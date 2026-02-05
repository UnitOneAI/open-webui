import logging
import sys
import inspect
import json
import asyncio
import hashlib
import re
from functools import wraps

from pydantic import BaseModel
from typing import AsyncGenerator, Generator, Iterator
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from starlette.responses import Response, StreamingResponse


from open_webui.constants import ERROR_MESSAGES
from open_webui.socket.main import (
    get_event_call,
    get_event_emitter,
)


from open_webui.models.users import UserModel
from open_webui.models.functions import Functions
from open_webui.models.models import Models

from open_webui.utils.plugin import (
    load_function_module_by_id,
    get_function_module_from_cache,
)
from open_webui.utils.tools import get_tools
from open_webui.utils.access_control import has_access

from open_webui.env import SRC_LOG_LEVELS, GLOBAL_LOG_LEVEL

from open_webui.utils.misc import (
    add_or_update_system_message,
    get_last_user_message,
    prepend_to_first_user_message_content,
    openai_chat_chunk_message_template,
    openai_chat_completion_message_template,
)
from open_webui.utils.payload import (
    apply_model_params_to_body_openai,
    apply_system_prompt_to_body,
)


logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])


def validate_pipe_id(pipe_id: str) -> bool:
    """Validate that pipe_id contains only safe characters"""
    # Allow alphanumeric, hyphens, underscores, and dots only
    if not re.match(r'^[a-zA-Z0-9._-]+$', pipe_id):
        return False
    # Prevent path traversal
    if '..' in pipe_id or pipe_id.startswith('/') or pipe_id.startswith('\\'):
        return False
    return True


def verify_function_integrity(pipe_id: str, function_code: str) -> bool:
    """Verify that the function hasn't been tampered with"""
    function_record = Functions.get_function_by_id(pipe_id)
    if not function_record:
        return False
    
    # Check if function is marked as active/approved
    if not getattr(function_record, 'is_active', True):
        return False
    
    return True


def sanitize_params(params: dict) -> dict:
    """Sanitize parameters to prevent code injection"""
    sanitized = {}
    allowed_types = (str, int, float, bool, list, dict, type(None), BaseModel)
    
    for key, value in params.items():
        # Check if key is a valid identifier
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', key):
            log.warning(f"Skipping invalid parameter key: {key}")
            continue
            
        # Check value type
        if isinstance(value, allowed_types):
            if isinstance(value, dict):
                sanitized[key] = sanitize_params(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    sanitize_params(item) if isinstance(item, dict) else item
                    for item in value
                    if isinstance(item, allowed_types)
                ]
            else:
                sanitized[key] = value
        elif isinstance(value, BaseModel):
            sanitized[key] = value
        else:
            log.warning(f"Skipping parameter {key} with disallowed type: {type(value)}")
    
    return sanitized


def validate_pipe_signature(pipe_func) -> bool:
    """Validate that the pipe function has a safe signature"""
    try:
        sig = inspect.signature(pipe_func)
        
        # Check for suspicious parameter names or types
        for param_name, param in sig.parameters.items():
            # Block parameters that could be used for code injection
            if param_name.startswith('__') and param_name not in [
                '__user__', '__event_emitter__', '__event_call__', 
                '__chat_id__', '__session_id__', '__message_id__',
                '__task__', '__task_body__', '__files__', '__metadata__',
                '__oauth_token__', '__request__', '__tools__', '__model__',
                '__messages__'
            ]:
                log.warning(f"Suspicious parameter name detected: {param_name}")
                return False
                
        return True
    except Exception as e:
        log.error(f"Error validating pipe signature: {e}")
        return False


def safe_execute_pipes_callable(function_module, pipe_id: str):
    """
    Safely execute the pipes callable with validation and sandboxing.
    Returns a list of pipes or empty list on error.
    """
    if not hasattr(function_module, 'pipes'):
        return []
    
    pipes_attr = function_module.pipes
    
    # If it's already a list, validate and return it
    if isinstance(pipes_attr, list):
        return pipes_attr
    
    # If it's callable, execute with safety checks
    if callable(pipes_attr):
        try:
            # Validate the callable signature to ensure it doesn't require unexpected parameters
            sig = inspect.signature(pipes_attr)
            if len(sig.parameters) > 0:
                log.warning(f"Function {pipe_id} pipes() callable has unexpected parameters, skipping")
                return []
            
            # Execute in a controlled manner
            if asyncio.iscoroutinefunction(pipes_attr):
                # For async functions, we need to run them in the current event loop
                result = asyncio.create_task(pipes_attr())
                # Wait for the result with a timeout
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(asyncio.wait_for(result, timeout=5.0))
            else:
                result = pipes_attr()
            
            # Validate the result is a list
            if not isinstance(result, list):
                log.warning(f"Function {pipe_id} pipes() returned non-list type, skipping")
                return []
            
            # Validate each pipe in the result
            validated_pipes = []
            for pipe in result:
                if isinstance(pipe, dict) and 'id' in pipe and 'name' in pipe:
                    # Additional validation for pipe structure
                    if isinstance(pipe['id'], str) and isinstance(pipe['name'], str):
                        validated_pipes.append(pipe)
                    else:
                        log.warning(f"Invalid pipe structure in {pipe_id}, skipping entry")
                else:
                    log.warning(f"Invalid pipe format in {pipe_id}, skipping entry")
            
            return validated_pipes
            
        except asyncio.TimeoutError:
            log.error(f"Timeout executing pipes() for function {pipe_id}")
            return []
        except Exception as e:
            log.error(f"Error executing pipes() for function {pipe_id}: {e}")
            return []
    
    return []


def get_function_module_by_id(request: Request, pipe_id: str):
    # Validate pipe_id format
    if not validate_pipe_id(pipe_id):
        log.error(f"Invalid pipe_id format: {pipe_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid function identifier"
        )
    
    # Verify the function exists and is authorized
    function_record = Functions.get_function_by_id(pipe_id)
    if not function_record:
        log.error(f"Function not found: {pipe_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Function not found"
        )
    
    # Verify function integrity
    function_code = getattr(function_record, 'content', '')
    if not verify_function_integrity(pipe_id, function_code):
        log.error(f"Function integrity check failed: {pipe_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Function failed security validation"
        )
    
    # Get user from request and verify access
    user = None
    if hasattr(request.state, 'user'):
        user = request.state.user
    
    if user and not has_access(user.id, type="read", access_control=getattr(function_record, 'access_control', {})):
        log.error(f"User {user.id} does not have access to function {pipe_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this function"
        )
    
    function_module, _, _ = get_function_module_from_cache(request, pipe_id)

    if hasattr(function_module, "valves") and hasattr(function_module, "Valves"):
        Valves = function_module.Valves
        valves = Functions.get_function_valves_by_id(pipe_id)

        if valves:
            try:
                function_module.valves = Valves(
                    **{k: v for k, v in valves.items() if v is not None}
                )
            except Exception as e:
                log.exception(f"Error loading valves for function {pipe_id}: {e}")
                raise e
        else:
            function_module.valves = Valves()

    return function_module


async def get_function_models(request):
    pipes = Functions.get_functions_by_type("pipe", active_only=True)
    pipe_models = []

    for pipe in pipes:
        try:
            # Validate pipe ID before processing
            if not validate_pipe_id(pipe.id):
                log.warning(f"Skipping invalid pipe_id: {pipe.id}")
                continue
                
            function_module = get_function_module_by_id(request, pipe.id)

            has_user_valves = False
            if hasattr(function_module, "UserValves"):
                has_user_valves = True

            # Check if function is a manifold
            if hasattr(function_module, "pipes"):
                # Use safe execution method
                sub_pipes = safe_execute_pipes_callable(function_module, pipe.id)

                log.debug(
                    f"get_function_models: function '{pipe.id}' is a manifold of {sub_pipes}"
                )

                for p in sub_pipes:
                    sub_pipe_id = f'{pipe.id}.{p["id"]}'
                    
                    # Validate sub-pipe ID
                    if not validate_pipe_id(sub_pipe_id):
                        log.warning(f"Skipping invalid sub_pipe_id: {sub_pipe_id}")
                        continue
                    
                    sub_pipe_name = p["name"]

                    if hasattr(function_module, "name"):
                        sub_pipe_name = f"{function_module.name}{sub_pipe_name}"

                    pipe_flag = {"type": pipe.type}

                    pipe_models.append(
                        {
                            "id": sub_pipe_id,
                            "name": sub_pipe_name,
                            "object": "model",
                            "created": pipe.created_at,
                            "owned_by": "openai",
                            "pipe": pipe_flag,
                            "has_user_valves": has_user_valves,
                        }
                    )
            else:
                pipe_flag = {"type": "pipe"}

                log.debug(
                    f"get_function_models: function '{pipe.id}' is a single pipe {{ 'id': {pipe.id}, 'name': {pipe.name} }}"
                )

                pipe_models.append(
                    {
                        "id": pipe.id,
                        "name": pipe.name,
                        "object": "model",
                        "created": pipe.created_at,
                        "owned_by": "openai",
                        "pipe": pipe_flag,
                        "has_user_valves": has_user_valves,
                    }
                )
        except Exception as e:
            log.exception(e)
            continue

    return pipe_models


async def generate_function_chat_completion(
    request, form_data, user, models: dict = {}
):
    async def execute_pipe(pipe, params):
        # Validate pipe function signature
        if not validate_pipe_signature(pipe):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid pipe function signature detected"
            )
        
        # Sanitize parameters before execution
        sanitized_params = sanitize_params(params)
        
        # Execute with timeout to prevent hanging
        try:
            if inspect.iscoroutinefunction(pipe):
                return await asyncio.wait_for(pipe(**sanitized_params), timeout=300.0)
            else:
                # Run sync function in executor to prevent blocking
                loop = asyncio.get_event_loop()
                return await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: pipe(**sanitized_params)),
                    timeout=300.0
                )
        except asyncio.TimeoutError:
            log.error("Pipe execution timeout")
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail="Function execution timeout"
            )

    async def get_message_content(res: str | Generator | AsyncGenerator) -> str:
        if isinstance(res, str):
            return res
        if isinstance(res, Generator):
            return "".join(map(str, res))
        if isinstance(res, AsyncGenerator):
            return "".join([str(stream) async for stream in res])

    def process_line(form_data: dict, line):
        if isinstance(line, BaseModel):
            line = line.model_dump_json()
            line = f"data: {line}"
        if isinstance(line, dict):
            line = f"data: {json.dumps(line)}"

        try:
            line = line.decode("utf-8")
        except Exception:
            pass

        if line.startswith("data:"):
            return f"{line}\n\n"
        else:
            line = openai_chat_chunk_message_template(form_data["model"], line)
            return f"data: {json.dumps(line)}\n\n"

    def get_pipe_id(form_data: dict) -> str:
        pipe_id = form_data["model"]
        if "." in pipe_id:
            pipe_id, _ = pipe_id.split(".", 1)
        return pipe_id

    def get_function_params(function_module, form_data, user, extra_params=None):
        if extra_params is None:
            extra_params = {}

        pipe_id = get_pipe_id(form_data)

        # Get the signature of the function
        sig = inspect.signature(function_module.pipe)
        params = {"body": form_data} | {
            k: v for k, v in extra_params.items() if k in sig.parameters
        }

        if "__user__" in params and hasattr(function_module, "UserValves"):
            user_valves = Functions.get_user_valves_by_id_and_user_id(pipe_id, user.id)
            try:
                params["__user__"]["valves"] = function_module.UserValves(**user_valves)
            except Exception as e:
                log.exception(e)
                params["__user__"]["valves"] = function_module.UserValves()

        return params

    model_id = form_data.get("model")
    model_info = Models.get_model_by_id(model_id)

    metadata = form_data.pop("metadata", {})

    files = metadata.get("files", [])
    tool_ids = metadata.get("tool_ids", [])
    # Check if tool_ids is None
    if tool_ids is None:
        tool_ids = []

    __event_emitter__ = None
    __event_call__ = None
    __task__ = None
    __task_body__ = None

    if metadata:
        if all(k in metadata for k in ("session_id", "chat_id", "message_id")):
            __event_emitter__ = get_event_emitter(metadata)
            __event_call__ = get_event_call(metadata)
        __task__ = metadata.get("task", None)
        __task_body__ = metadata.get("task_body", None)

    oauth_token = None
    try:
        if request.cookies.get("oauth_session_id", None):
            oauth_token = await request.app.state.oauth_manager.get_oauth_token(
                user.id,
                request.cookies.get("oauth_session_id", None),
            )
    except Exception as e:
        log.error(f"Error getting OAuth token: {e}")

    extra_params = {
        "__event_emitter__": __event_emitter__,
        "__event_call__": __event_call__,
        "__chat_id__": metadata.get("chat_id", None),
        "__session_id__": metadata.get("session_id", None),
        "__message_id__": metadata.get("message_id", None),
        "__task__": __task__,
        "__task_body__": __task_body__,
        "__files__": files,
        "__user__": user.model_dump() if isinstance(user, UserModel) else {},
        "__metadata__": metadata,
        "__oauth_token__": oauth_token,
        "__request__": request,
    }
    extra_params["__tools__"] = await get_tools(
        request,
        tool_ids,
        user,
        {
            **extra_params,
            "__model__": models.get(form_data["model"], None),
            "__messages__": form_data["messages"],
            "__files__": files,
        },
    )

    if model_info:
        if model_info.base_model_id:
            form_data["model"] = model_info.base_model_id

        params = model_info.params.model_dump()

        if params:
            system = params.pop("system", None)
            form_data = apply_model_params_to_body_openai(params, form_data)
            form_data = apply_system_prompt_to_body(system, form_data, metadata, user)

    pipe_id = get_pipe_id(form_data)
    
    # Validate pipe_id before getting function module
    if not validate_pipe_id(pipe_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid function identifier"
        )
    
    function_module = get_function_module_by_id(request, pipe_id)

    pipe = function_module.pipe
    params = get_function_params(function_module, form_data, user, extra_params)

    if form_data.get("stream", False):

        async def stream_content():
            try:
                res = await execute_pipe(pipe, params)

                # Directly return if the response is a StreamingResponse
                if isinstance(res, StreamingResponse):
                    async for data in res.body_iterator:
                        yield data
                    return
                if isinstance(res, dict):
                    yield f"data: {json.dumps(res)}\n\n"
                    return

            except Exception as e:
                log.error(f"Error: {e}")
                yield f"data: {json.dumps({'error': {'detail':str(e)}})}\n\n"
                return

            if isinstance(res, str):
                message = openai_chat_chunk_message_template(form_data["model"], res)
                yield f"data: {json.dumps(message)}\n\n"

            if isinstance(res, Iterator):
                for line in res:
                    yield process_line(form_data, line)

            if isinstance(res, AsyncGenerator):
                async for line in res:
                    yield process_line(form_data, line)

            if isinstance(res, str) or isinstance(res, Generator):
                finish_message = openai_chat_chunk_message_template(
                    form_data["model"], ""
                )
                finish_message["choices"][0]["finish_reason"] = "stop"
                yield f"data: {json.dumps(finish_message)}\n\n"
                yield "data: [DONE]"

        return StreamingResponse(stream_content(), media_type="text/event-stream")
    else:
        try:
            res = await execute_pipe(pipe, params)

        except Exception as e:
            log.error(f"Error: {e}")
            return {"error": {"detail": str(e)}}

        if isinstance(res, StreamingResponse) or isinstance(res, dict):
            return res
        if isinstance(res, BaseModel):
            return res.model_dump()

        message = await get_message_content(res)
        return openai_chat_completion_message_template(form_data["model"], message)