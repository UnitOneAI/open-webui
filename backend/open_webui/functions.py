import logging
import sys
import inspect
import json
import asyncio
import hashlib

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
    """
    Validate pipe_id to ensure it only contains safe characters.
    Prevents path traversal and code injection attacks.
    """
    if not pipe_id:
        return False
    
    # Allow alphanumeric, hyphens, underscores, and dots only
    import re
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', pipe_id):
        return False
    
    # Prevent path traversal attempts
    if '..' in pipe_id or '/' in pipe_id or '\\' in pipe_id:
        return False
    
    return True


def sanitize_params(params: dict) -> dict:
    """
    Sanitize parameters to prevent code injection.
    Creates a deep copy and validates nested structures.
    """
    import copy
    
    # Create a deep copy to avoid modifying original
    sanitized = copy.deepcopy(params)
    
    # Remove any potentially dangerous keys
    dangerous_keys = ['__import__', '__builtins__', 'eval', 'exec', 'compile', '__code__', '__globals__']
    
    def clean_dict(d):
        if not isinstance(d, dict):
            return d
        
        for key in list(d.keys()):
            if key in dangerous_keys:
                del d[key]
            elif isinstance(d[key], dict):
                d[key] = clean_dict(d[key])
            elif isinstance(d[key], list):
                d[key] = [clean_dict(item) if isinstance(item, dict) else item for item in d[key]]
        
        return d
    
    return clean_dict(sanitized)


def validate_pipe_function(pipe_function, pipe_id: str) -> bool:
    """
    Validate that the pipe function is safe to execute.
    """
    if not callable(pipe_function):
        log.error(f"Pipe function is not callable for {pipe_id}")
        return False
    
    # Check if the function has dangerous attributes
    if hasattr(pipe_function, '__globals__'):
        globals_dict = pipe_function.__globals__
        dangerous_builtins = ['eval', 'exec', 'compile', '__import__']
        
        for builtin in dangerous_builtins:
            if builtin in globals_dict and callable(globals_dict[builtin]):
                log.warning(f"Pipe function {pipe_id} has access to dangerous builtin: {builtin}")
    
    return True


def get_function_module_by_id(request: Request, pipe_id: str):
    # Validate pipe_id before processing
    if not validate_pipe_id(pipe_id):
        log.error(f"Invalid pipe_id format: {pipe_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid function ID format"
        )
    
    # Verify the function exists in the database before loading
    function_record = Functions.get_function_by_id(pipe_id.split('.')[0])
    if not function_record:
        log.error(f"Function not found in database: {pipe_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Function not found"
        )
    
    try:
        function_module, _, _ = get_function_module_from_cache(request, pipe_id)
    except Exception as e:
        log.exception(f"Error loading function module {pipe_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading function module"
        )

    if hasattr(function_module, "valves") and hasattr(function_module, "Valves"):
        Valves = function_module.Valves
        valves = Functions.get_function_valves_by_id(pipe_id)

        if valves:
            try:
                # Validate valves data before instantiation
                if not isinstance(valves, dict):
                    raise ValueError("Invalid valves data format")
                
                function_module.valves = Valves(
                    **{k: v for k, v in valves.items() if v is not None}
                )
            except Exception as e:
                log.exception(f"Error loading valves for function {pipe_id}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error loading function valves: {str(e)}"
                )
        else:
            function_module.valves = Valves()

    return function_module


async def get_function_models(request):
    pipes = Functions.get_functions_by_type("pipe", active_only=True)
    pipe_models = []

    for pipe in pipes:
        try:
            # Validate pipe.id before processing
            if not validate_pipe_id(pipe.id):
                log.warning(f"Skipping function with invalid ID: {pipe.id}")
                continue
                
            function_module = get_function_module_by_id(request, pipe.id)

            has_user_valves = False
            if hasattr(function_module, "UserValves"):
                has_user_valves = True

            # Check if function is a manifold
            if hasattr(function_module, "pipes"):
                sub_pipes = []

                # Handle pipes being a list, sync function, or async function
                try:
                    if callable(function_module.pipes):
                        if asyncio.iscoroutinefunction(function_module.pipes):
                            sub_pipes = await function_module.pipes()
                        else:
                            sub_pipes = function_module.pipes()
                    else:
                        sub_pipes = function_module.pipes
                except Exception as e:
                    log.exception(e)
                    sub_pipes = []

                log.debug(
                    f"get_function_models: function '{pipe.id}' is a manifold of {sub_pipes}"
                )

                for p in sub_pipes:
                    # Validate sub-pipe ID
                    if not isinstance(p, dict) or 'id' not in p or 'name' not in p:
                        log.warning(f"Skipping invalid sub-pipe: {p}")
                        continue
                    
                    if not validate_pipe_id(p["id"]):
                        log.warning(f"Skipping sub-pipe with invalid ID: {p['id']}")
                        continue
                        
                    sub_pipe_id = f'{pipe.id}.{p["id"]}'
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
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            log.exception(e)
            continue

    return pipe_models


async def generate_function_chat_completion(
    request, form_data, user, models: dict = {}
):
    async def execute_pipe(pipe, params, pipe_id: str):
        """
        Safely execute a pipe function with validated parameters.
        """
        # Validate the pipe function before execution
        if not validate_pipe_function(pipe, pipe_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Pipe function failed security validation"
            )
        
        # Sanitize parameters to prevent code injection
        safe_params = sanitize_params(params)
        
        try:
            if inspect.iscoroutinefunction(pipe):
                return await pipe(**safe_params)
            else:
                return pipe(**safe_params)
        except TypeError as e:
            log.error(f"Parameter error executing pipe {pipe_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid parameters for pipe function"
            )
        except Exception as e:
            log.error(f"Error executing pipe {pipe_id}: {e}")
            raise

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
    
    # Validate model_id
    if not model_id or not isinstance(model_id, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid model ID"
        )
    
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
    
    # Validate pipe_id
    if not validate_pipe_id(pipe_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid function ID"
        )
    
    function_module = get_function_module_by_id(request, pipe_id)

    pipe = function_module.pipe
    params = get_function_params(function_module, form_data, user, extra_params)

    if form_data.get("stream", False):

        async def stream_content():
            try:
                res = await execute_pipe(pipe, params, pipe_id)

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
            res = await execute_pipe(pipe, params, pipe_id)

        except Exception as e:
            log.error(f"Error: {e}")
            return {"error": {"detail": str(e)}}

        if isinstance(res, StreamingResponse) or isinstance(res, dict):
            return res
        if isinstance(res, BaseModel):
            return res.model_dump()

        message = await get_message_content(res)
        return openai_chat_completion_message_template(form_data["model"], message)