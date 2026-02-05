import logging
import sys
import inspect
import json
import asyncio
import hashlib
import hmac

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


def validate_valves(valves_dict: dict, valves_class) -> bool:
    """
    Validates that valve values are safe and match expected types.
    Returns True if valid, False otherwise.
    """
    if not valves_dict:
        return True
    
    try:
        # Get the expected fields from the Valves class
        if hasattr(valves_class, '__fields__'):
            expected_fields = valves_class.__fields__
            
            # Check that all provided valves are expected
            for key in valves_dict.keys():
                if key not in expected_fields:
                    log.warning(f"Unexpected valve key: {key}")
                    return False
            
            # Validate types match expectations
            for key, value in valves_dict.items():
                if value is not None and key in expected_fields:
                    expected_type = expected_fields[key].annotation
                    # Basic type checking
                    if hasattr(expected_type, '__origin__'):
                        # Handle generic types
                        continue
                    if not isinstance(value, expected_type):
                        # Try to convert basic types
                        if expected_type in (int, float, str, bool):
                            continue
                        log.warning(f"Type mismatch for valve {key}: expected {expected_type}, got {type(value)}")
                        return False
        
        return True
    except Exception as e:
        log.error(f"Error validating valves: {e}")
        return False


def sanitize_valve_value(value):
    """
    Sanitizes a single valve value to prevent code injection.
    """
    if value is None:
        return None
    
    # If it's a string, check for dangerous patterns
    if isinstance(value, str):
        # Reject strings that look like code or system commands
        dangerous_patterns = [
            '__import__',
            'exec(',
            'eval(',
            'compile(',
            'os.system',
            'subprocess',
            '__builtins__',
            '__globals__',
            '__code__',
            'open(',
            'file(',
        ]
        
        value_lower = value.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in value_lower:
                log.warning(f"Potentially dangerous pattern detected in valve value: {pattern}")
                raise ValueError(f"Invalid valve value: contains prohibited pattern")
    
    # For other basic types, return as-is
    if isinstance(value, (int, float, bool, list, dict)):
        # Recursively sanitize lists and dicts
        if isinstance(value, list):
            return [sanitize_valve_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: sanitize_valve_value(v) for k, v in value.items()}
        return value
    
    # For complex objects, only allow if they're from safe types
    if hasattr(value, '__dict__'):
        log.warning(f"Complex object passed as valve value: {type(value)}")
        raise ValueError(f"Invalid valve value type: {type(value)}")
    
    return value


def get_function_module_by_id(request: Request, pipe_id: str):
    function_module, _, _ = get_function_module_from_cache(request, pipe_id)

    if hasattr(function_module, "valves") and hasattr(function_module, "Valves"):
        Valves = function_module.Valves
        valves = Functions.get_function_valves_by_id(pipe_id)

        if valves:
            try:
                # Validate valves before using them
                if not validate_valves(valves, Valves):
                    log.error(f"Invalid valves configuration for function {pipe_id}")
                    raise ValueError("Invalid valves configuration")
                
                # Sanitize valve values
                sanitized_valves = {}
                for k, v in valves.items():
                    if v is not None:
                        try:
                            sanitized_valves[k] = sanitize_valve_value(v)
                        except ValueError as e:
                            log.error(f"Error sanitizing valve {k} for function {pipe_id}: {e}")
                            raise ValueError(f"Invalid valve value for {k}")
                
                # Create valves instance with sanitized values
                function_module.valves = Valves(**sanitized_valves)
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
        except Exception as e:
            log.exception(e)
            continue

    return pipe_models


async def generate_function_chat_completion(
    request, form_data, user, models: dict = {}
):
    async def execute_pipe(pipe, params):
        if inspect.iscoroutinefunction(pipe):
            return await pipe(**params)
        else:
            return pipe(**params)

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
                # Validate and sanitize user valves
                if validate_valves(user_valves, function_module.UserValves):
                    sanitized_user_valves = {}
                    for k, v in user_valves.items():
                        try:
                            sanitized_user_valves[k] = sanitize_valve_value(v)
                        except ValueError as e:
                            log.error(f"Error sanitizing user valve {k}: {e}")
                            continue
                    params["__user__"]["valves"] = function_module.UserValves(**sanitized_user_valves)
                else:
                    params["__user__"]["valves"] = function_module.UserValves()
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