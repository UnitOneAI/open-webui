import os
import re
import subprocess
import sys
from importlib import util
import types
import tempfile
import logging
import hashlib
import uuid

from open_webui.env import SRC_LOG_LEVELS, PIP_OPTIONS, PIP_PACKAGE_INDEX_OPTIONS
from open_webui.models.functions import Functions
from open_webui.models.tools import Tools

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])


def validate_function_id(function_id: str) -> bool:
    """
    Validate that the function_id is a valid UUID or alphanumeric string.
    """
    try:
        # Try parsing as UUID
        uuid.UUID(function_id)
        return True
    except ValueError:
        # Check if it's a safe alphanumeric string (with underscores and hyphens)
        return bool(re.match(r'^[a-zA-Z0-9_-]+$', function_id))


def validate_tool_id(tool_id: str) -> bool:
    """
    Validate that the tool_id is a valid UUID or alphanumeric string.
    """
    try:
        # Try parsing as UUID
        uuid.UUID(tool_id)
        return True
    except ValueError:
        # Check if it's a safe alphanumeric string (with underscores and hyphens)
        return bool(re.match(r'^[a-zA-Z0-9_-]+$', tool_id))


def validate_content_integrity(content: str, expected_hash: str = None) -> bool:
    """
    Validate content integrity using hash if provided.
    """
    if expected_hash is None:
        return True
    
    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    return content_hash == expected_hash


def sanitize_code_content(content: str) -> str:
    """
    Perform basic sanitization checks on code content.
    Raises exception if dangerous patterns are detected.
    """
    dangerous_patterns = [
        r'__import__\s*\(\s*["\']os["\']',
        r'__import__\s*\(\s*["\']subprocess["\']',
        r'eval\s*\(',
        r'compile\s*\(',
        r'globals\s*\(\s*\)',
        r'locals\s*\(\s*\)',
        r'setattr\s*\(',
        r'delattr\s*\(',
        r'__builtins__',
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            log.warning(f"Potentially dangerous pattern detected: {pattern}")
    
    return content


def extract_frontmatter(content):
    """
    Extract frontmatter as a dictionary from the provided content string.
    """
    frontmatter = {}
    frontmatter_started = False
    frontmatter_ended = False
    frontmatter_pattern = re.compile(r"^\s*([a-z_]+):\s*(.*)\s*$", re.IGNORECASE)

    try:
        lines = content.splitlines()
        if len(lines) < 1 or lines[0].strip() != '"""':
            # The content doesn't start with triple quotes
            return {}

        frontmatter_started = True

        for line in lines[1:]:
            if '"""' in line:
                if frontmatter_started:
                    frontmatter_ended = True
                    break

            if frontmatter_started and not frontmatter_ended:
                match = frontmatter_pattern.match(line)
                if match:
                    key, value = match.groups()
                    frontmatter[key.strip()] = value.strip()

    except Exception as e:
        log.exception(f"Failed to extract frontmatter: {e}")
        return {}

    return frontmatter


def replace_imports(content):
    """
    Replace the import paths in the content.
    """
    replacements = {
        "from utils": "from open_webui.utils",
        "from apps": "from open_webui.apps",
        "from main": "from open_webui.main",
        "from config": "from open_webui.config",
    }

    for old, new in replacements.items():
        content = content.replace(old, new)

    return content


def load_tool_module_by_id(tool_id, content=None):
    # Validate tool_id
    if not validate_tool_id(tool_id):
        raise ValueError(f"Invalid tool_id format: {tool_id}")

    if content is None:
        tool = Tools.get_tool_by_id(tool_id)
        if not tool:
            raise Exception(f"Toolkit not found: {tool_id}")

        content = tool.content

        content = replace_imports(content)
        Tools.update_tool_by_id(tool_id, {"content": content})
    else:
        # Sanitize content
        content = sanitize_code_content(content)
        
        frontmatter = extract_frontmatter(content)
        # Install required packages found within the frontmatter
        install_frontmatter_requirements(frontmatter.get("requirements", ""))

    module_name = f"tool_{tool_id}"
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module

    # Create a temporary file and use it to define `__file__` so
    # that it works as expected from the module's perspective.
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.close()
    try:
        with open(temp_file.name, "w", encoding="utf-8") as f:
            f.write(content)
        module.__dict__["__file__"] = temp_file.name

        # Executing the modified content in the created module's namespace
        exec(content, module.__dict__)
        frontmatter = extract_frontmatter(content)
        log.info(f"Loaded module: {module.__name__}")

        # Create and return the object if the class 'Tools' is found in the module
        if hasattr(module, "Tools"):
            return module.Tools(), frontmatter
        else:
            raise Exception("No Tools class found in the module")
    except Exception as e:
        log.error(f"Error loading module: {tool_id}: {e}")
        del sys.modules[module_name]  # Clean up
        raise e
    finally:
        os.unlink(temp_file.name)


def load_function_module_by_id(function_id: str, content: str | None = None):
    # Validate function_id
    if not validate_function_id(function_id):
        raise ValueError(f"Invalid function_id format: {function_id}")

    if content is None:
        function = Functions.get_function_by_id(function_id)
        if not function:
            raise Exception(f"Function not found: {function_id}")
        content = function.content

        content = replace_imports(content)
        Functions.update_function_by_id(function_id, {"content": content})
    else:
        # Sanitize content
        content = sanitize_code_content(content)
        
        frontmatter = extract_frontmatter(content)
        install_frontmatter_requirements(frontmatter.get("requirements", ""))

    module_name = f"function_{function_id}"
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module

    # Create a temporary file and use it to define `__file__` so
    # that it works as expected from the module's perspective.
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.close()
    try:
        with open(temp_file.name, "w", encoding="utf-8") as f:
            f.write(content)
        module.__dict__["__file__"] = temp_file.name

        # Execute the modified content in the created module's namespace
        exec(content, module.__dict__)
        frontmatter = extract_frontmatter(content)
        log.info(f"Loaded module: {module.__name__}")

        # Create appropriate object based on available class type in the module
        if hasattr(module, "Pipe"):
            return module.Pipe(), "pipe", frontmatter
        elif hasattr(module, "Filter"):
            return module.Filter(), "filter", frontmatter
        elif hasattr(module, "Action"):
            return module.Action(), "action", frontmatter
        else:
            raise Exception("No Function class found in the module")
    except Exception as e:
        log.error(f"Error loading module: {function_id}: {e}")
        # Cleanup by removing the module in case of error
        del sys.modules[module_name]

        Functions.update_function_by_id(function_id, {"is_active": False})
        raise e
    finally:
        os.unlink(temp_file.name)


def get_tool_module_from_cache(request, tool_id, load_from_db=True):
    # Validate tool_id
    if not validate_tool_id(tool_id):
        raise ValueError(f"Invalid tool_id format: {tool_id}")

    if load_from_db:
        # Always load from the database by default
        tool = Tools.get_tool_by_id(tool_id)
        if not tool:
            raise Exception(f"Tool not found: {tool_id}")
        content = tool.content

        new_content = replace_imports(content)
        if new_content != content:
            content = new_content
            # Update the tool content in the database
            Tools.update_tool_by_id(tool_id, {"content": content})

        if (
            hasattr(request.app.state, "TOOL_CONTENTS")
            and tool_id in request.app.state.TOOL_CONTENTS
        ) and (
            hasattr(request.app.state, "TOOLS") and tool_id in request.app.state.TOOLS
        ):
            if request.app.state.TOOL_CONTENTS[tool_id] == content:
                return request.app.state.TOOLS[tool_id], None

        tool_module, frontmatter = load_tool_module_by_id(tool_id, content)
    else:
        if hasattr(request.app.state, "TOOLS") and tool_id in request.app.state.TOOLS:
            return request.app.state.TOOLS[tool_id], None

        tool_module, frontmatter = load_tool_module_by_id(tool_id)

    if not hasattr(request.app.state, "TOOLS"):
        request.app.state.TOOLS = {}

    if not hasattr(request.app.state, "TOOL_CONTENTS"):
        request.app.state.TOOL_CONTENTS = {}

    request.app.state.TOOLS[tool_id] = tool_module
    request.app.state.TOOL_CONTENTS[tool_id] = content

    return tool_module, frontmatter


def get_function_module_from_cache(request, function_id, load_from_db=True):
    # Validate function_id
    if not validate_function_id(function_id):
        raise ValueError(f"Invalid function_id format: {function_id}")

    if load_from_db:
        # Always load from the database by default
        # This is useful for hooks like "inlet" or "outlet" where the content might change
        # and we want to ensure the latest content is used.

        function = Functions.get_function_by_id(function_id)
        if not function:
            raise Exception(f"Function not found: {function_id}")
        content = function.content

        new_content = replace_imports(content)
        if new_content != content:
            content = new_content
            # Update the function content in the database
            Functions.update_function_by_id(function_id, {"content": content})

        if (
            hasattr(request.app.state, "FUNCTION_CONTENTS")
            and function_id in request.app.state.FUNCTION_CONTENTS
        ) and (
            hasattr(request.app.state, "FUNCTIONS")
            and function_id in request.app.state.FUNCTIONS
        ):
            if request.app.state.FUNCTION_CONTENTS[function_id] == content:
                return request.app.state.FUNCTIONS[function_id], None, None

        function_module, function_type, frontmatter = load_function_module_by_id(
            function_id, content
        )
    else:
        # Load from cache (e.g. "stream" hook)
        # This is useful for performance reasons

        if (
            hasattr(request.app.state, "FUNCTIONS")
            and function_id in request.app.state.FUNCTIONS
        ):
            return request.app.state.FUNCTIONS[function_id], None, None

        function_module, function_type, frontmatter = load_function_module_by_id(
            function_id
        )

    if not hasattr(request.app.state, "FUNCTIONS"):
        request.app.state.FUNCTIONS = {}

    if not hasattr(request.app.state, "FUNCTION_CONTENTS"):
        request.app.state.FUNCTION_CONTENTS = {}

    request.app.state.FUNCTIONS[function_id] = function_module
    request.app.state.FUNCTION_CONTENTS[function_id] = content

    return function_module, function_type, frontmatter


def install_frontmatter_requirements(requirements: str):
    if requirements:
        try:
            req_list = [req.strip() for req in requirements.split(",")]
            
            # Validate package names to prevent command injection
            safe_package_pattern = re.compile(r'^[a-zA-Z0-9_\-\.\[\]<>=]+$')
            for req in req_list:
                if not safe_package_pattern.match(req):
                    raise ValueError(f"Invalid package specification: {req}")
            
            log.info(f"Installing requirements: {' '.join(req_list)}")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install"]
                + PIP_OPTIONS
                + req_list
                + PIP_PACKAGE_INDEX_OPTIONS
            )
        except Exception as e:
            log.error(f"Error installing packages: {' '.join(req_list)}")
            raise e

    else:
        log.info("No requirements found in frontmatter.")


def install_tool_and_function_dependencies():
    """
    Install all dependencies for all admin tools and active functions.

    By first collecting all dependencies from the frontmatter of each tool and function,
    and then installing them using pip. Duplicates or similar version specifications are
    handled by pip as much as possible.
    """
    function_list = Functions.get_functions(active_only=True)
    tool_list = Tools.get_tools()

    all_dependencies = ""
    try:
        for function in function_list:
            frontmatter = extract_frontmatter(replace_imports(function.content))
            if dependencies := frontmatter.get("requirements"):
                all_dependencies += f"{dependencies}, "
        for tool in tool_list:
            # Only install requirements for admin tools
            if tool.user and tool.user.role == "admin":
                frontmatter = extract_frontmatter(replace_imports(tool.content))
                if dependencies := frontmatter.get("requirements"):
                    all_dependencies += f"{dependencies}, "

        install_frontmatter_requirements(all_dependencies.strip(", "))
    except Exception as e:
        log.error(f"Error installing requirements: {e}")