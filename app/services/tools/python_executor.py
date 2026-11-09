import ast
import math
import statistics
from typing import Dict, Any


class RestrictedPythonExecutor:
    """
    Safely executes controlled Python data analysis and calculation scripts.
    Strictly forbids file system, network, shell execution, and reflection.
    """

    FORBIDDEN_IMPORTS = {
        "os", "sys", "subprocess", "socket", "shutil", "builtins",
        "requests", "urllib", "http", "ftplib", "telnetlib", "posix",
        "winreg", "ctypes", "pathlib", "importlib", "inspect", "platform"
    }

    FORBIDDEN_BUILTINS = {
        "eval", "exec", "open", "compile", "__import__", "getattr",
        "setattr", "delattr", "globals", "locals", "input", "breakpoint",
        "exit", "quit", "memoryview"
    }

    ALLOWED_BUILTINS = {
        "abs": abs,
        "all": all,
        "any": any,
        "bin": bin,
        "bool": bool,
        "dict": dict,
        "divmod": divmod,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "format": format,
        "int": int,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "pow": pow,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "print": print,
    }

    @classmethod
    def validate_ast(cls, code: str) -> None:
        """Parse code into AST and ensure no disallowed nodes or calls exist."""
        tree = ast.parse(code)
        for node in ast.walk(tree):
            # Block import statements
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in getattr(node, "names", []):
                    name = alias.name.split(".")[0]
                    if name in cls.FORBIDDEN_IMPORTS:
                        raise SecurityError(f"Importing module '{name}' is strictly prohibited.")
            
            # Block dunder attribute access (e.g., __subclasses__, __globals__)
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    raise SecurityError(f"Accessing private dunder attribute '{node.attr}' is prohibited.")

            # Block call to forbidden builtins
            if isinstance(node, ast.Name):
                if node.id in cls.FORBIDDEN_BUILTINS:
                    raise SecurityError(f"Calling built-in '{node.id}' is strictly prohibited.")

    @classmethod
    def execute(cls, code: str) -> Dict[str, Any]:
        """Safely execute snippet and return captured local variables and output."""
        try:
            cls.validate_ast(code)
        except SecurityError as se:
            return {"success": False, "error": f"SecurityViolation: {str(se)}"}
        except SyntaxError as syn:
            return {"success": False, "error": f"SyntaxError: {str(syn)}"}

        safe_globals = {
            "__builtins__": cls.ALLOWED_BUILTINS,
            "math": math,
            "statistics": statistics,
        }
        safe_locals: Dict[str, Any] = {}

        try:
            # Execute within restricted namespace
            exec(code, safe_globals, safe_locals)
            
            # Extract serializable outputs
            serializable_locals = {}
            for k, v in safe_locals.items():
                if not k.startswith("_") and isinstance(v, (int, float, str, bool, list, dict, set, tuple)):
                    serializable_locals[k] = v

            return {
                "success": True,
                "result": serializable_locals,
                "summary": f"Computed {len(serializable_locals)} variable(s): {', '.join(serializable_locals.keys())}"
            }
        except Exception as e:
            return {"success": False, "error": f"ExecutionError: {type(e).__name__}: {str(e)}"}


class SecurityError(Exception):
    pass

