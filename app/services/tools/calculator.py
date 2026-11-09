import ast
import math
import operator
from typing import Dict, Any, Union


class SafeCalculator:
    """
    Safe mathematical expression evaluator using AST parsing.
    Does not use arbitrary eval(); only allows arithmetic and standard math functions.
    """
    
    _OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    _FUNCTIONS = {
        "abs": abs,
        "round": round,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "floor": math.floor,
        "ceil": math.ceil,
        "factorial": math.factorial,
    }

    _CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
    }

    @classmethod
    def _eval_node(cls, node: ast.AST) -> Union[int, float]:
        if isinstance(node, ast.Expression):
            return cls._eval_node(node.body)
        
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")

        elif isinstance(node, ast.Name):
            if node.id in cls._CONSTANTS:
                return cls._CONSTANTS[node.id]
            raise ValueError(f"Unknown variable or constant: {node.id}")

        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type in cls._OPERATORS:
                return cls._OPERATORS[op_type](cls._eval_node(node.operand))
            raise ValueError(f"Unsupported unary operator: {op_type}")

        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type in cls._OPERATORS:
                left = cls._eval_node(node.left)
                right = cls._eval_node(node.right)
                if op_type == ast.Div and right == 0:
                    raise ZeroDivisionError("Division by zero")
                return cls._OPERATORS[op_type](left, right)
            raise ValueError(f"Unsupported binary operator: {op_type}")

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in cls._FUNCTIONS:
                func = cls._FUNCTIONS[node.func.id]
                args = [cls._eval_node(arg) for arg in node.args]
                return func(*args)
            raise ValueError(f"Unsupported function call: {getattr(node.func, 'id', 'unknown')}")

        else:
            raise ValueError(f"Unsupported AST node expression: {type(node).__name__}")

    @classmethod
    def evaluate(cls, expression: str) -> Dict[str, Any]:
        """Safely evaluate a mathematical expression."""
        clean_expr = expression.strip()
        # Replace common symbol variants
        clean_expr = clean_expr.replace("^", "**").replace("×", "*").replace("÷", "/")
        
        try:
            tree = ast.parse(clean_expr, mode='eval')
            result = cls._eval_node(tree)
            return {
                "success": True,
                "expression": expression,
                "result": result,
                "formatted": f"{result:,.6g}" if isinstance(result, float) else f"{result:,}"
            }
        except ZeroDivisionError:
            return {"success": False, "expression": expression, "error": "ZeroDivisionError: Division by zero"}
        except Exception as e:
            return {"success": False, "expression": expression, "error": f"EvaluationError: {str(e)}"}

