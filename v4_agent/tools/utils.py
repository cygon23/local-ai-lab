"""
tools/utils.py
--------------
Simple utility tools: datetime and calculator.

These seem trivial but teach an important lesson:
  LLMs are bad at two things: current time and exact math.
  Time: the model's training has a cutoff — it doesn't know "now".
  Math: LLMs predict tokens — they don't compute, they guess.
        "What is 847 * 293?" — the model might say 248,121 instead of 248,171.
  
  Tools fix both of these completely. The model DECIDES to use the tool,
  but Python does the actual work. Perfect accuracy.
"""

from datetime import datetime
import ast
import operator


def get_datetime() -> str:
    """Return current date and time as a human-readable string."""
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y at %H:%M:%S")


# Safe math operators — we don't use eval() for security
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}


def _safe_eval(node):
    """
    Recursively evaluate an AST node using only safe math operations.
    This is safer than eval() — it only allows arithmetic, no function calls,
    no imports, no arbitrary code execution.
    """
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported operation: {op_type}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return SAFE_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operation: {op_type}")
        operand = _safe_eval(node.operand)
        return SAFE_OPERATORS[op_type](operand)
    else:
        raise ValueError(f"Unsupported expression node: {type(node)}")


def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    Uses AST parsing instead of eval() to prevent code injection.

    Examples:
      "15 * 8 + 200"  → "320"
      "(100 / 3) * 2" → "66.666..."
      "2 ** 10"        → "1024"
    """
    try:
        # Clean up the expression
        expression = expression.strip()

        # Parse into AST
        tree = ast.parse(expression, mode='eval')

        # Evaluate safely
        result = _safe_eval(tree.body)

        # Format output: integers as int, floats with reasonable precision
        if isinstance(result, float):
            if result == int(result):
                return str(int(result))
            return f"{result:.6g}"
        return str(result)

    except ZeroDivisionError:
        return "Error: Division by zero"
    except ValueError as e:
        return f"Error: {e}"
    except SyntaxError:
        return f"Error: Invalid expression '{expression}'"
    except Exception as e:
        return f"Error: {e}"