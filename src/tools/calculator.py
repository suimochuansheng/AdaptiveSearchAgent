"""安全数学计算器工具 — 包装 math_safe_calc，标准化输入输出。"""

from src.utils.math_safe_calc import (
    MathEvalError,
    SecurityError,
    safe_math_calculate,
)


async def calculate_math(expression: str) -> str:
    """安全计算数学表达式，返回人类可读的结果字符串。

    Args:
        expression: 数学表达式，如 "2*x**2 + 3*x - 5"（x 已替换为具体值）。

    Returns:
        计算结果字符串。异常时返回错误描述。
    """
    expr = expression.strip()
    try:
        result = safe_math_calculate(expr)
        # 整数结果不显示 .0
        if isinstance(result, float) and result == int(result):
            result = int(result)
        return f"{expr} = {result}"
    except SecurityError as e:
        return f"表达式包含不允许的操作：{e}"
    except MathEvalError as e:
        return f"数学计算错误：{e}"
    except SyntaxError as e:
        return f"表达式语法错误：{e}"
