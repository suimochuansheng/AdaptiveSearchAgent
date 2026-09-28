"""安全数学计算器：基于 AST 白名单的数学表达式求值模块。

核心原则：
    绝对禁止使用 eval / exec，所有表达式通过 Python AST 语法树解析后，
    在白名单约束下递归求值。任何不在白名单内的节点/函数/语法都会立即抛出异常。

白名单设计：
    - 节点类型：Constant, BinOp, UnaryOp, Call, Name, Load, Expression
    - 运算符：Add(+), Sub(-), Mult(*), Div(/), Pow(**), USub(-x)
    - 数学函数：sin, cos, tan, sqrt, log, exp, abs
    - 超出白名单的任何 AST 类型 → SecurityError

使用方式：
    from src.utils.math_safe_calc import safe_math_calculate
    result = safe_math_calculate("2 + 3 * sin(0.5)")
"""

import ast
import math
import operator
from typing import Any

# ═══════════════════════════════════════════════════════════════════════
# 白名单定义
# ═══════════════════════════════════════════════════════════════════════

# AST 节点白名单：只有这 9 种节点类型允许出现在表达式中
# Expression  ─ 顶层包装（ast.parse 自动生成）
# Constant    ─ 数字字面量（如 3, 2.5）
# BinOp       ─ 二元运算（如 a + b）
# UnaryOp     ─ 一元运算（如 -x）
# Call        ─ 函数调用（如 sin(0.5)）
# Name        ─ 名称引用（函数名，必须是 Load 上下文）
# Load        ─ 变量读取上下文（配合 Name 使用）
# Add/Sub/Mult/Div/Pow/USub ─ 具体运算符（嵌在 BinOp/UnaryOp 内部）
_ALLOWED_NODE_TYPES: frozenset[type[ast.AST]] = frozenset(
    {
        ast.Expression,
        ast.Constant,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        # 运算符节点自身：
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
    }
)

# 安全数学函数白名单：映射函数名 → Python math 实现
# 注意 abs 使用内置函数，其余来自 math 模块
_ALLOWED_FUNCTIONS: dict[str, Any] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "log": math.log,  # 自然对数 ln(x)，单参数
    "exp": math.exp,
    "abs": abs,  # 内置绝对值
}

# 二元运算符映射：AST 运算符节点类型 → Python operator 函数
_BINOP_MAP: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,  # 真除法，返回 float
    ast.Pow: operator.pow,
}

# 一元运算符映射
_UNARYOP_MAP: dict[type[ast.unaryop], Any] = {
    ast.USub: operator.neg,  # 取负：-x
}


# ═══════════════════════════════════════════════════════════════════════
# 自定义异常
# ═══════════════════════════════════════════════════════════════════════


class MathSafeCalcError(Exception):
    """安全计算器异常基类。"""


class SecurityError(MathSafeCalcError):
    """安全违规：表达式包含白名单外的语法/函数/操作。"""


class MathEvalError(MathSafeCalcError):
    """数学求值错误：如除零、负数开方、log 非正数等。"""


# ═══════════════════════════════════════════════════════════════════════
# AST 白名单校验（在求值前先全面检查，尽早发现问题）
# ═══════════════════════════════════════════════════════════════════════


def _validate_constant(node: ast.Constant) -> None:
    """校验常量节点：只允许 int / float 数字。"""
    if not isinstance(node.value, int | float):
        raise SecurityError(f"不允许的常量类型: {type(node.value).__name__}，只允许数字。")


def _validate_call(node: ast.Call) -> None:
    """校验函数调用节点：函数名必须在白名单中，且不能有关键字参数。"""
    if not isinstance(node.func, ast.Name):
        raise SecurityError("只允许直接调用函数名（如 sin(0.5)），不支持模块前缀。")
    func_name = node.func.id
    if func_name not in _ALLOWED_FUNCTIONS:
        raise SecurityError(
            f"函数 '{func_name}' 不在白名单中。允许的函数: {', '.join(sorted(_ALLOWED_FUNCTIONS))}"
        )
    if node.keywords:
        raise SecurityError("函数调用不支持关键字参数。")
    for arg in node.args:
        _validate_ast(arg)


def _validate_ast(node: ast.AST) -> None:
    """递归遍历整个 AST 树，确保每个节点都在白名单内。

    这层校验是安全的第一道防线：在求值之前就拦截所有非法语法，
    确保后续 _eval_node 永远不会遇到非预期节点。
    """
    node_type = type(node)

    # 第一步：节点类型必须在白名单中
    if node_type not in _ALLOWED_NODE_TYPES:
        raise SecurityError(
            f"不允许的语法节点: {node_type.__name__}。"
            f"表达式只能包含数字、四则运算、乘方和 sin/cos/tan/sqrt/log/exp/abs 函数。"
        )

    # 第二步：按类型分派到对应的校验子函数
    if isinstance(node, ast.Constant):
        _validate_constant(node)
    elif isinstance(node, ast.BinOp):
        _validate_ast(node.left)
        _validate_ast(node.right)
        if type(node.op) not in _BINOP_MAP:
            raise SecurityError(f"不允许的二元运算符: {type(node.op).__name__}")
    elif isinstance(node, ast.UnaryOp):
        _validate_ast(node.operand)
        if type(node.op) not in _UNARYOP_MAP:
            raise SecurityError(f"不允许的一元运算符: {type(node.op).__name__}")
    elif isinstance(node, ast.Call):
        _validate_call(node)
    elif isinstance(node, ast.Name):
        raise SecurityError(f"不允许的变量引用: '{node.id}'。表达式只能包含数字和数学函数。")
    elif isinstance(node, ast.Expression):
        _validate_ast(node.body)


# ═══════════════════════════════════════════════════════════════════════
# AST 递归求值
# ═══════════════════════════════════════════════════════════════════════


def _eval_node(node: ast.AST) -> float:
    """在白名单校验通过后，递归求值 AST 节点。

    所有节点类型已在 _validate_ast 中验证，此处不再做类型检查，
    仅执行对应的数学计算。如果计算过程发生数学异常（除零等），
    捕获并转为 MathEvalError。
    """
    if isinstance(node, ast.Expression):
        # 顶层 Expression(body=...) → 求值 body
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        # 数字字面量 → 统一转为 float 返回
        # _validate_ast 已保证 node.value 是 int/float，但 mypy 无法跨函数窄化类型
        return float(node.value)  # type: ignore[arg-type]

    if isinstance(node, ast.BinOp):
        # 二元运算：先递归求值左右子树，再执行运算符
        left_val = _eval_node(node.left)
        right_val = _eval_node(node.right)
        op_func = _BINOP_MAP[type(node.op)]
        try:
            return float(op_func(left_val, right_val))
        except ZeroDivisionError as e:
            raise MathEvalError("数学错误：除以零。") from e
        except (ValueError, OverflowError) as e:
            raise MathEvalError(f"数学计算错误: {e}") from e

    if isinstance(node, ast.UnaryOp):
        # 一元运算：目前仅支持取负 (-x)
        op_func = _UNARYOP_MAP[type(node.op)]
        operand_val = _eval_node(node.operand)
        return float(op_func(operand_val))

    if isinstance(node, ast.Call):
        # 函数调用：从白名单获取 Python 实现，传入已求值的实参
        func_name = node.func.id  # type: ignore[attr-defined]
        func_impl = _ALLOWED_FUNCTIONS[func_name]
        args = [_eval_node(arg) for arg in node.args]
        try:
            result = func_impl(*args)
            return float(result)
        except (ValueError, ZeroDivisionError) as e:
            raise MathEvalError(f"数学计算错误: {e}") from e

    # 理论上不会到达这里（_validate_ast 已过滤），但作为最后防线保留
    raise SecurityError(f"内部错误：未处理的节点类型 {type(node).__name__}")


# ═══════════════════════════════════════════════════════════════════════
# 公开入口
# ═══════════════════════════════════════════════════════════════════════


def safe_math_calculate(expression: str) -> float:
    """安全地计算数学表达式，返回浮点结果。

    基于 Python AST 白名单解析，绝对不使用 eval / exec。
    只允许：数字、+、-、*、/、**、负号、以及 sin/cos/tan/sqrt/log/exp/abs 函数。

    Args:
        expression: 数学表达式字符串，例如 "2 + 3 * sin(0.5)"。

    Returns:
        float: 计算结果。

    Raises:
        SecurityError: 表达式包含非法语法/函数/变量。
        MathEvalError: 数学计算错误（如除零、负数开方等）。
        SyntaxError: 表达式语法不合法（Python 解析器层面）。
    """
    if not expression or not expression.strip():
        raise MathEvalError("表达式不能为空。")

    # ── 步骤 1：Python 解析为 AST ────────────────────────────────
    # ast.parse 本身是安全的：它只生成语法树，不执行任何代码
    # 将数学公式转成AST节点对象树-语法树
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as e:
        # 只要捕获了SyntaxError 错误，就用下面raise的方式报错
        raise SyntaxError(f"表达式语法错误: {e.msg} (位置: 第{e.lineno}行 第{e.offset}列)") from e

    # ── 步骤 2：白名单安全校验 ───────────────────────────────────
    # 递归遍历整个 AST，确保每个节点都在白名单内
    _validate_ast(tree)

    # ── 步骤 3：递归求值 ─────────────────────────────────────────
    # 校验通过后才开始计算
    try:
        return _eval_node(tree)
    except MathEvalError:
        # 数学异常直接向上抛
        raise
    except Exception as e:
        raise SecurityError(f"求值过程中发生意外错误: {type(e).__name__}: {e}") from e
