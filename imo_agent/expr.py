"""Guard expressions.

Pipeline configs are shared and edited by hand, so guard conditions are
evaluated through a whitelist of AST nodes -- comparisons, boolean ops,
names and literals -- rather than eval().
"""

from __future__ import annotations

import ast
import operator
from typing import Any

_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


class ExprError(ValueError):
    pass


def evaluate(expression: str, names: dict[str, Any]) -> bool:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExprError(f"cannot parse guard expression {expression!r}: {exc}") from exc
    return bool(_eval(tree.body, names, expression))


def identifiers(expression: str) -> set[str]:
    tree = ast.parse(expression, mode="eval")
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def _eval(node: ast.AST, names: dict[str, Any], src: str) -> Any:
    if isinstance(node, ast.BoolOp):
        values = [_eval(v, names, src) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        return any(values)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval(node.operand, names, src)

    if isinstance(node, ast.Compare):
        left = _eval(node.left, names, src)
        for op, comparator in zip(node.ops, node.comparators):
            fn = _COMPARE.get(type(op))
            if fn is None:
                raise ExprError(f"operator {type(op).__name__} not allowed in {src!r}")
            right = _eval(comparator, names, src)
            if not fn(left, right):
                return False
            left = right
        return True

    if isinstance(node, ast.Name):
        if node.id not in names:
            known = ", ".join(sorted(names)) or "(none)"
            raise ExprError(f"unknown name {node.id!r} in {src!r}; available: {known}")
        return names[node.id]

    if isinstance(node, ast.Constant):
        return node.value

    raise ExprError(f"{type(node).__name__} not allowed in guard expression {src!r}")
