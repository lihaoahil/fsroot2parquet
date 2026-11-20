"""Cut expressions of the config files, evaluated on columns.

An expression is Python arithmetic on column names, element-wise on NumPy arrays:

    (E_beam > 8.2) & (E_beam < 8.8)
    abs(M_Lambda - 1.115) < 0.010
    ((run < 70000) & (E_beam > 8.2)) | ((run >= 70000) & (E_beam > 8.0))

Use ``&``, ``|``, ``~`` with parentheses around comparisons (``and`` / ``or`` are accepted and mean the same).
Only names, numbers, arithmetic, comparisons and the functions below are allowed.
"""
from __future__ import annotations

import ast

import numpy as np

FUNCTIONS = {"abs": np.abs, "sqrt": np.sqrt, "minimum": np.minimum, "maximum": np.maximum, "cos": np.cos, "sin": np.sin}
_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Name, ast.Constant, ast.Load,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Invert, ast.BitAnd, ast.BitOr,
            ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq)


class _BoolToBit(ast.NodeTransformer):
    """a and b -> (a) & (b);  a or b -> (a) | (b);  not a -> ~(a)."""

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        op = ast.BitAnd() if isinstance(node.op, ast.And) else ast.BitOr()
        out = node.values[0]
        for value in node.values[1:]:
            out = ast.BinOp(out, op, value)
        return out

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        return ast.UnaryOp(ast.Invert(), node.operand) if isinstance(node.op, ast.Not) else node


def compile_expr(text: str):
    """Parsed and checked expression: (code, names it uses)."""
    tree = ast.fix_missing_locations(_BoolToBit().visit(ast.parse(str(text).strip(), mode="eval")))
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"not allowed in a cut expression: {type(node).__name__} in {text!r}")
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            raise ValueError(f"only plain function calls are allowed: {text!r}")
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return compile(tree, "<cut>", "eval"), names


def names_of(text: str) -> set[str]:
    return compile_expr(text)[1]


def evaluate(text: str, columns, functions=None):
    """The value of an expression; ``columns`` is any mapping from names to arrays.  A missing name raises KeyError."""
    code, names = compile_expr(text)
    scope = dict(FUNCTIONS) | dict(functions or {})
    for name in names:
        if name not in scope:
            scope[name] = columns[name]                         # KeyError: unknown column
    return eval(code, {"__builtins__": {}}, scope)             # noqa: S307 -- the tree was checked above


def all_of(conditions, columns, n: int, functions=None) -> np.ndarray:
    """Logical AND of a list of conditions, as a boolean array of length n."""
    mask = np.ones(n, dtype=bool)
    for text in conditions or []:
        mask &= np.asarray(evaluate(text, columns, functions), dtype=bool)
    return mask
