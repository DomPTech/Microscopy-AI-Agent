import ast
from ast import AST, NodeTransformer, Call, Name

from smolagents import Tool


class _KwargTransformer(NodeTransformer):
    """
    AST transformer that converts positional arguments to keyword arguments for calls to static tools.
    """
    def __init__(self, static_tools):
        self.static_tools = static_tools
        super().__init__()

    def visit_Call(self, node: Call) -> AST:
        self.generic_visit(node)
        if isinstance(node.func, Name):
            func_name = node.func.id
            if func_name in self.static_tools:
                tool = self.static_tools[func_name]
                if isinstance(tool, Tool) and node.args:
                    input_keys = list(tool.inputs.keys())
                    if len(node.args) <= len(input_keys):
                        existing_kw = {kw.arg for kw in node.keywords if kw.arg}
                        for i, arg in enumerate(node.args):
                            key = input_keys[i]
                            if key not in existing_kw:
                                node.keywords.append(ast.keyword(arg=key, value=arg))
                        node.args = []
        return node
