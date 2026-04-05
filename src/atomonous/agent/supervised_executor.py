import ast

from smolagents import LocalPythonExecutor
from atomonous.config import settings

class SupervisedExecutor(LocalPythonExecutor):
    """
    A supervised executor that allows for human intervention at key steps.
    This can be used during development to monitor the agent's behavior and 
    ensure it is making reasonable decisions before allowing it to execute actions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.user_prompt = "Please provide input: "
        self.confirmation_prompt = "Do you want to proceed? (y/n): "

        # Tools that require explicit approval before execution.
        self.dangerous_tools = {
            "set_beam_current",
            "blank_beam",
            "unblank_beam",
            "place_beam",
            "execute_workflow",
        }

    def _is_autorun_enabled(self) -> bool:
        return settings.agent_autorun

    def _get_called_tool_names(self, code_action: str) -> list[str]:
        """
        Return tool names called in the code snippet that are in the dangerous_tools list.
        Only names present in this executor's static tools are considered.
        """
        static_tools = getattr(self, "static_tools", {}) or {}
        tool_names = set(static_tools.keys())
        if not tool_names:
            return []

        called = set()
        try:
            tree = ast.parse(code_action)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    if isinstance(func, ast.Name):
                        called.add(func.id)
                    elif isinstance(func, ast.Attribute):
                        called.add(func.attr)
        except SyntaxError:
            for name in tool_names:
                if f"{name}(" in code_action:
                    called.add(name)

        return sorted(called & tool_names & self.dangerous_tools)

    def request_user_input(self, prompt=None):
        """
        Request input from the user with a custom or default prompt.
        """
        prompt = prompt or self.user_prompt
        return input(prompt)

    def request_confirmation(self, prompt=None):
        """
        Request confirmation from the user with a custom or default prompt.
        """
        prompt = prompt or self.confirmation_prompt
        while True:
            try:
                response = input(prompt).strip().lower()
            except EOFError:
                # Non-interactive environment: fail closed.
                print("No interactive input available. Denying execution by default.")
                return False

            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

    def __call__(self, code_action):
        """
        Execute code actions with per-tool approval, unless autorun is enabled.
        """
        if not self._is_autorun_enabled():
            called_tools = self._get_called_tool_names(code_action)
            for func in called_tools:
                print(f"The agent is trying to call this tool: {func}")
                if not self.request_confirmation(f"Approve tool call '{func}'? (y/n): "):
                    msg = f"Execution aborted by user. Tool call '{func}' was not approved."
                    print(msg)
                    return msg

        return super().__call__(code_action)