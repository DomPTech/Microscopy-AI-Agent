from smolagents import LocalPythonExecutor

class SupervisedExecutor(LocalPythonExecutor):
    """
    A supervised executor that allows for human intervention at key steps.
    This can be used during development to monitor the agent's behavior and 
    ensure it is making reasonable decisions before allowing it to execute actions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Variables for user input requests
        self.user_prompt = "Please provide input: "
        self.confirmation_prompt = "Do you want to proceed? (y/n): "
        # List of dangerous functions
        self.dangerous_functions = ["submit_experiment"]

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
            response = input(prompt).strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

    def __call__(self, code_action):
        """
        Execute the code action, but prompt the user if a dangerous function is detected.
        """
        for func in self.dangerous_functions:
            if func in code_action:
                print(f"The agent is trying to call this function: {func}")
                if not self.request_confirmation("Do you want to proceed with this action? (y/n): "):
                    print("Execution aborted by user.")
                    return None
        return super().__call__(code_action)