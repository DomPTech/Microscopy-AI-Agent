from smolagents import CodeAgent, WebSearchTool, TransformersModel

class Agent:
    def __init__(self):
        self.model = model = TransformersModel(
            model_id="Qwen/Qwen3-Next-80B-A3B-Thinking",
            max_new_tokens=4096,
            device_map="auto"
        )
        self.agent = CodeAgent(tools=[WebSearchTool()], model=model, stream_outputs=True)

    def chat(self, query: str) -> str:
        """
        Process user input and return a response.
        """
        self.agent.run(query)
        return "This is a dummy agent response."
