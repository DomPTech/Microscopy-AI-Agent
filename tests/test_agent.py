import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings
settings.hf_cache_dir = "/lustre/isaac24/scratch/dpelaia/hf_cache/"

from app.agent.core import Agent

def main():
    # Initialize the agent with a specific model
    model_name = "Qwen/Qwen2.5-14B-Instruct"
    agent = Agent(model_id=model_name)

    settings.instrument_host = "localhost"
    settings.instrument_port = 9001
    settings.autoscript_port = 9001

    # Define the prompt to test
    prompt = """
    Execute an experiment to get an image on a mock microscope.
    """

    # Call the agent with the prompt
    response = agent.chat(prompt)

if __name__ == "__main__":
    main()