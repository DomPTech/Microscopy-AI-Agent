import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings
settings.hf_cache_dir = "/lustre/isaac24/scratch/dpelaia/hf_cache/"

from app.agent.core import Agent

def main():
    agent = Agent(model_id="Qwen/Qwen2.5-32B-Instruct")

    settings.instrument_host = "localhost"
    settings.instrument_port = 9001
    settings.autoscript_port = 9001

    prompt = """
    Get an image on a mock microscope.
    """

    response = agent.chat(prompt)

if __name__ == "__main__":
    main()