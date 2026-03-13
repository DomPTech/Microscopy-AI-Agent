import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings
settings.hf_cache_dir = "/lustre/isaac24/scratch/dpelaia/hf_cache/"

from app.agent.core import Agent
from app.tools.microscopy import close_microscope, connect_client, start_server
from smolagents import LiteLLMModel
from dotenv import load_dotenv

load_dotenv()

model = LiteLLMModel(
    model_id="openai/meta-llama/llama-3.3-70b-instruct",  # or any Novita model slug
    api_base="https://api.novita.ai/v3/openai",
    api_key=os.environ.get("NOVITA_API_KEY")
)

def _require_success(result: str, expected_substrings: tuple[str, ...], step_name: str) -> str:
    if not any(substring in result for substring in expected_substrings):
        raise RuntimeError(f"{step_name} failed: {result}")
    return result

def main():
    try:
        start_result = start_server(mode="mock")
        print(start_result)
        _require_success(
            start_result,
            (
                "Started PyTango asyncroscopy servers",
                "PyTango asyncroscopy servers already running",
            ),
            "Server startup",
        )

        connect_result = connect_client()
        print(connect_result)
        _require_success(
            connect_result,
            ("Connected to PyTango microscope",),
            "Microscope connection",
        )

        agent = Agent(model=model)

        prompt = input("Enter a command: ")
        response = agent.chat(prompt)
        print(response)
    finally:
        print(close_microscope())

if __name__ == "__main__":
    main()