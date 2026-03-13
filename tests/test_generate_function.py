import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings
settings.hf_cache_dir = "/lustre/isaac24/scratch/dpelaia/hf_cache/"
from app.agent.core import Agent

agent = Agent.from_model_id(model_id="Qwen/Qwen2.5-0.5B-Instruct")

# Consume the generator to stream output and get final response
for output in agent.generate("Explain microscopy to me!"):
    if isinstance(output, tuple):
        tag, text = output
        if tag == "final":
            print(f"\n[Final] {text}")
        elif tag == "error":
            print(f"\n[Error] {text}")
    else:
        print(output, end="", flush=True)