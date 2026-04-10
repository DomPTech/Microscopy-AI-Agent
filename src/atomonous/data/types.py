from pathlib import Path
from typing import Union
from PIL.Image import Image

# A type that can be ingested by the AI
type AIFormat = Union[str, Image]

# A type that represents a filepath
type Filepath = Union[str, Path]