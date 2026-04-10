import io
from pathlib import Path
from typing import List, Type, Union, Any
import numpy as np
import tifffile
from PIL import Image
from ..base import DataConverter, HeuristicMismatchError, Filepath

class TiffConverter(DataConverter[Filepath]):
    """
    Converts TIFF files to PIL.Image.Image.
    """

    @property
    def input_type(self) -> Type:
        return (Filepath)

    def can_handle(self, data: Any) -> bool:
        if isinstance(data, (Path, str)):
            p = Path(data)
            return p.suffix.lower() in [".tiff", ".tif"]
        return False

    def convert(self, data: Filepath) -> Image.Image:
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"TIFF file not found: {path}")
            
        img_data = tifffile.imread(str(path))
        
        # Normalize to 8-bit if needed for PIL consumption
        if img_data.dtype != np.uint8:
            img_data = ((img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-5) * 255).astype(np.uint8)
            
        return Image.fromarray(img_data)

class NumpyImageConverter(DataConverter[Union[np.ndarray, Filepath]]):
    """
    Uses heuristics to convert numpy arrays (or .npy files) to PIL.Image.Image.
    """

    @property
    def input_type(self) -> Type:
        return (np.ndarray, Filepath)

    def can_handle(self, data: Any) -> bool:
        if isinstance(data, np.ndarray):
            return True
        if isinstance(data, (Path, str)):
            return Path(data).suffix.lower() == ".npy"
        return False

    def convert(self, data: Union[np.ndarray, Filepath]) -> Image.Image:
        if isinstance(data, (Path, str)):
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"NPY file not found: {path}")
            arr = np.load(str(path))
        else:
            arr = data

        # Heuristic check: 2D or 3D (small channels)
        is_plausible_image = False
        if arr.ndim == 2:
            is_plausible_image = True
        elif arr.ndim == 3 and arr.shape[-1] in [1, 3, 4]:
            is_plausible_image = True

        if not is_plausible_image:
            raise HeuristicMismatchError(f"Numpy array shape {arr.shape} is not image-like.")

        # Normalize to 8-bit if needed for PIL consumption
        if arr.dtype != np.uint8:
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-5) * 255).astype(np.uint8)

        return Image.fromarray(arr)