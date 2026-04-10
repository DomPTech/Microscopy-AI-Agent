import json
from pathlib import Path
from typing import List, Dict, Any, Union, Type
import numpy as np
import pandas as pd
import h5py
from ..base import DataConverter, HeuristicMismatchError, Filepath

class CsvConverter(DataConverter[Union[pd.DataFrame, Filepath]]):
    """
    Converts CSV files or DataFrames to strings.
    """

    @property
    def input_type(self) -> Type:
        return (pd.DataFrame, Path, str)

    def can_handle(self, data: Any) -> bool:
        if isinstance(data, pd.DataFrame):
            return True
        if isinstance(data, (Path, str)):
            return Path(data).suffix.lower() == ".csv"
        return False

    def convert(self, data: Union[pd.DataFrame, Filepath]) -> str:
        if isinstance(data, (Path, str)):
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"CSV file not found: {path}")
            df = pd.read_csv(str(path))
        else:
            df = data

        summary = {
            "columns": list(df.columns),
            "rows": len(df),
            "preview": df.head(5).to_dict(orient="records")
        }
        
        return json.dumps(summary, indent=2)

class Hdf5SummaryConverter(DataConverter[Union[h5py.File, h5py.Group, Filepath]]):
    """
    Converts HDF5 files or groups to hierarchical strings.
    """

    @property
    def input_type(self) -> Type:
        return (h5py.File, h5py.Group, Path, str)

    def can_handle(self, data: Any) -> bool:
        if isinstance(data, (h5py.File, h5py.Group)):
            return True
        if isinstance(data, (Path, str)):
            return Path(data).suffix.lower() in [".h5", ".hdf5"]
        return False

    def _summarize(self, item: Union[h5py.File, h5py.Group]) -> Dict[str, Any]:
        summary = {}
        for name, sub_item in item.items():
            if isinstance(sub_item, (h5py.File, h5py.Group)):
                summary[name] = {"type": "group", "children": self._summarize(sub_item)}
            elif isinstance(sub_item, h5py.Dataset):
                summary[name] = {
                    "type": "dataset",
                    "shape": list(sub_item.shape),
                    "dtype": str(sub_item.dtype)
                }
        return summary

    def convert(self, data: Union[h5py.File, h5py.Group, Filepath]) -> str:
        if isinstance(data, (Path, str)):
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"HDF5 file not found: {path}")
            with h5py.File(str(path), 'r') as f:
                full_summary = self._summarize(f)
        else:
            full_summary = self._summarize(data)

        return json.dumps(full_summary, indent=2)

class DictConverter(DataConverter[dict]):
    """Default converter for dictionaries."""
    @property
    def input_type(self) -> Type:
        return dict

    def convert(self, data: dict) -> str:
        return json.dumps(data, indent=2)