import os
import unittest
import numpy as np
import pandas as pd
import tifffile
import h5py
from pathlib import Path
from PIL import Image
import json
import base64
import io
from atomonous.data.factory import ConverterFactory
from atomonous.data.base import DataConverter
from typing import Type

# Mock class
class MicroscopeImage:
    def __init__(self, data: np.ndarray, name: str):
        self.data = data
        self.name = name

# Mock Converter returning PIL Image
class MicroscopeImageConverter(DataConverter[MicroscopeImage]):
    @property
    def input_type(self) -> Type:
        return MicroscopeImage

    def convert(self, data: MicroscopeImage) -> Image.Image:
        # Normalize to 8-bit if needed for PIL consumption
        arr = data.data
        if arr.dtype != np.uint8:
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-5) * 255).astype(np.uint8)
        return Image.fromarray(arr)

class TestDal(unittest.TestCase):
    def setUp(self):
        self.default_factory = ConverterFactory(register_default=True)

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("/Users/dominick/Microscopy-AI-Agent-Demo/tmp_test_simplified")
        cls.test_dir.mkdir(exist_ok=True)
        
        # Prep sample files
        cls.csv_path = cls.test_dir / "test.csv"
        pd.DataFrame({"X": [1, 2]}).to_csv(cls.csv_path, index=False)
        
        cls.tiff_path = cls.test_dir / "test.tiff"
        img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        tifffile.imwrite(str(cls.tiff_path), img)

    @classmethod
    def tearDownClass(cls):
        for file in cls.test_dir.glob("*"):
            file.unlink()
        cls.test_dir.rmdir()

    def test_csv_to_string(self):
        result = self.default_factory.convert(self.csv_path)
        self.assertIsInstance(result, str)
        self.assertIn("columns", result)

    def test_tiff_to_pil(self):
        result = self.default_factory.convert(self.tiff_path)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (50, 50))

    def test_custom_class_to_pil(self):
        self.default_factory.register_converter(MicroscopeImageConverter())
        mi = MicroscopeImage(np.random.randint(0, 255, (64, 64), dtype=np.uint8), "Test")
        result = self.default_factory.convert(mi)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (64, 64))

    def test_dict_to_string(self):
        result = self.default_factory.convert({"status": "ok"})
        self.assertIsInstance(result, str)
        self.assertIn('"status": "ok"', result)

if __name__ == "__main__":
    unittest.main()
