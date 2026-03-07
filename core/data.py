from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from pathlib import Path
from typing import Any

from . import io_utils


@dataclass
class MeshData:
    verts: np.ndarray
    faces: np.ndarray
    name: str


class NpzData:

    def __init__(self, path: str | Path):
        if isinstance(path, str):
            path = Path(path)
        if not path.suffix.endswith("npz"):
            raise RuntimeError(f"Invalid file extension: {path.suffix} !")
        if path.is_dir():
            raise RuntimeError("Path must be a file path not a directory !")
        if not path.exists():
            raise RuntimeError(f"file not found at: {path}")
            
        self._path = path
        self._data = io_utils.read_npz(path)
    
    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    
    def __iter__(self):
        return iter(self._data)
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path: {self._path})"

    @property
    def data(self) -> dict:
        return self._data

    def get(self, key: str) -> Any:
        if key not in self._data:
            raise RuntimeError(f"Key {key} does not exist in {self._path}")
        return self._data[key]
    
    @property
    def path(self) -> Path:
        return self._path