from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class MeshData:
    verts: np.ndarray
    faces: np.ndarray
    name: str