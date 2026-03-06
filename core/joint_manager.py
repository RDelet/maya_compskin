from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np
from pathlib import Path
from typing import List, Optional

from .data import MeshData
from . import math as cp_math


@dataclass
class JointResult:
    joint_positions: np.ndarray
    joint_vertex_indices: np.ndarray
    deformation_score: np.ndarray
    deformation_mask: Optional[np.ndarray]

    def dump(self, path: str | Path):
        if isinstance(path, str):
            path = Path(path)

        data = {"joints": {"positions": self.joint_positions}}
        with open(path, "w") as f:
            json.dump(data, f, indent=4)


class JointManager:

    def __init__(self, path: str | Path):
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise RuntimeError(f"Joint positions filedoes not exists: {path}")

        self._path = path
        self._load()

    @property
    def count(self) -> int:
        return len(self._data["joints"])
    
    def _load(self):
        with open(self._path, "r") as f:
            self._data = json.load(f)
        if "joints" not in self._data or len(self._data["joints"]) == 0:
            raise RuntimeError("JSON must contain a non-empty 'joints' list.")

    @property
    def names(self) -> List[str]:
        return [j["name"] for j in self._data["joints"]]

    @property
    def path(self) -> Path:
        return self._path

    @property
    def positions(self) -> np.ndarray:
        """(P, 3) float32 array of joint world positions."""
        return np.array([j["position"] for j in self._data["joints"]], dtype=np.float32)


def place_joints_on_mesh(mesh : MeshData, joint_count: int,
                         feature_weight: float = 2.0,
                         deform_mask: Optional[np.ndarray] = None) -> JointResult:
    score = cp_math.compute_rest_score(mesh.verts, mesh.faces)
    joint_verts = cp_math.uniform_biased_fps(mesh.verts, score, joint_count,
                                             feature_weight=feature_weight,
                                             deform_mask=deform_mask)

    return JointResult(joint_positions=mesh.verts[joint_verts],
                       joint_vertex_indices = np.array(joint_verts),
                       deformation_score = score,
                       deformation_mask = deform_mask)
