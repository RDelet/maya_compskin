from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import List


class JointManager:

    def __init__(self, json_path: str | Path):
        if isinstance(json_path, str):
            json_path = Path(json_path)
        if not json_path.exists():
            raise RuntimeError(f"Joint positions file not found: {json_path}")

        with open(json_path, "r") as f:
            self._data = json.load(f)

        if "joints" not in self._data or len(self._data["joints"]) == 0:
            raise RuntimeError("JSON must contain a non-empty 'joints' list.")

    @property
    def positions(self) -> np.ndarray:
        """(P, 3) float32 array of joint world positions."""
        return np.array([j["position"] for j in self._data["joints"]], dtype=np.float32)

    @property
    def names(self) -> List[str]:
        return [j["name"] for j in self._data["joints"]]

    @property
    def count(self) -> int:
        return len(self._data["joints"])
