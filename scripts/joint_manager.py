from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import List

import igl
from .constants import logger


class JointManager:
    """
    Loads user-defined joint positions from a JSON file and computes
    surface-distance-based initial skinning weights.

    Expected JSON format:
    {
        "joints": [
            {"name": "jaw",      "position": [0.0, 12.5, 2.3]},
            {"name": "eye_left", "position": [-3.1, 14.0, 1.8]},
            ...
        ]
    }
    """

    def __init__(self, json_path: str | Path):
        if isinstance(json_path, str):
            json_path = Path(json_path)
        if not json_path.exists():
            raise RuntimeError(f"Joint positions file not found: {json_path}")

        with open(json_path, "r") as f:
            self._data = json.load(f)

        if "joints" not in self._data or len(self._data["joints"]) == 0:
            raise RuntimeError("JSON must contain a non-empty 'joints' list.")

    # --------------------------------------------------
    # Properties
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Weight initialisation
    # --------------------------------------------------

    def compute_initial_weights(
        self,
        rest_verts: np.ndarray,
        rest_faces: np.ndarray,
        max_influences: int = 8,
        use_geodesic: bool = True,
    ) -> np.ndarray:
        """
        Compute (N, P) normalised weight matrix from surface distances.

        Args:
            rest_verts:     (N, 3) vertex positions.
            rest_faces:     (F, 3) triangle indices (quads are split automatically).
            max_influences: Maximum non-zero weights per vertex.
            use_geodesic:   True  → exact geodesic via igl (slower, better quality).
                            False → euclidean fallback (fast).

        Returns:
            weights: (N, P) float32, rows sum to 1.
        """
        joint_positions = self.positions  # (P, 3)

        # Ensure triangulated mesh for igl
        tris = self._triangulate(rest_faces)

        # Find the mesh vertex closest to each joint position
        source_vertices = self._closest_vertex_indices(rest_verts, joint_positions)

        if use_geodesic:
            distances = self._geodesic_distances(rest_verts, tris, source_vertices)
        else:
            distances = self._euclidean_distances(rest_verts, joint_positions)

        return self._distances_to_weights(distances, max_influences)

    # --------------------------------------------------
    # Internals
    # --------------------------------------------------

    @staticmethod
    def _triangulate(faces: np.ndarray) -> np.ndarray:
        """Split quads into triangles if needed."""
        if faces.shape[1] == 3:
            return faces
        return np.concatenate([faces[:, :3], faces[:, [0, 2, 3]]], axis=0)

    @staticmethod
    def _closest_vertex_indices(verts: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Return (P,) array of vertex indices closest to each joint."""
        # (N, P, 3) → (N, P) distances → argmin over N axis → (P,)
        dists = np.linalg.norm(verts[:, None, :] - positions[None, :, :], axis=-1)
        return np.argmin(dists, axis=0).astype(np.intp)

    @staticmethod
    def _geodesic_distances(
        verts: np.ndarray, tris: np.ndarray, source_verts: np.ndarray
    ) -> np.ndarray:
        """(P, N) geodesic distance matrix via igl.exact_geodesic."""
        P = len(source_verts)
        N = len(verts)
        distances = np.zeros((P, N), dtype=np.float32)
        verts_f64 = verts.astype(np.float64)
        tris_i32 = tris.astype(np.int32)
        all_targets = np.arange(N, dtype=np.intp)

        for i, src_idx in enumerate(source_verts):
            logger.debug(f"Geodesic distance: joint {i + 1}/{P}")
            d = igl.exact_geodesic(
                verts_f64, tris_i32,
                np.array([src_idx], dtype=np.intp),
                all_targets,
            )
            distances[i] = d.astype(np.float32)

        return distances

    @staticmethod
    def _euclidean_distances(verts: np.ndarray, joint_positions: np.ndarray) -> np.ndarray:
        """(P, N) euclidean distance matrix."""
        return np.linalg.norm(
            verts[None, :, :] - joint_positions[:, None, :], axis=-1
        ).astype(np.float32)

    @staticmethod
    def _distances_to_weights(distances: np.ndarray, max_influences: int) -> np.ndarray:
        """
        Convert (P, N) distance matrix to (N, P) normalised weight matrix.
        Uses inverse-distance weighting with hard pruning to max_influences.
        """
        weights = (1.0 / (distances.T + 1e-8)).astype(np.float32)  # (N, P)

        # Prune: keep only the top-k weights per vertex
        if max_influences < weights.shape[1]:
            threshold_indices = np.argsort(weights, axis=1)[:, :-max_influences]
            rows = np.repeat(np.arange(len(weights)), max_influences if max_influences < weights.shape[1]
                             else 0)
            weights[
                np.arange(len(weights))[:, None],
                threshold_indices
            ] = 0.0

        # Normalise rows
        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return weights / row_sums