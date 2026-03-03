from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from .constants import logger, in_directory, out_directory


def dump_json(data: dict, output_path: str | Path):
    if isinstance(output_path, str):
        output_path = Path(output_path)
    
    try:
        with open(output_path.as_posix(), 'w') as f:
            json.dump(data, f, indent=4)
        logger.debug(f"Successfully saved JSON to '{output_path.as_posix()}'")
    except Exception:
        raise RuntimeError("Failed to write JSON file.")


def read_npz(npz_path: str | Path):
    if isinstance(npz_path, str):
        npz_path = Path(npz_path)
    
    try:
        logger.info(f"Read npz file: '{npz_path.as_posix()}'.")
        return np.load(npz_path.as_posix(), allow_pickle=True)
    except:
        raise RuntimeError("Error on read npz file.")


def get_input_from_name(file_name: str) -> Path:
    path = in_directory / f"{file_name}.npz"
    if not path.exists():
        raise RuntimeError(f'File "{file_name}.npz" not found !')
    
    return path


def get_output_from_name(file_name: str) -> Path:
    path = out_directory / file_name / "result.npz"
    if not path.exists():
        raise RuntimeError(f'No result file found for "{file_name}" !')
    
    return path
