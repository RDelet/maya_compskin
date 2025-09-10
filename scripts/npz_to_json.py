from __future__ import annotations

from pathlib import Path

from .utils import dump_json, read_npz


def get_output_path(npz_path: str | Path, extension: str = '.json') -> Path:
    if isinstance(npz_path, str):
        npz_path = Path(npz_path)

    return npz_path.with_suffix(extension)


def run(npz_path: str | Path):
    
    if isinstance(npz_path, str):
        npz_path = Path(npz_path)
    
    data = {}
    npz_data = read_npz(npz_path)
    for key in npz_data.keys():
        array = npz_data[key]
        if array.dtype == 'O':
            data[key] = array.item()
        else:
            data[key] = array.tolist()

    dump_json(data, get_output_path(npz_path))
