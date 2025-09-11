from __future__ import annotations

from pathlib import Path

from .utils import dump_json, read_npz


class Converter:
    
    def __init__(self, npz_path: str | Path):
        if isinstance(npz_path, str):
            npz_path = Path(npz_path)
        if npz_path.is_dir():
            raise RuntimeError("Path must be a file path not a directory !")
        if not npz_path.exists():
            raise RuntimeError(f"file not found at: {npz_path}")
        
        self.path = npz_path

    def get_output_path(npz_path: str | Path, extension: str) -> Path:
        if isinstance(npz_path, str):
            npz_path = Path(npz_path)

        return npz_path.with_suffix(extension)

    def to_json(self):
        
        data = {}
        npz_data = read_npz(self.path)
        for key in npz_data.keys():
            array = npz_data[key]
            if array.dtype == 'O':
                data[key] = array.item()
            else:
                data[key] = array.tolist()

        dump_json(data, self.get_output_path(self.path, '.json'))
