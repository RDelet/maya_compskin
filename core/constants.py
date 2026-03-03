from pathlib import Path

import logging


_current_path = Path(__file__)

module_path = _current_path.parent.parent
in_directory = module_path / "in"
out_directory = module_path / "out"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(_current_path.parent.name)
logger.setLevel(logging.DEBUG)
