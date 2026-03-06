from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

_root_path= Path(__file__)
sys.path.insert(0, str(_root_path.parent.parent))

from maya_compskin.core import constants, io_utils
from maya_compskin.core.constants import logger
from maya_compskin.core.settings import Settings
from maya_compskin.core.trainer import Trainer


def main():

    faces = [# "bowen",
             # "jupiter",
             # "proteus",
             "aura"]
    for face in faces:
        try:
            npz_path = io_utils.get_input_from_name(face)
        except Exception as e:
            logger.error(f"Impossible to load {face}: {e}")
            continue

        settings = Settings(input_file=str(npz_path),
                            output_dir=constants.out_directory / npz_path.stem,
                            joint_file=npz_path.parent / f"{face}Joints.json",
                            p_bones=200,
                            max_influences=8,
                            total_nnz_brt=6000,
                            power=2,
                            alpha=50,
                            lr=1e-3,
                            iter1=20000,
                            iter2=20000,
                            seed=12345,
                            init_weight=1e-6)

        try:
            logger.info(f"--- Starting Programmatic Compressed Skinning Training for {npz_path.stem} ---")
            trainer = Trainer(settings)
            trainer.train()
            trainer.save_results()
            logger.info(f"--- Training of {npz_path.stem} finished successfully ---")
            logger.info(f"Results saved in: {os.path.join(settings.output_dir, 'result.npz')}")
        except Exception as e:
            logger.exception("An error occurred during training")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(e)
        logger.debug(traceback.format_exc())
