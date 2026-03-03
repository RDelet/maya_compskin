from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Settings:
    """Training settings for the Compressed Skinning model.

    Base parameters
        input_file  : NPZ file containing rest_verts, rest_faces, deltas.
        joint_file  : Optional JSON file with user-defined joint positions.
                      When provided, proxy joints are initialised at those
                      positions and skinning weights are seeded from
                      surface-distance to each joint.
        output_dir  : Directory where result.npz will be written.

    Skinning model parameters
        p_bones         : Number of proxy bones.
        max_influences  : Maximum non-zero skinning weights per vertex.
        total_nnz_brt   : Maximum number of non-zero entries in Brt.
                          Keeps transformations sparse so each joint only
                          affects the blendshapes it is anatomically
                          responsible for (unlike dense SSDR).

    Learning settings
        power       : Exponent for the reconstruction error term.
                      power=2 → standard MSE.  Higher values penalise large
                      outlier errors much more heavily, forcing the model to
                      capture fine surface details at the cost of average
                      quality.
        alpha       : Weight of the Laplacian smoothness regularisation.
                      High alpha → rigid/smooth deformations.
                      Low  alpha → allows highly localised, "chaotic" deforms.
        lr          : Adam learning rate.  Too large → training diverges;
                      too small → very slow convergence.
        iter1       : Iterations for pass 1 (no weight normalisation).
        iter2       : Iterations for pass 2 (with weight normalisation).
        seed        : Random seed for reproducibility.
        init_weight : Scale of the random initialisation for Brt.
                      Small values start near the identity (no deformation).
    """
    input_file: str
    output_dir: str

    joint_file: str | None = None
    p_bones: int = 40
    max_influences: int = 8
    total_nnz_brt: int = 6000
    power: int = 2
    alpha: float = 10.0
    lr: float = 1e-3
    iter1: int = 10000
    iter2: int = 10000
    seed: int = 12345
    init_weight: float = 1e-3
