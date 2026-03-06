from __future__ import annotations

import numpy as np
import scipy as sp
import torch
import time
import igl
import os

from . import math as cp_math
from .constants import logger
from .settings import Settings
from .joint_manager import JointManager


class Trainer:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(self.settings.seed)

        # Learned tensors
        self.W = None   # skinning weights
        self.Brt = None   # Lie-algebra coefficients
        self.TR = None   # fixed basis matrices

        # Data tensors
        self.A = None   # ground-truth deltas (target)
        self.L = None   # Laplacian matrix
        self.rest_pose = None
        self.quads = None
        self.joint_positions = None
        self._init_weights = None

        self.n_bs = 0   # number of blendshapes
        self.N = 0   # number of vertices

        # Diagnostics 
        self.loss_list   = []
        self.abserr_list = []

        logger.info(f"Using device: {self.device}")
        self._load_data()
        self._initialize_tensors()

    def _load_data(self):
        logger.info(f"Loading data from {self.settings.input_file}...")
        self.npb_data = np.load(self.settings.input_file, allow_pickle=True)

        # Blendshape deltas → matrix A
        self.n_bs = self.npb_data["deltas"].shape[0]
        deltas = self.npb_data["deltas"].transpose(1, 0, 2).reshape(-1, self.n_bs * 3).transpose()
        self.A = torch.from_numpy(deltas).float().to(self.device)
        self.N = self.A.shape[1]
        self.quads = self.npb_data["rest_faces"]
        rest = self.npb_data["rest_verts"]
        logger.info(f"Loaded model with {self.N} vertices and {self.n_bs} blendshapes.")

        # Laplacian regularisation matrix L
        # The Laplacian encodes the mesh connectivity.  Applied to a set of
        # per-vertex vectors, L * V measures how much each vertex deviates
        # from the average of its 1-ring neighbours ("how non-smooth" the
        # deformation is).  Minimising ||L * BX||² during training forces
        # adjacent vertices to move similarly → smooth, skin-like deforms.
        #
        # We use the normalised Laplacian:
        #   L[i, i] = -1
        #   L[i, k] = 1/|N(i)|  for k in the 1-ring of i
        #   L[i, k] = 0          otherwise
        #
        # This normalisation makes the smoothness penalty scale-independent
        # (a vertex with many neighbours is treated the same as one with few).
        adj = igl.adjacency_matrix(self.quads)
        adj_diag = np.array(np.sum(adj, axis=1)).squeeze()
        Lg = sp.sparse.diags(1 / adj_diag) @ (adj - sp.sparse.diags(adj_diag))
        self.L = torch.from_numpy(Lg.todense()).float().to(self.device).to_sparse()

        # Rest pose: centre and convert to homogeneous coordinates
        # Centering subtracts the mesh barycentre so the origin sits at the
        # centre of the face.  This improves numerical stability (smaller
        # numbers → smaller gradients → more stable training) and means the
        # rotation generators in TR act around a meaningful central point by
        # default when no joint positions are given.
        rest_centered = cp_math.add_homog_coordinate(rest - rest.mean(axis=0), 1)
        self.rest_pose = torch.from_numpy(rest_centered).float().to(self.device)

        # Joint positions et poids initiaux
        self.joint_positions = None
        self._init_weights = None
        if self.settings.joint_file is not None:
            jnt_manager = JointManager(self.settings.joint_file)
            if self.settings.p_bones != jnt_manager.count:
                self.settings.p_bones = jnt_manager.count

            logger.info(f"Computing initial weights from {jnt_manager.count} joint positions...")
            self.joint_positions = jnt_manager.positions
            self._init_weights = cp_math.compute_initial_weights(
                rest,
                jnt_manager.positions,
                max_influences=self.settings.max_influences,
            )

    def _initialize_tensors(self):
        logger.info("Initializing tensors for optimization...")

        self.TR = cp_math.buildTR(self.device)

        # Brt: Lie-algebra coefficients, shape (6, n_bs, P, 1, 1)
        # Initialised with small random values so training starts near the
        # identity transform (almost no deformation). init_weight controls
        # this scale: very small → very slow start but more stable;
        # larger → faster initial movement but risk of early divergence.
        self.Brt = (
            (self.settings.init_weight
             * torch.randn((6, self.n_bs, self.settings.p_bones, 1, 1)))
            .clone().float().to(self.device).requires_grad_()
        )

        if self._init_weights is not None:
            # Seed from surface-distance weights computed in _load_data.
            # _init_weights is (N, P); W expects (P, N) → transpose.
            # A tiny amount of noise breaks perfect symmetry so the optimiser
            # can differentiate between bones that start with identical weights.
            w0 = torch.from_numpy(self._init_weights.T).float().to(self.device)
            self.W = (w0 + 1e-8 * torch.randn_like(w0)).clone().requires_grad_()
            logger.info("Weights initialised from joint positions.")
        else:
            # No joint file: start from near-zero random weights.
            # The optimiser will discover which bones should influence which
            # vertices entirely from the blendshape data.
            self.W = (
                1e-8 * torch.randn(self.settings.p_bones, self.N)
            ).clone().float().to(self.device).requires_grad_()

    def _train_pass(self, num_iter, optimizer, normalizeW=False):
        """Run one training pass of num_iter gradient-descent steps.

        normalizeW=False  (pass 1): W can be any non-negative values.
                                    The model learns the overall scale of
                                    weights freely, which helps early
                                    exploration of the solution space.

        normalizeW=True   (pass 2): Each vertex's weights are normalised to
                                    sum to 1, enforcing the partition-of-unity
                                    constraint required by standard LBS.
                                    This ensures the final rig behaves
                                    correctly when driven in Maya.
        """
        st = time.time()
        for i in range(num_iter):

            # Weight normalisation
            # Clamp the column sums to a small positive value to avoid
            # division by zero for vertices that have no active bone yet.
            if normalizeW:
                col_sums = self.W.sum(axis=0).clamp(min=1e-8)
                Wn = self.W / col_sums
            else:
                Wn = self.W

            # Forward pass: predicted blendshape deltas
            # BX shape: (n_bs*3, N)
            # Each entry BX[s*3 + c, v] is the predicted displacement of
            # vertex v along axis c (0=x,1=y,2=z) for blendshape s.
            BX, _, _ = cp_math.compBX(Wn, self.Brt, self.TR, self.n_bs, self.rest_pose)

            # Reconstruction loss
            # Measures how well BX approximates the ground-truth deltas A.
            #
            # Standard MSE would be:  mean( (BX - A)² )
            #
            # Here we use a generalised Lp norm with exponent `power`:
            #   loss_recon = ( mean( |BX - A|^power ) )^(2/power)
            #
            # The outer ^(2/power) keeps the loss in "distance squared" units
            # regardless of power, so the learning rate stays comparable
            # across different power values.
            #
            # With power=2  → standard MSE (treats all errors equally).
            # With power=12 → focuses almost entirely on the single worst
            #                 vertex, forcing the model to eliminate outliers.
            weighed_error = BX - self.A
            loss = weighed_error.pow(self.settings.power).mean().pow(2 / self.settings.power)

            # Laplacian smoothness regularisation
            # Adds a penalty proportional to how "non-smooth" the predicted
            # deformations are across the mesh surface.
            #
            #   reg = alpha * mean( ||L * BX^T||² )
            #
            # L * BX^T computes, for each vertex, the difference between its
            # predicted displacement and the average of its neighbours'.
            # Squaring and averaging gives a scalar penalty.
            #
            # This prevents the model from learning implausible deformations
            # where adjacent vertices fly off in opposite directions.
            if self.settings.alpha is not None:
                loss += self.settings.alpha * (self.L @ BX.transpose(0, 1)).pow(2).mean()

            # Backward pass + parameter update
            optimizer.zero_grad()  # clear gradients from previous step
            loss.backward() # compute ∂loss/∂Brt and ∂loss/∂W
            optimizer.step()  # Adam update: W ← W - lr * adaptive_grad

            # Sparsity enforcement (no gradient, in-place)
            # These steps enforce the compression constraints AFTER each
            # gradient update.  They are wrapped in torch.no_grad() because
            # we are manually editing the parameter values, not backpropping.
            with torch.no_grad():

                # W pruning: keep only the top-k bones per vertex.
                # Any bone ranked below max_influences is zeroed out.
                # This enforces the max_influences constraint so the Maya
                # skinCluster stays fast to evaluate at runtime.
                Wcutoff = torch.topk(self.W, self.settings.max_influences + 1, dim=0).values[-1, :]
                Wmask = self.W > Wcutoff
                self.W.copy_(Wmask * self.W)
                self.W.clamp_(min=0)   # no negative weights (physically meaningless)

                # Brt pruning: keep only the total_nnz_brt largest entries
                # across ALL (DOF, blendshape, bone) combinations.
                # Zero out everything below the threshold.
                # This ensures each joint only "knows about" the blendshapes
                # it geometrically participates in → compressed representation.
                Bdecider = self.Brt.abs()
                Bcutoff = torch.topk(Bdecider.flatten(), self.settings.total_nnz_brt).values[-1]
                Bmask = Bdecider >= Bcutoff
                self.Brt.copy_(Bmask * self.Brt)

            # Logging every 200 steps
            if i % 200 == 0:
                BX, _, _ = cp_math.compBX(Wn, self.Brt, self.TR, self.n_bs, self.rest_pose)
                trunc_err = (BX - self.A).abs().max().item()

                if self.device == "cuda":
                    torch.cuda.synchronize()

                logger.info(
                    f"{i:05d}({time.time() - st:.3f}s) "
                    f"loss={loss.item():.5e}  "
                    f"max_err={trunc_err:.5e}  "
                    f"B_nnz={(self.Brt.abs() > 1e-4).count_nonzero().item()}  "
                    f"W_nnz={(self.W.abs() > 1e-4).count_nonzero().item()}"
                )
                self.loss_list.append(loss.item())
                self.abserr_list.append(trunc_err)
                st = time.time()

    def train(self):
        """Two-pass training strategy.

        Pass 1 (iter1 steps, no weight normalisation):
            Lets the optimiser freely scale weights up or down to explore the
            loss landscape.  Often converges quickly to a good approximate
            solution without the constraint of summing-to-one.

        Pass 2 (iter2 steps, with weight normalisation):
            Enforces partition-of-unity (weights sum to 1 per vertex).
            Refines the solution so the final rig satisfies the LBS constraint
            and behaves correctly in Maya without additional post-processing.
        """
        param_list = [self.Brt, self.W]

        logger.info("--- Pass 1: free weights ---")
        opt1 = torch.optim.Adam(param_list, lr=self.settings.lr, betas=(0.9, 0.9))
        self._train_pass(self.settings.iter1, opt1, normalizeW=False)

        logger.info("--- Pass 2: normalised weights ---")
        opt2 = torch.optim.Adam(param_list, lr=self.settings.lr, betas=(0.9, 0.9))
        self._train_pass(self.settings.iter2, opt2, normalizeW=True)

        logger.info("Training finished.")

    def save_results(self):
        logger.info("--- Evaluating and saving results ---")
        # Final normalised weights (same formula as pass 2)
        col_sums = self.W.sum(axis=0).clamp(min=1e-8)
        Wn = self.W / col_sums
        logger.info(f"Final weight range: min={Wn.min().item():.4f}  max={Wn.max().item():.4f}")

        # Reconstruct deltas with the same joint-position-aware compBX used
        # during training so the saved shapeXform is consistent with Wn.
        BX, B, _ = cp_math.compBX(Wn, self.Brt, self.TR, self.n_bs, self.rest_pose)

        # Reshape back to (N, n_bs, 3) to compare with ground truth
        orig_deltas = cp_math.npf(self.A.transpose(1, 0).reshape(-1, self.n_bs, 3))
        our_deltas  = cp_math.npf(BX.transpose(1, 0).reshape(-1, self.n_bs, 3))
        max_delta  = np.abs(orig_deltas - our_deltas).max()
        mean_delta = np.abs(orig_deltas - our_deltas).mean()
        logger.info(f"Reconstruction error – max: {max_delta:.6f}  mean: {mean_delta:.6f}")

        # B is the shapeXform: (n_bs*3, P*4)
        # converter.py uses this to reconstruct bone transforms at runtime via
        # _generateXforms(), which multiplies pose weights by B to get the
        # (3×4) matrix for each joint at each animation frame.
        shapeXforms = B.detach().cpu().numpy()

        os.makedirs(self.settings.output_dir, exist_ok=True)
        output_path = os.path.join(self.settings.output_dir, "result.npz")

        np.savez(
            output_path,
            rest = cp_math.npf(self.rest_pose[:, :3]),
            quads = self.quads,
            weights = cp_math.npf(Wn).transpose(),
            restXform = np.array([np.eye(3, 4)] * self.settings.p_bones),
            shapeXform = shapeXforms,
            jointPositions = self.joint_positions,
        )
        logger.info(f"Result saved to {output_path}")