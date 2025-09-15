import numpy as np
import scipy as sp
import torch
import time
import igl
import os
from dataclasses import dataclass

from .scripts import constants
from .scripts.constants import logger


def buildTR(device):
    # fmt: off
    ebase = torch.tensor([[[0, 0,  0, 0],
                           [0, 0, -1, 0],
                           [0, 1,  0, 0]],

                          [[ 0, 0, 1, 0],
                           [ 0, 0, 0, 0],
                           [-1, 0, 0, 0]],

                          [[0, -1, 0, 0],
                           [1,  0, 0, 0],
                           [0,  0, 0, 0]],

                          [[0, 0, 0, 1],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]],

                          [[0, 0, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, 0, 0]],

                          [[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 1]]], dtype=torch.float32).to(device)
    return ebase.reshape(6, 1, 1, 3, 4)
    # fmt: on


def add_homog_coordinate(M, dim):
    x = list(M.shape)
    x[dim] = 1
    return np.concatenate([M, np.ones(x)], axis=dim).astype(M.dtype)


def compBX(Wn, Brt, TR, n_bs, P, rest_pose):
    # calculates Linear Blend Skinning
    # Wn ∈ PxN   (numBones x numVertices)
    # Brt - 6 degree of freedom  per blendshape per bone  (6, n_bs, numBones, 1, 1)
    # TR  - 6 base matrices (n_bx, numBones, 3, 4)[6]  one per degree of freedom these are used to convert Brt to B
    # rest_pose  ∈ nx4
    # X :  rest_pose.p * weight
    #         vertex...vertex
    #            0      N
    #         ┌           ┐
    # bone0  x│┌───┐      │
    #        y││  →│...   │
    #        z││w*p│      │
    #        w│└───┘      │
    # bone1  x│           │
    #        y│           │
    #        z│           │
    #        w│           │
    #         ┆           ┆
    # boneP  w│           │
    #         └           ┘
    X = (Wn.unsqueeze(2) * rest_pose).permute(0, 2, 1).reshape(4 * P, -1)
    B = Brt[0, ...] * TR[0]
    for i in range(1, 6):
        B += Brt[i, ...] * TR[i]
    B = B.permute(0, 2, 1, 3).reshape(n_bs * 3, P * 4)
    # B current bone transforms
    #               bone 0... bone N
    #                0123     0123
    #              ┌               ┐
    # blendshape0 0│┌────┐   ┌────┐│
    #             1││TM  │...│TM  ││
    #             2│└────┘   └────┘│
    # blendshape1 0│┌────┐   ┌────┐│
    #             1││TM  │...│TM  ││
    #             2│└────┘   └────┘│
    #              │               │
    #              ┆               ┆
    #              └               ┘
    return B @ X, B, X


def npf(T):
    return T.detach().cpu().numpy()


@dataclass
class Settings:
    """!@Brief Trainning settings
    
        Base parameters
            input_file: NPZ file input.
                rest pose: Orig vertices position
                polygon data: poly count / poly connect as array [ [0, 2, 4], [3, 5, 7], ... ]
                              All polygon need to have a same number of vertices.
                deltas: Shape deltas.
            output_dir: NPZ output path.
        
        Skinning Model Parameters
            p_bones: Number of proxy bones.
            max_influences: Max bone influences per vertex.
            total_nnz_brt: The total number of non-zero values ​​allowed in the transformation matrix.
                           This is what makes it possible to isolate a deformation to a joint rather than to all the vertices like the SSDR
        
        Learning Settings
            power: corresponds to the mean square error
                   Changes how error is measured.
                   Higher power penalizes large errors very heavily, forcing the model to better reproduce fine details, at the potential expense of other aspects.
            alpha: The weight of the Laplacian regularization term.
                   This regularization encourages deformations to be "smooth" on the mesh surface.
                   A high alpha value gives more rigid and smooth deformations, a low value allows more "chaotic" deformations.
            lr: The "learning rate" of the Adam optimizer.
                Controls the size of the steps the optimizer takes to minimize the error.
                Too large a value can cause learning to "explode", too small a value can make it very slow.
            iter1: The number of iterations for the two training passes without weight normalisation
            iter2: The number of iterations for the two training passes with weight normalisation
            seed: Training begins with randomly initialized weights and transformations.
                  Setting the seed ensures that this randomness is the same on every run, making the results
            init_weight: Controls the scale of random values ​​used to initialize transformations
                         Determines whether learning begins with very small or larger initial deformations
    """
    input_file: str
    output_dir: str
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


class Trainer:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(self.settings.seed)

        # Data and Tensors
        self.npb_data = None
        self.A = None
        self.L = None
        self.W = None
        self.Brt = None
        self.TR = None
        self.rest_pose = None
        self.quads = None
        self.n_bs = 0
        self.N = 0  # num_vertices

        # Training state
        self.loss_list = []
        self.abserr_list = []

        logger.info(f"Using device: {self.device}")
        self._load_data()
        self._initialize_tensors()

    def _load_data(self):
        logger.info(f"Loading data from {self.settings.input_file}...")
        self.npb_data = np.load(self.settings.input_file, allow_pickle=True)

        self.n_bs = self.npb_data["deltas"].shape[0]
        deltas = self.npb_data["deltas"].transpose(1, 0, 2).reshape(-1, self.n_bs * 3).transpose()
        self.A = torch.from_numpy(deltas).float().to(self.device)
        self.N = self.A.shape[1]

        self.quads = self.npb_data["rest_faces"]
        rest = self.npb_data["rest_verts"]

        logger.info(f"Loaded model with {self.N} vertices and {self.n_bs} blendshapes.")

        # Laplacian
        adj = igl.adjacency_matrix(self.quads)
        adj_diag = np.array(np.sum(adj, axis=1)).squeeze()
        # Rigidness Laplacian regularization.
        # ⎧-1        if k = i
        # ⎨1/|N(i)|, if k ∈ N(i)
        # ⎩0         otherwise.
        # Where N(i) denotes all the 1-ring neighbours of i
        Lg = sp.sparse.diags(1 / adj_diag) @ (adj - sp.sparse.diags(adj_diag))
        self.L = torch.from_numpy((Lg).todense()).float().to(self.device).to_sparse()

        # Rest pose
        rest_centered = rest - rest.mean(axis=0)
        self.rest_pose = torch.from_numpy(add_homog_coordinate(rest_centered, 1)).float().to(self.device)

    def _initialize_tensors(self):
        logger.info("Initializing tensors for optimization...")
        self.TR = buildTR(self.device)
        self.Brt = ((self.settings.init_weight * torch.randn((6, self.n_bs, self.settings.p_bones, 1, 1)))
                    .clone()
                    .float()
                    .to(self.device)
                    .requires_grad_())
        self.W = (1e-8 * torch.randn(self.settings.p_bones, self.N)).clone().float().to(self.device).requires_grad_()

    def _train_pass(self, num_iter, optimizer, normalizeW=False):
        st = time.time()
        for i in range(num_iter):
            Wn = self.W / self.W.sum(axis=0) if normalizeW else self.W
            BX, _, _ = compBX(Wn, self.Brt, self.TR, self.n_bs, self.settings.p_bones, self.rest_pose)
            weighed_error = BX - self.A

            loss = weighed_error.pow(self.settings.power).mean().pow(2 / self.settings.power)
            if self.settings.alpha is not None:
                loss += self.settings.alpha * (self.L @ (BX).transpose(0, 1)).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                Wcutoff = torch.topk(self.W, self.settings.max_influences + 1, dim=0).values[-1, :]
                Wmask = self.W > Wcutoff
                Wpruned = Wmask * self.W
                self.W.copy_(Wpruned)
                self.W.clamp_(min=0)

                Bdecider = self.Brt.abs()
                Bcutoff = torch.topk(Bdecider.flatten(), self.settings.total_nnz_brt).values[-1]
                Bmask = Bdecider >= Bcutoff
                Bpruned = Bmask * self.Brt
                self.Brt.copy_(Bpruned)

            if i % 200 == 0:
                BX, _, _ = compBX(Wn, self.Brt, self.TR, self.n_bs, self.settings.p_bones, self.rest_pose)
                trunc_err = (BX - self.A).abs().max().item()

                if self.device == "cuda":
                    torch.cuda.synchronize()

                logger.info(f"{i:05d}({time.time() - st:.3f}) loss={loss.item():.5e} err={trunc_err:.5e} "
                            f"B_nnz={(self.Brt.abs() > 1e-4).count_nonzero().item()} W_nnz={(self.W.abs() > 1e-4).count_nonzero().item()}")
                self.loss_list.append(loss.item())
                self.abserr_list.append(trunc_err)
                st = time.time()

    def train(self):
        param_list = [self.Brt, self.W]
        logger.info("--- First training pass ---")
        optimizer1 = torch.optim.Adam(param_list, lr=self.settings.lr, betas=(0.9, 0.9))
        self._train_pass(self.settings.iter1, optimizer1, normalizeW=False)
        logger.info("--- Second training pass (with weight normalization) ---")
        optimizer2 = torch.optim.Adam(param_list, lr=self.settings.lr, betas=(0.9, 0.9))
        self._train_pass(self.settings.iter2, optimizer2, normalizeW=True)
        logger.info("Training finished.")

    def save_results(self):
        logger.info("--- Evaluating and Saving Results ---")
        Wn = self.W / self.W.sum(axis=0)
        logger.info(f"Final weight range: min={Wn.min().item()}, max={Wn.max().item()}")

        BX, B, _ = compBX(Wn, self.Brt, self.TR, self.n_bs, self.settings.p_bones, self.rest_pose)
        orig_deltas = npf(self.A.transpose(1, 0).reshape(-1, self.n_bs, 3))
        our_deltas = npf(BX.transpose(1, 0).reshape(-1, self.n_bs, 3))

        maxDelta = np.abs(orig_deltas - our_deltas).max()
        meanDelta = np.abs(orig_deltas - our_deltas).mean()
        logger.info(f"Max reconstruction delta: {maxDelta}")
        logger.info(f"Mean reconstruction delta: {meanDelta}")

        shapeXforms = B.detach().cpu().numpy()

        if not os.path.exists(self.settings.output_dir):
            os.makedirs(self.settings.output_dir)

        output_path = os.path.join(self.settings.output_dir, "result.npz")
        np.savez(output_path,
                 rest=npf(self.rest_pose[:, :3]),
                 quads=self.quads,
                 weights=npf(Wn).transpose(),
                 restXform=np.array([np.eye(3, 4)] * self.settings.p_bones),
                 shapeXform=shapeXforms)
        logger.info(f"Result saved to {output_path}")


if __name__ == "__main__":
    npz_path = constants.in_directory.rglob("*.npz")
    for path in npz_path:
        face_name = path.stem

        settings = Settings(input_file=path,
                            output_dir=constants.out_directory / face_name,
                            p_bones=250,
                            max_influences=8,
                            total_nnz_brt=10000,
                            power=12,
                            alpha=50,
                            lr=1e-5,
                            iter1=20000,
                            iter2=20000,
                            seed=12345,
                            init_weight=1e-6)

        try:
            logger.info(f"--- Starting Programmatic Compressed Skinning Training for {face_name} ---")
            trainer = Trainer(settings)
            trainer.train()
            trainer.save_results()
            logger.info("--- Training of {face_name} finished successfully ---")
            logger.info(f"Results saved in: {os.path.join(settings.output_dir, 'result.npz')}")
        except Exception as e:
            logger.exception("An error occurred during training")
        
        break


"""
# Launch training from maya
from maya_compskin import model_fit


npz_path = utils.get_in_from_name("aura")
face_name = path.stem

settings = model_fit.Settings(input_file=path,
                              output_dir=constants.out_directory / face_name,
                              p_bones=20,
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
    logger.info(f"--- Starting Programmatic Compressed Skinning Training for {face_name} ---")
    trainer = model_fit.Trainer(settings)
    trainer.train()
    trainer.save_results()
    logger.info("--- Training of {face_name} finished successfully ---")
    logger.info(f"Results saved in: {os.path.join(settings.output_dir, 'result.npz')}")
except Exception as e:
    logger.exception("An error occurred during training")
"""

"""
# ToDo: Test data
for i in range(num_frames):
    T = generateXforms(anim_weights[i, :], shapeXforms)
    X = npf((Wn.unsqueeze(2) * rest_pose).permute(0, 2, 1).reshape(4 * P, -1))
    anim_verts = T @ X
    igl.write_obj(f"out/anim_frame{i:05d}.obj", anim_verts.transpose(), quads)
"""