from __future__ import annotations

import numpy as np
import torch


def compute_initial_weights(rest_verts: np.ndarray, joint_positions: np.ndarray, max_influences: int = 8) -> np.ndarray:

    n_verts = len(rest_verts)
    n_joints = len(joint_positions)

    diff = rest_verts[:, np.newaxis, :] - joint_positions[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    weights = 1.0 / (distances + 1e-8)
    if max_influences < n_joints:
        near_indices = np.argpartition(distances, max_influences, axis=1)[:, :max_influences]
        mask = np.zeros((n_verts, n_joints), dtype=bool)
        np.put_along_axis(mask, near_indices, True, axis=1)
        weights[~mask] = 0.0

    row_sums = weights.sum(axis=1, keepdims=True)
    weights  = weights / row_sums

    return weights


# ─────────────────────────────────────────────────────────────────────────────
# buildTR  –  The 6 basis matrices of a rigid-body transformation
# ─────────────────────────────────────────────────────────────────────────────
#
# A 3D rigid-body transformation (rotation + translation) has 6 degrees of
# freedom (DOF): 3 rotation axes (X, Y, Z) and 3 translation axes (X, Y, Z).
#
# In Lie group theory, any small rigid transformation can be written as a
# linear combination of 6 "generator" matrices, which are the basis elements
# of the corresponding Lie algebra se(3).
#
# Each generator is a (3×4) matrix that, when multiplied by a homogeneous
# point (x, y, z, 1), produces the infinitesimal displacement caused by a
# unit motion along that DOF:
#
#   ebase[0]  =  rotation around X-axis       [[0, 0, 0, 0],
#                  (right-hand rule:            [0, 0,-1, 0],
#                   +Y→+Z direction)            [0,+1, 0, 0]]
#
#   ebase[1]  =  rotation around Y-axis       [[ 0, 0,+1, 0],
#                  (+Z→+X direction)            [ 0, 0, 0, 0],
#                                              [-1, 0, 0, 0]]
#
#   ebase[2]  =  rotation around Z-axis       [[0,-1, 0, 0],
#                  (+X→+Y direction)            [+1, 0, 0, 0],
#                                              [ 0, 0, 0, 0]]
#
#   ebase[3]  =  translation along X          [[0,0,0,1],
#                                              [0,0,0,0],
#                                              [0,0,0,0]]
#
#   ebase[4]  =  translation along Y          [[0,0,0,0],
#                                              [0,0,0,1],
#                                              [0,0,0,0]]
#
#   ebase[5]  =  translation along Z          [[0,0,0,0],
#                                              [0,0,0,0],
#                                              [0,0,0,1]]
#
# IMPORTANT: These bases are entirely independent of where a joint is located
# in space. They describe *how* a transformation is parameterized (its shape),
# not *where* it is applied. The joint position is handled separately in
# compBX by expressing each vertex in the joint's local frame before applying
# the transformation.
#
# The output is reshaped to (6, 1, 1, 3, 4) so it can broadcast directly
# against Brt of shape (6, n_bs, n_bones, 1, 1) via element-wise multiply.
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


# ─────────────────────────────────────────────────────────────────────────────
# add_homog_coordinate  –  append a column/row of ones to a numpy array
# ─────────────────────────────────────────────────────────────────────────────
# Homogeneous coordinates extend a point (x, y, z) → (x, y, z, 1) so that
# translations can be expressed as matrix multiplications (just like rotations).
# Without this extra 1, a (3×4) matrix could only apply rotation+scale, not
# translation.
def add_homog_coordinate(M, dim):
    x = list(M.shape)
    x[dim] = 1
    return np.concatenate([M, np.ones(x)], axis=dim).astype(M.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# compBX  –  Core of the Compressed Skinning model: compute predicted deltas
# ─────────────────────────────────────────────────────────────────────────────
#
# This function implements the forward pass of the model. Given the current
# learned parameters (Brt, W) it predicts the vertex displacements for every
# blendshape, which are then compared to the ground-truth deltas (A) to
# compute the reconstruction error.
#
# ── Inputs ───────────────────────────────────────────────────────────────────
#   Wn              (P, N)       Skinning weights: how much each proxy bone
#                                influences each vertex. Rows sum to 1 after
#                                normalisation. P = n_bones, N = n_vertices.
#
#   Brt             (6, n_bs, P, 1, 1)
#                                The 6 scalar coefficients that, when combined
#                                with the 6 basis matrices TR, define a (3×4)
#                                rigid transformation for each (blendshape,
#                                bone) pair. These are the LEARNED parameters.
#
#   TR              (6, 1, 1, 3, 4)
#                                The 6 fixed basis matrices from buildTR.
#                                Broadcasting dimensions allow a single matmul
#                                to cover all blendshapes and all bones.
#
#   n_bs            int          Number of blendshapes.
#   P               int          Number of proxy bones.
#
#   rest_pose       (N, 4)       Centered rest-pose vertices in homogeneous
#                                coordinates.  The centering (subtracting the
#                                mesh barycentre) is done once in _load_data
#                                for numerical stability.
#
#   joint_positions (P, 3) | None
#                                World-space positions of the proxy joints
#                                AFTER the same centering applied to rest_pose.
#                                When provided, each vertex is expressed in the
#                                local frame of each joint before the skinning
#                                transform is applied, so rotations act around
#                                the joint rather than around the origin.
#
# ── What X represents ────────────────────────────────────────────────────────
#
# Without joint positions (original formulation):
#   For each bone j and vertex v:  X[j, v] = w_j(v) * v_rest
#   The transform T_j acts on the global rest position, so rotations
#   implicitly pivot around the mesh barycentre (the origin after centering).
#
# With joint positions (new formulation):
#   For each bone j at position p_j and vertex v:
#     X[j, v] = w_j(v) * (v_rest - p_j)
#   By subtracting p_j first we express v in the LOCAL frame of joint j.
#   When T_j (which encodes a rotation) is then applied, the rotation
#   pivots around p_j in world space, which is the physically correct behaviour.
#
#   Proof: for a pure rotation R around p_j:
#     R * (v - p_j)  =  R*v - R*p_j
#   This is exactly a rotation centred on p_j.  The translation component
#   of Brt absorbs any residual offset.
#
# Layout of X after reshape:
#
#         vertex_0 ... vertex_N
#      ┌                       ┐
# bone0│  w0*vx0  ...  w0*vxN  │  ← x coordinate weighted by bone 0
#      │  w0*vy0  ...  w0*vyN  │  ← y
#      │  w0*vz0  ...  w0*vzN  │  ← z
#      │  w0* 1   ...  w0* 1   │  ← homogeneous (enables translation)
# bone1│  w1*vx0  ...  w1*vxN  │
#      │  ...                  │
# boneP│  ...                  │
#      └                       ┘
#   shape: (4*P, N)
#
# ── What B represents ────────────────────────────────────────────────────────
#
# B is obtained by summing the 6 Lie-algebra generators scaled by Brt:
#   B_raw = Σ_{k=0}^{5}  Brt[k] * TR[k]    shape: (n_bs, P, 3, 4)
#
# Each (3×4) slice B[s, j] is the transformation matrix for blendshape s,
# bone j.  It encodes a linearised rigid motion (small-angle approximation).
#
# After permute + reshape:
#
#         bone0_cols bone1_cols ... boneP_cols
#      ┌                                      ┐
# bs0 │  [3×4 TM]    [3×4 TM]  ...  [3×4 TM] │  ← 3 rows per blendshape
# bs1 │  ...                                  │
#     │  ...                                  │
# bsK │  ...                                  │
#      └                                      ┘
#   shape: (n_bs*3, 4*P)
#
# ── Final product BX ─────────────────────────────────────────────────────────
#
#   BX = B @ X    shape: (n_bs*3, N)
#
# This is exactly the predicted per-vertex displacement for every blendshape,
# packed as (n_bs*3 rows, N columns).  It is compared to A (same shape) to
# compute the reconstruction loss.
"""
def compBX(Wn, Brt, TR, n_bs, P, rest_pose, joint_positions=None):

    if joint_positions is not None:
        # ── Express each vertex in the local frame of each joint ──────────
        #
        # rest_pose[:, :3]  :  (N, 3)  vertex positions (centred world space)
        # joint_positions   :  (P, 3)  joint positions  (same centred space)
        #
        # We want  v_local[j, v] = v_world[v] - p_j
        # using broadcasting:
        pj      = joint_positions.unsqueeze(1)   # (P, 1, 3)  – one pos per bone
        v_world = rest_pose[:, :3].unsqueeze(0)  # (1, N, 3)  – all vertices

        # Subtract: each bone sees every vertex relative to itself
        v_local = v_world - pj                   # (P, N, 3)

        # Re-append the homogeneous coordinate (=1) so translation terms in
        # the (3×4) transform matrix have something to act on.
        ones = torch.ones(*v_local.shape[:2], 1, device=rest_pose.device)  # (P, N, 1)
        rest_local = torch.cat([v_local, ones], dim=-1)                     # (P, N, 4)

        # Weight each vertex by the bone's skinning weight, then reshape to
        # the (4*P, N) layout expected by the matrix product B @ X.
        #   Wn          : (P, N)    → unsqueeze → (P, N, 1)
        #   rest_local  : (P, N, 4)
        #   product     : (P, N, 4)  → permute(0,2,1) → (P, 4, N) → reshape → (4P, N)
        X = (Wn.unsqueeze(2) * rest_local).permute(0, 2, 1).reshape(4 * P, -1)

    else:
        # ── Original formulation: joints implicitly at the centred origin ──
        # Wn         : (P, N) → unsqueeze → (P, N, 1)
        # rest_pose  : (N, 4) → unsqueeze broadcasts to (P, N, 4)
        X = (Wn.unsqueeze(2) * rest_pose).permute(0, 2, 1).reshape(4 * P, -1)

    # ── Reconstruct the (3×4) transformation matrices from Brt and TR ──────
    #
    # Brt : (6, n_bs, P, 1, 1)  – scalar coefficients, LEARNED
    # TR  : (6, 1,    1, 3, 4)  – fixed basis matrices
    #
    # Element-wise multiply broadcasts to (6, n_bs, P, 3, 4), then we sum
    # over the 6 DOF dimension to get one (3×4) matrix per (blendshape, bone).
    #
    #   B_raw[s, j] = Σ_k  Brt[k, s, j] * TR[k]    shape: (n_bs, P, 3, 4)
    #
    # This is the vectorised equivalent of the original loop:
    #   B = Brt[0]*TR[0]; for i in 1..5: B += Brt[i]*TR[i]
    B = (Brt * TR).sum(dim=0)            # (n_bs, P, 3, 4)

    # Rearrange from (n_bs, P, 3, 4) to (n_bs*3, P*4) so that the matrix
    # product B @ X computes all blendshapes simultaneously.
    #   permute(0,2,1,3) → (n_bs, 3, P, 4)
    #   reshape           → (n_bs*3, P*4)
    B = B.permute(0, 2, 1, 3).reshape(n_bs * 3, P * 4)

    # ── BX: predicted vertex displacements ──────────────────────────────────
    # (n_bs*3, P*4) @ (P*4, N)  =  (n_bs*3, N)
    # Each column is a vertex; each group of 3 rows is one blendshape (x,y,z).
    return B @ X, B, X
"""
def compBX(Wn, Brt, TR, n_bs, P, rest_pose, joint_positions=None):

    if joint_positions is not None:
        joint_positions = joint_positions.unsqueeze(1)
        vertex_world    = rest_pose[:, :3].unsqueeze(0)
        vertex_local    = vertex_world - joint_positions
        ones       = torch.ones(*vertex_local.shape[:2], 1, device=rest_pose.device)
        rest_local = torch.cat([vertex_local, ones], dim=-1)
        weighted = Wn.unsqueeze(2) * rest_local
    else:
        weighted = Wn.unsqueeze(2) * rest_pose

    P_actual = weighted.shape[0]
    X = weighted.permute(0, 2, 1).reshape(4 * P_actual, -1)  # (4P, N)

    B = Brt[0, ...] * TR[0]
    for i in range(1, 6):
        B += Brt[i, ...] * TR[i]
    B = B.permute(0, 2, 1, 3).reshape(n_bs * 3, P_actual * 4)

    return B @ X, B, X


def npf(T):
    """Detach a tensor from the computation graph and return it as a numpy array."""
    return T.detach().cpu().numpy()


def generateXforms(weights, shapeXforms):
        # weights ... (num_shapes, 1), output of riglogic
        # shapeXforms ... (3*num_shapes, 4*num_proxy_bones) matrix
        # returns: (num_proxy_bones, 3, 4) skinning transforms, input to skinCluster
        nShapes = weights.shape[0]
        nBones = shapeXforms.shape[1] // 4
        Z = weights.reshape(1, 1, nShapes) * np.dstack([np.eye(3)] * nShapes)
        # Z:
        # ┌      ┐┌      ┐┌      ┐
        # │w₁   0││w₂   0││w₃   0│
        # │  w₁  ││  w₂  ││  w₃  │  ───▶ axis 2
        # │0   w₁││0   w₂││0   w₃│
        # └      ┘└      ┘└      ┘
        #
        # Z.transpose(0, 2, 1).reshape(3, -1)
        # ┌                  ┐
        # │w₁0 0 w₂0 0 w₃0 0 │
        # │0 w₁0 0 w₂0 0 w₃0 │
        # │0 0 w₁0 0 w₂0 0 w₃│
        # └                  ┘
        # Z.transpose(0, 2, 1).reshape(3, -1) @ shapeXforms
        #   weighted sum of blendshape transfomrs (3, 4 * num_bones)
        #
        # Z.transpose(0, 2, 1).reshape(3, -1) @ shapeXforms + np.array([np.eye(3, 4)] * nBones).transpose(1, 0, 2).reshape(3, -1)
        #   add 1 to diagonals for every transform (befor was 0)
        res = Z.transpose(0, 2, 1).reshape(3, -1) @ shapeXforms + np.array(
            [np.eye(3, 4)] * nBones
        ).transpose(1, 0, 2).reshape(3, -1)

        return res