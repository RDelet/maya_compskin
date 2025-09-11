from __future__ import annotations

import numpy as np
from pathlib import Path

from maya import cmds
from maya.api import OpenMaya as om

from . import utils
from . import constants


def add_homog_coordinate(M, dim):
    """Ajoute une coordonnée homogène à une matrice."""
    x = list(M.shape)
    x[dim] = 1

    return np.concatenate([M, np.ones(x)], axis=dim).astype(M.dtype)


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


def buil_skin(mesh_obj: om.MObject, npz_path: str | Path):
    # Get data
    npz_data = utils.read_npz(npz_path)

    skin_weights = npz_data['weights']
    shape_xforms = npz_data['shapeXform']
    num_bones = skin_weights.shape[1]
    num_blendshapes = shape_xforms.shape[0] // 3

    # Compute joint position at frame 0
    # pose_weights = np.random.rand(num_blendshapes)
    pose_weights = np.zeros(num_blendshapes, dtype=np.float32)
    bind_pose_transforms = generateXforms(pose_weights, shape_xforms)
    bind_pose_transforms = bind_pose_transforms.reshape(3, 4, num_bones).transpose(2, 0, 1)

    # Create joints
    joints = []
    joint_grp = cmds.createNode("transform", name="JOINTS_GRP")
    for i in range(num_bones):
        transform_3x4 = bind_pose_transforms[i]
        transform_4x4 = np.vstack([transform_3x4, [0, 0, 0, 1]])
        jnt = cmds.createNode("joint", name=f"proxy_joint_{i}", parent=joint_grp)
        cmds.xform(jnt, matrix=transform_4x4.T.flatten().tolist(), worldSpace=True)
        joints.append(jnt)

    # Create skin
    skin_obj = utils.create_skin(mesh_obj, joints)
    utils.set_skin_weights(skin_obj, skin_weights)


def build_npz(face_name: str):
    cmds.file(new=True, force=True)

    in_npz_path = constants.in_directory / f"{face_name}.npz"
    if not in_npz_path.exists():
        raise RuntimeError(f"Input file not found at: {in_npz_path}")
    
    out_npz_path = constants.out_directory / face_name / "result.npz"
    if not in_npz_path.exists():
        raise RuntimeError(f"Output file not found at: {in_npz_path}")
    
    mesh_obj = utils.npz_to_mesh(in_npz_path)
    buil_skin(mesh_obj, out_npz_path)


"""
import imp
from maya_compskin.scripts import apply_skin
imp.reload(apply_skin)

apply_skin.build_npz("aura")
"""
