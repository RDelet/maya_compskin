from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import List

from . import utils

from maya import cmds
from maya.api import OpenMaya as om


class Converter:
    
    def __init__(self, npz_path: str | Path):
        if isinstance(npz_path, str):
            npz_path = Path(npz_path)
        if npz_path.is_dir():
            raise RuntimeError("Path must be a file path not a directory !")
        if not npz_path.exists():
            raise RuntimeError(f"file not found at: {npz_path}")
        
        self.path = npz_path

    def _get_output_path(self, npz_path: str | Path, extension: str) -> Path:
        if isinstance(npz_path, str):
            npz_path = Path(npz_path)

        return npz_path.with_suffix(extension)

    def to_json(self):
        
        data = {}
        npz_data = utils.read_npz(self.path)
        for key in npz_data.keys():
            array = npz_data[key]
            if array.dtype == 'O':
                data[key] = array.item()
            else:
                data[key] = array.tolist()

        utils.dump_json(data, self._get_output_path(self.path, '.json'))
    
    def to_mesh(self, build_blendshapes: bool = False) -> om.MObject:
        
        npz_data = utils.read_npz(self.path)
        mesh_name = self.path.stem

        # Get data
        rest_points = om.MPointArray(npz_data.get('rest_verts'))
        rest_faces = npz_data.get('rest_faces')
        poly_count = om.MIntArray([len(f) for f in rest_faces])
        poly_connect = om.MIntArray(rest_faces.ravel())
        u_values = []
        v_values = []

        # Build mesh
        mesh_transform = utils.get_object(cmds.createNode("transform", name=mesh_name))
        mfn_mesh = om.MFnMesh()
        mesh_obj = mfn_mesh.create(rest_points, poly_count, poly_connect, u_values, v_values, mesh_transform)
        utils.assign_shader(mesh_obj)

        mdg_mod = om.MDGModifier()
        mdg_mod.renameNode(mesh_obj, f"{mesh_name}Shape")
        mdg_mod.doIt()
        
        if build_blendshapes:
            deltas = npz_data.get('deltas')
            if deltas.shape[0] > 0:
                utils.set_blendshape_targets(mesh_obj, deltas)
    
        return mesh_obj
        
    def to_skin(self, shape: om.MObject) -> om.MObject:

        npz_data = utils.read_npz(self.path)
        rest_vertices = npz_data['rest']
        skin_weights = npz_data['weights']
        num_bones = skin_weights.shape[1]

        joint_bind_matrices = self._compute_joint_bindpose(num_bones, skin_weights, rest_vertices)
        joint_grp = cmds.createNode("transform", name="JOINTS_GRP")
        joints = self._create_joints(joint_bind_matrices, parent=joint_grp)

        skin_obj = utils.create_skin(shape, joints)
        utils.set_skin_weights(skin_obj, skin_weights)
        
        return skin_obj, joints
    
    def to_pose(self, joints: List[str]):
        # ToDO: ...
        npz_data = utils.read_npz(self.path)
        joint_count = len(joints)
        shape_xforms = npz_data['shapeXform']
        num_blendshapes = shape_xforms.shape[0] // 3
        
        pose_weights = np.random.uniform(0, 1.0, size=num_blendshapes)
        # pose_weights = np.zeros(num_blendshapes, dtype=np.float32)
        bind_pose_transforms = self._generateXforms(pose_weights, shape_xforms)
        bind_pose_transforms = bind_pose_transforms.reshape(3, joint_count, 4).transpose(1, 0, 2)

        for node in joints:
            transform_3x4 = bind_pose_transforms[i]
            transform_4x4 = np.vstack([transform_3x4, [0, 0, 0, 1]])
            cmds.xform(node, matrix=transform_4x4.T.flatten().tolist(), worldSpace=True)
            joints.append(node)
    
    @staticmethod
    def _compute_joint_bindpose(num_bones: int, skin_weights: np.array, rest_verts: np.array) -> np.array:
        joint_bind_matrices = []
        for i in range(num_bones):
            bone_weights = skin_weights[:, i]
            total_weight = np.sum(bone_weights)

            if total_weight > 1e-6:
                weighted_center = np.average(rest_verts, axis=0, weights=bone_weights)
            else:
                weighted_center = np.zeros(3)

            bind_matrix = np.eye(4, dtype=np.float64)
            bind_matrix[:3, 3] = weighted_center
            joint_bind_matrices.append(bind_matrix)
        
        return np.array(joint_bind_matrices)
    
    @staticmethod
    def _create_joints(matrices: np.array, parent: str = None) -> List[str]:
        joints = []
        for i in range(matrices.shape[0]):
            jnt = cmds.createNode("joint", name=f"proxy_joint_{i}", parent=parent)
            cmds.xform(jnt, matrix=matrices[i].T.flatten().tolist(), worldSpace=True)
            joints.append(jnt)
        
        return joints
    
    def _add_homog_coordinate(M, dim):
        x = list(M.shape)
        x[dim] = 1

        return np.concatenate([M, np.ones(x)], axis=dim).astype(M.dtype)


    def _generateXforms(weights, shapeXforms):
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


"""
import imp

from maya import cmds

from maya_compskin.scripts import converter, utils
imp.reload(converter)
imp.reload(utils)

cmds.file(new=True, force=True)

aura_in = utils.get_in_from_name("aura")
aura_out = utils.get_out_from_name("aura")

in_converter = converter.Converter(aura_in)
shape_obj = in_converter.to_mesh()
# shape_obj = in_converter.to_mesh(build_blendshapes=True)
out_converter = converter.Converter(aura_out)
skin_obj, joints = out_converter.to_skin(shape_obj)
"""