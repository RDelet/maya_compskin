from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import List

from . import utils, constants

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
        self._data = utils.read_npz(npz_path)
    
    def get(self, key: str) -> np.array:
        if key not in self._data:
            raise RuntimeError(f"Key {key} does not exist in {self.path}")
        
        return self._data[key]

    def _get_output_path(self, npz_path: str | Path, extension: str) -> Path:
        if isinstance(npz_path, str):
            npz_path = Path(npz_path)

        return npz_path.with_suffix(extension)

    def to_json(self):
        
        data = {}
        for key in self._data.keys():
            array = self.get(key)
            if array.dtype == 'O':
                data[key] = array.item()
            else:
                data[key] = array.tolist()

        utils.dump_json(data, self._get_output_path(self.path, '.json'))
    
    def to_mesh(self, build_blendshapes: bool = False) -> om.MObject:
        # Get data
        mesh_name = self.path.stem
        rest_points = om.MPointArray(self.get('rest_verts'))
        rest_faces = self.get('rest_faces')
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
            deltas = self.get('deltas')
            if deltas.shape[0] > 0:
                utils.set_blendshape_targets(mesh_obj, deltas)
    
        return mesh_obj
        
    def to_skin(self, shape: om.MObject) -> om.MObject:
        rest_vertices = self.get("rest")
        skin_weights = self.get("weights")
        num_bones = skin_weights.shape[1]

        joint_grp = cmds.createNode("transform", name="JOINTS_GRP")
        joints = self._create_joints(num_bones, parent=joint_grp)

        skin_obj = utils.create_skin(shape, joints)
        utils.set_skin_weights(skin_obj, skin_weights)
        
        return skin_obj, joints
    
    def to_anim(self, joints: List[str], shape_xform: np.array):
        anim_weights = self.get("weights")
        num_frames = anim_weights.shape[0]
        target_count = anim_weights.shape[1]
        num_blendshapes = shape_xform.shape[0] // 3
        min_target_count = min(target_count, num_blendshapes)
        
        joint_count = len(joints)
        
        cmds.playbackOptions(minTime=0, maxTime=num_frames, animationStartTime=0, animationEndTime=num_frames)
        current_time = cmds.currentTime(query=True)
        cmds.refresh(suspend=True, force=True)
        try:
            for i in range(num_frames):
                cmds.currentTime(i)
                
                pose_weights = np.zeros(num_blendshapes, dtype=np.float32)
                pose_weights[:min_target_count] = anim_weights[i][:min_target_count]
                xform = self._generateXforms(pose_weights, shape_xform)
                pose_matrices = xform.reshape(3, joint_count, 4).transpose(1, 0, 2)
                
                for j, node in enumerate(joints):
                    transform_3x4 = pose_matrices[j]
                    transform_4x4 = np.vstack([transform_3x4, [0, 0, 0, 1]])
                    matrix = transform_4x4.T.flatten().tolist()
                    cmds.xform(node, matrix=matrix, worldSpace=True)

                cmds.setKeyframe(joints)
        except Exception as e:
            constants.logger.error(e)
        finally:
            cmds.currentTime(current_time)
            cmds.refresh(suspend=False, force=True)
    
    @staticmethod
    def _create_joints(num_joints, parent: str = None) -> List[str]:
        joints = []
        for i in range(num_joints):
            jnt = cmds.createNode("joint", name=f"proxy_joint_{i}", parent=parent)
            cmds.xform(jnt, matrix=utils.identity_matrix, worldSpace=True)
            cmds.setAttr(f"{jnt}.bindPose", utils.identity_matrix, type="matrix")
            joints.append(jnt)
        
        return joints
    
    @staticmethod
    def _add_homog_coordinate(M, dim):
        x = list(M.shape)
        x[dim] = 1

        return np.concatenate([M, np.ones(x)], axis=dim).astype(M.dtype)

    @staticmethod
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
test_anim = utils.get_in_from_name("test_anim")

in_converter = converter.Converter(aura_in)
shape_obj = in_converter.to_mesh()
out_converter = converter.Converter(aura_out)
skin_obj, joints = out_converter.to_skin(shape_obj)

anim_converter = converter.Converter(test_anim)
anim_converter.to_anim(joints, out_converter.get("shapeXform"))
"""

"""
# apply weights on blendshape to compare animation

in_converter = converter.Converter(aura_in)
shape_bs_obj = in_converter.to_mesh(build_blendshapes=True)
bs_node = utils.find_deformer(shape_bs_obj, om.MFn.kBlendShape)
bs_node_name = utils.name_of(bs_node[0])

anim_converter = converter.Converter(test_anim)
weights = anim_converter.get("weights")
num_frame = weights.shape[0]
num_target = weights.shape[1]
current_time = cmds.currentTime(query=True)
cmds.refresh(suspend=True, force=True)
try:
    for i in range(num_frame):
        cmds.currentTime(i)
        for j in range(num_target):
            cmds.setAttr(f"{bs_node_name}.weight[{j}]", float(weights[i][j]))
        cmds.setKeyframe(bs_node_name)
except Exception as e:
    cmds.error(e)
finally:
    cmds.currentTime(current_time)
    cmds.refresh(suspend=False, force=True)
"""