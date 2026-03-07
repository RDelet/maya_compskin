from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from typing import List

from maya import cmds
from maya.api import OpenMaya as om

from ..core import constants, math as cp_math
from ..core.joint_manager import JointManager
from ..core.data import NpzData
from ..maya import maya_utils, mesh


class Converter:

    def __init__(self, npz_path: str | Path):
        self._data = NpzData(npz_path)
        self.json = _JsonConverter(self._data)
        self.mesh = _MeshConverter(self._data)
        self.skin = _SkinConverter(self._data)
        self.anim = _AnimationConverter(self._data)
    
    @property
    def data(self) -> NpzData:
        return self._data


class NpzConverter(ABC):

    def __init__(self, data: NpzData):
        self._data = data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path: {self._data.path})"
    
    @abstractmethod
    def convert(self):
        pass


class _JsonConverter(NpzConverter):

    def _get_output_path(self, npz_path: str | Path, extension: str) -> Path:
        if isinstance(npz_path, str):
            npz_path = Path(npz_path)
        return npz_path.with_suffix(extension)

    def convert(self):
        data = {}
        for key in self._data.keys():
            array = self._data[key]
            data[key] = array.item() if array.dtype == 'O'else array.tolist()

        maya_utils.dump_json(data, self._get_output_path(self.path, '.json'))


class _MeshConverter(NpzConverter):

    def convert(self, with_blendshapes: bool = False) -> om.MObject:
        mesh_name = self._data.path.stem
        rest_points = om.MPointArray(self._data.get('rest_verts'))
        rest_faces = self._data.get('rest_faces')
        poly_count = om.MIntArray([len(f) for f in rest_faces])
        poly_connect = om.MIntArray(rest_faces.ravel())
        u_values = []
        v_values = []

        mesh_transform = maya_utils.get_object(cmds.createNode("transform", name=mesh_name))
        mfn_mesh = om.MFnMesh()
        mesh_obj = mfn_mesh.create(rest_points, poly_count, poly_connect, u_values, v_values, mesh_transform)
        maya_utils.assign_shader(mesh_obj)

        mdg_mod = om.MDGModifier()
        mdg_mod.renameNode(mesh_obj, f"{mesh_name}Shape")
        mdg_mod.doIt()
        
        if with_blendshapes:
            deltas = self._data.get('deltas')
            if deltas.shape[0] > 0:
                maya_utils.set_blendshape_targets(mesh_obj, deltas)
    
        return mesh_obj


class _SkinConverter(NpzConverter):

    def convert(self, shape: str | om.MObject, joint_manager: JointManager | None = None) -> tuple:
        skin_weights = self._data.get("weights")
        num_bones = skin_weights.shape[1]

        # TODO: use data in input file
        # joint_position = self._data.get("jointPositions")
        joint_grp = cmds.createNode("transform", name="JOINTS_GRP")
        if joint_manager is not None:
            joints = self._create_joints_at_positions(
                joint_manager.positions,
                joint_manager.names,
                parent=joint_grp,
            )
        else:
            joints = self._create_joints(num_bones, parent=joint_grp)

        skin_obj = maya_utils.create_skin(shape, joints)
        mesh.set_skin_weights(skin_obj, skin_weights)

        return skin_obj, joints

    @staticmethod
    def _create_joints(num_joints, parent: str = None) -> List[str]:
        joints = []
        for i in range(num_joints):
            jnt = cmds.createNode("joint", name=f"proxy_joint_{i}", parent=parent)
            joints.append(jnt)
        
        return joints

    @staticmethod
    def _create_joints_at_positions(positions: np.ndarray, names: list[str], parent: str | None = None) -> list[str]:
        joints = []
        for i, (pos, name) in enumerate(zip(positions, names)):
            jnt = cmds.createNode("joint", name=name, parent=parent)
            rest_matrix = [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                float(pos[0]), float(pos[1]), float(pos[2]), 1,
            ]
            cmds.xform(jnt, matrix=rest_matrix, worldSpace=True)
            cmds.setAttr(f"{jnt}.bindPose", rest_matrix, type="matrix")

            joints.append(jnt)

        return joints


class _AnimationConverter(NpzConverter):

    def convert(self, joints: List[str], shape_xform: np.ndarray):
        anim_weights = self._data.get("weights")
        num_frames = anim_weights.shape[0]
        target_count = anim_weights.shape[1]
        num_blendshapes = shape_xform.shape[0] // 3
        min_target_count = min(target_count, num_blendshapes)
        joint_count = len(joints)

        cmds.playbackOptions(minTime=0, maxTime=num_frames, animationStartTime=0, animationEndTime=num_frames)

        # J_rest[j] = [[I | p_j], [0,0,0,1]]  — joint j en bind pose world space
        # Utilisé pour composer J_anim = T_j @ J_rest_j
        if "jointPositions" in self._data:
            joint_positions = self._data.get("jointPositions")
            rest_matrices = np.tile(np.eye(4), (joint_count, 1, 1))
            rest_matrices[:, :3, 3] = joint_positions
        else:
            rest_matrices = None

        cmds.refresh(suspend=True)

        try:
            anim_data = {jnt: [] for jnt in joints}
            for i in range(num_frames):
                pose_weights = np.zeros(num_blendshapes, dtype=np.float32)
                pose_weights[:min_target_count] = anim_weights[i][:min_target_count]

                # T_j shape : (joint_count, 3, 4)
                # generateXforms retourne I à la pose de repos (tous weights=0)
                xform = cp_math.generateXforms(pose_weights, shape_xform)
                pose_matrices = xform.reshape(3, joint_count, 4).transpose(1, 0, 2)

                for j, jnt in enumerate(joints):
                    T_3x4 = pose_matrices[j]                         # (3, 4)
                    T_4x4 = np.vstack([T_3x4, [0, 0, 0, 1]])        # (4, 4)

                    if rest_matrices is not None:
                        # J_anim = T_j @ J_rest_j
                        #
                        # Preuve (régime linéaire) :
                        #   T_j = [[I+R_j, t_j], [0,1]]
                        #   J_rest = [[I, p_j], [0,1]]
                        #   T @ J_rest ≈ [[I+R_j, p_j+t_j], [0,1]]
                        #
                        # Maya calcule : J_anim @ bpm @ v
                        #   bpm = [[I, -p_j], [0,1]]  (ton maya_utils.create_skin)
                        #   = [[I+R_j, p_j+t_j]] @ [[I,-p_j]] @ v
                        #   = [[I+R_j, t_j]] @ v
                        #   = v + R_j@v + t_j
                        #   = v + T_j @ (v - p_j)  ✓  (régime linéaire R_j petit)
                        # J_anim = T_4x4 @ rest_matrices[j]
                        J_anim = rest_matrices[j] @ T_4x4
                    else:
                        J_anim = T_4x4

                    anim_data[jnt].append(J_anim.T.flatten().tolist())

            for jnt, data in anim_data.items():
                # continue
                maya_utils.anim_from_matrice(jnt, data)

        except Exception as e:
            constants.logger.error(e)
        finally:
            cmds.refresh(suspend=False)


"""
from maya import cmds

from maya_compskin.core import io_utils
from maya_compskin.maya.converter import Converter
from maya_compskin.core.joint_manager import JointManager

cmds.file(new=True, force=True)

aura_in = io_utils.get_input_from_name("aura")
aura_out = io_utils.get_output_from_name("aura")
test_anim = io_utils.get_input_from_name("test_anim")
joint_manager = JointManager(aura_in.parent / "Aura_JointPosition.json")

converter_in = Converter(aura_in)
converter_out = Converter(aura_out)
converter_anim = Converter(test_anim)

mesh_obj = converter_in.mesh.convert()
skin_obj, joints = converter_out.skin.convert(mesh_obj, joint_manager)
converter_anim.anim.convert(joints, converter_out.data["shapeXform"])
"""
