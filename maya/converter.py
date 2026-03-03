from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import List, Any

from ..core import constants, io_utils, math as cp_math
from ..core.joint_manager import JointManager
from ..maya import maya_utils

from maya import cmds
from maya.api import OpenMaya as om


class AbstractConverter:

    def __init__(self, npz_path: str | Path | None = None):
        self.path = None
        self._data = None
        self._is_initialized = False

        if npz_path:
            if isinstance(npz_path, str):
                npz_path = Path(npz_path)
            if not npz_path.suffix.endswith("npz"):
                raise RuntimeError(f"Invalid file extension: {npz_path.suffix} !")
            if npz_path.is_dir():
                raise RuntimeError("Path must be a file path not a directory !")
            if not npz_path.exists():
                raise RuntimeError(f"file not found at: {npz_path}")
            
            self.path = npz_path
            self._data = io_utils.read_npz(npz_path)
            self._is_initialized = True
    
    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def data(self) -> dict:
        if not self._is_initialized:
            raise RuntimeError("No data initialized !")
        return self._data

    @classmethod
    def from_converter(cls, other: AbstractConverter) -> AbstractConverter:
        if not isinstance(other, AbstractConverter):
            raise TypeError("Value must be an AbstractConverter !")
        if not other.is_initialized:
            raise RuntimeError("No data initialized on source converter !")

        new_cls = cls()
        new_cls.path = other.path
        new_cls._data = other.data
        new_cls._is_initialized = True

        return new_cls

    def get(self, key: str) -> Any:
        if not self._is_initialized:
            raise RuntimeError("No data initialized !")

        if key not in self._data:
            raise RuntimeError(f"Key {key} does not exist in {self.path}")

        return self._data[key]
        
    def convert(self, *args, **kwargs):
        if not self._is_initialized:
            raise RuntimeError("No data initialized !")


class Converter(AbstractConverter):

    def __init__(self, npz_path: str | Path):
        super().__init__(npz_path)

        self.json = _JsonConverter.from_converter(self)
        self.mesh = _MeshConverter.from_converter(self)
        self.skin = _SkinConverter.from_converter(self)
        self.anim = _AnimationConverter.from_converter(self)


class _JsonConverter(AbstractConverter):

    def _get_output_path(self, npz_path: str | Path, extension: str) -> Path:
        if isinstance(npz_path, str):
            npz_path = Path(npz_path)
        return npz_path.with_suffix(extension)

    def convert(self):
        super().convert()
        data = {}
        for key in self._data.keys():
            array = self.get(key)
            data[key] = array.item() if array.dtype == 'O'else array.tolist()

        maya_utils.dump_json(data, self._get_output_path(self.path, '.json'))


class _MeshConverter(AbstractConverter):

    def convert(self, with_blendshapes: bool = False) -> om.MObject:
        super().convert()
        mesh_name = self.path.stem
        rest_points = om.MPointArray(self.get('rest_verts'))
        rest_faces = self.get('rest_faces')
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
            deltas = self.get('deltas')
            if deltas.shape[0] > 0:
                maya_utils.set_blendshape_targets(mesh_obj, deltas)
    
        return mesh_obj


class _SkinConverter(AbstractConverter):

    """TODO: Refacto ! """

    def convert(self, shape: str | om.MObject, joint_manager: JointManager | None = None) -> tuple:
        super().convert()
        skin_weights = self.get("weights")
        num_bones = skin_weights.shape[1]

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
        maya_utils.set_skin_weights(skin_obj, skin_weights)

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


class _AnimationConverter(AbstractConverter):

    def convert(self, joints: list[str], shape_xform: np.ndarray, joint_manager=None):
        super().convert()

        anim_weights    = self.get("weights")
        num_frames      = anim_weights.shape[0]
        target_count    = anim_weights.shape[1]
        num_blendshapes = shape_xform.shape[0] // 3
        min_target_count = min(target_count, num_blendshapes)
        joint_count     = len(joints)

        # J_rest[j] = [[I | p_j], [0,0,0,1]]  — joint j en bind pose world space
        # Utilisé pour composer J_anim = T_j @ J_rest_j
        if joint_manager is not None:
            positions    = joint_manager.positions               # (P, 3) world space
            rest_matrices = np.tile(np.eye(4), (joint_count, 1, 1))
            rest_matrices[:, :3, 3] = positions                 # translation en colonne
        else:
            rest_matrices = None

        cmds.playbackOptions(
            minTime=0, maxTime=num_frames,
            animationStartTime=0, animationEndTime=num_frames,
        )
        current_time = cmds.currentTime(query=True)
        cmds.refresh(suspend=True)

        try:
            anim_data = {jnt: [] for jnt in joints}
            for i in range(num_frames):
                cmds.currentTime(i)

                pose_weights = np.zeros(num_blendshapes, dtype=np.float32)
                pose_weights[:min_target_count] = anim_weights[i][:min_target_count]

                # T_j shape : (joint_count, 3, 4)
                # _generateXforms retourne I à la pose de repos (tous weights=0)
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
                        J_anim = T_4x4 @ rest_matrices[j]
                    else:
                        J_anim = T_4x4

                    anim_data[jnt].append(J_anim.T.flatten().tolist())
                    # # Maya attend la matrice en row-major (transposée par rapport à numpy)
                    # cmds.xform(jnt, matrix=J_anim.T.flatten().tolist(), worldSpace=True)

                # cmds.setKeyframe(joints)
                # if i == 200:
                #     break
            for jnt, data in anim_data.items():
                maya_utils.anim_from_matrice(jnt, data)

        except Exception as e:
            constants.logger.error(e)
        finally:
            cmds.currentTime(current_time)
            cmds.refresh(suspend=False)


"""
import imp

from maya import cmds

from maya_compskin.core import io_utils
from maya_compskin.maya import converter
from maya_compskin.core.joint_manager import JointManager

imp.reload(converter)
imp.reload(io_utils)

cmds.file(new=True, force=True)

aura_in = io_utils.get_input_from_name("aura")
aura_out = io_utils.get_output_from_name("aura")
test_anim = io_utils.get_input_from_name("test_anim")
joint_manager = JointManager(aura_in.parent / "Aura_JointPosition.json")

converter_in = converter.Converter(aura_in)
converter_out = converter.Converter(aura_out)
converter_anim = converter.Converter(test_anim)

mesh_obj = converter_in.mesh.convert()
skin_obj, joints = converter_out.skin.convert(mesh_obj, joint_manager)
converter_anim.anim.convert(joints, converter_out["shapeXform"])
"""
