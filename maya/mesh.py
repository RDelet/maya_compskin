from __future__ import annotations

import numpy as np
from typing import List, Optional

from maya import cmds
from maya.api import OpenMaya as om, OpenMayaAnim as oma

from ..core import math as cp_math
from ..core.data import MeshData
from ..core.joint_manager import JointResult, place_joints_on_mesh
from . import maya_utils


def get_selected() -> om.MDagPath:
    selected = om.MGlobal.getActiveSelectionList()
    if selected.isEmpty():
        raise RuntimeError("No node selected!")

    dag_path = selected.getDagPath(0)
    if dag_path.hasFn(om.MFn.kTransform):
        dag_path.extendToShape()
    if dag_path.apiType() != om.MFn.kMesh:
        raise RuntimeError(f"Object {dag_path.fullPathName()} is not a mesh!")

    return dag_path


def extract_data(path: str | om.MObject | om.MDagPath) -> MeshData:
    if not isinstance(path, om.MDagPath):
        path = maya_utils.get_path(path)
    if path.hasFn(om.MFn.kTransform):
        path.extendToShape()
    if path.apiType() != om.MFn.kMesh:
        raise RuntimeError(f"Object {path.fullPathName()} is not a mesh!")

    mesh_fn = om.MFnMesh(path)
    points = mesh_fn.getPoints(om.MSpace.kObject)
    _, tri_verts = mesh_fn.getTriangles()

    verts = np.array(points)[:, :3]
    faces = np.array(tri_verts, dtype=np.int32).reshape(-1, 3)

    return MeshData(verts, faces, path.fullPathName())


def create_joints(result: JointResult, mesh_name: str = "mesh") -> List[str]:
    short_name = mesh_name.split("|")[-1].split(":")[-1]
    base_name = f"cp_joints_{short_name}"
    group = cmds.createNode("transform", name=base_name)

    output = []
    for idx, pos in enumerate(result.joint_positions):
        jnt = cmds.createNode("joint", name=f"{base_name}_JNT{idx:02d}", parent=group)
        cmds.setAttr(f"{jnt}.translate", *pos.tolist())
        output.append(jnt)

    return output


def get_components(mesh_obj: str | om.MObject) -> om.MObject:
    if isinstance(mesh_obj, str):
        mesh_obj = maya_utils.get_object(mesh_obj)
    if not mesh_obj.hasFn(om.MFn.kMesh):
        raise TypeError(f"Node {maya_utils.name_of(mesh_obj)} must be a mesh not {mesh_obj.apiTypeStr()}")

    mit_vtx = om.MItMeshVertex(maya_utils.get_path(mesh_obj))
    single_component = om.MFnSingleIndexedComponent()
    component = single_component.create(om.MFn.kMeshVertComponent)
    while not mit_vtx.isDone():
        single_component.addElement(mit_vtx.index())
        mit_vtx.next()

    return component


def auto_place_joints(mesh: str | om.MObject | om.MDagPath, joint_count: int,
                      feature_weight: float = 2.0,
                deform_mask: Optional[np.ndarray] = None):
    mesh   = extract_data(mesh)
    result = place_joints_on_mesh(mesh, joint_count,
                                  feature_weight=feature_weight,
                                  deform_mask=deform_mask)

    return create_joints(result, mesh_name=mesh.name)


def auto_skin(path: str | om.MObject | om.MDagPath, joints: List[str]):
    if not isinstance(path, om.MDagPath):
        path = maya_utils.get_path(path)
    if path.hasFn(om.MFn.kTransform):
        path.extendToShape()
    if path.apiType() != om.MFn.kMesh:
        raise RuntimeError(f"Object {path.fullPathName()} is not a mesh!")

    mfn = om.MFnMesh(path)
    points = np.array(mfn.getPoints(om.MSpace.kObject))[:, :3]
    joint_positions = np.array([cmds.xform(x, query=True, translation=True, worldSpace=True) for x in joints])
    weights = cp_math.compute_initial_weights(points, np.array(joint_positions))

    skin_obj = maya_utils.create_skin(path, joints)
    set_skin_weights(skin_obj, weights)


def set_skin_weights(skin_obj: str | om.MObject, weights: np.array, normalize: bool = True):
    if isinstance(skin_obj, str):
        skin_obj = maya_utils.get_object(skin_obj)

    skin_name = maya_utils.name_of(skin_obj)
    skin_fn = oma.MFnSkinCluster(skin_obj)
    output_shapes = skin_fn.getOutputGeometry()
    if len(output_shapes) == 0:
        raise RuntimeError(f"No output geometry found on {skin_name} !")

    mesh_obj = output_shapes[0]
    mesh_path = maya_utils.get_path(mesh_obj)
    component = get_components(mesh_obj)
    influences_ids = om.MIntArray(list(range(weights.shape[1])))
    weights = om.MDoubleArray(weights.flatten())    
    
    return skin_fn.setWeights(mesh_path, component, influences_ids, weights, normalize, returnOldWeights=True)



"""
import json
import numpy as np

from maya import cmds

from maya_compskin.core import io_utils, math as cp_math
from maya_compskin.maya import converter, maya_utils, mesh

cmds.file(new=True, force=True)

face = "aura"
aura_in = io_utils.get_input_from_name(face)
converter_in = converter.Converter(aura_in)

mesh_obj = converter_in.mesh.convert()

deltas = np.array(converter_in["deltas"])
mask = cp_math.get_mask_from_deltas(deltas, threshold=1e-3)
joints = mesh.auto_place_joints(mesh_obj, 200, 2.0, deform_mask=mask)
skin_obj = mesh.auto_skin(mesh_obj, joints)
"""