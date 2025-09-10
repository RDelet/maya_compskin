from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from maya import cmds
from maya.api import OpenMaya as om

from .constants import logger

__msl = om.MSelectionList()


# --------------------------------------------------
# File
# --------------------------------------------------

def dump_json(data: dict, output_path: str | Path):
    if isinstance(output_path, str):
        output_path = Path(output_path)
    
    try:
        with open(output_path.as_posix(), 'w') as f:
            json.dump(data, f, indent=4)
        logger.debug(f"Successfully saved JSON to '{output_path.as_posix()}'")
    except Exception:
        raise RuntimeError("Failed to write JSON file.")


def read_npz(npz_path: str | Path):
    if isinstance(npz_path, str):
        npz_path = Path(npz_path)
    
    try:
        logger.info(f"Read npz file: '{npz_path.as_posix()}'.")
        return np.load(npz_path.as_posix(), allow_pickle=True)
    except:
        raise RuntimeError("Error on read npz file.")


# --------------------------------------------------
# Maya
# --------------------------------------------------

def name_of(node: om.MObject | om.MDagPath) -> str:
    if isinstance(node, om.MObject):
        if node.hasFn(om.MFn.kDagNode):
            return om.MFnDagNode(node).fullPathName()
        else:
            return om.MFnDependencyNode(node).name()
    else:
        return node.fullPathName()


def assign_shader(mesh: om.MObject, shader_name: str = "initialShadingGroup"):
    shader_fn = om.MFnSet(get_object(shader_name))
    shader_fn.addMember(mesh)


def get_object(node: str) -> om.MObject:
    try:
        __msl.clear()
        __msl.add(node)
        return __msl.getDependNode(0)
    except Exception:
        if len(cmds.ls(node)) > 1:
            raise RuntimeError(f'Multi node found with name "{node}" !')
        raise RuntimeError(f'Node "{node}" does not exist !')


def get_path(node: str | om.MObject) -> om.MObject:
    if isinstance(node, om.MObject):
        return om.MDagPath.getAPathTo(node)

    try:
        __msl.clear()
        __msl.add(node)
        return __msl.getDagPath(0)
    except Exception:
        if len(cmds.ls(node)) > 1:
            raise RuntimeError(f'Multi node found with name "{node}" !')
        raise RuntimeError(f'Node "{node}" does not exist !')


def create_orig(mesh_obj: om.MObject) -> om.MObject:
    transform_path = get_path(mesh_obj)
    transform_path.pop()
    transform_fn = om.MFnDagNode(transform_path)

    orig_obj = om.MFnMesh().copy(mesh_obj, transform_path.node())
    orig_fn = om.MFnDagNode(orig_obj)
    plug = orig_fn.findPlug("intermediateObject", False)
    plug.setBool(True)
    
    mdg_mod = om.MDGModifier()
    mdg_mod.renameNode(orig_obj, f"{transform_fn.name()}OrigShape")
    mdg_mod.doIt()
    
    return orig_obj


def create_skin(mesh_obj: om.MObject, influences: list) -> om.MObject:
    skin_name = cmds.createNode("skinCluster")
    mesh_name = name_of(mesh_obj)
    orig_obj = create_orig(mesh_obj)
    orig_name = name_of(orig_obj)

    cmds.connectAttr(f"{skin_name}.outputGeometry[0]", f"{mesh_name}.inMesh", force=True)
    cmds.connectAttr(f"{orig_name}.outMesh", f"{skin_name}.input[0].inputGeometry", force=True)
    cmds.connectAttr(f"{orig_name}.worldMesh[0]", f"{skin_name}.originalGeometry[0]", force=True)
    
    for i, influence in enumerate(influences):
        matrix = om.MMatrix(cmds.xform(influence, query=True, matrix=True, worldSpace=True))
        cmds.connectAttr(f"{influence}.worldMatrix[0]", f"{skin_name}.matrix[{i}]", force=True)
        cmds.setAttr(f"{skin_name}.bindPreMatrix[{i}]", matrix.inverse(), type="matrix")
    
    return get_object(skin_name)


def set_weights(skin_obj: om.MObject, weights: np.array):
    skin_name = name_of(skin_obj)
    vtx_count = weights.shape[0]
    inf_count = weights.shape[1]
    pouet = weights.tolist()
    for i in range(vtx_count):
        for j in range(inf_count):
            cmds.setAttr(f"{skin_name}.weightList[{i}].weights[{j}]", pouet[i][j])


# --------------------------------------------------
# Misc
# --------------------------------------------------

def npz_to_mesh(npz_path: Path):

    npz_data = read_npz(npz_path)
    mesh_name = npz_path.stem

    # Get data
    rest_points = om.MPointArray(npz_data.get('rest_verts'))
    rest_faces = npz_data.get('rest_faces')
    poly_count = om.MIntArray([len(f) for f in rest_faces])
    poly_connect = om.MIntArray(rest_faces.ravel())
    u_values = []
    v_values = []

    # Build mesh
    mesh_transform = get_object(cmds.createNode("transform", name=mesh_name))
    mfn_mesh = om.MFnMesh()
    mesh_obj = mfn_mesh.create(rest_points, poly_count, poly_connect, u_values, v_values, mesh_transform)
    assign_shader(mesh_obj)

    mdg_mod = om.MDGModifier()
    mdg_mod.renameNode(mesh_obj, f"{mesh_name}Shape")
    mdg_mod.doIt()
    
    return mesh_obj
