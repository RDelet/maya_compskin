from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from maya import cmds
from maya.api import OpenMaya as om, OpenMayaAnim as oma

from .constants import logger, in_directory, out_directory

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


def get_in_from_name(file_name: str) -> Path:
    path = in_directory / f"{file_name}.npz"
    if not path.exists():
        raise RuntimeError(f'File "{file_name}.npz" not found !')
    
    return path


def get_out_from_name(file_name: str) -> Path:
    path = out_directory / file_name / "result.npz"
    if not path.exists():
        raise RuntimeError(f'No result file found for "{file_name}" !')
    
    return path


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


def is_valid(handle: om.MObject | om.MDagPath | om.MObjectHandle) -> bool:
    """!@brief Checks whether the MObject is still valid (i.e. associated to an existing object) or not."""
    if isinstance(handle, om.MDagPath):
        handle = handle.node()
    if isinstance(handle, om.MObject):
        handle = om.MObjectHandle(handle)

    return handle.isAlive() and handle.isValid() and not handle.object().isNull()


def graph_iterator(node: str | om.MObject | om.MDagPath,
                   mfn_type: om.MFn = om.MFn.kDependencyNode,
                   direction=om.MItDependencyGraph.kUpstream,
                   traversal=om.MItDependencyGraph.kBreadthFirst,
                   level=om.MItDependencyGraph.kNodeLevel) -> om.MItDependencyGraph:
    if isinstance(node, str):
        node = get_object(node)
    elif isinstance(node, om.MDagPath):
        node = node.node()

    return om.MItDependencyGraph(node, mfn_type, direction, traversal, level)


def find_deformer(shape_obj: om.MObject, deformer_type: int = om.MFn.kGeometryFilt) -> om.MObject:
    it = graph_iterator(shape_obj, deformer_type)
    output = []
    while not it.isDone():
        output.append(it.currentNode())
        it.next()

    return output


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


def get_mesh_components(mesh_obj: str | om.MObject) -> om.MObject:
    if isinstance(mesh_obj, str):
        mesh_obj = get_object(mesh_obj)
    if not mesh_obj.hasFn(om.MFn.kMesh):
        raise TypeError(f"Node {name_of(mesh_obj)} must be a mesh not {mesh_obj.apiTypeStr()}")

    mit_vtx = om.MItMeshVertex(get_path(mesh_obj))
    single_component = om.MFnSingleIndexedComponent()
    component = single_component.create(om.MFn.kMeshVertComponent)
    while not mit_vtx.isDone():
        single_component.addElement(mit_vtx.index())
        mit_vtx.next()

    return component


def _build_deformer(mesh_obj: om.MObject, deformer_type: str, deformer_name: str) -> om.MObject:
    deformer = cmds.createNode(deformer_type, name=deformer_name)
    mesh_name = name_of(mesh_obj)
    orig_obj = create_orig(mesh_obj)
    orig_name = name_of(orig_obj)

    cmds.connectAttr(f"{deformer}.outputGeometry[0]", f"{mesh_name}.inMesh", force=True)
    cmds.connectAttr(f"{orig_name}.outMesh", f"{deformer}.input[0].inputGeometry", force=True)
    cmds.connectAttr(f"{orig_name}.worldMesh[0]", f"{deformer}.originalGeometry[0]", force=True)
    
    return get_object(deformer)


def create_skin(mesh_obj: om.MObject, influences: list) -> om.MObject:
    mesh_name = name_of(mesh_obj).split("|")[-1].split(":")[-1]
    deformer = _build_deformer(mesh_obj, "skinCluster", f"SKIN_{mesh_name}")
    deformer_name = name_of(deformer)
    
    for i, influence in enumerate(influences):
        matrix = om.MMatrix(cmds.xform(influence, query=True, matrix=True, worldSpace=True))
        cmds.connectAttr(f"{influence}.worldMatrix[0]", f"{deformer_name}.matrix[{i}]", force=True)
        cmds.setAttr(f"{deformer_name}.bindPreMatrix[{i}]", matrix.inverse(), type="matrix")
    
    return deformer


def create_blendshapes(mesh_obj: om.MObject) -> om.MObject:
    mesh_name = name_of(mesh_obj).split("|")[-1].split(":")[-1]

    return _build_deformer(mesh_obj, "blendShape", f"BS_{mesh_name}")


def set_skin_weights(skin_obj: om.MObject, weights: np.array, normalize: bool = True):
    skin_name = name_of(skin_obj)
    skin_fn = oma.MFnSkinCluster(skin_obj)
    output_shapes = skin_fn.getOutputGeometry()
    if len(output_shapes) == 0:
        raise RuntimeError(f"No output geometry found on {skin_name} !")
    mesh_obj = output_shapes[0]
    mesh_path = get_path(mesh_obj)
    component = get_mesh_components(mesh_obj)
    influences_ids = om.MIntArray(list(range(weights.shape[1])))
    weights = om.MDoubleArray(weights.flatten())    
    
    return skin_fn.setWeights(mesh_path, component, influences_ids, weights, normalize, returnOldWeights=True)


def set_blendshape_targets(shape: om.MObject, deltas: np.array) -> om.MObject:
    bs_obj = create_blendshapes(shape)
    bs_name = name_of(bs_obj)

    input_target_plug = om.MFnDependencyNode(bs_obj).findPlug("inputTarget", False)
    input_target_element = input_target_plug.elementByLogicalIndex(0)
    input_target_group = input_target_element.child(0)

    single_component = om.MFnSingleIndexedComponent()
    components = single_component.create(om.MFn.kMeshVertComponent)
    single_component.addElements(np.arange(len(deltas[0])))
    components_fn = om.MFnComponentListData()
    components_obj = components_fn.create()
    components_fn.add(components)

    for i in range(deltas.shape[0]):
        cmds.setAttr(f"{bs_name}.weight[{i}]", 0.0)
        cmds.aliasAttr(f"hodor{i}", f'{bs_name}.weight[{i}]')

        input_target_group_element = input_target_group.elementByLogicalIndex(i)
        input_target_item = input_target_group_element.child(0)
        input_target_item_element = input_target_item.elementByLogicalIndex(6000)

        point_array_data = om.MFnPointArrayData()
        data_obj = point_array_data.create(om.MPointArray(deltas[i]))
        input_target_item_element.child(3).setMObject(data_obj)

        input_target_item_element.child(4).setMObject(components_obj)
    
    return bs_obj



"""
from maya_compskin.scripts import utils, constants

npz_path = constants.in_directory / "aura.npz"
utils.npz_to_mesh(npz_path, build_blendshapes=False)
"""