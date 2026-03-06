from __future__ import annotations

from typing import List

import numpy as np

from maya import cmds
from maya.api import OpenMaya as om, OpenMayaAnim as oma

from ..core.constants import logger


__msl = om.MSelectionList()


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


def get_object(node: str | om.MDagPath) -> om.MObject:
    if isinstance(node, om.MDagPath):
        return node.node()

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


def _build_deformer(mesh_obj: str | om.MObject | om.MDagPath, deformer_type: str, deformer_name: str) -> om.MObject:
    if not isinstance(mesh_obj, om.MObject):
        mesh_obj = get_object(mesh_obj)

    deformer = cmds.createNode(deformer_type, name=deformer_name)
    mesh_name = name_of(mesh_obj)
    orig_obj = create_orig(mesh_obj)
    orig_name = name_of(orig_obj)

    cmds.connectAttr(f"{deformer}.outputGeometry[0]", f"{mesh_name}.inMesh", force=True)
    cmds.connectAttr(f"{orig_name}.outMesh", f"{deformer}.input[0].inputGeometry", force=True)
    cmds.connectAttr(f"{orig_name}.worldMesh[0]", f"{deformer}.originalGeometry[0]", force=True)
    
    return get_object(deformer)


def create_skin(mesh: str | om.MObject | om.MDagPath, influences: list) -> om.MObject:
    if not isinstance(mesh, str):
        mesh = name_of(mesh)
    mesh_short = mesh.split("|")[-1].split(":")[-1]
    deformer = _build_deformer(mesh, "skinCluster", f"SKIN_{mesh_short}")
    deformer_name = name_of(deformer)
    
    for i, influence in enumerate(influences):
        matrix = np.array(cmds.xform(influence, query=True, matrix=True, worldSpace=True))
        inv_matrix = np.linalg.inv(matrix.reshape(4, 4)).flatten()
        cmds.connectAttr(f"{influence}.worldMatrix[0]", f"{deformer_name}.matrix[{i}]", force=True)
        cmds.setAttr(f"{deformer_name}.bindPreMatrix[{i}]", inv_matrix.tolist(), type="matrix")
    
    return deformer


def create_blendshapes(mesh_obj: om.MObject) -> om.MObject:
    mesh_name = name_of(mesh_obj).split("|")[-1].split(":")[-1]

    return _build_deformer(mesh_obj, "blendShape", f"BS_{mesh_name}")


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
        cmds.aliasAttr(f"target_{i}", f'{bs_name}.weight[{i}]')

        input_target_group_element = input_target_group.elementByLogicalIndex(i)
        input_target_item = input_target_group_element.child(0)
        input_target_item_element = input_target_item.elementByLogicalIndex(6000)

        point_array_data = om.MFnPointArrayData()
        data_obj = point_array_data.create(om.MPointArray(deltas[i]))
        input_target_item_element.child(3).setMObject(data_obj)

        input_target_item_element.child(4).setMObject(components_obj)
    
    return bs_obj


def current_time(current) -> int:
    return cmds.currentTime(current)


def get_animation_end() -> int:
    return int(cmds.playbackOptions(query=True, animationEndTime=True))


def get_animation_start() -> int:
    return int(cmds.playbackOptions(query=True, animationStartTime=True))


def create_animation(node: str | om.MObject, attr: str, keys: List[float]) -> om.MObject:
    if isinstance(node, str):
        node = get_object(node)

    try:
        plug = om.MFnDependencyNode(node).findPlug(attr, False)
    except Exception as e:
        logger.error(f"Attribute {attr} does not exists on {name_of(node)} !")
        raise e
    
    start_frame = int(get_animation_start())
    curve_fn = oma.MFnAnimCurve()
    curve_obj = curve_fn.create(plug)
    k_linear = curve_fn.kTangentLinear
    times = [om.MTime(start_frame + i, om.MTime.uiUnit()) for i in range(len(keys))]
    curve_fn.addKeysWithTangents(times, keys, tangentInType=k_linear, tangentOutType=k_linear)

    return curve_obj


def srt_from_matrix(matrix: list | tuple | om.MMatrix) -> List[List[float], List[float], List[float]]:
    if not isinstance(matrix, om.MMatrix):
        matrix = om.MMatrix(matrix)
    
    transformation = om.MTransformationMatrix(matrix)
    
    scale = transformation.scale(om.MSpace.kWorld)
    rotate = transformation.rotation(asQuaternion=False)
    translate = transformation.translation(om.MSpace.kWorld)

    return scale, rotate, translate


def anim_from_matrice(node: str | om.MObect, anim_matrices: List[om.MMatrix]) -> List[om.MObject]:
    anim_data = {"sx": [], "sy": [], "sz": [],
                 "rx": [], "ry": [], "rz": [],
                 "tx": [], "ty": [], "tz": []}

    for matrix in anim_matrices:
        scale, rotate, translate = srt_from_matrix(matrix)
        anim_data["sx"].append(scale[0])
        anim_data["sy"].append(scale[1])
        anim_data["sz"].append(scale[2])
        anim_data["rx"].append(rotate[0])
        anim_data["ry"].append(rotate[1])
        anim_data["rz"].append(rotate[2])
        anim_data["tx"].append(translate[0])
        anim_data["ty"].append(translate[1])
        anim_data["tz"].append(translate[2])
    
    output = []
    for attr, keys in anim_data.items():
        output.append(create_animation(node, attr, keys))
    
    return output
