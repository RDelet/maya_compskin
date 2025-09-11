from pathlib import Path
from typing import Tuple
import numpy as np

from . import utils
from .constants import logger

from maya import cmds
from maya.api import OpenMaya as om


class Exporter:
    
    def __init__(self, mesh_name: str):
        if not cmds.objExists(mesh_name):
            raise RuntimeError(f"Mesh {mesh_name} does not exist !")
        if not cmds.nodeType(mesh_name) == "mesh":
            raise RuntimeError(f"Node {mesh_name} is not a mesh !")

        self._handle = om.MObjectHandle(utils.get_object(mesh_name))
        self._mesh_fn = om.MFnMesh(self.shape)
    
    @property
    def shape(self) -> om.MObject:
        return self._handle.object()
    
    @staticmethod
    def dump(*args, **kwargs):
        file_path = args[0]
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        output_dir = file_path.parent
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        
        try:
            # Save exemplre
            # np.savez_compressed(file_path.as_posix(),
            #                     rest_verts=rest_vertices,
            #                     rest_faces=rest_faces,
            #                     deltas=deltas)
            np.savez_compressed(*args, **kwargs)
            logger.info(f"\n--- Succès ! ---")
            logger.info(f"Fichier sauvegardé ici : {file_path.as_posix()}")
        except Exception as e:
            raise RuntimeError(f"Error on save npz file {file_path.as_posix()} !")

    def from_blendshape(self):
        rest_vertices, rest_faces = self._get_mesh_data()
        deltas = self._get_delta_from_blendshape()
        
        return rest_vertices, rest_faces, deltas
    
    def from_rig(self):
        # ToDo
        raise NotImplementedError("Export mesh from rig not implemented yet !")

    def _get_mesh_data(self) -> Tuple[np.array, np.array]:
        logger.info(f"Get mesh data from {utils.name_of(self.shape)}")
        rest_vertices = np.array(self._mesh_fn.getPoints(om.MSpace.kObject))
        triangle_counts, triangle_vertices = self._mesh_fn.getTriangles()

        triangles = []
        vertex_index = 0
        for poly_index in range(len(triangle_counts)):
            num_triangles = triangle_counts[poly_index]
            for _ in range(num_triangles):
                v1 = triangle_vertices[vertex_index]
                v2 = triangle_vertices[vertex_index + 1] 
                v3 = triangle_vertices[vertex_index + 2]
                triangles.append([v1, v2, v3])
                vertex_index += 3
        
        return rest_vertices, np.array(triangles)

    def _get_delta_from_blendshape(self) -> np.array:
        logger.info(f"Get blendshape data from {utils.name_of(self.shape)}")
        deformers = utils.find_deformer(self.shape, om.MFn.kBlendShape)
        if not deformers:
            raise RuntimeError(f"No blendshape found on {utils.name_of(self.shape)}")
        if len(deformers) != 1:
            raise RuntimeError(f"Multi blendshape found on {utils.name_of(self.shape)}")
        bs_obj = deformers[0]
        vtx_count = self._mesh_fn.numVertices

        input_target_plug = om.MFnDependencyNode(bs_obj).findPlug("inputTarget", False)
        input_target_element = input_target_plug.elementByLogicalIndex(0)
        input_target_group = input_target_element.child(0)
        target_count = input_target_group.numElements()

        deltas = np.zeros((target_count, vtx_count, 3), dtype=np.float32)
        for i in range(target_count):
            input_target_group_element = input_target_group.elementByPhysicalIndex(i)
            input_target_item = input_target_group_element.child(0)
            input_target_item_element = input_target_item.elementByLogicalIndex(6000)
            fn_point_data = om.MFnPointArrayData(input_target_item_element.child(3).asMObject())
            fn_comp_data = om.MFnComponentListData(input_target_item_element.child(4).asMObject())
            components = []
            for i in range(fn_comp_data.length()):
                fn_single = om.MFnSingleIndexedComponent(fn_comp_data.get(0))
                components.extend(fn_single.getElements())

            if len(components) == 0:
                continue
            deltas[i][np.array(components)] = np.array(fn_point_data.array())[:, :3]


"""
import imp

from maya_compskin.scripts import exporter
imp.reload(exporter)
from maya_compskin.scripts.exporter import Exporter
from maya_compskin.scripts import constants

output_path = constants.in_directory / "Jin.npz"
exporter = Exporter("_MESH_:JIN_HEADShape")
rest_vertices, rest_faces, deltas = exporter.from_blendshape()
exporter.dump(output_path.as_posix(), rest_vertices=rest_vertices, rest_faces=rest_faces, deltas=deltas)
"""
