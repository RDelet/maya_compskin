import numpy as np

from .utils import get_object, name_of

from maya.api import OpenMaya as om


mesh = ""
mesh_obj = get_object(mesh)
mesh_fn = om.MFnMesh(mesh_obj)


rest_verts = np.array(mesh_fn.getPoints(om.MSpace.kObject))
triangle_counts, triangle_vertices = mesh_fn.getTriangles()
vtx_count = len(rest_verts)

triangles = []
vertex_index = 0
for poly_index in range(len(triangle_counts)):
    num_triangles = triangle_counts[poly_index]
    for tri in range(num_triangles):
        v1 = triangle_vertices[vertex_index]
        v2 = triangle_vertices[vertex_index + 1] 
        v3 = triangle_vertices[vertex_index + 2]
        triangles.append([v1, v2, v3])
        vertex_index += 3
rest_faces = np.array(triangles)

# ToDo: Create BlendShape utils
"""
bs_obj = BlendShape.find(mesh_obj)
if not bs_obj:
    raise RuntimeError(f"No blendshape found on {name_of(mesh_obj)}")
qd_bs = BlendShape(bs_obj)
target_count = qd_bs.count()
deltas = np.zeros((target_count, vtx_count, 3), dtype=np.float32)
for i in range(target_count):
    components = np.array(qd_bs.get_components_indices(i), dtype=np.int32)
    if len(components) == 0:
        continue
    d = qd_bs.get_target_offset(i)
    deltas[i][components] = np.array([[d[j].x, d[j].y, d[j].z] for j in range(d.length())])
"""

try:
    file_path = r""
    np.savez_compressed(file_path,
                        rest_verts=rest_verts,
                        rest_faces=rest_faces,
                        deltas=deltas)
    print(f"\n--- Succès ! ---")
    print(f"Fichier sauvegardé ici : {file_path}")
except Exception as e:
    raise RuntimeError(f"Error on save npz file {file_path} !")
    cmds.error(f"Une erreur est survenue lors de la sauvegarde du fichier .npz : {e}")