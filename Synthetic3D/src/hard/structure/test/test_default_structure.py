import numpy as np
from Synthetic3D.src.utilities.view_data import view_vtk_3D_data
from Synthetic3D.src.hard.structure.octahedron import create_octahedron_with_main_point
from Synthetic3D.src.hard.structure.section import Section
from Synthetic3D.src.hard.structure.vector import Vector
from Synthetic3D.src.hard.structure.triangle import Triangle
from Synthetic3D.src.hard.structure.vertex import Vertex, get_int_coords_from_vertex_list
from Synthetic3D.src.hard.drawing_and_filliing.draw_triangle import draw_voxel_triangle

def test_Octahedron():
    print("StartTestOctahedron")
    test_data = np.zeros((128, 128, 128, 3), dtype=np.uint8)

    position = Vector(64,64,64)

    octahedoron = create_octahedron_with_main_point(position,
                                                    32,32,32)

    vertices, edges, triangles = octahedoron
    color = (255, 0, 0)

    int_coords_list = get_int_coords_from_vertex_list(vertices)

    for triangle in triangles:
        v1, v2, v3 = triangle.get_values_by_vertex_indices_from_list(int_coords_list)
        print(v1, v2, v3)
        draw_voxel_triangle(test_data, v1, v2, v3, color, 0)

    view_vtk_3D_data(test_data, vertices, position)

def test_Section():
    print("StartTestSection")
    test_data = np.zeros((128, 128, 128, 3), dtype=np.uint8)

    position = Vector(64, 64, 64)

    direction = Vector(np.random.randn(3))

    section = Section(position, direction, 32, 32, 0)

    vertices = section.contour_list + [Vertex(position, direction)]

    triangles = [
        Triangle((0, 1, 4), (0, 0, 0)),
        Triangle((1, 2, 4), (0, 0, 0)),
        Triangle((2, 3, 4), (0, 0, 0)),
        Triangle((3, 0, 4), (0, 0, 0))
    ]

    int_coords_list = get_int_coords_from_vertex_list(vertices)

    color = (255, 0, 0)
    for triangle in triangles:
        v1, v2, v3 = triangle.get_values_by_vertex_indices_from_list(int_coords_list)
        print(v1, v2, v3)
        draw_voxel_triangle(test_data, v1, v2, v3, color, 0)

    view_vtk_3D_data(test_data, vertices)

if __name__ == "__main__":
    test_Octahedron()
    test_Section()
