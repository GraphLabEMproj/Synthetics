from Synthetic3D.src.hard.structure.edge import Edge
from Synthetic3D.src.hard.structure.vertex import Vertex
from Synthetic3D.src.hard.structure.vector import Vector
from Synthetic3D.src.organells.mitohondrion import Mitohondrion
from Synthetic3D.src.hard.structure.triangle import Triangle

import cv2
import numpy as np
from Synthetic3D.src.utilities.view_data import view_vtk_3D_data
from Synthetic3D.src.hard.drawing_and_filliing.draw_triangle import draw_voxel_triangle

from Synthetic3D.src.utilities.logging_config import clear_log_file

def test_frame_mito():
    print("StartTestMitohondrionFrame")
    test_data = np.zeros((512, 512, 1536, 3), dtype=np.uint8)

    dict_params = {
        "mitohondrion": {
            "radius_of_section": 15,
            "len_of_mitohondrion": (512, 1024),
            "thickness": 1
        }
    }

    mito = Mitohondrion(dict_params)

    new_pos = Vector(256, 256, 256)

    print("Len of mito", mito.mito_len)
    print("Number of section:", mito.count_of_section)

    print(mito.view_shell.vertex_list[1])

    vertex_counter = 0
    vertex_list = []
    triangles_list = []

    for i, section in enumerate(mito.section_list):

        vertices = section.contour_list + [section.position]
        print("len_section", len(vertices))
        triangles = [
            Triangle((vertex_counter  , vertex_counter+2, vertex_counter+4), (0, 0, 0)),
            Triangle((vertex_counter+1, vertex_counter+2, vertex_counter+4), (0, 0, 0)),
            Triangle((vertex_counter  , vertex_counter+3, vertex_counter+4), (0, 0, 0)),
            Triangle((vertex_counter+1, vertex_counter+3, vertex_counter+4), (0, 0, 0))
        ]
        triangles_list += triangles.copy()
        vertex_list += vertices
        vertex_counter+=5

    new_vertex_list = []
    for vertex in vertex_list:
        new_vertex_list.append(Vertex(vertex.position + new_pos, vertex.normal))



    print("len triangles_list", len(triangles_list))
    print("len vertex_list", len(vertex_list))

    color = (255, 0, 0)
    for triangle in triangles_list:
        v1, v2, v3 = triangle.get_values_by_vertices_indexes_from_list(new_vertex_list)
        print(v1, v2, v3)
        draw_voxel_triangle(test_data, v1.position, v2.position, v3.position, color, 0)

    view_vtk_3D_data(test_data, new_vertex_list)

def test_Mitohondrion_orientation():
    print("StartTestMitohondrion")

    test_data = np.zeros((512, 512, 512, 3), dtype=np.uint8)

    dict_params = {
        "mitohondrion": {
            "radius_of_section": 22,
            "len_of_mitohondrion": (128, 256),
            "membrane_thickness": 3,
            "membrane_color": (255, 20, 30),
            "num_partition_of_triangles":0
       }
    }

    mito = Mitohondrion(dict_params)

    print("WARNINGS")
    for warning in mito.warnings:
        print(warning)
    print()

    print("ChangePosition")
    print(mito.view_shell.get_frame(0))
    mito.ChangePosition(Vector(256,256,256))
    print(mito.view_shell.get_frame(0))

    print("Rotate")
    print(mito.view_shell.get_frame(0))
    mito.Rotate(Vector(np.random.randint(0, 360, size=3)))
    print(mito.view_shell.get_frame(0))

    print("Len of mito", mito.mito_len)
    print("Number of section:", mito.count_of_section)

    print("Num of sections:", len(mito.section_list))
    print("Num of frames:", len(mito.view_shell.frame_points))

    '''
    for pos in mito.view_shell.get_positions():
        if not isinstance(pos, (Vector, np.ndarray)):
            print(f"positions_list contein grab!!! {type(pos)}")
    for norm in mito.view_shell.get_normales():
        if not isinstance(norm, (Vector, np.ndarray)):
            print(f"normals_list contein grab!!! {type(norm)}")
    for edge in mito.view_shell.edge_list:
        if not isinstance(edge, Edge):
            print(f"edge_list contein grab!!! {type(edge)}")
    '''
    print()
    print("ANGLE", mito.angle)
    mito.Draw(test_data)

    #for i in range(len(mito.view_shell.get_frames())-1):
    #   draw_line_3D(test_data,  mito.view_shell.get_frame(i+1).to_int(), mito.view_shell.get_frame(i).to_int(), (0, 255, 0), 1)

    view_vtk_3D_data(test_data)#, view_vertexes, mito.position)# mito.view_shell.vertex_list[1].vertex)
def test_Mitohondrion():
    print("StartTestMitohondrion")

    test_data = np.zeros((256, 256, 512, 3), dtype=np.uint8)

    dict_params = {
        "mitohondrion": {
            "radius_of_section": 42,
            "len_of_mitohondrion": (128, 384),
            "membrane_thickness": 3,
            "membrane_color": (255, 20, 30),
            "num_partition_of_triangles":5
       }
    }

    mito = Mitohondrion(dict_params)

    print("WARNINGS")
    for warning in mito.warnings:
        print(warning)
    print()

    print("ChangePosition")
    print(mito.view_shell.get_frame(0))
    mito.ChangePosition(Vector(256,128,128))
    print(mito.view_shell.get_frame(0))

    print("Rotate")
    print(mito.view_shell.get_frame(0))
    mito.Rotate(Vector(0, 90, 0))
    print(mito.view_shell.get_frame(0))

    print("Len of mito", mito.mito_len)
    print("Number of section:", mito.count_of_section)

    print("Num of sections:", len(mito.section_list))
    print("Num of frames:", len(mito.view_shell.frame_points))

    '''
    for pos in mito.view_shell.get_positions():
        if not isinstance(pos, (Vector, np.ndarray)):
            print(f"positions_list contein grab!!! {type(pos)}")
    for norm in mito.view_shell.get_normales():
        if not isinstance(norm, (Vector, np.ndarray)):
            print(f"normals_list contein grab!!! {type(norm)}")
    for edge in mito.view_shell.edge_list:
        if not isinstance(edge, Edge):
            print(f"edge_list contein grab!!! {type(edge)}")
    '''
    print()

    mito.Draw(test_data)

    for_gif_data = []
    for z in range(test_data.shape[0]):
        slice = test_data[z,:,:,:]
        if np.any(slice[:,:] != 0):
            for_gif_data.append(slice)
        cv2.imwrite(f"test_gen_mitohondrion/mitohondrion_slice_{z}.png", slice)

    cv2.imwritemulti(f"test_gen_mitohondrion/mitohondrion_shell.gif", for_gif_data)

    #for i in range(len(mito.view_shell.get_frames())-1):
    #   draw_line_3D(test_data,  mito.view_shell.get_frame(i+1).to_int(), mito.view_shell.get_frame(i).to_int(), (0, 255, 0), 1)

    view_vertexes = mito.view_shell.dict_of_shells["external"].get_union_vertex_list()

    view_vtk_3D_data(test_data)#, view_vertexes, mito.position)# mito.view_shell.vertex_list[1].vertex)
def test_Mitohondrion_BIG():
    print("StartTestMitohondrion")

    test_data = np.zeros((512, 512, 1024, 3), dtype=np.uint8)

    dict_params = {
        "mitohondrion": {
            "radius_of_section": 42,
            "len_of_mitohondrion": (384, 768),
            "membrane_thickness": 3,
            "membrane_color": (255, 20, 30),
            "num_partition_of_triangles":5
       }
    }

    mito = Mitohondrion(dict_params)

    print("WARNINGS")
    for warning in mito.warnings:
        print(warning)
    print()

    print("ChangePosition")
    print(mito.view_shell.get_frame(0))
    mito.ChangePosition(Vector(512,256,256))
    print(mito.view_shell.get_frame(0))

    print("Rotate")
    print(mito.view_shell.get_frame(0))
    mito.Rotate(Vector(0, 90, 0))
    print(mito.view_shell.get_frame(0))

    print("Len of mito", mito.mito_len)
    print("Number of section:", mito.count_of_section)

    print("Num of sections:", len(mito.section_list))
    print("Num of frames:", len(mito.view_shell.frame_points))

    '''
    for pos in mito.view_shell.get_positions():
        if not isinstance(pos, (Vector, np.ndarray)):
            print(f"positions_list contein grab!!! {type(pos)}")
    for norm in mito.view_shell.get_normales():
        if not isinstance(norm, (Vector, np.ndarray)):
            print(f"normals_list contein grab!!! {type(norm)}")
    for edge in mito.view_shell.edge_list:
        if not isinstance(edge, Edge):
            print(f"edge_list contein grab!!! {type(edge)}")
    '''
    print()

    mito.Draw(test_data)

    for_gif_data = []
    for z in range(test_data.shape[0]):
        slice = test_data[z,:,:,:]
        if np.any(slice[:,:] != 0):
            for_gif_data.append(slice)
        cv2.imwrite(f"test_gen_mitohondrion/mitohondrion_slice_{z}.png", slice)

    cv2.imwritemulti(f"test_gen_mitohondrion/mitohondrion_shell.gif", for_gif_data)

    #for i in range(len(mito.view_shell.get_frames())-1):
    #   draw_line_3D(test_data,  mito.view_shell.get_frame(i+1).to_int(), mito.view_shell.get_frame(i).to_int(), (0, 255, 0), 1)

    view_vertexes = mito.view_shell.dict_of_shells["external"].get_union_vertex_list()

    view_vtk_3D_data(test_data)#, view_vertexes, mito.position)# mito.view_shell.vertex_list[1].vertex)

if __name__ == "__main__":
    clear_log_file()

    #test_frame_mito()

    #while True:
    #    test_Mitohondrion_orientation()

    test_Mitohondrion()
