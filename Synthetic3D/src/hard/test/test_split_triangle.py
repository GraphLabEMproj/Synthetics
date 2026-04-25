



from Synthetic3D.src.hard.structure.octahedron import create_octahedron_with_main_point
from Synthetic3D.src.hard.structure.shells import OuterShell
from Synthetic3D.src.hard.structure.triangle import list_of_triangle_to_2_arrs_vertexes_and_edges

import numpy as np
import pyvista as pv

def expand_faces(faces):
    """
    Быстро расширяет массив треугольников (Mx3) в (Mx4),
    где первый элемент каждой строки — 3.
    """

    # Создаем массив (M,4), заполняем его нулями
    faces_expanded = np.zeros((faces.shape[0], 4), dtype=int)
    # Первый столбец — 3
    faces_expanded[:, 0] = 3
    # Остальные — исходные вершины
    faces_expanded[:, 1:] = faces
    return faces_expanded

def view_shell(shell):

    """
    :param data:        - трехмерный массив для рисования
    :param shell:       - замкнутая оболочка для рисования, содержащая вершины и треугольники
    :param color:       - цвет заполнения
    :return: заполненный цветом color shell в data
    """

    arr_pos = shell.vertexes.positions.to_int().copy_data_as_array().astype(float)
    #print(arr_pos)
    triangle_list = shell.triangle_list
    triangle_arr_indexes, _ = list_of_triangle_to_2_arrs_vertexes_and_edges(triangle_list)
    # PyVista требует формат faces: [n_pts, v0, v1, v2, ...]
    # где n_pts — число вершин в многоугольнике (для треугольника всегда 3)
    faces_pv = expand_faces(triangle_arr_indexes)
    #print(faces_pv)

    mesh = pv.PolyData(arr_pos, faces_pv)

    # Визуализация
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color=(255,0,0), show_edges=False)
    plotter.show()

def test_split_triangles():
    v, e, t = create_octahedron_with_main_point((20, 20, 20), 10, 10, 10)


    shell = OuterShell()

    shell.add_vertex_list(v)
    shell.add_edges_list(e)
    shell.add_triangles_list(t)

    for i in range(3):
        print(shell)

        for ty in shell.triangle_list:
            print("\t", ty)
            for edeas in ty.get_values_by_edge_indices_from_list(shell.edge_list):
                print(edeas)

        view_shell(shell)
        shell.Partition_of_triangles(1)

    view_shell(shell)




if __name__ == "__main__":
    test_split_triangles()
