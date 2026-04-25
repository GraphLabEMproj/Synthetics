from Synthetic3D.src.hard.structure.section import Section
from Synthetic3D.src.hard.structure.vector import Vector
from Synthetic3D.src.hard.structure.shells import OuterShell
from Synthetic3D.src.hard.structure.triangle import list_of_triangle_to_2_arrs_vertexes_and_edges
from Synthetic3D.src.hard.structure.edge import Edge
from Synthetic3D.src.utilities.view_data import view_vtk_3D_data

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

def added_edges_in_plotter(plotter, arr_pos, shell):
    # Создаем списки для точек и линий
    all_points = []
    all_lines = []
    colors_list = []

    # создаем массив линий
    start_idx = 0
    for edge in shell.edge_list:
        # добавляем точки
        all_points.append(arr_pos[edge.v1_index])
        all_points.append(arr_pos[edge.v2_index])

        all_lines.extend([2, start_idx, start_idx + 1])
        start_idx += 2

        # генерируем цвет для этого ребра
        color = np.clip(np.random.normal(0.5, 0.2, 3), 0, 1)
        colors_list.append(color)

    # объединяем все точки
    all_points = np.array(all_points)
    all_lines = np.array(all_lines)
    colors_array = np.array(colors_list)

    # создаем PolyData с одним объектом
    line_mesh = pv.PolyData(all_points, lines=all_lines)

    # задаем цвет
    line_mesh['colors'] = np.repeat(colors_array, 2, axis=0)  # повторяем для каждой
    plotter.add_mesh(line_mesh, scalars='colors', line_width=3, show_scalar_bar=False)

    # --- Добавляем подписи вершинам ---
    # Формируем список индексов вершин (например, все уникальные вершины)
    vertex_indices = [edge.v1_index for edge in shell.edge_list] + [edge.v2_index for edge in shell.edge_list]
    vertex_indices = list(set(vertex_indices))  # уникальные индексы

    # Получаем координаты подписей
    label_points = arr_pos[vertex_indices]
    # Создаем подписи индексами
    labels = [str(idx) for idx in vertex_indices]
    # Добавляем подписи
    plotter.add_point_labels(label_points, labels, font_size=12, point_color='yellow', point_size=20)


def added_triangle_edges_in_plotter(plotter, arr_pos, shell):
    # Создаем списки для точек и линий
    all_points = []
    all_lines = []
    colors_list = []

    # создаем массив линий
    start_idx = 0
    for triangle in shell.triangle_list:
        for edge_index in triangle.edge_indexes:
            # добавляем точки
            edge = shell.edge_list[edge_index]
            all_points.append(arr_pos[edge.v1_index])
            all_points.append(arr_pos[edge.v2_index])

            all_lines.extend([2, start_idx, start_idx + 1])
            start_idx += 2

            # генерируем цвет для этого ребра
            color = np.clip(np.random.normal(0.5, 0.2, 3), 0, 1)
            colors_list.append(color)

    # объединяем все точки
    all_points = np.array(all_points)
    all_lines = np.array(all_lines)
    colors_array = np.array(colors_list)

    # создаем PolyData с одним объектом
    line_mesh = pv.PolyData(all_points, lines=all_lines)

    # задаем цвет
    line_mesh['colors'] = np.repeat(colors_array, 2, axis=0)  # повторяем для каждой
    plotter.add_mesh(line_mesh, scalars='colors', line_width=3, show_scalar_bar=False)

    # --- Добавляем подписи вершинам ---
    # Формируем список индексов вершин (например, все уникальные вершины)
    vertex_indices = [edge.v1_index for edge in shell.edge_list] + [edge.v2_index for edge in shell.edge_list]
    vertex_indices = list(set(vertex_indices))  # уникальные индексы

    # Получаем координаты подписей
    label_points = arr_pos[vertex_indices]
    # Создаем подписи индексами
    labels = [str(idx) for idx in vertex_indices]
    # Добавляем подписи
    plotter.add_point_labels(label_points, labels, font_size=12, point_color='yellow', point_size=20)


def view_shell(shell):

    """
    :param data:        - трехмерный массив для рисования
    :param shell:       - замкнутая оболочка для рисования, содержащая вершины и треугольники
    :param color:       - цвет заполнения
    :return: заполненный цветом color shell в data
    """

    arr_pos = shell.vertexes.positions.copy_data_as_array().astype(float)
    #print(arr_pos)
    triangle_list = shell.triangle_list
    triangle_arr_indexes, _ = list_of_triangle_to_2_arrs_vertexes_and_edges(triangle_list)
    # PyVista требует формат faces: [n_pts, v0, v1, v2, ...]
    # где n_pts — число вершин в многоугольнике (для треугольника всегда 3)

    faces_pv = expand_faces(triangle_arr_indexes)
    print(faces_pv)
    #print(faces_pv)

    mesh = pv.PolyData(arr_pos, faces_pv)

    # Визуализация
    plotter = pv.Plotter()
    #plotter.add_mesh(mesh, color=(255,0,0), show_edges=False)
    #added_edges_in_plotter(plotter, arr_pos, shell)
    added_triangle_edges_in_plotter(plotter, arr_pos, shell)
    plotter.show()

def test_split_triangles():
    sec1 = Section(Vector(0, 0, 0), Vector(1, 0, 0), 100, 100, angle=0)

    shell = OuterShell()
    shell.add_vertex_list(sec1.contour_list)
    shell.add_edges(Edge(0, 1),  # 1-3
                    Edge(1, 2),  # 2-3
                    Edge(2, 3),  # 1-4
                    Edge(3, 0))  # 2-4

    last_indexes = [[0, 1, 2, 3], [0, 1, 2, 3]]

    for i in range(2):
        new_sec = Section(Vector(50*(1+i) , 0, 0), Vector(1, 0, 0), 100, 100, angle=0)

        # NEW SECTION INDEXES
        new_v_index = shell.get_vertex_count()
        shell.add_vertex_list(new_sec.contour_list)
        new_v_i_1 = new_v_index
        new_v_i_2 = new_v_index + 1
        new_v_i_3 = new_v_index + 2
        new_v_i_4 = new_v_index + 3

        new_ed_index = shell.get_edge_count()
        ed1 = Edge(new_v_i_1, new_v_i_2)
        ed2 = Edge(new_v_i_2, new_v_i_3)
        ed3 = Edge(new_v_i_3, new_v_i_4)
        ed4 = Edge(new_v_i_1, new_v_i_4)
        shell.add_edges(ed1, ed2, ed3, ed4)

        new_e_i_1 = new_ed_index
        new_e_i_2 = new_ed_index + 1
        new_e_i_3 = new_ed_index + 2
        new_e_i_4 = new_ed_index + 3

        new_section_indexes = [[new_v_i_1, new_v_i_2, new_v_i_3, new_v_i_4],
                               [new_e_i_1, new_e_i_2, new_e_i_3, new_e_i_4]]

        Section.AddSection(last_indexes, new_section_indexes, shell)
        last_indexes = new_section_indexes

    np.set_printoptions(precision=3, suppress=True)

    shell.triangle_list = shell.triangle_list[:]

    for i in range(3):
        print(shell)
        print("POS PRINT *********************************")
        for pos in shell.vertexes.positions:
            print(pos)
        for efe in shell.edge_list:
            print("\t", efe)
        for ty in shell.triangle_list:
            print("\t", ty)
            vertex_list = ty.vertex_indexes
            for edeas in ty.get_values_by_edge_indices_from_list(shell.edge_list):
                print(edeas)
                if not (edeas.v1_index in vertex_list or edeas.v2_index in vertex_list):
                    raise ValueError(f"Нет индексов {edeas} в треугольнике {vertex_list}")

        view_shell(shell)
        shell.Partition_of_triangles(1)

    print(shell)
    print("POS PRINT *********************************")
    for pos in shell.vertexes.positions:
        print(pos)
    for efe in shell.edge_list:
        print("\t", efe)
    for ty in shell.triangle_list:
        print("\t", ty)

        for edeas in ty.get_values_by_edge_indices_from_list(shell.edge_list):
            print(edeas)
    view_shell(shell)

    view_vtk_3D_data(np.ones((1,1,1)), shell.get_union_vertex_list())



def test_split_triangles_reverse():
    sec1 = Section(Vector(0, 0, 0), Vector(0, 0, 1), 100, 100, angle=0)

    shell = OuterShell()
    shell.add_vertex_list(sec1.contour_list)
    shell.add_edges(Edge(0, 1),  # 1-3
                    Edge(1, 2),  # 2-3
                    Edge(2, 3),  # 1-4
                    Edge(3, 0))  # 2-4

    shell.reverse()
    now_index_val = shell.get_vertex_count()
    last_indexes = [[now_index_val-1, now_index_val-2, now_index_val-3, now_index_val-4],
                    [now_index_val-1, now_index_val-2, now_index_val-3, now_index_val-4]]

    current_pos = Vector()

    for i in range(4):
        vec = Vector(0, 0, -1) + Vector((*np.random.random(size=2), 0))
        vec /= np.linalg.norm(vec)

        current_pos += vec*100

        new_sec = Section(current_pos, vec, 100, 100, angle=0, reversed=-1)

        # NEW SECTION INDEXES
        new_v_index = shell.get_vertex_count()
        shell.add_vertex_list(new_sec.contour_list)
        new_v_i_1 = new_v_index
        new_v_i_2 = new_v_index + 1
        new_v_i_3 = new_v_index + 2
        new_v_i_4 = new_v_index + 3

        new_ed_index = shell.get_edge_count()
        ed1 = Edge(new_v_i_1, new_v_i_2)
        ed2 = Edge(new_v_i_2, new_v_i_3)
        ed3 = Edge(new_v_i_3, new_v_i_4)
        ed4 = Edge(new_v_i_4, new_v_i_1)
        shell.add_edges(ed1, ed2, ed3, ed4)

        new_e_i_1 = new_ed_index
        new_e_i_2 = new_ed_index + 1
        new_e_i_3 = new_ed_index + 2
        new_e_i_4 = new_ed_index + 3

        new_section_indexes = [[new_v_i_1, new_v_i_2, new_v_i_3, new_v_i_4],
                               [new_e_i_1, new_e_i_2, new_e_i_3, new_e_i_4]]

        Section.AddSection(last_indexes, new_section_indexes, shell)
        last_indexes = new_section_indexes

    np.set_printoptions(precision=3, suppress=True)

    shell.triangle_list = shell.triangle_list[:]

    for i in range(3):
        print(shell)
        print("POS PRINT *********************************")
        for pos in shell.vertexes.positions:
            print(pos)
        for efe in shell.edge_list:
            print("\t", efe)
        for ty in shell.triangle_list:
            print("\t", ty)
            vertex_list = ty.vertex_indexes
            for edeas in ty.get_values_by_edge_indices_from_list(shell.edge_list):
                print(edeas)
                if not (edeas.v1_index in vertex_list or edeas.v2_index in vertex_list):
                    raise ValueError(f"Нет индексов {edeas} в треугольнике {vertex_list}")

        view_shell(shell)
        shell.Partition_of_triangles(1)

    print(shell)
    print("POS PRINT *********************************")
    for pos in shell.vertexes.positions:
        print(pos)
    for efe in shell.edge_list:
        print("\t", efe)
    for ty in shell.triangle_list:
        print("\t", ty)

        for edeas in ty.get_values_by_edge_indices_from_list(shell.edge_list):
            print(edeas)
    view_shell(shell)

    view_vtk_3D_data(np.ones((1, 1, 1)), shell.get_union_vertex_list())

if __name__ == "__main__":
    #test_split_triangles()
    test_split_triangles_reverse()
