from Synthetic3D.src.hard.structure.simple_tubular_cristae import create_tubular_cristae

import numpy as np
from Synthetic3D.src.hard.structure.section import Section
from Synthetic3D.src.hard.structure.vector import Vector
from Synthetic3D.src.hard.structure.shells import FrameShell
from Synthetic3D.src.hard.structure.edge import Edge
from Synthetic3D.src.hard.structure.triangle import list_of_triangle_to_2_arrs_vertexes_and_edges
from Synthetic3D.src.utilities.view_data import view_vtk_3D_data

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

def added_frames_in_plotter(plotter, arr_pos):
    # Создаем списки для точек и линий
    all_lines = []
    colors_list = []

    if len(arr_pos) < 2:
        print(f"arr_pos to small", arr_pos)
    # создаем массив линий
    for i in range(len(arr_pos) - 1):
        all_lines.extend([2, i, i + 1])

        # генерируем цвет для этого ребра
        color = np.clip(np.random.normal(0.5, 0.2, 3), 0, 1)
        colors_list.append(color)

    # объединяем все точки
    all_lines = np.array(all_lines)
    colors_array = np.array(colors_list)
    all_points = np.array(arr_pos)

    # создаем PolyData с одним объектом
    line_mesh = pv.PolyData(all_points, lines=all_lines)

    # задаем цвет
    line_mesh['colors'] = colors_array #np.repeat(colors_array, 2, axis=0)  # повторяем для каждой
    plotter.add_mesh(line_mesh, scalars='colors', line_width=3, show_scalar_bar=False)

    # --- Добавляем подписи вершинам ---
    # Формируем список индексов вершин (например, все уникальные вершины)
    vertex_indices = [i for i in range(len(arr_pos))]
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


def added_cylinder_mesh(plotter, base_center, direction, radius):
    """
    Создает меш цилиндра по основанию, направлению, радиусу и высоте.
    """
    # Нормализуем направление
    direction = np.array(direction)
    height = np.linalg.norm(direction)
    direction /= height

    # Создаем цилиндр
    cylinder = pv.Cylinder(center=base_center, direction=direction, radius=radius, height=2 * height, resolution=50)
    plotter.add_mesh(cylinder, color='lightblue', opacity=0.5)  # полупрозрачный

def added_arrow_mesh(plotter, direction):
    # Создайте стрелку
    arrow = pv.Arrow(np.array((0, 0, 0)), direction, scale=50)
    # Добавьте стрелку в плоттер
    plotter.add_mesh(arrow, color='red')

def view_frame_shell(shell_frames, print_mesh=True, cristaes_frames=None, cylinder_data=None, target_dir=None):

    """
    :param shell:       - замкнутая оболочка для рисования, содержащая вершины и треугольники
    :return: заполненный цветом color shell в data
    """

    arr_pos = shell_frames.vertexes.positions.copy_data_as_array()
    #print(arr_pos)
    triangle_list = shell_frames.triangle_list
    triangle_arr_indexes, _ = list_of_triangle_to_2_arrs_vertexes_and_edges(triangle_list)
    # PyVista требует формат faces: [n_pts, v0, v1, v2, ...]
    # где n_pts — число вершин в многоугольнике (для треугольника всегда 3)

    faces_pv = expand_faces(triangle_arr_indexes)
    #print(faces_pv)
    #print(faces_pv)

    mesh = pv.PolyData(arr_pos, faces_pv)

    frame_arr_pos = shell_frames.get_frames().copy_data_as_array()

    # Визуализация
    plotter = pv.Plotter()
    if print_mesh:
        plotter.add_mesh(mesh, color=(255,0,0), show_edges=False)
    added_frames_in_plotter(plotter, frame_arr_pos)
    if cylinder_data is not None:
        added_cylinder_mesh(plotter, cylinder_data[0], cylinder_data[1], cylinder_data[2])
    #added_triangle_edges_in_plotter(plotter, arr_pos, shell)

    print("cristaes_frames", cristaes_frames)
    if cristaes_frames is not None:
        for cristaes_frame in cristaes_frames:
            arr_frame = np.array(cristaes_frame)
            added_frames_in_plotter(plotter, arr_frame)

    if target_dir is not None:
        added_arrow_mesh(plotter, target_dir)


    plotter.show()





def test_create_tubular_cristae():

    section_list = []

    sec1 = Section(Vector(0, 0, 0), Vector(0, 0, 1), 40, 40, angle=0)

    section_list.append(sec1)

    shell = FrameShell()
    shell.add_vertex_list(sec1.contour_list)
    shell.add_edges(Edge(0, 1),  # 1-3
                    Edge(1, 2),  # 2-3
                    Edge(2, 3),  # 1-4
                    Edge(3, 0))  # 2-4

    shell.add_frame_point(Vector(0, 0, 0))

    #shell.reverse()
    now_index_val = shell.get_vertex_count()
    last_indexes = [[0, 1, 2, 3],
                    [0, 1, 2, 3]]

    current_pos = Vector()

    for i in range(4):
        vec = Vector(0, 0, 1) + Vector((*np.random.random(size=2), 0))
        vec /= np.linalg.norm(vec)

        current_pos += vec*50

        new_sec = Section(current_pos, vec, 40, 40, angle=0)

        shell.add_frame_point(current_pos.copy())
        section_list.append(new_sec)

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

    view_frame_shell(shell, print_mesh=True)


    print("cristaes_frames_len", len(shell.get_frames()))
    print("section_list_len", len(section_list))

    cristae_frame_list, cristae_radius_list = create_tubular_cristae(shell.get_frames(),
                                                                     section_list,
                                                                     cristae_step = 10,
                                                                     cristae_radius = (5, 7),
                                                                     cristae_gap = 2,
                                                                     cristae_angle_deviation=15,
                                                                     density_cristae=0.25,
                                                                     max_count_added_cristae=1000,
                                                                     max_count_added_cristae_continue=1000,
                                                                     angle_by_frame_dir=45)

    view_frame_shell(shell, print_mesh=False, cristaes_frames=cristae_frame_list)


if __name__ == "__main__":
    test_create_tubular_cristae()
