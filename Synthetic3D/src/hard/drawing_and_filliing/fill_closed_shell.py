
from Synthetic3D.src.hard.structure.triangle import list_of_triangle_to_2_arrs_vertexes_and_edges
#from Synthetic3D.src.hard.structure.shells import FrameShell

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

def fill_closed_shell(data, shell, color):
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

    #surface = mesh.extract_surface(algorithm='dataset_surface')
    #print(mesh)
    voxel_grid = mesh.voxelize(spacing=1)
    #voxel_grid = surface.voxelize(spacing=1)

    '''
    # Визуализация
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color=color, show_edges=False)
    plotter.show()
    '''

    # Генерируем случайные цвета для каждого треугольника
    num_triangles = mesh.n_cells
    colors = np.random.randint(0, 255, size=(num_triangles, 3))
    mesh.cell_data["random_colors"] = colors

    '''
    # Визуализация, передавая массив цветов
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars="random_colors", rgb=True, show_edges=True)

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

    plotter.show()
    '''

    # Получаем координаты вокселей
    voxel_points = voxel_grid.points  # Nx3 массив
    #print(arr_pos.shape, faces_pv.shape, voxel_points.shape)

    # Предполагаем, что координаты совпадают с индексами data или масштабированы
    # Если координаты — это целые индексы, то:
    indices = np.round(voxel_points).astype(int)

    # Ограничиваем индексами внутри границ
    D, H, W = data.shape[:3]
    valid_mask = (
            (0 <= indices[:, 0]) & (indices[:, 0] < W) &
            (0 <= indices[:, 1]) & (indices[:, 1] < H) &
            (0 <= indices[:, 2]) & (indices[:, 2] < D)
    )
    # Отбираем только допустимые индексы
    valid_indices = indices[valid_mask]

    # Присваиваем цвет по индексам в массиве data с помощью numpy advanced indexing
    data[valid_indices[:, 2], valid_indices[:, 1], valid_indices[:, 0]] = color
    return data

def get_valid_shell_points(data, shell):
    """
    :param data:        - трехмерный массив для рисования
    :param shell:       - замкнутая оболочка для рисования, содержащая вершины и треугольники
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
    surface = mesh.extract_surface(algorithm='dataset_surface')
    #print(mesh)
    voxel_grid = surface.voxelize(spacing=1)

    # Получаем координаты вокселей
    voxel_points = voxel_grid.points  # Nx3 массив
    #print(arr_pos.shape, faces_pv.shape, voxel_points.shape)

    # Предполагаем, что координаты совпадают с индексами data или масштабированы
    # Если координаты — это целые индексы, то:
    indices = np.round(voxel_points).astype(int)

    # Ограничиваем индексами внутри границ
    D, H, W = data.shape[:3]
    valid_mask = (
            (0 <= indices[:, 0]) & (indices[:, 0] < W) &
            (0 <= indices[:, 1]) & (indices[:, 1] < H) &
            (0 <= indices[:, 2]) & (indices[:, 2] < D)
    )
    # Отбираем только допустимые индексы
    valid_indices = indices[valid_mask]
    return valid_indices
