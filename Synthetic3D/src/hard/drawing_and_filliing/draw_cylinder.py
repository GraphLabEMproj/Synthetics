import numpy as np

from Synthetic3D.src.hard.drawing_and_filliing.draw_line_3d import draw_line_3D, generate_points_by_line_3D
from Synthetic3D.src.hard.vector_operation import create_new_2d_plane_by_point_and_normal, plane_coord_to_3d, array_plane_coords_to_3d, normalize_vector
from Synthetic3D.src.hard.drawing_and_filliing.draw_2d import get_circle_point_2D
from Synthetic3D.src.utilities.logging_config import logger
from Synthetic3D.src.hard.drawing_and_filliing.fill_sphere import fill_small_sphere


def draw_2d_points_in_3d(data, UVpoints, start_point, direction, color, scale):
    u_axis, v_axis = create_new_2d_plane_by_point_and_normal(direction)
    end_point = start_point + direction

    d, h, w = data.shape[:3]

    UVpoints = np.array(UVpoints, dtype=float)

    def write_cycle(point_3d):
        draw_points = np.round(array_plane_coords_to_3d(UVpoints, point3d, u_axis, v_axis)).astype(int)
        # Создаем булеву маску для фильтрации
        mask = (
                (draw_points[:, 0] >= 0) & (draw_points[:, 0] < w) &  # x в диапазоне
                (draw_points[:, 1] >= 0) & (draw_points[:, 1] < h) &  # y в диапазоне
                (draw_points[:, 2] >= 0) & (draw_points[:, 2] < d)    # z в диапазоне
        )
        # Применяем маску
        filtered_coords = draw_points[mask]
        if len(filtered_coords) > 0:
            # Распакуем координаты
            x_indices = filtered_coords[:, 0]
            y_indices = filtered_coords[:, 1]
            z_indices = filtered_coords[:, 2]
            # Создаем массив индексов для обращения
            # Используем np.r_ для индексов
            indices = (z_indices, y_indices, x_indices)
            data[indices] = color

    point3d = start_point.copy().astype(float)
    norm_dir = normalize_vector(direction) / scale
    while np.linalg.norm(point3d - end_point) > 1 / scale:
        write_cycle(point3d)
        point3d += norm_dir
    write_cycle(end_point)

def draw_small_cylinder(data, point, direction, radius, color, thickness, delta=0.1, scale=2):
    """
    Функция рисует оболочку на расстоянии radius от вектора direction с началом в point при помощи параллельных линий.

    Args:
        data:       - Трехмерный массив данных для рисования
        point:      - Начальная точка цилиндра
        direction:  - Вектор направления, также задающий и длину
        radius:     - Радиус цилиндра
        color:      - Цвет для оболочки
        thickness:  - Толщина оболочки
    """


    #UVpoints = get_circle_point_2D(0, 0, radius)
    """
        ТУТ КОД ДЛЯ СОЗДАНИЯ ОКРУЖНОСТИ В 2Д с заданной толщиной  
    """
    UVpoints = []

    square_r_max = (radius + thickness + delta)**2
    square_r_min = (max(radius - thickness - 0.95 + delta, 0))**2

    thickness_radius = thickness + radius

    for v in np.arange(-thickness_radius, thickness_radius+1, 1/scale):
        square_len_v = v**2
        for u in np.arange(-thickness_radius, thickness_radius+1, 1/scale):
            square_len_vu = square_len_v + u**2
            if square_r_min <= square_len_vu <= square_r_max:
                UVpoints.append((u, v))

    draw_2d_points_in_3d(data, UVpoints, point, direction, color, scale)


def fill_small_cylinder(data, point, direction, radius, color, delta = 0.2, scale=2):
    """
    Функция рисует оболочку на расстоянии radius от вектора direction с началом в point при помощи параллельных линий.

    Args:
        data:       - Трехмерный массив данных для рисования
        point:      - Начальная точка цилиндра
        direction:  - Вектор направления, также задающий и длину
        radius:     - Радиус цилиндра
        color:      - Цвет для оболочки
    """

    UVpoints = []
    square_r_max = (radius + delta)**2

    for v in np.arange(-radius, radius+1, 1/scale):
        square_len_v = v**2
        for u in np.arange(-radius, radius+1, 1/scale):
            square_len_vu = square_len_v + u**2
            if square_len_vu <= square_r_max:
                UVpoints.append((u, v))

    if len(UVpoints) == 0:
        logger.warning(f"CYLINDER {point}, {point+direction}, {direction}, {radius}, НЕТ ТОЧЕК !")
    else:
        draw_2d_points_in_3d(data, UVpoints, point, direction, color, scale)


def draw_small_cylinder_with_filling(data, point, direction, radius, color_outer, color_inner=None, thickness=0, delta=0.2, scale=2):
    if color_inner is None:
        color_inner = color_outer

    if radius-thickness > 0:
        fill_small_cylinder(data, point, direction, radius-thickness, color_inner, delta, scale)
    draw_small_cylinder(data, point, direction,radius,color_outer, thickness, delta, scale)


def fill_small_capsule(data, point, direction, radius, color, delta = 0.2, scale=2):
    fill_small_sphere(data, point, radius, color, delta)
    fill_small_sphere(data, point+direction, radius, color, delta)
    fill_small_cylinder(data, point, direction, radius, color, delta, scale)


def draw_small_capsule_with_filling(data, point, direction, radius, color_outer, color_inner=None, thickness=0, delta=0.2, scale=2):
    if color_inner is None:
        color_inner = color_outer

    if thickness > 0:
        fill_small_capsule(data, point, direction, radius, color_outer, delta, scale)
    if radius-thickness > 0:
        fill_small_capsule(data, point, direction, radius-thickness, color_inner, delta, scale)
