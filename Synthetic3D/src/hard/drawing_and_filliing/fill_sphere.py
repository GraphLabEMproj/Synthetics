import numpy as np
from Synthetic3D.src.hard.structure.vector import Vector

import pyvista as pv

def fill_small_sphere(data, position:Vector, radius, color, delta_radius=0.25):
    """
    Для сфер малого радиуса нагрузка на рисование будет небольшая, так что использование неоптимального, но простого
    алгоритма не станет серьёзной проблемой.

    :param data:        Входные трехмерные данные для рисования
    :param position:    Центр сферы
    :param radius:      радиус сферы
    :param color:       цвет рисования оболочки
    :param delta_radius:   ЭКСПЕРИМЕНТАЛЬНОЕ ПОЛЕ ДЛЯ ВЫБОРА РАЗНЫХ СПОСОБОВ ОТСЕИВАНИЯ ЛИШНИХ ПИКСЕЛЕЙ ПРИНАДЛЕЖАВШИХ ОСИ
    :return: list[str]  список варнингов для дебагинга

    Заметка:
    Есть вариант сделать очень быстрое рисование через послойное рисование двумерными окружностями с заполнением сверху
    и снизу, но этот вариант будет предпочтительнее для сфер приличного размера
    """

    # ПРОВЕРКА ВОЗМОЖНОСТИ ПРОСТРАНСТВЕННОГО ПЕРЕМЕЩЕНИЯ И УСТАНОВКИ ЦВЕТА
    assert len(data.shape) > 2
    c = 1 if len(data.shape) == 3 else data.shape[3]
    if isinstance(color, int):
        if c != 1:
            raise Exception(
                f"Размерность цвета не соврадает с количеством каналов изображения ! Color = '{color}', shape data = '{data.shape}'")
    elif len(color) != c:
        raise Exception(
            f"Размерность цвета не соврадает с количеством каналов изображения ! Color = '{color}', shape data = '{data.shape}'")

    assert isinstance(radius, int), "radius can be int!"

    x, y, z = np.round(position).astype(int) # и для векторов и всего итеррируемого
    d, h, w = data.shape[:3]

    warnings_draw = []

    max_radius_compare = (radius+delta_radius)**2

    for k_z in range(-radius, radius+1, 1):
        now_z = k_z + z
        if 0<=now_z<d:
            len_of_now_vector_by_z = k_z ** 2
            for k_x in range(-radius, radius+1, 1):
                now_x = k_x + x
                if 0 <= now_x < w:
                    len_of_now_vector_by_zx = len_of_now_vector_by_z + k_x**2
                    for k_y in range(-radius, radius+1, 1):
                        now_y = k_y + y
                        if 0 <= now_y <h:
                            len_of_now_vector_by_zxy = len_of_now_vector_by_zx + k_y ** 2
                            if len_of_now_vector_by_zxy <= max_radius_compare:
                                data[now_z, now_y, now_x] = color
                        else:
                            warnings_draw.append(f"Выход за пределы поля при рисовании сферы по y. Индекс  {now_y} из [0:{h})")
                else:
                    warnings_draw.append(f"Выход за пределы поля при рисовании сферы по y. Индекс {now_x} из [0:{w})")
        else:
            warnings_draw.append(f"Выход за пределы поля при рисовании сферы по z. Индекс {now_z} из [0:{d})")

    return warnings_draw

def fill_sphere_pyvista(data, center, radius, color):
    """
    Закрашивает сферу (целиком) в трёхмерном массиве data.

    Параметры
    ---------
    data : np.ndarray, форма (D, H, W) или (D, H, W, C)
        Массив вокселей.
    center : tuple (x, y, z)
        Координаты центра сферы (в индексах).
    radius : float
        Радиус сферы (в единицах вокселей).
    color : int или tuple
        Цвет. Если data одноканальная — int, если многоканальная — tuple длины C.
    """
    # Размерности
    if data.ndim == 3:
        channels = 1
    elif data.ndim == 4:
        channels = data.shape[3]
    else:
        raise ValueError("data должна быть 3D или 4D массивом")

    if isinstance(color, int) and channels != 1:
        raise ValueError("Цвет не соответствует числу каналов")
    if not isinstance(color, int) and len(color) != channels:
        raise ValueError("Цвет не соответствует числу каналов")

    D, H, W = data.shape[:3]

    # Создаём сферу-меш
    sphere = pv.Sphere(radius=radius, center=center)

    # Вокселизация с шагом 1 (получаем точки, попадающие внутрь меша)
    # По умолчанию voxelize() вокселизирует только внутренность (solid).
    voxel_grid = sphere.voxelize(spacing=1)

    # Извлекаем индексы вокселей (округляем до целых)
    voxel_points = voxel_grid.points  # (N, 3) float
    indices = np.round(voxel_points).astype(int)

    # Оставляем только те, что попадают в границы data
    valid = (
        (indices[:, 0] >= 0) & (indices[:, 0] < W) &
        (indices[:, 1] >= 0) & (indices[:, 1] < H) &
        (indices[:, 2] >= 0) & (indices[:, 2] < D)
    )
    idx = indices[valid]

    if len(idx) == 0:
        return

    # Присваиваем цвет
    if channels == 1:
        data[idx[:, 2], idx[:, 1], idx[:, 0]] = color
    else:
        data[idx[:, 2], idx[:, 1], idx[:, 0]] = np.array(color, dtype=data.dtype)
