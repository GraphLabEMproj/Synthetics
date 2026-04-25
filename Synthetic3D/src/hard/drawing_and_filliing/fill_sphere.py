import numpy as np
from Synthetic3D.src.hard.structure.vector import Vector

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
