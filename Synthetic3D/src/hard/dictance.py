import numpy as np
from Synthetic3D.src.hard.structure.vector import Vector

def distance_point_to_segment(point, seg_start, seg_end):
    # Вектор от начала сегмента к точке
    v = point - seg_start
    # Вектор от начала сегмента к конечной точке
    s = seg_end - seg_start
    # Параметр проекции точки на линию сегмента
    t = np.dot(v, s) / np.dot(s, s)
    # Ограничиваем t в диапазоне [0, 1], чтобы получить точку на сегменте
    t = max(0, min(1, t))
    # Находим ближайшую точку на сегменте
    closest_point = seg_start + t * s
    # Расстояние между точкой и этой точкой
    return np.linalg.norm(point - closest_point)

def min_distance_between_segments(a1: Vector|np.ndarray,
                                  a2: Vector|np.ndarray,
                                  b1: Vector|np.ndarray,
                                  b2: Vector|np.ndarray):

    """
    Находит минимальное расстояние между двумя отрезками в 3D пространстве.
    a1, a2, b1, b2 - np.array или списки координат [x, y, z]
    """

    # Направляющие векторы
    v1 = a2 - a1
    v2 = b2 - b1
    r = a1 - b1

    #  возвращает скалярное произведение
    a = np.dot(v1, v1)
    b = np.dot(v2, v2)

    epsilon = 1e-10  # чтобы избежать деления на ноль

    if a <= epsilon and b <= epsilon:
        # Оба сегмента — точки
        return np.linalg.norm(a1 - b1)
    if a <= epsilon:
        # Первый сегмент — точка
        return distance_point_to_segment(a1, b1, b2)
    if b <= epsilon:
        # Второй сегмент — точка
        return distance_point_to_segment(b1, a1, a2)

    c = np.dot(v1, r)
    d = np.dot(v1, v2)
    denom = a * b - d * d
    f = np.dot(v2, r)

    if abs(denom) > epsilon:
        s = (d * f - c * b) / denom
        t = (a * f - d * c) / denom
        s = max(0, min(1, s))
        t = max(0, min(1, t))

        closest_point_seg1 = a1 + s * v1
        closest_point_seg2 = b1 + t * v2
        return np.linalg.norm(closest_point_seg1 - closest_point_seg2)
    else:
        # Отрезки параллельны
        t = f / b
        t = max(0, min(1, t))

        closest_point_seg2 = b1 + t * v2
        closest_point_seg1_1 = a1         # s1 = 0
        closest_point_seg1_2 = a1 + v1    # s2 = 1

        dist1 = np.linalg.norm(closest_point_seg1_1 - closest_point_seg2)
        dist2 = np.linalg.norm(closest_point_seg1_2 - closest_point_seg2)

        if dist1 < dist2:
            return dist1
        else:
            return dist2
