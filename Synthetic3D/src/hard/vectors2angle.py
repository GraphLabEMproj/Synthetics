import numpy as np
import math
from Synthetic3D.src.hard.structure.vector import Vector

def vector_in_degrees(vector:Vector):
    # Вычисляем длину вектора
    length = np.linalg.norm(vector)
    if length == 0:
        return (0, 0, 0)

    # Нормализуем вектор
    nx, ny, nz = vector/length

    # Вычисляем углы относительно каждой оси
    # Угол между вектором и каждой осью можно определить как arccos компоненты
    angle_x = math.acos(nx)  # угол относительно оси X в радианах
    angle_y = math.acos(ny)  # угол относительно оси Y в радианах
    angle_z = math.acos(nz)  # угол относительно оси Z в радианах

    # Переводим в градусы для удобства
    angle_x_deg = math.degrees(angle_x)
    angle_y_deg = math.degrees(angle_y)
    angle_z_deg = math.degrees(angle_z)

    return Vector(angle_x_deg, angle_y_deg, angle_z_deg)

########################################################################################################################## изучить теорию поворота углов !
def vector2angles_in_degrees(current_vector:Vector, target_vector:Vector):
    """
    Вычисляет углы поворота вокруг осей X, Y, Z в градусах,
    чтобы повернуть current_vector в direction target_vector.

    Args:
        current_vector (list или np.ndarray): исходный вектор [x, y, z].
        target_vector (list или np.ndarray): целевой вектор [x, y, z].

    Returns:
        Vector: (angle_x, angle_y, angle_z) — углы в градусах.
    """

    # Нормализуем векторы
    v1 = current_vector/np.linalg.norm(current_vector)
    v2 = target_vector/np.linalg.norm(target_vector)

    # Вычисляем ось вращения (вектор оси)
    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)

    # Если векторы совпадают или противоположны
    if axis_norm < 1e-8:
        # Вектора параллельны или противоположны
        dot = np.dot(v1, v2)
        if dot > 0:
            # Вектора совпадают, угол 0
            return 0.0, 0.0, 0.0
        else:
            # Вектора противоположны, поворот на 180 градусов вокруг любой ортогональной оси
            # Здесь выбираем произвольную ортогональную ось
            orthogonal = np.array([1, 0, 0]) if not np.allclose(v1, [1, 0, 0]) else np.array([0, 1, 0])
            axis = np.cross(v1, orthogonal)
            axis /= np.linalg.norm(axis)
            angle = np.pi
    else:
        axis /= axis_norm
        # Вычисляем угол между векторами
        dot = np.dot(v1, v2)
        dot = np.clip(dot, -1.0, 1.0)
        angle = np.arccos(dot)

    # Создаем кватерон для поворота
    half_angle = angle / 2.0
    sin_half = np.sin(half_angle)
    q = np.array([
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half,
        np.cos(half_angle)
    ])

    # Преобразуем кватернион в углы Эйлера (порядок ZYX)
    def quaternion_to_euler(q):
        x, y, z, w = q
        # Углы вокруг осей
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        angle_x = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        angle_y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        angle_z = np.arctan2(t3, t4)

        return angle_x, angle_y, angle_z

    angles_rad = quaternion_to_euler(q)
    # Переводим радианы в градусы
    return Vector([np.degrees(angle) for angle in angles_rad])

def vector2angles_in_quaterion(current_vector:Vector, target_vector:Vector):
    """
    Вычисляет углы поворота вокруг осей X, Y, Z в градусах,
    чтобы повернуть current_vector в direction target_vector.

    Args:
        current_vector (list или np.ndarray): исходный вектор [x, y, z].
        target_vector (list или np.ndarray): целевой вектор [x, y, z].

    Returns:
        Vector: (angle_x, angle_y, angle_z) — углы в градусах.
    """

    # Нормализуем векторы
    v1 = current_vector/np.linalg.norm(current_vector)
    v2 = target_vector/np.linalg.norm(target_vector)

    # Вычисляем ось вращения (вектор оси)
    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)

    # Если векторы совпадают или противоположны
    if axis_norm < 1e-8:
        # Вектора параллельны или противоположны
        dot = np.dot(v1, v2)
        if dot > 0:
            # Вектора совпадают, угол 0
            return np.array([0.0, 0.0, 0.0, 1.0])
        else:
            # Вектора противоположны, поворот на 180 градусов вокруг любой ортогональной оси
            # Здесь выбираем произвольную ортогональную ось
            orthogonal = np.array([1, 0, 0]) if not np.allclose(v1, [1, 0, 0]) else np.array([0, 1, 0])
            axis = np.cross(v1, orthogonal)
            axis /= np.linalg.norm(axis)
            angle = np.pi
    else:
        axis /= axis_norm
        # Вычисляем угол между векторами
        dot = np.dot(v1, v2)
        dot = np.clip(dot, -1.0, 1.0)
        angle = np.arccos(dot)

    # Создаем кватерон для поворота
    half_angle = angle / 2.0
    sin_half = np.sin(half_angle)
    q = np.array([
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half,
        np.cos(half_angle)
    ])

    return q
