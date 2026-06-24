import math
import numpy as np
from Synthetic3D.src.hard.structure.vector import Vector

# специализированные повороты в 2D

def rotate_2d_by_z(vector, angle):
    rz = math.radians(angle)
    x, y, z = vector
    x_new = x * math.cos(rz) - y * math.sin(rz)
    y_new = x * math.sin(rz) + y * math.cos(rz)
    return Vector(x_new, y_new, z)

# общие повороты в 3D
def rotate_3d(point:Vector, angle:Vector):
    '''
    Поворачивает вестор в 3D пространстве на заданные углы по осям x, y, z.
    :param point: Vector - 3 числа в в (x, y, z)
    :param angle: Vector - угол поворота вокруг осей x, y и z в градусах
    :return: Vector - новая точка после поворота (x, y, z)
    '''

    # Преобразуем градусы в радианы
    rx = math.radians(angle[0])
    ry = math.radians(angle[1])
    rz = math.radians(angle[2])

    x, y, z = point

    # Поворот вокруг оси x
    y1 = y * math.cos(rx) - z * math.sin(rx)
    z1 = y * math.sin(rx) + z * math.cos(rx)
    x1 = x

    # Поворот вокруг оси y
    x2 = x1 * math.cos(ry) + z1 * math.sin(ry)
    z2 = -x1 * math.sin(ry) + z1 * math.cos(ry)
    y2 = y1

    # Поворот вокруг оси z
    x3 = x2 * math.cos(rz) - y2 * math.sin(rz)
    y3 = x2 * math.sin(rz) + y2 * math.cos(rz)
    z3 = z2

    return Vector(x3, y3, z3)

def get_rotate_matrix(angle:Vector):
       # Перевод углов в радианы
       rx = math.radians(angle[0])
       ry = math.radians(angle[1])
       rz = math.radians(angle[2])

       # Матрица поворота вокруг оси X (для вектора-столбца)
       Rx = np.array([
           [1, 0, 0],
           [0, math.cos(rx), -math.sin(rx)],
           [0, math.sin(rx),  math.cos(rx)]
       ])

       # Матрица поворота вокруг оси Y
       Ry = np.array([
           [ math.cos(ry), 0, math.sin(ry)],
           [0,             1, 0],
           [-math.sin(ry), 0, math.cos(ry)]
       ])

       # Матрица поворота вокруг оси Z
       Rz = np.array([
           [ math.cos(rz), -math.sin(rz), 0],
           [ math.sin(rz),  math.cos(rz), 0],
           [0,              0,            1]
       ])

       # Суммарная матрица поворота: сначала Rx, затем Ry, затем Rz
       # Для вектора-столбца v' = Rz * Ry * Rx * v
       R = Rz @ Ry @ Rx
       return R


def rotate_around_point_3d(point:Vector, angle:Vector, turning_pos:Vector = Vector()):
    '''
    Поворачивает вестор в 3D пространстве на заданные углы по осям x, y, z.
    :param point: Vector - 3 числа в в (x, y, z)
    :param angle: Vector - угол поворота вокруг осей x, y и z в градусах
    :return: Vector - новая точка после поворота (x, y, z)
    '''

    # Преобразуем градусы в радианы
    rx = math.radians(angle[0])
    ry = math.radians(angle[1])
    rz = math.radians(angle[2])

    x, y, z = point-turning_pos

    # Поворот вокруг оси x
    y1 = y * math.cos(rx) - z * math.sin(rx)
    z1 = y * math.sin(rx) + z * math.cos(rx)
    x1 = x

    # Поворот вокруг оси y
    x2 = x1 * math.cos(ry) + z1 * math.sin(ry)
    z2 = -x1 * math.sin(ry) + z1 * math.cos(ry)
    y2 = y1

    # Поворот вокруг оси z
    x3 = x2 * math.cos(rz) - y2 * math.sin(rz)
    y3 = x2 * math.sin(rz) + y2 * math.cos(rz)
    z3 = z2

    return Vector(x3, y3, z3) + turning_pos

def quaternion_rotate_3d(point:Vector|np.ndarray, quaternion:np.ndarray):
    """
        Поворачивает точку в 3D пространстве по кватерниону.

        Args:
            point (Vector): точка (x, y, z).
            quaternion (np.ndarray): кватернион (x, y, z, w).

        Returns:
            Vector: повернутая точка (x, y, z).
        """

    # Представляем точку как кватернион с нулевой w-компонентой
    p = np.array([point[0], point[1], point[2], 0.0])

    # Обратный кватернион (сопряжённый, так как кватернион единичный)
    q_conj = np.array([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]])

    # Вращение: q * p * q_conj
    def quat_mult(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        ])

    q_result = quat_mult(quat_mult(quaternion, p), q_conj)

    # Возвращаем только x, y, z компоненты
    return Vector(q_result[:3])
