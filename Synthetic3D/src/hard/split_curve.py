import math

import numpy as np
from scipy.optimize import least_squares

from Synthetic3D.src.hard.vector_operation import normalize_vector, scale_vector_to_length


######################################################################################################################### переписать модуль !

# old
def ellipse3D_through_points_and_tangents(P1, T1, P2, T2):
    normal = np.cross(T1, T2)
    norm_normal = np.linalg.norm(normal)
    if norm_normal < 1e-8:
        raise ValueError("Касательные слишком параллельны или нулевые.")
    normal /= norm_normal

    v1 = T1 / np.linalg.norm(T1)
    v2 = np.cross(normal, v1)
    v2 /= np.linalg.norm(v2)

    def project_point(P, origin, v1, v2):
        vec = P - origin
        x = np.dot(vec, v1)
        y = np.dot(vec, v2)
        return np.array([x, y])

    origin = P1
    p1_2d = project_point(P1, origin, v1, v2)
    p2_2d = project_point(P2, origin, v1, v2)

    # Начальные параметры: радиусы примерно равны расстоянию между точками
    a0 = np.linalg.norm(p1_2d)
    b0 = np.linalg.norm(p2_2d)
    # Проверка на нулевые касательные
    T1_norm = np.linalg.norm(T1)
    T2_norm = np.linalg.norm(T2)
    if T1_norm < 1e-8 or T2_norm < 1e-8:
        raise ValueError("Касательные слишком малы или нулевые.")
    initial_guess = [a0 if a0 > 0 else 1.0, b0 if b0 > 0 else 1.0, 0, 0, 0]

    bounds = ([1e-6, 1e-6, -np.inf, -np.inf, -np.pi], [np.inf, np.inf, np.inf, np.inf, np.pi])

    def residuals(params):
        a, b, cx, cy, angle = params
        if a <= 0 or b <= 0:
            return np.array([np.inf, np.inf])
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        def rotate_point(x, y):
            return cos_a * x - sin_a * y, sin_a * x + cos_a * y

        def point_residual(P_2d):
            x, y = P_2d
            x_rot, y_rot = rotate_point(x - cx, y - cy)
            return (x_rot / a) ** 2 + (y_rot / b) ** 2 - 1

        def tangent_residual(P_2d, T_3d):
            T_2d = T_3d[:2]
            norm_T2d = np.linalg.norm(T_2d)
            if norm_T2d < 1e-8:
                return 0  # или какое-то значение по умолчанию
            T_2d = T_2d / norm_T2d
            x, y = P_2d
            x_rot, y_rot = rotate_point(x - cx, y - cy)
            tangent_x = x_rot / (a ** 2)
            tangent_y = y_rot / (b ** 2)
            tangent_vec = np.array([tangent_x, tangent_y])
            norm_tangent_vec = np.linalg.norm(tangent_vec)
            if norm_tangent_vec < 1e-8:
                return 0
            tangent_vec /= norm_tangent_vec
            return np.dot(tangent_vec, T_2d)

        res1 = point_residual(p1_2d)
        res2 = point_residual(p2_2d)
        tan_res1 = tangent_residual(p1_2d, T1)
        tan_res2 = tangent_residual(p2_2d, T2)

        return [res1, res2, tan_res1 - 1, tan_res2 - 1]

    result = least_squares(residuals, initial_guess, bounds=bounds)

    a, b, cx, cy, angle = result.x

    def ellipse_point(t):
        x = a * np.cos(t)
        y = b * np.sin(t)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x_rot = cos_a * x - sin_a * y + cx
        y_rot = sin_a * x + cos_a * y + cy
        point_3d = origin + v1 * x_rot + v2 * y_rot
        return point_3d

    t_mid = np.pi / 4
    point_between = ellipse_point(t_mid)
    return point_between



def find_circle_center(P1, T1, P2, T2):
    """
    Находит центр окружности, проходящей через точки P1 и P2
    с касательными T1 и T2 в этих точках.
    """
    # Нормализуем касательные
    T1 = normalize_vector(T1)
    T2 = normalize_vector(T2)

    # Построение системы уравнений для поиска центра
    A = np.array([T1, -T2]).T
    b = P2 - P1

    if np.linalg.matrix_rank(A) < 2:
        print("Касательные параллельны или совпадают, центр не определен однозначно.")
        return None

    lambdas = np.linalg.lstsq(A, b, rcond=None)[0]
    lambda1, lambda2 = lambdas

    # Точки пересечения линий, содержащих касательные
    point_line1 = P1 + lambda1 * T1
    point_line2 = P2 + lambda2 * T2

    # Центр окружности — середина между двумя точками пересечения
    O = (point_line1 + point_line2) / 2

    return O

def compute_radius(P, center):
    """Вычисляет радиус как расстояние между точкой и центром окружности."""
    return np.linalg.norm(P - center)

######################################################################################################################### переписать модуль !
def calculate_orto_vec(vec, normal):
    if all(element == 0 for element in normal):
        print("Zero normal vector detected!")
        return normal

    #vec_len = math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    #vec = vec/vec_len
    # вычисление проекции vec на вектор normal ( = cos(Q) * |vec|)
    #                                                cos(Q) = vec*normal2 / (|vec| * |normal|)
    scalar_vec_normal = vec[0] * normal[0] + vec[1] * normal[1] + vec[2] * normal[2]
    proec_vec_to_normal = (scalar_vec_normal / (normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)) * normal

    # вычисление касательной
    cas_vec = vec - proec_vec_to_normal
    #print(proec_vec_to_normal, normal, vec, scalar_vec_normal, cas_vec)
    #print(f"user check val in fun calculate_orto_vec. cas_vec*proec_vec_to_normal : {(cas_vec * proec_vec_to_normal).sum()}  == 0 ?")

    len_cas_vec = math.sqrt(cas_vec[0] ** 2 + cas_vec[1] ** 2 + cas_vec[2] ** 2)

    # если направление совпадает с нормалью, то и касательная будет нормалью
    if all(element == 0 for element in cas_vec):
        return normal
    else:
        return cas_vec #/len_cas_vec



def vector_sum_with_save_len(vec1, vec2):
    # вычиcлить результирующую длину как среднюю от двух векторов
    len_v1 = np.linalg.norm(vec1)
    len_v2 = np.linalg.norm(vec2)

    len_result = (len_v1+len_v2)/2

    # вычислить направление результирующего вектора
    res_vec = np.array((vec1[0] + vec2[0],
                        vec1[1] + vec2[1],
                        vec1[2] + vec2[2]))

    # привести результирующий вектор к нужной длине
    temp_len_res = np.linalg.norm(res_vec)
    if temp_len_res != 0:
        return res_vec * (len_result/temp_len_res)
    else:
        print("ERROR! Resulting vector has no length!")
        return (0,0,0)

def hermite_point_at_half(p0, p1, m0, m1):
    """
    Вычисляет точку сплайна Эрмита в t=0.5 (середина кривой),
    заданную двумя точками и касательными в них.
    p0, p1: начальная и конечная точки (numpy массивы 3D)
    m0, m1: касательные в точках (numpy массивы 3D)
    """
    t = 0.5
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2

    point = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
    return point

def split_half_curve(point1, normal1, point2, normal2):

    split_normal = vector_sum_with_save_len(normal1, normal2)
    #split_normal = normalize_vector(split_normal)

    if False:# all(np.cross(point1, point2) == 0):
        half_point = (int((point1[0] + point2[0])//2),
                      int((point1[1] + point2[1])//2),
                      int((point1[2] + point2[2])//2))
    else:
        # вычисление вектора, лежащего в плоскости p1 p2 и n2 и ортоганального n2 (считаем касательную в p2)

        cas_v2_v1_and_n1 = calculate_orto_vec(point1-point2, normal1)
        cas_v1_v2_and_n2 = calculate_orto_vec(point2-point1, normal2)

        if all(element == 0 for element in cas_v2_v1_and_n1):
            print(f"ошибка первой касательной {point1}, {point2} и нормаль1 {normal1}")
        if all(element == 0 for element in cas_v1_v2_and_n2):
            print(f"ошибка второй касательной {point2}, {point1} и нормаль2 {normal2}")

        # вычисление вектора, лежащего в плоскости p1 p2 и n1 и ортоганального n1 (считаем касательную в p1)


        center = find_circle_center(point1, -cas_v2_v1_and_n1, point2, cas_v1_v2_and_n2)

        #half_point = ellipse3D_through_points_and_tangents(point1, -cas_v2_v1_and_n1, point2, cas_v1_v2_and_n2)

        if center is not None:
            # Вычисляем радиус, исходя из одной из точек
            radius = compute_radius(point1, center) * 1.5

            # Масштабируем касательные до длины радиуса
            cas_v2_v1_and_n1 = scale_vector_to_length(-cas_v2_v1_and_n1, radius)
            cas_v1_v2_and_n2 = scale_vector_to_length(cas_v1_v2_and_n2, radius)

            half_point = hermite_point_at_half(point1, point2, cas_v2_v1_and_n1, cas_v1_v2_and_n2)
        else:
            # лежат на прямой
            half_point = np.array([(point1[0] + point2[0]) / 2,
                                   (point1[1] + point2[1]) / 2,
                                   (point1[2] + point2[2]) / 2])

        #half_point = tuple(np.round(((point1 + point2)*4 + cas_v2_v1_and_n1 - cas_v1_v2_and_n2)/8).astype(int).tolist())

        #a = 10
        #b = 10
        #half_point = np.round(((point1 + point2)*4 + 3*a*normal1 - 3*b*normal2)/8).astype(int)

        #print(vertex1, vertex2)
        #print(half_point, split_normal)

        #raise Exception("user stop")

        #print(point1, point2, half_point)
        #raise Exception("User ex")

    return half_point, split_normal
