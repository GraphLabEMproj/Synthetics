import cv2
import numpy as np

from Synthetic3D.src.hard.structure.vertex import Vertex
from Synthetic3D.src.hard.drawing_and_filliing.draw_triangle import draw_voxel_triangle
from Synthetic3D.src.utilities.view_data import view_vtk_3D_data

def rotate_vertices_around_point(vertices, center, axis, angle_degrees):
    """
    Вращает вершины вокруг заданной точки по выбранной оси на указанный угол.

    :param vertices: список вершин (каждая — список координат)
    :param center: точка центра вращения (список или массив координат)
    :param axis: ось вращения ('x', 'y' или 'z')
    :param angle_degrees: угол поворота в градусах
    :return: список повернутых вершин
    """
    center = np.array(center)
    # Смещаем вершины так, чтобы центр был в начале координат
    shifted_vertices = [np.array(v) - center for v in vertices]

    # Создаем матрицу вращения
    angle_radians = np.radians(angle_degrees)
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians)],
            [0, np.sin(angle_radians), np.cos(angle_radians)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_radians), 0, np.sin(angle_radians)],
            [0, 1, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians)]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Ось должна быть 'x', 'y' или 'z'.")

    # Вращаем каждую вершину
    rotated_vertices = [rotation_matrix @ v for v in shifted_vertices]
    # Возвращаем вершины в изначальную систему координат
    rotated_vertices = [v + center for v in rotated_vertices]
    return [np.round(vertex).astype(int).tolist() for vertex in rotated_vertices]


def generate_integer_tetrahedron_vertices(center, radius, scale_factor=1):
    """
    Генерирует вершины правильного тетраэдра с центром в 'center' и радиусом 'radius'.
    Вершины возвращаются как целочисленные координаты, масштабированные на 'scale_factor'.
    После генерации вершины будут повернуты так, чтобы одна грань была параллельна плоскости XY.
    """
    # Исходные вершины правильного тетраэдра
    vertices_directions = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ]) / np.sqrt(3)  # нормализация

    # Выбираем грань, которую нужно сделать параллельной XY
    face_vertices = vertices_directions[:3]

    # Находим два вектора на грани
    v1 = face_vertices[1] - face_vertices[0]
    v2 = face_vertices[2] - face_vertices[0]

    # Находим нормаль грани
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)  # нормализация

    # Цель: повернуть normal так, чтобы он стал параллелен оси Z
    target_normal = np.array([0, 0, 1])

    # Ось и угол вращения
    rotation_axis = np.cross(normal, target_normal)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    if rotation_axis_norm < 1e-8:
        # Нормаль уже параллельна целевой оси
        rotation_matrix = np.eye(3)
    else:
        rotation_axis /= rotation_axis_norm
        angle = np.arccos(np.clip(np.dot(normal, target_normal), -1.0, 1.0))
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    # Применяем вращение к всем вершинам
    vertices_directions = vertices_directions @ rotation_matrix.T

    # Масштабируем и смещаем вершины
    vertices_scaled = vertices_directions * radius * scale_factor
    vertices = np.array(center) + vertices_scaled

    # Округляем до целых чисел
    return [np.round(vertex).astype(int).tolist() for vertex in vertices]


def create_cube(center, size):
    """
    Создает вершины куба и его грани.

    :param center: центр куба (список или массив координат)
    :param size: длина ребра куба
    :return: список вершин, список граней (каждая грань — список индексов вершин)
    """
    c = np.array(center)
    s = size / 2

    # Вершины куба
    vertices = [
        tuple(np.round(c + np.array([x, y, z])).astype(int).tolist()) for x in [-s, s] for y in [-s, s] for z in [-s, s]
    ]

    # Грани куба (по 4 вершины каждая, индексы вершин)
    faces = [
        [0, 1, 3, 2],  # Нижняя
        [4, 5, 7, 6],  # Верхняя
        [0, 1, 5, 4],  # Передняя
        [2, 3, 7, 6],  # Задняя
        [0, 2, 6, 4],  # Левая
        [1, 3, 7, 5],  # Правая
    ]

    return vertices, faces


def triangulate_faces(vertices, faces):
    """
    Триангулирует каждую грань квадрата в два треугольника.

    :param vertices: список вершин
    :param faces: список граней (каждая — список индексов вершин)
    :return: список треугольных граней (каждая — список из 3 индексов вершин)
    """
    triangles = []
    for face in faces:
        # Делим четырехугольник на два треугольника
        triangles.append([vertices[face[0]], vertices[face[1]], vertices[face[2]]])
        triangles.append([vertices[face[0]], vertices[face[2]], vertices[face[3]]])
    return triangles

def test_cycle_from_tetraedr_3D():
    color = 255
    thickness = 0

    img_size = 300

    center = (img_size // 2, img_size // 2, img_size // 2)
    print(center)
    half_size_of_tetraedr = 60

    coords_list = generate_integer_tetrahedron_vertices(center, half_size_of_tetraedr)
    #coords_list = rotate_vertices_around_point(coords_list, center, 'z', 30)
    #coords_list = rotate_vertices_around_point(coords_list, center, 'y', 30)

    #coords_list, faces = create_cube(center, 100)

    normal_list = [(vertex[0]-center[0],
                    vertex[1]-center[1],
                    vertex[2]-center[2]) for vertex in coords_list]

    #import math
    #for i in range(len(normal_list)):
    #    normal = normal_list[i]
    #    normal_len = math.sqrt(normal[0]**2 + normal[1] ** 2 + normal[2] ** 2)
    #    normal_list[i] = (normal[0]/normal_len, normal[1]/normal_len, normal[2]/normal_len)


    vertex_list = [Vertex(coords_list[i], normal_list[i]) for i in range(len(coords_list))]

    #'''
    triangle_list = [
        (vertex_list[0], vertex_list[1], vertex_list[2]),
        (vertex_list[0], vertex_list[2], vertex_list[3]),
        (vertex_list[0], vertex_list[1], vertex_list[3]),
        (vertex_list[1], vertex_list[2], vertex_list[3]),
    ]
    #'''

    #triangle_list = triangulate_faces(vertex_list, faces)

    '''
    for i in range(1):
        print(f"{i+1} этап разбиения")
        new_triangle_list = []
        for triangle in triangle_list:
            new_triangle_list += split_triangle(*triangle)
        triangle_list = new_triangle_list
    '''

    print(len(triangle_list))

    data = np.zeros((img_size, img_size*2, img_size*3), dtype=np.uint8)

    resize_coef_for_2D_slice = 4

    for triangle in triangle_list:
        print("v ", triangle[0].position, triangle[1].position, triangle[2].position, "  n ", triangle[0].normal, triangle[1].normal, triangle[2].normal)
        draw_voxel_triangle(data, triangle[0].position, triangle[1].position, triangle[2].position, color, thickness)

    for i in range(0, img_size):
        cv2.imwrite(f"test_gen_data_split/{i}_slice.png",
                    cv2.resize(data[i], (img_size * resize_coef_for_2D_slice, img_size * resize_coef_for_2D_slice),
                               interpolation=cv2.INTER_NEAREST))

    data_color = np.zeros((img_size, img_size*2, img_size*3, 3), dtype=np.uint8)
    data_color[:, :, :, 0] = data
    #data_color[:, :, :, 1] = data
    data_color[:, :, :, 2] = data

    view_vtk_3D_data(data_color)


test_cycle_from_tetraedr_3D()
