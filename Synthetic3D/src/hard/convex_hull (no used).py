import numpy as np
from scipy.spatial import ConvexHull
from Synthetic3D.src.hard.structure.vector import Vector
from Synthetic3D.src.hard.structure.triangle import Triangle
from Synthetic3D.src.hard.vector_operation import normalize_vector

def convex_hull_3d(points):
    """
        Находит выпуклую оболочку для набора точек.
        Возвращает:
        - список треугольников (каждый — тройка индексов входных точек)
        - список индексов точек, составляющих оболочку
    """

    pts = np.array(points)
    hull = ConvexHull(pts)

    # Индексы точек, входящих в оболочку
    hull_vertex_indices = hull.vertices

    # Грани оболочки — треугольники по индексам
    triangles_indices = []
    for simplex in hull.simplices:
        # simple — это тройка индексов точек, входящих в оболочку
        triangles_indices.append(Triangle(simplex))

    return triangles_indices, hull_vertex_indices

def shell_expansion(poins_list, radius_list, center_shell_point, work_indexes):
    """
        Данная функция возвращает точки оболочки, смещенные от центра на 2 радиуса
    """

    new_convex_list = []

    for index in work_indexes:
        point = poins_list[index]
        radius = radius_list[index]
        normal = normalize_vector(point - center_shell_point)
        new_convex_list.append(Vector(point + normal*2*radius))

    return new_convex_list
