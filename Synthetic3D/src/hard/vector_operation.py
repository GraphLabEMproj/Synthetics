import numpy as np
from Synthetic3D.src.hard.structure.vector import Vector


def normalize_vector(v):
    """Нормализует вектор v."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def scale_vector_to_length(vec, value):
    """
    Масштабирует касательную vec так, чтобы ее длина была равна value.
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Вектор не должен быть нулевым!")
    return (vec / norm) * value


def random_dir_change(dir_vector, change_value):
    #dir_vector = np.array(dir_vector, dtype=float)

    def get_orthogonal_component(vector):
        # Генерируем случайный вектор
        random_vec = np.random.randn(3)
        # Проецируем его на v, чтобы получить компонент, параллельную v
        projection = np.dot(random_vec, vector) / np.dot(vector, vector) * vector
        # Получаем ортогональный компонент
        return random_vec - projection

    orthogonal_component = get_orthogonal_component(dir_vector)
    len_orthogonal_component = np.linalg.norm(orthogonal_component)
    while len_orthogonal_component == 0:
        orthogonal_component = get_orthogonal_component(dir_vector)
        len_orthogonal_component = np.linalg.norm(orthogonal_component)
    # Нормализуем ортогональный вектор и масштабируем так, чтобы итоговая длина была равна original_length
    orthogonal_component /= len_orthogonal_component
    # Складываем с исходным вектором
    result = dir_vector + orthogonal_component * change_value  # масштабирование единичного вектора
    # Нормализуем итоговый результат
    result /= np.linalg.norm(result)
    return Vector(result)

def get_pos_and_norm_from_vertex_short_edge(vertex1_1, vertex1_2, vertex2_1, vertex2_2):
    point1_1 = vertex1_1.position
    point1_2 = vertex1_2.position
    point2_1 = vertex2_1.position
    point2_2 = vertex2_2.position

    vector1 = point1_2 - point1_1
    vector2 = point2_2 - point2_1

    if np.linalg.norm(vector1) < np.linalg.norm(vector2):
        return vertex1_1.position, vertex1_1.normal, vertex1_2.position, vertex1_2.normal
    else:
        return vertex2_1.position, vertex2_1.normal, vertex2_2.position, vertex2_2.normal

def create_new_2d_plane_by_point_and_normal(normal):
    # Проверяем, какой вектор выбрать для перекрестного произведения
    # чтобы избежать нулевого или очень малого вектора
    if np.allclose(normal, [0, 0, 1], atol=1e-8):
        u_candidate = np.array([1, 0, 0])
    else:
        u_candidate = np.array([0, 0, 1])  # другой вектор для вычислений

    u = np.cross(normal, u_candidate)
    if np.linalg.norm(u) < 1e-8:
        # В случае, если у нас всё равно ноль, выбираем другой вектор
        u_candidate = np.array([1, 0, 0])
        u = np.cross(normal, u_candidate)

    u = normalize_vector(u)
    v = np.cross(normal, u)
    v = normalize_vector(v)

    return u, v

def plane_coord_to_3d(u_coord, v_coord, u_axis, v_axis, point):
    return Vector(point + u_coord * u_axis + v_coord * v_axis)

def array_plane_coords_to_3d(points_2d_array, point, u_axis, v_axis):
    """
    Преобразует массив 2D-координат points_2d_array (N x 2) в массив 3D-координат.

    Parameters:
    - points_2d_array: numpy массив формы (N, 2)
    - point: точка на плоскости (numpy массив формы (3,))
    - normal: нормаль к плоскости (numpy массив формы (3,))

    Returns:
    - points_3d: numpy массив формы (N, 3)
    """
    # Расширим u и v для broadcasting
    u = u_axis.reshape(1, 3)
    v = v_axis.reshape(1, 3)

    # Расширим points_2d для broadcast
    u_coords = points_2d_array[:, 0].reshape(-1, 1)
    v_coords = points_2d_array[:, 1].reshape(-1, 1)

    # Вычисляем 3D точки
    points_3d = np.array(point) + u_coords * u + v_coords * v
    return points_3d


