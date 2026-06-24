import numpy as np
from Synthetic3D.src.hard.random_params import get_rand_int
from Synthetic3D.src.utilities.logging_config import logger

from Synthetic3D.src.hard.vector_operation import create_new_2d_plane_by_point_and_normal, \
            plane_coord_to_3d

from Synthetic3D.src.hard.dictance import min_distance_between_segments

from collections import defaultdict

# ------------------- Пространственный индекс для отрезков крист -------------------
class CristaeSegmentIndex:
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.grid = defaultdict(list)  # ключ: (ix, iy, iz) -> список отрезков

    def _cell_key(self, point):
        return tuple(int(coord // self.cell_size) for coord in point)

    def add_segment(self, cristae_id, seg_idx, p_start, p_end, radius):
        """Добавляет отрезок в ячейки, пересекающие его bounding box, расширенный на radius."""
        mins = np.minimum(p_start, p_end) - radius
        maxs = np.maximum(p_start, p_end) + radius
        min_cell = self._cell_key(mins)
        max_cell = self._cell_key(maxs)
        for ix in range(min_cell[0], max_cell[0] + 1):
            for iy in range(min_cell[1], max_cell[1] + 1):
                for iz in range(min_cell[2], max_cell[2] + 1):
                    self.grid[(ix, iy, iz)].append(
                        (cristae_id, seg_idx, np.array(p_start), np.array(p_end), radius)
                    )

    def query_near_segment(self, p_start, p_end, search_radius, exclude_cristae_id=None):
        """Возвращает кандидаты-отрезки, чьи AABB пересекаются с областью поиска."""
        mins = np.minimum(p_start, p_end) - search_radius
        maxs = np.maximum(p_start, p_end) + search_radius
        min_cell = self._cell_key(mins)
        max_cell = self._cell_key(maxs)
        candidates = []
        for ix in range(min_cell[0], max_cell[0] + 1):
            for iy in range(min_cell[1], max_cell[1] + 1):
                for iz in range(min_cell[2], max_cell[2] + 1):
                    for item in self.grid.get((ix, iy, iz), []):
                        if exclude_cristae_id is not None and item[0] == exclude_cristae_id:
                            continue
                        candidates.append(item)
        # Убираем дубликаты (один отрезок мог попасть в несколько ячеек)
        unique = {}
        for c in candidates:
            uid = (c[0], c[1])
            if uid not in unique:
                unique[uid] = c
        return list(unique.values())

# ================================================================
# Пространственный индекс для точек (поиск ближайшей)
# ================================================================
class PointGrid:
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.grid = defaultdict(list)

    def _cell_key(self, point):
        return tuple(int(coord // self.cell_size) for coord in point)

    def add_point(self, point):
        self.grid[self._cell_key(point)].append(np.array(point))

    def query_nearest(self, point, max_search_radius):
        """Возвращает ближайшую точку в пределах max_search_radius или None."""
        center = np.array(point)
        min_cell = self._cell_key(center - max_search_radius)
        max_cell = self._cell_key(center + max_search_radius)
        best_dist = float('inf')
        best_point = None
        for ix in range(min_cell[0], max_cell[0] + 1):
            for iy in range(min_cell[1], max_cell[1] + 1):
                for iz in range(min_cell[2], max_cell[2] + 1):
                    for p in self.grid.get((ix, iy, iz), []):
                        d = np.linalg.norm(p - center)
                        if d < best_dist:
                            best_dist = d
                            best_point = p
        return best_point


def generate_point_in_circle(max_radius):
    r = max_radius * np.sqrt(np.random.rand())
    theta = np.random.uniform(0, 2*np.pi)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def is_intersection_with_added_points(x, y,
                                      first_list_on_plane,
                                      first_list_of_radiuse_cristae,
                                      new_radius,
                                      cristae_gap):

    for point_2d, radius in zip(first_list_on_plane, first_list_of_radiuse_cristae):
        if ((point_2d[0] - x) ** 2 + (point_2d[1] - y) ** 2) < (radius + new_radius + cristae_gap)**2:
            return True
    return False

def rotate_vector_random_direction(vector, angle_degrees, attempt_expansion=0):
    """
    Поворачивает вектор на заданный угол в случайном направлении.
    Возвращает повернутый вектор.
    """
    v = np.array(vector)

    # Выбираем сторону отклонения: +1 или -1
    direction_sign = np.random.choice([1, -1])

    expand_old_attempt_angle = angle_degrees
    if attempt_expansion > 0.5: # Добавка от angle_degrees до 90 градусов после половины попыток
        expand_old_attempt_angle += int((90-angle_degrees) * (attempt_expansion-0.5) * 2)

    angle_radians = np.radians(expand_old_attempt_angle * direction_sign)

    # Находим случайную ось, перпендикулярную текущему вектору
    rand_vec = np.random.randn(3)
    axis = np.cross(v, rand_vec)
    while np.linalg.norm(axis) == 0:
        rand_vec = np.random.randn(3)
        axis = np.cross(v, rand_vec)

    axis /= np.linalg.norm(axis)

    # Поворот по формуле Родрига
    v_rot = (v * np.cos(angle_radians) +
             np.cross(axis, v) * np.sin(angle_radians) +
             axis * np.dot(axis, v) * (1 - np.cos(angle_radians)))

    return v_rot

def transform_deviated_vector(prev_frame_dir, new_frame_dir, prev_deviated_vector):
    """
    Поворачивает прошлый отклонённый вектор так, чтобы он соответствовал новому положению фрейма.
    """
    if np.allclose(prev_frame_dir, new_frame_dir, atol=1e-08):
        return prev_deviated_vector.copy()

    prev_dir = np.array(prev_frame_dir)
    new_dir = np.array(new_frame_dir)
    deviated_vec = np.array(prev_deviated_vector)

    # Нормализуем направления
    prev_dir = prev_dir/np.linalg.norm(prev_dir)
    new_dir = new_dir/np.linalg.norm(new_dir)

    # Создаем систему координат для старого направления
    # и для нового направления
    # Построим базис: вектор направления + два перпендикуляра
    # Для этого возьмем произвольный вектор, не параллельный
    def create_basis(forward):
        # Создаем базис
        z = forward
        # выбираем произвольный вектор, не параллельный z
        rand_vec = np.array([1, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1, 0])
        x = np.cross(rand_vec, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        return np.vstack([x, y, z])

    basis_prev = create_basis(prev_dir)
    basis_new = create_basis(new_dir)

    # Проецируем отклонённый вектор в базисе старого фрейма
    coeffs = np.linalg.lstsq(basis_prev.T, deviated_vec, rcond=None)[0]

    # Восстанавливаем вектор в новом базисе
    new_deviated_vec = basis_new.T @ coeffs

    return new_deviated_vec

def filter_point_indices_by_side(points_list, plane_normal, intersection_point):
    """
    Возвращает индексы точек, находящихся с одной стороны от второй плоскости.

    :param points_list: список точек, каждая точка — список или кортеж из 3 чисел [x, y, z]
    :param plane_normal: список или кортеж из 3 чисел — нормаль второй плоскости
    :param intersection_point: список или кортеж из 3 чисел — точка пересечения двух плоскостей
    :return: список индексов
    """
    points = np.array(points_list)
    n = np.array(plane_normal)
    P0 = np.array(intersection_point)

    vectors = points - P0
    dot_products = np.dot(vectors, n)

    # Находим индексы точек с положительным скалярным произведением
    indices = np.where(dot_products <= 0)[0]
    return indices

def is_point_inside_cylinder(point, base_cylinder, direction, radius):
    """
        Возвращает код положения точки относительно цилиндра:
        0 - внутри цилиндра
        1 - над верхним основанием
        2 - под нижним основанием
        3 - за боковой стороной (снаружи)
    """

    axis_length = np.linalg.norm(direction)
    V_hat = direction / axis_length

    # Проекция точки на ось
    t = np.dot(point - base_cylinder, V_hat)

    # Проекция на ось для определения положения
    projection = base_cylinder + t * V_hat

    # Расстояние от точки до оси
    d = np.linalg.norm(point - projection)

    # Проверка, внутри ли радиуса
    if d <= radius:
        if t < 0:
            return 2  # под нижним основанием
        elif t > axis_length:
            return 1  # над верхним основанием
        else:
            return 0  # внутри цилиндра
    else:
        return 3  # за боковой стороной

def get_sequences(point_to_check_index, max_length):
    sequences = []
    n = len(point_to_check_index)
    i = 0

    while i < n:
        start_idx = point_to_check_index[i]

        # Начинаем формировать диапазон
        start = start_idx
        end = start_idx

        # Расширяем влево
        left_neighbor = start - 1
        while left_neighbor >= 0 and left_neighbor in point_to_check_index:
            start = left_neighbor
            i += 1
            left_neighbor -= 1

        # Продолжаем расширять вправо
        right_neighbor = end + 1
        while right_neighbor < max_length and right_neighbor in point_to_check_index:
            end = right_neighbor
            i += 1
            right_neighbor += 1

        sequences.append(list(range(start, end + 1)))
        i += 1  # переходим к следующему неподтвержденному элемент

    return sequences

# ================================================================
# Быстрая проверка пересечений через индекс отрезков
# ================================================================
def is_intersection_with_last_cristae_fast(point_last, point_new, cristae_id, cristae_radius,
                                           cristae_gap, segment_index, max_other_radius):
    search_margin = cristae_radius + max_other_radius + cristae_gap
    candidates = segment_index.query_near_segment(
        point_last, point_new, search_margin, exclude_cristae_id=cristae_id
    )
    for cand in candidates:
        _, _, seg_start, seg_end, other_radius = cand
        if min_distance_between_segments(point_last, point_new, seg_start, seg_end) < \
                cristae_radius + other_radius + cristae_gap:
            return True
    return False

def return_point_on_cylinder(point_last, point_new, cylinder_radius, base_center_cylinder_point, presections_dir):
    p1 = point_last
    p2 = point_new
    c = base_center_cylinder_point
    d = presections_dir

    v = p2 - p1
    d_norm = d / np.linalg.norm(d)

    v_cross_d = v - np.dot(v, d_norm) * d_norm
    p1_cross_d = p1 - c - np.dot(p1 - c, d_norm) * d_norm

    a = np.dot(v_cross_d, v_cross_d)
    b = 2 * np.dot(v_cross_d, p1_cross_d)
    c_coef = np.dot(p1_cross_d, p1_cross_d) - cylinder_radius**2

    discriminant = b**2 - 4 * a * c_coef

    t_side = None
    if abs(a) > 1e-8 and discriminant >= 0:
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b + sqrt_discriminant) / (2 * a)
        t2 = (-b - sqrt_discriminant) / (2 * a)
        t_candidates = [t for t in [t1, t2] if t >= 0]
        if t_candidates:
            t_side = min(t_candidates)

    # Проверка пересечения с нижним основанием
    t_base = None
    denom_base = np.dot(v, -d_norm)
    if abs(denom_base) > 1e-8:
        t_base_candidate = np.dot(c - p1, -d_norm) / denom_base
        if t_base_candidate >= 0:
            point_base = p1 + t_base_candidate * v
            vec_in_base = point_base - c
            vec_in_base_proj = vec_in_base - np.dot(vec_in_base, d_norm) * d_norm
            if np.linalg.norm(vec_in_base_proj) <= cylinder_radius + 1e-8:
                t_base = t_base_candidate

    # Проверка пересечения с верхним основанием
    t_top = None
    top_center = c + d
    denom_top = np.dot(v, -d_norm)
    if abs(denom_top) > 1e-8:
        t_top_candidate = np.dot(top_center - p1, -d_norm) / denom_top
        if t_top_candidate >= 0:
            point_top = p1 + t_top_candidate * v
            vec_in_top = point_top - top_center
            vec_in_top_proj = vec_in_top - np.dot(vec_in_top, d_norm) * d_norm
            if np.linalg.norm(vec_in_top_proj) <= cylinder_radius + 1e-8:
                t_top = t_top_candidate

    # Собираем все возможные пересечения
    candidates = []
    if t_side is not None:
        candidates.append((t_side, 3))
    if t_base is not None:
        candidates.append((t_base, 1))
    if t_top is not None:
        candidates.append((t_top, 2))

    if not candidates:
        # Нет пересечения
        return None, None

    # Выбираем минимальное t
    t_min, hit_type = min(candidates, key=lambda x: x[0])
    intersection_point = p1 + t_min * v
    return intersection_point, hit_type

# ================================================================
# Продолжение кристы внутри цилиндра (исправленная версия)
# ================================================================
def continue_cristae_while_in_cylinder(cristae_frame_list, cristae_radius, target_cristae_dir,
                                       cristae_step, cristae_angle_deviation,
                                       max_number_of_attempt_continue, cristae_id,
                                       segment_index, max_other_radius, cristae_gap,
                                       cylinder_radius, base_center_cylinder_point, presections_dir):
    now_cristae_pos = cristae_frame_list[-1]

    new_cristae_dir = target_cristae_dir
    len_new = np.linalg.norm(new_cristae_dir)
    new_cristae_dir = new_cristae_dir / len_new * cristae_step
    new_cristae_pos = now_cristae_pos + new_cristae_dir

    intersection_status = is_intersection_with_last_cristae_fast(
        now_cristae_pos, new_cristae_pos, cristae_id, cristae_radius,
        cristae_gap, segment_index, max_other_radius
    )

    continue_expansion = True
    brak_cristae = False
    status_intersection_point = None

    while continue_expansion:
        counter_attempt_continue = 0
        while intersection_status and counter_attempt_continue < max_number_of_attempt_continue:
            new_cristae_dir = rotate_vector_random_direction(
                target_cristae_dir, cristae_angle_deviation,
                counter_attempt_continue / max_number_of_attempt_continue
            )
            len_new = np.linalg.norm(new_cristae_dir)
            new_cristae_dir = new_cristae_dir / len_new * cristae_step
            new_cristae_pos = now_cristae_pos + new_cristae_dir

            intersection_status = is_intersection_with_last_cristae_fast(
                now_cristae_pos, new_cristae_pos, cristae_id, cristae_radius,
                cristae_gap, segment_index, max_other_radius
            )
            counter_attempt_continue += 1

        if counter_attempt_continue == max_number_of_attempt_continue:
            continue_expansion = False
            brak_cristae = True
        else:
            status_intersection_point = is_point_inside_cylinder(
                new_cristae_pos, base_center_cylinder_point, presections_dir, cylinder_radius
            )
            if status_intersection_point == 0:
                cristae_frame_list.append(new_cristae_pos)
                now_cristae_pos = new_cristae_pos
                intersection_status = True   # следующий виток попыток
            else:
                continue_expansion = False
                new_cristae_on_cylinder, status_intersection_point = return_point_on_cylinder(
                    now_cristae_pos, new_cristae_pos, cylinder_radius,
                    base_center_cylinder_point, presections_dir
                )
                cristae_frame_list.append(new_cristae_on_cylinder)

    if brak_cristae:
        return -1
    return status_intersection_point

def generate_point_on_half_cylinder(base_center, axis_vector, radius, direction_vector):
    """
    Генерирует одну точку на боковой поверхности цилиндра, находящуюся только на одной половине,
    определенной путем проекции direction_vector на основание цилиндра.
    """
    axis_vector = np.array(axis_vector)
    direction_vector = np.array(direction_vector)

    # Нормаль оси цилиндра
    axis_norm = axis_vector / np.linalg.norm(axis_vector)

    # Проекция direction_vector на плоскость, ортогональную к оси цилиндра
    proj = direction_vector - np.dot(direction_vector, axis_norm) * axis_norm

    # Если проекция нулевая (direction_vector параллелен оси), то выбрать любой перпендикуляр
    if np.linalg.norm(proj) < 1e-8:
        # выбрать произвольный вектор, перпендикулярный axis_norm
        if abs(axis_norm[0]) < 0.99:
            arbitrary = np.array([1, 0, 0])
        else:
            arbitrary = np.array([0, 1, 0])
        proj = np.cross(axis_norm, arbitrary)

    proj /= np.linalg.norm(proj)  # нормализуем

    # Генерация случайного угла
    theta = np.random.uniform(0, 2 * np.pi)
    # Высота на цилиндре
    h = np.random.uniform(0, np.linalg.norm(axis_vector))

    # Создаем базис для боковой поверхности
    arbitrary_v = np.array([1, 0, 0]) if abs(axis_norm[0]) < 0.99 else np.array([0, 1, 0])
    v1 = np.cross(axis_norm, arbitrary_v)
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(axis_norm, v1)
    v2 /= np.linalg.norm(v2)

    # Генерация точки
    point = base_center + axis_norm * h + radius * (np.cos(theta) * v1 + np.sin(theta) * v2)

    # Проверка, с какой стороны относительно плоскости деления
    vec_to_point = point - base_center
    if np.dot(vec_to_point, proj) < 0:
        return point
    else:
        # Можно повторить генерацию, чтобы получить точку с нужной стороны
        return generate_point_on_half_cylinder(base_center, axis_vector, radius, direction_vector)

def is_intersection_with_3d_new_cristae(new_point,
                                        new_radius,
                                        new_sections_points,
                                        new_sections_radiuses,
                                        cristae_gap):

    for last_point, last_radius in zip(new_sections_points, new_sections_radiuses):
        if np.linalg.norm(last_point-new_point) < new_radius + last_radius + cristae_gap:
            return True
    return False

def delete_small_len_cristae(cristae_list:list, cristae_radius_list:list, minimum_len:float = 2):
    filtred_list = []
    filtred_radius_list = []
    delete_counter = 0
    for i, cristae in enumerate(cristae_list):
        # по коду в них должно быть минимум 2 точки
        # также по коду укороченной может быть только последняя, так как она укорачивается цилиндром.
        if np.linalg.norm(cristae[-1]-cristae[-2]) < minimum_len:
            cristae.pop(-2)

        if len(cristae) > 1:
            filtred_list.append(cristae)
            filtred_radius_list.append(cristae_radius_list[i])
        else:
            delete_counter += 1

    logger.cristae(f"Удалено {delete_counter} крист из {len(cristae_list)} меньше {minimum_len}")
    return filtred_list, filtred_radius_list

def create_tubular_cristae_fast(frames, list_of_section, cristae_step, cristae_radius_param,
                                cristae_gap, cristae_angle_deviation=45, density_cristae=0.5,
                                max_count_added_cristae=1000, max_count_added_cristae_continue=1000,
                                angle_by_frame_dir=45, overlap_radius=5):
    # Параметры радиусов
    if isinstance(cristae_radius_param, (int, np.integer, float, np.floating)):
        max_possible_radius = cristae_radius_param
        mean_cristae_radius = cristae_radius_param
    else:
        max_possible_radius = max(cristae_radius_param)
        mean_cristae_radius = np.mean(cristae_radius_param)

    # Размер ячеек для индексов
    cell_size_seg = max_possible_radius * 2 + cristae_gap          # для отрезков
    cell_size_pt  = max_possible_radius * 2 + cristae_gap          # для точек
    segment_index = CristaeSegmentIndex(cell_size_seg)
    point_index   = PointGrid(cell_size_pt)

    next_cristae_id = 0

    list_of_cristae = []
    list_of_radiuse_cristae = []
    to_continue_cristae_step = []
    to_continue_cristae_radiuce_step = []

    start_section = list_of_section[0]
    max_radius = start_section.max_radius - overlap_radius

    # ---------- INIT (генерация точек на первой плоскости) ----------
    count_of_start_cristae = int(round((max_radius / (mean_cristae_radius + cristae_gap/2))**2 * density_cristae))
    logger.cristae(f"\tgen_radius {max_radius} mean_cristae_radius {mean_cristae_radius}")
    logger.cristae(f"\tcount_of_start_cristae {count_of_start_cristae}")

    row_first_step_list_of_cristae = []
    row_first_step_list_of_radiuse_cristae = []
    first_list_on_plane = []

    for _ in range(count_of_start_cristae):
        attempt_count = 0
        x, y = generate_point_in_circle(max_radius)
        use_radius = get_rand_int(cristae_radius_param)
        while is_intersection_with_added_points(x, y, first_list_on_plane,
                                                row_first_step_list_of_radiuse_cristae,
                                                use_radius, cristae_gap) \
                and attempt_count < max_count_added_cristae:
            x, y = generate_point_in_circle(max_radius)
            use_radius = get_rand_int(cristae_radius_param)
            attempt_count += 1
        if attempt_count == max_count_added_cristae:
            logger.cristae("\t\tПропуск кристы при инициализации")
        else:
            first_list_on_plane.append((x, y))
            row_first_step_list_of_radiuse_cristae.append(use_radius)

    presections_dir = frames[1] - frames[0]
    first_frame_len = np.linalg.norm(presections_dir)
    u_axis, v_axis = create_new_2d_plane_by_point_and_normal(presections_dir)
    for point_2d in first_list_on_plane:
        point_3d = plane_coord_to_3d(*point_2d, u_axis, v_axis, frames[0])
        row_first_step_list_of_cristae.append(point_3d)

    target_dir = rotate_vector_random_direction(presections_dir, angle_by_frame_dir)
    filter_indexes = filter_point_indices_by_side(row_first_step_list_of_cristae, target_dir, frames[0])
    first_step_list_of_cristae = [[row_first_step_list_of_cristae[i]] for i in filter_indexes]
    first_step_list_of_radiuse_cristae = [row_first_step_list_of_radiuse_cristae[i] for i in filter_indexes]

    # Добавляем начальные точки всех будущих крист в point_index
    for cr_pos in first_step_list_of_cristae:
        point_index.add_point(cr_pos[0])

    logger.cristae(f"\twork_first_step points {len(first_step_list_of_cristae)}")

    # ---------- INIT TRASE (первый цилиндр) ----------
    overlap_delete_index = 0
    len_of_cristae = len(first_step_list_of_cristae)
    for i in range(len_of_cristae):
        idx = i - overlap_delete_index
        cr_pos = first_step_list_of_cristae[idx]
        cr_r = first_step_list_of_radiuse_cristae[idx]
        cristae_id = next_cristae_id
        next_cristae_id += 1

        end_status = continue_cristae_while_in_cylinder(
            cr_pos, cr_r, target_dir, cristae_step, cristae_angle_deviation,
            max_count_added_cristae_continue,
            cristae_id, segment_index, max_possible_radius, cristae_gap,
            max_radius, frames[0], presections_dir
        )

        if end_status is None or end_status == -1 or len(cr_pos) < 2:
            last_point_i = cr_pos[-1]
            if last_point_i is None:
                first_step_list_of_cristae.pop(idx)
                first_step_list_of_radiuse_cristae.pop(idx)
                overlap_delete_index += 1
            else:
                # Быстрый поиск ближайшей точки через point_index
                nearest_point = point_index.query_nearest(
                    last_point_i,
                    max_search_radius=cristae_step * 2 + max_possible_radius * 2 + cristae_gap
                )
                if nearest_point is not None:
                    cr_pos.append(nearest_point)
                    list_of_cristae.append(cr_pos)
                    list_of_radiuse_cristae.append(cr_r)
                    # Регистрируем отрезки и все точки в индексах
                    for seg_idx in range(len(cr_pos) - 1):
                        segment_index.add_segment(cristae_id, seg_idx,
                                                  cr_pos[seg_idx], cr_pos[seg_idx + 1], cr_r)
                    for pt in cr_pos:
                        point_index.add_point(pt)
        elif end_status == 2:
            to_continue_cristae_step.append(cr_pos)
            to_continue_cristae_radiuce_step.append(cr_r)
            # Добавляем все точки кристы в point_index
            for pt in cr_pos:
                point_index.add_point(pt)
        else:
            list_of_cristae.append(cr_pos)
            list_of_radiuse_cristae.append(cr_r)
            for seg_idx in range(len(cr_pos) - 1):
                segment_index.add_segment(cristae_id, seg_idx,
                                          cr_pos[seg_idx], cr_pos[seg_idx + 1], cr_r)
            for pt in cr_pos:
                point_index.add_point(pt)

    logger.cristae(f"\t\tЗавершённых: {len(list_of_cristae)}, "
                   f"в продолжении: {len(to_continue_cristae_step)}, "
                   f"удалено: {overlap_delete_index}")

    # ---------- FIRST SECTION TRASE (боковая поверхность первого цилиндра) ----------
    count_of_section_cristae = int(round(max_radius * first_frame_len /
                                         (mean_cristae_radius + cristae_gap/2)**2 * density_cristae))
    new_sections_points = []
    new_sections_radiuses = []

    for k in range(count_of_section_cristae):
        new_radius = get_rand_int(cristae_radius_param)
        new_point = generate_point_on_half_cylinder(frames[0], presections_dir, max_radius, target_dir)
        attempt_count = 0
        while is_intersection_with_3d_new_cristae(new_point, new_radius,
                                                  new_sections_points, new_sections_radiuses,
                                                  cristae_gap) and attempt_count < max_count_added_cristae:
            new_radius = get_rand_int(cristae_radius_param)
            new_point = generate_point_on_half_cylinder(frames[0], presections_dir, max_radius, target_dir)
            attempt_count += 1
        if attempt_count == max_count_added_cristae:
            logger.cristae(f"\t\tПропуск кристы {k} при инициализации")
        else:
            new_sections_points.append(new_point)
            new_sections_radiuses.append(new_radius)

    section_step_list_of_cristae = [[pt] for pt in new_sections_points]
    section_step_list_of_radiuse_cristae = new_sections_radiuses

    # Добавляем начальные точки новых крист в point_index
    for cr_pos in section_step_list_of_cristae:
        point_index.add_point(cr_pos[0])

    union_cristae_list = list_of_cristae + to_continue_cristae_step + section_step_list_of_cristae
    union_radius_list = list_of_radiuse_cristae + to_continue_cristae_radiuce_step + section_step_list_of_radiuse_cristae
    overlap_section_index = len(list_of_cristae) + len(to_continue_cristae_step)
    overlap_delete_index = 0
    len_of_cristae = len(section_step_list_of_cristae)

    for i in range(len_of_cristae):
        idx = i - overlap_delete_index
        cr_pos = section_step_list_of_cristae[idx]
        cr_r = section_step_list_of_radiuse_cristae[idx]
        cristae_id = next_cristae_id
        next_cristae_id += 1

        end_status = continue_cristae_while_in_cylinder(
            cr_pos, cr_r, target_dir, cristae_step, cristae_angle_deviation,
            max_count_added_cristae_continue,
            cristae_id, segment_index, max_possible_radius, cristae_gap,
            max_radius, frames[0], presections_dir
        )

        if end_status is None or end_status == -1 or len(cr_pos) < 2:
            last_point_i = cr_pos[-1]
            if last_point_i is None:
                section_step_list_of_cristae.pop(idx)
                section_step_list_of_radiuse_cristae.pop(idx)
                union_cristae_list.pop(idx + overlap_section_index)
                union_radius_list.pop(idx + overlap_section_index)
                overlap_delete_index += 1
            else:
                nearest_point = point_index.query_nearest(
                    last_point_i,
                    max_search_radius=cristae_step * 2 + max_possible_radius * 2 + cristae_gap
                )
                if nearest_point is not None:
                    cr_pos.append(nearest_point)
                    list_of_cristae.append(cr_pos)
                    list_of_radiuse_cristae.append(cr_r)
                    for seg_idx in range(len(cr_pos) - 1):
                        segment_index.add_segment(cristae_id, seg_idx,
                                                  cr_pos[seg_idx], cr_pos[seg_idx + 1], cr_r)
                    for pt in cr_pos:
                        point_index.add_point(pt)
        elif end_status == 2:
            to_continue_cristae_step.append(cr_pos)
            to_continue_cristae_radiuce_step.append(cr_r)
            for pt in cr_pos:
                point_index.add_point(pt)
        else:
            list_of_cristae.append(cr_pos)
            list_of_radiuse_cristae.append(cr_r)
            for seg_idx in range(len(cr_pos) - 1):
                segment_index.add_segment(cristae_id, seg_idx,
                                          cr_pos[seg_idx], cr_pos[seg_idx + 1], cr_r)
            for pt in cr_pos:
                point_index.add_point(pt)

    # ---------- SECTION TRASE (последующие секции) ----------
    last_frame_dir = presections_dir
    last_target_dir = target_dir

    for frame_i in range(len(frames) - 2):
        now_frame_i = frame_i + 1
        logger.cristae(f"calculete {now_frame_i} section")
        now_frame_dir = frames[now_frame_i + 1] - frames[now_frame_i]
        now_target_dir = transform_deviated_vector(last_frame_dir, now_frame_dir, last_target_dir)

        section_step_list_of_cristae = to_continue_cristae_step
        section_step_list_of_radiuse_cristae = to_continue_cristae_radiuce_step
        to_continue_cristae_step = []
        to_continue_cristae_radiuce_step = []

        # Добавляем точки from_continue в point_index (они уже могли быть добавлены, но на всякий случай)
        for cr_pos in section_step_list_of_cristae:
            for pt in cr_pos:
                point_index.add_point(pt)

        frame_len = np.linalg.norm(now_frame_dir)
        section_radius = list_of_section[frame_i].max_radius - overlap_radius
        count_of_section_cristae = int(round(section_radius * frame_len /
                                             (mean_cristae_radius + cristae_gap/2)**2 * density_cristae))

        new_sections_points = []
        new_sections_radiuses = []
        for k in range(count_of_section_cristae):
            new_radius = get_rand_int(cristae_radius_param)
            new_point = generate_point_on_half_cylinder(frames[now_frame_i], now_frame_dir,
                                                        section_radius, now_target_dir)
            attempt_count = 0
            while is_intersection_with_3d_new_cristae(new_point, new_radius,
                                                      new_sections_points, new_sections_radiuses,
                                                      cristae_gap) and attempt_count < max_count_added_cristae:
                new_radius = get_rand_int(cristae_radius_param)
                new_point = generate_point_on_half_cylinder(frames[now_frame_i], now_frame_dir,
                                                            section_radius, now_target_dir)
                attempt_count += 1
            if attempt_count == max_count_added_cristae:
                logger.cristae(f"\tПропуск кристы {k} при инициализации")
            else:
                new_sections_points.append(new_point)
                new_sections_radiuses.append(new_radius)

        for pt in new_sections_points:
            section_step_list_of_cristae.append([pt])
            point_index.add_point(pt)   # начальная точка сразу в индексе
        section_step_list_of_radiuse_cristae += new_sections_radiuses

        overlap_delete_index = 0
        len_of_cristae = len(section_step_list_of_cristae)
        for i in range(len_of_cristae):
            idx = i - overlap_delete_index
            cr_pos = section_step_list_of_cristae[idx]
            cr_r = section_step_list_of_radiuse_cristae[idx]
            cristae_id = next_cristae_id
            next_cristae_id += 1

            end_status = continue_cristae_while_in_cylinder(
                cr_pos, cr_r, now_target_dir, cristae_step, cristae_angle_deviation,
                max_count_added_cristae_continue,
                cristae_id, segment_index, max_possible_radius, cristae_gap,
                section_radius, frames[now_frame_i], now_frame_dir
            )

            if end_status is None or end_status == -1 or len(cr_pos) < 2:
                last_point_i = cr_pos[-1]
                if last_point_i is None:
                    section_step_list_of_cristae.pop(idx)
                    section_step_list_of_radiuse_cristae.pop(idx)
                    overlap_delete_index += 1
                else:
                    nearest_point = point_index.query_nearest(
                        last_point_i,
                        max_search_radius=cristae_step * 2 + max_possible_radius * 2 + cristae_gap
                    )
                    if nearest_point is not None:
                        cr_pos.append(nearest_point)
                        list_of_cristae.append(cr_pos)
                        list_of_radiuse_cristae.append(cr_r)
                        for seg_idx in range(len(cr_pos) - 1):
                            segment_index.add_segment(cristae_id, seg_idx,
                                                      cr_pos[seg_idx], cr_pos[seg_idx + 1], cr_r)
                        for pt in cr_pos:
                            point_index.add_point(pt)
            elif end_status == 2:
                to_continue_cristae_step.append(cr_pos)
                to_continue_cristae_radiuce_step.append(cr_r)
                for pt in cr_pos:
                    point_index.add_point(pt)
            else:
                list_of_cristae.append(cr_pos)
                list_of_radiuse_cristae.append(cr_r)
                for seg_idx in range(len(cr_pos) - 1):
                    segment_index.add_segment(cristae_id, seg_idx,
                                              cr_pos[seg_idx], cr_pos[seg_idx + 1], cr_r)
                for pt in cr_pos:
                    point_index.add_point(pt)

        logger.cristae(f"\t\tСекция {now_frame_i}: завершённых {len(list_of_cristae)}, "
                       f"в продолжении {len(to_continue_cristae_step)}")

        last_frame_dir = now_frame_dir
        last_target_dir = now_target_dir

    # Добавляем оставшиеся продолжающиеся кристы
    list_of_cristae += to_continue_cristae_step
    list_of_radiuse_cristae += to_continue_cristae_radiuce_step

    return delete_small_len_cristae(list_of_cristae, list_of_radiuse_cristae, 2)
