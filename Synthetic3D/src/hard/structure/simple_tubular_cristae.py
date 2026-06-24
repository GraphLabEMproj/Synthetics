import numpy as np
from Synthetic3D.src.hard.random_params import get_rand_int
from Synthetic3D.src.utilities.logging_config import logger

from Synthetic3D.src.hard.vector_operation import create_new_2d_plane_by_point_and_normal, \
            plane_coord_to_3d

from Synthetic3D.src.hard.dictance import min_distance_between_segments


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

def is_intersection_with_last_cristae(point_last, point_new, i, i_radius, cristae_step,
                                      list_of_cristae, cristae_radius_list, cristae_gap):

    for j in range(i): # проверяем все предыдущие
        last_cristae = list_of_cristae[j]
        last_cristae_radius = cristae_radius_list[j]
        # проверяем все точки, на расстоянии в 2 шага от добавляемой
        cristae_2_step = 2 * cristae_step + i_radius + last_cristae_radius
        point_to_check_index = []
        for k, point in enumerate(last_cristae):
            if np.linalg.norm(point_new-point) < cristae_2_step:
                point_to_check_index.append(k)
            else:
                continue

        segments_sequences = get_sequences(point_to_check_index, len(last_cristae))
        for segments_indexes in segments_sequences:
            last_cristae_one = last_cristae[segments_indexes[0]]
            for l in range(len(segments_indexes) - 1):
                last_cristae_two = last_cristae[segments_indexes[l+1]]
                if min_distance_between_segments(point_last, point_new, last_cristae_one, last_cristae_two) < \
                                                i_radius + last_cristae_radius + cristae_gap:
                    return True
                last_cristae_one = last_cristae_two
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


def continue_cristae_while_in_cylinder(cristae_frame_list,
                                       cristae_radius,
                                       target_cristae_dir,
                                       cristae_step,
                                       cristae_angle_deviation,
                                       max_number_of_attempt_coutinue,
                                       i,
                                       list_of_cristae,
                                       cristae_radius_list,
                                       cristae_gap,
                                       cylinder_radius,
                                       base_center_cylinder_point,
                                       presections_dir):

    now_cristae_pos = cristae_frame_list[-1]
    #print("now_cristae_pos", now_cristae_pos, type(now_cristae_pos))

    # do {
    new_cristae_dir = target_cristae_dir
    len_of_new_cristae_dir = np.linalg.norm(new_cristae_dir)
    new_cristae_dir = new_cristae_dir / len_of_new_cristae_dir * cristae_step

    new_cristae_pos = now_cristae_pos + new_cristae_dir

    intersection_status = is_intersection_with_last_cristae(now_cristae_pos,
                                                            new_cristae_pos,
                                                            i,
                                                            cristae_radius,
                                                            cristae_step,
                                                            list_of_cristae,
                                                            cristae_radius_list,
                                                            cristae_gap)
    # }

    continue_expantion = True
    brak_cristae = False
    status_intersection_point = None

    while continue_expantion:
        counter_attempt_coutinue = 0
        while intersection_status is True and counter_attempt_coutinue < max_number_of_attempt_coutinue:
            # do in while {
            new_cristae_dir = rotate_vector_random_direction(target_cristae_dir,
                                                             cristae_angle_deviation,
                                                             counter_attempt_coutinue/max_number_of_attempt_coutinue)
            len_of_new_cristae_dir = np.linalg.norm(new_cristae_dir)
            new_cristae_dir = new_cristae_dir / len_of_new_cristae_dir * cristae_step
            new_cristae_pos = now_cristae_pos + new_cristae_dir

            intersection_status = is_intersection_with_last_cristae(now_cristae_pos,
                                                                    new_cristae_pos,
                                                                    i,
                                                                    cristae_radius,
                                                                    cristae_step,
                                                                    list_of_cristae,
                                                                    cristae_radius_list,
                                                                    cristae_gap)
            # }

            counter_attempt_coutinue += 1

        if counter_attempt_coutinue == max_number_of_attempt_coutinue:
            #logger.cristae(f"Не удалось продолжить кристу {i} за {max_number_of_attempt_coutinue} попыток")
            continue_expantion = False
            brak_cristae = True
        else:
            status_intersection_point = is_point_inside_cylinder(new_cristae_pos,
                                                                 base_center_cylinder_point, presections_dir,
                                                                 cylinder_radius)
            if status_intersection_point == 0:
                cristae_frame_list.append(new_cristae_pos)
                now_cristae_pos = new_cristae_pos
                intersection_status=True
            else:
                continue_expantion = False
                new_cristae_on_cylinder, status_intersection_point = return_point_on_cylinder(now_cristae_pos,
                                                                                              new_cristae_pos,
                                                                                              cylinder_radius,
                                                                                              base_center_cylinder_point,
                                                                                              presections_dir)
                cristae_frame_list.append(new_cristae_on_cylinder)

    # в случае ошибок сообщить об удалении из крист
    if brak_cristae:
        return -1
    else:
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

def create_tubular_cristae(frames,
                           list_of_section,
                           cristae_step,
                           cristae_radius_param,
                           cristae_gap,
                           cristae_angle_deviation = 45,
                           density_cristae = 0.5,
                           max_count_added_cristae = 1000,
                           max_count_added_cristae_continue = 1000,
                           angle_by_frame_dir=45,
                           overlap_radius = 5):

    ############### INIT ################################
    list_of_cristae = []
    list_of_radiuse_cristae = []

    start_section = list_of_section[0]
    max_radius = start_section.max_radius - overlap_radius

    if isinstance(cristae_radius_param, (int, np.integer, float, np.floating)):
        mean_cristae_radius = cristae_radius_param
    else:
        mean_cristae_radius = (cristae_radius_param[0] + cristae_radius_param[1]) / 2

    count_of_start_cristae = int(round((max_radius / (mean_cristae_radius+cristae_gap/2))**2 * density_cristae))

    logger.cristae(f"\tgen_radius {max_radius} mean_cristae_radius {mean_cristae_radius}")
    logger.cristae(f"\tcount_of_start_cristae {count_of_start_cristae}")

    row_first_step_list_of_cristae = []
    row_first_step_list_of_radiuse_cristae = []
    first_list_on_plane = []
    for s in range(count_of_start_cristae):
        attempt_count = 0
        # do while
        x, y = generate_point_in_circle(max_radius)
        use_cristae_radius = get_rand_int(cristae_radius_param)

        while (is_intersection_with_added_points(x, y,
                                                 first_list_on_plane,
                                                 row_first_step_list_of_radiuse_cristae,
                                                 use_cristae_radius,
                                                 cristae_gap)) and \
                attempt_count < max_count_added_cristae:

            x, y = generate_point_in_circle(max_radius)
            use_cristae_radius = get_rand_int(cristae_radius_param)

            attempt_count += 1

        if attempt_count == max_count_added_cristae:
            logger.cristae(f"\t\tПропуск кристы {s} при инициализации")
        else:
            first_list_on_plane.append((x, y))
            row_first_step_list_of_radiuse_cristae.append(use_cristae_radius)


    # перевод в 3Д
    presections_dir = frames[1] - frames[0]
    first_frame_len = np.linalg.norm(presections_dir)
    u_axis, v_axis = create_new_2d_plane_by_point_and_normal(presections_dir)
    for point_2d in first_list_on_plane:
        point_3d = plane_coord_to_3d(*point_2d, u_axis, v_axis, frames[0])
        row_first_step_list_of_cristae.append(point_3d)


    ########## INIT FILTER ###############################

    # Отсеить точки, что будут за куполом и часто будут пересекать оболочку несколько раз
    target_dir = rotate_vector_random_direction(presections_dir,
                                                angle_by_frame_dir)
    filter_indexes = filter_point_indices_by_side(row_first_step_list_of_cristae, target_dir, frames[0])
    first_step_list_of_cristae = [[row_first_step_list_of_cristae[i]] for i in filter_indexes]
    first_step_list_of_radiuse_cristae = [row_first_step_list_of_radiuse_cristae[i] for i in filter_indexes]

    #first_step_list_of_cristae = [[point] for point in row_first_step_list_of_cristae]
    #first_step_list_of_radiuse_cristae = row_first_step_list_of_radiuse_cristae

    logger.cristae(f"\twork_first_step points {len(first_step_list_of_cristae)}")

    ########## INIT TRASE ###############################

    #cylinder
    #   max_radius
    #   presections_dir
    #   start_point = frames[0]

    to_continue_cristae_step = []
    to_continue_cristae_radiuce_step = []

    overlap_delete_index = 0
    len_of_cristae = len(first_step_list_of_cristae)
    for i in range(len_of_cristae):
        cr_pos = first_step_list_of_cristae[i-overlap_delete_index]
        cr_r = first_step_list_of_radiuse_cristae[i-overlap_delete_index]

        #target_dir
        end_status = continue_cristae_while_in_cylinder(cr_pos,
                                                        cr_r,
                                                        target_dir,
                                                        cristae_step,
                                                        cristae_angle_deviation,
                                                        max_count_added_cristae_continue,
                                                        i - overlap_delete_index,
                                                        first_step_list_of_cristae,
                                                        first_step_list_of_radiuse_cristae,
                                                        cristae_gap,
                                                        max_radius,
                                                        frames[0],
                                                        presections_dir)

        # Если возникла ошибка, то удаляем из рассмотрения ############### вместо удаление попробовать прицепить к ближайшей
        if end_status is None or end_status == -1 or len(cr_pos) < 2:
            #print("delete_cristae")

            ############################################################## эксперимент
            last_point_i = cr_pos[-1]

            if last_point_i is None: ############################################################# старый блок
                first_step_list_of_cristae.pop(i-overlap_delete_index)
                first_step_list_of_radiuse_cristae.pop(i-overlap_delete_index)
                overlap_delete_index += 1

            else:                   ############################################################ экспериментальный блок
                min_cristae_pos_list = []
                min_cristae_val_list = []
                for j in range(i):
                    min_value = 1000
                    min_point = None
                    cr_pos_j = first_step_list_of_cristae[j]
                    for point_j in cr_pos_j:
                        dist_to_cr = np.linalg.norm(last_point_i - point_j)
                        if dist_to_cr < min_value:
                            min_value = dist_to_cr
                            min_point = point_j
                    if min_point is not None:
                        min_cristae_pos_list.append(min_point)
                        min_cristae_val_list.append(min_value)

                closest_neighbor_crist_pos = min_cristae_pos_list[min_cristae_val_list.index(min(min_cristae_val_list))]
                cr_pos.append(closest_neighbor_crist_pos)

                list_of_cristae.append(cr_pos)
                list_of_radiuse_cristae.append(cr_r)
                ############################################################## эксперимент

        elif end_status == 2:
            to_continue_cristae_step.append(cr_pos)
            to_continue_cristae_radiuce_step.append(cr_r)
        else:
            list_of_cristae.append(cr_pos)
            list_of_radiuse_cristae.append(cr_r)

    if overlap_delete_index != 0:
        logger.cristae(f"\t\tОбработано {len_of_cristae} крист из них удалено {overlap_delete_index} и в продолжении {len(to_continue_cristae_step)}. Завершенных {len(list_of_cristae)}")
    else:
        logger.cristae(
            f"\t\tОбработано {len_of_cristae} крист из них в продолжении {len(to_continue_cristae_step)}. Завершенных {len(list_of_cristae)}")

    ################## FIRST SECTION TRASE ####################################################

    section_step_list_of_cristae = [] # to_continue_cristae_step
    section_step_list_of_radiuse_cristae = [] # to_continue_cristae_radiuce_step

    count_of_section_cristae = int(round(max_radius*first_frame_len / (mean_cristae_radius+cristae_gap/2)**2 * density_cristae))

    new_sections_points = []
    new_sections_radiuses = []

    logger.cristae(f"\twork_first_section_step points {count_of_section_cristae}")

    for k in range(count_of_section_cristae):
        # do {

        new_radius = get_rand_int(cristae_radius_param)
        new_point = generate_point_on_half_cylinder(frames[0],
                                                    presections_dir,
                                                    max_radius,
                                                    target_dir)

        attempt_count = 0
        while is_intersection_with_3d_new_cristae(new_point,
                                                  new_radius,
                                                  new_sections_points,
                                                  new_sections_radiuses,
                                                  cristae_gap) and attempt_count < max_count_added_cristae:

            new_radius = get_rand_int(cristae_radius_param)
            new_point = generate_point_on_half_cylinder(frames[0],
                                                        presections_dir,
                                                        max_radius,
                                                        target_dir)
            attempt_count += 1

        if attempt_count == max_count_added_cristae:
            logger.cristae(f"\t\tПропуск кристы {k} при инициализации")
        else:
            new_sections_points.append(new_point)
            new_sections_radiuses.append(new_radius)

    for point in new_sections_points:
        section_step_list_of_cristae.append([point])
    section_step_list_of_radiuse_cristae += new_sections_radiuses

    overlap_delete_index = 0
    overlap_section_index = len(list_of_cristae) + len(to_continue_cristae_step)

    union_first_cristae_section_list = list_of_cristae + to_continue_cristae_step + section_step_list_of_cristae
    union_first_radius_section_list = list_of_radiuse_cristae + to_continue_cristae_radiuce_step + section_step_list_of_radiuse_cristae

    len_of_cristae = len(section_step_list_of_cristae)
    for i in range(len_of_cristae):
        cr_pos = section_step_list_of_cristae[i - overlap_delete_index]
        cr_r = section_step_list_of_radiuse_cristae[i - overlap_delete_index]

        # target_dir
        end_status = continue_cristae_while_in_cylinder(cr_pos,
                                                        cr_r,
                                                        target_dir,
                                                        cristae_step,
                                                        cristae_angle_deviation,
                                                        max_count_added_cristae_continue,
                                                        i - overlap_delete_index + overlap_section_index,
                                                        union_first_cristae_section_list,
                                                        union_first_radius_section_list,
                                                        cristae_gap,
                                                        max_radius,
                                                        frames[0],
                                                        presections_dir)

        # Если возникла ошибка, то удаляем из рассмотрения ############### вместо удаление попробовать прицепить к ближайшей
        if end_status is None or end_status == -1 or len(cr_pos) < 2:
            # print("delete_cristae")

            ############################################################## эксперимент
            last_point_i = cr_pos[-1]

            if last_point_i is None: ############################################################# старый блок
                section_step_list_of_cristae.pop(i - overlap_delete_index)
                section_step_list_of_radiuse_cristae.pop(i - overlap_delete_index)

                union_first_cristae_section_list.pop(i - overlap_delete_index + overlap_section_index)
                union_first_radius_section_list.pop(i - overlap_delete_index + overlap_section_index)

                overlap_delete_index += 1
            else:                   ############################################################ экспериментальный блок
                min_cristae_pos_list = []
                min_cristae_val_list = []
                for j in range(i - overlap_delete_index + overlap_section_index):
                    min_value = 1000
                    min_point = None
                    cr_pos_j = union_first_cristae_section_list[j]
                    for point_j in cr_pos_j:
                        #print(point_j, last_point_i)
                        dist_to_cr = np.linalg.norm(last_point_i - point_j)
                        if dist_to_cr < min_value:
                            min_value = dist_to_cr
                            min_point = point_j
                    if min_point is not None:
                        min_cristae_pos_list.append(min_point)
                        min_cristae_val_list.append(min_value)

                closest_neighbor_crist_pos = min_cristae_pos_list[min_cristae_val_list.index(min(min_cristae_val_list))]
                cr_pos.append(closest_neighbor_crist_pos)

                list_of_cristae.append(cr_pos)
                list_of_radiuse_cristae.append(cr_r)
                ############################################################## эксперимент

        elif end_status == 2:
            to_continue_cristae_step.append(cr_pos)
            to_continue_cristae_radiuce_step.append(cr_r)
        else:
            list_of_cristae.append(cr_pos)
            list_of_radiuse_cristae.append(cr_r)

    if overlap_delete_index != 0:
        logger.cristae(f"\t\tОбработано {len_of_cristae} крист из них удалено {overlap_delete_index} и в продолжении {len(to_continue_cristae_step)}. Завершенных {len(list_of_cristae)}")
    else:
        logger.cristae(
            f"\t\tОбработано {len_of_cristae} крист из них в продолжении {len(to_continue_cristae_step)}. Завершенных {len(list_of_cristae)}")

    ################## SECTION TRASE ####################################################
    last_frame_dir = presections_dir
    last_target_dir = target_dir
    for frame_i in range(len(frames)-2):
        now_frame_i = frame_i+1
        logger.cristae(f"calculete {now_frame_i} section")
        now_frame_dir = frames[now_frame_i+1] - frames[now_frame_i]
        now_target_dir = transform_deviated_vector(last_frame_dir, now_frame_dir, last_target_dir)

        section_step_list_of_cristae = to_continue_cristae_step
        section_step_list_of_radiuse_cristae = to_continue_cristae_radiuce_step

        to_continue_cristae_step = []
        to_continue_cristae_radiuce_step = []

        frame_len = np.linalg.norm(now_frame_dir)
        section_radius = list_of_section[frame_i].max_radius - overlap_radius
        count_of_section_cristae = int(round(section_radius * frame_len / (mean_cristae_radius+cristae_gap/2) ** 2 * density_cristae))

        new_sections_points = []
        new_sections_radiuses = []

        logger.cristae(f"\twork_{now_frame_i}_section_step points {count_of_section_cristae}")

        for k in range(count_of_section_cristae):
            # do {
            new_radius = get_rand_int(cristae_radius_param)
            new_point = generate_point_on_half_cylinder(frames[now_frame_i],
                                                        now_frame_dir,
                                                        section_radius,
                                                        now_target_dir)

            attempt_count = 0
            while is_intersection_with_3d_new_cristae(new_point,
                                                      new_radius,
                                                      new_sections_points,
                                                      new_sections_radiuses,
                                                      cristae_gap) and attempt_count < max_count_added_cristae:
                new_radius = get_rand_int(cristae_radius_param)
                new_point = generate_point_on_half_cylinder(frames[now_frame_i],
                                                            now_frame_dir,
                                                            section_radius,
                                                            now_target_dir)
                attempt_count += 1

            if attempt_count == max_count_added_cristae:
                logger.cristae(f"\tПропуск кристы {k} при инициализации")
            else:
                new_sections_points.append(new_point)
                new_sections_radiuses.append(new_radius)

        for point in new_sections_points:
            section_step_list_of_cristae.append([point])
        section_step_list_of_radiuse_cristae += new_sections_radiuses

        overlap_delete_index = 0
        len_of_cristae = len(section_step_list_of_cristae)
        for i in range(len_of_cristae):
            cr_pos = section_step_list_of_cristae[i - overlap_delete_index]
            cr_r = section_step_list_of_radiuse_cristae[i - overlap_delete_index]

            # target_dir
            end_status = continue_cristae_while_in_cylinder(cr_pos,
                                                            cr_r,
                                                            now_target_dir,
                                                            cristae_step,
                                                            cristae_angle_deviation,
                                                            max_count_added_cristae_continue,
                                                            i - overlap_delete_index,
                                                            section_step_list_of_cristae,
                                                            section_step_list_of_radiuse_cristae,
                                                            cristae_gap,
                                                            section_radius,
                                                            frames[now_frame_i],
                                                            now_frame_dir)

            # Если возникла ошибка, то удаляем из рассмотрения
            if end_status is None or end_status == -1 or len(cr_pos) < 2:
                # print("delete_cristae")
                ############################################################## эксперимент
                last_point_i = cr_pos[-1]

                if last_point_i is None: ############################################################# старый блок

                    section_step_list_of_cristae.pop(i - overlap_delete_index)
                    section_step_list_of_radiuse_cristae.pop(i - overlap_delete_index)
                    overlap_delete_index += 1

                else:                   ############################################################ экспериментальный блок
                    min_cristae_pos_list = []
                    min_cristae_val_list = []
                    for j in range(i - overlap_delete_index):
                        min_value = 1000
                        min_point = None
                        cr_pos_j = section_step_list_of_cristae[j]
                        for point_j in cr_pos_j:
                            dist_to_cr = np.linalg.norm(last_point_i - point_j)
                            if dist_to_cr < min_value:
                                min_value = dist_to_cr
                                min_point = point_j
                        if min_point is not None:
                            min_cristae_pos_list.append(min_point)
                            min_cristae_val_list.append(min_value)

                    closest_neighbor_crist_pos = min_cristae_pos_list[min_cristae_val_list.index(min(min_cristae_val_list))]
                    cr_pos.append(closest_neighbor_crist_pos)

                    list_of_cristae.append(cr_pos)
                    list_of_radiuse_cristae.append(cr_r)
                    ############################################################## эксперимент

            elif end_status == 2:
                to_continue_cristae_step.append(cr_pos)
                to_continue_cristae_radiuce_step.append(cr_r)
            else:
                list_of_cristae.append(cr_pos)
                list_of_radiuse_cristae.append(cr_r)

        if overlap_delete_index != 0:
            logger.cristae(
                f"\t\tОбработано {len_of_cristae} крист из них удалено {overlap_delete_index} и в продолжении {len(to_continue_cristae_step)}. Завершенных {len(list_of_cristae)}")
        else:
            logger.cristae(
                f"\t\tОбработано {len_of_cristae} крист из них в продолжении {len(to_continue_cristae_step)}. Завершенных {len(list_of_cristae)}")

    ################### END TRASE ########################################

    list_of_cristae += to_continue_cristae_step
    list_of_radiuse_cristae += to_continue_cristae_radiuce_step

    return delete_small_len_cristae(list_of_cristae, list_of_radiuse_cristae, 2)
