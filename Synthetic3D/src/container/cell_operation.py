import numpy as np
from numba import njit

from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure

from Synthetic3D.src.hard.structure.vector import Vector
from Synthetic3D.src.hard.random_params import get_rand_int

@njit
def ExpansionOfPoint26Dir_v3(data: np.ndarray,
                             position: np.ndarray,
                             index: int,
                             index_kernel: np.ndarray,
                             probability_kernel: np.ndarray,
                             #list_of_boarders: list = []
                             ):
    d, h, w = data.shape
    p_x, p_y, p_z = position
    next_iteration_work_points = []
    repit_check_now_point = False
    check_pos = index_kernel + position
    valid_mask = (
            (0 <= check_pos[:, 0]) & (check_pos[:, 0] < w) &
            (0 <= check_pos[:, 1]) & (check_pos[:, 1] < d) &
            (0 <= check_pos[:, 2]) & (check_pos[:, 2] < h)
    )
    # Отбираем только допустимые индексы
    # work_pos = check_pos[valid_mask]
    # work_probability = probability_kernel[valid_mask]

    for i in range(check_pos.shape[0]):
        if not valid_mask[i]:
            continue
        x, y, z = check_pos[i]
        probability = probability_kernel[i]

        # если не занято, то занимаем с некоторой вероятностью, иначе отправляем на пересмотр
        if data[z, y, x] == 0:
            # 1/3 для углов, 1/2 для середин ребер и 1 для середин граней
            if np.random.random() < probability:
                next_iteration_work_points.append((x, y, z))
                data[z, y, x] = index
            else:
                repit_check_now_point = True

        # попали в соседа self.index или в свою границу
        elif data[z, y, x] == index or data[z, y, x] == -index:
            continue
        # попали в гранчное состояние:
        #   - попали в другой регион
        #   - дотянулись до чужой границы
        # говорим что текущая позиция - граница этой клетки если она уже не обозначена
        else:
            if data[p_z, p_y, p_x] != -index:
                data[p_z, p_y, p_x] = -index
                #list_of_boarders.append((p_z, p_y, p_x))
            else:
                continue

    if repit_check_now_point:
        next_iteration_work_points.append((p_x, p_y, p_z))

    return next_iteration_work_points

def ExpansionOfPoint6Dir(data,
                         position: Vector | np.ndarray,
                         index: int,
                         probability=None,
                         #list_of_boarders: list = []
                         ):
    assert len(data.shape) == 3 or data.shape[3] == 1, "Данные для алгоритма разрастания должны иметь 3 оси и 1 канал."
    d, h, w = data.shape
    p_x, p_y, p_z = position
    next_iteration_work_points = []
    repit_check_now_point = False

    def check_point(x, y, z):
        # если не занято, то занимаем с некоторой вероятностью, иначе отправляем на пересмотр
        if data[z, y, x] == 0:
            if np.random.random() < probability:
                next_iteration_work_points.append(Vector(x, y, z, dtype=int))
                data[z, y, x] = index
                return False
            else:
                return True

        # попали в соседа self.index или в свою границу
        elif data[z, y, x] == index or data[z, y, x] == -index:
            return False
        # попали в гранчное состояние:
        #   - попали в другой регион
        #   - дотянулись до чужой границы
        # говорим что текущая позиция - граница этой клетки если она уже не обозначена
        else:
            if data[p_z, p_y, p_x] != -index:
                data[p_z, p_y, p_x] = -index
                #list_of_boarders.append((p_z, p_y, p_x))
            return False

    # z check
    if p_z > 0 and check_point(p_x, p_y, p_z-1):
        repit_check_now_point = True
    if p_z < d-1 and check_point(p_x, p_y, p_z + 1):
        repit_check_now_point = True

    # y check
    if p_y > 0 and check_point(p_x, p_y - 1, p_z):
        repit_check_now_point = True
    if p_y < h - 1 and check_point(p_x, p_y + 1, p_z):
        repit_check_now_point = True

    # x check
    if p_x > 0 and check_point(p_x - 1, p_y, p_z):
        repit_check_now_point = True
    if p_x < w - 1 and check_point(p_x + 1, p_y, p_z):
        repit_check_now_point = True

    if repit_check_now_point:
        next_iteration_work_points.append(Vector(p_x, p_y, p_z, dtype=int))
    return next_iteration_work_points

def _update_labels_from_mask(cell_mask, label, border_value=1):
    """
    Создаёт массив меток для одной клетки:
    +label внутри, -label на границе.
    """
    struct = generate_binary_structure(3, 1)          # 6-связность
    interior = binary_erosion(cell_mask, structure=struct, border_value=border_value)
    boundary = cell_mask & ~interior

    result = np.zeros(cell_mask.shape, dtype=np.int32)
    result[interior] = label
    result[boundary] = -label
    return result

def shift_boundary(cell_fields, label, distance, shift_mask=None):
    """
    Быстрая версия: EDT только внутри bounding box клетки.
    """
    # 1. Маска клетки и источник сдвига
    cell_mask = np.abs(cell_fields) == label
    if shift_mask is None:
        source = (cell_fields == -label)
    else:
        source = shift_mask & (cell_fields == -label)

    if not np.any(source):
        return cell_fields

    # 2. Координаты клетки и отступ
    coords = np.argwhere(cell_mask)
    margin = int(np.ceil(distance)) + 2
    shape = np.array(cell_fields.shape)
    min_coords = np.maximum(0, coords.min(axis=0) - margin)
    max_coords = np.minimum(shape - 1, coords.max(axis=0) + margin)

    # 3. Срезы и вырезание нужных массивов
    slices = tuple(slice(mi, ma + 1) for mi, ma in zip(min_coords, max_coords))
    crop_cell_mask = cell_mask[slices]
    crop_source = source[slices]

    # 4. EDT внутри ограниченной области
    binary_for_edt = ~crop_source  # True везде, кроме source (нули для EDT)
    dist_crop = distance_transform_edt(binary_for_edt)

    # 5. Новая маска клетки внутри crop-области
    new_cell_mask_crop = crop_cell_mask & (dist_crop > distance)

    # 6. Очищаем всю старую клетку в исходном массиве
    cell_fields[cell_mask] = 0

    if not np.any(new_cell_mask_crop):
        return cell_fields  # клетка исчезла

    # 7. Строим расширенную маску для корректного определения границ
    padded_shape = tuple(s + 2 for s in new_cell_mask_crop.shape)
    padded = np.zeros(padded_shape, dtype=bool)
    padded[1:-1, 1:-1, 1:-1] = new_cell_mask_crop

    # Заполняем отступы: 1, если грань примыкает к границе тензора
    if min_coords[0] == 0:
        padded[0, :, :] = 1
    if max_coords[0] == shape[0] - 1:
        padded[-1, :, :] = 1
    if min_coords[1] == 0:
        padded[:, 0, :] = 1
    if max_coords[1] == shape[1] - 1:
        padded[:, -1, :] = 1
    if min_coords[2] == 0:
        padded[:, :, 0] = 1
    if max_coords[2] == shape[2] - 1:
        padded[:, :, -1] = 1

    # 8. Получаем метки на расширенной маске (border_value=0, т.к. отступы уже заданы)
    padded_labels = _update_labels_from_mask(padded, label, border_value=0)

    # 9. Вырезаем внутреннюю часть и вставляем в исходный массив
    inner_labels = padded_labels[1:-1, 1:-1, 1:-1]
    cell_fields[slices][new_cell_mask_crop] = inner_labels[new_cell_mask_crop]

    return cell_fields


def shift_boundary_with_shell(cell_fields, label, distance):
    """
    Сдвигает ВСЮ границу клетки внутрь на `distance` и возвращает
    маску «оболочки», появившейся между старой и новой границами.
    Изменяет `cell_fields` на месте.

    Параметры
    ----------
    cell_fields : np.ndarray (3D, int32)
    label : int (>0)
    distance : float

    Возвращает
    -------
    cell_fields : обновлённый массив
    shell_mask : np.ndarray (3D, bool)
        True в тех вокселях, которые исчезли из клетки (были внутри,
        но стали фоном).
    """
    # Запоминаем маску старой клетки
    old_cell_mask = np.abs(cell_fields) == label

    # Сдвигаем всю границу (shift_mask=None)
    shift_boundary(cell_fields, label, distance, shift_mask=None)

    # Маска новой клетки
    new_cell_mask = np.abs(cell_fields) == label

    # Оболочка – всё, что было в клетке и теперь отсутствует
    shell_mask = old_cell_mask & ~new_cell_mask

    cell_fields[shell_mask == True] = -label

    return shell_mask

def triple_shift_boundary_with_shell(cell_fields, label, distance1, distance2, distance3):
    """
    Сдвигает ВСЮ границу клетки внутрь на `distance1`, `distance2` и `distance3` и возвращает
    3 маски «оболочки», появившейся между старой и новой границами.
    Изменяет `cell_fields` на месте. Внутренние области помечаются как граница, поскольку в этой области лучше ничего не делать

    Параметры
    ----------
    cell_fields : np.ndarray (3D, int32)
    label : int (>0)
    distance1 : float
    distance2 : float
    distance3 : float

    Возвращает
    -------
    cell_fields : обновлённый массив
    shell_mask : np.ndarray (3D, bool)
        True в тех вокселях, которые исчезли из клетки (были внутри,
        но стали фоном).
    """
    ########## MASK1 ############
    # Запоминаем маску старой клетки
    old_cell_mask = np.abs(cell_fields) == label

    # Сдвигаем всю границу (shift_mask=None)
    shift_boundary(cell_fields, label, distance1, shift_mask=None)

    # Маска новой клетки
    new_cell_mask1 = np.abs(cell_fields) == label

    ########## MASK2 ############
    # Оболочка – всё, что было в клетке и теперь отсутствует
    shell_mask1 = old_cell_mask & ~new_cell_mask1

    # Сдвигаем всю границу (shift_mask=None)
    shift_boundary(cell_fields, label, distance2, shift_mask=None)

    # Маска новой клетки
    new_cell_mask2 = np.abs(cell_fields) == label

    # Оболочка – всё, что было в клетке и теперь отсутствует
    shell_mask2 = new_cell_mask1 & ~new_cell_mask2

    ########## MASK3 ############
    # Сдвигаем всю границу (shift_mask=None)
    shift_boundary(cell_fields, label, distance3, shift_mask=None)

    # Маска новой клетки
    new_cell_mask3 = np.abs(cell_fields) == label

    # Оболочка – всё, что было в клетке и теперь отсутствует
    shell_mask3 = new_cell_mask2 & ~new_cell_mask3

    cell_fields[shell_mask1 == True] = -label
    cell_fields[shell_mask2 == True] = -label
    cell_fields[shell_mask3 == True] = -label

    return shell_mask1, shell_mask2, shell_mask3


def generate_spheres_mask(shape, density, radius_range):
    """
    Быстрая генерация маски непересекающихся сфер (512^3 за секунды).
    """
    mask = np.zeros(shape, dtype=bool)
    total_voxels = np.prod(shape)
    target_full_volume = density * total_voxels

    centers = []
    radii = []
    current_volume = 0.0
    max_attempts = 20000
    attempts = 0

    # Пространственная сетка для ускорения проверки пересечений
    # Размер ячейки = 2 * max_radius, чтобы любая сфера пересекала не более 8 ячеек
    min_r, max_r = radius_range
    cell_size = 2 * max_r
    grid_shape = tuple(int(np.ceil(s / cell_size)) for s in shape)
    grid = {}  # ключ (i,j,k) -> список индексов сфер в этой ячейке

    while current_volume < target_full_volume and attempts < max_attempts:
        c = np.array([get_rand_int((0, shape[0]-1)),
                      get_rand_int((0, shape[1]-1)),
                      get_rand_int((0, shape[2]-1))])
        r = get_rand_int(radius_range)

        # В каких ячейках может находиться сфера (по bounding box)
        min_cell = np.floor((c - r) / cell_size).astype(int)
        max_cell = np.floor((c + r) / cell_size).astype(int)
        # Ограничиваем размерами сетки
        min_cell = np.maximum(min_cell, 0)
        max_cell = np.minimum(max_cell, np.array(grid_shape) - 1)

        # Проверяем пересечения только с теми сферами, что лежат в этих ячейках
        collision = False
        for i in range(min_cell[0], max_cell[0] + 1):
            for j in range(min_cell[1], max_cell[1] + 1):
                for k in range(min_cell[2], max_cell[2] + 1):
                    for idx in grid.get((i, j, k), []):
                        oc = centers[idx]
                        or_ = radii[idx]
                        if np.sum((c - oc)**2) < (r + or_)**2:
                            collision = True
                            break
                    if collision:
                        break
                if collision:
                    break
            if collision:
                break

        if not collision:
            idx = len(centers)
            centers.append(c)
            radii.append(r)
            current_volume += (4.0 / 3.0) * np.pi * r**3

            # Добавляем индекс сферы во все ячейки, которые пересекает её bounding box
            for i in range(min_cell[0], max_cell[0] + 1):
                for j in range(min_cell[1], max_cell[1] + 1):
                    for k in range(min_cell[2], max_cell[2] + 1):
                        grid.setdefault((i, j, k), []).append(idx)
            attempts = 0
        else:
            attempts += 1

    # Быстрая отрисовка сфер по отдельным bounding box'ам
    for c, r in zip(centers, radii):
        # Целочисленные границы сферы, обрезанные тензором
        zmin = max(0, int(np.floor(c[0] - r)))
        zmax = min(shape[0] - 1, int(np.ceil(c[0] + r)))
        ymin = max(0, int(np.floor(c[1] - r)))
        ymax = min(shape[1] - 1, int(np.ceil(c[1] + r)))
        xmin = max(0, int(np.floor(c[2] - r)))
        xmax = min(shape[2] - 1, int(np.ceil(c[2] + r)))

        # Локальные координаты (одномерные массивы) – экономят память
        z_loc = np.arange(zmin, zmax + 1) - c[0]
        y_loc = np.arange(ymin, ymax + 1) - c[1]
        x_loc = np.arange(xmin, xmax + 1) - c[2]

        # Вычисляем расстояние через broadcasting (без meshgrid)
        Z, Y, X = np.ogrid[:len(z_loc), :len(y_loc), :len(x_loc)]
        # Приводим к настоящим координатам (не индексам)
        Z = z_loc[Z]
        Y = y_loc[Y]
        X = x_loc[X]
        dist2 = Z*Z + Y*Y + X*X

        # Обновляем срез маски
        mask[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1] |= (dist2 <= r*r)

    return mask




