import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import sys
from Synthetic3D.src.hard.random_params import get_color_fun_by_param, get_color_index_fun_by_param, color_dim_check


from numba import njit


def fill_tensor(shape, mode, val1, val2, c, gaussian_radius, gaussian_sigma):
    """
    Заполняет массив размером shape либо константой (mode=0),
    либо случайными значениями из нормального распределения (mode=1)
    с параметрами mean=val1, scatter=val2 (где scatter = 3*sigma).
    Для 4D‑массивов все каналы получают одинаковое значение для каждого вокселя.
    """
    if mode == 0:
        print("val1", val1)
        return np.full(shape, val1, dtype=np.uint8)
    spatial_shape = shape[:3] if len(shape) == 4 else shape
    # Генерируем случайный шум
    random_vals = np.random.normal(loc=0.0, scale=val2, size=spatial_shape)
    # Преобразование шума в области с плавным изменением интенсивности
    smoothed = gaussian_filter(random_vals, radius=gaussian_radius, sigma=gaussian_sigma)

    # Масштабируем до нужного sigma и добавляем среднее
    actual_std = np.std(smoothed)
    result_vals = (smoothed / actual_std) * val2 + val1

    # Обрезаем до [0, 255]
    result_vals = np.clip(np.round(result_vals), 0, 255).astype(np.uint8)

    if len(shape) == 3:
        return result_vals
    res = np.empty(shape, dtype=np.uint8)
    res[:] = result_vals[..., None]
    return res

def draw_sphere(result, i,j,k, cycle_range, shape, nodule_color_iter):
    size = np.random.uniform(cycle_range[0], cycle_range[1])
    # создаем сферу
    for x in range(max(0, int(i - size)), min(shape[0], int(i + size) + 1)):
        for y in range(max(0, int(j - size)), min(shape[1], int(j + size) + 1)):
            for z in range(max(0, int(k - size)), min(shape[2], int(k + size) + 1)):
                if (x - i) ** 2 + (y - j) ** 2 + (z - k) ** 2 <= size ** 2:
                    result[x, y, z] = nodule_color_iter

@njit
def numba_draw_sphere(result, i, j, k, cycle_range, shape, color_array):
    """
    Заливает сферу в массиве result.
    color_array – одномерный массив длины c (число каналов).
    """
    size = np.random.uniform(cycle_range[0], cycle_range[1])
    # Границы по осям
    x_start = max(0, int(i - size))
    x_end = min(shape[0], int(i + size) + 1)
    y_start = max(0, int(j - size))
    y_end = min(shape[1], int(j + size) + 1)
    z_start = max(0, int(k - size))
    z_end = min(shape[2], int(k + size) + 1)

    # Определяем размерность массива
    if result.ndim == 3:          # один канал
        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                for z in range(z_start, z_end):
                    if (x - i) ** 2 + (y - j) ** 2 + (z - k) ** 2 <= size ** 2:
                        result[x, y, z] = color_array[0]
    else:                         # многоканальный (ndim == 4)
        c = color_array.shape[0]
        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                for z in range(z_start, z_end):
                    if (x - i) ** 2 + (y - j) ** 2 + (z - k) ** 2 <= size ** 2:
                        for ch in range(c):
                            result[x, y, z, ch] = color_array[ch]


def draw_cylinder(result, i,j,k, cylinder_length_range, cylinder_radius_range, shape, nodule_color_iter):
    # параметры цилиндра
    x0, y0, z0 = i, j, k
    phi = np.random.uniform(0, 2 * np.pi)
    theta = np.random.uniform(0, np.pi)
    dx = np.sin(theta) * np.cos(phi)
    dy = np.sin(theta) * np.sin(phi)
    dz = np.cos(theta)
    length = np.random.uniform(cylinder_length_range[0], cylinder_length_range[1])
    radius = np.random.uniform(cylinder_radius_range[0], cylinder_radius_range[1])

    x1 = x0 + dx * length
    y1 = y0 + dy * length
    z1 = z0 + dz * length

    num_points = int(length * 3)
    xs = np.linspace(x0, x1, num_points)
    ys = np.linspace(y0, y1, num_points)
    zs = np.linspace(z0, z1, num_points)

    for xi, yi, zi in zip(xs, ys, zs):
        xi_int = int(round(xi))
        yi_int = int(round(yi))
        zi_int = int(round(zi))
        # добавляем вокруг линии цилиндра (радиус)
        for x in range(max(0, xi_int - int(radius)), min(shape[0], xi_int + int(radius) + 1)):
            for y in range(max(0, yi_int - int(radius)), min(shape[1], yi_int + int(radius) + 1)):
                for z in range(max(0, zi_int - int(radius)), min(shape[2], zi_int + int(radius) + 1)):
                    if (x - xi) ** 2 + (y - yi) ** 2 + (z - zi) ** 2 <= radius ** 2:
                        result[x, y, z] = nodule_color_iter

@njit
def numba_draw_cylinder(result, i, j, k, cylinder_length_range, cylinder_radius_range, shape, color_array):
    """
    Заливает цилиндр (заданный случайной ориентацией и параметрами).
    color_array – одномерный массив длины c.
    """
    x0, y0, z0 = i, j, k
    phi = np.random.uniform(0, 2 * np.pi)
    theta = np.random.uniform(0, np.pi)
    dx = np.sin(theta) * np.cos(phi)
    dy = np.sin(theta) * np.sin(phi)
    dz = np.cos(theta)
    length = np.random.uniform(cylinder_length_range[0], cylinder_length_range[1])
    radius = np.random.uniform(cylinder_radius_range[0], cylinder_radius_range[1])

    x1 = x0 + dx * length
    y1 = y0 + dy * length
    z1 = z0 + dz * length

    num_points = int(length * 3)
    xs = np.linspace(x0, x1, num_points)
    ys = np.linspace(y0, y1, num_points)
    zs = np.linspace(z0, z1, num_points)

    # Для ускорения преобразуем радиус в int
    r_int = radius

    if result.ndim == 3:
        for idx in range(num_points):
            xi = xs[idx]
            yi = ys[idx]
            zi = zs[idx]
            xi_int = int(round(xi))
            yi_int = int(round(yi))
            zi_int = int(round(zi))
            x_start = max(0, xi_int - r_int)
            x_end = min(shape[0], xi_int + r_int + 1)
            y_start = max(0, yi_int - r_int)
            y_end = min(shape[1], yi_int + r_int + 1)
            z_start = max(0, zi_int - r_int)
            z_end = min(shape[2], zi_int + r_int + 1)
            for x in range(x_start, x_end):
                for y in range(y_start, y_end):
                    for z in range(z_start, z_end):
                        if (x - xi) ** 2 + (y - yi) ** 2 + (z - zi) ** 2 <= radius ** 2:
                            result[x, y, z] = color_array[0]
    else:
        c = color_array.shape[0]
        for idx in range(num_points):
            xi = xs[idx]
            yi = ys[idx]
            zi = zs[idx]
            xi_int = int(round(xi))
            yi_int = int(round(yi))
            zi_int = int(round(zi))
            x_start = max(0, xi_int - r_int)
            x_end = min(shape[0], xi_int + r_int + 1)
            y_start = max(0, yi_int - r_int)
            y_end = min(shape[1], yi_int + r_int + 1)
            z_start = max(0, zi_int - r_int)
            z_end = min(shape[2], zi_int + r_int + 1)
            for x in range(x_start, x_end):
                for y in range(y_start, y_end):
                    for z in range(z_start, z_end):
                        if (x - xi) ** 2 + (y - yi) ** 2 + (z - zi) ** 2 <= radius ** 2:
                            for ch in range(c):
                                result[x, y, z, ch] = color_array[ch]


def add_internal_structures(
    volume,
    density=0.01,
    nodule_color_param=(122, 6),
    background_color_param=(200, 1),
    shape_type='both',
    cycle_range=(1, 3),
    cylinder_length_range=(10.0, 20.0),
    cylinder_radius_range=(1, 2),
    gaussian_radius=3,
    gaussian_sigma=1.0
):
    """
    Добавляет внутри объемов вкрапления для имитации микроскопической текстуры.
    """

    shape = volume.shape
    if len(shape) == 3:
        c = 1
    else:
        c = shape[3]

    work_rule_back = get_color_index_fun_by_param(background_color_param)
    print("work_rule_back", work_rule_back)
    if work_rule_back != 2:
        mode = 0
        val1 = background_color_param
        val2 = 0
    else:
        mode = 1
        val1 = background_color_param[0]
        val2 = background_color_param[1]

    result = fill_tensor(volume.shape, mode, val1, val2, c, 15, 10)

    color_fun = get_color_fun_by_param(nodule_color_param)

    for i in tqdm(range(shape[0]), desc="Заполнение текстурой клетки", colour="GREEN", file=sys.stdout):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if np.random.rand() < density:
                    nodule_color_iter = np.array(color_dim_check(color_fun(), c))
                    obj_type = shape_type
                    if shape_type == 'both':
                        obj_type = 'sphere' if np.random.rand() < 0.66 else 'cylinder'
                    if obj_type == 'sphere':
                        numba_draw_sphere(result=result,
                                          i=i, j=j, k=k,
                                          cycle_range = cycle_range,
                                          shape=shape,
                                          color_array=nodule_color_iter)
                    elif obj_type == 'cylinder':
                        numba_draw_cylinder(result=result,
                                            i=i, j=j, k=k,
                                            cylinder_length_range = cylinder_length_range,
                                            cylinder_radius_range = cylinder_radius_range,
                                            shape=shape,
                                            color_array=nodule_color_iter)

    # размытие для натуральности текстуры
    result = gaussian_filter(result, radius=gaussian_radius, sigma=gaussian_sigma)
    return result

def CreatePossionNoise(shape_of_data, noise_value):
    """
        реализация из 2Д версии
    """
    ## apply a 5x5 blur to the noisy image to smooth out high-frequency noise
    # noisy = cv2.blur(noisy, (5, 5))
    ## create a second array of ones with the same shape and data type as the first one
    # noisy2 = np.random.poisson(np.ones_like(noisy) * poisson_noise - poisson_noise
    ## add the two noisy arrays together
    # noisy = noisy + noisy2

    noisy1 = np.random.poisson(size=shape_of_data + (1,)) * noise_value - noise_value
    noisy1 = gaussian_filter(noisy1, radius=2, sigma=10)
    noisy2 = np.random.poisson(size=shape_of_data + (1,)) * noise_value - noise_value
    # print(gray_noise.min(), gray_noise.max(), gray_noise.mean())
    return noisy1 + noisy2


def AddGaussianBlur(data, radius, sigma):
    return gaussian_filter(data, radius=radius, sigma=sigma)
