import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import sys


def add_internal_structures(
    volume,
    density=0.01,
    nodule_color=140,
    background_color=200,
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
    result = np.full_like(volume, background_color, dtype=np.float32)
    shape = volume.shape

    for i in tqdm(range(shape[0]), desc="Заполнение текстурой клетки", colour="GREEN", file=sys.stdout):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if np.random.rand() < density:

                    nodule_color_iter = nodule_color if isinstance(nodule_color, int) else int(np.random.uniform(*nodule_color))

                    obj_type = shape_type
                    if shape_type == 'both':
                        obj_type = 'sphere' if np.random.rand() < 0.66 else 'cylinder'

                    size = np.random.uniform(*cycle_range)

                    if obj_type == 'sphere':
                        # создаем сферу
                        for x in range(max(0, int(i - size)), min(shape[0], int(i + size) + 1)):
                            for y in range(max(0, int(j - size)), min(shape[1], int(j + size) + 1)):
                                for z in range(max(0, int(k - size)), min(shape[2], int(k + size) + 1)):
                                    if (x - i) ** 2 + (y - j) ** 2 + (z - k) ** 2 <= size ** 2:
                                        result[x, y, z] = nodule_color_iter
                    elif obj_type == 'cylinder':
                        # параметры цилиндра
                        x0, y0, z0 = i, j, k
                        phi = np.random.uniform(0, 2 * np.pi)
                        theta = np.random.uniform(0, np.pi)
                        dx = np.sin(theta) * np.cos(phi)
                        dy = np.sin(theta) * np.sin(phi)
                        dz = np.cos(theta)
                        length = np.random.uniform(*cylinder_length_range)
                        radius = np.random.uniform(*cylinder_radius_range)

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
