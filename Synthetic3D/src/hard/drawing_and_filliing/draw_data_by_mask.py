import numpy as np
from scipy.ndimage import gaussian_filter
from Synthetic3D.src.hard.random_params import choise_use_color_by_param, color_dim_check, get_color_index_fun_by_param

def draw_data_by_mask_and_random_value(data, mask, color_param):
    spatial_shape = data.shape[:3] if data.ndim == 4 else data.shape

    index_fun = get_color_index_fun_by_param(color_param)
    if index_fun != 2:
        data[mask] = color_dim_check(choise_use_color_by_param(color_param), data.shape)
        return data

    # --- Ветка index_fun == 2 ---
    val1, val2 = color_param
    target_std = val2 / 3.0
    if not np.any(mask):
        return data

    sigma = 10
    radius = 5
    support = max(radius, int(4 * sigma)) + 1

    mask_idx = np.argwhere(mask)                       # координаты True-пикселей
    min_coords = mask_idx.min(axis=0) - support
    max_coords = mask_idx.max(axis=0) + support + 1
    for i in range(len(spatial_shape)):
        min_coords[i] = max(min_coords[i], 0)
        max_coords[i] = min(max_coords[i], spatial_shape[i])

    crop_shape = tuple(int(max_c - min_c) for min_c, max_c in zip(min_coords, max_coords))

    # Генерация только в урезанной области
    random_crop = np.random.normal(loc=0.0, scale=1.0, size=crop_shape)
    smoothed_crop = gaussian_filter(random_crop, radius=radius, sigma=sigma)
    actual_std = np.std(smoothed_crop)
    result_crop = (smoothed_crop / actual_std) * target_std + val1
    result_crop = np.clip(np.round(result_crop), 0, 255).astype(np.uint8)

    # Пересчёт глобальных индексов маски в локальные координаты подобласти
    offsets = min_coords
    local_idx = mask_idx - offsets

    # Присваиваем только нужные пиксели
    if data.ndim == 3:
        data[tuple(mask_idx.T)] = result_crop[tuple(local_idx.T)]
    else:
        # 4D: повторяем значение по всем каналам, как в оригинале result_vals[mask, None]
        data[tuple(mask_idx.T)] = result_crop[tuple(local_idx.T)][:, None]

    return data

