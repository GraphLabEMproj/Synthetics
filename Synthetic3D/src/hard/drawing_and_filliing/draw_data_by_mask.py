import numpy as np
from scipy.ndimage import gaussian_filter
from Synthetic3D.src.hard.random_params import choise_use_color_by_param, color_dim_check, get_color_index_fun_by_param

def draw_data_by_mask_and_random_value(data, mask, color_param):
        spatial_shape = data.shape[:3] if len(data.shape) == 4 else data.shape

        index_fun = get_color_index_fun_by_param(color_param)
        if index_fun == 2:
            val1, val2 = color_param
            target_std = val2/3

            random_vals = np.random.normal(loc=0.0, scale=1, size=spatial_shape)
            # Преобразование шума в области с плавным изменением интенсивности
            smoothed = gaussian_filter(random_vals, radius=5, sigma=10)

            # Масштабируем до нужного sigma и добавляем среднее
            actual_std = np.std(smoothed)
            result_vals = (smoothed / actual_std) * target_std + val1

            # Обрезаем до [0, 255]
            result_vals = np.clip(np.round(result_vals), 0, 255).astype(np.uint8)

            if len(data.shape) == 3:
                data[mask==True] = result_vals[mask==True]
                return data
            data[mask==True] = result_vals[mask==True, None]
            return data

        else:
            data[mask==True] = color_dim_check(choise_use_color_by_param(color_param), data.shape)

