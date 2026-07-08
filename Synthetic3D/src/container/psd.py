import numpy as np

from Synthetic3D.src.utilities.check_of_params import check_param, update_param
from Synthetic3D.src.utilities.logging_config import logger
from Synthetic3D.src.hard.random_params import get_rand_int, color_dim_check
from Synthetic3D.src.hard.drawing_and_filliing.draw_data_by_mask import draw_data_by_mask_and_random_value

from scipy.ndimage import distance_transform_edt, generate_binary_structure, binary_erosion

class PSD:
    def __init__(self,
                cell_fields,
                center,
                radius,
                params,
                unique_indexes=None,
                mask_sphere=None   # опционально – для уменьшения вычислений в дальнейшем
                ):
        """
        Параметры:
            cell_fields : 3D массив меток клеток (положительные – внутренность, отрицательные – мембрана)
            center : tuple (z, y, x) центр PSD
            radius : радиус сферы, в которой строится PSD
            params : словарь параметров (ключ 'psd' содержит настройки)
            mask_sphere : опционально, готовая булева маска сферы
            unique_indexes : список из двух индексов [индекс_клетки1, индекс_клетки2]
        """
        self.params = params.get("psd", {})
        self._check_and_set_default_params()
        self.Create(cell_fields, center, radius, params, mask_sphere, unique_indexes)

    def _check_and_set_default_params(self):
        warning_list = []

        warning_list += check_param(self.params, "presynapse_thickness", (5,10))    # толщина пресинаптической мембраны
        warning_list += check_param(self.params, "postsynapse_thickness", (5,10))   # толщина постсинаптической мембраны
        warning_list += check_param(self.params, "synapse_color_param", (64, 10))

        warning_list += check_param(self.params, "gap_thickness", (10, 20))         # толщина синаптической щели (дополнительные слои между мембранами)
        warning_list += check_param(self.params, "gap_color_param", (95, 20))

        warning_list += check_param(self.params, "out_thickness", (5, 20))          # максимальная выпуклость линзы затемнения
        warning_list += check_param(self.params, "out_color_param", (95, 20))

        warning_list += check_param(self.params, "membrane_mask_color", 255)

        if len(warning_list) != 0:
            logger.config(f'\tWarning PSD!\n{"\n".join(warning_list)}')
            warning_list = ["Warning PSD!"] + warning_list
        return warning_list

    def update_draw_config(self, config):
        new_params = config.get("psd", None)
        if new_params is not None:
            update_param(self.params, new_params, "synapse_color_param")
            update_param(self.params, new_params, "gap_color_param")
            update_param(self.params, new_params, "out_color_param")


    def _expand_inside(self, seed_mask, cell_mask, thickness):
        if thickness <= 0:
            return np.zeros_like(seed_mask, dtype=bool)
        edt_input = np.ones(seed_mask.shape, dtype=bool)
        edt_input[seed_mask] = False
        dist = distance_transform_edt(edt_input)
        expanded = (dist <= thickness) & cell_mask
        expanded = expanded | seed_mask
        return expanded


    def _get_boarder_mask(self, mask, border_value=True):
        struct = generate_binary_structure(3, 1)          # 6-связность (крестик)
        interior = binary_erosion(mask, structure=struct, border_value=border_value)
        return mask & ~interior

    def _get_crop_slices(self, cell_fields_shape, center, radius, margin):
        """
        Вычисляет подкуб с запасом и возвращает всё необходимое для работы в подкубе:
        - slices: кортеж срезов для извлечения подкуба и вставки масок обратно
        - crop: копия подкуба из cell_fields
        - center_crop: координаты центра в подкубе
        - sphere_crop: маска сферы в подкубе
        """
        shape = cell_fields_shape
        z_min = max(0, center[0] - radius - margin)
        z_max = min(shape[0], center[0] + radius + margin + 1)
        y_min = max(0, center[1] - radius - margin)
        y_max = min(shape[1], center[1] + radius + margin + 1)
        x_min = max(0, center[2] - radius - margin)
        x_max = min(shape[2], center[2] + radius + margin + 1)
        slices = (slice(z_min, z_max), slice(y_min, y_max), slice(x_min, x_max))

        return slices

    def create_sphere_mask(self, slice_data, center, radius):
        # Размеры подкуба (предполагаем, что срезы идут с положительным шагом)
        shape = (
            slice_data[0].stop - slice_data[0].start,
            slice_data[1].stop - slice_data[1].start,
            slice_data[2].stop - slice_data[2].start
        )
        # Относительный центр внутри подкуба
        center_crop = (
            center[0] - slice_data[0].start,
            center[1] - slice_data[1].start,
            center[2] - slice_data[2].start
        )

        # Сетка координат в подкубе
        Z = np.arange(shape[0])
        Y = np.arange(shape[1])
        X = np.arange(shape[2])
        ZZ, YY, XX = np.meshgrid(Z, Y, X, indexing='ij')
        dist = np.sqrt((ZZ - center_crop[0])**2 + (YY - center_crop[1])**2 + (XX - center_crop[2])**2)
        sphere_mask = dist <= radius
        return sphere_mask   # размер shape

    def Create(self, cell_fields, center, radius, params, mask_sphere, unique_indexes):
        # ---- Выносим параметры в локальные переменные ----
        presynapse_thickness = self.params["presynapse_thickness"]
        postsynapse_thickness = self.params["postsynapse_thickness"]
        gap_thickness = self.params["gap_thickness"]
        out_thickness = self.params["out_thickness"]

        shape = cell_fields.shape

        if unique_indexes is None:
            raise NotImplementedError()
            # на данный момент не требуется, поскольку конвеер уже это делаеи при проверке расположения. Возможно будет сделана при полной реализации позже


        # ---- Случайные толщины (используем локальные переменные) ----
        gap_th = get_rand_int(gap_thickness)
        pre_th = get_rand_int(presynapse_thickness)
        post_th = get_rand_int(postsynapse_thickness)
        out_th = get_rand_int(out_thickness)

        #print(gap_th, pre_th, post_th, out_th)

        max_th = max(gap_th+pre_th, gap_th+post_th+out_th) + 5
        margin = int(np.ceil(max_th)) + 2

        if unique_indexes is None or len(unique_indexes) != 2:
            raise ValueError("unique_indexes должен содержать ровно два индекса.")
        self.idx1, self.idx2 = unique_indexes

        # ---- Получаем подкуб ----
        slices = self._get_crop_slices(cell_fields.shape, center, radius, margin)

        if mask_sphere is None:
            sphere_crop = self.create_sphere_mask(slices, center, radius)
            mask_sphere = np.zeros(shape, dtype=bool)
            mask_sphere[slices] = sphere_crop
        else:
            sphere_crop = mask_sphere[slices].copy()

        crop = cell_fields[slices].copy()

        # 2. Маски клеток и их границ
        cell1_mask = (np.abs(crop) == self.idx1)
        cell2_mask = ~cell1_mask

        # Граница клетки: маска & ~эрозия(маска)
        # Пересечение со сферой – начальная область контакта
        contact1 = self._get_boarder_mask(cell1_mask) & sphere_crop
        contact2 = self._get_boarder_mask(cell2_mask) & sphere_crop

        # 4. Смещение для gap: половина gap_thickness
        gap2 = gap_th // 2
        gap1 = gap_th - gap2

        gap1_mask = self._expand_inside(contact1, cell1_mask, gap1)
        gap2_mask = self._expand_inside(contact2, cell2_mask, gap2)
        gap_crop = gap1_mask | gap2_mask

        start_seed_mask_1 = gap_crop if np.any(gap_crop) else contact1
        membrane1_crop = self._expand_inside(start_seed_mask_1, cell1_mask, pre_th) & ~gap_crop & sphere_crop
        start_seed_mask_2 = gap_crop if np.any(gap_crop) else contact2
        membrane2_crop = self._expand_inside(start_seed_mask_2, cell2_mask, post_th) & ~gap_crop & sphere_crop
        shading_crop = self._expand_inside(membrane2_crop, cell2_mask, out_th) & ~membrane2_crop & sphere_crop

        # ---- Переносим маски в полный размер ----
        self.mask_gap = np.zeros(shape, dtype=bool)
        self.mask_pre_membrane = np.zeros(shape, dtype=bool)
        self.mask_post_membrane = np.zeros(shape, dtype=bool)
        self.mask_shading = np.zeros(shape, dtype=bool)

        self.mask_gap[slices] = gap_crop
        self.mask_pre_membrane[slices] = membrane1_crop
        self.mask_post_membrane[slices] = membrane2_crop
        self.mask_shading[slices] = shading_crop

        self.psd_mask = self.mask_pre_membrane | self.mask_post_membrane | self.mask_gap

        # ---- Зануляем в оригинальном cell_fields ----
        cell_fields[self.psd_mask | self.mask_shading] = 0
        cell_fields[mask_sphere & (cell_fields < 0)] = 0

    def Draw(self, data):
        draw_data_by_mask_and_random_value(data, self.mask_shading  , self.params["out_color_param"])
        draw_data_by_mask_and_random_value(data, self.mask_pre_membrane, self.params["synapse_color_param"])
        draw_data_by_mask_and_random_value(data, self.mask_post_membrane, self.params["synapse_color_param"])
        draw_data_by_mask_and_random_value(data, self.mask_gap      , self.params["gap_color_param"])

    def DrawMask(self, mask_data, color=None):
        """Закрашивает маску PSD указанным цветом"""
        work_color = self.params["membrane_mask_color"] if color is None else color
        work_color = color_dim_check(work_color, mask_data.shape)
        mask_data[self.psd_mask] = work_color
