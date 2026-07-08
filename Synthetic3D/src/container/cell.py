import numpy as np
from Synthetic3D.src.hard.structure.vector import Vector
from Synthetic3D.src.hard.drawing_and_filliing.draw_element import edge_check_3D
from Synthetic3D.src.organells.empty_organelle import EmptyOrganelle

from Synthetic3D.src.utilities.check_of_params import check_param, update_param
from Synthetic3D.src.container.cell_operation import ExpansionOfPoint26Dir_v3, ExpansionOfPoint6Dir

from Synthetic3D.src.container.cell_operation import shift_boundary_with_shell, shift_boundary
from Synthetic3D.src.utilities.logging_config import logger


from Synthetic3D.src.hard.drawing_and_filliing.draw_data_by_mask import draw_data_by_mask_and_random_value

from Synthetic3D.src.hard.random_params import get_rand_int, get_color_index_fun_by_param, choise_use_color_by_param

from Synthetic3D.src.container.psd import PSD

class Cell():
    """
    Класс представляющий клетку и содержащий код для её расширения. Стартует из зоны вокруг органелл.
    index - метка данной клетки в датасете. Отрицательное значение обозначает границу.

    """

    def __init__(self, index, list_of_organels=[EmptyOrganelle()], params={}):
        self.warnings = []
        self.list_of_organells = list_of_organels
        self.work_points = []
        self.index = index
        self.params = params.get("cell", {})

        self.warnings += check_param(self.params, "probability_of_expansion", 0.5)
        self.warnings += check_param(self.params, "membrane_thickness", 2)
        self.warnings += check_param(self.params, "distance_cell_shift_range", (1,2))
        self.warnings += check_param(self.params, "membrane_color_param", (0, 127, 0))
        self.warnings += check_param(self.params, "membrane_mask_color", 255)

    # рисует минимальную маску для разрастания регионов, полностью покрывая внутренние органеллы
    ####################################################################################################################### Нуждается в улучшении
    def DrawKernelOfArea(self, data):
        d,h,w = data.shape[:3]
        for organell in self.list_of_organells:
            t_work_points = organell.DrawArea(data, self.index)
            # отсеить краевые и перевести в целое, так как далее работа только с индексами
            for point in t_work_points:
                if not edge_check_3D(*point, w=w, h=h, d=d):
                    self.work_points.append(np.round(point).astype(int))

    '''
    ######################################################################### Подумать над необходимостью увеличения каналлов
    def ExpansionOfPoint26Dir(self, data, position: Vector | np.ndarray):
        assert len(data.shape) == 3 or data.shape[3] == 1, "Данные для алгоритма разрастания должны иметь 3 оси и 1 канал."
        d, h, w = data.shape
        p_x, p_y, p_z = position
        next_iteration_work_points = []
        repit_check_now_point = False

        for i_z in (-1, 0, 1):
            n_z = p_z + i_z
            for i_x in (-1, 0, 1):
                n_x = p_x + i_x
                for i_y in (-1, 0, 1):
                    n_y = p_y + i_y
                    if p_x == n_x and p_y == n_y and p_z == n_z: # текущая точка нам не интересна. только окресность
                        continue
                    elif not edge_check_3D(n_x,n_y,n_z, w, h, d): # если мы внутри data
                        # если не занято, то занимаем с некоторой вероятностью, иначе отправляем на пересмотр
                        if data[n_z, n_y, n_x] == 0:
                            # 1/3 для углов, 1/2 для середин ребер и 1 для середин граней
                            if np.random.random() < 1/(abs(i_z) + abs(i_y) + abs(i_x)):
                                next_iteration_work_points.append(Vector(n_x, n_y, n_z, dtype=int))
                                data[n_z, n_y, n_x] = self.index
                            else:
                                repit_check_now_point = True

                        # попали в соседа self.index или в свою границу
                        elif data[n_z, n_y, n_x] == self.index or data[n_z, n_y, n_x] == -self.index:
                            continue
                        # попали в гранчное состояние:
                        #   - попали в другой регион
                        #   - дотянулись до чужой границы
                        # говорим что текущая позиция - граница этой клетки если она уже не обозначена
                        else:
                            if data[p_z, p_y, p_x] != -self.index:
                                data[p_z, p_y, p_x] = -self.index
                            else:
                                continue

            if repit_check_now_point:
                next_iteration_work_points.append(Vector(p_x, p_y, p_z, dtype=int))

        return next_iteration_work_points
    '''

    def ExpansionOfRegion(self, data):
        assert len(data.shape) == 3 or data.shape[3] == 1, "Данные для алгоритма разрастания должны иметь 3 оси и 1 канал."
        index_kernel = np.array([
            (dz, dy, dx)
            for dz in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if not (dz == 0 and dy == 0 and dx == 0)
        ], dtype=int)
        probability_kernel = np.array([
            1 / (abs(dz) + abs(dy) + abs(dx))
            for dz in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if not (dz == 0 and dy == 0 and dx == 0)
        ])

        if len(self.work_points) > 0:
            self.work_points = np.array(self.work_points, dtype=np.int32)
            new_points = []
            for work_point in self.work_points:
                new_points += ExpansionOfPoint26Dir_v3(data, work_point, self.index, index_kernel, probability_kernel)
                #new_points += ExpansionOfPoint6Dir(data, work_point, self.index, self.params["probability_of_expansion"])

            self.work_points = new_points
            return True
        else:
            return False

    def SetPositionAndAngle(self, pos, angle):
        for org in self.list_of_organells:
            org.SetPositionAndAngle(pos, angle)


    def PostExpansionMembrance(self, cell_fields, mask_for_shift_boundary=None):
        distance_cell_shift_range = self.params["distance_cell_shift_range"]
        membrane_thickness = self.params["membrane_thickness"]

        # Делаем отслоение мембраны по маске
        size_of_shift = get_rand_int(distance_cell_shift_range)
        if size_of_shift > 0:
            shift_boundary(cell_fields, self.index, size_of_shift, mask_for_shift_boundary)

        # Делаем утолщение мембраны
        added_thickness = get_rand_int(membrane_thickness) - 1 # минус один, так как толщину 1 дает алгоритм разрастания
        if added_thickness > 0:
            shift_boundary_with_shell(cell_fields, self.index, added_thickness)

        logger.cell(f"Cell {self.index} shift_boundary complited")


    def Create_PSD(self, cell_fields, axon_indices = [], params={}):
        """
        Создаёт маску локального утолщения мембраны (PSD) на стыке двух клеток.
        Возвращает булеву маску мембран в области PSD и добавляет объект PSD в list_of_organells.

        Параметры:
            cell_fields : 3D numpy array с метками клеток (положительные – внутренность,
                          отрицательные – мембрана, значение = -индекс_клетки)
            axon_indices : список индексов клеток-аксонов (для исключения)
            params : словарь дополнительных параметров (может переопределить self.params)

        Возвращает:
            mask_psd : 3D булева маска, где True – мембранные воксели внутри выбранной области,
                       или None, если не удалось найти подходящую область.
        """
        # Параметры – приоритет у переданного params, иначе из self.params
        gap_area = 10
        radius = get_rand_int(params.get("psd", {"radius": 12}).get("radius", 12))
        radius_with_gap = radius + gap_area                                                                            ########################### PSD
        max_attempt = params.get("psd", {"max_attempt": 2000}).get("max_attempt", 2000)

        # 1. Все координаты мембраны данной клетки (отрицательные значения = -self.index)
        membrane_coords = np.argwhere(cell_fields == -self.index)  # (N, 3) в порядке (z,y,x)
        if membrane_coords.shape[0] == 0:
            logger.cell(f"Нет мембраны для клетки {self.index}")
            return None

        shape = cell_fields.shape  # (Z, Y, X)
        found = False
        center = None

        for attempt in range(max_attempt):
            # 2. Случайная точка на мембране
            idx = np.random.choice(membrane_coords.shape[0])
            center = membrane_coords[idx]  # (z, y, x)

            # 3. Ограниченный подкуб вокруг центра
            z_min = max(0, center[0] - radius_with_gap)
            z_max = min(shape[0], center[0] + radius_with_gap + 1)
            y_min = max(0, center[1] - radius_with_gap)
            y_max = min(shape[1], center[1] + radius_with_gap + 1)
            x_min = max(0, center[2] - radius_with_gap)
            x_max = min(shape[2], center[2] + radius_with_gap + 1)

            Z = np.arange(z_min, z_max)
            Y = np.arange(y_min, y_max)
            X = np.arange(x_min, x_max)
            ZZ, YY, XX = np.meshgrid(Z, Y, X, indexing='ij')
            dist = np.sqrt((ZZ - center[0])**2 + (YY - center[1])**2 + (XX - center[2])**2)
            sphere_mask = dist <= radius_with_gap

            # Полная маска сферы в размере исходного тензора
            #full_mask = np.zeros(shape, dtype=bool)
            #full_mask[z_min:z_max, y_min:y_max, x_min:x_max] = sphere_mask

            # 4. Мембранные значения внутри сферы
            membrane_values = cell_fields[z_min:z_max, y_min:y_max, x_min:x_max]
            membrane_values = np.abs(membrane_values[sphere_mask==True])

            if membrane_values.size == 0:
                continue

            unique_cells = np.unique(np.abs(membrane_values))

            # ПРОВЕРКА: если среди уникальных клеток есть хотя бы один аксон – пропускаем эту точку
            if any(c in axon_indices for c in unique_cells):
                continue

            # 5. Проверка: есть наша клетка и хотя бы один сосед (не аксон)
            other_cells = [self.index] + [c for c in unique_cells if c != self.index and c not in axon_indices]

            if len(other_cells) == 2:
                found = True
                break

        if not found:
            print(f"Не удалось найти PSD-область для клетки {self.index} за {max_attempt} попыток")
            return None

        # 6. Создаём объект PSD и добавляем его в список органелл
        # Передаём cell_fields (для доступа к данным), центр, радиус, параметры и маску сферы
        # (можно также передать mask_psd, если он нужен для быстрого доступа к мембранам)
        psd = PSD(
            cell_fields=cell_fields,
            center=center,
            radius=radius,
            params=params,
            unique_indexes=other_cells
        )
        self.list_of_organells.insert(0, psd) # PSD должны рисоваться раньше везикул.

    def DrawOrganelles(self, data, cell_fields=None):
        for organell in self.list_of_organells:
            if not isinstance(organell, PSD):
                organell.Draw(data)

    def DrawMembrane(self, data, cell_fields=None):
        if cell_fields is not None:
            # РИСОВАНИЕ МЕМБРАН
            membrane_color_param = self.params["membrane_color_param"]                                ######## PARAM
            get_fun_type = get_color_index_fun_by_param(membrane_color_param)
            if get_fun_type == 2:################################################################### задание основного и цвета в меньшем диапазоне для понижения разнообразия цвета мембраны каждой клетки
                main_membrane_color = choise_use_color_by_param(membrane_color_param)
                min_range = membrane_color_param[0] - membrane_color_param[1]
                max_range = membrane_color_param[0] + membrane_color_param[1]
                change_3sigma = min(abs(main_membrane_color - min_range),
                                    abs(max_range - main_membrane_color))
                use_membrane_color_param = (main_membrane_color, change_3sigma)

                draw_data_by_mask_and_random_value(data,
                                                   cell_fields[:,:,:] == -self.index,
                                                   use_membrane_color_param)
            else:
                draw_data_by_mask_and_random_value(data,
                                                   cell_fields[:,:,:] < 0,
                                                   membrane_color_param)

        for organell in self.list_of_organells:
            if isinstance(organell, PSD):
                organell.Draw(data)


    def DrawMembraneMask(self, data, cell_fields):
        draw_data_by_mask_and_random_value(data,
                                           cell_fields[:,:,:] == -self.index,
                                           self.params["membrane_mask_color"])                   ######## PARAM

    def update_draw_config(self, config):
        new_params = config.get("cell", None)
        if new_params is not None:
            update_param(self.params, new_params, "membrane_color_param")

        for organell in self.list_of_organells:
            organell.update_draw_config(config)
