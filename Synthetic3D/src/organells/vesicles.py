import numpy as np

from Synthetic3D.src.organells.abstract_organell import Organell
from Synthetic3D.src.hard.structure.vector import Vector
from Synthetic3D.src.hard.random_params import get_rand_int, get_rand_float, get_bool_rand_probability, color_dim_check, choise_use_color_by_param, get_color_index_fun_by_param
from Synthetic3D.src.utilities.check_of_params import check_param, update_param

from Synthetic3D.src.utilities.logging_config import logger
from Synthetic3D.src.hard.structure.vesicle import get_vesicle_type, get_random_type_of_vesicles_name, AbstractVesicle, CapsuleVesicle

from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

ACCEPTABLE_TYPES_OF_VESICLE_CLUSTER_SHAPES = ["Cuboid", "Ellipsoid"]

from tqdm import tqdm

def make_ball(radius):
    """Создать 3D-массив шара заданного радиуса."""
    size = 2 * radius + 1
    center = radius
    y, x, z = np.ogrid[-center:size-center, -center:size-center, -center:size-center]
    return (x*x + y*y + z*z) <= radius*radius


class Vesicles(Organell):
    def __init__(self, params=None):
        super().__init__()
        if params is not None and "vesicles" in params:
            self.params.update(params["vesicles"])
            if True:
                self.comment = "Vesicles cloud by params"
            else:
                self.comment = "Default one Vesicle"
        else:
            self.comment = "Default Vesicles cloud"
        #print(self.comment)
        self.warnings = self._check_and_set_default_params()
        self.warnings += self._Create()

        #print(self.params)

    def _check_and_set_default_params(self) -> list[str]:
        warning_list = []

        warning_list+=check_param(self.params, "radius_of_vesicle", (3,6))
        warning_list+=check_param(self.params, "probability_of_vesicle_filling", 0.5)
        warning_list+=check_param(self.params, "thickness", 1)
        warning_list+=check_param(self.params, "membrane_color", (255, 0, 0))
        warning_list+=check_param(self.params, "inner_color", (0, 255, 0))

        #warning_list+=check_param(self.params, "number_of_vesicles", (25, 100))
        warning_list+=check_param(self.params, "vesicle_packing_fraction", (0.2, 0.4))

        warning_list+=check_param(self.params, "max_num_of_attempts2cloud", 1000)
        warning_list+=check_param(self.params, "gap_of_vesicules", 1)

        warning_list+=check_param(self.params, "type_of_shape", "Ellipsoid")
        if self.params["type_of_shape"] == "Cuboid":
            warning_list+=check_param(self.params, "cuboud_shape_radius", (75,50,50))
        elif self.params["type_of_shape"] == "Ellipsoid":
            warning_list+=check_param(self.params, "ellipsoid_shape_radius", (90,60,45))
        else:
            msg = f"ERROR! Type of shape of vesicules cloud may be only: {ACCEPTABLE_TYPES_OF_VESICLE_CLUSTER_SHAPES}"
            raise ValueError(msg)

        warning_list+=check_param(self.params, "vesicle_type", get_random_type_of_vesicles_name())

        if self.params["vesicle_type"] == "CAPSULE":
            warning_list+=check_param(self.params, "vesicle_half_capsule_len", (2,9))
            #warning_list+=check_param(self.params, "vesicle_type", get_random_type_of_vesicles_name())


        if len(warning_list) != 0:
            logger.config(f'\tWarning Vesicles!\n{"\n".join(warning_list)}')
            warning_list = ["Warning Vesicle!"] + warning_list
        return warning_list

    def _calculate_number_of_vesicles(self):
        vesicle_packing_fraction = self.params["vesicle_packing_fraction"]
        if isinstance(vesicle_packing_fraction, (int, np.integer)):
            return 1
        elif get_color_index_fun_by_param(vesicle_packing_fraction) == 2 or\
            isinstance(vesicle_packing_fraction, float):
                radius_param = self.params["radius_of_vesicle"]
                if get_color_index_fun_by_param(radius_param) == 2:
                    avg_radius = (radius_param[0] + radius_param[1])/2
                else:
                    avg_radius = radius_param

                if isinstance(vesicle_packing_fraction, float):
                    use_vesicle_packing_fraction = vesicle_packing_fraction
                else:
                    use_vesicle_packing_fraction = np.random.uniform(vesicle_packing_fraction[0], vesicle_packing_fraction[1])
                    self.params["vesicle_packing_fraction"] = use_vesicle_packing_fraction

                # Объём одной сферы
                vesicle_volume = (4.0 / 3.0) * np.pi * (avg_radius ** 3)

                if self.params["vesicle_type"] == "CAPSULE":
                    vesicle_half_capsule_len = self.params["vesicle_half_capsule_len"]
                    mean_len = vesicle_half_capsule_len if isinstance(vesicle_half_capsule_len, (int, float)) else sum(vesicle_half_capsule_len)
                    cylinder_volume = np.pi * (avg_radius ** 2) * mean_len
                    vesicle_volume += cylinder_volume

                # Максимальное число сфер при плотнейшей упаковке
                theoretical_max_count = (self.cloude_volume * 0.7404) / vesicle_volume

                # Желаемое число сфер согласно выбранной доле
                desired_count = use_vesicle_packing_fraction * theoretical_max_count

                # Округляем вниз, так как дробное количество невозможно
                return int(np.floor(desired_count))

        else:
            raise ValueError(f'The function "_calculate_number_of_weights" does not work with the value "{self.params["vesicle_packing_fraction"]}"')


    def update_draw_config(self, config):
        new_params = config.get("vesicles", None)
        if new_params is not None:
            update_param(self.params, new_params, "membrane_color")
            update_param(self.params, new_params, "inner_color")


    def _Create(self):
        warnings_list = []
        self.vesicle_objects = []   # список добавленных объектов Vesicle

        self.VesicleClass = get_vesicle_type(self.params["vesicle_type"])                           # ЭКСПЕРИМЕНТАЛЬНЫЙ ТИП ОДИН ДЛЯ ВСЕГО СКОПЛЕНИЯ

        if self.params["type_of_shape"] == "Cuboid":
            type_of_shape_param = self.params["cuboud_shape_radius"]
            if isinstance(type_of_shape_param, (list, tuple)):
                x_radius = get_rand_int(type_of_shape_param[0])
                y_radius = get_rand_int(type_of_shape_param[1])
                z_radius = get_rand_int(type_of_shape_param[2])
                self.params["cuboud_shape_radius"] = (x_radius, y_radius, z_radius)
                logger.organelle(f'Vesicle cloude size choice Cuboid as half axis {x_radius}, {y_radius} and {z_radius} fo xyz')
            else:
                x_radius, y_radius, z_radius = type_of_shape_param

            self.max_cloud_radius = np.linalg.norm((x_radius,y_radius,z_radius))
            self.cloude_volume = x_radius * y_radius * z_radius * 8 # W*H*D
        elif self.params["type_of_shape"] == "Ellipsoid":
            type_of_shape_param = self.params["ellipsoid_shape_radius"]
            if isinstance(type_of_shape_param, (list, tuple)):
                x_radius = get_rand_int(type_of_shape_param[0])
                y_radius = get_rand_int(type_of_shape_param[1])
                z_radius = get_rand_int(type_of_shape_param[2])
                self.params["ellipsoid_shape_radius"] = (x_radius, y_radius, z_radius)
                logger.organelle(f'Vesicle cloude size choice Ellipsoid as hals axis {x_radius}, {y_radius} and {z_radius} for xyz')
            else:
                x_radius, y_radius, z_radius = type_of_shape_param

            self.max_cloud_radius = max(x_radius, y_radius, z_radius)
            self.cloude_volume = 4 / 3 * np.pi *  x_radius * y_radius * z_radius# 4/3*pi*rx*ry*rx

        if get_color_index_fun_by_param(self.params["radius_of_vesicle"]) == 2: ################## Установка малой вариативности внутри одного скопления для повышения реализма
            min_r, max_r = self.params["radius_of_vesicle"]
            work_radius = get_rand_int((min_r, max_r))

            avaleble_radiuce_shith = 1   ######################################################################################################################## PARAM !

            min_work_r = max(min_r, work_radius-avaleble_radiuce_shith)
            max_work_r = min(max_r, work_radius+avaleble_radiuce_shith)

            self.params["radius_of_vesicle"] = (min_work_r, max_work_r)

        radius_param = self.params["radius_of_vesicle"]
        self.max_radius = radius_param if isinstance(radius_param, int) else max(radius_param)
        self.desired_vesicles_count = self._calculate_number_of_vesicles()

        logger.organelle(f'Vesicle desired count {self.desired_vesicles_count} and radius {self.params["radius_of_vesicle"]} with packing fraction {self.params["vesicle_packing_fraction"]}')

        if isinstance(self.params["thickness"], (float, int)):
            self.max_thickness = self.params["thickness"]
        else:
            self.max_thickness = max(self.params["thickness"])

        if self.desired_vesicles_count == 1:
            warnings_list = self.CreateAlone()
        else:
            warnings_list = self.CreateCloude()
        return warnings_list

    def CreateAlone(self):
        ######################################################################################################################## PARAM !
        radius_of_vesicle_param = self.params["radius_of_vesicle"]
        thickness_of_vesicle_param = self.params["thickness"]
        dop_param = self.params["vesicle_half_capsule_len"] if self.params["vesicle_type"] == "CAPSULE" else None

        vesicle = self.VesicleClass(get_rand_int(radius_of_vesicle_param),
                                    get_rand_int(thickness_of_vesicle_param),
                                    get_rand_int(dop_param) if dop_param is not None else None)

        add_point_list = vesicle.point_generation(Vector(0, 0, 0))

        start_index_val = len(self.shell.frame_points)
        self.shell.add_frame_point_list(add_point_list)
        vesicle.indices = [i + start_index_val for i in range(len(add_point_list))]

        self.vesicle_objects.append(vesicle)
        ######################################################################################################################## PARAM !
        self.filling_list = [get_bool_rand_probability(self.params["probability_of_vesicle_filling"])]
        self._CalculateViewData()
        return []

    def CreateCloude(self):
        warnings_list = []
        num_point = self.desired_vesicles_count

        ######################################################################################################################## PARAM !
        radius_of_vesicle_param = self.params["radius_of_vesicle"]
        thickness_of_vesicle_param = self.params["thickness"]
        dop_param = self.params["vesicle_half_capsule_len"] if self.params["vesicle_type"] == "CAPSULE" else None

        vesicle = self.VesicleClass(get_rand_int(radius_of_vesicle_param),
                                    get_rand_int(thickness_of_vesicle_param),
                                    get_rand_int(dop_param) if dop_param is not None else None)

        new_points = vesicle.point_generation(self.GenNewPoint())

        start_index_val = len(self.shell.frame_points)
        self.shell.add_frame_point_list(new_points)
        vesicle.indices = [i + start_index_val for i in range(len(new_points))]
        self.vesicle_objects.append(vesicle)

        ######################################################################################################################## PARAM !
        probability_of_vesicle_filling = self.params["probability_of_vesicle_filling"]
        self.filling_list = [get_bool_rand_probability(probability_of_vesicle_filling)]

        ######################################################################################################################## PARAM !
        max_num_of_attempts2cloud = self.params["max_num_of_attempts2cloud"]

        # Создание облака точек
        miss_point_counter = 0
        self.global_miss_point_counter = 0
        for i in tqdm(range(num_point-1), desc="Добавление везикул в скопление"):
            attempt_counter = 0

            new_vesicle = self.VesicleClass(get_rand_int(radius_of_vesicle_param),
                                            get_rand_int(thickness_of_vesicle_param),
                                            get_rand_int(dop_param) if dop_param is not None else None)
            new_points = new_vesicle.point_generation(self.GenNewPoint())

            while self.CheckIntersection(new_vesicle, new_points) and\
                  attempt_counter < max_num_of_attempts2cloud:
                new_vesicle = self.VesicleClass(get_rand_int(radius_of_vesicle_param),
                                                get_rand_int(thickness_of_vesicle_param),
                                                get_rand_int(dop_param) if dop_param is not None else None)
                new_points = new_vesicle.point_generation(self.GenNewPoint())

                attempt_counter += 1

            if attempt_counter == max_num_of_attempts2cloud:
                miss_point_counter+=1
            else:
                start_index_val = len(self.shell.frame_points)
                self.shell.add_frame_point_list(new_points)
                new_vesicle.indices = [i + start_index_val for i in range(len(new_points))]
                self.vesicle_objects.append(new_vesicle)
                self.filling_list.append(get_bool_rand_probability(probability_of_vesicle_filling))

            self.global_miss_point_counter += attempt_counter

        if miss_point_counter != 0:
            warning_mesg = 'Class "Vesicles". '+\
                           f'Failed to add {miss_point_counter} point of {num_point} to cluster. '+\
                           f'Maximum number of attempts reached {max_num_of_attempts2cloud}.'
            warnings_list.append("WARNING! " + warning_mesg)
            logger.organelle(warning_mesg)
        warnings_list.append(f"Mean of error attempt: {self.global_miss_point_counter/num_point}")
        self._CalculateViewData()
        return warnings_list

    def CheckIntersection(self, new_vesicle, new_points):
        ######################################################################################################################## PARAM !
        gap_of_vesicules = self.params["gap_of_vesicules"]

        for cloude_ves in self.vesicle_objects:
            cloude_ves_pos = cloude_ves.get_coords(self.shell.frame_points)

            if AbstractVesicle.check_intersection(new_vesicle,
                                                  new_points,
                                                  cloude_ves,
                                                  cloude_ves_pos,
                                                  gap_of_vesicules):
                return True
        return False

    def GenNewPoint(self):
        # тут правила генерации внутри заданной формы. Самая простая - кубоид

        ######################################################################################################################## PARAM !
        type_of_shape = self.params["type_of_shape"]

        if type_of_shape == "Cuboid":
            ######################################################################################################################## PARAM !
            x_radius, y_radius, z_radius = self.params["cuboud_shape_radius"]

            x = np.random.randint(-x_radius, x_radius+1)
            y = np.random.randint(-y_radius, y_radius+1)
            z = np.random.randint(-z_radius, z_radius+1)

            return Vector(x, y, z)

        if type_of_shape == "Ellipsoid":
            # Параметры радиусов по осям эллипсоида
            x_radius, y_radius, z_radius = self.params["ellipsoid_shape_radius"]

            # Генерация точки внутри единичного эллипсоида (x^2/a^2 + y^2/b^2 + z^2/c^2 <= 1)
            attempt_gen_counter = 0
            max_attempt_gen_counter = 10000
            while attempt_gen_counter < max_attempt_gen_counter:
                x = np.random.randint(-x_radius, x_radius+1)
                y = np.random.randint(-y_radius, y_radius+1)
                z = np.random.randint(-z_radius, z_radius+1)
                # Проверка, внутри ли точка эллипсоида
                if (x / x_radius) ** 2 + (y / y_radius) ** 2 + (z / z_radius) ** 2 <= 1:
                    return Vector(x, y, z)

                attempt_gen_counter+=1

            msg = "ERROR! Error in the formula for generating points inside an ellipse"
            raise RuntimeError(msg)

        #elif
        else:
            msg = f"ERROR! Type of shape of vesicules cloud may be only: {ACCEPTABLE_TYPES_OF_VESICLE_CLUSTER_SHAPES}"
            raise ValueError(msg)


    ##################################################################################################################### не реализованна

    def Draw(self, data):
        self.view_shell.transform_coords2int()

        membrane_color_param = self.params["membrane_color"]
        get_fun_type = get_color_index_fun_by_param(membrane_color_param)
        # задание основного и цвета в диапазоне для снижения разброса интенсивностей в одном скоплении
        if get_fun_type == 2:
            main_interior_color = choise_use_color_by_param(membrane_color_param)
            min_range = membrane_color_param[0] - membrane_color_param[1]
            max_range = membrane_color_param[0] + membrane_color_param[1]
            change_3sigma = min(abs(main_interior_color - min_range),
                                abs(max_range - main_interior_color))
            use_membrane_color_param = (main_interior_color, change_3sigma)
        else:
            use_membrane_color_param = membrane_color_param


        for i, vesicle in enumerate(self.vesicle_objects):
            work_membrane_color = color_dim_check(choise_use_color_by_param(use_membrane_color_param), data.shape)
            if self.filling_list[i]:
                work_inner_color = color_dim_check(choise_use_color_by_param(self.params["inner_color"]), data.shape)
                vesicle.DrawOneVesicle(data, work_membrane_color, work_inner_color, True, self.view_shell.get_frames())
            else:
                vesicle.DrawOneVesicle(data, work_membrane_color, None, False, self.view_shell.get_frames())

    def DrawMask(self, mask_data, color=None):
        self.view_shell.transform_coords2int()

        ######################################################################################################################## PARAM !
        work_color = self.params["mask_color"] if color is None else color
        work_color = color_dim_check(work_color, mask_data.shape)

        for vesicle in self.vesicle_objects:
            vesicle.DrawOneVesicleMask(mask_data, work_color, self.view_shell.get_frames())

    def DrawArea(self, cell_data, color) -> list[Vector]:
        self.view_shell.transform_coords2int()

        # 1. Рисование исходных везикул
        for vesicle in self.vesicle_objects:
            vesicle.DrawOneVesicleMask(cell_data, color, self.view_shell.get_frames())

        # 2. Морфологическое закрытие: 20 дилатаций + 20 эрозий ядром 3×3×3
        mask = (cell_data == color)                    # бинарная маска объекта
        struct = generate_binary_structure(3, 1)

        # Расширение (заполняет промежутки между везикулами, сглаживает впадины)
        dilated = binary_dilation(mask, structure=struct, iterations=40)
        # Сужение (восстанавливает размер, сглаживая выступы)
        closed_mask = binary_erosion(dilated, structure=struct, iterations=40, border_value=True)
        # Не пересекаем чужую область
        free_mask = closed_mask & (cell_data == 0)
        # Обновляем тензор: все воксели внутри закрытой маски получают цвет объекта
        cell_data[free_mask] = color

        # 3. Вычисление граничных вокселей (поверхность)
        transform_mask = (cell_data == color)
        eroded = binary_erosion(transform_mask, structure=struct, iterations=1)
        boundary_mask = transform_mask & ~eroded          # объект минус его эрозия

        coords = np.argwhere(boundary_mask)            # массив (N, 3) координат
        new_pos_list = [Vector(x, y, z) for z, y, x in coords]
        return new_pos_list