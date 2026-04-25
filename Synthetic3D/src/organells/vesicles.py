import numpy as np
from Synthetic3D.src.hard.drawing_and_filliing.draw_sphere import draw_small_sphere
from Synthetic3D.src.hard.drawing_and_filliing.fill_sphere import fill_small_sphere

from Synthetic3D.src.organells.abstract_organell import Organell
from Synthetic3D.src.hard.structure.vector import Vector
from Synthetic3D.src.hard.random_params import get_rand_int, get_bool_rand_probability, color_dim_check, choise_use_color_by_param
from Synthetic3D.src.utilities.check_of_params import check_param

from Synthetic3D.src.utilities.logging_config import logger


ACCEPTABLE_TYPES_OF_VESICLE_CLUSTER_SHAPES = ["Cuboid"]


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
        warning_list+=check_param(self.params, "thickness", 0.5)
        warning_list+=check_param(self.params, "color", (255, 0, 0))
        warning_list+=check_param(self.params, "color_inner", (0, 255, 0))

        warning_list+=check_param(self.params, "number_of_vesicles", (25, 100))
        warning_list+=check_param(self.params, "max_num_of_attempts2cloud", 1000)
        warning_list+=check_param(self.params, "gap_of_vesicules", 1)

        warning_list+=check_param(self.params, "type_of_shape", "Cuboid")
        warning_list+=check_param(self.params, "cuboud_shape_radius", (50,25,25))

        if len(warning_list) != 0:
            logger.config(f'\tWarning Vesicles!\n{"\n".join(warning_list)}')
            warning_list = ["Warning Vesicle!"] + warning_list
        return warning_list

    def _Create(self):
        warnings_list = []
        # radius_list нужен для расчета пересечений
        self.radius_list = []
        if isinstance(self.params["number_of_vesicles"], int) and self.params["number_of_vesicles"] == 1:
            warnings_list = self.CreateAlone()
        else:
            warnings_list = self.CreateCloude()
        return warnings_list

    def CreateAlone(self):
        print("WARNING!!! CREATE TEST ONE VESICLES")
        self.shell.add_frame_point(Vector(0, 0, 0))
        ######################################################################################################################## PARAM !
        self.radius_list = [get_rand_int(self.params["radius_of_vesicle"])]
        ######################################################################################################################## PARAM !
        self.filling_list = [get_bool_rand_probability(self.params["probability_of_vesicle_filling"])]
        self._CalculateViewData()
        return []


    ##################################### ТЕСТОВАЯ РЕАЛИЗАЦИЯ №№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
    def CreateCloude(self):
        print("WARNING!!! CREATE TEST ONE VESICLES CLOUDE")

        warnings_list = []

        ######################################################################################################################## PARAM !
        num_point = get_rand_int(self.params["number_of_vesicles"])

        ######################################################################################################################## PARAM !
        radius_of_vesicle = self.params["radius_of_vesicle"]

        self.shell.add_frame_point(self.GenNewPoint())
        self.radius_list = [get_rand_int(radius_of_vesicle)]

        ######################################################################################################################## PARAM !
        probability_of_vesicle_filling = self.params["probability_of_vesicle_filling"]
        self.filling_list = [get_bool_rand_probability(probability_of_vesicle_filling)]

        ######################################################################################################################## PARAM !
        max_num_of_attempts2cloud = self.params["max_num_of_attempts2cloud"]

        # Создание облака точек
        miss_point_counter = 0
        self.global_miss_point_counter = 0
        for i in range(num_point-1):
            attempt_counter = 0

            new_point = self.GenNewPoint()
            new_radius = get_rand_int(radius_of_vesicle)

            while self.CheckIntersection(new_point, new_radius) and\
                  attempt_counter < max_num_of_attempts2cloud:
                new_point = self.GenNewPoint()
                new_radius = get_rand_int(radius_of_vesicle)
                attempt_counter += 1

            if attempt_counter == max_num_of_attempts2cloud:
                miss_point_counter+=1
            else:
                self.shell.add_frame_point(new_point)
                self.radius_list.append(new_radius)
                self.filling_list.append(get_bool_rand_probability(probability_of_vesicle_filling))

            self.global_miss_point_counter += attempt_counter

        if miss_point_counter != 0:
            warnings_list.append('WARNING! Class "Vesicles". '+\
                                 f'Failed to add {miss_point_counter} point of {num_point} to cluster. '+\
                                 f'Maximum number of attempts reached {max_num_of_attempts2cloud}.')

        warnings_list.append(f"Mean of error attempt: {self.global_miss_point_counter/num_point}")
        self._CalculateViewData()
        return warnings_list

    def CheckIntersection(self, point, radius):
        ######################################################################################################################## PARAM !
        gap_of_vesicules = self.params["gap_of_vesicules"]

        for i, cloud_point in enumerate(self.shell.get_frames()):
            cloud_radius = self.radius_list[i]
            dist_between_points = np.linalg.norm(cloud_point-point)
            ######################################################################################################################## PARAM !
            if dist_between_points < cloud_radius+radius+gap_of_vesicules+self.params["thickness"]:
                return True
        return False

    def GenNewPoint(self):
        # тут правила генерации внутри заданной формы. Самая простая - кубоид

        ######################################################################################################################## PARAM !
        type_of_shape = self.params["type_of_shape"]

        if type_of_shape == "Cuboid":
            ######################################################################################################################## PARAM !
            x_radius, y_radius, z_radius = self.params["cuboud_shape_radius"]

            x = np.random.randint(-x_radius, x_radius)
            y = np.random.randint(-y_radius, y_radius)
            z = np.random.randint(-z_radius, z_radius)

            return Vector(x, y, z)

        #elif
        else:
            msg = f"ERROR! Type of shape of vesicules cloud may be only: {ACCEPTABLE_TYPES_OF_VESICLE_CLUSTER_SHAPES}"
            raise ValueError(msg)


    ##################################################################################################################### не реализованна

    def Draw(self, data):
        self.view_shell.transform_coords2int()

        for i, frame_point in enumerate(self.view_shell.get_frames()):
            work_membrane_color = color_dim_check(choise_use_color_by_param(self.params["color"]), data.shape)
            if self.filling_list[i]:
                work_inner_color = color_dim_check(choise_use_color_by_param(self.params["color_inner"]), data.shape)
                fill_small_sphere(data,
                                  frame_point,
                                  self.radius_list[i],
                                  ######################################################################################################################## PARAM !
                                  work_inner_color)
                draw_small_sphere(data,
                                  frame_point,
                                  self.radius_list[i],
                                  ######################################################################################################################## PARAM !
                                  work_membrane_color,
                                  ######################################################################################################################## PARAM !
                                  self.params["thickness"])
            else:
                fill_small_sphere(data,
                                  frame_point,
                                  ######################################################################################################################## PARAM !
                                  self.radius_list[i]+self.params["thickness"],
                                  ######################################################################################################################## PARAM !
                                  work_membrane_color)

    def DrawMask(self, mask_data, color=None):
        self.view_shell.transform_coords2int()

        ######################################################################################################################## PARAM !
        work_color = self.params["mask_color"] if color is None else color
        work_color = color_dim_check(work_color, mask_data.shape)

        for i, frame_point in enumerate(self.view_shell.get_frames()):
            fill_small_sphere(mask_data,
                              frame_point,
                              self.radius_list[i] + self.params["thickness"],
                              work_color)


    ###################################################################################################################### НЕ ОЧЕНЬ ХОРОШАЯ РЕАЛИЗАЦИЯ
    def DrawArea(self, cell_data, color) -> list[Vector]:
        print("НУЖНО ДОДЕЛАТЬ DrawArea Vesicules")
        pos_list = []

        #work_color = color_dim_check(color, cell_data.shape)

        d, h, w = cell_data.shape[:3]

        for i, frame_point in enumerate(self.view_shell.get_frames()):
            radius = self.radius_list[i] + self.params["thickness"] + 1
            pos = np.round(frame_point).astype(int)

            max_radius_compare = (radius + 0.5) ** 2
            min_radius_compare = (radius - 0.5) ** 2

            for z in range(-radius, radius + 1, 1):
                now_z = pos[2] + z
                if 0 <= now_z < d:
                    radius_sum_z = z ** 2
                    for x in range(-radius, radius + 1, 1):
                        now_x = pos[0] + x
                        if 0 <= now_x < w:
                            radius_sum_zx = radius_sum_z + x ** 2
                            for y in range(-radius, radius + 1, 1):
                                now_y = pos[1] + y
                                if 0 <= now_y < h:
                                    radius_sum_zxy = radius_sum_zx + y ** 2
                                    if radius_sum_zxy <= max_radius_compare:
                                        cell_data[now_z, now_y, now_x] = color
                                        if min_radius_compare <= radius_sum_zxy:
                                            pos_list.append(Vector(now_x, now_y, now_z, dtype=int))
        return pos_list
