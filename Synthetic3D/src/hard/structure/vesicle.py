from abc import ABC, abstractmethod
import numpy as np
import random

from Synthetic3D.src.hard.drawing_and_filliing.draw_sphere import draw_small_sphere
from Synthetic3D.src.hard.drawing_and_filliing.fill_sphere import fill_small_sphere

from Synthetic3D.src.hard.dictance import min_distance_between_segments, distance_point_to_segment
from Synthetic3D.src.hard.structure.vector import Vector
from Synthetic3D.src.hard.random_params import get_rand_int, random_unit_vector

from Synthetic3D.src.hard.drawing_and_filliing.draw_cylinder import fill_small_capsule


ACCEPTABLE_TYPES_OF_VESICLE = ["SPHERE", "CAPSULE"] #"DUMBBELLS"]


class AbstractVesicle(ABC):
    def __init__(self, radius, thickness):
        self.radius = radius
        self.thickness = thickness
        self.indices = None   # будет установлен после добавления в shell

    def get_coords(self, shell):
        """Возвращает координаты опорных точек по индексам (если есть)."""
        if self.indices is None:
            raise RuntimeError("Vesicle not yet added to shell, no indices set.")
        return [shell[i] for i in self.indices]

    def get_bounding_sphere(self, coords):
        """
        Вычисляет центр и радиус охватывающей сферы по переданным координатам опорных точек.
        coords – список numpy-массивов (опорные точки).
        """
        if len(coords) == 1:
            center = coords[0]
            radius = self.radius + self.thickness
        else:
            center = np.mean(coords, axis=0)
            max_dist = max(np.linalg.norm(p - center) for p in coords)
            radius = max_dist + self.radius + self.thickness
        return center, radius

    @staticmethod
    def check_intersection(ves1, coords1, ves2, coords2, gap):
        """
        Проверяет пересечение двух везикул, используя переданные координаты опорных точек.
        Возвращает True, если есть пересечение с учётом зазора gap.
        """
        # Быстрая проверка охватывающих сфер
        c1, r1 = ves1.get_bounding_sphere(coords1)
        c2, r2 = ves2.get_bounding_sphere(coords2)
        if np.linalg.norm(c1 - c2) > r1 + r2 + gap:
            return False
        # Точная проверка в зависимости от типов
        return AbstractVesicle._exact_intersection(ves1, coords1, ves2, coords2, gap)

    @staticmethod
    def _exact_intersection(ves1, coords1, ves2, coords2, gap):
        """Реализует точную проверку пересечения для конкретных типов."""
        # Здесь нужно определить типы везикул и вызвать соответствующие функции.
        # Можно использовать isinstance или паттерн посетитель.
        # Ниже приведён пример для Sphere и Capsule.
        if isinstance(ves1, SphereVesicle) and isinstance(ves2, SphereVesicle):
            p1 = coords1[0]
            p2 = coords2[0]
            return np.linalg.norm(p1 - p2) < (ves1.radius + ves1.thickness + ves2.radius + ves2.thickness + gap)
        elif isinstance(ves1, SphereVesicle) and isinstance(ves2, CapsuleVesicle):
            p = coords1[0]
            a, b = coords2[0], coords2[1]
            dist =  distance_point_to_segment(p, a, b)
            return dist < (ves1.radius + ves1.thickness + ves2.radius + ves2.thickness + gap)
        elif isinstance(ves1, CapsuleVesicle) and isinstance(ves2, SphereVesicle):
            # симметрично
            a, b = coords1[0], coords1[1]
            p = coords2[0]
            dist = distance_point_to_segment(p, a, b)
            return dist < (ves1.radius + ves1.thickness + ves2.radius + ves2.thickness + gap)
        elif isinstance(ves1, CapsuleVesicle) and isinstance(ves2, CapsuleVesicle):
            a1, b1 = coords1[0], coords1[1]
            a2, b2 = coords2[0], coords2[1]
            dist = min_distance_between_segments(a1, b1, a2, b2)
            return dist < (ves1.radius + ves1.thickness + ves2.radius + ves2.thickness + gap)
        else:
            raise NotImplementedError(f"Intersection not implemented for {type(ves1)} and {type(ves2)}")

    @abstractmethod
    def point_generation(self, point):
        pass

    @abstractmethod
    def DrawOneVesicle(self, data, membrane_color, inner_color, fill_probability, shell):
        pass

    @abstractmethod
    def DrawOneVesicleMask(self, mask_data, color, shell):
        pass

    @abstractmethod
    def DrawOneVesicleArea(self, cell_data, color, shell) -> list:
        pass



class SphereVesicle(AbstractVesicle):
    def __init__(self, radius, thickness, param=None, param2=None):
        super().__init__(radius, thickness)

    def point_generation(self, point):
        return [point]

    def DrawOneVesicle(self, data, membrane_color, inner_color, fill_probability, shell):
        frame_point = self.get_coords(shell)[0]
        if fill_probability:
            fill_small_sphere(data,
                              frame_point,
                              self.radius,
                              inner_color)
            draw_small_sphere(data,
                              frame_point,
                              self.radius,
                              membrane_color,
                              self.thickness)
        else:
            fill_small_sphere(data,
                              frame_point,
                              self.radius+self.thickness,
                              membrane_color)

    def DrawOneVesicleMask(self, mask_data, color, shell):
        frame_point = self.get_coords(shell)[0]

        fill_small_sphere(mask_data,
                          frame_point,
                          self.radius + self.thickness,
                          color)

    def DrawOneVesicleArea(self, cell_data, color, shell):
        frame_point = self.get_coords(shell)[0]
        radius = self.radius + self.thickness + 1
        pos = np.round(frame_point).astype(int)

        max_radius_compare = (radius + 0.5) ** 2
        min_radius_compare = (radius - 0.5) ** 2

        d, h, w = cell_data.shape[:3]

        pos_list = []

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


class CapsuleVesicle(AbstractVesicle):
    def __init__(self, radius, thickness, half_distance_param=None, main_dir = None):
        super().__init__(radius, thickness)
        if half_distance_param is None:
            self.half_distance_param = get_rand_int((2,5))
        else:
            self.half_distance_param = half_distance_param
        self.main_dir = main_dir

    def point_generation(self, point):
        if self.main_dir is None:
            rand_vec = random_unit_vector() * get_rand_int(self.half_distance_param)
        else:
            rand_vec = self.main_dir * get_rand_int(self.half_distance_param)

        rand_vec = np.round(rand_vec).astype(int)

        return [point - rand_vec, point + rand_vec]

    def DrawOneVesicle(self, data, membrane_color, inner_color, fill_probability, shell):
        a, b = self.get_coords(shell)[:2]

        fill_small_capsule(data, a, b-a, self.radius+self.thickness, membrane_color)
        if fill_probability:
            fill_small_capsule(data, a, b-a, self.radius, inner_color)

    def DrawOneVesicleMask(self, mask_data, color, shell):
        a, b = self.get_coords(shell)[:2]

        fill_small_capsule(mask_data, a, b-a, self.radius+self.thickness, color)

    def DrawOneVesicleArea(self, cell_data, color, shell):
        """
        Возвращает список точек на поверхности капсулы (толщиной ~1 пиксель)
        для использования в DrawArea.
        """

        pos_list = []
        # Получаем координаты концов капсулы из shell
        a, b = self.get_coords(shell)[:2]  # a и b - numpy массивы
        R_ext = self.radius + self.thickness + 1   # внешний радиус (поверхность мембраны)

        direction = b - a

        fill_small_capsule(cell_data, a, direction, R_ext, color)

        # Направление и длина капсулы
        direction = b - a
        length = np.linalg.norm(direction)
        dir_unit = direction / length

        # Определяем ограничивающий куб для перебора (с запасом)
        margin = int(np.ceil(R_ext)) + 1
        min_x = int(min(a[0], b[0]) - margin)
        max_x = int(max(a[0], b[0]) + margin)
        min_y = int(min(a[1], b[1]) - margin)
        max_y = int(max(a[1], b[1]) + margin)
        min_z = int(min(a[2], b[2]) - margin)
        max_z = int(max(a[2], b[2]) + margin)

        # Ограничиваем размеры массива cell_data (предполагаем порядок z, y, x)
        d, h, w = cell_data.shape[:3]
        min_x = max(min_x, 0)
        max_x = min(max_x, w - 1)
        min_y = max(min_y, 0)
        max_y = min(max_y, h - 1)
        min_z = max(min_z, 0)
        max_z = min(max_z, d - 1)

        # Перебираем все точки в ограничивающем кубе
        for z in range(min_z, max_z + 1):
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    p = np.array([x, y, z], dtype=float)

                    # Вычисляем расстояние от точки до поверхности капсулы
                    # 1. Проекция на ось
                    v = p - a
                    t = np.dot(v, dir_unit)

                    if t < 0:
                        # Ближайшая точка – конец a (сфера)
                        dist_center = np.linalg.norm(p - a)
                        dist_surf = dist_center - R_ext
                    elif t > length:
                        # Ближайшая точка – конец b (сфера)
                        dist_center = np.linalg.norm(p - b)
                        dist_surf = dist_center - R_ext
                    else:
                        # Ближайшая точка на оси цилиндра
                        closest_on_axis = a + t * dir_unit
                        dist_axis = np.linalg.norm(p - closest_on_axis)
                        dist_surf = dist_axis - R_ext

                    # Если точка лежит на поверхности с точностью 0.5 пикселя
                    if abs(dist_surf) <= 0.5:
                        cell_data[z, y, x] = color
                        pos_list.append(Vector(x, y, z, dtype=int))

        return pos_list



def get_vesicle_type(type_of_vesicle):
    if type_of_vesicle in ACCEPTABLE_TYPES_OF_VESICLE:
        if type_of_vesicle == "SPHERE":
            return SphereVesicle
        elif type_of_vesicle == "CAPSULE":
            return CapsuleVesicle
        elif type_of_vesicle == "DUMBBELLS":
            raise NotImplementedError("Форма гантельки ещё не сделана")

    else:
        msg = f"ERROR type of vesicle! Shape can only be {ACCEPTABLE_TYPES_OF_VESICLE}. Received {type_of_vesicle}"
        raise AttributeError(msg)

def get_random_type_of_vesicles_name():
    return random.choice(ACCEPTABLE_TYPES_OF_VESICLE)


