from abc import ABC, abstractmethod
import numpy as np
import random

from Synthetic3D.src.hard.drawing_and_filliing.draw_sphere import draw_small_sphere
from Synthetic3D.src.hard.drawing_and_filliing.fill_sphere import fill_small_sphere, fill_sphere_pyvista

from Synthetic3D.src.hard.dictance import min_distance_between_segments, distance_point_to_segment
from Synthetic3D.src.hard.structure.vector import Vector
from Synthetic3D.src.hard.random_params import get_rand_int, random_unit_vector

from Synthetic3D.src.hard.drawing_and_filliing.draw_cylinder import fill_small_capsule, fill_capsule_vista


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

class SphereVesicle(AbstractVesicle):
    def __init__(self, radius, thickness, param=None, param2=None):
        super().__init__(radius, thickness)

    def point_generation(self, point):
        return [point]

    def DrawOneVesicle(self, data, membrane_color, inner_color, fill_probability, shell):
        frame_point = self.get_coords(shell)[0]
        if fill_probability:
            fill_sphere_pyvista(data,
                                frame_point,
                                self.radius + self.thickness,
                                membrane_color)
            fill_sphere_pyvista(data,
                                frame_point,
                                self.radius,
                                inner_color)
        else:
            fill_sphere_pyvista(data,
                                frame_point,
                                self.radius+self.thickness,
                                membrane_color)

    def DrawOneVesicleMask(self, mask_data, color, shell):
        frame_point = self.get_coords(shell)[0]

        fill_sphere_pyvista(mask_data,
                          frame_point,
                          self.radius + self.thickness,
                          color)

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

        fill_capsule_vista(data, a, b-a, self.radius+self.thickness, membrane_color)
        if fill_probability:
            fill_capsule_vista(data, a, b-a, self.radius, inner_color)

    def DrawOneVesicleMask(self, mask_data, color, shell):
        a, b = self.get_coords(shell)[:2]
        fill_capsule_vista(mask_data, a, b-a, self.radius+self.thickness, color)

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


