import numpy as np
from Synthetic3D.src.organells.abstract_organell import Organell
from Synthetic3D.src.hard.structure.vector import Vector
from Synthetic3D.src.hard.random_params import get_rand_int
from Synthetic3D.src.utilities.check_of_params import check_param

class EmptyOrganelle(Organell):
    def __init__(self, params=None):
        super().__init__()

        if params is not None and "empty_organelle" in params:
            self.params.update(params["empty_organelle"])
            self.comment = "EmptyOrganelle by params"
        else:
            self.comment = "Default EmptyOrganelle"

        self.warnings = self._check_and_set_default_params()
        self.warnings += self._Create()

    def _check_and_set_default_params(self) -> list[str]:
        warning_list = []
        warning_list+=check_param(self.params, "radius", (15, 50))
        if len(warning_list) != 0:
            warning_list = ["Warning EmptyOrganelle!"] + warning_list
        return warning_list

    def _Create(self):
        warnings_list = []
        ################################################################################################# PARAM
        self.radius = get_rand_int(self.params["radius"])
        return warnings_list

    def Draw(self, data): pass # не умеет рисоваться, только создавать маски зерна территории
    def DrawMask(self, mask_data, color): pass # не умеет рисоваться, только создавать маски зерна территории

    def update_draw_config(self, config):
        pass

    ####################################################################################################################### придумать что-то поэффективнее и универсальнее чем копия 2 алгоритмов рисования и заполнения малой сферы
    def DrawArea(self, cell_data, color) -> list[Vector]:
        #work_color = color_dim_check(color, cell_data.shape)

        pos = np.round(self.position).astype(int)
        pos_list = []

        d,h,w = cell_data.shape[:3]

        max_radius_compare = (self.radius + 0.5)**2
        min_radius_compare = (self.radius - 0.5)**2

        for z in range(-self.radius, self.radius+1, 1):
            now_z = pos[2] + z
            if 0 <= now_z < d:
                radius_sum_z = z**2
                for x in range(-self.radius, self.radius + 1, 1):
                    now_x = pos[0] + x
                    if 0 <= now_x < w:
                        radius_sum_zx = radius_sum_z + x**2
                        for y in range(-self.radius, self.radius+1, 1):
                            now_y = pos[1] + y
                            if 0 <= now_y < h:
                                radius_sum_zxy = radius_sum_zx + y**2
                                if radius_sum_zxy <= max_radius_compare:
                                    cell_data[now_z,now_y,now_x] = color
                                    if min_radius_compare <= radius_sum_zxy:
                                        pos_list.append(Vector(now_x, now_y, now_z,dtype=int))
        return pos_list
