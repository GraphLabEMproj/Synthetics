from Synthetic3D.src.container.cell import Cell
from Synthetic3D.src.organells.empty_organelle import EmptyOrganelle
from Synthetic3D.src.utilities.check_of_params import check_param, update_param
from Synthetic3D.src.utilities.logging_config import logger

from Synthetic3D.src.container.cell_operation import triple_shift_boundary_with_shell
from Synthetic3D.src.hard.random_params import get_rand_int #, get_color_index_fun_by_param, choise_use_color_by_param
from Synthetic3D.src.hard.drawing_and_filliing.draw_data_by_mask import draw_data_by_mask_and_random_value

import numpy as np


class Axon(Cell):
    def __init__(self, index, list_of_organels=[EmptyOrganelle()], params={}):
        super().__init__(index, list_of_organels)
        self.params = params.get("axon", {})
        self.warnings = self._check_and_set_default_params()

        self.membrane_draw_list = []

    def _check_and_set_default_params(self):
        warning_list = []

        warning_list += check_param(self.params, "membrane_thickness", (5,10))
        warning_list += check_param(self.params, "membrane_color_param", (64, 10))
        warning_list += check_param(self.params, "inner_thickness", (10, 20))
        warning_list += check_param(self.params, "inner_color_param", (95, 20))
        warning_list += check_param(self.params, "input_membrane_thickness", (2,5))
        warning_list += check_param(self.params, "membrane_mask_color", 255)

        if len(warning_list) != 0:
            logger.config(f'\tWarning Axon!\n{"\n".join(warning_list)}')
            warning_list = ["Warning Axon!"] + warning_list
        return warning_list

    def PostExpansionMembrance(self, cell_fields, mask_for_shift_boundary=None):
        logger.cell("create Axon")

        membrane_thickness_param = self.params["membrane_thickness"]                      ######## PARAM
        membrane_thickness = get_rand_int(membrane_thickness_param)

        inner_thickness_param = self.params["inner_thickness"]                            ######## PARAM
        inner_thickness = get_rand_int(inner_thickness_param)

        input_membrane_thickness_param = self.params["input_membrane_thickness"]
        input_membrane_thickness = get_rand_int(input_membrane_thickness_param)

        axon_outer_membrane_mask,\
        axon_inner_mask,\
        axon_input_membrane_mask  = triple_shift_boundary_with_shell(cell_fields,
                                                                     self.index,
                                                                     membrane_thickness,
                                                                     inner_thickness,
                                                                     input_membrane_thickness)

        self.membrane_draw_list.append(axon_outer_membrane_mask)
        self.membrane_draw_list.append(axon_inner_mask)
        self.membrane_draw_list.append(axon_input_membrane_mask)

        logger.cell("Axon created")


    def DrawMembrane(self, data, cell_fields):
        axon_outer_membrane_mask = self.membrane_draw_list[0]
        axon_inner_mask          = self.membrane_draw_list[1]
        axon_input_membrane_mask = self.membrane_draw_list[2]

        if np.any(axon_outer_membrane_mask):
            draw_data_by_mask_and_random_value(data,
                                               axon_outer_membrane_mask,
                                               self.params["membrane_color_param"])         ######## PARAM

            if np.any(axon_inner_mask):
                draw_data_by_mask_and_random_value(data,
                                                   axon_inner_mask,
                                                   self.params["inner_color_param"])         ######## PARAM

                if np.any(axon_input_membrane_mask):
                    draw_data_by_mask_and_random_value(data,
                                                       axon_input_membrane_mask,
                                                       self.params["membrane_color_param"])         ######## PARAM

    def DrawOrganelles(self, data, cell_fields=None):
        pass

    def DrawMembraneMask(self, data, cell_fields):
        axon_outer_membrane_mask = self.membrane_draw_list[0]
        axon_inner_mask          = self.membrane_draw_list[1]
        axon_input_membrane_mask = self.membrane_draw_list[2]

        color_mask_param = self.params["membrane_mask_color"]                        ######## PARAM

        if np.any(axon_outer_membrane_mask):
            draw_data_by_mask_and_random_value(data,
                                               axon_outer_membrane_mask,
                                               color_mask_param)

            if np.any(axon_inner_mask):
                draw_data_by_mask_and_random_value(data,
                                                   axon_inner_mask,
                                                   color_mask_param)

                if np.any(axon_input_membrane_mask):
                    draw_data_by_mask_and_random_value(data,
                                                       axon_input_membrane_mask,
                                                       color_mask_param)

    def update_draw_config(self, config):
        new_params = config.get("axon", None)
        if new_params is not None:
            update_param(self.params, new_params, "membrane_color_param")
            update_param(self.params, new_params, "inner_color_param")

        for organell in self.list_of_organells:
            organell.update_draw_config(config)

