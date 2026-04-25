import datetime
import json
import os.path

import numpy as np
import random
from Synthetic3D.src.organells.empty_organelle import EmptyOrganelle
from Synthetic3D.src.organells.vesicles import Vesicles
from Synthetic3D.src.organells.mitohondrion import Mitohondrion
from Synthetic3D.src.container.cell import Cell
from Synthetic3D.src.hard.random_params import get_rand_int, choise_use_color_by_param, color_dim_check
from Synthetic3D.src.hard.structure.vector import Vector
from Synthetic3D.src.container.inrersections_rules import CheckTwoOrganells

from Synthetic3D.src.hard.noise_and_blur import add_internal_structures, CreatePossionNoise, AddGaussianBlur

from Synthetic3D.src.utilities.check_of_params import check_param
from Synthetic3D.src.hard.write_data import write_dataset, write_datamask

from Synthetic3D.src.utilities.logging_config import logger, set_log_file_path, log_execution
from Synthetic3D.src.utilities.check_to_json import to_json_format

class MainField:
    def __init__(self, param:dict={}):
        self.data = None
        self.masks = [] # num channel = num target or make list of masks ?

        self.cell_list = []

        # instance labels of cells
        self.cell_fields = None

        self.params = param
        self.warnings = self._check_and_set_default_params()

    def _check_and_set_default_params(self) -> list[str]:
        warning_list = []
        warning_list+=check_param(self.params, "shape_of_data", (256, 256, 256)) #(512, 512, 512))
        warning_list+=check_param(self.params, "channel_num", 3)
        warning_list+=check_param(self.params, "dict_of_organells",  {"Mitohondrion":0,
                                                                      "Vesicles": 1,
                                                                      "Membranes": 2})

        warning_list += check_param(self.params, "max_count_of_organells", {"Mitohondrion": 4,
                                                                                                        "Vesicles": 3,
                                                                                                        "EmptyCell": 5})

        warning_list+=check_param(self.params, "membrane_color", (0, 127, 0))
        warning_list+=check_param(self.params, "membrane_mask_color", 255)
        warning_list+=check_param(self.params, "poisson_noise", 15)
        warning_list+=check_param(self.params, "max_count_attempt_for_adding_cell", 10000)

        warning_list+=check_param(self.params, "backgraund",  {
                                                               "background_color": 185,
                                                               "background_noise_color": (135, 150),
                                                               "density": 0.001,
                                                               "gaussian_radius":3,
                                                               "gaussian_sigma": 5
                                                               })

        warning_list += check_param(self.params, "generation_offset_range_from_the_edge", (0, 0, 0))
        warning_list += check_param(self.params, "generation_rotation_range", ((0, 359),(0, 359),(0, 359)))

        warning_list += check_param(self.params, "gaussian_blur_radius", 5)
        warning_list += check_param(self.params, "gaussian_blur_sigma", 2)

        warning_list += check_param(self.params, "mask_blur_threshold", 128)

        warning_list+=check_param(self.params,
                                  "save_dataset_dir",
                                  f"datasets/{datetime.datetime.now().strftime("dataset_%Y_%m_%d__%H_%M_%S")}")

        if len(warning_list) != 0:
            logger.config(f'Main check params\n{"\n".join(warning_list)}')
            warning_list = ["Warning MainField!"] + warning_list
        return warning_list

    ################################################################################################################################################################### НУЖНА ДЛЯ ОТЛАДКИ
    def DrawMembranes(self):
        if self.cell_fields is None:
            raise RuntimeError("Мембраны ещё не считались! Для получения мембран нужно запустить алгоритм разрастания регионов!")

        view_data = np.zeros(self.cell_fields.shape + (3,), dtype=np.uint8)
        ################################################################################### PARAM
        membrane_color = color_dim_check(choise_use_color_by_param(self.params["membrane_color"]), 3)
        ################################################################################### LOGPARAM
        #logger.main(f"\used_membrane_color = {membrane_color}")
        self.params["used_membrane_color"] = to_json_format(membrane_color)

        view_data[self.cell_fields[:,:,:] < 0] = membrane_color
        return view_data

    #################################################################################################################### Нуждается в ускорении!
    @log_execution(level="main")
    def CreateDataBackGround(self):
        ################################################################################################################## PARAM
        backgraund_param = self.params["backgraund"]

        density = backgraund_param.get("density", 0.001)
        nodule_color = backgraund_param.get("background_noise_color", (125, 160))
        background_color = backgraund_param.get("background_color", 200)
        background_color = color_dim_check(choise_use_color_by_param(background_color), self.data.shape)
        ################################################################################### LOGPARAM
        #logger.main(f"\tused_background_color = {background_color}")
        self.params["used_background_color"] = to_json_format(background_color)

        gaussian_radius = backgraund_param.get("gaussian_radius", 2)
        gaussian_sigma = backgraund_param.get("gaussian_sigma", 2)

        cycle_range = backgraund_param.get("cycle_range", (2, 4))
        cylinder_length_range = backgraund_param.get("cylinder_length_range", (10.0, 40.0))
        cylinder_radius_range = backgraund_param.get("cylinder_radius_range", (2, 4))

        self.data = add_internal_structures(
                                            self.data,
                                            density=density,  # плотность вкраплений (вероятность добавления на воксель)
                                            nodule_color=nodule_color,  # цвет вкраплений
                                            background_color=background_color,  # цвет фона
                                            shape_type='both',  # 'sphere', 'cylinder', или 'both'
                                            gaussian_radius=gaussian_radius,
                                            gaussian_sigma=gaussian_sigma,  # параметр размытия
                                            cycle_range = cycle_range,
                                            cylinder_length_range = cylinder_length_range,
                                            cylinder_radius_range = cylinder_radius_range)

        #print("WARNING!!! CreateDataBackGround no implemented!")

    @log_execution(level="main")
    def CreateData(self):

        ################################################################################################################## PARAM
        d, h, w = self.params["shape_of_data"]
        ################################################################################################################## PARAM
        c = self.params["channel_num"]

        shape_of_data = (d, h, w, c)
        self.data = np.zeros(shape_of_data, np.uint8)

        ################################################################################################################## PARAM
        for i in range(len(self.params["dict_of_organells"])):
            self.masks.append(np.zeros(shape_of_data[:3], np.uint8))

        #self.CreateDataBackGround()

    @log_execution(level="main")
    def AddBlur(self):
        gaussian_blur_radius = self.params["gaussian_blur_radius"]
        gaussian_blur_sigma  = self.params["gaussian_blur_sigma"]

        data = np.clip(AddGaussianBlur(self.data.astype(int), gaussian_blur_radius, gaussian_blur_sigma),
                       0,
                       255)

        # расширение/подрезка масок в зависимости от порога
        for i in range(len(self.masks)):
            mask = self.masks[i]
            blur_mask = AddGaussianBlur(mask.astype(int), gaussian_blur_radius, gaussian_blur_sigma)
            blur_mask[blur_mask[:,:,:] < self.params["mask_blur_threshold"]] = 0
            blur_mask[blur_mask[:,:,:] > 0] = 255
            self.masks[i] = blur_mask.astype(np.uint8)
        self.data = data.astype(np.uint8)

    @log_execution(level="main")
    def AddNoise(self):
        ############################################################################################################ PARAM
        possion_noise = self.params["poisson_noise"]

        gray_noise = CreatePossionNoise(self.data.shape[:3], possion_noise)
        data = self.data.astype(int) + gray_noise.repeat(self.params.get("channel_num", 1), -1)
        data = np.clip(data, 0, 255)
        self.data = data.astype(np.uint8)

    def ExpansionOfRegionsIter(self, arr_of_num_work_points): ############################################################## вынесено чтобы смотреть между итерациями
        repit_flag = False
        random.shuffle(self.cell_list)  # перемешивание для минимизации приоритетоности
        for cell in self.cell_list:
            if cell.ExpansionOfRegion(self.cell_fields):
                repit_flag = True
                arr_of_num_work_points[cell.index - 1] = len(cell.work_points)
        return repit_flag

    @log_execution(level="main")
    def ExpansionOfRegions(self, repit_flag=True):
        self.cell_fields = np.zeros(self.data.shape[:3], np.int64) # можено сделать и побольше

        for cell in self.cell_list:
            cell.DrawKernelOfArea(self.cell_fields)

        iteration_counter = 0
        summ_of_work_point = 0

        arr_of_num_work_points = np.zeros(len(self.cell_list))
        while repit_flag:
            repit_flag = self.ExpansionOfRegionsIter(arr_of_num_work_points)

            iteration_counter += 1
            print(
                f"{iteration_counter}-я итерация, num_of_points {arr_of_num_work_points}, work_of_points {arr_of_num_work_points.sum()} и {summ_of_work_point} of {self.data.shape[0] * self.data.shape[1] * self.data.shape[2]}")

            summ_of_work_point += arr_of_num_work_points.sum()

        print(f"ExpansionOfRegions completed in {iteration_counter} iterations")

    ###################################################################################################################################### переделать функцию
    def DrawMembransMask(self):
        ################################################################################### PARAM
        membrane_mask_color = choise_use_color_by_param(self.params["membrane_mask_color"])
        ################################################################################### LOGPARAM
        self.params["used_membrane_mask_color"] = to_json_format(membrane_mask_color)
        #logger.main(f"\tused_membrane_mask_color = {membrane_mask_color}")

        ################################################################################### PARAM
        self.masks[self.params["dict_of_organells"]["Membranes"]][self.cell_fields[:,:,:] < 0] = membrane_mask_color # подумать как модифицировать на выбор семантик - инстенс

    @log_execution(level="main")
    def DrawMasks(self):
        for cell in self.cell_list:
            for organell in cell.list_of_organells:
                if isinstance(organell, Vesicles):
                    organell.DrawMask(self.masks[self.params["dict_of_organells"]["Vesicles"]])
                elif isinstance(organell, EmptyOrganelle):
                    continue # не отображается
                elif isinstance(organell, Mitohondrion):
                    organell.DrawMask(self.masks[self.params["dict_of_organells"]["Mitohondrion"]])
                else:
                    raise ValueError(f"This organelle type {type(organell)} is not supported yet!")

        if self.cell_fields is not None:
            self.DrawMembransMask()

    @log_execution(level="main")
    def DrawDataset(self):
        for cell in self.cell_list:
            for organell in cell.list_of_organells:
                organell.Draw(self.data)

        if self.cell_fields is not None:
            ################################################################################### PARAM
            membrane_color = color_dim_check(choise_use_color_by_param(self.params["membrane_color"]), self.data.shape)
            ################################################################################### LOGPARAM
            self.params["used_membrane_color"] = to_json_format(membrane_color)
            self.data[self.cell_fields[:,:,:] < 0] = membrane_color

    @log_execution(level="main")
    def WriteDataset(self, list_of_str=None):
        path_to_save = self.params["save_dataset_dir"]
        write_dataset(self.data, path_to_save, "original", 0)

        for class_name, index_dataset in self.params["dict_of_organells"].items():
            write_datamask(self.masks[index_dataset], path_to_save, class_name.lower())

        self.params["logs"] = list_of_str
        set_log_file_path(os.path.join(path_to_save, "generate.log"))

        with open(os.path.join(path_to_save, 'dataset_config.json'), 'w', encoding='utf-8') as f:
            json.dump(self.params, f, indent=4, ensure_ascii=False)

    def SetNewPosition(self, cell):
        ################################################################################################################## PARAM
        d, h, w = self.params["shape_of_data"]

        ################################################################################################################## PARAM
        ################################################################################################################## PARAM
        overlap_x = self.params["generation_offset_range_from_the_edge"][0]
        overlap_y = self.params["generation_offset_range_from_the_edge"][1]
        overlap_z = self.params["generation_offset_range_from_the_edge"][2]

        cell.SetPositionAndAngle(Vector(get_rand_int((overlap_x, w-1-overlap_x)),
                                              get_rand_int((overlap_y, h-1-overlap_y)),
                                              get_rand_int((overlap_z, d-1-overlap_z))),
                                 Vector(get_rand_int(self.params["generation_rotation_range"][0]),
                                              get_rand_int(self.params["generation_rotation_range"][1]),
                                              get_rand_int(self.params["generation_rotation_range"][2])))

        return cell

    def CheckIntersections(self, newCell: Cell):
        for cell in self.cell_list:
            for organel in cell.list_of_organells:
                for new_organell in newCell.list_of_organells:
                    if CheckTwoOrganells(organel, new_organell):
                        return True
        return False

    def AddNewCell(self, cell:Cell):

        count_attempt = 0
        ################################################################################### PARAM
        while self.CheckIntersections(cell) and count_attempt < self.params["max_count_attempt_for_adding_cell"]:
            self.SetNewPosition(cell)
            count_attempt += 1

        ################################################################################### PARAM
        if count_attempt == self.params["max_count_attempt_for_adding_cell"]:
            print(f"НЕУДАЛОСЬ ДОБАВИТЬ ОРГАНЕЛЛУ ЗА {self.params["max_count_attempt_for_adding_cell"]} ПОПЫТОК")
        else:
            self.cell_list.append(cell)

    def CreateAndAddCell(self):
        i = np.random.randint(3)
        if i == 0: # if 0 - Empty else Vesicles
            new_organelle = EmptyOrganelle
        elif i == 1:
            new_organelle = Vesicles
        else:
            new_organelle = Mitohondrion

        new_index = len(self.cell_list)+1
        new_cell = Cell(new_index, [new_organelle(self.params)], params=self.params)
        self.SetNewPosition(new_cell)
        self.AddNewCell(new_cell)

    @log_execution(level="main")
    def AddCellsByConfig(self):
        max_count_cell_config = self.params["max_count_of_organells"]
        max_count_of_mito = max_count_cell_config.get("Mitohondrion", 0)
        max_count_of_ves = max_count_cell_config.get("Vesicles", 0)
        max_count_of_empty = max_count_cell_config.get("EmptyCell", 0)

        list_of_organells = []
        if max_count_of_mito > 0:
            list_of_organells.append([max_count_of_mito, Mitohondrion])
        if max_count_of_ves > 0:
            list_of_organells.append([max_count_of_ves, Vesicles])
        if max_count_of_empty > 0:
            list_of_organells.append([max_count_of_empty, EmptyOrganelle])

        while len(list_of_organells) > 0:
            choise_index = np.random.randint(len(list_of_organells))

            max_count, new_organelle = list_of_organells[choise_index]
            # индекс 0 зарезервирован под пустоту
            new_index = len(self.cell_list) + 1
            new_cell = Cell(new_index, [new_organelle(self.params)], params=self.params)
            self.SetNewPosition(new_cell)
            self.AddNewCell(new_cell)

            if max_count > 1:
                list_of_organells[choise_index][0] -= 1
            else:
                list_of_organells.pop(choise_index)
