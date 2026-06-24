import datetime
import json
import os.path

import numpy as np
import random
from Synthetic3D.src.organells.empty_organelle import EmptyOrganelle
from Synthetic3D.src.organells.vesicles import Vesicles
from Synthetic3D.src.organells.mitohondrion import Mitohondrion
from Synthetic3D.src.container.cell import Cell
from Synthetic3D.src.hard.random_params import get_rand_int, choise_use_color_by_param, color_dim_check, get_color_index_fun_by_param
from Synthetic3D.src.hard.structure.vector import Vector
from Synthetic3D.src.container.inrersections_rules import CheckTwoOrganells

from Synthetic3D.src.hard.noise_and_blur import add_internal_structures, CreatePossionNoise, AddGaussianBlur
from Synthetic3D.src.hard.drawing_and_filliing.draw_data_by_mask import draw_data_by_mask_and_random_value

from Synthetic3D.src.utilities.check_of_params import check_param
from Synthetic3D.src.hard.write_data import write_dataset, write_datamask

from Synthetic3D.src.utilities.logging_config import logger, set_log_file_path, log_execution
from Synthetic3D.src.utilities.check_to_json import to_json_format

from Synthetic3D.src.container.cell_operation import generate_spheres_mask, shift_boundary_with_shell, shift_boundary

class MainField:
    def __init__(self, param:dict={}):
        self.data = None
        self.masks = [] # num channel = num target or make list of masks ?

        self.cell_list = []

        # instance labels of cells
        self.cell_fields = None

        self.params = param
        self.warnings = self._check_and_set_default_params()

        self.count_organells = 0

    def _check_and_set_default_params(self) -> list[str]:
        warning_list = []
        warning_list+=check_param(self.params, "shape_of_data", (256, 256, 256)) #(512, 512, 512))
        warning_list+=check_param(self.params, "channel_num", 3)
        warning_list+=check_param(self.params, "dict_of_organells",  {"Mitohondrion":0,
                                                                      "Vesicles": 1,
                                                                      "Membranes": 2})

        warning_list += check_param(self.params, "max_count_of_organells", {"Mitohondrion": 4,
                                                                            "Vesicles": 3,
                                                                            "EmptyCell": 5,
                                                                            "AxonShell": 1})

        warning_list+=check_param(self.params, "membrane_color", (0, 127, 0))
        warning_list += check_param(self.params, "distance_cell_shift_range", (1,2))

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

        warning_list += check_param(self.params, "axon_membrane_size", (5,10))
        warning_list += check_param(self.params, "axon_color_membrane_param", (64, 10))

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

    #################################################################################################################### Нуждается в ускорении!
    @log_execution(level="main")
    def CreateDataBackGround(self):
        ################################################################################################################## PARAM
        backgraund_param = self.params["backgraund"]

        density = backgraund_param.get("density", 0.001)
        nodule_color = backgraund_param.get("background_noise_color", (125, 15))
        background_color = backgraund_param.get("background_color", (200, 1))

        gaussian_radius = backgraund_param.get("gaussian_radius", 2)
        gaussian_sigma = backgraund_param.get("gaussian_sigma", 2)

        cycle_range = backgraund_param.get("cycle_range", (2, 3))
        cylinder_length_range = backgraund_param.get("cylinder_length_range", (10.0, 40.0))
        cylinder_radius_range = backgraund_param.get("cylinder_radius_range", (2, 3))

        self.data = add_internal_structures(
                                            self.data,
                                            density=density,  # плотность вкраплений (вероятность добавления на воксель)
                                            nodule_color_param=nodule_color,  # цвет вкраплений
                                            background_color_param=background_color,  # цвет фона
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
            logger.main(
                f"{iteration_counter}-я итерация, num_of_points {arr_of_num_work_points}, work_of_points {arr_of_num_work_points.sum()} и {summ_of_work_point} of {self.data.shape[0] * self.data.shape[1] * self.data.shape[2]}")

            summ_of_work_point += arr_of_num_work_points.sum()

        logger.main(f"ExpansionOfRegions completed in {iteration_counter} iterations")

    def _expanded_mask(self, mask, value):
        pass

    ###################################################################################################################################### переделать функцию
    def DrawMembransMask(self):
        if self.cell_fields is None:
            raise RuntimeError("Мембраны ещё не считались! Для получения мембран нужно запустить алгоритм разрастания регионов!")

        mode = "semantic"
        if mode == "semantic":
            for cell in self.cell_list:
                if cell.index in self.axon_cell_index_list:
                    continue
                else:
                    draw_data_by_mask_and_random_value(self.masks[self.params["dict_of_organells"]["Membranes"]],
                                                       self.cell_fields[:,:,:] == -cell.index,
                                                       self.params["membrane_mask_color"])        ######## PARAM
        elif mode == "instance":
            NotImplementedError("Membrane mode 'instance' no realised")
        else:
            raise NotImplementedError("Now membrane mode only 'semantic'")

    ################################################################################################################################################################### НУЖНА ДЛЯ ОТЛАДКИ
    def DrawMembranesForView(self):
        if self.cell_fields is None:
            raise RuntimeError("Мембраны ещё не считались! Для получения мембран нужно запустить алгоритм разрастания регионов!")

        view_data = np.zeros(self.cell_fields.shape + (3,), dtype=np.uint8)
        return draw_data_by_mask_and_random_value(view_data,
                                                  self.cell_fields[:,:,:] < 0,
                                                  self.params["membrane_mask_color"]) ######## PARAM

    @log_execution(level="main")
    def DrawMasks(self):
        for cell in self.cell_list:
            for organell in cell.list_of_organells:
                if isinstance(organell, Vesicles) and "Vesicles" in self.params["dict_of_organells"].keys():
                    organell.DrawMask(self.masks[self.params["dict_of_organells"]["Vesicles"]])
                elif isinstance(organell, EmptyOrganelle):
                    continue # не отображается
                elif isinstance(organell, Mitohondrion) and "Mitohondrion" in self.params["dict_of_organells"].keys():
                    organell.DrawMask(self.masks[self.params["dict_of_organells"]["Mitohondrion"]])
                else:
                    raise ValueError(f"This organelle type {type(organell)} is not supported yet!")

        if self.cell_fields is not None and "Membranes" in self.params["dict_of_organells"].keys():
            self.DrawMembransMask()

    @log_execution(level="main")
    def DrawDataset(self):
        logger.main("Draw organells")
        for cell in self.cell_list:
            for organell in cell.list_of_organells:
                organell.Draw(self.data)

        if self.cell_fields is not None:
            logger.main("Draw axon and membrans")
            ############################################################################################################################ ЭКСПЕРИМЕНТАЛЬНАЯ ФУНКЦИЯ
            mask_for_shift_boundary = generate_spheres_mask(self.cell_fields.shape,
                                                            density= 0.25,
                                                            radius_range = (4,30),
                                                            )

            ############################################################################################################################################################## AXON PARAM
            logger.main("Calculate shitfing membranes")
            distance_cell_shift_range = self.params["distance_cell_shift_range"]

            if "AxonShell" in self.params["max_count_of_organells"].keys():
                count_of_axons = self.params["max_count_of_organells"]["AxonShell"]
            else:
                count_of_axons = 0
            self.axon_cell_index_list = []

            capillary_count = 1 ############################################################################################ Создание обычной клетки с более толстой оболочкой (возможно капиляры)

            # РИСОВАНИЕ АКСОНА
            for cell in self.cell_list:
                if isinstance(cell.list_of_organells[0], EmptyOrganelle) and count_of_axons > 0:
                    logger.organelle("Draw EmptyOrganell as AxonShell")
                    axon_membrane_size = get_rand_int(self.params["axon_membrane_size"])
                    axon_color_membrane_param = self.params["axon_color_membrane_param"]

                    _, axon_mask = shift_boundary_with_shell(self.cell_fields, cell.index, axon_membrane_size)
                    draw_data_by_mask_and_random_value(self.data,
                                                       axon_mask==True,
                                                       axon_color_membrane_param)  ######## PARAM

                    self.axon_cell_index_list.append(cell.index)
                    count_of_axons -= 1
                    logger.main("AxonShell created")

                elif isinstance(cell.list_of_organells[0], EmptyOrganelle) and capillary_count > 0:
                    logger.organelle("Draw EmptyOrganell as СapillaryShell")
                    capillary_membrane_size = get_rand_int((1,3))   ############################################# Прям толстенькая
                    _, capillary_mask = shift_boundary_with_shell(self.cell_fields, cell.index, capillary_membrane_size)
                    capillary_count -= 1
                    logger.main("Сapillary created")

                else:
                    size_of_shift = get_rand_int(distance_cell_shift_range)
                    if size_of_shift > 0:
                        shift_boundary(self.cell_fields, cell.index, size_of_shift, mask_for_shift_boundary)
                    added_thickness = get_rand_int(1) # Пока что константа в 2 вокселя ##############################################################################
                    if added_thickness > 0:
                        _, new_mask = shift_boundary_with_shell(self.cell_fields, cell.index, added_thickness)
                    logger.main(f"Cell {cell.index} shift_boundary complited")

            logger.main("Draw membranes")
            # РИСОВАНИЕ МЕМБРАН
            membrane_color_param = self.params["membrane_color"]
            get_fun_type = get_color_index_fun_by_param(membrane_color_param)
            if get_fun_type == 2:################################################################### задание основного и цвета в диапазоне для повышения разнообразия цвета каждой клетки
                for cell in self.cell_list:
                    if cell.index in self.axon_cell_index_list:
                        continue
                    else:
                        main_membrane_color = choise_use_color_by_param(membrane_color_param)
                        min_range = membrane_color_param[0] - membrane_color_param[1]
                        max_range = membrane_color_param[0] + membrane_color_param[1]
                        change_3sigma = min(abs(main_membrane_color - min_range),
                                            abs(max_range - main_membrane_color))
                        use_interior_color_param = (main_membrane_color, change_3sigma)

                        draw_data_by_mask_and_random_value(self.data,
                                                           self.cell_fields[:,:,:] == -cell.index,
                                                           use_interior_color_param)                ######## PARAM

            else:
                draw_data_by_mask_and_random_value(self.data,
                                                   self.cell_fields[:,:,:] < 0,
                                                   membrane_color_param)                        ######## PARAM

    @log_execution(level="main")
    def WriteDataset(self, list_of_str=None):
        path_to_save = self.params["save_dataset_dir"]
        write_dataset(self.data, path_to_save, "original", 0, save_as_gif = True)

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
            logger.main(f"НЕУДАЛОСЬ ДОБАВИТЬ ОРГАНЕЛЛУ ЗА {self.params["max_count_attempt_for_adding_cell"]} ПОПЫТОК")
        else:
            self.cell_list.append(cell)
            self.count_organells += 1

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
