import datetime
import json
import numpy as np
import os.path
import random
import sys
import time

from tqdm import tqdm

from Synthetic3D.src.container.cell import Cell
from Synthetic3D.src.container.axon import Axon
from Synthetic3D.src.container.psd import PSD

from Synthetic3D.src.organells.empty_organelle import EmptyOrganelle
from Synthetic3D.src.organells.mitohondrion import Mitohondrion
from Synthetic3D.src.organells.vesicles import Vesicles

from Synthetic3D.src.container.inrersections_rules import CheckTwoOrganells

from Synthetic3D.src.hard.structure.vector import Vector
from Synthetic3D.src.hard.random_params import get_rand_int, get_bool_rand_probability

from Synthetic3D.src.hard.noise_and_blur import add_internal_structures, CreatePossionNoise, AddGaussianBlur
from Synthetic3D.src.hard.drawing_and_filliing.draw_data_by_mask import draw_data_by_mask_and_random_value
from Synthetic3D.src.container.cell_operation import generate_spheres_mask, shift_boundary_with_shell
from Synthetic3D.src.hard.write_data import write_dataset, write_datamask

#from Synthetic3D.src.utilities.check_to_json import to_json_format
from Synthetic3D.src.utilities.check_of_params import check_param, update_param
from Synthetic3D.src.utilities.logging_config import logger, set_log_file_path, log_execution



def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', empty=' ', print_end='\n'):
    """
    Выводит прогресс-бар в консоль с новой строки.
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + empty * (length - filled_length)
    print(f'\r{prefix} {iteration} of {total} |{bar}| {percent}% {suffix}', end=print_end, flush=True)


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
                                                                      "Membranes": 2,
                                                                      "Axon": 3,
                                                                      "PSD": 4})

        warning_list += check_param(self.params, "max_count_of_organells", {"Mitohondrion": 4,
                                                                            "Vesicles": 3,
                                                                            "EmptyCell": 5,
                                                                            "Axon": 1,
                                                                            "PSD": 4})

        warning_list+=check_param(self.params, "cell", {"membrane_thickness": 2,
                                                        "distance_cell_shift_range": (1,2),
                                                        "membrane_color": (0, 127, 0),
                                                        "membrane_mask_color": 255})

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

    def ExpansionOfRegionsIter(self, arr_of_num_work_points, early_stop_cell_idexes_and_count=None): ############################################################## вынесено чтобы смотреть между итерациями
        repit_flag = False

        indexes_cell_list = [i for i in range(len(self.cell_list))]
        random.shuffle(indexes_cell_list)  # перемешивание для минимизации приоритетоности

        #print(len(indexes_cell_list), (len(early_stop_cell_idexes_and_count) if early_stop_cell_idexes_and_count is not None else None), "$$$$$$$$$$$$$$$$$$$$$$$$$", flush=True)
        for i in indexes_cell_list:
            cell = self.cell_list[i]
            cell_index = cell.index

            if early_stop_cell_idexes_and_count is not None and\
               cell_index in early_stop_cell_idexes_and_count.keys():
                #print("early stop", cell_index)
                if early_stop_cell_idexes_and_count[cell_index] > 0:
                    #print("\t", cell_index, early_stop_cell_idexes_and_count[cell_index])
                    if cell.ExpansionOfRegion(self.cell_fields):
                        repit_flag = True
                        arr_of_num_work_points[cell.index - 1] = len(cell.work_points)
                        early_stop_cell_idexes_and_count[cell_index] -= 1
                    else:
                        early_stop_cell_idexes_and_count[cell_index] = 0
                else:
                    continue
            else:
                #print("default", cell_index)
                if cell.ExpansionOfRegion(self.cell_fields):
                    repit_flag = True
                    arr_of_num_work_points[cell.index - 1] = len(cell.work_points)

        #print("$$$$$$$$$$$$$$$$$$$$$$$$$", flush=True)
        return repit_flag

    @log_execution(level="main")
    def ExpansionOfRegions(self, repit_flag=True):
        self.cell_fields = np.zeros(self.data.shape[:3], np.int64) # можено сделать и побольше

        for cell in self.cell_list:
            cell.DrawKernelOfArea(self.cell_fields)

        iteration_counter = 0
        summ_of_work_point = 0

        probability_earli_stop_of_vesicles = 0.75                                                                                                # EXPERIMENT PARAM !!!!!!!
        count_of_garanty_early_step_iters = (3, 20)                                                                                             # EXPERIMENT PARAM !!!!!!!

        early_stop_cell_idexes_and_count = {}
        for cell in self.cell_list:
            if isinstance(cell.list_of_organells[0], Vesicles):
                if get_bool_rand_probability(probability_earli_stop_of_vesicles):
                    early_stop_cell_idexes_and_count[cell.index] = get_rand_int(count_of_garanty_early_step_iters)

        arr_of_num_work_points = np.zeros(len(self.cell_list), dtype=int)

        logger.main("Расширение основного циккла")
        # Расширение для всех, но с лимитом остановочных
        cycle_repit = repit_flag
        while cycle_repit:
            cycle_repit = self.ExpansionOfRegionsIter(arr_of_num_work_points, early_stop_cell_idexes_and_count)
            iteration_counter += 1
            logger.main(
                f"{iteration_counter}-я итерация, num_of_points {arr_of_num_work_points}, work_of_points {arr_of_num_work_points.sum()} и {summ_of_work_point} of {self.data.shape[0] * self.data.shape[1] * self.data.shape[2]}")
            summ_of_work_point += arr_of_num_work_points.sum()

        logger.main("Расширение остатков")
        # Расширение тех, кто рано остановился для того чтобы правильно закончили внутрянку
        cycle_repit = repit_flag
        while cycle_repit:
            cycle_repit = self.ExpansionOfRegionsIter(arr_of_num_work_points)
            iteration_counter += 1
            logger.main(
                f"{iteration_counter}-я итерация, num_of_points {arr_of_num_work_points}, work_of_points {arr_of_num_work_points.sum()} и {summ_of_work_point} of {self.data.shape[0] * self.data.shape[1] * self.data.shape[2]}")
            summ_of_work_point += arr_of_num_work_points.sum()


        self._ExpansionOfMembranens()
        self._AddedPSD(early_stop_cell_idexes_and_count)

        logger.main(f"ExpansionOfRegions completed in {iteration_counter} iterations")

    def _ExpansionOfMembranens(self):
        logger.cell("Expansion of membrans and create axons")
        if self.cell_fields is not None:
            # Создание регионов для расслоения (Пока что пробная версия)
            mask_for_shift_boundary = generate_spheres_mask(self.cell_fields.shape,
                                                            density= 0.20,                                                                                             # EXPERIMENT PARAM !!!!!!!
                                                            radius_range = (4,30),                                                                                     # EXPERIMENT PARAM !!!!!!!
                                                            )
            logger.main("Calculate shitfing membranes")

            ############################################## ЭКСПЕРИМЕНТАЛЬНОЕ ##########################################
            capillary_count = 1 ############################################################################################ Создание обычной клетки с более толстой оболочкой (возможно капиляры)) # EXPERIMENT PARAM !!!!!!!

            for cell in self.cell_list:
                if (not isinstance(cell, Axon)) and isinstance(cell.list_of_organells[0], EmptyOrganelle) and capillary_count > 0:
                    logger.organelle("Draw EmptyOrganell as СapillaryShell")
                    capillary_membrane_size = get_rand_int((4,7)) - 1   ############################################# Прям толстенькая
                    shift_boundary_with_shell(self.cell_fields, cell.index, capillary_membrane_size)
                    capillary_count -= 1
                    logger.main("Сapillary created")
                else:
                    cell.PostExpansionMembrance(self.cell_fields, mask_for_shift_boundary)

    def _AddedPSD(self, dict_of_vesicles_indexes):
        count_of_PSD = (self.params["max_count_of_organells"]["PSD"] if "PSD" in self.params["max_count_of_organells"].keys() else 0)
        logger.main(f"Добавление {count_of_PSD} PSD")


        axon_indices = [cell.index for cell in self.cell_list if isinstance(cell, Axon)]
        no_axon_list = [cell for cell in self.cell_list if not isinstance(cell, Axon)]

        list_of_indexes_vec_cells = [*dict_of_vesicles_indexes.keys()]
        for i in range(count_of_PSD):
            if len(list_of_indexes_vec_cells) > 0:
                cell = self.cell_list[list_of_indexes_vec_cells.pop() - 1] # индекс на 1 больше, поскольку 0 это индекс фона
            else:
                cell = random.choice(no_axon_list)  ##################################################################################################################### Нужно доработать

            cell.Create_PSD(self.cell_fields, axon_indices, self.params)

    '''
    ###################################################################################################################################### переделать функцию
    def DrawMembransMask(self):
        if self.cell_fields is None:
            raise RuntimeError("Мембраны ещё не считались! Для получения мембран нужно запустить алгоритм разрастания регионов!")

        mode = "semantic"                       ######################################### ПАРАМЕТР НА БУДУЩЕЕ
        if mode == "semantic":
            for cell in self.cell_list:
                if isinstance(cell, Axon):
                    continue
                else:
                    cell.DrawMembraneMask

        elif mode == "instance":
            NotImplementedError("Membrane mode 'instance' no realised")
        else:
            raise NotImplementedError("Now membrane mode only 'semantic'")
    '''

    ################################################################################################################################################################### НУЖНА ДЛЯ ОТЛАДКИ
    def DrawMembranesForView(self):
        if self.cell_fields is None:
            raise RuntimeError("Мембраны ещё не считались! Для получения мембран нужно запустить алгоритм разрастания регионов!")

        view_data = np.zeros(self.cell_fields.shape + (3,), dtype=np.uint8)
        for cell in self.cell_list:
            if isinstance(cell, Axon):
                continue
            else:
                draw_data_by_mask_and_random_value(view_data,
                                                   self.cell_fields[:,:,:] == -cell.index,
                                                   self.params["membrane_mask_color"])        ######## PARAM
        return view_data

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
                elif isinstance(organell, PSD) and "PSD" in self.params["dict_of_organells"].keys():
                    organell.DrawMask(self.masks[self.params["dict_of_organells"]["PSD"]])
                else:
                    raise ValueError(f"This organelle type {type(organell)} is not supported yet!")

            if isinstance(cell, Axon):
                cell.DrawMembraneMask(self.masks[self.params["dict_of_organells"]["Axon"]], self.cell_fields)
            else:
                if self.cell_fields is not None and "Membranes" in self.params["dict_of_organells"].keys():
                    cell.DrawMembraneMask(self.masks[self.params["dict_of_organells"]["Membranes"]], self.cell_fields)

    @log_execution(level="main")
    def DrawDataset(self):
        total_cells = len(self.cell_list)

        # Чтобы на затенение PSD могли наложиться везикулы
        for cell in tqdm(self.cell_list, desc='Draw cells membrans, axon shells and PSD', colour="GREEN", file=sys.stdout):
            cell.DrawMembrane(self.data, self.cell_fields)
            # Выводим прогресс на новой строке

        for idx, cell in enumerate(self.cell_list, start=1):
            # Выводим прогресс на новой строке
            print_progress_bar(
                idx,
                total_cells,
                prefix='Draw cells organells',
                suffix='',
                length=40)
            cell.DrawOrganelles(self.data, self.cell_fields)

    @log_execution(level="main")
    def WriteDataset(self, list_of_str=None):
        path_to_save = self.params["save_dataset_dir"]
        write_dataset(self.data, path_to_save, "original", 0, save_as_gif = True)

        for class_name, index_dataset in self.params["dict_of_organells"].items():
            write_datamask(self.masks[index_dataset], path_to_save, class_name.lower())

        self.params["logs"] = list_of_str
        with open(os.path.join(path_to_save, 'dataset_config.json'), 'w', encoding='utf-8') as f:
            json.dump(self.params, f, indent=4, ensure_ascii=False)

        set_log_file_path(os.path.join(path_to_save, "generate.log"))


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

    def CheckIntersections(self, newCell: Cell|Axon):
        for cell in self.cell_list:
            for organel in cell.list_of_organells:
                for new_organell in newCell.list_of_organells:
                    if CheckTwoOrganells(organel, new_organell):
                        return True
        return False

    def AddNewCell(self, cell:Cell|Axon):
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

    def update_draw_config(self, config):
        if config is not None:
            update_param(self.params, config, "poisson_noise")
            if "backgraund" in config.keys():
                self.params["backgraund"].update(config["backgraund"])

            update_param(self.params, config, "gaussian_blur_radius")
            update_param(self.params, config, "gaussian_blur_sigma")
            update_param(self.params, config, "mask_blur_threshold")

            for cell in self.cell_list:
                cell.update_draw_config(config)


    @log_execution(level="main")
    def AddCellsByConfig(self):
        max_count_cell_config = self.params["max_count_of_organells"]
        max_count_of_mito = max_count_cell_config.get("Mitohondrion", 0)
        max_count_of_ves = max_count_cell_config.get("Vesicles", 0)
        max_count_of_empty = max_count_cell_config.get("EmptyCell", 0)
        max_count_of_axons = max_count_cell_config.get("Axon", 0)

        list_of_organells = []
        if max_count_of_mito > 0:
            list_of_organells.append([max_count_of_mito, Mitohondrion, Cell])
        if max_count_of_ves > 0:
            list_of_organells.append([max_count_of_ves, Vesicles, Cell])
        if max_count_of_empty > 0:
            list_of_organells.append([max_count_of_axons, EmptyOrganelle, Axon]) # Основа клетки - пустая органелла
        if max_count_of_empty > 0:
            list_of_organells.append([max_count_of_empty, EmptyOrganelle, Cell])

        count_of_added_organells = sum(self.params["max_count_of_organells"].values()) - (self.params["max_count_of_organells"]["PSD"] if "PSD" in self.params["max_count_of_organells"].keys() else 0)

        # Можно засечь время для отображения затраченного времени (опционально)
        start_time = time.time()
        suffix = ''
        for i in range(count_of_added_organells):
            print_progress_bar(i+1, count_of_added_organells,
                               prefix='Добавление органелл',
                               suffix=suffix,
                               length=40)  # длина полосы

            #choise_index = np.random.randint(len(list_of_organells))
            choise_index = 0 ##################################################################################### Пусть идут по порядку. Сначала огромные митохондрии, потом более маленькие везикулы, а потом все остальные
            max_count, new_organelle, cell_class = list_of_organells[choise_index]
            # индекс 0 зарезервирован под пустоту
            new_index = len(self.cell_list) + 1
            new_cell = cell_class(new_index, [new_organelle(self.params)], params=self.params)
            self.SetNewPosition(new_cell)
            self.AddNewCell(new_cell)

            if max_count > 1:
                list_of_organells[choise_index][0] -= 1
            else:
                list_of_organells.pop(choise_index)

            # Выводим прогресс на новой строке
            elapsed = time.time() - start_time
            # Добавим время, если нужно
            suffix = f'[Время: {elapsed:.1f}с]'

        print(f'Добавление {count_of_added_organells} органелл завершено за {elapsed:.1f}с')
