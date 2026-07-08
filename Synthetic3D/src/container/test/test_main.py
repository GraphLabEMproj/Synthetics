import time

import numpy as np

from Synthetic3D.src.container.main_field import MainField
from Synthetic3D.src.utilities.view_data import view_vtk_3D_data
from Synthetic3D.src.utilities.logging_config import logger, clear_log_file

import cv2

def DrawAllArea(cell_list, data):
    for cell in cell_list:
        for organell in cell.list_of_organells:
            organell.DrawArea(data, cell.index)


def test_of_adding_noises_MainField():
    main = MainField()
    main.CreateData()
    main.CreateDataBackGround()
    main.AddNoise()

    for z in range(main.data.shape[0]):
        slice = main.data[z,:,:,:]
        cv2.imwrite(f"test_main_data/main_data_{z}.png", slice)



def test_expansion_MainField():
    main = MainField()
    main.CreateData()
    for i in range(10):
        logger.main(f"Generation {i} organelle of 10")
        main.CreateAndAddCell()

    field_area = np.zeros(main.data.shape[:3], dtype=int)

    print("DrawAllArea")
    DrawAllArea(main.cell_list, field_area)

    if np.all(field_area == 0):
        print("Data clean")

    print("len of cell", len(main.cell_list))

    field_area[field_area[:,:,:] > 0] = 255

    print(field_area.shape)
    print(field_area.max(), field_area.min())

    view_vtk_3D_data(field_area.astype(np.uint8))

    start_time = time.time()
    main.ExpansionOfRegions(repit_flag=False)

    arr_of_num_work_points = np.zeros(len(main.cell_list), dtype=int)
    iter = 0
    summ_of_work_point = 0
    while (main.ExpansionOfRegionsIter(arr_of_num_work_points)):
        iter += 1
        view_data = main.cell_fields.copy()
        view_data[view_data[:,:,:] > 0] = 255
        print(
            f"{iter}-я итерация, num_of_points {arr_of_num_work_points}, work_of_points {arr_of_num_work_points.sum()} и {summ_of_work_point} of {main.data.shape[0] * main.data.shape[1] * main.data.shape[2]}")
        summ_of_work_point += arr_of_num_work_points.sum()
        if iter % 10 == 0:
            view_vtk_3D_data(view_data)

    end_time = time.time()
    print(f"На разрастание регионов размером {main.cell_fields.shape} потребовалось {end_time-start_time} секунд")

    view_data = main.DrawMembranes()
    view_data[field_area[:,:,:] > 0] = (255, 0, 0)

    for z in range(view_data.shape[0]):
        slice = view_data[z,:,:,:]
        cv2.imwrite(f"test_main/cell_field_{z}.png", slice)

    view_vtk_3D_data(view_data)


if __name__ == "__main__":
    clear_log_file()
    #test_of_adding_noises_MainField()
    test_expansion_MainField()