import numpy as np

from Synthetic3D.src.hard.drawing_and_filliing.draw_triangle import draw_voxel_triangle


def get_min_max_coordinate_volue_in_list_of_triple_vertex(list_of_data):
    # поставить стартовой первый элемент списка и убрать его из рассмотрения
    temp_val = list_of_data[0][0]
    min_val = [temp_val[0], temp_val[1], temp_val[2]]
    max_val = [temp_val[0], temp_val[1], temp_val[2]]

    for triple in list_of_data:
        for vertex in triple:
            # перебор по координатам
            for i in range(3):
                if vertex[i] < min_val[i]:
                    min_val[i] = vertex[i]
                if max_val[i] < vertex[i]:
                    max_val[i] = vertex[i]

    return min_val, max_val

def fill_figure_by_mask_and_overlap(data, mask, color, overlap_data=(0,0,0), overlap_mask=(0,0,0)):

    shape_of_data = data.shape
    shape_of_mask = mask.shape

    # учитывает начальное местоположение объекта на данных
    size_of_work_data = (shape_of_data[0]-overlap_data[2],
                         shape_of_data[1]-overlap_data[1],
                         shape_of_data[2]-overlap_data[0])

    # учитывает "обрезку"  маски вышедшей за начальную границу размера данных
    size_of_work_mask = (shape_of_mask[0]-overlap_mask[2],
                         shape_of_mask[1]-overlap_mask[1],
                         shape_of_mask[2]-overlap_mask[0])

    # обеспечивает обрезку части маски, которая не влезла в конечную границу данных
    intersection_size = (min(size_of_work_data[0], size_of_work_mask[0]),
                         min(size_of_work_data[1], size_of_work_mask[1]),
                         min(size_of_work_data[2], size_of_work_mask[2]))

    # вычисляется область пересечения маски и данных и окрашивается цветом
    data[overlap_data[2]: overlap_data[2]+intersection_size[0],
         overlap_data[1]: overlap_data[1]+intersection_size[1],
         overlap_data[0]: overlap_data[0]+intersection_size[2]]\
        [mask[overlap_mask[2]: overlap_mask[2]+intersection_size[0],
              overlap_mask[1]: overlap_mask[1]+intersection_size[1],
              overlap_mask[0]: overlap_mask[0]+intersection_size[2]]==1] = color
    return data

def search_voxel_in_closed_area_of_volume_mask(list_of_triple_vertex):
    x_coord = 0
    y_coord = 0
    z_coord = 0

    for tiple in list_of_triple_vertex:
        for vertex in tiple:
            x_coord += vertex[0]
            y_coord += vertex[1]
            z_coord += vertex[2]

    # для теста затравка - центр отрезка от центра масс до вершин

    center = (int(x_coord/(3*len(list_of_triple_vertex))),
            int(y_coord/(3*len(list_of_triple_vertex))),
            int(z_coord/(3*len(list_of_triple_vertex))))
    ret_points = []
    for tiple in list_of_triple_vertex:
        for vertex in tiple:
            ret_points.append((
                    (vertex[0] + center[0])//2,
                    (vertex[1] + center[1])//2,
                    (vertex[2] + center[2])//2
            ))
    return ret_points

def recursive_mask_filling(mask, points_in_fig):
    list_of_work_wertex = points_in_fig
    while len(list_of_work_wertex) != 0:
        next_work_list = []
        for vertex in list_of_work_wertex:
            # x - step
            for x in (-1, +1):
                if 0 < vertex[0]+x < mask.shape[2]:
                    if mask[vertex[2], vertex[1], vertex[0] + x] == 0:
                        mask[vertex[2], vertex[1], vertex[0] + x] = 1
                        next_work_list.append((vertex[0]+x, vertex[1], vertex[2]))

            # y - step
            for y in (-1, +1):
                if 0 < vertex[1]+y < mask.shape[1]:
                    if mask[vertex[2], vertex[1]+y, vertex[0]] == 0:
                        mask[vertex[2], vertex[1]+y, vertex[0]] = 1
                        next_work_list.append((vertex[0], vertex[1]+y, vertex[2]))

            # z - step
            for z in (-1, +1):
                if 0 < vertex[2]+z < mask.shape[0]:
                    if mask[vertex[2]+z, vertex[1], vertex[0]] == 0:
                        mask[vertex[2]+z, vertex[1], vertex[0]] = 1
                        next_work_list.append((vertex[0], vertex[1], vertex[2]+z))
        list_of_work_wertex = next_work_list

def recursive_fill_closed_figure(data, list_of_triangles_in_vertex_form, point_in_fig, color):
    min_fig_val, max_fig_val = get_min_max_coordinate_volue_in_list_of_triple_vertex(
        list_of_triangles_in_vertex_form)

    shape_of_fig = (max_fig_val[2] - min_fig_val[2] + 1,
                    max_fig_val[1] - min_fig_val[1] + 1,
                    max_fig_val[0] - min_fig_val[0] + 1)

    mod_vertex_coord = []
    for triangle in list_of_triangles_in_vertex_form:
        vertex_triple = []
        for vertex in triangle:
            vertex_triple.append(
                (vertex[0] - min_fig_val[0], vertex[1] - min_fig_val[1], vertex[2] - min_fig_val[2]))

        mod_vertex_coord.append(vertex_triple)

    mask_of_data = np.zeros(shape_of_fig, dtype=np.uint8)

    for triangle in mod_vertex_coord:
        draw_voxel_triangle(mask_of_data, triangle[0], triangle[1], triangle[2], 1)

    list_points_in_fig = point_in_fig if isinstance(point_in_fig, list) else[point_in_fig]
    list_points_in_mask = [(vertex[0]-min_fig_val[0],
                            vertex[1]-min_fig_val[1],
                            vertex[2]-min_fig_val[2]) for vertex in list_points_in_fig]

    recursive_mask_filling(mask_of_data, list_points_in_mask)

    fill_figure_by_mask_and_overlap(data, mask_of_data, color, min_fig_val)

    return data
