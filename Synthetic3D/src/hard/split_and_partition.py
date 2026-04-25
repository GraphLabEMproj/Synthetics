import numpy as np

from Synthetic3D.src.hard.split_curve import split_half_curve
from Synthetic3D.src.hard.structure.edge import Edge
from Synthetic3D.src.hard.structure.triangle import Triangle

from tqdm import tqdm

def split_edge(shell_with_vertex_and_edge, index_edge, list_for_adding_edge):
    edge = shell_with_vertex_and_edge.get_edge(index_edge)

    if edge.vertex_mean is None:
        p1, n1 = shell_with_vertex_and_edge.get_position_and_normal(edge.v1_index)
        p2, n2 = shell_with_vertex_and_edge.get_position_and_normal(edge.v2_index)

        new_position, new_normal = split_half_curve(p1, n1, p2, n2)
        new_index = shell_with_vertex_and_edge.get_vertex_count()
        shell_with_vertex_and_edge.add_vertex(new_position, new_normal)
        edge.vertex_mean = new_index

    if edge.new_fist_index_pair is None:
        new_index_half_edges = len(list_for_adding_edge)
        edge.new_fist_index_pair = new_index_half_edges
        list_for_adding_edge.append(Edge(edge.v1_index, edge.vertex_mean))
        list_for_adding_edge.append(Edge(edge.vertex_mean, edge.v2_index))

    return edge.v1_index, edge.vertex_mean, edge.v2_index, edge.new_fist_index_pair, edge.new_fist_index_pair+1

def partition_of_triangle(triangle:Triangle,
                          current_shell,
                          list_for_adding_new_edge=[]) -> list[Triangle]:
    if triangle.edge_indexes is None:
        print("Warning! Попытка разбить треугольник без граней !")
        return []
    else:
        # тюрпл из 5 индексов
        split_edge_res_1 = split_edge(current_shell, triangle.edge_indexes[0], list_for_adding_new_edge)
        split_edge_res_2 = split_edge(current_shell, triangle.edge_indexes[1], list_for_adding_new_edge)
        split_edge_res_3 = split_edge(current_shell, triangle.edge_indexes[2], list_for_adding_new_edge)

        v1_1, v_mean1, v2_1, ed1_1, ed2_1 = split_edge_res_1
        # поиск общих вершин из первой
        if split_edge_res_1[0] == split_edge_res_2[0]:
            v1_2, v_mean2, v2_2, ed1_2, ed2_2 = split_edge_res_2

            if v2_1 == split_edge_res_3[0]:
                v1_3, v_mean3, v2_3, ed1_3, ed2_3 = split_edge_res_3
            else:
                v2_3, v_mean3, v1_3, ed2_3, ed1_3 = split_edge_res_3

        elif split_edge_res_1[0] == split_edge_res_2[2]:
            v2_2, v_mean2, v1_2, ed2_2, ed1_2 = split_edge_res_2

            if v2_1 == split_edge_res_3[0]:
                v1_3, v_mean3, v2_3, ed1_3, ed2_3 = split_edge_res_3
            else:
                v2_3, v_mean3, v1_3, ed2_3, ed1_3 = split_edge_res_3

        elif split_edge_res_1[0] == split_edge_res_3[0]:
            v1_2, v_mean2, v2_2, ed1_2, ed2_2 = split_edge_res_3

            if v2_1 == split_edge_res_2[0]:
                v1_3, v_mean3, v2_3, ed1_3, ed2_3 = split_edge_res_2
            else:
                v2_3, v_mean3, v1_3, ed2_3, ed1_3 = split_edge_res_2

        else: # v1_1 == v2_3
            v2_2, v_mean2, v1_2, ed2_2, ed1_2 = split_edge_res_3

            if v2_1 == split_edge_res_2[0]:
                v1_3, v_mean3, v2_3, ed1_3, ed2_3 = split_edge_res_2
            else:
                v2_3, v_mean3, v1_3, ed2_3, ed1_3 = split_edge_res_2

        new_edge_index = len(list_for_adding_new_edge)

        list_for_adding_new_edge.append(Edge(v_mean1, v_mean2))
        list_for_adding_new_edge.append(Edge(v_mean1, v_mean3))
        list_for_adding_new_edge.append(Edge(v_mean2, v_mean3))

        tr1 = Triangle((v1_1, v_mean1, v_mean2), (ed1_1, new_edge_index, ed1_2))

        tr2 = Triangle((v_mean1, v_mean2, v_mean3),
                       (new_edge_index, new_edge_index+1, new_edge_index+2))

        tr3 = Triangle((v2_1, v_mean1, v_mean3),
                       (new_edge_index+1, ed2_1, ed1_3))

        tr4 = Triangle((v_mean3, v_mean2, v2_2),
                       (new_edge_index+2, ed2_2, ed2_3))

        return [tr1, tr2, tr3, tr4]


# Не используется из-за слишком долгой работы.
def remove_no_used_edges(shell):
    edge_list = shell.edge_list
    triangle_list = shell.triangle_list

    new_edge_list = []
    counter_of_grab = 0

    print(len(edge_list), len(triangle_list))


    for i, edge in enumerate(tqdm(edge_list, desc="Проверка ребер")):
        founded = False
        for triangle in triangle_list:
            if any([val == i for val in triangle.edge_indexes]):
                founded = True
                break
        if founded:
            new_edge_list.append(edge)
        else:
            counter_of_grab += 1
            '''
            for triangle in triangle_list:
                ed_i1, ed_i2, ed_i3 = triangle.edge_indexes
                if ed_i1>i:
                    ed_i1-=1
                if ed_i2>i:
                    ed_i2-=1
                if ed_i3>i:
                    ed_i3-=1
                triangle.edge_indexes = np.array([ed_i1, ed_i2, ed_i3], dtype=int)

            for edge in edge_list:
                if edge.new_fist_index_pair is not None:
                    if edge.new_fist_index_pair > i:
                        edge.new_fist_index_pair -= 1

            '''
    #shell.edge_list = new_edge_list
    return counter_of_grab

