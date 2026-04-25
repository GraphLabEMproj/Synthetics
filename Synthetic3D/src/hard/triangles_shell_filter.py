
from Synthetic3D.src.hard.drawing_and_filliing.draw_element import edge_check_3D

def triangle_shell_filter(shell, shape_of_data) -> int:
    d, h, w = shape_of_data[:3]

    new_triangle_list = []
    drop_counter = 0
    for triangle in shell.triangle_list:
        v1, v2, v3 = triangle.get_values_by_vertex_indices_from_list(shell.get_positions())

        if edge_check_3D(*v1, w, h, d) and \
           edge_check_3D(*v2, w, h, d) and \
           edge_check_3D(*v3, w, h, d):
            drop_counter += 1
        else:
            new_triangle_list.append(triangle)
    shell.triangle_list = new_triangle_list

    return drop_counter
