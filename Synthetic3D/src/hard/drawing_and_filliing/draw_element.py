import numpy as np
from Synthetic3D.src.hard.structure.vector import Vector

def edge_check(x, y, w, h):
    if x < 0 or x >= w or y < 0 or y >= h:
        return True
    else:
        return False

def setPixel2D(img, x, y, color, thickness):
    c = 1 if len(img.shape) == 2 else img.shape[2]

    if isinstance(color, (int, np.integer, float, np.floating)):
        if c != 1:
            raise Exception(
                f"Размерность цвета не соврадает с количеством каналов изображения ! Color = '{color}', shape img = '{img.shape}'")
    elif len(color) != c:
        raise Exception(f"Размерность цвета не соврадает с количеством каналов изображения ! Color = '{color}', shape img = '{img.shape}'")

    h, w = img.shape[0:2]

    if not edge_check(x, y, w, h):
        img[y, x] = color
    #else:
    #    print(f"WARNING! Выход за границы изображения ! x = '{x}', y = '{y}'. Shape img '{img.shape}'")


    # draw with thickness
    for y_i in range(0, thickness + 1):
        y_pos_up = y + y_i
        y_pos_down = y - y_i
        for x_i in range(0, thickness + 1):
            if y_i == 0 and x_i == 0:
                continue
            if x_i + y_i > 1.5 * thickness:
                continue

            x_pos_right = x + x_i
            x_pos_left = x - x_i

            if not edge_check(x_pos_right, y_pos_up, w, h):
                img[y_pos_up, x_pos_right] = color
            if not edge_check(x_pos_left, y_pos_up, w, h):
                img[y_pos_up, x_pos_left] = color
            if not edge_check(x_pos_right, y_pos_down, w, h):
                img[y_pos_down, x_pos_right] = color
            if not edge_check(x_pos_left, y_pos_down, w, h):
                img[y_pos_down, x_pos_left] = color

    return img

def edge_check_3D(x, y, z, w, h, d):
    if 0 <=x<w and 0<=y<h and 0<=z<d:
        return False
    else:
        return True

def setVoxel3D(data, x, y, z, color, thickness):
    assert len(data.shape) > 2

    c = 1 if len(data.shape) == 3 else data.shape[3]

    if isinstance(color, (int, np.integer, float, np.floating)):
        if c != 1:
            raise Exception(
                f"Размерность цвета не соврадает с количеством каналов изображения ! Color = '{color}', shape data = '{data.shape}'")
    elif len(color) != c:
        raise Exception(f"Размерность цвета не соврадает с количеством каналов изображения ! Color = '{color}', shape data = '{data.shape}'")

    d, h, w = data.shape[0:3]

    if not edge_check_3D(x, y, z, w, h, d):
        data[z, y, x] = color
    #else:
    #    #print(f"WARNING! Выход за границы изображения ! x = '{x}', y = '{y}', z = '{z}'. Shape data '{data.shape}'")

    # draw with thickness
    for z_i in range(0, thickness + 1):
        z_pos_in = z + z_i
        z_pos_out = z - z_i
        for y_i in range(0, thickness + 1):
            y_pos_up = y + y_i
            y_pos_down = y - y_i
            for x_i in range(0, thickness + 1):
                # игнор центра
                if y_i == 0 and x_i == 0 and z_i == 0:
                    continue
                # для ускорения работы расширение не сферой, а срезанием углов плоскостями
                if x_i + y_i + z_i > 1.5 * thickness:
                    continue

                x_pos_right = x + x_i
                x_pos_left = x - x_i

                if not edge_check_3D(x_pos_right, y_pos_up, z_pos_in, w, h, d):
                    data[z_pos_in, y_pos_up, x_pos_right] = color
                if not edge_check_3D(x_pos_right, y_pos_up, z_pos_out, w, h, d):
                    data[z_pos_out, y_pos_up, x_pos_right] = color

                if not edge_check_3D(x_pos_right, y_pos_down, z_pos_in, w, h, d):
                    data[z_pos_in, y_pos_down, x_pos_right] = color
                if not edge_check_3D(x_pos_right, y_pos_down, z_pos_out, w, h, d):
                    data[z_pos_out, y_pos_down, x_pos_right] = color


                if not edge_check_3D(x_pos_left, y_pos_up, z_pos_in, w, h, d):
                    data[z_pos_in, y_pos_up, x_pos_left] = color
                if not edge_check_3D(x_pos_left, y_pos_up, z_pos_out, w, h, d):
                    data[z_pos_out, y_pos_up, x_pos_left] = color

                if not edge_check_3D(x_pos_left, y_pos_down, z_pos_in, w, h, d):
                    data[z_pos_in, y_pos_down, x_pos_left] = color
                if not edge_check_3D(x_pos_left, y_pos_down, z_pos_out, w, h, d):
                    data[z_pos_out, y_pos_down, x_pos_left] = color
    return data
