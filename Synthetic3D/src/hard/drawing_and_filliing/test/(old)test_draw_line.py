import cv2
import numpy as np
import math

from Synthetic3D.src.hard.drawing_and_filliing.draw_2d import draw_line_2D, draw_circle_2D, draw_ellips_2D
from Synthetic3D.src.hard.drawing_and_filliing.draw_line_3d import draw_line_3D
from Synthetic3D.src.hard.drawing_and_filliing.draw_triangle import draw_voxel_triangle
from Synthetic3D.src.utilities.view_data import view_vtk_3D_data
from Synthetic3D.src.hard.drawing_and_filliing.filling_area import recursive_fill_closed_figure

def test_line_2D():

    color = 128
    thickness = 0

    img_size = 512
    resize_coef = 2

    index = 0
    key = ""
    while key != ord('q'):
        img = np.zeros((img_size, img_size), dtype=np.uint8)

        x = img_size//2
        y = img_size//2

        x2 = x + 60 * math.cos(index)
        y2 = y + 125 * math.sin(index)

        index += 0.05

        draw_line_2D(img, x,y, x2, y2, color, int(thickness*5*index)%20)

        draw_circle_2D(img, x,y, 70, int(color+index*20)%255, thickness)

        draw_ellips_2D(img, x, y, 80, 240, 255, thickness)
        draw_ellips_2D(img, x, y, int(20 * index) % 240 + 3, int(60 * index) % 240 + 3, 255, thickness)


        draw_circle_2D(img, x, y, 75, 255, thickness)
        draw_circle_2D(img, x, y, 245, 255, thickness)

        draw_ellips_2D(img, 50, 50, int(2*index)%6+3, int(1*index)%6+3, 255, thickness)

        cv2.imshow(f"test_line_2D", cv2.resize(img, (img_size*resize_coef, img_size*resize_coef)))
        key = cv2.waitKey(50)

def test_3D_line_and_triangle():
    color = 255
    thickness = 3

    img_size = 256
    resize_coef = 4


    key = ""

    data = np.zeros((img_size, img_size, img_size), dtype=np.uint8)

    # сетка пирамидки

    center = (img_size//2, img_size//2, img_size//4)

    delta_triangle = 180

    up_vertex = (center[0], center[1], center[2] + delta_triangle)


    overlap_line = 100
    l_d_point = (center[0], center[1] - overlap_line, center[2]-50)
    l_u_point = (center[0], center[1] + overlap_line, center[2])
    r_d_point = (center[0] + overlap_line, center[1], center[2]-50)
    r_u_point = (center[0] - overlap_line, center[1], center[2])

    points = [l_d_point, r_u_point, l_u_point, r_d_point]
    # верхний треугольник
    for i in range(0, 4):
        draw_line_3D(data, up_vertex, points[i], color, thickness)

    for i in range(0, 4):
        draw_line_3D(data, points[i], points[(i + 1)%4], color, thickness)

    draw_voxel_triangle(data, up_vertex, l_d_point, r_u_point, color//2, 0)
    draw_voxel_triangle(data, up_vertex, l_d_point, l_u_point, color // 2, 0)

    for i in range(0, img_size):
        cv2.imwrite(f"test_gen_data/{i}_slice.png", cv2.resize(data[i], (img_size*resize_coef, img_size*resize_coef), interpolation=cv2.INTER_NEAREST))

    view_vtk_3D_data(data)

def get_tetraedr(img_size, color, thickness):


    data = np.zeros((img_size, img_size, img_size), dtype=np.uint8)

    # сетка пирамидки

    center = (img_size // 2, img_size // 2, img_size // 4)

    delta_triangle = 180

    up_vertex = (center[0], center[1], center[2] + delta_triangle)

    overlap_line = 100
    l_d_point = (center[0], center[1] - overlap_line, center[2] - 50)
    l_u_point = (center[0], center[1] + overlap_line, center[2])
    r_d_point = (center[0] + overlap_line, center[1], center[2] - 50)
    r_u_point = (center[0] - overlap_line, center[1], center[2])

    draw_voxel_triangle(data, up_vertex, l_d_point, r_u_point, color, thickness)
    draw_voxel_triangle(data, up_vertex, r_d_point, l_u_point, color, thickness)
    draw_voxel_triangle(data, up_vertex, r_d_point, l_d_point, color, thickness)
    draw_voxel_triangle(data, up_vertex, r_u_point, l_u_point, color, thickness)

    draw_voxel_triangle(data, l_u_point, r_d_point, l_d_point, color, thickness)
    draw_voxel_triangle(data, l_d_point, r_u_point, l_u_point, color, thickness)

    return data

def test_3D():
    color = 255
    thickness = 0

    img_size = 256
    resize_coef_for_2D_slice = 4
    data = get_tetraedr(img_size, color, thickness)

    for i in range(0, img_size):
        cv2.imwrite(f"test_gen_data/{i}_slice.png",
                    cv2.resize(data[i], (img_size * resize_coef_for_2D_slice, img_size * resize_coef_for_2D_slice),
                               interpolation=cv2.INTER_NEAREST))

    view_vtk_3D_data(data)

def test_full_3D_data():
    color = 255
    thickness = 1

    img_size = 512
    resize_coef_for_2D_slice = 4

    data = np.zeros((img_size, img_size, img_size), dtype=np.uint8)
    # сетка пирамидки
    center = (img_size // 2, img_size // 2, img_size // 4)
    delta_triangle = 180
    up_vertex = (center[0], center[1], center[2] + delta_triangle)

    overlap_line = 100
    l_d_point = (center[0], center[1] - overlap_line, center[2] - 50)
    l_u_point = (center[0], center[1] + overlap_line, center[2])
    r_d_point = (center[0] + overlap_line, center[1], center[2] - 50)
    r_u_point = (center[0] - overlap_line, center[1], center[2])

    triangle_list = [
        (up_vertex, l_d_point, r_u_point),
        (up_vertex, r_d_point, l_u_point),
        (up_vertex, r_d_point, l_d_point),
        (up_vertex, r_u_point, l_u_point),

        (l_u_point, r_d_point, l_d_point),
        (l_d_point, r_u_point, l_u_point)
    ]

    center = (center[0]-50, center[1], center[2])

    delta_triangle = 180
    up_vertex = (center[0], center[1], center[2] + delta_triangle)

    overlap_line = 100
    l_d_point = (center[0], center[1] - overlap_line, center[2] - 50)
    l_u_point = (center[0], center[1] + overlap_line, center[2])
    r_d_point = (center[0] + overlap_line, center[1], center[2] - 50)
    r_u_point = (center[0] - overlap_line, center[1], center[2])

    triangle_list2 = [
        (up_vertex, l_d_point, r_u_point),
        (up_vertex, r_d_point, l_u_point),
        (up_vertex, r_d_point, l_d_point),
        (up_vertex, r_u_point, l_u_point),

        (l_u_point, r_d_point, l_d_point),
        (l_d_point, r_u_point, l_u_point)
    ]

    recursive_fill_closed_figure(data, triangle_list+triangle_list2, (up_vertex[0], up_vertex[1], up_vertex[2]-4), 128)

    #for triangle in triangle_list+triangle_list2:
    #    draw_voxel_triangle(data, triangle[0], triangle[1], triangle[2], 255, thickness)



    for i in range(0, img_size):
        cv2.imwrite(f"test_gen_data/{i}_slice.png",
                    cv2.resize(data[i], (img_size * resize_coef_for_2D_slice, img_size * resize_coef_for_2D_slice),
                               interpolation=cv2.INTER_NEAREST))

    view_vtk_3D_data(data)

#test_line_2D()
#test_3D_line_and_triangle()
#test_3D()
test_full_3D_data()
