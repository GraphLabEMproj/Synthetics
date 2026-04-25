import cv2
import numpy as np
import math
import random

from Synthetic3D.src.hard.drawing_and_filliing.draw_cylinder import draw_small_cylinder, fill_small_cylinder, draw_small_cylinder_with_filling
from Synthetic3D.src.hard.structure.vector import Vector
from Synthetic3D.src.utilities.view_data import view_vtk_3D_data
from Synthetic3D.src.hard.rotate import rotate_3d
from Synthetic3D.src.hard.vector_operation import normalize_vector

from Synthetic3D.src.hard.drawing_and_filliing.draw_line_3d import draw_line_3D

def distribute_angles(N):
    angles_list = []
    for _ in range(N):
        z = random.uniform(-1, 1)
        r = math.sqrt(1 - z ** 2)
        theta = random.uniform(0, 2 * math.pi)

        # Координаты на сфере
        x = r * math.cos(theta)
        y = r * math.sin(theta)

        # Углы для вращения
        azimuth_deg = math.degrees(math.atan2(y, x))
        if azimuth_deg < 0:
            azimuth_deg += 360
        inclination_deg = math.degrees(math.acos(z))
        # Можно оставить третий угол случайным или равномерным
        third_deg = random.uniform(0, 360)

        angles_list.append(Vector(
            azimuth_deg,
            inclination_deg,
            third_deg
        ))
    return angles_list


#angles = distribute_angles(20)
angles = np.random.randint(0, 359, size=(20, 3))  # distribute_angles(20)


for idx, triple in enumerate(angles):
    print(f"Углы {idx + 1}: {triple}")


def test_draw_cylinder(data):

    center = Vector(75, 50, 50)

    for i in range(20):
        r = np.random.randint(2, 10)
        thickness = 0 #np.random.randint(0, 3)
        dirs = rotate_3d(Vector(1, 0, 0), angle=angles[i]) * np.random.randint(25, 75)
        draw_small_cylinder(data, center+(r+2)*normalize_vector(dirs), dirs, r, (255, 0, 0), thickness=thickness)


def test_fill_cylinder(data):

    center = Vector(200, 50, 50)

    for i in range(10):
        r = np.random.randint(1, 10)
        dirs = rotate_3d(Vector(1, 0, 0), angle=angles[i]) * np.random.randint(25, 75)
        fill_small_cylinder(data, center + (r+2) * normalize_vector(dirs), dirs, r, (0, 255, 0))

def test_draw_and_fill_cylinder(data):
    center = Vector(300, 50, 50)

    for i in range(10):
        r = np.random.randint(1, 5)
        thickness = np.random.randint(0, 3)
        dirs = rotate_3d(Vector(1, 0, 0), angle=angles[i]) * np.random.randint(25, 75)
        draw_small_cylinder_with_filling(data, center + (r+2) * normalize_vector(dirs), dirs, r, (0, 0, 255), (0, 255, 0), thickness)



if __name__ == "__main__":

    data = np.zeros((128, 128, 400, 3), dtype=np.uint8)

    draw_line_3D(data, (0,0,0), (0,0,64), (0,0,255), 0)

    test_draw_cylinder(data)
    if np.all(data == 0):
        print("ERROR DRAW cylinder!!!")

    test_fill_cylinder(data)
    if np.all(data == 0):
        print("ERROR FILL cylinder!!!")

    test_draw_and_fill_cylinder(data)

    for z in range(data.shape[0]):
        slice = data[z,:,:,:]
        cv2.imwrite(f"test_draw_cylinder/cylinder_slice_{z}.png", slice)

    view_vtk_3D_data(data)
