import cv2
import numpy as np

from Synthetic3D.src.hard.drawing_and_filliing.draw_sphere import draw_small_sphere
from Synthetic3D.src.hard.drawing_and_filliing.fill_sphere import fill_small_sphere
from Synthetic3D.src.hard.structure.vector import Vector
from Synthetic3D.src.utilities.view_data import view_vtk_3D_data


def test_draw_sphere(data):
    #draw_small_sphere(data, Vector(64, 64, 64), 10, (255, 0, 0))
    #raise Exception("User Stop")

    warnings = []
    r_last = 0
    for r in range(10):
        for j in range(20):
            center = Vector(64 + j*40, r_last +1+ 16 + 40+j, 32)
            warnings += draw_small_sphere(data, center, r, (255, 0, 0), thickness=0, delta_radius=j/20)
        r_last += 2*r+4

    return warnings

def test_fill_sphere(data):
    warnings = []

    for j in range(20):
        center = Vector(64 + j*40, 160 + 40+j, 32)
        warnings += fill_small_sphere(data, center, j, (0, 0, 255), delta_radius=j/20)

    return warnings

if __name__ == "__main__":

    data = np.zeros((64, 256, 1024, 3), dtype=np.uint8)

    w1 = test_draw_sphere(data)
    if np.all(data == 0):
        print("ERROR DRAW SPHERE!!!")

    w2 = test_fill_sphere(data)
    if np.all(data == 0):
        print("ERROR FILL SPHERE!!!")

    print("WARNINGS w1 ", w1)
    print("WARNINGS w2 ", w2)

    for z in range(data.shape[0]):
        slice = data[z,:,:,:]
        cv2.imwrite(f"test_draw_sphere/sphere_slice_{z}.png", slice)

    view_vtk_3D_data(data)
