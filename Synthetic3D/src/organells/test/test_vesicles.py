import cv2
import numpy as np
from Synthetic3D.src.utilities.view_data import view_vtk_3D_data
from Synthetic3D.src.organells.vesicles import Vesicles
from Synthetic3D.src.hard.structure.vector import Vector

from Synthetic3D.src.utilities.logging_config import clear_log_file
def TestVesicle(test_data):

    param = {
        "vesicles": {
            "number_of_vesicles": 1,
            "radius_of_vesicle": 20
        }
    }

    vesicle = Vesicles(param)

    print(f"Warnings: {vesicle.warnings}")

    print("ChangePosition")
    print(vesicle.view_shell.get_frames()[0])
    vesicle.ChangePosition(Vector((127,127,40)))
    print(vesicle.view_shell.get_frames()[0])

    print("Rotate")
    print(vesicle.view_shell.get_frames()[0])
    vesicle.Rotate(Vector((0, 0, 90)))
    print(vesicle.view_shell.get_frames()[0])

    print("num of vesicles:", len(vesicle.view_shell.get_frames()))
    vesicle.Draw(test_data)

    vesicle2 = Vesicles(param)
    vesicle2.SetPosition(Vector([127, 127, 80]))
    vesicle2.Draw(test_data)

def TestVesicles(test_data):
    cloud_vesicle = Vesicles()
    print(f"Warnings: {cloud_vesicle.warnings}")

    print("ChangePosition")
    print(cloud_vesicle.view_shell.get_frames()[0])
    cloud_vesicle.ChangePosition(Vector((256+127,127,64)))
    print(cloud_vesicle.view_shell.get_frames()[0])

    print("Rotate")
    print(cloud_vesicle.view_shell.get_frames()[0])
    cloud_vesicle.Rotate(Vector((0, 0, 90)))
    print(cloud_vesicle.view_shell.get_frames()[0])

    print("num of vesicles:", len(cloud_vesicle.view_shell.get_frames()))
    cloud_vesicle.Draw(test_data)

if __name__ == "__main__":
    clear_log_file()
    print("StartTestVesicles")
    test_data = np.zeros((128, 256, 512, 3), dtype=np.uint8)
    TestVesicle(test_data)
    TestVesicles(test_data)

    for z in range(test_data.shape[0]):
        slice = test_data[z,:,:,:]
        cv2.imwrite(f"test_gen_vesicles/vesicules_slice_{z}.png", slice)

    view_vtk_3D_data(test_data) #" vesicle.view_shell.vertex_list, vesicle.position)# vesicle.view_shell.vertex_list[1].vertex)
