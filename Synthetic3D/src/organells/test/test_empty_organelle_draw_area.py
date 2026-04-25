from Synthetic3D.src.hard.structure.vector import Vector
from Synthetic3D.src.organells.empty_organelle import EmptyOrganelle
from Synthetic3D.src.utilities.view_data import view_vtk_3D_data

import numpy as np

def TestEmptyOrganelle():


    test_data = np.zeros((256, 256, 256), dtype=np.uint8)

    organell = EmptyOrganelle()

    print(f"Warnings: {organell.warnings}")

    print("ChangePosition")
    print(organell.position)
    organell.ChangePosition(Vector((127, 127, 127)))
    print(organell.position)

    print("Rotate")
    print(organell.angle)
    organell.Rotate(Vector((0, 0, 90)))
    print(organell.angle)

    print("num of vertexes", organell.shell.get_vertex_count())
    print("len of triangles:", len(organell.shell.triangle_list))

    organell.Draw(np.zeros((1,1,1), dtype=int))
    organell.DrawMask(np.zeros((1,1,1), dtype=int))
    list_of_points = organell.DrawArea(test_data, 127)
    print("Count_of_shell_poinst", len(list_of_points))


    view_vtk_3D_data(test_data)


if __name__ == "__main__":
    TestEmptyOrganelle()
