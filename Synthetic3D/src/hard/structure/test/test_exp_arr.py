from Synthetic3D.src.hard.structure.expandable_array import ExpandableVectorArray
from Synthetic3D.src.hard.structure.vector import Vector
import numpy as np


def test_exp_arr():
    v1 = ExpandableVectorArray()
    v2 = ExpandableVectorArray(Vector(1, 2, 3))
    v3 = ExpandableVectorArray(np.ones((5,3)))
    v4 = ExpandableVectorArray([Vector(2, 3, 4)])

    print(v1)
    print(v2)
    print(v3)
    print(v4)

    print(v1 + v2 + v3 + v4)

    print("Shift test")
    v4.shift_by_vector(Vector(-1,-2,-3))
    print(v4)


    v5 = ExpandableVectorArray([Vector(0, 0, 1)])
    print("rotate test")
    v5.rotate(Vector(-90,0,0))
    print(v5)

    print(v5.to_int())

    print((v1 + v2 + v3 + v4)[0])

def test_exp_list_arr():


    v2 = ExpandableVectorArray(Vector(1, 2, 3))
    v2.append(Vector(1, 5, 6))
    v2.append(Vector(2, 5, 6))
    v2.append(Vector(3, 5, 6))
    v2.append(Vector(4, 5, 6))

    v3 = v2.to_int()
    v3.append(Vector(-1, 5, 6))
    v3.append(Vector(-2, 5, 6))
    v3.append(Vector(-3, 5, 6))
    v3.append(Vector(-4, 5, 6))

    print("forward")
    for i in range(len(v3)):
        print(v3[i])

    print("revers")
    for i in range(len(v3)):
        print(v3[-1-i])

    v3._list_to_array()
    print(v3)

    print("revers arr")
    v3.reverse()
    print(v3)

if __name__ == "__main__":
    #test_exp_arr()
    test_exp_list_arr()
