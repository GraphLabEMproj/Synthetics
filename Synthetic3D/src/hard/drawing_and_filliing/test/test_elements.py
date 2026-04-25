import numpy as np
from Synthetic3D.src.hard.structure.vector import Vector

def test_vector_init():
    vector = Vector()
    assert vector[0] == 0
    assert vector[1] == 0
    assert vector[2] == 0
    assert len(vector) == 3
    assert vector.dtype == float

    point = Vector(1, 2, 3.7)
    assert point[0] == 1
    assert point[1] == 2
    assert point[2] == 3.7
    assert len(point) == 3
    assert point.dtype == float


def test_vector_set():
    vector = Vector([1, 2, 3])
    assert vector[0] == 1.
    assert vector[1] == 2.
    assert vector[2] == 3.
    assert len(vector) == 3
    assert vector.dtype == float

    vector = Vector([4.0, 2.0, 3.3])
    assert vector[0] == 4
    assert vector[1] == 2
    assert vector[2] == 3.3
    assert len(vector) == 3
    assert vector.dtype == float

    vector = Vector((1, 6.5, 3))
    assert vector[0] == 1
    assert vector[1] == 6.5
    assert vector[2] == 3
    assert len(vector) == 3
    assert vector.dtype == float

    vector = Vector((1.0, 8.0, 9.3))
    assert vector[0] == 1
    assert vector[1] == 8
    assert vector[2] == 9.3
    assert len(vector) == 3
    assert vector.dtype == float

    vector = Vector(np.array([1.0, 2.6, 3.3]))
    assert vector[0] == 1
    assert vector[1] == 2.6
    assert vector[2] == 3.3
    assert len(vector) == 3
    assert vector.dtype == float

def test_operation():
    # Создание Vector
    v1 = Vector([1.0, 2.5, 3.0])
    v2 = Vector([4.0, 5.5, 6.0])

    # Проверка сложения Vector + Vector
    v1221 = v1 + v2
    assert isinstance(v1221, Vector)
    expected_v55 = np.array([5, 8, 9])
    assert np.allclose(v1221, expected_v55), f"Ошибка в p1 + 10.4: {v1221} != {expected_v55}"

    # Проверка сложения Vector + scalar
    v12 = v1 + 10
    assert isinstance(v12, Vector)
    expected_v12 = np.array([11, 12.5, 13])
    assert np.allclose(v12, expected_v12), f"Ошибка в p1 + 10: {v12} != {expected_v12}"

    # Проверка сложения Vector + scalar
    v122 = v1 + 10.5
    assert isinstance(v122, Vector)
    expected_v122 = np.array([11.5, 13, 13.5])
    assert np.allclose(v122, expected_v122), f"Ошибка в p1 + 10: {v122} != {expected_v122}"

    # Проверка сложения Vector - scalar
    v1223 = v1 - 10
    assert isinstance(v1223, Vector)
    expected_v1sadasda2 = np.array([-9, -7.5, -7])
    assert np.allclose(v1223, expected_v1sadasda2), f"Ошибка в v1 - 10: {v1223} != {expected_v1sadasda2}"

    # Проверка сложения Vector - scalar
    v122sd3 = v1 - 10.5
    assert isinstance(v122sd3, Vector)
    expected_v1sadasadsda2 = np.array([-9.5, -8, -7.5])
    assert np.allclose(v122sd3, expected_v1sadasadsda2), f"Ошибка в v1 - 10.5: {v122sd3} != {expected_v1sadasadsda2}"

    # Проверка вычитания Vector - Vector
    v4 = v2 - v1
    assert isinstance(v4, Vector)
    expected_v4 = np.array([3, 3, 3])
    assert np.allclose(v4, expected_v4), f"Ошибка в v2 - v1: {v4} != {expected_v4}"

    print("Все тесты пройдены успешно!")

def test_zero_check():
    v = Vector()
    assert v.is_zero()

    v2 = Vector([2,4,5])
    assert not v2.is_zero()

def test_equel():

    p1 = Vector()
    p2 = Vector()

    assert p1 == p2

    p3 = Vector((1,2,3))

    p4 = Vector([1,2,3])

    assert p1 != p3

    assert p3 == p4


if __name__ == "__main__":
    test_vector_init()
    test_vector_set()
    test_operation()
    test_zero_check()
    test_equel()
