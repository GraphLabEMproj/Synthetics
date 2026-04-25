import numpy as np

def is_zero(vector):
    return np.array_equal(vector, np.array([0, 0, 0]))

def vector_equal(vec1, vec2):
    return np.array_equal(vec1, vec2)

def VectorList2Array(list_of_vectors, to_int=False):
    res_arr = np.array(list_of_vectors)

    assert res_arr.shape[0] == len(list_of_vectors)
    assert res_arr.shape[1] == 3

    if to_int:
        return np.round(res_arr).astype(int)
    else:
        return res_arr

class Vector(np.ndarray):
    def __new__(cls, *args, dtype=float):
        if len(args) == 0:
            input_array = [0, 0, 0]
        elif len(args) == 1:
            input_array = args[0]
        elif len(args) == 3:
            input_array = args
        else:
            raise ValueError("Можно создать Vector из 0, 1 или 3 аргументов")
        obj = np.asarray(input_array, dtype=dtype).view(cls)
        if obj.shape != (3,):
            raise ValueError("Vector должен иметь размер 3")
        return obj

    def __eq__(self, other):
        if isinstance(other, (Vector, list, tuple, np.ndarray)):
            return np.array_equal(self, other)
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        if isinstance(other, (int, float, Vector, list, tuple, np.ndarray)):
            return Vector(np.add(self, other))
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (int, float, Vector, list, tuple, np.ndarray)):
            return Vector(np.subtract(self, other))
        else:
            return NotImplemented

    def to_int(self):
        if np.issubdtype(self.dtype, np.floating):
            return Vector(np.round(self).astype(int), dtype=int)
        return self

    '''
    def transform2int(self):
        if np.issubdtype(self.dtype, np.floating):
            self[:] = np.round(self).astype(int)
            print("here")
        print(self.dtype)
        return self
    '''

    def __repr__(self):
        return f"Vector({list(self)})"

def vector_to_int(vector:Vector|np.ndarray):
    if np.issubdtype(vector.dtype, np.floating):
        return np.round(vector).astype(int)
    return vector
