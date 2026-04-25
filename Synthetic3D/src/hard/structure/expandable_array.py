import numpy as np
from Synthetic3D.src.hard.rotate import rotate_3d
from Synthetic3D.src.hard.structure.vector import Vector
#from Synthetic3D.src.utilities.logging_config import logger

class ExpandableVectorArray:
    def __init__(self, *args, dtype=float):
        array_count = len(args)
        if array_count > 0:
            if array_count == 1:
                if isinstance(args[0], np.ndarray):
                    if len(args[0].shape) == 1:
                        self._array = np.array(args, dtype=dtype)
                    else:
                        self._array = args[0].astype(dtype)

                elif isinstance(args[0], (list, tuple)):
                    self._array = np.array(args[0], dtype=dtype)
                else: # one element of data or Vector
                    self._array = np.array(args, dtype=dtype)
            else:
                self._array = np.array(args, dtype=dtype)
        else:
            self._array = np.zeros((0,3), dtype=dtype)
        self.dtype = self._array.dtype
        self._new_data = []

    def size(self) -> int:
        return len(self._new_data) + len(self._array)

    def __len__(self):
        return len(self._new_data) + len(self._array)

    def __getitem__(self, index):
        if index < 0:
            len_list = len(self._new_data)
            if -index < len_list + 1:
                return self._new_data[index]
            else:
                return self._array[index+len_list]

        else:
            len_arr = len(self._array)
            if index < len_arr:
                return self._array[index]
            else:
                return self._new_data[index - len_arr]

    def __setitem__(self, index, value):
        if index < 0:
            len_list = len(self._new_data)
            if -index < len_list + 1:
                self._new_data[index] = value
            else:
                self._array[index+len_list] = value

        else:
            len_arr = len(self._array)
            if index < len_arr:
                self._array[index] = value
            else:
                self._new_data[index - len_arr] = value


    def __call__(self, index):
        return self.__getitem__(index)

    def __deepcopy__(self, memodict={}):
        new_elem = ExpandableVectorArray(self.copy_data_as_array())
        return new_elem

    def __iadd__(self, other):
        self._list_to_array()
        if len(self._array) > 0:
            self._array = np.append(self._array, other.copy_data_as_array(), axis=0)
        else:
            self._array = other.copy_data_as_array()

        self.dtype = self._array.dtype
        return self

    def add_data(self, other):
        new_data = ExpandableVectorArray(self.copy_data_as_array())
        new_data += other
        return new_data

    def __add__(self, other):
        return self.add_data(other)
    def __str__(self):
        str_val = f"ExpandableVectorArray: vectors in arr {len(self._array)}, vectors list {len(self._new_data)}"+\
                  f", dtype {self.dtype}\n\tArray data {self._array}, List data {self._new_data}\n"
        return str_val

    def get_elemens(self):
        for elem in self._array:
            yield elem
        for elem in self._new_data:
            yield elem

    def _list_to_array(self, dtype=None):
        """
            Переносит список в массив.
        """

        len_list = len(self._new_data)

        if dtype is None:
            dtype = self.dtype
        else:
            self.dtype = dtype

        if len_list > 0:
            len_arr = len(self._array)
            new_array = np.zeros((len_arr+len_list, 3), dtype=dtype)
            if len_arr > 0:
                new_array[:len_arr] = self._array
            for i, elem in enumerate(self._new_data):
                if not np.issubdtype(elem.dtype, dtype) and np.issubdtype(elem.dtype, (np.integer, int)):
                    elem = np.round(elem).astype(dtype)
                new_array[i+len_arr] = elem
            self._array = new_array
            self._new_data = []

    def copy_data_as_array(self, dtype=None) -> np.ndarray:
        """
            Возвращает копию данных в виде массива
        """
        len_arr = len(self._array)
        len_list = len(self._new_data)

        if dtype is None:
            dtype = self.dtype

        if len_list + len_arr == 0:
            return []
        else:
            new_array = np.zeros((len_arr+len_list, 3), dtype=dtype)
            new_array[:len_arr] = self._array
            for i, elem in enumerate(self._new_data):
                new_array[i+len_arr] = elem
            return new_array

    def to_int(self):
        return ExpandableVectorArray(np.round(self.copy_data_as_array()).astype(int), dtype=int)

    def append(self, value):
        self._new_data.append(value)

    def shift_by_vector(self, vector:Vector):
        if vector.dtype == self.dtype:
            self._list_to_array()
        else:
            np.issubdtype(self.dtype, (int, np.integer))  # Нужно для возможности к целочисленному прибавить флоат
            self._list_to_array(vector.dtype)

        if len(self._array) > 0:
            self._array += vector

    def rotate(self, angle:Vector):
        self._list_to_array()
        #####################################################################################можно улучшить переведя расчеты по матрице массива
        roteted_array = np.zeros_like(self._array, dtype=float)
        for i in range(len(self._array)):
            roteted_array[i] = rotate_3d(self._array[i], angle)
        self._array = roteted_array
        self.dtype=float

    def reverse(self):
        self._list_to_array()
        self._array = np.flip(self._array, 0)
