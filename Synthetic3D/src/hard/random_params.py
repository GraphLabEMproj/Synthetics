import numpy as np

def get_rand_double_int_val(r: int|tuple[int,int]) -> tuple[int,int]:
    if isinstance(r, int):
        return r, -r
    else:
        return (np.random.randint(r[0], r[1] + 1),
                -np.random.randint(r[0], r[1] + 1))

def get_rand_int(r: int|tuple[int,int]) -> int:
    if isinstance(r, int):
        return r
    else:
        return np.random.randint(r[0], r[1] + 1)

def get_rand_float(r: float|tuple[float,float]) -> float:
    if isinstance(r, (float, int)):
        return r
    else:
        return np.random.uniform(r[0], r[1])


def random_unit_vector() -> np.ndarray:
    """
    Генерирует случайный единичный вектор в трёхмерном пространстве.
    Распределение равномерное на сфере.
    """
    # Метод сферических координат (равномерно по телесному углу)
    theta = np.random.uniform(0, 2 * np.pi)          # азимутальный угол
    phi = np.arccos(2 * np.random.uniform() - 1)     # полярный угол (cos(phi) равномерен)
    return np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])

def get_bool_rand_probability(probability:float) -> bool:
    """
    Возвращает результат True c вероятность pobability, иначе False
    :param probability:   - значение вероятности
    :return: True or False
    """

    assert 0 <= probability <= 1, "probability can only be in the range [0-1]!"

    if probability == 1:
        return True
    else:
        if probability < np.random.random():
            return True
        else:
            return False

def get_random_int_value_with_gaussian_distribution(mean, scatter_range):
    """
    Возвращает случайное целое число по гауссову распределению с помощью numpy.
    :param mean:   - среднее значение распределения
    :param scatter_range: -  максимальный диапазон отклонения +-3 сигма
    :return: целое число
    """

    return int(np.random.normal(mean, scatter_range/3)+0.495)


def get_random_color_with_gaussian_distribution(mean, scatter_range):
    """
    Возвращает случайное целое число по гауссову распределению с помощью numpy, ограниченное диапазоном [0-255].
    :param mean:   - среднее значение распределения
    :param scatter_range: -  максимальный диапазон отклонения +-3 сигма
    :return: целое число
    """

    return np.clip(get_random_int_value_with_gaussian_distribution(mean, scatter_range), 0, 255)


def choise_use_color_by_param(color_param):
    if isinstance(color_param, (int, np.integer)): # INT
        return color_param
    elif len(color_param) == 1: # INT
        return color_param[0]
    elif len(color_param) == 2: # MEAN AND RANGE
        return get_random_color_with_gaussian_distribution(*color_param)
    elif len(color_param) == 3: # RGB or complex parameter (such as a list of lists)
        return [choise_use_color_by_param(val) for val in color_param]
    else:
        raise ValueError(f'Невозможно преобразовать входные данные "{color_param}" в цвет.')

def get_color_index_fun_by_param(color_param):
    if isinstance(color_param, (int, np.integer)): # INT
        return 0
    elif len(color_param) == 1: # INT
        return 1
    elif len(color_param) == 2: # MEAN AND RANGE
        return 2
    elif len(color_param) == 3: # RGB or complex parameter (such as a list of lists)
        return 3
    else:
        raise ValueError(f'Невозможно преобразовать входные данные "{color_param}" в цвет.')

def get_color_fun_by_param(color_param):
    if isinstance(color_param, (int, np.integer)): # INT
        return lambda: color_param
    elif len(color_param) == 1: # INT
        return lambda: color_param[0]
    elif len(color_param) == 2: # MEAN AND RANGE
        def gaussian_distribution_wiper():
            return get_random_color_with_gaussian_distribution(*color_param)
        return gaussian_distribution_wiper
    elif len(color_param) == 3: # RGB or complex parameter (such as a list of lists)
        return lambda: [choise_use_color_by_param(val) for val in color_param]
    else:
        raise ValueError(f'Невозможно преобразовать входные данные "{color_param}" в цвет.')


def color_dim_check(color, data_shape):
    assert isinstance(data_shape,  (np.integer, int)) or len(data_shape) > 2, f"Работа только с трехмерными данными. Пришел data_shape {data_shape}."

    if isinstance(data_shape, (int, np.integer)):
        c = data_shape
    elif len(data_shape) == 3:
        c = 1
    else:
        c = data_shape[3]

    if isinstance(color, (int, np.integer)):
        if c == 1:
            return color
        else:
            return [color] * c
    else:
        if len(color) == c:
            return color
        else:
            raise ValueError(f'Невозможно преобразовать нестандарный цвет "{color}" в размерность "{c}"')
