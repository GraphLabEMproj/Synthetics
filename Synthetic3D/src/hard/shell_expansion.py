import numpy as np
from Synthetic3D.src.hard.structure.shells import OuterShell, FrameShell
from Synthetic3D.src.hard.structure.vector import Vector
from Synthetic3D.src.hard.vector_operation import normalize_vector
from Synthetic3D.src.hard.structure.expandable_array import ExpandableVectorArray
from copy import deepcopy

from Synthetic3D.src.utilities.logging_config import logger

def shell_vertex_expansion(shell:FrameShell, value=0.0) -> tuple[ExpandableVectorArray, ExpandableVectorArray]:
    """
        Данная функция создает новый набор вертексов (позиций с нормалями), смещенных на направлении нормалей и
    возвращает оболочку.

    :param shell:   - Входная оболочка для модификаций.
    :param value:   - Дистанция смещения. Положительные смещают наружу, а отрицательные внутрь.
    :return:        - Возвращает массив, так что запрещает добавление и удаление точек. Возможно следует заменить на
                      список.
    """

    if value == 0:
        return deepcopy(shell.get_positions()), deepcopy(shell.get_normales())
    else:
        len_of_points = shell.get_vertex_count()
        new_position = np.zeros((len_of_points, 3), dtype=float)
        new_normals = np.zeros((len_of_points, 3), dtype=float)

        logger.warning(f"shell_vertex_expansion нуждается в доработке функции поиска средней точки при разбиении")
        for i in range(len_of_points):
            pos, norm = shell.get_position_and_normal(i)
            mod_norm = normalize_vector(norm) * value

            new_position[i] = pos.astype(float) + mod_norm
            if np.linalg.norm(mod_norm) < np.linalg.norm(norm):
                new_normals[i] = norm + mod_norm
            else: # если модификация сильно меняет направление, то изменить размер нормали на 1/10 от её длины
                new_normals[i] = norm * (value/10 + 1)

        return ExpandableVectorArray(new_position), ExpandableVectorArray(new_normals)

def shell_expansion(shell:FrameShell, value=0.0) -> OuterShell:
    """
        Данная функция создает оболочку, со смещенным положением точек в направлении нормалей на value и возвращает
    новую оболочку.

    :param shell:Shell  - Входная оболочка для модификаций
    :param value:float  - Дистанция смещения. Положительные смещают наружу, а отрицательные внутрь.
    :return: Shell      - Новая модифицированная оболочка
    """

    if value == 0:
        return deepcopy(shell)
    else:
        new_shell = OuterShell()
        new_position, new_normals = shell_vertex_expansion(shell, value)
        new_shell.vertexes.positions = new_position
        new_shell.vertexes.normales = new_normals
        # ребра и треугольники не меняются, но они указывают на смещенные вершины
        new_shell.edge_list = deepcopy(shell.edge_list)
        new_shell.triangle_list = deepcopy(shell.triangle_list)

        return new_shell
