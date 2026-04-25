from copy import deepcopy
import numpy as np

from Synthetic3D.src.hard.structure.edge import Edge
from Synthetic3D.src.hard.structure.triangle import Triangle
from Synthetic3D.src.hard.structure.expandable_array import ExpandableVectorArray
from Synthetic3D.src.hard.structure.vertex_as_SoA import VertexAsSoA
from Synthetic3D.src.utilities.logging_config import logger
from Synthetic3D.src.hard.triangles_shell_filter import triangle_shell_filter

from Synthetic3D.src.hard.split_and_partition import partition_of_triangle, remove_no_used_edges

class OuterShell:
    """
        Класс для хранения поверхностных точек без осевых точек frame
    """

    def __init__(self):
        self.vertexes = VertexAsSoA()
        self.edge_list = []
        self.triangle_list = []

    def __str__(self):
        str = "Class OuterShell:\n" + \
              f"\tlen position list: {len(self.vertexes.positions)}\n" + \
              f"\tlen normals  list: {len(self.vertexes.normales)}\n" + \
              f"\tlen edge     list: {len(self.edge_list)}\n" + \
              f"\tlen triangles list: {len(self.triangle_list)}\n"
        return str

    def __deepcopy__(self, memo):
        new_shell = OuterShell()
        new_shell.vertexes = deepcopy(self.vertexes, memo)
        new_shell.edge_list = deepcopy(self.edge_list, memo)
        new_shell.triangle_list = deepcopy(self.triangle_list, memo)
        return new_shell

    def create_with_copy_vertexs(self):
        # Создает массив как ссылку на текущий массив вершин
        new_shell = OuterShell()
        new_shell.vertexes = self.vertexes
        return new_shell

    def add_edge(self, edge):
        self.edge_list.append(edge)

    def add_edges(self, *args):
        for edge in args:
            self.add_edge(edge)

    def add_edges_list(self, edges_list):
        self.edge_list += edges_list

    def add_triangle(self, tr):
        self.triangle_list.append(tr)

    def add_triangles(self, *args):
        for tr in args:
            self.add_triangle(tr)

    def add_triangles_list(self, triangles_list):
        self.triangle_list += triangles_list

    def add_vertex(self, *args):
        self.vertexes.add_vertex(*args)

    def add_vertex_list(self, list_of_vertex):
        self.vertexes.add_vertex_list(list_of_vertex)

    def get_edge(self, index):
        return self.edge_list[index]

    def get_vertex(self, index):
        return self.vertexes.get_vertex(index)

    def get_positions(self):
        return self.vertexes.get_positions()

    def get_normales(self):
        return self.vertexes.get_normales()

    def get_position_and_normal(self, index):
        return self.vertexes.get_position_and_normal(index)

    def get_poligon(self, index):
        return self.triangle_list[index]

    def get_generator_of_3_pos_by_triangles(self):
        for triangle in self.triangle_list:
            yield triangle.get_values_by_vertex_indices_from_list(self.get_positions())

    def get_vertex_count(self):
        return self.vertexes.size()

    def get_edge_count(self):
        return len(self.edge_list)

    def shift_coords_by_value(self, vector):
        self.vertexes.shift_coords_by_value(vector)

    def rotate_coords_and_normals(self, angle):
        self.vertexes.rotate_coords_and_normals(angle)

    def transform_coords2int(self):
        self.vertexes.transform_coords2int()

    def get_union_vertex_list(self):
        return self.vertexes.get_union_vertex_list()

    def add_shell(self, other):
        new_shell = deepcopy(self)
        new_shell += other
        return new_shell

    def __iadd__(self, other):
        now_vertex_index = self.get_vertex_count()
        self.vertexes += other.vertexes

        now_edge_index = self.get_edge_count()
        for edge in other.edge_list:
            new_edge = Edge(edge.v1_index + now_vertex_index,
                            edge.v2_index + now_vertex_index)
            if edge.vertex_mean is not None:
                new_edge.vertex_mean = edge.vertex_mean + now_vertex_index
            self.edge_list.append(new_edge)

        for triangle in other.triangle_list:
            vertex_indexes = triangle.vertex_indexes
            edge_indexes = triangle.edge_indexes

            overlap_vertex_indexes = (vertex_indexes[0] + now_vertex_index,
                                      vertex_indexes[1] + now_vertex_index,
                                      vertex_indexes[2] + now_vertex_index)

            overlap_edge_indexes = (edge_indexes[0] + now_edge_index,
                                    edge_indexes[1] + now_edge_index,
                                    edge_indexes[2] + now_edge_index)

            self.triangle_list.append(Triangle(overlap_vertex_indexes, overlap_edge_indexes))

    def __add__(self, other):
        return self.add_shell(other)

    def reverse(self):
        self.vertexes.reverse()

        now_vertex_index = self.get_vertex_count() - 1
        now_edge_index = self.get_edge_count() - 1

        # замена системы координат при смене порядка следования списка
        new_edge_list = []
        for edge in self.edge_list:
            new_edge = Edge(now_vertex_index - edge.v1_index,
                            now_vertex_index - edge.v2_index)
            if edge.vertex_mean is not None:
                new_edge.vertex_mean = now_vertex_index - edge.vertex_mean
            new_edge_list.append(new_edge)
        self.edge_list = new_edge_list

        # Смена направления для соответствия вершинам.
        self.edge_list.reverse()

        new_triangle_list = []
        for triangle in self.triangle_list:
            vertex_indexes = triangle.vertex_indexes
            edge_indexes = triangle.edge_indexes

            overlap_vertex_indexes = (now_vertex_index - vertex_indexes[0],
                                      now_vertex_index - vertex_indexes[1],
                                      now_vertex_index - vertex_indexes[2])

            overlap_edge_indexes = (now_edge_index - edge_indexes[0],
                                    now_edge_index - edge_indexes[1],
                                    now_edge_index - edge_indexes[2])
            new_triangle_list.append(Triangle(overlap_vertex_indexes, overlap_edge_indexes))
        self.triangle_list = new_triangle_list
        self.triangle_list.reverse()

    def Partition_of_triangles_while_len_of_edge_triangle_more_then_value(self, value=1, shape_of_data=None):
        counter_of_drops = 0

        if shape_of_data is not None:
            drop_count = triangle_shell_filter(self, shape_of_data)
            logger.shell(f"Удалено изначальных треугольников за пределами кадра: {drop_count}")

        check_triangle = self.triangle_list
        finish_check_triangle = []
        while len(check_triangle) > 0:
            logger.shell(f"{counter_of_drops}, Количество треугольников на разбитие {len(check_triangle)}")
            new_check_triangle = []
            for triangle in check_triangle:
                if triangle.calculate_shortest_edge_length_by_position_list(self.get_positions()) > value:
                    new_check_triangle += partition_of_triangle(triangle, self, self.edge_list)
                else:
                    finish_check_triangle.append(triangle)
            check_triangle = new_check_triangle
            counter_of_drops += 1

        self.triangle_list = finish_check_triangle

        logger.shell(f"Дробление закончено за {counter_of_drops} итераций")

        if shape_of_data is not None:
            drop_count = triangle_shell_filter(self, shape_of_data)
            logger.shell(f"Удалено конечных треугольников за пределами кадра: {drop_count}")

        #counter_of_grab = remove_no_used_edges(self)
        #logger.shell(f"Удалено неиспользуемых ребер: {counter_of_grab}")

    def Partition_of_triangles(self, number_of_iteration, shape_of_data=None):
        if shape_of_data is not None:
            drop_count = triangle_shell_filter(self, shape_of_data)
            logger.shell(f"Удалено изначальных треугольников за пределами кадра: {drop_count}")

        for i in range(number_of_iteration):
            new_edge_list = []
            new_triange_list = []
            for triangle in self.triangle_list:
                new_triange_list += partition_of_triangle(triangle, self, new_edge_list)
            self.edge_list = new_edge_list
            self.triangle_list = new_triange_list

        if shape_of_data is not None:
            drop_count = triangle_shell_filter(self, shape_of_data)
            logger.shell(f"Удалено конечных треугольников за пределами кадра: {drop_count}")

class FrameShell(OuterShell):
    """
    Класс для хранения поверхностных точек c фреймом
    """

    def __init__(self):
        super().__init__()
        self.frame_points = ExpandableVectorArray()

    def __str__(self):
        str = "Class FrameShell:\n" +\
              f"\tlen position list: {len(self.vertexes.positions)}\n" +\
              f"\tlen normals  list: {len(self.vertexes.normales)}\n" +\
              f"\tlen edge     list: {len(self.edge_list)}\n" +\
              f"\tlen triangles list: {len(self.triangle_list)}\n" + \
              f"\tlen frames   list: {len(self.frame_points)}\n"
        return str

    def __deepcopy__(self, memo):
        new_shell = FrameShell()
        new_shell.vertexes = deepcopy(self.vertexes, memo)
        new_shell.edge_list = deepcopy(self.edge_list, memo)
        new_shell.triangle_list = deepcopy(self.triangle_list, memo)
        new_shell.frame_points = deepcopy(self.frame_points, memo)
        return new_shell

    def create_with_copy_vertexs(self):
        new_shell = FrameShell()
        new_shell.vertexes = self.vertexes
        new_shell.frame_points = self.frame_points
        return new_shell

    def add_frame_point(self, point):
        self.frame_points.append(point)

    def get_frames(self):
        return self.frame_points

    def get_frames_gen(self):
        return self.frame_points.get_elemens()

    def get_frame(self, index):
        return self.frame_points[index]

    def shift_coords_by_value(self, vector):
        super().shift_coords_by_value(vector)
        self.frame_points.shift_by_vector(vector)

    def rotate_coords_and_normals(self, angle):
        super().rotate_coords_and_normals(angle)
        self.frame_points.rotate(angle)

    def transform_coords2int(self):
        super().transform_coords2int()
        self.frame_points = self.frame_points.to_int()

    def add_shell(self, other):
        new_shell = deepcopy(self)
        new_shell += other
        return new_shell

    def __iadd__(self, other):
        super().__add__(other)
        self.frame_points += other.frame_points

    def reverse(self):
        super().reverse()
        self.frame_points.reverse()


class ThickVertexShell:
    """
        Класс хранит несколько массивов вершин с общей структурой
        Таким образом получается некоторое масштабирование при помощи смещения вершин без необходимости хранить 2 разные
        структуры с общими треугольниками и ребрами.
    """

    def __init__(self, dict_of_vertexes, frame_points, edge_list, triangle_list):
        self.dict_of_vertexes = dict_of_vertexes
        self.frame_points = frame_points
        self.edge_list = edge_list
        self.triangle_list = triangle_list

    def create_with_copy_vertexs(self):
        new_shell = FrameShell()
        new_shell.dict_of_vertexes = self.dict_of_vertexes
        new_shell.frame_points = self.frame_points
        return new_shell

    def shift_coords_by_value(self, value):
        for vertexes in self.dict_of_vertexes.values():
            vertexes.shift_coords_by_value(value)
        self.frame_points.shift_by_vector(value)

    def transform_coords2int(self):
        for vertexes in self.dict_of_vertexes.values():
            vertexes.transform_coords2int()
        self.frame_points = self.frame_points.to_int()

    def rotate_coords_and_normals(self, angle):
        for vertexes in self.dict_of_vertexes.values():
            vertexes.rotate_coords_and_normals(angle)
        self.frame_points.rotate(angle)

    def get_generator_of_3_pos_by_triangles(self):
        for vertexes in self.dict_of_vertexes.values():
            for triangle in self.triangle_list:
                yield triangle.get_values_by_vertex_indices_from_list(vertexes.get_positions())

    def add_frame_point(self, point):
        self.frame_points.append(point)

    def get_frames(self):
        return self.frame_points

    def get_frames_gen(self):
        return self.frame_points.get_elemens()

    def get_frame(self, index):
        return self.frame_points[index]


class ThickShell:
    """
        Класс хранит несколько оболочек с общей структурой фрейма.
        Необходима в случае максимального детализированного отображения, так как в случае создания 2 смещенных
        вовнутрь и наружу оболочек они будут состоять из разного числа точек, и, следовательно, и треугольников.
        В этом случае внутренние фреймы оболочки не важны.
    """

    def __init__(self, dict_of_shells, frame_points):
        self.dict_of_shells = dict_of_shells
        self.frame_points = frame_points

    def create_with_copy_vertexs(self):
        new_shell = FrameShell()
        new_shell.dict_of_shells = self.dict_of_shells
        new_shell.frame_points = self.frame_points
        return new_shell

    def shift_coords_by_value(self, value):
        for shell in self.dict_of_shells.values():
            shell.shift_coords_by_value(value)
        self.frame_points.shift_by_vector(value)

    def transform_coords2int(self):
        for shell in self.dict_of_shells.values():
            shell.transform_coords2int()
        self.frame_points = self.frame_points.to_int()

    def rotate_coords_and_normals(self, angle):
        for shell in self.dict_of_shells.values():
            shell.rotate_coords_and_normals(angle)
        self.frame_points.rotate(angle)

    def get_generator_of_3_pos_by_triangles(self):
        for shell in self.dict_of_shells.values():
            for v in shell.get_generator_of_3_pos_by_triangles():
                yield v

    def add_frame_point(self, point):
        self.frame_points.append(point)

    def get_frames(self):
        return self.frame_points

    def get_frames_gen(self):
        return self.frame_points.get_elemens()

    def get_frame(self, index):
        return self.frame_points[index]

    def Partition_of_triangles_while_len_of_edge_triangle_more_then_value(self, value=1, shape_of_data=None):
        for shell in self.dict_of_shells.values():
            shell.Partition_of_triangles_while_len_of_edge_triangle_more_then_value(value, shape_of_data)

    def Partition_of_triangles(self, count):
        for shell in self.dict_of_shells.values():
            shell.Partition_of_triangles(count)

class MultipleShell:
    def __init__(self, dict_of_shells = {}):
        self.dict_of_shells = dict_of_shells

    def __call__(self, name:str):
        return self.dict_of_shells[name]

    def __getitem__(self, name:str):
        return self.dict_of_shells[name]

    def get_generator_of_3_pos_by_triangles(self):
        for shell in self.dict_of_shells.values():
            for v in shell.get_generator_of_3_pos_by_triangles():
                yield v

    def shift_coords_by_value(self, value):
        for shell in self.dict_of_shells.values():
            shell.shift_coords_by_value(value)

    def transform_coords2int(self):
        for shell in self.dict_of_shells.values():
            shell.transform_coords2int()

    def rotate_coords_and_normals(self, angle):
        for shell in self.dict_of_shells.values():
            shell.rotate_coords_and_normals(angle)

    def Partition_of_triangles_while_len_of_edge_triangle_more_then_value(self, value=1, shape_of_data=None):
        for shell in self.dict_of_shells.values():
            shell.Partition_of_triangles_while_len_of_edge_triangle_more_then_value(value, shape_of_data)

    def Partition_of_triangles(self, count):
        for shell in self.dict_of_shells.values():
            shell.Partition_of_triangles(count)
