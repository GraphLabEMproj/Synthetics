import numpy as np
from Synthetic3D.src.utilities.logging_config import logger

class Triangle:
    def __init__(self, triple_vertex_index, triple_edge_indexes=None):
        assert_error_str = "Треугольник состоит из 3 индексов вершин, а также, дополнительно, 3 индексов ребер."
        assert len(triple_vertex_index) == 3, assert_error_str + f" Пришло не 3 индекса вершины: {triple_vertex_index}."
        assert triple_edge_indexes is None or len(triple_edge_indexes) == 3, assert_error_str +\
                                                    f" Пришел ошибочный набор индексов ребер: {triple_edge_indexes}."

        self.vertex_indexes = triple_vertex_index
        self.edge_indexes = triple_edge_indexes
    def get_values_by_vertex_indices_from_list(self, vertex_list):
        return (vertex_list[self.vertex_indexes[0]],
                vertex_list[self.vertex_indexes[1]],
                vertex_list[self.vertex_indexes[2]])

    def get_values_by_edge_indices_from_list(self, edge_list):
        if self.edge_indexes is None:
            logger.warning("Warning Triangle! Попытка доступа отсутствующему списку ребер треугольника !")
            return None
        else:
            return (edge_list[self.edge_indexes[0]],
                    edge_list[self.edge_indexes[1]],
                    edge_list[self.edge_indexes[2]])

    def check_vertex_indices_in_indices_within_edges_in_edges_list(self, edges_list):
        for index_edge in self.edge_indexes:
            edge = edges_list[index_edge]
            if not edge.v1_index in self.vertex_indexes or not edge.v2_index in self.vertex_indexes:
                raise ValueError(f"Edge with ingex {index_edge} {edge} has no index in triagles vertex index" +\
                                 f"{self.vertex_indexes} and edges indexes {self.edge_indexes}")

    def __str__(self):
        str = "Triangle:\n" +\
              f"\tVertex indexes: {self.vertex_indexes}\n" + \
              f"\tEdge indexes: {self.edge_indexes}\n"
        return str

    def square_by_position_list(self, pos_list):
        point1 = pos_list[self.vertex_indexes[0]]
        point2 = pos_list[self.vertex_indexes[1]]
        point3 = pos_list[self.vertex_indexes[2]]

        # Проверка совпадения точек
        if point1 == point2 or point2 == point3 or point1 == point3:
            return 0.0  # Площадь равна 0, если есть совпадающие точки
        else:
            # Распаковка координат
            x1, y1, z1 = point1
            x2, y2, z2 = point2
            x3, y3, z3 = point3

            # Векторные разности
            v1 = (x2 - x1, y2 - y1, z2 - z1)
            v2 = (x3 - x1, y3 - y1, z3 - z1)

            # Векторное произведение v1 x v2
            cross_x = v1[1] * v2[2] - v1[2] * v2[1]
            cross_y = v1[2] * v2[0] - v1[0] * v2[2]
            cross_z = v1[0] * v2[1] - v1[1] * v2[0]

            # Длина векторного произведения
            cross_length = (cross_x ** 2 + cross_y ** 2 + cross_z ** 2) ** 0.5

            # Площадь треугольника = половина длины векторного произведения
            return cross_length / 2

    def calculate_shortest_edge_length_by_position_list(self, pos_list):
        point1 = pos_list[self.vertex_indexes[0]]
        point2 = pos_list[self.vertex_indexes[1]]
        point3 = pos_list[self.vertex_indexes[2]]

        # Вычисляем длины всех трёх ребер
        d1 = np.linalg.norm(point2 - point1)
        d2 = np.linalg.norm(point3 - point2)
        d3 = np.linalg.norm(point1 - point3)

        # Возвращаем минимальную длину ребра
        return min(d1, d2, d3)

def list_of_triangle_to_2_arrs_vertexes_and_edges(triangle_list):
    vertex_index_arr = np.zeros((len(triangle_list), 3), dtype=int)
    edge_index_arr = np.zeros((len(triangle_list), 3), dtype=int)

    for i, triangle in enumerate(triangle_list):
        vertex_index_arr[i] = triangle.vertex_indexes
        edge_index_arr[i] = triangle.edge_indexes

    return vertex_index_arr, edge_index_arr
