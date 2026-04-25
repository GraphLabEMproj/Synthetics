from Synthetic3D.src.hard.structure.edge import Edge
from Synthetic3D.src.hard.structure.triangle import Triangle
from Synthetic3D.src.hard.structure.vertex import Vertex
from Synthetic3D.src.hard.structure.shells import FrameShell, OuterShell
from Synthetic3D.src.hard.rotate import rotate_3d, rotate_2d_by_z, quaternion_rotate_3d
from Synthetic3D.src.hard.structure.vector import Vector
from Synthetic3D.src.hard.random_params import get_rand_double_int_val, get_rand_int
from Synthetic3D.src.hard.split_curve import split_half_curve
from Synthetic3D.src.hard.vectors2angle import vector2angles_in_degrees, vector2angles_in_quaterion

######################################################################################################################## для имитации сфер лучше всего подходит исокаэдр (12 угольник)

class Section:
    """
    Класс являющийся переборкой вдоль тела. Имеет эллиптическую форму ориентированной в плоскости, проходящей через
    точку position и нормалью direction.

    :param
    ru       : int|(int:int) - задает радиус оси u в плоскости секции. 2 числа задают диапозон для случайного выбора.
    rv       : int|(int:int) - задает радиус оси v в плоскости секции. 2 числа задают диапозон для случайного выбора.
    angle    : float|(float, float) - задает угол в градусах для поворота контура внутри плоскости секции. 2 числа                  # на данный момент в реализации применяются только int !!!!!!!!!!!!!!
                                      задают диапозон для случайного выбора eukf/
    position : (int, int, int) - местоположение центра секции
    direction: (float, float, float) - вектор нормали секции

    :argument
    contour_list: list[Vertex] - в общем случае хранит 4 точки переборки
    """

    def __init__(self, position:Vector, direction:Vector, ru, rv, angle=(-22,22), reversed=1):
        print("position", position, "direction", direction)
        self.position = Vertex(position, direction) # хранит центральную точку и вектор нормали к плоскости секции
        assert abs(reversed) == 1, "reversed can be onli 1 or -1"
        self.reversed = reversed
        self.CreateSection(ru,rv,angle)

    def CreateSection(self, ru, rv, angle):
        use_ru = get_rand_double_int_val(ru)
        use_rv = get_rand_double_int_val(rv)

        use_angle = get_rand_int(angle)
        # создание точек в плоскости ху c единичными векторами (нормаль = (0, 0, 1))
        self.max_radius = max(use_ru[0], -use_ru[1], use_rv[0], -use_rv[1])

        # работаем по очереди
        t_contour_list = [
            Vertex((use_ru[0], 0, 0), (self.max_radius/use_ru[0], 0, 0)),
            Vertex((0, use_rv[0], 0), (0, self.max_radius/use_rv[0], 0)),
            Vertex((use_ru[1], 0, 0), (self.max_radius/use_ru[1], 0, 0)),
            Vertex((0, use_rv[1], 0), (0, self.max_radius/use_ru[1], 0)),
        ]

        # поворот точек в плоскости секции
        contour_list_into_section = []
        for vertex in t_contour_list:
            t_point = rotate_2d_by_z(vertex.position, use_angle)
            t_normal = rotate_2d_by_z(vertex.normal, use_angle)
            contour_list_into_section.append(Vertex(t_point, t_normal))

        # поворот секции на вектор нормали и смещении в нужное место
        self.contour_list = []
        '''
        angle_turn = vector2angles_in_degrees(Vector(0, 0, self.reversed), self.position.normal)
        for vertex in contour_list_into_section:
            point = rotate_3d(vertex.position, angle_turn)
            normal = rotate_3d(vertex.normal, angle_turn)
            self.contour_list.append(Vertex(point + self.position.position, normal))
        '''

        quaterion = vector2angles_in_quaterion(Vector(0, 0, self.reversed), self.position.normal)
        for vertex in contour_list_into_section:
            point = quaternion_rotate_3d(vertex.position, quaterion)
            normal = quaternion_rotate_3d(vertex.normal, quaterion)
            self.contour_list.append(Vertex(point + self.position.position, normal))

    @staticmethod
    def AddEdgeEndPoint(last_section_indexes:[list[int], list[int]],
                        new_vertex:Vertex,
                        storage_shell:FrameShell|OuterShell): # добавляет к предыдущей секции крышку
        new_p_i = storage_shell.get_vertex_count()
        storage_shell.add_vertex(new_vertex)

        vertex_indexes, edge_indexes = last_section_indexes

        v_i_1 = vertex_indexes[0] # v_i_1 = last_section_indexes[0] #######       v_1         +=
        v_i_2 = vertex_indexes[1] # v_i_2 = last_section_indexes[1] #######   v_4             -=
        v_i_3 = vertex_indexes[2] # v_i_3 = last_section_indexes[2] #######        pi  v_2    =+
        v_i_4 = vertex_indexes[3] # v_i_4 = last_section_indexes[3] #######       v_3         =-

        edge_start_index = storage_shell.get_edge_count()
        ed1 = Edge(v_i_1, new_p_i)
        ed2 = Edge(v_i_2, new_p_i)
        ed3 = Edge(v_i_3, new_p_i)
        ed4 = Edge(v_i_4, new_p_i)
        storage_shell.add_edges(ed1, ed2, ed3, ed4)

        tr1 = Triangle((v_i_1, v_i_2, new_p_i),
                       (edge_indexes[0], edge_start_index, edge_start_index+1))
        tr2 = Triangle((v_i_2, v_i_3, new_p_i),
                       (edge_indexes[1], edge_start_index + 1, edge_start_index + 2))
        tr3 = Triangle((v_i_3, v_i_4, new_p_i),
                        (edge_indexes[2], edge_start_index + 2, edge_start_index + 3))
        tr4 = Triangle((v_i_4, v_i_1, new_p_i),
                        (edge_indexes[3], edge_start_index + 3, edge_start_index))

        storage_shell.add_triangles(tr1, tr2, tr3, tr4)

        '''
        print("vertex_i", v_i_1, v_i_2, v_i_3, v_i_4, new_p_i)
        print("edges_i", edge_indexes + [edge_start_index+j for j in range(4)])

        for i in edge_indexes:
            print(storage_shell.edge_list[i])
        for j in range(4):
            print(storage_shell.edge_list[edge_start_index+j])
        '''
        for triangle in storage_shell.triangle_list:
            triangle.check_vertex_indices_in_indices_within_edges_in_edges_list(storage_shell.edge_list)


    @staticmethod
    def AddSection(last_section_indexes, new_section_indexes, storage_shell:FrameShell|OuterShell): # добавляет к предыдущей секции новую
        # LAST SECTION INDEXES
        vertex_indexes, edge_indexes = last_section_indexes
        v_i_1 = vertex_indexes[0]  # v_i_1 = last_section_indexes[0] #######       v_1         +=
        v_i_2 = vertex_indexes[1]  # v_i_2 = last_section_indexes[1] #######   v_4             -=
        v_i_3 = vertex_indexes[2]  # v_i_3 = last_section_indexes[2] #######        pi  v_2    =+
        v_i_4 = vertex_indexes[3]  # v_i_4 = last_section_indexes[3] #######       v_3         =-

        e_i_1 = edge_indexes[0]
        e_i_2 = edge_indexes[1]
        e_i_3 = edge_indexes[2]
        e_i_4 = edge_indexes[3]

        # NEW SECTION INDEXES
        new_vertex_indexes, new_edge_indexes = new_section_indexes
        new_v_i_1 = new_vertex_indexes[0]
        new_v_i_2 = new_vertex_indexes[1]
        new_v_i_3 = new_vertex_indexes[2]
        new_v_i_4 = new_vertex_indexes[3]

        new_e_i_1 = new_edge_indexes[0]
        new_e_i_2 = new_edge_indexes[1]
        new_e_i_3 = new_edge_indexes[2]
        new_e_i_4 = new_edge_indexes[3]

        # создание ребер вдоль 2 секций (1-new1, 2-new2 и т.д)
        stringer_index_1_1 = storage_shell.get_edge_count()
        st_ed_1_1 = Edge(v_i_1, new_v_i_1)
        st_ed_2_2 = Edge(v_i_2, new_v_i_2)
        st_ed_3_3 = Edge(v_i_3, new_v_i_3)
        st_ed_4_4 = Edge(v_i_4, new_v_i_4)
        storage_shell.add_edges(st_ed_1_1, st_ed_2_2, st_ed_3_3, st_ed_4_4)
        stringer_index_2_2 = stringer_index_1_1 + 1
        stringer_index_3_3 = stringer_index_1_1 + 2
        stringer_index_4_4 = stringer_index_1_1 + 3

        # Создание промежуточных точек для разбиения квадратного полигона на треугольники
        half_v1_pos, half_v1_norm = split_half_curve(*storage_shell.get_position_and_normal(v_i_1),
                                                     *storage_shell.get_position_and_normal(new_v_i_1))
        half_v2_pos, half_v2_norm = split_half_curve(*storage_shell.get_position_and_normal(v_i_2),
                                                     *storage_shell.get_position_and_normal(new_v_i_2))
        half_v3_pos, half_v3_norm = split_half_curve(*storage_shell.get_position_and_normal(v_i_3),
                                                     *storage_shell.get_position_and_normal(new_v_i_3))
        half_v4_pos, half_v4_norm = split_half_curve(*storage_shell.get_position_and_normal(v_i_4),
                                                     *storage_shell.get_position_and_normal(new_v_i_4))

        now_new_vertex_index = storage_shell.get_vertex_count()
        st_ed_1_1.vertex_mean = now_new_vertex_index
        storage_shell.add_vertex(half_v1_pos, half_v1_norm)
        st_ed_2_2.vertex_mean = now_new_vertex_index + 1
        storage_shell.add_vertex(half_v2_pos, half_v2_norm)
        st_ed_3_3.vertex_mean = now_new_vertex_index + 2
        storage_shell.add_vertex(half_v3_pos, half_v3_norm)
        st_ed_4_4.vertex_mean = now_new_vertex_index + 3
        storage_shell.add_vertex(half_v4_pos, half_v4_norm)

        vertex_1_2 = Vertex(*split_half_curve(half_v1_pos, half_v1_norm,
                                              half_v2_pos, half_v2_norm))
        vertex_2_3 = Vertex(*split_half_curve(half_v2_pos, half_v2_norm,
                                              half_v3_pos, half_v3_norm))
        vertex_3_4 = Vertex(*split_half_curve(half_v3_pos, half_v3_norm,
                                              half_v4_pos, half_v4_norm))
        vertex_4_1 = Vertex(*split_half_curve(half_v1_pos, half_v1_norm,
                                              half_v4_pos, half_v4_norm))

        # добавить полигон как добавление точки
        Section.AddEdgeEndPoint([[v_i_1, v_i_2, new_v_i_2, new_v_i_1],
                                 [e_i_1, stringer_index_2_2, new_e_i_1, stringer_index_1_1]],
                                vertex_1_2,
                                storage_shell)

        Section.AddEdgeEndPoint([[v_i_2, v_i_3, new_v_i_3, new_v_i_2],
                                [e_i_2, stringer_index_3_3, new_e_i_2, stringer_index_2_2]],
                                vertex_2_3,
                                storage_shell)

        Section.AddEdgeEndPoint([[v_i_3, v_i_4, new_v_i_4, new_v_i_3],
                                 [e_i_3, stringer_index_4_4, new_e_i_3, stringer_index_3_3]],
                                vertex_3_4,
                                storage_shell)

        Section.AddEdgeEndPoint([[v_i_4, v_i_1, new_v_i_1, new_v_i_4],
                                 [e_i_4, stringer_index_1_1, new_e_i_4, stringer_index_4_4]],
                                vertex_4_1,
                                storage_shell)
