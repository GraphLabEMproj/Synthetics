from Synthetic3D.src.hard.structure.edge import Edge
from Synthetic3D.src.hard.structure.triangle import Triangle
from Synthetic3D.src.hard.structure.vertex import Vertex
from Synthetic3D.src.hard.structure.vector import Vector

def create_octahedron_with_main_point(center_point: Vector,
                                      rx: int,
                                      ry: int,
                                      rz: int,
                                      v_i: int = 0,
                                      e_i: int = 0
                                      ) -> [list[Vertex], list[Edge], list[Triangle]]:

    normal_len_max = max(rx,ry,rz)

    v1 = Vertex(center_point + Vector(rx, 0, 0), (normal_len_max/rx, 0, 0))
    v2 = Vertex(center_point - Vector(rx, 0, 0), (-normal_len_max/rx, 0, 0))

    v3 = Vertex(center_point + Vector(0, ry, 0), (0, normal_len_max/ry, 0))
    v4 = Vertex(center_point - Vector(0, ry, 0), (0, -normal_len_max/ry, 0))

    v5 = Vertex(center_point + Vector(0, 0, rz), (0, 0, normal_len_max/rz))
    v6 = Vertex(center_point - Vector(0, 0, rz), (0, 0, -normal_len_max/rz))

    vertices = [v1, v2, v3, v4, v5, v6]

    ed1 = Edge(0 + v_i, 2 + v_i) # v1 v3
    ed2 = Edge(0 + v_i, 3 + v_i) # v1 v4
    ed3 = Edge(0 + v_i, 4 + v_i) # v1 v5
    ed4 = Edge(0 + v_i, 5 + v_i) # v1 v6
    ed5 = Edge(1 + v_i, 2 + v_i) # v2 v3
    ed6 = Edge(1 + v_i, 3 + v_i) # v2 v4
    ed7 = Edge(1 + v_i, 4 + v_i) # v2 v5
    ed8 = Edge(1 + v_i, 5 + v_i) # v2 v6
    ed9 = Edge(2 + v_i, 4 + v_i) # v3 v5
    ed10= Edge(2 + v_i, 5 + v_i) # v3 v6
    ed11= Edge(3 + v_i, 4 + v_i) # v4 v5
    ed12= Edge(3 + v_i, 5 + v_i) # v4 v6

    edges = [ed1, ed2, ed3, ed4, ed5, ed6, ed7, ed8, ed9, ed10, ed11, ed12]

    tr1 = Triangle((0 + v_i, 2 + v_i, 4 + v_i), (0 + e_i, 2 + e_i, 8 + e_i))  # v1 v3 v5
    tr2 = Triangle((0 + v_i, 2 + v_i, 5 + v_i), (0 + e_i, 3 + e_i, 9 + e_i))  # v1 v3 v6
    tr3 = Triangle((0 + v_i, 3 + v_i, 4 + v_i), (1 + e_i, 2 + e_i, 10 + e_i)) # v1 v4 v5
    tr4 = Triangle((0 + v_i, 3 + v_i, 5 + v_i), (1 + e_i, 3 + e_i, 11 + e_i)) # v1 v4 v6

    tr5 = Triangle((1 + v_i, 2 + v_i, 4 + v_i), (4 + e_i, 6 + e_i, 8 + e_i))  # v2 v3 v5
    tr6 = Triangle((1 + v_i, 2 + v_i, 5 + v_i), (4 + e_i, 7 + e_i, 9 + e_i))  # v2 v3 v6
    tr7 = Triangle((1 + v_i, 3 + v_i, 4 + v_i), (5 + e_i, 6 + e_i, 10 + e_i)) # v2 v4 v5
    tr8 = Triangle((1 + v_i, 3 + v_i, 5 + v_i), (5 + e_i, 7 + e_i, 11 + e_i)) # v2 v4 v6

    triangles = [tr1, tr2, tr3, tr4, tr5, tr6, tr7, tr8]

    return vertices, edges, triangles
