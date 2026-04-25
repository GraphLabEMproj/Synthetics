from Synthetic3D.src.hard.structure.vector import Vector

class Vertex:
    def __init__(self, vertex : tuple|list|Vector, normal : tuple|list|Vector = (0, 0, 0)):
       # координата точки
        self.position = Vector(vertex)
        # вектор нормали
        self.normal = Vector(normal)

    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return f"vertex: {self.position}, normal: [{', '.join([f'{val:.04}' for val in self.normal])}]"

def Vertex_list_to_2_list(vertex_list: list[Vertex]):
    pos_list = []
    norm_list = []

    for vertex in vertex_list:
        pos_list.append(vertex.position)
        norm_list.append(vertex.normal)

    return pos_list, norm_list

def transform_coords_to_int(vertex_list: list[Vertex]):
    new_list = []
    for vertex in vertex_list:
        new_list.append(Vertex(vertex.position.to_int(),
                               vertex.normal))
    return new_list

def get_int_coords_from_vertex_list(vertex_list: list[Vertex]):
    new_list = []
    for vertex in vertex_list:
        new_list.append(vertex.position.to_int())
    return new_list
