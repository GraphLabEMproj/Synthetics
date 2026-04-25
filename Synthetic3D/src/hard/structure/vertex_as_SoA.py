import numpy as np
from copy import deepcopy

from Synthetic3D.src.hard.structure.vertex import Vertex
from Synthetic3D.src.hard.structure.expandable_array import ExpandableVectorArray

class VertexAsSoA:
    def __init__(self,
                 positions=None,
                 normales=None):
        self.positions = ExpandableVectorArray() if positions is None else positions
        self.normales  = ExpandableVectorArray() if normales is None else normales

    def __deepcopy__(self, memo):
        new_vertexes = VertexAsSoA()
        new_vertexes.positions = deepcopy(self.positions, memo)
        new_vertexes.normales = deepcopy(self.normales, memo)
        return new_vertexes

    def add_vertex(self, *args):
        if len(args) == 1:
            if isinstance(args[0], Vertex):
                self.positions.append(args[0].position)
                self.normales.append(args[0].normal)
            else:
                raise ValueError("You can add either one Vertex or two Vectors")

        elif len(args) == 2:
            assert len(args[0]) == 3, "Coordinates must have length 3"
            assert len(args[1]) == 3, "Normals must have length 3"

            self.positions.append(args[0])
            self.normales.append(args[1])
        else:
            raise ValueError("You can add either one Vertex or two Vectors")

    def add_vertex_list(self, list_of_vertex):
        for vertex in list_of_vertex:
            if isinstance(vertex, Vertex):
                self.positions.append(vertex.position)
                self.normales.append(vertex.normal)
            else:
                raise ValueError("Only list of Vertex can be added to Shell")

    def get_vertex(self, index):
        return Vertex(self.positions[index],
                      self.normales[index])

    def get_positions(self):
        return self.positions

    def get_positions_gen(self):
        return self.positions.get_elemens()

    def get_normales(self):
        return self.normales
    def get_normales_gen(self):
        return self.normales.get_elemens()

    def get_position_and_normal(self, index):
        return self.positions[index], self.normales[index]

    def size(self):
        return len(self.positions)

    def shift_coords_by_value(self, vector):
        self.positions.shift_by_vector(vector)

    def rotate_coords_and_normals(self, angle):
        self.positions.rotate(angle)
        self.normales.rotate(angle)

    def transform_coords2int(self):
        self.positions = self.positions.to_int()

    def get_union_vertex_list(self):
        list_of_vertex = [Vertex(*val) for val in zip(self.positions.get_elemens(), self.normales.get_elemens())]
        return list_of_vertex

    def __iadd__(self, other):
        self.positions += other.positions
        self.normales += other.normales

    def add_vertexes(self, other):
        new_shell = deepcopy(self)
        new_shell += other
        return new_shell

    def __add__(self, other):
        return self.add_vertexes(other)

    def reverse(self):
        self.positions.reverse()
        self.normales.reverse()
