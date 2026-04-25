class Edge:
    def __init__(self, v1_index, v2_index):
        self.v1_index = v1_index
        self.v2_index = v2_index
        self.vertex_mean = None

        self.new_fist_index_pair = None

    def __str__(self):
        str = "Edge:\n" +\
              f"\tvertex_index_1: {self.v1_index}\n" +\
              f"\tvertex_index_2: {self.v2_index}\n" +\
              f"\tvertex_mean_index: {"None" if self.vertex_mean is None else self.vertex_mean}" + \
              f"\tnew_fist_index_pair: {"None" if self.new_fist_index_pair is None else self.new_fist_index_pair}\n"
        return str
