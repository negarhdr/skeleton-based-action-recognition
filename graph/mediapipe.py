import sys

sys.path.extend(['../'])
from graph import tools

num_node = 33
self_link = [(i, i) for i in range(num_node)]
inward = [(0, 1), (0, 4), (1, 2), (2, 3), (3, 7), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                    (11, 23), (12, 14), (12, 24), (13, 15), (14, 16), (15, 21), (15, 17), (15, 19), (16, 22), (16, 18),
                    (16, 20), (17, 19), (18, 20), (23, 24), (23, 25), (24, 26), (25, 27),
                    (26, 28), (27, 29), (27, 31), (28, 30), (28, 32), (29, 31), (30, 32)]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
