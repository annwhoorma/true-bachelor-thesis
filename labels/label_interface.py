import numpy as np
import json
import random

import globalenv

def calculate_num_edges(num_nodes):
    '''
    calculate maximum number of edges for
    '''
    return num_nodes * (num_nodes - 1)

class LabelInterface:
    num_nodes = globalenv.NUM_NODES
    num_regions = globalenv.NUM_REGIONS
    nodes_per_region = globalenv.NUM_NODES_PER_REGION
    def __init__(self, mask, label_name, l_dist, h_dist):
        self.num_edges = calculate_num_edges(self.num_nodes)
        self.name = label_name
        self.mask = mask
        self.l_dist = l_dist
        self.h_dist = h_dist
        self.A = np.zeros((self.num_nodes, self.num_nodes))
        self.all_possible_pairs = [(i, j) for i in range(self.num_regions) for j in range(i, self.num_regions)]

    def _choose_random_regions(self, regions, number):
        return random.sample(regions, k=number)

    def _make_symmetric(self, matrix):
        return (matrix + matrix.T) / 2

    def _generate_regions(self):
        n = self.num_regions
        npr = self.nodes_per_region
        self.regions = [range(i*npr, (i+1)*npr) for i in range(0, n)]
        self.weights = {
            'low': self.l_dist, 'high': self.h_dist
        }
        self.connections = {}

    def _fill_regions_with_values(self, r1, r2, values):
        future_region = np.reshape(values, (len(r1), len(r2)))
        # it should be okay if there are no overlaps between regions
        self.A[r1[0]:r1[-1]+1, r2[0]:r2[-1]+1] = future_region
        self.A[r2[0]:r2[-1]+1, r1[0]:r1[-1]+1] = future_region

    def _generate_patterns(self):
        for conn, regss in self.connections.items():
            ws = self.weights[conn]
            for regs in regss:
                num_values = len(self.regions[regs[0]]) * len(self.regions[regs[1]])
                values = np.random.normal(ws['mu'], ws['sigma'], num_values)
                values[values < globalenv.MIN_VALUE] = globalenv.MIN_VALUE
                values[values > globalenv.MAX_VALUE] = globalenv.MAX_VALUE
                self._fill_regions_with_values(self.regions[regs[0]], self.regions[regs[1]], values)
        # normalize the matrix
        self.A /= np.max(self.A)

    def get_connected_regions(self):
        pass # connections

    def _generate_edge_index_coo_format(self):
        self.edge_index_coo = np.array(self.mask.nonzero())

    def _generate_edge_attr(self):
        edge_index_T = np.transpose(self.edge_index_coo)
        edge_attr = []
        for edge in edge_index_T:
            feat = self.A[edge[0], edge[1]]
            edge_attr.append(feat)
        self.edge_attr = np.array(edge_attr)

    def to_json(self, path, save_adj=True):
        '''
        save_adj=True when we need to visualize
        '''
        self._generate_edge_index_coo_format()
        self._generate_edge_attr()
        if save_adj:
            data = {
                # 'num_nodes': self.num_nodes,
                'adj': json.dumps(self.A.tolist()),
                # 'edge_attr': json.dumps(self.edge_attr.tolist()),
                # 'edge_index': json.dumps(self.edge_index_coo.tolist())
            }
        else:
            data = {
                'num_nodes': self.num_nodes,
                'edge_attr': json.dumps(self.edge_attr.tolist()),
                'edge_index': json.dumps(self.edge_index_coo.tolist())
            }
        json.dump(data, open(path/f'{self.name}.json', 'w'))


if __name__ == '__main__':
    pass