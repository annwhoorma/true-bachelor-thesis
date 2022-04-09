import networkx as nx
from explainer import Explainer
from labels import TestLabel, Label1, Label2, Label3
from dataset import generate_mask
import globalenv
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def graph_to_nxgraph(weights_matrix) -> nx.Graph:
    graph = nx.from_numpy_matrix(weights_matrix, parallel_edges=False)
    return graph

def get_global_efficiency_score(g: nx.Graph):
    shortest_paths = [[nx.dijkstra_path_length(g, i, j, weight='weight') for i in range(5)] for j in range(5)]
    shortest_paths = np.array(shortest_paths)
    np.fill_diagonal(shortest_paths, 0)
    N = shortest_paths.shape[0]
    nodal_efficiency = (1.0 / (N-1)) * np.apply_along_axis(sum, 0, shortest_paths)
    return np.mean(nodal_efficiency)

def get_global_clustering_score(graph: nx.Graph) -> float:
    return nx.average_clustering(graph, weight="weight")

def calculate_for_nx(graph: TestLabel):
    nxgraph = graph_to_nxgraph(graph.A)
    eff = get_global_clustering_score(nxgraph)
    clus = get_global_efficiency_score(nxgraph)
    return eff, clus

def calculate_avg(num_graphs, explainer, d, segr):
    effs, cluss = [], []
    num_nodes = globalenv.NUM_NODES
    num_edges = num_nodes * (num_nodes - 1) # undirected
    for _ in range(num_graphs):
        mask = generate_mask(num_nodes, num_edges)
        d_low, d_high = explainer.get_l_distribution(d), explainer.get_h_distribution(d)
        graph = TestLabel(mask, 'test', d_low, d_high, segr)
        eff, clus = calculate_for_nx(graph)
        effs.append(eff); cluss.append(clus)
    return np.mean(effs), np.mean(cluss)

def create_csvs():
    graph_type = 'integration'
    graph_types = {
        'integration': Label1,
        'neutral': Label2,
        'segregation': Label3,
    }
    filename = graph_type
    with open(f'res_{filename}.csv', 'w') as f:
        f.write('cr,d,efficiency,clustering\n')
        explainer = Explainer()
        for cr in range(2, 10):
            if graph_type == 'segregation':
                globalenv.INNERCONNECTED_REGIONS = cr
            elif graph_type == 'integration':
                globalenv.CONNECTED_REGIONS = cr
            for d in globalenv.DS:
                eff, clus = calculate_avg(10, explainer, d, )
                f.write(f'{cr},{d},{eff},{clus}\n')

if __name__ == '__main__':
    create_csvs()