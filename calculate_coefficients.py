import networkx as nx
from tqdm import tqdm
from explainer import Explainer
from labels import Label1, Label2, Label3
from dataset import generate_mask
import globalenv
import numpy as np
import networkx.algorithms.community as nx_comm
from pathlib import Path
import bct

import warnings

from labels.label_interface import LabelInterface
warnings.filterwarnings("ignore")

def graph_to_nxgraph(weights_matrix) -> nx.Graph:
    graph = nx.from_numpy_matrix(weights_matrix, parallel_edges=False)
    return graph

def get_global_efficiency_score(g: nx.Graph) -> float:
    shortest_paths = [[nx.dijkstra_path_length(g, i, j, weight='weight') for i in range(globalenv.NUM_NODES)] for j in range(globalenv.NUM_NODES)]
    shortest_paths = np.array(shortest_paths)
    np.fill_diagonal(shortest_paths, 0)
    N = shortest_paths.shape[0]
    nodal_efficiency = (1.0 / (N-1)) * np.apply_along_axis(sum, 0, shortest_paths)
    return np.mean(nodal_efficiency)

def get_global_clustering_score(graph: nx.Graph) -> float:
    return nx.average_clustering(graph, weight="weight")

def calculate_with_bct(graph: LabelInterface, regions: 'list[set[int]]') -> 'tuple[float]':
    regions = [region_num for region_num, region in enumerate(regions) for _ in region]
    modularity = bct.modularity_und(graph.A, kci=regions)[1]
    participation = np.mean(bct.participation_coef(graph.A, ci=regions))
    return modularity, participation

def are_in_the_same_region(u, v, regions):
    for region in regions:
        if u in region and v in region:
            return True
    return False

def get_modularity_coefficient(graph: nx.Graph, regions: 'list[set[int]]') -> float:
    weight = "weight"
    q_is = []

    out_degree = dict(graph.degree(weight=weight))
    deg_sum = sum(out_degree.values()) / 2
    ks = {v: sum(graph.adj[i][u]['weight'] for (i, u) in graph.edges() if i == v or u == v) for v in graph.nodes()}

    for (i, j) in graph.edges():
        w_ij = graph.adj[i][j]['weight']
        k_i = ks[i]
        k_j = ks[j]
        term = w_ij - k_i * k_j / deg_sum
        sigmami_sigmamj = int(are_in_the_same_region(i, j, regions))
        q_i = term * sigmami_sigmamj
        q_is.append(q_i)
    return np.sum(q_is) / deg_sum

def get_participation_coefficient(graph: nx.Graph, regions: 'list[set[int]]'):
    # если связи внутри сегр. регионов усиливаются, этот коэфф падает
    global_pc = []
    ks = {v: sum(graph.adj[i][u]['weight'] for (i, u) in graph.edges() if i == v or u == v) for v in graph.nodes()}
    for v in graph.nodes():
        k_i = ks[v]
        temp = []
        for region in regions:
            ki_m = sum(graph.adj[v][u]['weight'] for u in region if (v, u) in graph.edges())
            term = ki_m / k_i
            temp.append(term ** 2)
        pc_i = 1 - sum(temp)
        global_pc.append(pc_i)
    return np.mean(global_pc)

def calculate_for_nx(graph: LabelInterface, regions: 'list[set[int]]') -> 'tuple[float]':
    nxgraph = graph_to_nxgraph(graph.A)
    eff = get_global_clustering_score(nxgraph)
    clus = get_global_efficiency_score(nxgraph)
    modularity = get_modularity_coefficient(nxgraph, regions)
    participation = get_participation_coefficient(nxgraph, regions)
    return eff, clus, modularity, participation

def calculate_average_metrics(num_graphs: int, explainer: Explainer, d: int, ClassLabel: LabelInterface, save_graphs: bool, to_save: str) -> 'tuple[float]':
    effs, cluss, moduls, particips = [], [], [], []
    num_nodes = globalenv.NUM_NODES
    num_edges = num_nodes * (num_nodes - 1) # undirected
    for i in range(num_graphs):
        mask = generate_mask(num_nodes, num_edges)
        d_low, d_high = explainer.get_l_distribution(d), explainer.get_h_distribution(d)
        graph = ClassLabel(mask, str(i), d_low, d_high)
        graph_regions = graph.regions
        if save_graphs:
            graph.to_json(Path(to_save), save_adj=True)
        modularity, particip = calculate_with_bct(graph, graph_regions)
        moduls.append(modularity); particips.append(particip)
    return np.mean(moduls), np.mean(particips)

def create_index_row(metrics):
    index_row = 'cr,d,'
    for metric in metrics:
        index_row += metric.value + ','
    return index_row + '\n'

def create_csvs():
    metrics = {
        # globalenv.WeightedMetric.GlobalEfficiency,
        # globalenv.WeightedMetric.GlobalClustering,
        globalenv.WeightedMetric.GlobalModularity,
        globalenv.WeightedMetric.GlobalParticipation
    }
    global_path = 'graphs_for_metrics'
    Path(global_path).mkdir(exist_ok=True)
    index_row = create_index_row(metrics)
    all_graph_types = [globalenv.CalculateMetrics.segregation, globalenv.CalculateMetrics.integration]
    for graph_type in all_graph_types:
        global_path = f'{global_path}/{graph_type.value}'
        Path(global_path).mkdir(exist_ok=True)
        filename = graph_type.value
        with open(f'res_{filename}.csv', 'w') as f:
            f.write(index_row)
            explainer = Explainer()
            for i in tqdm(globalenv.CS, desc=graph_type.value):
                if graph_type == globalenv.CalculateMetrics.segregation:
                    globalenv.INNERCONNECTED_REGIONS = i
                    cr = globalenv.INNERCONNECTED_REGIONS
                    ClassLabel = Label3
                    to_save = f'{global_path}/icr={cr}'
                elif graph_type == globalenv.CalculateMetrics.integration:
                    globalenv.CONNECTED_REGIONS = i
                    icr = globalenv.CONNECTED_REGIONS
                    ClassLabel = Label1
                    to_save = f'{global_path}/cr={icr}'
                elif graph_type == globalenv.CalculateMetrics.neutral:
                    ClassLabel = Label2
                    to_save = f'{global_path}'
                Path(to_save).mkdir(exist_ok=True)
                for d in tqdm(globalenv.DS, desc=f'(i)cr={i}'):
                    local_to_save = f'{to_save}/d={d}'
                    Path(local_to_save).mkdir(exist_ok=True)
                    mod, part = calculate_average_metrics(20, explainer, d, ClassLabel, save_graphs=False, to_save=local_to_save)
                    f.write(f'{i},{d},{mod},{part}\n')

if __name__ == '__main__':
    create_csvs()