import networkx as nx
from tqdm import tqdm
from explainer import GaussianExplainer, UniformExplainer
from labels import Label1, Label2, Label3
from dataset import generate_mask
import globalenv
import numpy as np
from pathlib import Path
import bct

import warnings

from labels.label_interface import LabelInterface
warnings.filterwarnings("ignore")

def graph_to_nxgraph(weights_matrix) -> nx.Graph:
    graph = nx.from_numpy_matrix(weights_matrix, parallel_edges=False)
    return graph

def get_global_efficiency_score(num_nodes, g: nx.Graph) -> float:
    shortest_paths = np.zeros((num_nodes, num_nodes))
    for j in range(num_nodes):
        for i in range(j+1, num_nodes):
            try:
                shortest_paths[i, j] = 1 / nx.dijkstra_path_length(g, i, j, weight='weight')
            except:
                # if node is unreachable
                shortest_paths[j, i] = 0
    shortest_paths += shortest_paths.T
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

def calculate_average_metrics(num_graphs: int, explainer, d: int, ClassLabel: LabelInterface, save_graphs: bool, to_save: str) -> 'tuple[float]':
    effs, cluss, moduls, particips = [], [], [], []
    num_nodes = globalenv.NUM_NODES
    num_edges = num_nodes * (num_nodes - 1) # undirected
    for i in range(num_graphs):
        mask = generate_mask(num_nodes, num_edges)
        d_low, d_high = explainer.get_l_distribution(d), explainer.get_h_distribution(d)
        graph = ClassLabel(mask, str(i), d_low, d_high, globalenv.DIST_TYPE)
        graph_regions = graph.regions
        if save_graphs:
            graph.to_json(Path(to_save), save_adj=True)
        # calculate modularity and participation
        modularity, particip = calculate_with_bct(graph, graph_regions)
        moduls.append(modularity); particips.append(particip)
        # calculate clustering and efficiency
        nxgraph = graph_to_nxgraph(graph.A)
        clustering = get_global_clustering_score(nxgraph)
        efficiency = get_global_efficiency_score(num_nodes, nxgraph)
        effs.append(efficiency)
        cluss.append(clustering)
    return np.mean(effs), np.mean(cluss), np.mean(moduls), np.mean(particips)
    # return np.mean(moduls), np.mean(particips)

def create_index_row(metrics):
    index_row = 'type,cr,n,'
    for metric in metrics:
        index_row += metric.value + ','
    return index_row + '\n'

# for labels, not for fully-connected ones
def create_csvs():
    metrics = [
        globalenv.WeightedMetric.GlobalEfficiency,
        globalenv.WeightedMetric.GlobalClustering,
        globalenv.WeightedMetric.GlobalModularity,
        globalenv.WeightedMetric.GlobalParticipation
    ]
    global_path = 'graphs_for_metrics'
    Path(global_path).mkdir(exist_ok=True)
    index_row = create_index_row(metrics)
    print(index_row)
    all_graph_types = [globalenv.CalculateMetrics.segregation, globalenv.CalculateMetrics.integration]
    explainer_name = globalenv.DIST_TYPE.value
    with open(f'efficiency_for_labels_{explainer_name}.csv', 'a') as f:
        for graph_type in all_graph_types:
            global_path = f'{global_path}/{graph_type.value}'
            Path(global_path).mkdir(exist_ok=True)
            f.write(index_row)
            explainer = UniformExplainer() if globalenv.DIST_TYPE is globalenv.DistributionType.Uniform else GaussianExplainer()
            for i in tqdm(globalenv.REGIONS_RANGE, desc=graph_type.value):
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
                for n in tqdm(globalenv.NS, desc=f'(i)cr={i}'):
                    local_to_save = f'{to_save}/n={n}'
                    Path(local_to_save).mkdir(exist_ok=True)
                    # set save_graphs to True if you want to visualize the graphs later
                    eff, clus, mod, part = calculate_average_metrics(2, explainer, n, ClassLabel, save_graphs=False, to_save=local_to_save)
                    f.write(f'{graph_type.value},{i},{n},{eff},{clus},{mod},{part}\n')
                    # mod, part = calculate_average_metrics(10, explainer, n, ClassLabel, save_graphs=False, to_save=local_to_save)
                    # f.write(f'{i},{n},{mod},{part}\n')

if __name__ == '__main__':
    create_csvs()