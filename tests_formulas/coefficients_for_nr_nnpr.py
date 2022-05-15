import random
from test_bct import *
import numpy as np
from pathlib import Path
from math import ceil
import networkx as nx



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

# старые метрики
def generate_cluster_effic_for_nn_nnpr():
    all_num_regions = range(2, 13)
    all_num_nodes_per_region = range(2, 8)
    all_net_types = [NetworkType.Integration, NetworkType.Segregation]
    with open('cluster_effic_nn_nnpr.csv', 'w') as f:
        f.write('net_type,num_regions,num_nodes_per_region,clustering,efficiency\n')
        for net_type in all_net_types:
            print(net_type.value)
            for num_regions in all_num_regions:
                print(f'nr: {num_regions}')
                for num_nodes_per_region in all_num_nodes_per_region:
                    _, adj = create_network(num_regions, num_nodes_per_region, net_type)
                    clustering, efficiency = calculate_old(adj)
                    f.write(f'{net_type.value},{num_regions},{num_nodes_per_region},{clustering},{efficiency}\n')



def remove_edge(adj, regions, k_edges, ne_to_leave_zeros, in_same_region: bool):
    if in_same_region:
        stop_cond = lambda i, j: are_in_one_region(i, j, regions)
    else:
        stop_cond =lambda i, j: not are_in_one_region(i, j, regions)
    while k_edges > 0:
        all_zeros_i, all_zeros_j = (adj == 1).nonzero()
        if len(all_zeros_i) <= ne_to_leave_zeros:
            return adj
        idx = random.choice(range(len(all_zeros_i)))
        i, j = all_zeros_i[idx], all_zeros_j[idx]
        if stop_cond(i, j) and i != j:
            adj[i, j] = 0
            adj[j, i] = 0
            k_edges -= 1
    return adj

def add_edge(adj, regions, k_edges, in_same_region: bool):
    nr = adj.shape[0]
    if in_same_region:
        stop_cond = lambda i, j: are_in_one_region(i, j, regions)
    else:
        stop_cond =lambda i, j: not are_in_one_region(i, j, regions)
    while k_edges > 0 and np.sum(adj) < nr * (nr - 1):
        all_zeros_i, all_zeros_j = (adj == 0).nonzero()
        idx = random.choice(range(len(all_zeros_i)))
        i, j = all_zeros_i[idx], all_zeros_j[idx]
        if stop_cond(i, j) and i != j:
            adj[i, j] = 1
            adj[j, i] = 1
            k_edges -= 1
    return adj


def add_integrated_connection(adj: np.ndarray, regions, k_edges=1):
    return add_edge(adj, regions, k_edges, in_same_region=False)

def add_segregated_connection(adj: np.ndarray, regions, k_edges=1):
    return add_edge(adj, regions, k_edges, in_same_region=True)

def remove_segregated_connection(adj: np.array, regions, ne_to_leave, k_edges=1):
    return remove_edge(adj, regions, k_edges, ne_to_leave, in_same_region=True)


def test_generate_edge():
    net_type = NetworkType.Segregation
    regions, adj = create_network(2, 3, net_type)
    nr = adj.shape[0]
    print(adj, regions, sep='\n')
    # while the matrix has at least one zero not in the main diagonal
    while np.sum(adj) < nr * (nr - 1):
        if net_type == NetworkType.Integration:
            adj = add_segregated_connection(adj, regions, k_edges=2)
        elif net_type == NetworkType.Segregation:
            adj = add_integrated_connection(adj, regions, k_edges=2)
        print(adj, end='\n\n')

def test_remove_edge():
    net_type = NetworkType.FullyConnected
    nr = 3
    nnpr = 3
    ne = (nr * nnpr) ** 2
    regions, adj = create_network(nr, nnpr, net_type)
    nn_to_leave = (nnpr ** 2) * nr
    print(adj, regions, sep='\n')
    while np.sum(adj) > ne - nn_to_leave:
        adj = remove_segregated_connection(adj, regions, k_edges=1)
    print(adj, end='\n\n')


def get_k_edges(nr, adj, filled, num_steps=100):
    '''
    calculates the ratio of edges to change at every iterations
    '''
    num_places_to_fill = len((adj == filled).nonzero()[0]) - adj.shape[0] # all zeros without the main diagonal
    # print(f'num_places_to_fill: {num_places_to_fill}')
    return max(ceil((num_places_to_fill) / (num_steps * 2)), 1)

def get_k_edgesv2(nr, nnpr, num_steps=100):
    '''
    calculates the ratio of edges to change at every iterations
    '''
    num_places_to_fill = nr * (nnpr ** 2) - nr * nnpr # all zeros without the main diagonal
    # print(f'num_places_to_fill: {num_places_to_fill}')
    return max(ceil((num_places_to_fill) / (num_steps * 2)), 1)


def from_integrated_to_segregated_for_nn_nnpr(nr=10, nnpr=40, filename=None):
    if filename is None:
        filename = f'coeffs_i2s_nr_nnpr_{nr/nnpr}.csv'
    net_type = NetworkType.Segregation
    regions, adj = create_network(nr, nnpr, net_type)
    k_edges = get_k_edges(nr, adj, filled=0)
    print(f'original k_edges: {k_edges}')
    nn = adj.shape[0]
    ne = nn ** 2
    kci = [idx for idx, region in enumerate(regions) for _ in region]
    step_num = 0
    with open(filename, 'w') as f:
        f.write('net_type,step,bct_modularity,bct_participation\n')
        # segregated to fully-connected
        while np.sum(adj) < nn * (nn - 1):
            adj = add_integrated_connection(adj, regions, k_edges)
            modularity, participation = calculate_coeffs(adj, kci)
            f.write(f'{net_type.value},{step_num},{modularity},{participation}\n')
            step_num += 1
        print('finished s->fc')
        # fully-connected
        net_type = NetworkType.FullyConnected
        modularity, participation = calculate_coeffs(adj, kci)
        f.write(f'{net_type.value},{step_num},{modularity},{participation}\n')
        step_num += 1
        print(f'finished fc at step: {step_num}')
        net_type = NetworkType.Segregation
        ne_to_leave = ne - nr * (nnpr ** 2)
        k_edges = get_k_edgesv2(nr, nnpr)
        print(f'new k_edges: {k_edges}')
        # fully-connected to integrated
        while np.sum(adj) > ne_to_leave:
            adj = remove_segregated_connection(adj, regions, ne_to_leave, k_edges)
            modularity, participation = calculate_coeffs(adj, kci)
            f.write(f'{net_type.value},{step_num},{modularity},{participation}\n')
            step_num += 1
        print('finished fc->i')


# если хотим получить соотношения, получить их разными способами и усреднять
# nnpr зафиксировать: [5, 10, (15), 20], для него построить несколько картинок per metric
# на одном - меняется nr [5, 10, (15), 20]
def from_integrated_to_segregated_for_fixed_nnprs():
    folder = 'for_fixed_nnpr/'
    num_steps = 15
    Path(folder).mkdir(exist_ok=True)
    for nnpr in [6, 12, 18]:
        filename = f'{folder}/nnpr={nnpr}.csv'
        with open(filename, 'w') as f:
            f.write('net_type,nr,step,bct_modularity,bct_participation\n')
            for nr in [6, 12, 18]:
                print(f'nnpr = {nnpr} | nr = {nr}')
                net_type = NetworkType.Segregation
                regions, adj = create_network(nr, nnpr, net_type)
                k_edges = get_k_edges(nr, adj, filled=0, num_steps=num_steps)
                print(f'>>> num edges for S->FC: {k_edges}')
                nn = adj.shape[0]
                ne = nn ** 2
                kci = [idx for idx, region in enumerate(regions) for _ in region]
                step_num = 0
                # segregated to fully-connected
                while np.sum(adj) < nn * (nn - 1):
                    adj = add_integrated_connection(adj, regions, k_edges)
                    modularity, participation = calculate_coeffs(adj, kci)
                    f.write(f'{net_type.value},{nr},{step_num},{modularity},{participation}\n')
                    step_num += 1
                # fully-connected
                net_type = NetworkType.FullyConnected
                modularity, participation = calculate_coeffs(adj, kci)
                f.write(f'{net_type.value},{nr},{step_num},{modularity},{participation}\n')
                step_num += 1
                print(f'>>> finished fc at step: {step_num}')
                net_type = NetworkType.Segregation
                ne_to_leave = ne - nr * (nnpr ** 2)
                k_edges = get_k_edgesv2(nr, nnpr, num_steps)
                print(f'>>> num edges for FC->I: {k_edges}')
                # fully-connected to integrated
                while np.sum(adj) > ne_to_leave:
                    adj = remove_segregated_connection(adj, regions, ne_to_leave, k_edges)
                    modularity, participation = calculate_coeffs(adj, kci)
                    f.write(f'{net_type.value},{nr},{step_num},{modularity},{participation}\n')
                    step_num += 1
                print(f'>>> finished at step {step_num}', end='\n')

def calculate_old(adj):
    nxgraph = graph_to_nxgraph(adj)
    clustering = get_global_clustering_score(nxgraph)
    num_nodes = adj.shape[0]
    efficiency = get_global_efficiency_score(num_nodes, nxgraph)
    return clustering, efficiency

def from_segreg_to_integrated_for_fixed_nnprs_with_old_metrics():
    folder = 'for_fixed_nnpr_with_old/'
    num_steps = 15
    Path(folder).mkdir(exist_ok=True)
    for nnpr in [10]: # it doesnt affect the result
        filename = f'{folder}/nnpr={nnpr}.csv'
        with open(filename, 'w') as f:
            f.write('net_type,nr,step,modularity,participation,clustering,efficiency\n')
            for nr in [6, 12, 18]:
                print(f'nnpr = {nnpr} | nr = {nr}')
                net_type = NetworkType.Segregation
                regions, adj = create_network(nr, nnpr, net_type)
                k_edges = get_k_edges(nr, adj, filled=0, num_steps=num_steps)
                print(f'>>> num edges for S->FC: {k_edges}')
                nn = adj.shape[0]
                ne = nn ** 2
                kci = [idx for idx, region in enumerate(regions) for _ in region]
                step_num = 0
                # segregated to fully-connected
                while np.sum(adj) < nn * (nn - 1):
                    adj = add_integrated_connection(adj, regions, k_edges)
                    modularity, participation = calculate_coeffs(adj, kci)
                    clustering, efficiency = calculate_old(adj)
                    f.write(f'{net_type.value},{nr},{step_num},{modularity},{participation},{clustering},{efficiency}\n')
                    step_num += 1
                # fully-connected
                net_type = NetworkType.FullyConnected
                modularity, participation = calculate_coeffs(adj, kci)
                clustering, efficiency = calculate_old(adj)
                f.write(f'{net_type.value},{nr},{step_num},{modularity},{participation},{clustering},{efficiency}\n')
                step_num += 1
                print(f'>>> finished fc at step: {step_num}')
                net_type = NetworkType.Segregation
                ne_to_leave = ne - nr * (nnpr ** 2)
                k_edges = get_k_edgesv2(nr, nnpr, num_steps)
                print(f'>>> num edges for FC->I: {k_edges}')
                # fully-connected to integrated
                while np.sum(adj) > ne_to_leave:
                    adj = remove_segregated_connection(adj, regions, ne_to_leave, k_edges)
                    modularity, participation = calculate_coeffs(adj, kci)
                    clustering, efficiency = calculate_old(adj)
                    f.write(f'{net_type.value},{nr},{step_num},{modularity},{participation},{clustering},{efficiency}\n')
                    step_num += 1
                print(f'>>> finished at step {step_num}', end='\n')


if __name__ == '__main__':
    generate_cluster_effic_for_nn_nnpr()
