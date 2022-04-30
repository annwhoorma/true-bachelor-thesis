import bct
import numpy as np
from enum import Enum

class NetworkType(Enum):
    Segregation = 'segregation'
    Integration = 'integration'
    FullyConnected = 'fully-connected'

def are_in_one_region(i: int, j: int, regions: list):
    for region in regions:
        if i in region and j in region:
            return True
    return False


def create_fully_connected(nn: int):
    adj = np.zeros((nn, nn))
    for i in range(nn):
        for j in range(i + 1, nn):
            adj[i][j] = 1
    return adj


def create_fully_segregated(nn: int, regions: list):
    '''
    nn: number of nodes
    '''
    adj = np.zeros((nn, nn))
    for i in range(nn):
        for j in range(i + 1, nn):
            adj[i][j] = int(are_in_one_region(i, j, regions))
    return adj

def create_fully_integrated(nn: int, regions: list):
    '''
    nn: number of nodes
    '''
    adj = np.zeros((nn, nn))
    for i in range(nn):
        for j in range(i + 1, nn):
            adj[i][j] = int(not are_in_one_region(i, j, regions))
    return adj


def create_network(nr: int, nnpr: int, net_type: NetworkType):
    '''
    nr: number of regions
    nnpr: number of nodes per region
    type: type of network
    '''
    nn = nnpr * nr
    regions = [range(i*nnpr, (i+1)*nnpr) for i in range(0, nr)]
    # print(f'regions: {regions}')
    if net_type == NetworkType.Segregation:
        adj = create_fully_segregated(nn, regions)
    elif net_type == NetworkType.Integration:
        adj = create_fully_integrated(nn, regions)
    elif net_type == NetworkType.FullyConnected:
        adj = create_fully_connected(nn)
    adj = (adj + adj.T)
    return regions, adj


def calculate_coeffs(adj, kci):
    modularity = bct.modularity_und(adj, kci=kci)[1]
    participation = bct.participation_coef(adj, ci=kci)
    g_participation = np.mean(participation)
    return round(modularity, 4), round(g_participation, 4)


if __name__ == '__main__':
    all_num_regions = range(2, 30)
    all_num_nodes_per_region = range(2, 21)
    all_net_types = [NetworkType.Integration, NetworkType.Segregation]
    with open('coefficients.csv', 'w') as f:
        f.write('net_type,num_regions,num_nodes_per_region,bct_modularity,bct_participation\n')
        for net_type in all_net_types:
            for num_regions in all_num_regions:
                for num_nodes_per_region in all_num_nodes_per_region:
                    connected_regions, adj = create_network(num_regions, num_nodes_per_region, net_type)
                    kci = [idx for idx, region in enumerate(connected_regions) for _ in region]
                    modularity, participation = calculate_coeffs(adj, kci)
                    f.write(f'{net_type.value},{num_regions},{num_nodes_per_region},{modularity},{participation}\n')