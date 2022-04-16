from tqdm import tqdm
from labels import *
import numpy as np
from pathlib import Path
import globalenv
from explainer import GaussianExplainer

class Dataset:
    def __init__(self, GaussianExplainer: GaussianExplainer, num_per_label: dict, num_nodes: int, num_edges: int, mask, dir: str):
        self.num_per_label = num_per_label
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.GaussianExplainer = GaussianExplainer
        Path(dir).mkdir(exist_ok=True)

        self.dir = dir
        self.mask = mask

    def generate(self):
        for d in globalenv.DS:
            dpath = f'd={d}'
            Path(self.dir, dpath).mkdir(exist_ok=True)
            d_low, d_high = self.GaussianExplainer.get_l_distribution(d), self.GaussianExplainer.get_h_distribution(d)
            print
            for LabelClass, (name, num) in tqdm(self.num_per_label.items(), desc=f'd={d}'):
                Path(self.dir, dpath, name).mkdir(exist_ok=True)
                for i in range(num):
                    graph = LabelClass(self.mask, f'{i+1}', d_low, d_high)
                    graph.to_json(Path(self.dir, dpath, name), save_adj=False)


def make_symmetric(matrix):
    return (matrix + matrix.T)

def generate_mask(num_nodes, num_edges, fully_conn=True):
    if fully_conn:
        mask = np.ones((num_nodes, num_nodes))
        np.fill_diagonal(mask, 0)
        return mask
    edges_ratio = num_edges / (num_nodes * (num_nodes - 1))
    mask = np.random.uniform(0, 1, num_nodes**2).reshape(num_nodes, num_nodes)
    symm_mask = make_symmetric(mask) / 2
    mask = np.where((symm_mask < edges_ratio), 1, 0)
    np.fill_diagonal(mask, 0)
    return mask


def generate_integration():
    train = {
        Label1: ('integration', 1),
        Label2: ('neutral', 1)
    }
    valid = {
        Label1: ('integration', 0),
        Label2: ('neutral', 0)
    }
    test = {
        Label1: ('integration', 0),
        Label2: ('neutral', 0)
    }
    return train, valid, test


def generate_segregation():
    train = {
        Label3: ('segregation', 1),
        Label2: ('neutral', 1)
    }
    valid = {
        Label3: ('segregation', 0),
        Label2: ('neutral', 0)
    }
    test = {
        Label3: ('segregation', 0),
        Label2: ('neutral', 0)
    }
    return train, valid, test


if __name__ == '__main__':
    gen_dataset = globalenv.GenDataset.segregation
    GaussianExplainer = GaussianExplainer()
    num_nodes = globalenv.NUM_NODES
    num_edges = num_nodes * (num_nodes - 1) # undirected
    mask = generate_mask(num_nodes, num_edges)

    if gen_dataset == globalenv.GenDataset.integration:
        train, valid, test = generate_integration()
    elif gen_dataset == globalenv.GenDataset.segregation:
        train, valid, test = generate_segregation()
    global_path = f'{gen_dataset.value}_dataset'
    Path(global_path).mkdir(exist_ok=True)
    for i in globalenv.CS:
        if gen_dataset == globalenv.GenDataset.integration:
            globalenv.CONNECTED_REGIONS = i
            cr = globalenv.CONNECTED_REGIONS
            to_save = f'{global_path}/cr={cr}'
        elif gen_dataset == globalenv.GenDataset.segregation:
            globalenv.INNERCONNECTED_REGIONS = i
            icr = globalenv.INNERCONNECTED_REGIONS
            to_save = f'{global_path}/icr={icr}'
        Path(to_save).mkdir(exist_ok=True)
        train_dataset = Dataset(GaussianExplainer, train, num_nodes, num_edges, mask, f'{to_save}/train')
        valid_dataset = Dataset(GaussianExplainer, valid, num_nodes, num_edges, mask, f'{to_save}/valid')
        test_dataset = Dataset(GaussianExplainer, test, num_nodes, num_edges, mask, f'{to_save}/test')
        print('>>> generating train...')
        train_dataset.generate()
        print('>>> generating valid...')
        valid_dataset.generate()
        print('>>> generating test...')
        test_dataset.generate()

        with open(f'{to_save}/readme.txt', 'w') as f:
            # f.write(f'number of nodes: {globalenv.NUM_NODES}\n')
            # f.write(f'number of regions: {globalenv.NUM_REGIONS}\n')
            # f.write(f'number of nodes per region: {globalenv.NUM_NODES_PER_REGION}\n')
            # if gen_dataset == globalenv.GenDataset.integration:
            #     f.write(f'inner-connected regions: {globalenv.INNERCONNECTED_REGIONS}\n')
            # elif gen_dataset == globalenv.GenDataset.segregation:
            #     f.write(f'inner-connected regions: {globalenv.INNERCONNECTED_REGIONS}\n')
            f.write(f'd means step number from {globalenv.DS[0]} to {globalenv.DS[-1]}')