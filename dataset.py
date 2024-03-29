from tqdm import tqdm
from labels import *
import numpy as np
from pathlib import Path
import globalenv
from explainer import GaussianExplainer, UniformExplainer

class Dataset:
    def __init__(self, explainer, num_per_label: dict, num_nodes: int, num_edges: int, mask, dir: str):
        self.num_per_label = num_per_label
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.explainer = explainer
        Path(dir).mkdir(exist_ok=True)

        self.dir = dir
        self.mask = mask

    def generate(self):
        for n in globalenv.NS:
            dpath = f'n={n}'
            Path(self.dir, dpath).mkdir(exist_ok=True)
            d_low, d_high = self.explainer.get_l_distribution(n), self.explainer.get_h_distribution(n)
            for LabelClass, (name, num) in tqdm(self.num_per_label.items(), desc=f'n={n}'):
                Path(self.dir, dpath, name).mkdir(exist_ok=True)
                for i in range(num):
                    graph = LabelClass(self.mask, f'{i+1}', d_low, d_high, globalenv.DIST_TYPE)
                    graph.to_json(Path(self.dir, dpath, name), save_adj=True) # save_adj TRUE ONLY FOR VISUALIZATION


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
        Label1: ('integration', 256),
    }
    valid = {
        Label1: ('integration', 32),
    }
    test = {
        Label1: ('integration', 32),
    }
    return train, valid, test


def generate_segregation():
    train = {
        Label3: ('segregation', 256),
    }
    valid = {
        Label3: ('segregation', 32),
    }
    test = {
        Label3: ('segregation', 32),
    }
    return train, valid, test


def generate_neutral():
    train = {
        Label2: ('neutral', 256),
    }
    valid = {
        Label2: ('neutral', 32),
    }
    test = {
        Label2: ('neutral', 32),
    }
    return train, valid, test


if __name__ == '__main__':
    gen_dataset = globalenv.DATASET_TYPE
    dist_type = globalenv.DIST_TYPE.value
    explainer = UniformExplainer() if globalenv.DIST_TYPE is globalenv.DistributionType.Uniform else GaussianExplainer()
    num_nodes = globalenv.NUM_NODES
    num_edges = num_nodes * (num_nodes - 1) # undirected
    mask = generate_mask(num_nodes, num_edges)
    if gen_dataset == globalenv.GenDataset.integration:
        train, valid, test = generate_integration()
    elif gen_dataset == globalenv.GenDataset.segregation:
        train, valid, test = generate_segregation()
    elif gen_dataset == globalenv.GenDataset.neutral:
        train, valid, test = generate_neutral()
    global_path = f'__{gen_dataset.value}_{dist_type}_dataset'
    Path(global_path).mkdir(exist_ok=True)
    if gen_dataset != globalenv.GenDataset.neutral:
        for i in globalenv.REGIONS_RANGE:
            if gen_dataset == globalenv.GenDataset.integration:
                globalenv.CONNECTED_REGIONS = i
                cr = globalenv.CONNECTED_REGIONS
                to_save = f'{global_path}/cr={cr}'
            elif gen_dataset == globalenv.GenDataset.segregation:
                globalenv.INNERCONNECTED_REGIONS = i
                icr = globalenv.INNERCONNECTED_REGIONS
                to_save = f'{global_path}/icr={icr}'
            Path(to_save).mkdir(exist_ok=True)
            train_dataset = Dataset(explainer, train, num_nodes, num_edges, mask, f'{to_save}/train')
            valid_dataset = Dataset(explainer, valid, num_nodes, num_edges, mask, f'{to_save}/valid')
            test_dataset = Dataset(explainer, test, num_nodes, num_edges, mask, f'{to_save}/test')
            print('>>> generating train...')
            train_dataset.generate()
            print('>>> generating valid...')
            valid_dataset.generate()
            print('>>> generating test...')
            test_dataset.generate()

            with open(f'{to_save}/readme.txt', 'w') as f:
                f.write(f'd means step number from {globalenv.NS[0]} to {globalenv.NS[-1]}')
    else:
        to_save = f'{global_path}'
        Path(to_save).mkdir(exist_ok=True)
        train_dataset = Dataset(explainer, train, num_nodes, num_edges, mask, f'{to_save}/train')
        valid_dataset = Dataset(explainer, valid, num_nodes, num_edges, mask, f'{to_save}/valid')
        test_dataset = Dataset(explainer, test, num_nodes, num_edges, mask, f'{to_save}/test')
        print('>>> generating train...')
        train_dataset.generate()
        print('>>> generating valid...')
        valid_dataset.generate()
        print('>>> generating test...')
        test_dataset.generate()

        with open(f'{to_save}/readme.txt', 'w') as f:
            f.write(f'd means step number from {globalenv.NS[0]} to {globalenv.NS[-1]}')