import torch
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.data.data import Data
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
# from torch.utils.data import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset
from datasets import SynGraphDataset, Mutag

def get_data_loaders(data_dir, dataset_name, batch_size, splits, random_state, mutag_x=False):
    multi_label = False
    assert dataset_name in ['ba_2motifs', 'mutag',  'ogbg_molhiv', 'ogbg_molbbbp', 'ogbg_molsider']

    if dataset_name == 'ba_2motifs':
        dataset = SynGraphDataset(data_dir, 'ba_2motifs')
        split_idx = get_random_split_idx(dataset, splits)
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset=dataset, split_idx=split_idx)
        train_set = dataset[split_idx["train"]]

    elif dataset_name == 'mutag':
        dataset = Mutag(root=data_dir / 'mutag')
        split_idx = get_random_split_idx(dataset, splits, mutag_x=mutag_x)
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset=dataset, split_idx=split_idx)
        train_set = dataset[split_idx['train']]

    elif 'ogbg' in dataset_name:
        dataset = PygGraphPropPredDataset(root=data_dir, name='-'.join(dataset_name.split('_')), pre_transform=pre_transform)
        split_idx = get_random_split_idx(dataset, splits)
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset=dataset, split_idx=split_idx)
        train_set = dataset[split_idx['train']]

    x_dim = test_set[0].x.shape[1]
    edge_attr_dim = 0 if test_set[0].edge_attr is None else test_set[0].edge_attr.shape[1]
    if isinstance(test_set, list):
        num_class = Batch.from_data_list(test_set).y.unique().shape[0]
    elif test_set.data.y.shape[-1] == 1 or len(test_set.data.y.shape) == 1:
        num_class = test_set.data.y.unique().shape[0]
    else:
        num_class = test_set.data.y.shape[-1]
        multi_label = True

    print('[INFO] Calculating degree...')
    # Compute in-degree histogram over training data.
    # deg = torch.zeros(10, dtype=torch.long)
    # for data in train_set:
    #     d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    #     deg += torch.bincount(d, minlength=deg.numel())
    batched_train_set = Batch.from_data_list(train_set)
    d = degree(batched_train_set.edge_index[1], num_nodes=batched_train_set.num_nodes, dtype=torch.long)
    deg = torch.bincount(d, minlength=10)

    aux_info = {'deg': deg, 'multi_label': multi_label}
    return loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info


def get_random_split_idx(dataset, splits, random_state=None, mutag_x=False):
    if random_state is not None:
        np.random.seed(random_state)

    print('[INFO] Randomly split dataset!')
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)

    if not mutag_x:
        n_train, n_valid = int(splits['train'] * len(idx)), int(splits['valid'] * len(idx))
        train_idx = idx[:n_train]
        valid_idx = idx[n_train:n_train+n_valid]
        test_idx = idx[n_train+n_valid:]
    else:
        print('[INFO] mutag_x is True!')
        n_train = int(splits['train'] * len(idx))
        train_idx, valid_idx = idx[:n_train], idx[n_train:]
        test_idx = [i for i in range(len(dataset)) if (dataset[i].y.squeeze() == 0 and dataset[i].edge_label.sum() > 0)]
    return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}


def get_loaders_and_test_set(batch_size, dataset=None, split_idx=None, dataset_splits=None):
    if split_idx is not None:
        assert dataset is not None
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)
        test_set = dataset.copy(split_idx["test"])  # For visualization
    else:
        assert dataset_splits is not None
        train_loader = DataLoader(dataset_splits['train'], batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset_splits['valid'], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset_splits['test'], batch_size=batch_size, shuffle=False)
        test_set = dataset_splits['test']  # For visualization
    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}, test_set
from torch_geometric.utils import to_dense_adj
def pre_transform(data):
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    if (edge_index.shape[1]==0):
        data.resistance = torch.ones([edge_index.shape[1]])
        return data

    A = to_dense_adj(edge_index)[0]
    if (A.shape[0]!=num_nodes):
        data.resistance = torch.ones([edge_index.shape[1]])
        return data
    num_edges = edge_index.shape[1]
    resistance = torch.ones([edge_index.shape[1]])
  
    D = torch.diag(A.sum(-1))
    L = D - A
    M = torch.linalg.pinv(L + torch.ones([num_nodes,num_nodes],device=A.device) / num_nodes)

    for i in range(num_edges):
        left = edge_index[0][i]
        right = edge_index[1][i]
        resistance[i] = (M[left][left] + M[right][right] - 2*M[left][right])
    data.resistance = resistance

    return data