# https://github.com/flyingdoog/PGExplainer/blob/master/MUTAG.ipynb

import yaml
import torch
import numpy as np
import pickle as pkl
from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data


class Mutag(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root=root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Mutagenicity_A.txt', 'Mutagenicity_edge_gt.txt', 'Mutagenicity_edge_labels.txt',
                'Mutagenicity_graph_indicator.txt', 'Mutagenicity_graph_labels.txt', 'Mutagenicity_label_readme.txt',
                'Mutagenicity_node_labels.txt', 'Mutagenicity.pkl']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        raise NotImplementedError

    def process(self):
        print("Precalculating...")

        with open(self.raw_dir + '/Mutagenicity.pkl', 'rb') as fin:
            _, original_features, original_labels = pkl.load(fin)
        # with open(self.raw_dir + '/RD.pkl', 'rb') as fin_1:
        #     RD = pkl.load(fin_1)

        edge_lists, graph_labels, edge_label_lists, node_type_lists = self.get_graph_data()
        # print(original_labels.shape[0])
        # print(len(RD))

        data_list = []
        # RD = []
        for i in range(original_labels.shape[0]):
            num_nodes = len(node_type_lists[i])
            edge_index = torch.tensor(edge_lists[i], dtype=torch.long).T

            y = torch.tensor(graph_labels[i]).float().reshape(-1, 1)
            x = torch.tensor(original_features[i][:num_nodes]).float()
            r = pre_calculate_RD(edge_index, num_nodes, i)
            assert original_features[i][num_nodes:].sum() == 0
            edge_label = torch.tensor(edge_label_lists[i]).float()
            if y.item() != 0:
                edge_label = torch.zeros_like(edge_label).float()

            node_label = torch.zeros(x.shape[0])
            signal_nodes = list(set(edge_index[:, edge_label.bool()].reshape(-1).tolist()))
            if y.item() == 0:
                node_label[signal_nodes] = 1

            if len(signal_nodes) != 0:
                node_type = torch.tensor(node_type_lists[i])
                node_type = set(node_type[signal_nodes].tolist())
                assert node_type in ({4, 1}, {4, 3}, {4, 1, 3})  # NO or NH

            if y.item() == 0 and len(signal_nodes) == 0:
                continue
            data_list.append(Data(x=x, y=y, edge_index=edge_index, node_label=node_label, edge_label=edge_label, resistance=r, node_type=torch.tensor(node_type_lists[i])))
            # RD.append()

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        # with open(self.raw_dir + '/RD.pkl', 'wb') as f:
        #     pkl.dump(RD, f)
        #     print('保存结束！')


    def get_graph_data(self):
        pri = self.raw_dir + '/Mutagenicity_'

        file_edges = pri + 'A.txt'
        # file_edge_labels = pri + 'edge_labels.txt'
        file_edge_labels = pri + 'edge_gt.txt'
        file_graph_indicator = pri + 'graph_indicator.txt'
        file_graph_labels = pri + 'graph_labels.txt'
        file_node_labels = pri + 'node_labels.txt'

        edges = np.loadtxt(file_edges, delimiter=',').astype(np.int32)
        try:
            edge_labels = np.loadtxt(file_edge_labels, delimiter=',').astype(np.int32)
        except Exception as e:
            print(e)
            print('use edge label 0')
            edge_labels = np.zeros(edges.shape[0]).astype(np.int32)

        graph_indicator = np.loadtxt(file_graph_indicator, delimiter=',').astype(np.int32)
        graph_labels = np.loadtxt(file_graph_labels, delimiter=',').astype(np.int32)

        try:
            node_labels = np.loadtxt(file_node_labels, delimiter=',').astype(np.int32)
        except Exception as e:
            print(e)
            print('use node label 0')
            node_labels = np.zeros(graph_indicator.shape[0]).astype(np.int32)

        graph_id = 1
        starts = [1]
        node2graph = {}
        for i in range(len(graph_indicator)):
            if graph_indicator[i] != graph_id:
                graph_id = graph_indicator[i]
                starts.append(i+1)
            node2graph[i+1] = len(starts)-1
        # print(starts)
        # print(node2graph)
        graphid = 0
        edge_lists = []
        edge_label_lists = []
        edge_list = []
        edge_label_list = []
        for (s, t), l in list(zip(edges, edge_labels)):
            sgid = node2graph[s]
            tgid = node2graph[t]
            if sgid != tgid:
                print('edges connecting different graphs, error here, please check.')
                print(s, t, 'graph id', sgid, tgid)
                exit(1)
            gid = sgid
            if gid != graphid:
                edge_lists.append(edge_list)
                edge_label_lists.append(edge_label_list)
                edge_list = []
                edge_label_list = []
                graphid = gid
            start = starts[gid]
            edge_list.append((s-start, t-start))
            edge_label_list.append(l)

        edge_lists.append(edge_list)
        edge_label_lists.append(edge_label_list)

        # node labels
        node_label_lists = []
        graphid = 0
        node_label_list = []
        for i in range(len(node_labels)):
            nid = i+1
            gid = node2graph[nid]
            # start = starts[gid]
            if gid != graphid:
                node_label_lists.append(node_label_list)
                graphid = gid
                node_label_list = []
            node_label_list.append(node_labels[i])
        node_label_lists.append(node_label_list)

        return edge_lists, graph_labels, edge_label_lists, node_label_lists

from torch_geometric.utils import to_dense_adj

def pre_calculate_RD(edge_index, num_nodes, i):
    A = to_dense_adj(edge_index)[0]
    # print("AAAAAA", A)
    if(A.shape[0]!=num_nodes):
        return torch.ones_like(edge_index[1])
    # print("A:", A.shape)
    # print("edge_index:", edge_index.shape)
    # print("num_nodes:", num_nodes)
    num_edges = edge_index.shape[1]
    RD = torch.ones(edge_index.shape[1])
  
    D = torch.diag(A.sum(-1))
    L = D - A
    # print(L)
    M = torch.linalg.pinv(L + torch.ones([num_nodes,num_nodes],device=A.device) / num_nodes)

    for i in range(num_edges):
        left = edge_index[0][i]
        right = edge_index[1][i]
        # print("left:", M[left][left])
        # print("right:", M[right][right])
        # print("M[left][right]:", M[left][right])
        # print("M[left][left] + M[right][right] - 2*M[left][right]",M[left][left] + M[right][right] - 2*M[left][right])

        RD[i] = M[left][left] + M[right][right] - 2*M[left][right]
        # print("RD[i]:", RD[i])
    return RD