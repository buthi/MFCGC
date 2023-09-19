import os.path as osp

import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn
import torch.nn
import torch.nn
import torch.nn.functional as F
from scipy.io import loadmat
from torch import tensor

from config import ACM_config, DBLP_config, IMDB_config
from config import config, Cora_config, Citeseer_config


def load_graph(graph):
    n, _ = graph.shape
    edges = np.array(graph, dtype=np.int32)
    idx = edges.nonzero()  # (row, col)
    data = edges[idx]
    adj = sp.coo_matrix((data, idx), shape=(n, n), dtype=np.float32)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_multi_graph_data(dataset, device):
    path = osp.join('../', 'datasets', dataset)

    if dataset == 'DBLP':
        mat = loadmat(osp.join(path, 'DBLP4057_GAT_with_idx.mat'))
        list = ['net_APCPA', 'net_APTPA', 'net_APA']
        n_clusters = 4
        args = DBLP_config()
    elif dataset == 'ACM':
        mat = loadmat(osp.join(path, 'ACM3025.mat'))
        list = ['PAP', 'PLP']
        n_clusters = 3
        args = ACM_config()
    else:
        mat = loadmat(osp.join(path, 'imdb5k.mat'))
        list = ['MDM', 'MAM']
        n_clusters = 3
        args = IMDB_config()

    graph_list = []
    for etype in list:
        adj = load_graph(mat[etype]).to(device)
        graph = dgl.graph((adj._indices()[0], adj._indices()[1]), num_nodes=len(mat[etype])).to(device)
        graph.remove_self_loop()
        graph.add_self_loop()
        edge_index = torch.stack(graph.edges())
        graph_list.append((graph, edge_index))

    x = tensor(mat['features'] if dataset == 'DBLP' else mat['feature'], dtype=torch.float).to(device)
    y = torch.argmax(tensor(mat['label']), -1).to(device)

    return x, y, graph_list, n_clusters, args

def load_mutil_feature_data(dataset, device, normalize=True):
    parser = config()

    if dataset == 'Cora':
        dataset = dgl.data.CoraGraphDataset()
        graph = dataset[0].to(device)
        graph.remove_self_loop()
        graph.add_self_loop()
        edge_index = torch.stack(graph.edges())
        n_clusters = 7
        args = Cora_config(parser)
    else:
        dataset = dgl.data.CiteseerGraphDataset()
        graph = dataset[0].to(device)
        graph.remove_self_loop()
        graph.add_self_loop()
        edge_index = torch.stack(graph.edges())
        n_clusters = 6
        args = Citeseer_config(parser)

    x = graph.ndata['feat']
    x2 = torch.mm(x, x.t())
    if normalize:
        x = F.normalize(x, dim=1)
        x2 = F.normalize(x2, dim=1)
    x_list = [x, x2]

    graph_list = []
    for i in range(2):
        graph_list.append((graph, edge_index))

    return x_list, graph.ndata['label'], graph, edge_index, n_clusters, args, graph_list
