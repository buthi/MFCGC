import torch
import torch.nn
import torch.nn.functional as F
import dgl
from torch import nn
from dgl.nn.pytorch import GraphConv


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, normalize=False):
        super(GConv, self).__init__()
        self.act = nn.PReLU()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.append(
            GraphConv(in_feats=input_dim, out_feats=hidden_dim, allow_zero_in_degree=True, norm='left')
        )
        for _ in range(1, num_layers):
            self.layers.append(
                GraphConv(in_feats=hidden_dim, out_feats=hidden_dim, allow_zero_in_degree=True, norm='left')
            )
        self.normalize = normalize

    def forward(self, x, graph):
        z = x

        for i, layer in enumerate(self.layers):
            z = layer(graph, z)
            z = self.act(z)
        if self.normalize:
            return F.normalize(z, dim=1)
        else:
            return z


class MultiFeatureEncoder(nn.Module):
    def __init__(self, device, x_list, augmentor, hidden_dim=256, n_clusters=3, num_layers=3, normalize=False):
        super(MultiFeatureEncoder, self).__init__()

        gconvs = nn.ModuleList()
        self.cluster_projectors = nn.ModuleList()
        self.instance_projectors = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for x in x_list:
            gconv = GConv(input_dim=x.size(1), hidden_dim=hidden_dim, num_layers=num_layers, normalize=normalize).to(
                device)
            gconvs.append(gconv)

            self.instance_projectors.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.PReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
            self.cluster_projectors.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.PReLU(),
                nn.Linear(hidden_dim, n_clusters),
                nn.Softmax(dim=1)
            ))
        self.encoder = gconvs
        self.augmentor = augmentor

        self.device = device

    def forward(self, x_list, graph, edge_index):
        aug1, aug2, aug3 = self.augmentor
        zs = []
        z1s = []
        z2s = []
        c1s = []
        c2s = []
        for index in range(len(x_list)):
            x = x_list[index]
            z = self.encoder[index](x, graph)

            x1, edge_index1, _ = aug1(x, edge_index, embeds=z, index=index)
            graph1 = dgl.graph((edge_index1[0], edge_index1[1]), num_nodes=graph.num_nodes()).to(self.device)
            graph1.remove_self_loop()
            graph1.add_self_loop()
            z1 = self.encoder[index](x1, graph1)

            x2, edge_index2, _ = aug1(x, edge_index, embeds=z, index=index)
            graph2 = dgl.graph((edge_index2[0], edge_index2[1]), num_nodes=graph.num_nodes()).to(self.device)
            graph2.remove_self_loop()
            graph2.add_self_loop()
            z2 = self.encoder[index](x2, graph2)

            x3, edge_index3, _ = aug1(x, edge_index, embeds=z, index=index)
            graph3 = dgl.graph((edge_index3[0], edge_index3[1]), num_nodes=graph.num_nodes()).to(self.device)
            graph3.remove_self_loop()
            graph3.add_self_loop()
            z3 = self.encoder[index](x3, graph3)

            c1 = F.normalize(self.cluster_projectors[index](z1), dim=1)
            c2 = F.normalize(self.cluster_projectors[index](z2), dim=1)
            z1 = self.instance_projectors[index](z1)
            z2 = self.instance_projectors[index](z2)
            z3 = self.instance_projectors[index](z3)

            zs.append(z3)
            z1s.append(z1)
            z2s.append(z2)
            c1s.append(c1)
            c2s.append(c2)
        return zs, z1s, z2s, c1s, c2s

    def cluster(self, x_list, graph):
        zs = []
        cs = []
        for index in range(len(x_list)):
            z = self.encoder[index](x_list[index], graph)
            c = self.cluster_projectors[index](z)
            z = self.instance_projectors[index](z)
            zs.append(z)
            cs.append(c)

        return zs, cs


class MultiGraphEncoder(nn.Module):
    def __init__(self, device, graph_list, augmentor, input_dim, hidden_dim=256, n_clusters=3, num_layers=3,
                 normalize=True):
        super(MultiGraphEncoder, self).__init__()

        gconvs = nn.ModuleList()
        self.cluster_projectors = nn.ModuleList()
        self.instance_projectors = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(len(graph_list)):
            gconv = GConv(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, normalize=normalize)
            gconvs.append(gconv)
            self.cluster_projectors.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, n_clusters),
                nn.Softmax(dim=1)
            ))
            self.instance_projectors.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
            self.decoders.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, input_dim),
            ))

        self.encoder = gconvs
        self.augmentor = augmentor

        self.device = device

    def forward(self, x, graph_list):
        aug1, aug2, aug3 = self.augmentor
        zs = []
        z1s = []
        z2s = []
        c1s = []
        c2s = []
        for i in range(len(graph_list)):
            graph, edge_index = graph_list[i]
            z = self.encoder[i](x, graph)

            x1, edge_index1, _ = aug1(x, edge_index, embeds=z, index=i)
            graph1 = dgl.graph((edge_index1[0], edge_index1[1]), num_nodes=graph.num_nodes()).to(self.device)
            graph1.remove_self_loop()
            graph1.add_self_loop()
            z1 = self.encoder[i](x1, graph1)

            x2, edge_index2, _ = aug2(x, edge_index, embeds=z, index=i)
            graph2 = dgl.graph((edge_index2[0], edge_index2[1]), num_nodes=graph.num_nodes()).to(self.device)
            graph2.remove_self_loop()
            graph2.add_self_loop()
            z2 = self.encoder[i](x2, graph2)

            x3, edge_index3, _ = aug3(x, edge_index, embeds=z, index=i)
            graph3 = dgl.graph((edge_index3[0], edge_index3[1]), num_nodes=graph.num_nodes()).to(self.device)
            graph3.remove_self_loop()
            graph3.add_self_loop()
            z3 = self.encoder[i](x3, graph3)

            c1 = F.normalize(self.cluster_projectors[i](z1), dim=1)
            c2 = F.normalize(self.cluster_projectors[i](z2), dim=1)

            z1 = self.instance_projectors[i](z1)
            z2 = self.instance_projectors[i](z2)

            zs.append(z3)
            z1s.append(z1)
            z2s.append(z2)
            c1s.append(c1)
            c2s.append(c2)

        return zs, z1s, z2s, c1s, c2s

    def cluster(self, x, graph_list):
        zs = []
        cs = []
        for i in range(len(graph_list)):
            graph, edge_index = graph_list[i]
            z = self.encoder[i](x, graph)
            c = self.cluster_projectors[i](z)
            zs.append(z)
            cs.append(c)

        return zs, cs
