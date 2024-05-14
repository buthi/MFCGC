import dgl
import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import random

from torch import nn
from munkres import Munkres
from sklearn import metrics
from torch_scatter import scatter_add


class InfoNCELoss(nn.Module):
    def __init__(self, tau, intra=True):
        super(InfoNCELoss, self).__init__()
        self.tau = tau
        self.intra = intra

    def sim(self, z1, z2):
        # normalize embeddings across feature dimension
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)

        s = torch.mm(z1, z2.t())
        return s

    def get_loss(self, z1, z2, ):
        # calculate SimCLR loss
        f = lambda x: torch.exp(x / self.tau)

        refl_sim = f(self.sim(z1, z1))  # intra-view pairs
        between_sim = f(self.sim(z1, z2))  # inter-view pairs

        # between_sim.diag(): positive pairs
        if self.intra:
            x1 = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
        else:
            x1 = between_sim.sum(1)
        # x1 = refl_sim.sum(1) + between_sim.sum(1)

        loss = -torch.log(between_sim.diag() / x1)

        return loss.mean()

    def forward(self, z1, z2):
        l1 = self.get_loss(z1, z2)
        l2 = self.get_loss(z2, z1)

        ret = (l1 + l2) * 0.5

        # return ret.mean()
        return ret


class Self_SupConLoss(nn.Module):
    def __init__(self, tau=0.07, contrast_mode='all', base_temperature=0.07):
        super(Self_SupConLoss, self).__init__()
        self.temperature = tau
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)  # embedding做乘积  zi * zj
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # 就是取每一列最大的值
        logits = anchor_dot_contrast - logits_max.detach()  # 之前的矩阵减去每一列的最大值
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)  # 行列都扩充两倍
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )  # 就是把对角线对的元素去掉了

        mask = mask * logits_mask  # 和原来的mask相乘 把对角线去掉
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # print(exp_logits.shape)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


class Augment:
    def __init__(self, *augments):
        self.augments = augments

    def forward(self, x, edge, edge_weights, embeds, index):
        for aug in self.augments:
            x, edge, edge_weights = aug.augment(x, edge, edge_weights, embeds, index)
        return x, edge, edge_weights

    def __call__(self, x, edge, edge_weights=None, embeds=None, index=None):
        return self.forward(x, edge, edge_weights, embeds, index)


class EdgeRemoving:
    def __init__(self, pe: float):
        self.pe = pe

    def augment(self, x, edge_index, edge_weights, embeds, index):
        if self.pe != 0:
            row, col = edge_index
            mask = torch.rand(row.size(0), device=edge_index.device) >= self.pe
            row, col, edge_attr = row[mask], col[mask], None if edge_weights is None else edge_weights[mask]
            edge_index = torch.stack([row, col], dim=0)
        return x, edge_index, edge_weights


class FeatureDropout:
    def __init__(self, pf: float):
        self.pf = pf

    def augment(self, x, edge_index, edge_weights, embeds, index):
        if self.pf != 0:
            x = F.dropout(x, p=1. - self.pf)
        return x, edge_index, edge_weights


class FeatureMasking:
    def __init__(self, pf: float):
        self.pf = pf

    def augment(self, x, edge_index, edge_weights, embeds, index):
        if self.pf != 0:
            device = x.device
            drop_mask = torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1) < self.pf
            drop_mask = drop_mask.to(device)
            x = x.clone()
            x[:, drop_mask] = 0
        return x, edge_index, edge_weights


class DegreeAugment:
    def __init__(self, graph_list, thresholds, edge_mask_rate):
        self.degrees = []
        self.num_nodes = []
        self.thresholds = thresholds
        self.edge_mask_rate = edge_mask_rate

        self.max_degrees = []
        self.node_dists = []
        self.src_idxs = []
        self.rest_idxs = []
        self.rest_node_degrees = []
        self.max_degrees = []
        for i in range(len(graph_list)):
            graph = graph_list[i]
            threshold = self.thresholds[i]
            edge_index = graph[1]
            num_node = graph[0].num_nodes()
            degree = graph[0].in_degrees().cpu().numpy()
            max_degree = np.max(degree)

            node_dist = self.get_node_dist(edge_index, num_node)
            src_idx = torch.LongTensor(np.argwhere(degree < threshold).flatten()).to(
                graph[0].device)
            rest_idx = torch.LongTensor(np.argwhere(degree >= threshold).flatten()).to(graph[0].device)
            rest_node_degree = degree[degree >= threshold]

            self.node_dists.append(node_dist)
            self.src_idxs.append(src_idx)
            self.rest_idxs.append(rest_idx)
            self.rest_node_degrees.append(rest_node_degree)
            self.max_degrees.append(max_degree)

    def get_node_dist(self, edge_index, num_node):
        """
        Compute adjacent node distribution.
        """
        row, col = edge_index[0], edge_index[1]

        dist_list = []
        for i in range(num_node):
            dist = torch.zeros([num_node], dtype=torch.float32, device=edge_index[0].device)
            idx = row[(col == i)]
            dist[idx] = 1
            dist_list.append(dist)
        dist_list = torch.stack(dist_list, dim=0)
        return dist_list

    def sim(self, z1, z2):
        # normalize embeddings across feature dimension
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)

        s = torch.mm(z1, z2.t())
        return s

    def neighbor_sampling(self, src_idx, dst_idx, node_dist, sim,
                          max_degree, aug_degree):
        phi = sim[src_idx, dst_idx].unsqueeze(dim=1)  # 选择对应的相似度
        phi = torch.clamp(phi, 0, 0.9)  # 将张量的值设置在0或者0.5，大于0.5的都是0.5，小于0.5的都是0

        # print('phi', phi)
        mix_dist = node_dist[dst_idx] * phi + node_dist[src_idx] * (1 - phi)  # 得到最终的相似度

        new_tgt = torch.multinomial(mix_dist + 1e-12, int(max_degree)).to(phi.device)
        tgt_idx = torch.arange(max_degree).unsqueeze(dim=0).to(phi.device)

        new_col = new_tgt[(tgt_idx - aug_degree.unsqueeze(dim=1) < 0)]  # 这里是一个传播机制
        new_row = src_idx.repeat_interleave(aug_degree)  # 这个就是将之前度比较小的节点扩增为
        return new_row, new_col

    def degree_mask_edge(self, idx, sim, max_degree, node_degree, mask_prob):
        aug_degree = (node_degree * (1 - mask_prob)).long().to(sim.device)
        sim_dist = sim[idx]

        # _, new_tgt = th.topk(sim_dist + 1e-12, int(max_degree))
        new_tgt = torch.multinomial(sim_dist + 1e-12, int(max_degree))
        tgt_idx = torch.arange(max_degree).unsqueeze(dim=0).to(sim.device)

        new_col = new_tgt[(tgt_idx - aug_degree.unsqueeze(dim=1) < 0)]
        new_row = idx.repeat_interleave(aug_degree)
        return new_row, new_col

    def augment(self, x, edge_index, edge_weights, embeds, index):
        node_dist = self.node_dists[index]
        src_idx = self.src_idxs[index]
        rest_idx = self.rest_idxs[index]
        rest_node_degree = self.rest_node_degrees[index]
        max_degree = self.max_degrees[index]

        sim = self.sim(embeds, embeds)
        sim = torch.clamp(sim, 0, 1)
        sim = sim - torch.diag_embed(torch.diag(sim))
        src_sim = sim[src_idx]
        dst_idx = torch.multinomial(src_sim + 1e-12, 1).flatten().to(
            x.device)

        rest_node_degree = torch.LongTensor(rest_node_degree)
        degree_dist = scatter_add(torch.ones(rest_node_degree.size()), rest_node_degree).to(
            x.device)
        prob = degree_dist.unsqueeze(dim=0).repeat(src_idx.size(0), 1)
        aug_degree = torch.multinomial(prob, 1).flatten().to(x.device)

        new_row_mix_1, new_col_mix_1 = self.neighbor_sampling(src_idx, dst_idx, node_dist, sim,
                                                              max_degree, aug_degree)
        new_row_rest_1, new_col_rest_1 = self.degree_mask_edge(rest_idx, sim, max_degree, rest_node_degree,
                                                               self.edge_mask_rate)
        nsrc1 = torch.cat((new_row_mix_1, new_row_rest_1)).cpu()
        ndst1 = torch.cat((new_col_mix_1, new_col_rest_1)).cpu()
        edge_index = torch.stack([nsrc1, ndst1], dim=0)
        return x, edge_index, edge_weights


def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')

    c = np.equal(new_predict, y_true).astype(int)

    return acc, f1_macro, c


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
