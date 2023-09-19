import torch.nn
import torch
import torch.nn
import torch.nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


def multi_feature_train1(encoder_model, contrast_model, cluster_model, optimizer, x_list, graph, edge_index, device):
    encoder_model.train()
    optimizer.zero_grad()
    zs, z1s, z2s, c1s, c2s = encoder_model(x_list, graph, edge_index)
    con_loss = 0
    for i in range(len(z1s)):
        con_loss += F.mse_loss(torch.mm(z1s[i], z2s[i].T), torch.eye(z1s[i].size(0)).to(device))
    for i in range(len(z1s)):
        con_loss += contrast_model(z1s[i], z2s[i])
    clu_loss = 0
    for i in range(len(z1s)):
        clu_loss += cluster_model(c1s[i].t(), c2s[i].t())

    re_loss = 0
    loss = con_loss + clu_loss

    loss.backward()
    optimizer.step()
    return loss.item(), con_loss.item(), clu_loss.item(), re_loss


def multi_feature_train2(encoder_model, contrast_model, contrast_model2, cluster_model, optimizer, x_list, graph,
           edge_index, device, scheduler=None):
    encoder_model.train()
    optimizer.zero_grad()
    zs, z1s, z2s, c1s, c2s = encoder_model(x_list, graph, edge_index)
    con_loss = 0
    for i in range(len(z1s)):
        con_loss += contrast_model(z1s[i], z2s[i])
    for i in range(len(z1s)):
        con_loss += F.mse_loss(torch.mm(z1s[i], z2s[i].t()), torch.eye(z1s[i].size(0)).to(device))

    mv_con_loss = 0
    for i in range(1, len(z1s)):
        mv_con_loss += contrast_model2(zs[0], zs[i])

    clu_loss = 0
    for i in range(len(z1s)):
        clu_loss += cluster_model(c1s[i].t(), c2s[i].t())

    loss = con_loss + clu_loss + mv_con_loss

    loss.backward()
    optimizer.step()
    if scheduler != None:
        scheduler.step()
    return loss.item(), con_loss.item(), clu_loss.item(), mv_con_loss.item()

def multi_feature_train3(encoder_model, contrast_model, contrast_model2, cluster_model, sup_contrast, optimizer, x_list, graph, edge_index, pred, high_confidence_idx, high_confidence_label, device, scheduler=None):

    encoder_model.train()
    optimizer.zero_grad()
    zs, z1s, z2s, c1s, c2s = encoder_model(x_list, graph, edge_index)
    con_loss = 0

    z1s_ = []
    z2s_ = []
    for i in range(len(z1s)):
        z1s_.append(z1s[i][torch.tensor(list(set(range(z1s[i].size(0))) - set(high_confidence_idx.tolist())))])
        z2s_.append(z2s[i][torch.tensor(list(set(range(z2s[i].size(0))) - set(high_confidence_idx.tolist())))])

    for i in range(len(z1s)):
        con_loss += contrast_model(z1s_[i], z2s_[i])

    for i in range(len(z1s)):
        con_loss += F.mse_loss(torch.mm(z1s[i], z2s[i].t()), torch.eye(z1s[i].size(0)).to(device))

    mv_con_loss = 0
    for i in range(1, len(z1s)):
        mv_con_loss += contrast_model2(z1s_[0], z1s_[i])
        mv_con_loss += contrast_model2(z2s_[0], z2s_[i])

    clu_loss = 0
    for i in range(len(z1s)):
        clu_loss += cluster_model(c1s[i], c2s[i])


    sup_loss = 0
    for i in range(len(z1s)):

        embeddings = torch.cat([z1s[i].unsqueeze(1), z2s[i].unsqueeze(1)], dim=1)
        embeddings = embeddings[high_confidence_idx]
        labels = torch.tensor(high_confidence_label)
        sup_loss += sup_contrast(embeddings, labels)

    loss = con_loss + clu_loss + mv_con_loss + sup_loss
    loss.backward()
    optimizer.step()
    if scheduler != None:
        scheduler.step()

    c_mat = []
    pred_c = []
    for i in range(len(z1s)):
        c_mat.append((c1s[i] + c2s[i]) / 2)
        i = c_mat[i].argmax(dim=-1)
        pred_c.append(i.data.cpu().numpy())

    return loss.item()

def multi_graph_train1(encoder_model, contrast_model, contrast_model2, cluster_model, optimizer, x, graph_list, device):
    encoder_model.train()
    optimizer.zero_grad()
    _, z1s, z2s, c1s, c2s = encoder_model(x, graph_list)
    con_loss = 0
    for i in range(len(z1s)):
        con_loss += contrast_model(z1s[i], z2s[i])
    for i in range(len(z1s)):
        con_loss += F.mse_loss(torch.mm(z1s[i], z2s[i].t()), torch.eye(z1s[i].size(0)).to(device))

    clu_loss = 0
    for i in range(len(z1s)):
        clu_loss += cluster_model(c1s[i].t(), c2s[i].t())

    re_loss = 0
    loss = con_loss
    loss.backward()
    optimizer.step()
    return loss.item(), con_loss.item(), clu_loss.item(), re_loss


def multi_graph_train2(encoder_model, contrast_model, contrast_model2, cluster_model, optimizer, x, graph_list, device, scheduler=None):
    encoder_model.train()
    optimizer.zero_grad()
    zs, z1s, z2s, c1s, c2s = encoder_model(x, graph_list)
    con_loss = 0
    for i in range(len(z1s)):
        con_loss += contrast_model(z1s[i], z2s[i])
    for i in range(len(z1s)):
        con_loss += F.mse_loss(torch.mm(z1s[i], z2s[i].t()), torch.eye(z1s[i].size(0)).to(device))

    mv_con_loss = 0
    for i in range(1, len(z1s)):
        mv_con_loss += contrast_model2(zs[0], zs[i])

    clu_loss = 0
    for i in range(len(z1s)):
        clu_loss += cluster_model(c1s[i].t(), c2s[i].t())

    re_loss = 0
    loss = con_loss + clu_loss + mv_con_loss
    loss.backward()
    optimizer.step()
    if scheduler != None:
        scheduler.step()
    return loss.item(), con_loss.item(), clu_loss.item(), mv_con_loss.item(), re_loss


def multi_graph_train3(encoder_model, contrast_model, contrast_model2, cluster_model, sup_contrast, optimizer, x, graph_list, pred, high_confidence_idx, high_confidence_label, device, scheduler=None):

    encoder_model.train()
    optimizer.zero_grad()
    zs, z1s, z2s, c1s, c2s = encoder_model(x, graph_list)
    con_loss = 0

    z1s_ = []
    z2s_ = []
    for i in range(len(z1s)):
        z1s_.append(z1s[i][torch.tensor(list(set(range(z1s[i].size(0))) - set(high_confidence_idx.tolist())))])
        z2s_.append(z2s[i][torch.tensor(list(set(range(z2s[i].size(0))) - set(high_confidence_idx.tolist())))])

    for i in range(len(z1s)):
        con_loss += contrast_model(z1s_[i], z2s_[i])

    for i in range(len(z1s)):
        con_loss += F.mse_loss(torch.mm(z1s[i], z2s[i].t()), torch.eye(z1s[i].size(0)).to(device))

    mv_con_loss = 0
    for i in range(1, len(z1s)):
        mv_con_loss += contrast_model2(z1s_[0], z1s_[i])
        mv_con_loss += contrast_model2(z2s_[0], z2s_[i])

    clu_loss = 0
    for i in range(len(z1s)):
        clu_loss += cluster_model(c1s[i], c2s[i])


    sup_loss = 0
    for i in range(len(z1s)):

        embeddings = torch.cat([z1s[i].unsqueeze(1), z2s[i].unsqueeze(1)], dim=1)
        embeddings = embeddings[high_confidence_idx]
        labels = torch.tensor(high_confidence_label)
        sup_loss += sup_contrast(embeddings, labels)

    loss = clu_loss + 10 * mv_con_loss + 0.2 * con_loss + 0.8 * sup_loss
    loss.backward()
    optimizer.step()
    if scheduler != None:
        scheduler.step()

    c_mat = []

    pred_c = []
    for i in range(len(z1s)):
        c_mat.append((c1s[i] + c2s[i]) / 2)

        c_mat_np = np.array(c_mat[i].cpu().detach())

        kmeans = KMeans(n_clusters=4).fit(c_mat_np)
        pred_c.append(kmeans.labels_)

    return loss.item(), pred_c

