import os

from data import load_multi_graph_data, load_mutil_feature_data
from model import MultiFeatureEncoder, MultiGraphEncoder
from train import multi_feature_train1, multi_feature_train2, multi_feature_train3, multi_graph_train1, \
    multi_graph_train2, multi_graph_train3
from utils import Augment, EdgeRemoving, FeatureMasking, FeatureDropout, InfoNCELoss, cluster_acc, Self_SupConLoss, \
    DegreeAugment, setup_seed

os.environ["OMP_NUM_THREADS"] = '12'

import warnings

warnings.filterwarnings("ignore")

import torch.nn
import numpy as np
from tqdm import tqdm
from torch.optim import Adam

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def multi_feature_eva(x_list, graph, edge_index, model, kmeans):
    with torch.no_grad():
        model.eval()
        zs, cs = model.cluster(x_list, graph)
        z = torch.cat(tuple([z for z in zs]), dim=1)
    pred = kmeans.fit_predict(z.cpu())
    nmi = normalized_mutual_info_score(pred, y.cpu())
    ari = adjusted_rand_score(pred, y.cpu())
    acc, f1, c = cluster_acc(y.cpu().numpy(), pred)
    return acc, f1, nmi, ari, pred, c, kmeans.transform(z.cpu())

def multi_graph_eva(model, kmeans):
    with torch.no_grad():
        model.eval()
        zs, cs = model.cluster(x, graph_list)
        z = torch.cat(tuple([z for z in zs]), dim=1)
    pred = kmeans.fit_predict(z.cpu())
    nmi = normalized_mutual_info_score(pred, y.cpu())
    ari = adjusted_rand_score(pred, y.cpu())
    acc, f1, c = cluster_acc(y.cpu().numpy(), pred)
    return acc, f1, nmi, ari, pred, c, kmeans.transform(z.cpu())


if __name__ == '__main__':
    device = torch.device('cuda:1')
    setup_seed(np.random.randint(1000))

    datasets = 'ACM'

    if datasets in ['Cora', 'Citeseer']:
        x_list, y, graph, edge_index, n_clusters, args, graph_list = load_mutil_feature_data(datasets, device, False)

        aug1 = Augment(EdgeRemoving(pe=args.pe), FeatureMasking(pf=args.pf1), FeatureDropout(pf=args.pf2))
        aug2 = Augment(EdgeRemoving(pe=args.pe), FeatureMasking(pf=args.pf1), FeatureDropout(pf=args.pf2))
        aug3 = Augment(EdgeRemoving(pe=args.pe), FeatureMasking(pf=args.pf1), FeatureDropout(pf=args.pf2))

        encoder_model = MultiFeatureEncoder(x_list=x_list, augmentor=(aug1, aug2, aug3), hidden_dim=args.hidden_dim,
                                            n_clusters=n_clusters, num_layers=args.num_layers, normalize=args.normalize,
                                            device=device).to(device)
        print("model have {} paramerters in total".format(sum(x.numel() for x in encoder_model.parameters())))

        contrast_model = InfoNCELoss(tau=args.tau).to(device)
        contrast_model2 = InfoNCELoss(tau=args.tau).to(device)
        cluster_model = InfoNCELoss(tau=args.tau).to(device)
        sup_contrast = Self_SupConLoss(tau=1)

        pre_optimizer = Adam(encoder_model.parameters(), lr=args.pt_lr, weight_decay=args.pt_wd)
        optimizer = Adam(encoder_model.parameters(), lr=args.lr, weight_decay=args.wd)
        finetune_optimizer = Adam(encoder_model.parameters(), lr=args.ft_lr, weight_decay=args.ft_wd)

        losss = []
        nmis = []
        aris = []
        accs = []
        f1s = []
        epochs1 = args.epochs1
        epochs2 = args.epochs2
        epochs3 = args.epochs3

        kmeans = KMeans(n_clusters=n_clusters, n_init=25)
        acc = 0
        with tqdm(total=epochs1, desc='(Pre)') as pbar:
            for epoch in range(epochs1):
                encoder_model.train()

                loss, con_loss, clu_loss, re_loss = multi_feature_train1(encoder_model, contrast_model, cluster_model,
                                                                         pre_optimizer, x_list,
                                                                         graph, edge_index, device)
                pbar.set_postfix({'acc': acc, 'loss': loss, 'con_loss': con_loss, 'clu_loss': clu_loss})
                pbar.update()

                if epoch % 20 == 0:
                    acc, f1, nmi, ari, _, _, _ = multi_feature_eva(x_list, graph, edge_index, encoder_model, kmeans)
                    nmis.append((epoch, nmi))
                    aris.append((epoch, ari))

                    accs.append((epoch, acc))
                    f1s.append((epoch, f1))

        if epochs1 != 0:
            print('[MAX]NMI: ', max([i[1] for i in nmis]))
            print('[MAX]ARI: ', max([i[1] for i in aris]))
            print('[MAX]ACC: ', max([i[1] for i in accs]))
            print('[MAX]F1: ', max([i[1] for i in f1s]))

        with tqdm(total=epochs2, desc='(Train)') as pbar:
            for epoch in range(epochs2):
                encoder_model.train()
                loss, con_loss, clu_loss, mv_con_loss = multi_feature_train2(encoder_model, contrast_model,
                                                                             contrast_model2,
                                                                             cluster_model,
                                                                             optimizer, x_list,
                                                                             graph, edge_index, device)
                pbar.set_postfix(
                    {'acc': acc, 'loss': loss, 'con_loss': con_loss, 'clu_loss': clu_loss, 'mv_con_loss': mv_con_loss})
                pbar.update()

                writer.add_scalar('loss', loss, epoch)

                if epoch % 20 == 0:
                    acc, f1, nmi, ari, _, _, _ = multi_feature_eva(x_list, graph, edge_index, encoder_model, kmeans)

                    nmis.append((epoch, nmi))
                    aris.append((epoch, ari))

                    accs.append((epoch, acc))
                    f1s.append((epoch, f1))

        if epochs2 != 0:
            print('[MAX]NMI: ', max([i[1] for i in nmis]))
            print('[MAX]ARI: ', max([i[1] for i in aris]))
            print('[MAX]ACC: ', max([i[1] for i in accs]))
            print('[MAX]F1: ', max([i[1] for i in f1s]))

        aug1 = Augment(DegreeAugment(graph_list, [11, 11], 0.3))
        aug2 = Augment(DegreeAugment(graph_list, [11, 11], 0.3))
        aug3 = Augment(DegreeAugment(graph_list, [11, 11], 0.3))
        encoder_model.augmentor = (aug1, aug2, aug3)

        with tqdm(total=epochs3, desc='(Finetune)') as pbar:
            acc, f1, nmi, ari, pred, c, distances = multi_feature_eva(x_list, graph, edge_index, encoder_model, kmeans)

            pred = torch.tensor(pred).to(device).long()
            high_confidence = torch.min(torch.tensor(distances), dim=1).values
            threshold = torch.sort(high_confidence).values[
                int(len(high_confidence) * 0.5)]
            high_confidence_idx = np.argwhere(high_confidence < threshold)[0]
            high_confidence_label = pred[high_confidence_idx]
            T_idx = np.where(c == 1)[0]
            common_elements = np.intersect1d(np.array(high_confidence_idx), T_idx)
            proportion = len(common_elements) / len(high_confidence_idx)

            for epoch in range(epochs3):
                encoder_model.train()
                loss = multi_feature_train3(encoder_model, contrast_model, contrast_model2, cluster_model, sup_contrast,
                                            finetune_optimizer, x_list, graph,
                                            edge_index, pred, high_confidence_idx, high_confidence_label, device)
                pbar.set_postfix({'acc': acc, 'loss': loss})
                pbar.update()
                losss.append(loss)

                if epoch % 10 == 0:
                    acc, f1, nmi, ari, pred, c, distances = multi_feature_eva(x_list, graph, edge_index, encoder_model, kmeans)

                    nmis.append((epochs1 + epochs2 + epoch, nmi))
                    aris.append((epochs1 + epochs2 + epoch, ari))
                    accs.append((epochs1 + epochs2 + epoch, acc))
                    f1s.append((epochs1 + epochs2 + epoch, f1))

                    pred = torch.tensor(pred).to(device).long()
                    high_confidence = torch.min(torch.tensor(distances), dim=1).values
                    threshold = torch.sort(high_confidence).values[
                        int(len(high_confidence) * 0.5)]
                    high_confidence_idx = np.argwhere(high_confidence < threshold)[0]
                    high_confidence_label = pred[high_confidence_idx]
                    T_idx = np.where(c == 1)[0]
                    common_elements = np.intersect1d(np.array(high_confidence_idx), T_idx)
                    proportion = len(common_elements) / len(high_confidence_idx)

        if epochs3 != 0:
            print('[MAX]NMI: ', max([i[1] for i in nmis]))
            print('[MAX]ARI: ', max([i[1] for i in aris]))
            print('[MAX]ACC: ', max([i[1] for i in accs]))
            print('[MAX]F1: ', max([i[1] for i in f1s]))
    elif datasets in ['ACM', 'DBLP', "IMDB"]:
        device = torch.device('cuda')

        x, y, graph_list, n_clusters, args = load_multi_graph_data(datasets, device)

        aug1 = Augment(EdgeRemoving(pe=args.pe), FeatureMasking(pf=args.pf1), FeatureDropout(pf=args.pf2))
        aug2 = Augment(EdgeRemoving(pe=args.pe), FeatureMasking(pf=args.pf1), FeatureDropout(pf=args.pf2))
        aug3 = Augment(EdgeRemoving(pe=args.pe), FeatureMasking(pf=args.pf1), FeatureDropout(pf=args.pf2))

        encoder_model = MultiGraphEncoder(device=device, graph_list=graph_list, augmentor=(aug1, aug2, aug3), input_dim=x.size(1),
                                hidden_dim=args.hidden_dim, n_clusters=n_clusters, num_layers=args.num_layers,
                                normalize=True).to(device)
        print("model have {} paramerters in total".format(sum(x.numel() for x in encoder_model.parameters())))

        contrast_model = InfoNCELoss(tau=args.tau).to(device)
        contrast_model2 = InfoNCELoss(tau=args.tau, intra=False).to(device)
        cluster_model = InfoNCELoss(tau=args.tau).to(device)
        sup_contrast = Self_SupConLoss(tau=0.2)

        pre_optimizer = Adam(encoder_model.parameters(), lr=args.pt_lr, weight_decay=args.pt_wd)
        optimizer = Adam(encoder_model.parameters(), lr=args.lr, weight_decay=args.wd)
        finetune_optimizer = Adam(encoder_model.parameters(), lr=args.ft_lr, weight_decay=args.ft_wd)

        losss = []
        nmis = []
        aris = []
        accs = []
        f1s = []
        epochs1 = args.epochs1
        epochs2 = args.epochs2
        epochs3 = args.epochs3

        kmeans = KMeans(n_clusters=n_clusters, n_init=20)
        acc = 0

        with tqdm(total=epochs1, desc='(Pre)') as pbar:
            for epoch in range(epochs1):
                encoder_model.train()

                loss, con_loss, clu_loss, re_loss = multi_graph_train1(encoder_model, contrast_model, contrast_model2,
                                                           cluster_model,
                                                           optimizer, x, graph_list, device)
                pbar.set_postfix({'acc': acc, 'loss': loss, 'con_loss': con_loss, 'clu_loss': clu_loss})
                pbar.update()

                # writer.add_scalar('loss', loss, epoch)

                if epoch % 20 == 0:
                    acc, f1, nmi, ari, _, _, _ = multi_graph_eva(encoder_model, kmeans)
                    nmis.append((epoch, nmi))
                    aris.append((epoch, ari))

                    accs.append((epoch, acc))
                    f1s.append((epoch, f1))

        if epochs1 != 0:
            mmax = max([i[1] for i in accs])
            mmax_index = [i[1] for i in accs].index(mmax)
            print('[MAX]NMI: ', [i[1] for i in nmis][mmax_index])
            print('[MAX]ARI: ', [i[1] for i in aris][mmax_index])
            print('[MAX]ACC: ', [i[1] for i in accs][mmax_index])
            print('[MAX]F1: ', [i[1] for i in f1s][mmax_index])

        losss = []

        with tqdm(total=epochs2, desc='(Train)') as pbar:
            for epoch in range(epochs2):
                encoder_model.train()
                loss, con_loss, clu_loss, mv_con_loss, re_loss = multi_graph_train2(encoder_model, contrast_model, contrast_model2,
                                                                        cluster_model, optimizer, x, graph_list, device)
                pbar.set_postfix(
                    {'acc': acc, 'loss': loss, 'con_loss': con_loss, 'clu_loss': clu_loss, 'mv_con_loss': mv_con_loss,
                     're_loss': re_loss
                     })
                pbar.update()
                losss.append(loss)
                writer.add_scalar('loss', loss, epoch)

                if epoch % 20 == 0:
                    acc, f1, nmi, ari, _, _, _ = multi_graph_eva(encoder_model, kmeans)

                    nmis.append((epochs1 + epoch, nmi))
                    aris.append((epochs1 + epoch, ari))

                    accs.append((epochs1 + epoch, acc))
                    f1s.append((epochs1 + epoch, f1))

        if epochs2 != 0:
            mmax = max([i[1] for i in accs])
            mmax_index = [i[1] for i in accs].index(mmax)
            print('[MAX]NMI: ', [i[1] for i in nmis][mmax_index])
            print('[MAX]ARI: ', [i[1] for i in aris][mmax_index])
            print('[MAX]ACC: ', [i[1] for i in accs][mmax_index])
            print('[MAX]F1: ', [i[1] for i in f1s][mmax_index])

        aug1 = Augment(DegreeAugment(graph_list, [1234, 1000, 14], 0.2))
        aug2 = Augment(DegreeAugment(graph_list, [1234, 1000, 14], 0.2))
        aug3 = Augment(DegreeAugment(graph_list, [1234, 1000, 14], 0.2))
        encoder_model.augmentor = (aug1, aug2, aug3)

        with tqdm(total=epochs3, desc='(Finetune)') as pbar:
            acc, f1, nmi, ari, pred, c, distances = multi_graph_eva(encoder_model, kmeans)

            pred = torch.tensor(pred).to(device).long()
            high_confidence = torch.min(torch.tensor(distances), dim=1).values
            threshold = torch.sort(high_confidence).values[
                int(len(high_confidence) * 0.9)]
            high_confidence_idx = np.argwhere(high_confidence < threshold)[0]
            high_confidence_label = pred[high_confidence_idx]
            T_idx = np.where(c == 1)[0]
            common_elements = np.intersect1d(np.array(high_confidence_idx), T_idx)
            proportion = len(common_elements) / len(high_confidence_idx)

            for epoch in range(epochs3):
                encoder_model.train()
                loss, pred_c = multi_graph_train3(encoder_model, contrast_model, contrast_model2, cluster_model, sup_contrast,
                                      finetune_optimizer, x, graph_list, pred, high_confidence_idx,
                                      high_confidence_label,
                                      device)

                y_pred = list(pred_c)
                acc_c = []
                nmi_c = []
                ari_c = []
                f1_c = []
                for i in range(3):
                    acc_1, f1_1, c_c = cluster_acc(y.cpu().numpy(), y_pred[i])
                    nmi_1 = normalized_mutual_info_score(y_pred[i], y.cpu())
                    ari_1 = adjusted_rand_score(y_pred[i], y.cpu())
                    nmi_c.append(nmi_1)
                    ari_c.append(ari_1)
                    f1_c.append(f1_1)
                    acc_c.append(acc_1)
                ###
                pbar.set_postfix({'acc': acc, 'loss': loss})
                print(acc_c[0], acc_c[1], acc_c[2], nmi_c[0], nmi_c[1], nmi_c[2], ari_c[0], ari_c[1], ari_c[2], f1_c[0],
                      f1_c[1], f1_c[2])

                pbar.update()
                losss.append(loss)

                if epoch % 10 == 0:
                    acc, f1, nmi, ari, pred, c, distances = multi_graph_eva(encoder_model, kmeans)

                    nmis.append((epochs1 + epochs2 + epoch, nmi))
                    aris.append((epochs1 + epochs2 + epoch, ari))
                    accs.append((epochs1 + epochs2 + epoch, acc))
                    f1s.append((epochs1 + epochs2 + epoch, f1))

                    pred = torch.tensor(pred).to(device).long()
                    high_confidence = torch.min(torch.tensor(distances), dim=1).values
                    threshold = torch.sort(high_confidence).values[
                        int(len(high_confidence) * 0.9)]  # 获取一个threshold的阈值，在threshold的位置的距离是多少
                    high_confidence_idx = np.argwhere(high_confidence < threshold)[0]  # 距离小于这个值的被视为高置信度的样本
                    high_confidence_label = pred[high_confidence_idx]
                    T_idx = np.where(c == 1)[0]
                    common_elements = np.intersect1d(np.array(high_confidence_idx), T_idx)
                    proportion = len(common_elements) / len(high_confidence_idx)

        if epochs3 != 0:
            mmax = max([i[1] for i in accs])
            mmax_index = [i[1] for i in accs].index(mmax)
            print('[MAX]NMI: ', [i[1] for i in nmis][mmax_index])
            print('[MAX]ARI: ', [i[1] for i in aris][mmax_index])
            print('[MAX]ACC: ', [i[1] for i in accs][mmax_index])
            print('[MAX]F1: ', [i[1] for i in f1s][mmax_index])
