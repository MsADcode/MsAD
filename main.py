import numpy as np
from diffuse.diffuse import Diffuse
import os
import torch
import random
import sys
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.set_printoptions(precision=3)
from castle.algorithms.pc.pc import find_skeleton
import networkx as nx
import community
import math
# sys.stdout = open('main.txt', 'w')


def compute_evaluate(W_p, W_true):
    assert (W_p.shape == W_true.shape and W_p.shape[0] == W_p.shape[1])
    TP = np.sum((W_p + W_true) == 2)
    TP_FP = W_p.sum(axis=1).sum()   #TP+FP
    TP_FN = W_true.sum(axis=1).sum()  #TP+FN
    TN = ((W_p + W_true) == 0).sum()

    accuracy = (TP + TN) / (W_p.shape[0]*W_p.shape[0])
    precision = TP / TP_FP
    recall = TP / TP_FN
    F1 = 2 * (recall * precision) / (recall + precision)
    shd = np.count_nonzero(W_p != W_true)

    mt = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'F1': F1,'shd': shd}
    for i in mt:
        mt[i] = round(mt[i], 4)  #4-digit decimal

    return mt

def main():
    # parameters-----------------------------------------------
    # train
    epochs = int(4000)
    batch_size = 200
    orderSize = 200

    # diffusion
    n_steps = int(1e4)
    beta_start = 0.0001
    beta_end=0.2
    n_votes = 9
    epsilon = -0.22

    # model
    dim = 64

    # prune
    prunSize = 200
    thresh = 0.18  # 0.19

    # datasets--------------------------------------------------
    val_ratio = 0.28
    subject_num=50

    # algorithm-------------------------------------------------
    if file_path == 'simsTxt/sim200.csv':
        X = pd.read_csv(file_path, header=None).to_numpy()
    elif file_path == 'simsTxt/sim500.csv':
        X = pd.read_csv(file_path, header=None).to_numpy()
    else:
        X = np.loadtxt(file_path)

    true_causal_matrix = np.loadtxt(gtrue_path)
    n_nodes = true_causal_matrix.shape[0]

    skeleton, _ = find_skeleton(X, float(1e-20), 'fisherz')

    graph = nx.Graph(skeleton)
    if X.shape[1] == 5:
        partition = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    else:
        partition = community.best_partition(graph, random_state=42)
    print(partition)
    community_ids = set(partition.values())
    num_communities = len(community_ids)
    adj_all = np.zeros((subject_num, n_nodes, n_nodes))
    print(num_communities)
    for community_id in range(num_communities):
        nodes_with_id = [[key for key, value in partition.items() if value == community_id]]
        subX = X[:, nodes_with_id].squeeze()

        sub_n_nodes=len(nodes_with_id[0])

        np.random.seed(42)
        random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        diffuse = Diffuse(dim, val_ratio,  epsilon, sub_n_nodes, n_votes, beta_start, beta_end, orderSize,
                        prunSize, thresh,
                        epochs, batch_size, n_steps)
        diffuse.fit1(subX)

        for id_subject in range(subject_num):
            start = int(id_subject * (prunSize))
            end = int((id_subject + 1) * (prunSize))
            prunX = subX[start:end, :]
            sub_adj_matrix = diffuse.fit2(prunX)

            for i_subject in range(sub_adj_matrix.shape[0]):
                for j_subject in range(sub_adj_matrix.shape[0]):
                    if sub_adj_matrix[i_subject][j_subject] == 1:
                        adj_all[id_subject][nodes_with_id[0][i_subject]][nodes_with_id[0][j_subject]] = 1


    #boundary----------------------------------
    if X.shape[1] >=10:

        for node_i in range(n_nodes):
            for node_j in range(node_i+1,n_nodes):
                if partition.get(node_i)!=partition.get(node_j):
                    if skeleton[node_i][node_j]==1:
                        nodes_with_id=[node_i,node_j]

                        subX = X[:, nodes_with_id].squeeze()

                        sub_n_nodes=len(nodes_with_id)

                        np.random.seed(42)
                        random.seed(42)
                        torch.manual_seed(42)
                        torch.cuda.manual_seed(42)
                        torch.cuda.manual_seed_all(42)
                        torch.backends.cudnn.benchmark = False
                        torch.backends.cudnn.deterministic = True

                        diffuse = Diffuse(dim, val_ratio,  epsilon, sub_n_nodes, n_votes, beta_start, beta_end, orderSize,
                                        prunSize, thresh,
                                        epochs, batch_size,  n_steps)
                        diffuse.fit1(subX)

                        for id_subject in range(subject_num):
                            start = int(id_subject * (prunSize))
                            end = int((id_subject + 1) * (prunSize))
                            prunX = subX[start:end, :]
                            sub_adj_order = diffuse.fit3(prunX)

                            adj_all[id_subject][nodes_with_id[sub_adj_order[0]]][nodes_with_id[sub_adj_order[1]]] = 1




    #f1---------------------------------------------------------------------
    accuracy_all = []
    precision_all = []
    recall_all = []
    f1_all = []
    shd_all = []

    for id_subject in range(subject_num):
        # np.savetxt("msad/" + str(n_nodes) + "/" + str(id_subject) + ".txt", adj_all[id_subject], fmt='%d')

        mt = compute_evaluate(adj_all[id_subject], true_causal_matrix)
        print(mt)

        accuracy_all.append(mt['accuracy'])
        precision_all.append(mt['precision'])
        recall_all.append(mt['recall'])
        f1_all.append(mt['F1'])
        shd_all.append(mt['shd'])

    mean_accuracy = np.mean(accuracy_all)
    std_accuracy = np.std(accuracy_all)
    mean_precision = np.mean(precision_all)
    std_precision = np.std(precision_all)
    mean_recall = np.mean(recall_all)
    std_recall = np.std(recall_all)
    mean_F1 = np.mean(f1_all)
    std_F1 = np.std(f1_all)
    mean_shd = np.mean(shd_all)
    std_shd = np.std(shd_all)

    print("{} mean+std--accuracy: {:.2f} + {:.2f}".format(n_nodes, mean_accuracy, std_accuracy))
    print("{} mean+std--precision: {:.2f} + {:.2f}".format(n_nodes, mean_precision, std_precision))
    print("{} mean+std--recall: {:.2f} + {:.2f}".format(n_nodes, mean_recall, std_recall))
    print("{} mean+std--F1: {:.2f} + {:.2f}".format(n_nodes, mean_F1, std_F1))
    print("{} mean+std--shd: {:.2f} + {:.2f}".format(n_nodes, mean_shd, std_shd))


if __name__ == "__main__":

    file_path = 'simsTxt/sim1.txt'
    gtrue_path = 'simsTxt/stand_5nodes.txt'
    main()
