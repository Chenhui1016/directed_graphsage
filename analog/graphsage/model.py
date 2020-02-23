import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import os

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from collections import defaultdict

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator
from graphsage.plot_confusion_matrix import plot_confusion_matrix

"""
Simple supervised GraphSAGE model for directed graph as well as examples running the model
on the EDA datasets.
"""

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10]))
        # pos_weight controls weight of "1" label in loss function

        self.weight = nn.Parameter(torch.FloatTensor(enc.embed_dim, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, pair1, pair2):
        embed1 = (self.enc(pair1)).t()
        embed2 = (self.enc(pair2))

        scores = torch.diag(torch.mm(embed1, self.weight.mm(embed2)))
        return scores

    def loss(self, pair1, pair2, labels):
        scores = self.forward(pair1, pair2)
        return self.xent(scores, labels)

def load_cora():
    data_dir = "data"
    feat_data = np.load("{}/feats.npy".format(data_dir))
    num_nodes = feat_data.shape[0]
    #labels = np.empty((num_nodes,1), dtype=np.int64)
    train = []
    test  = []
    train_label = []
    test_label = []
    with open("{}/labels.txt".format(data_dir)) as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            #labels.append(int(info[2]))
            if int(info[3]) == 1:
                train.append([int(info[0]), int(info[1])])
                train_label.append(int(info[2]))
            else:
                test.append([int(info[0]), int(info[1])])
                test_label.append(int(info[2]))

    adj_lists = defaultdict(lambda: defaultdict(set))
    with open("{}/all.edgelist".format(data_dir)) as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = int(info[0])
            paper2 = int(info[1])
            adj_lists[paper1]["out"].add(paper2)
            adj_lists[paper2]["in"].add(paper1)
    return feat_data, train_label, test_label, adj_lists, train, test

def shuffle_list(*ls):
    l =list(zip(*ls))
    random.shuffle(l)
    return zip(*l)

def main():
    np.random.seed(1)
    random.seed(1)
    feat_data, train_label, test_label, adj_lists, train, test = load_cora()
    num_nodes = feat_data.shape[0]
    feat_dim = feat_data.shape[1]
    hidden_dim = 15
    features = nn.Embedding(num_nodes, feat_dim)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    #features.cuda()

    agg1 = MeanAggregator(features, cuda=False)
    enc1 = Encoder(features, feat_dim, hidden_dim, adj_lists, agg1, gcn=False, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, hidden_dim, adj_lists, agg2,
            base_model=enc1, gcn=False, cuda=False)
    enc1.num_samples = 10
    enc2.num_samples = 10

    graphsage = SupervisedGraphSage(hidden_dim, enc2)
    #graphsage.cuda()

    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.01, weight_decay=1e-5)
    times = []
    epoch = 100
    batch_size = 5120
    num_batch = len(train)//batch_size
    best = 1e9
    cnt_wait = 0
    patience = 25
    best_epoch = 0
    best_batch = 0

    train_pair1 = []
    train_pair2 = []
    test_pair1 = []
    test_pair2 = []
    for x in train:
        train_pair1.append(x[0])
        train_pair2.append(x[1])
    for x in test:
        test_pair1.append(x[0])
        test_pair2.append(x[1])
    for e in range(epoch):
        # shuffle training set
        fused_train = [list(x) for x in shuffle_list(train_pair1,train_pair2,train_label)]
        train_pair1 = fused_train[0]
        train_pair2 = fused_train[1]
        train_label = fused_train[2]

        for i in range(num_batch):
            if i < num_batch - 1:
                pair1 = train_pair1[i*batch_size: i*batch_size + batch_size]
                pair2 = train_pair2[i*batch_size: i*batch_size + batch_size]
                sub_label = train_label[i*batch_size: i*batch_size + batch_size]
            else:
                pair1 = train_pair1[i*batch_size: len(train_pair1)]
                pair2 = train_pair2[i*batch_size: len(train_pair2)]
                sub_label = train_label[i*batch_size: len(train_pair1)]
            start_time = time.time()
            optimizer.zero_grad()
            loss = graphsage.loss(pair1, pair2,\
                Variable(torch.FloatTensor(np.asarray(sub_label))))

            '''
            if loss < best:
                best = loss
                best_epoch = e
                best_batch = i
                cnt_wait = 0
                torch.save(graphsage.state_dict(), 'best_model.pkl')

            else:
                cnt_wait += 1

            if cnt_wait == patience:
                print('Early stopping!')
                break
            '''

            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time-start_time)
            print("The {}-th epoch, The {}-th batch, ".format(e, i), "Loss: ", loss.item())

    #print('Loading {}th epoch {}th batch'.format(best_epoch, best_batch))
    #graphsage.load_state_dict(torch.load('best_model.pkl'))

    if len(test) < 100000:
        test_output = torch.sigmoid(graphsage.forward(test_pair1, test_pair2))
        pred = np.where(test_output.data.numpy() < 0.5, 0, 1)

    ### Inference on large graph, avoid out of memory
    else:
        chunk_size = 5120
        pred = []
        for j in range(len(test)//chunk_size):
            if j < (len(test)//chunk_size-1):
                pair1 = test_pair1[j*chunk_size:(j+1)*chunk_size]
                pair2 = test_pair2[j*chunk_size:(j+1)*chunk_size]
            else:
                pair1 = test_pair1[j*chunk_size:len(test_pair1)]
                pair2 = test_pair2[j*chunk_size:len(test_pair2)]
            test_output = torch.sigmoid(graphsage.forward(pair1, pair2))
            pred = np.concatenate((pred, np.where(test_output.data.numpy() < 0.5, 0, 1)), axis=None)
            print("Inference on the {}-th chunk".format(j))
    print("Test F1:", f1_score(np.asarray(test_label), pred, average="micro", labels=[1]))
    print("Test Recall:", recall_score(np.asarray(test_label), pred, average="micro", labels=[1]))
    print("Test Precision:", precision_score(np.asarray(test_label), pred, average="micro", labels=[1]))
    print("False Positive Rate:", 1-recall_score(np.asarray(test_label), pred, average="micro", labels=[0]))
    plot_confusion_matrix(np.asarray(test_label), pred, np.array([0, 1]), title='Confusion matrix, without normalization')

    print("Average batch time:", np.mean(times))


if __name__ == "__main__":
    main()
