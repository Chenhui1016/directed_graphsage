import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

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
        self.xent = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2]))

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        #return self.xent(scores, labels.squeeze())
        return self.xent(scores, labels)

def load_cora():
    data_dir = "dataset"
    feat_data = np.load("{}/feats.npy".format(data_dir))
    num_nodes = feat_data.shape[0]
    labels = np.empty((num_nodes,1), dtype=np.int64)
    train = []
    test  = []
    with open("{}/labels.txt".format(data_dir)) as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            labels[int(info[0])] = int(info[1])
            if int(info[2]) == 1:
                train.append(int(info[0]))
            else:
                test.append(int(info[0]))

    adj_lists = defaultdict(lambda: defaultdict(set))
    with open("{}/all.edgelist".format(data_dir)) as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = int(info[0])
            paper2 = int(info[1])
            adj_lists[paper1]["out"].add(paper2)
            adj_lists[paper2]["in"].add(paper1)
    return feat_data, labels, adj_lists, train, test

def main():
    np.random.seed(1)
    random.seed(1)
    feat_data, labels, adj_lists, train, test = load_cora()
    num_nodes = feat_data.shape[0]
    feat_dim = feat_data.shape[1]
    hidden_dim = 20
    features = nn.Embedding(num_nodes, feat_dim)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    #features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, feat_dim, hidden_dim, adj_lists, agg1, gcn=False, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, hidden_dim, adj_lists, agg2,
            base_model=enc1, gcn=False, cuda=False)
    enc1.num_samples = 10
    enc2.num_samples = 10

    graphsage = SupervisedGraphSage(1, enc2)
    #graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    #train = rand_indices[:400]
    #val = rand_indices[1000:1500]
    #test = rand_indices[400:]

    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.001, weight_decay=5e-4)
    times = []
    epoch = 1000
    batch_size = 512
    #num_batch = len(train)//batch_size
    num_batch = 1
    best = 1e9
    cnt_wait = 0
    patience = 20
    best_t = 0
    random.shuffle(train)
    for e in range(epoch):
        for i in range(num_batch):
            #if i < num_batch-1:
                #batch_nodes = train[i*batch_size: i*batch_size + batch_size]
            #else:
                #batch_nodes = train[i*batch_size: len(train)]
            start_time = time.time()
            optimizer.zero_grad()
            #loss = graphsage.loss(batch_nodes,\
            loss = graphsage.loss(train,\
                Variable(torch.FloatTensor(labels[np.array(train)])))
                #Variable(torch.FloatTensor(labels[np.array(batch_nodes)])))
                #Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time-start_time)
            print("The {}-th epoch ".format(e), "{}-th batch".format(i), "Loss: ", loss.item())


            if loss.item() < best:
                best_loss = loss.item()
                cnt_wait = 0
                best_t = e
                torch.save(graphsage.state_dict(), 'best_model.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == patience:
                print("early stopping!")
                break

    print('Loading {}th epoch'.format(best_t))
    graphsage.load_state_dict(torch.load('best_model.pkl'))

    if len(test) < 100000:
        test_output = torch.sigmoid(graphsage.forward(test))
        pred = (np.where(test_output.data.numpy() < 0.5, 0, 1))
        print("Test F1:", f1_score(labels[test], pred, average="micro", labels=[1]))
        print("Test Recall:", recall_score(labels[test], pred, average="micro", labels=[1]))
        print("Test Precision:", precision_score(labels[test], pred, average="micro", labels=[1]))
        plot_confusion_matrix(labels[test], pred, np.array([0, 1]), title='Confusion matrix, without normalization')

    ### Inference on large graph, avoid out of memory
    else:
        chunk_size = 5120
        pred = []
        for j in range(len(test)//chunk_size):
            if j < (len(test)//chunk_size-1):
                test_output = torch.sigmoid(graphsage.forward(test[j*chunk_size:(j+1)*chunk_size]))
            else:
                test_output = torch.sigmoid(graphsage.forward(test[j*chunk_size:len(test)]))
            pred += (np.where(test_output.data.numpy() < 0.5, 0, 1)).tolist()
            print("Inference on the {}-th chunk".format(j))
        print("Test F1 micro:", f1_score(labels[test], np.asarray(pred), average="micro", labels=[1]))
        print("Test Recall micro:", recall_score(labels[test], np.asarray(pred), average="micro", labels=[1]))
        print("Test Precision micro:", precision_score(labels[test], np.asarray(pred), average="micro", labels=[1]))
        plot_confusion_matrix(labels[test], np.asarray(pred), np.array([0, 1]), title='Confusion matrix, without normalization')

    print("Average batch time:", np.mean(times))


if __name__ == "__main__":
    main()
