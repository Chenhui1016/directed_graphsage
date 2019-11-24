import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator

"""
Simple supervised GraphSAGE model for directed graph as well as examples running the model
on the Cora datasets.
"""

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())

def load_cora():
    data_dir = "trojand"
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

def run_cora():
    np.random.seed(1)
    random.seed(1)
    feat_data, labels, adj_lists, train, test = load_cora()
    num_nodes = feat_data.shape[0]
    feat_dim = feat_data.shape[1]
    hidden_dim = 15
    features = nn.Embedding(num_nodes, feat_dim)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    #features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, feat_dim, hidden_dim, adj_lists, agg1, gcn=False, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, hidden_dim, adj_lists, agg2,
            base_model=enc1, gcn=False, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(2, enc2)
    #graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    #train = rand_indices[:400]
    #val = rand_indices[1000:1500]
    #test = rand_indices[400:]

    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.001)
    times = []
    for batch in range(300):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes,
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.item())

    if len(test) < 2560:
        test_output = graphsage.forward(test)
        print("Test F1:", f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro"))

    ### Inference on large graph, avoid out of memory
    else:
        chunk_size = 256
        pred = []
        for j in range(int(len(test)/chunk_size)):
            if j < (int(len(test)/chunk_size)-1):
                test_output = graphsage.forward(test[j*chunk_size:(j+1)*chunk_size])
            else:
                test_output = graphsage.forward(test[j*chunk_size:len(test)])
            pred += (test_output.data.numpy().argmax(axis=1)).tolist()
            print("Inference on the {}-th chunk".format(j))
        print("Test F1:", f1_score(labels[test], np.asarray(pred), average="micro"))
    print("Average batch time:", np.mean(times))


if __name__ == "__main__":
    run_cora()
