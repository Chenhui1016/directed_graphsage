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
from scipy.sparse import csr_matrix

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator
from graphsage.plot_confusion_matrix import plot_confusion_matrix

"""
Simple supervised GraphSAGE model for directed graph as well as examples running the model
on the EDA datasets.
"""

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc, edge_map):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([16]))

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)

        self.edge_map = edge_map

    def forward(self, edges):
        node_list1 = []
        node_list2 = []
        for e in edges:
            node_list1 += [self.edge_map[e][0]]
            node_list2 += [self.edge_map[e][1]]
        node_embed1 = (self.enc(node_list1)).t()
        node_embed2 = (self.enc(node_list2)).t()
        edge_embeds = (0.5 * (node_embed1 + node_embed2)).t()
        scores = self.weight.mm(edge_embeds)
        return scores.t()

    def loss(self, edges, labels):
        scores = self.forward(edges)
        return self.xent(scores.squeeze(), labels)
        #return self.xent(scores, labels)

def load_cora():
    data_dir = "dataset"
    feat_data = np.load("{}/feats.npy".format(data_dir))
    #num_nodes = feat_data.shape[0]
    #labels = np.empty((num_nodes,1), dtype=np.int64)
    labels = []
    train = []
    test  = []
    '''
    with open("{}/labels.txt".format(data_dir)) as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            labels[int(info[0])] = int(info[1])
            if int(info[2]) == 1:
                train.append(int(info[0]))
            else:
                test.append(int(info[0]))
    '''
    adj_lists = defaultdict(lambda: defaultdict(set))
    row = []
    col = []
    data = []
    edge_map = {}
    with open("{}/all.edgelist".format(data_dir)) as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = int(info[0])
            paper2 = int(info[1])
            adj_lists[paper1]["out"].add(paper2)
            adj_lists[paper2]["in"].add(paper1)
            labels.append(int(info[-2]))
            if int(info[-1]) == 1:
                train.append(i)
            else:
                test.append(i)
            row += [i, i]
            col += [paper1, paper2]
            data += [0.5, 0.5]
            num_edges = i+1
            edge_map[i] = [paper1, paper2]
    #incident_matrix = csr_matrix((data, (row, col)), shape=(num_edges, num_nodes))
    labels = np.asarray(labels)
    return feat_data, labels, adj_lists, train, test, edge_map

def main():
    np.random.seed(1)
    random.seed(1)
    feat_data, labels, adj_lists, train, test, edge_map = load_cora()
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

    graphsage = SupervisedGraphSage(1, enc2, edge_map)
    #graphsage.cuda()

    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.001, weight_decay=1e-5)
    times = []
    epoch = 10
    batch_size = 512
    num_batch = len(train)//batch_size
    best = 1e9
    cnt_wait = 0
    patience = 20
    best_t = 0
    for e in range(epoch):
        for i in range(num_batch):
            if i < num_batch-1:
                batch_nodes = train[i*batch_size: i*batch_size + batch_size]
            else:
                batch_nodes = train[i*batch_size: len(train)]
            start_time = time.time()
            optimizer.zero_grad()
            loss = graphsage.loss(batch_nodes,\
                Variable(torch.FloatTensor(labels[np.array(batch_nodes)])))
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
        print("Test F1:", f1_score(labels[test], pred, labels = [1], average="micro"))
        print("Test Recall:", recall_score(labels[test], pred, labels = [1], average="micro"))
        print("Test Precision:", precision_score(labels[test], pred, labels = [1], average="micro"))
        cm = plot_confusion_matrix(labels[test], pred, np.array([0, 1]), title='Confusion matrix, without normalization')
        #recall = cm[1][1]/(cm[1][0]+cm[1][1])
        #precision = cm[1][1]/(cm[1][1]+cm[0][1])
        #f1 = 2*recall*precision/(recall+precision)
        #print("Test F1 micro:", f1)
        #print("Test Recall micro:", recall)
        #print("Test Precision micro:", precision)

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
        cm = plot_confusion_matrix(labels[test], np.asarray(pred), np.array([0, 1]), title='Confusion matrix, without normalization')
        print("Test F1:", f1_score(labels[test], np.asarray(pred), labels = [1], average="micro"))
        print("Test Recall:", recall_score(labels[test], np.asarray(pred), labels = [1], average="micro"))
        print("Test Precision:", precision_score(labels[test], np.asarray(pred), labels = [1], average="micro"))


    print("Average batch time:", np.mean(times))


if __name__ == "__main__":
    main()
