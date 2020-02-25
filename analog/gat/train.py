from __future__ import division
from __future__ import print_function

import os
import networkx as nx

import sys
import glob
import time
import random
import argparse
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy, normalize_adj
from models import GAT, SpGAT
from plot_confusion_matrix import plot_confusion_matrix

def spy_sparse2torch_sparse(coo):
    """
    :param data: a scipy sparse coo matrix
    :return: a sparse torch tensor
    """
    values=coo.data
    indices=np.vstack((coo.row,coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    t = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return t

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_false', default=True, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=15, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=20, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
#adj, features, labels, idx_train, idx_val, idx_test = load_data()
data_dir = "../data"
features = torch.FloatTensor(np.load("{}/feats.npy".format(data_dir)))
num_nodes = features.size()[0]
G = nx.Graph()
with open("{}/all.edgelist".format(data_dir)) as ff:
    for i,line in enumerate(ff):
        info = line.split()
        G.add_edge(int(info[0]), int(info[1]))
# add isolated nodes
for i in range(num_nodes):
     G.add_node(i)
adj = nx.to_scipy_sparse_matrix(G)
# build symmetric adjacency matrix
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
adj = normalize_adj(adj + sp.eye(adj.shape[0]))
adj = torch.FloatTensor(np.array(adj.todense()))

train_label = []
test_label = []

train_row = []
train_col1 = []
train_col2 = []
train_data = []

test_row = []
test_col1 = []
test_col2 = []
test_data = []

train_cnt = 0
test_cnt = 0
with open("{}/labels.txt".format(data_dir)) as fp:
    for i,line in enumerate(fp):
        info = line.strip().split()
        if int(info[3]) == 1:
            train_label.append(int(info[2]))
            train_row.append(train_cnt)
            train_col1.append(int(info[0]))
            train_col2.append(int(info[1]))
            train_data.append(1)
            train_cnt += 1
        else:
            test_label.append(int(info[2]))
            test_row.append(test_cnt)
            test_col1.append(int(info[0]))
            test_col2.append(int(info[1]))
            test_data.append(1)
            test_cnt += 1

train_map1 = spy_sparse2torch_sparse(coo_matrix((train_data, (train_row, train_col1)), shape=(train_cnt, num_nodes)))
train_map2 = spy_sparse2torch_sparse(coo_matrix((train_data, (train_row, train_col2)), shape=(train_cnt, num_nodes)))

test_map1 = spy_sparse2torch_sparse(coo_matrix((test_data, (test_row, test_col1)), shape=(test_cnt, num_nodes)))
test_map2 = spy_sparse2torch_sparse(coo_matrix((test_data, (test_row, test_col2)), shape=(test_cnt, num_nodes)))

# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=features.shape[1],
                nhid=args.hidden,
                #nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha)
else:
    model = GAT(nfeat=features.shape[1],
                nhid=args.hidden,
                #nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    #labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

features, adj = Variable(features), Variable(adj)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    #output = model(features, adj)
    loss_train = model.loss(features, adj, Variable(torch.FloatTensor(np.asarray(train_label))), train_map1, train_map2)
    #acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    #if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        #model.eval()
        #output = model(features, adj)

    #loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    #acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          #'acc_train: {:.4f}'.format(acc_train.data.item()),
          #'loss_val: {:.4f}'.format(loss_val.data.item()),
          #'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_train.data.item()


def compute_test():
    model.eval()
    #output = model(features, adj)
    #loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    test_output = torch.sigmoid(model.forward(features, adj, test_map1, test_map2))
    pred = np.where(test_output.data.numpy() < 0.5, 0, 1)
    print("True Positive Rate:", recall_score(np.asarray(test_label), pred, average="micro", labels=[1]))
    print("False Positive Rate:", 1-recall_score(np.asarray(test_label), pred, average="micro", labels=[0]))
    plot_confusion_matrix(np.asarray(test_label), pred, np.array([0, 1]), title='Confusion matrix, without normalization')
    #acc_test = accuracy(output[idx_test], labels[idx_test])
    #print("Test set results:",
          #"loss= {:.4f}".format(loss_test.item()),
          #"accuracy= {:.4f}".format(acc_test.item()))

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
compute_test()
