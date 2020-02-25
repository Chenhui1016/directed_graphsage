import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer
from torch.nn import init


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.xent = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10]))
        self.weight = nn.Parameter(torch.FloatTensor(nhid, nhid))
        init.xavier_uniform_(self.weight, gain=1.414)

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj, pair1_map, pair2_map):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        embed1 = torch.sparse.mm(pair1_map, x)
        embed2 = torch.sparse.mm(pair2_map, x)
        scores = torch.diag(torch.mm(embed1, self.weight.mm(embed2.t())))
        return scores

    def loss(self, x, adj, labels, pair1_map, pair2_map):
        scores = self.forward(x, adj, pair1_map, pair2_map)
        return self.xent(scores, labels)

class SpGAT(nn.Module):
    #def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.xent = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10]))
        self.weight = nn.Parameter(torch.FloatTensor(nhid, nhid))
        init.xavier_uniform_(self.weight, gain=1.414)

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj, pair1_map, pair2_map):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        embed1 = torch.sparse.mm(pair1_map, x)
        embed2 = torch.sparse.mm(pair2_map, x)
        scores = torch.diag(torch.mm(embed1, self.weight.mm(embed2.t())))
        #return F.log_softmax(x, dim=1)
        return scores

    def loss(self, x, adj, labels, pair1_map, pair2_map):
        scores = self.forward(x, adj, pair1_map, pair2_map)
        return self.xent(scores, labels)


