##
# @file   readgraph.py
# @author Yibo Lin
# @date   Feb 2020
#

import os
import sys
import pdb
import numpy as np
import networkx as nx
import pickle
import networkx as nx
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import random

class SpiceEntry (object):
    def __init__(self):
        self.name = ""
        self.pins = []
        self.cell = None
        self.attributes = {}

    def __str__(self):
        content = "name: " + self.name
        content += "; pins: " + " ".join(self.pins)
        content += "; cell: " + self.cell
        content += "; attr: " + str(self.attributes)
        return content

    def __repr__(self):
        return self.__str__()

class SpiceSubckt (object):
    def __init__(self):
        self.name = ""
        self.pins = []
        self.entries = []

    def __str__(self):
        content = "subckt: " + self.name + "\n"
        content += "pins: " + " ".join(self.pins) + "\n"
        content += "entries: \n";
        for entry in self.entries:
            content += str(entry) + "\n"
        return content

class SpiceNode (object):
    def __init__(self):
        self.id = None
        self.attributes = {} # include name (named in hierarchy), cell
        self.pins = []
    def __str__(self):
        content = "SpiceNode( " + str(self.id) + ", " + str(self.attributes) + ", " + str(self.pins) + " )"
        return content
    def __repr__(self):
        return self.__str__()

class SpiceNet (object):
    def __init__(self):
        self.id = None
        self.attributes = {} # include name
        self.pins = []
    def __str__(self):
        content = "SpiceNet( " + str(self.id) + ", " + str(self.attributes) + ", " + str(self.pins) + " )"
        return content
    def __repr__(self):
        return self.__str__()

class SpicePin (object):
    def __init__(self):
        self.id = None
        self.node_id = None
        self.net_id = None
        self.attributes = {} # include type
    def __str__(self):
        content = "SpicePin( " + str(self.id) + ", node: " + str(self.node_id) + ", net: " + str(self.net_id) + " attributes: " + str(self.attributes) + " )"
        return content
    def __repr__(self):
        return self.__str__()

class SpiceGraph (object):
    def __init__(self):
        self.nodes = []
        self.pins = []
        self.nets = []
    def __str__(self):
        content = "SpiceGraph\n"
        for node in self.nodes:
            content += str(node) + "\n"
        for pin in self.pins:
            content += str(pin) + "\n"
        for net in self.nets:
            content += str(net) + "\n"
        return content

def draw_graph(G, labels, color):
    color_map = []
    for node in G:
        flag = 0
        for i in range(len(color)):
            if node in color[i]:
                color_map.append(10*i+10)
                flag = 1
        if flag == 0:
            color_map.append(10*len(color)+10)
    #for node in G:
        #if node in color:
            #color_map.append('skyblue')
        #else:
            #color_map.append('green')
    pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
    options = {'arrowstyle': '-|>', 'arrowsize': 12}
    nx.draw(G, font_weight='bold', pos=pos, node_color=color_map, **options, cmap=plt.cm.Blues)
    nx.draw_networkx_labels(G,pos,labels,font_size=16)
    plt.savefig('graph.pdf', dpi=120)

def convert(integer, length):
    bool_list = [0] * length
    bool_list[integer] = 1
    return bool_list

if __name__ == '__main__':
    # pickle file
    filename = sys.argv[1]

    with open(filename, "rb") as f:
        dataX, dataY = pickle.load(f)

    save_dir = "data/"
    G = nx.Graph()
    node_att = {}
    num_nodes = 0   # used to merge subgraphs by changing node indices
    all_pairs = []  # store all pos and neg node pairs
    node_type = []  # store types of all nodes
    ratio = 0.7     # #training_samples/#total_samples
    for i in range(len(dataX)):
        train = i < int(len(dataX)*ratio)   # split training and test set
            #train = True
        #else:
            #train = False
        subckts = dataX[i]["subckts"] # raw subcircuits read from spice netlist
        graph = dataX[i]["graph"] # hypergraph
        label = dataY[i] # symmetry pairs of node indices, self-symmetry if a pair only has one element
        pin_map = {}    # map pins to node id
        for g in graph.nodes:
            node_att[g.id+num_nodes] = g.attributes['cell']
            node_type.append(g.attributes['cell'])
            for p in g.pins:
                pin_map[p] = g.id
        for n in graph.nets:
            node_list = []
            for pin in n.pins:
                if pin_map[pin] not in node_list:
                    node_list.append(pin_map[pin])
            edges = combinations(node_list, 2)
            for edge in edges:
                G.add_edge(edge[0]+num_nodes, edge[1]+num_nodes)

        node_pairs = list(combinations(list(G.nodes()), 2)) # all possible node pairs
        # only add neg pair whose nodes are from the same subgraph 
        # node_pairs = list(combinations([t for t in range(num_nodes, num_nodes+len(graph.nodes))], 2))
        random.seed(1)
        random.shuffle(node_pairs)
        neg_pairs = []
        neg_size = 20
        for pair in node_pairs:
            if [pair[0]-num_nodes, pair[1]-num_nodes] in label:
                continue
            if train:
                neg_pairs.append([pair[0], pair[1], 0, 1])
                # first two cols are node ids, the third col is the label, the last col is train or test
            else:
                neg_pairs.append([pair[0], pair[1], 0, 0])
            if len(neg_pairs) >neg_size*len(label):   # select negative samples, whose size is controled by neg_size
                break

        pos_pairs = []
        for l in label:
            if len(l) == 1:
                continue
            if train:
                pos_pairs.append([l[0]+num_nodes, l[1]+num_nodes, 1, 1])
            else:
                pos_pairs.append([l[0]+num_nodes, l[1]+num_nodes, 1, 0])

        all_pairs += pos_pairs + neg_pairs
        num_nodes += len(graph.nodes)

        #draw_graph(G, node_att, label)

    # convert node types into one-hot vector
    all_type = {}
    for x in node_type:
        if x not in all_type:
            all_type[x] = len(all_type)
    num_types = len(all_type)
    feat = []
    for x in node_type:
        feat.append(convert(all_type[x], num_types))
    feats = np.array([np.array(x) for x in feat])

    # save all files
    np.save(save_dir+"feats.npy", feats)
    nx.write_edgelist(G, save_dir+"all.edgelist")
    with open(save_dir+"labels.txt", "w") as ff:
        for pair in all_pairs:
            ff.write((str(pair[0])+" "+str(pair[1])+" "+str(pair[2])+" "+str(pair[3])+"\n"))


