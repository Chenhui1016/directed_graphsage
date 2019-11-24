# Reference Directed GraphSAGE Implementation
### Author: William L. Hamilton, Chenhui Deng


Currently, only supervised versions of GraphSAGE-mean and GraphSAGE-GCN are implemented. 

#### Requirements

pytorch >0.2 is required.

#### Running examples

Execute `python -m graphsage.model` to run.
It assumes that CUDA is not being used, but modifying the run functions in `model.py` in the obvious way can change this.

#### Dataset format

Please refer to the file format in trojand/ directory.

The trojand/feats.npy is missing due to its large size. Please generate the node feature np.array by yourself with shape: (num\_nodes, feat\_dim).

This version can suppoer directed large graphs with millions of nodes.
