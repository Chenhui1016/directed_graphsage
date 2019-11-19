# Reference Directed GraphSAGE Implementation
### Author: William L. Hamilton, Chenhui Deng


Currently, only supervised versions of GraphSAGE-mean and GraphSAGE-GCN are implemented. 

#### Requirements

pytorch >0.2 is required.

#### Running examples

Execute `python -m graphsage.model` to run the Cora example.
It assumes that CUDA is not being used, but modifying the run functions in `model.py` in the obvious way can change this.
