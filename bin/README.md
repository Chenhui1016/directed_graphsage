# Reference Directed GraphSAGE Implementation
### Author: William L. Hamilton, Chenhui Deng


Currently, only supervised versions of GraphSAGE-mean and GraphSAGE-GCN are implemented. 

#### Requirements

pytorch >0.2 is required.

#### Running examples

Execute `python -m graphsage.model` to run.
It assumes that CUDA is not being used, but modifying the run functions in `model.py` in the obvious way can change this.

#### Dataset format

Please refer to the file format in dataset/ directory.

#### New
This version can suppoer directed large graphs with millions of nodes for binary classification.

#### Imbanlanced dataset

This version can handle imbalanced datasets by putting more focuses on minor class. Please change pos-weight value in BCE loss function.
