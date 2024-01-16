# Higher Order OCclusion (hoocs)
## Introduction
Hoocs implements a broad range of model-agnostic attributions.
- *PredDiff*  [[1]]([1]) [[2]]([2])
- Shapley values [[3]]([3])
- KernelSHAP [[4]]([4])
Recently, there has been increasing interest in more in-depth analysis of models. To meet this needs, the analysis of 
feature interactions is inevitable. Therefore, this package allows to calculate arbitrary higher-order explanations.

Importantly, it is easily extendable to other methods, which rely on marginalizing features in input space. 
This package puts a special focus on the on-manifoldness of attributions methods. To this end, we enable simple 
incorporation of new conditional distributions (*imputers*) for any kind of data modality. 

## Installation
```
pip install hoocs
```

## Implement new imputers
To add a new imputer to incorporate a suitable conditional distribution for the current data modality, 
the user is requested to inherent from the abstract base `Imputer` class 
in `hoocs.imputers.abstract_imputers.py`.
This class performs basic type checking and ensures a consistent interface. 

## References

###### [1] [Explaining classifications for individual instances.](https://ieeexplore.ieee.org/abstract/document/4407709)
###### [2] [PredDiff: Explanations and interactions from conditional expectations](https://www.sciencedirect.com/science/article/pii/S000437022200114X)
###### [3] [An efficient explanation of individual classifications using game theory](https://www.jmlr.org/papers/volume11/strumbelj10a/strumbelj10a.pdf?ref=https://githubhelp.com)
###### [4] [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)
