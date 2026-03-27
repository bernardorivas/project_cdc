import numpy as np
from MORALS.systems.system import BaseSystem


class SwitchedNetwork(BaseSystem):
    """System wrapper for state-dependent switched oscillator networks.

    State is a flat vector in R^{2N} (N nodes x 2D local dynamics).
    Transform is identity since the state space is Euclidean.
    Bounds default to [-3, 3]^{2N}; actual normalization is computed
    from data by DynamicsDataset.
    """

    def __init__(self, dims=6, **kwargs):
        super().__init__(**kwargs)
        self.name = "switched_network"
        self.state_bounds = np.array([[-3.0, 3.0]] * dims)
