import numpy as np


def average_fusion(expert_advice):
    return np.sum(expert_advice, axis=0) / 3