import pickle
import os

import numpy as np

from datasets import gen


def epanechnikov_kernel(u):
    return np.where(np.abs(u) <= 1, 3 * (1 - u**2) / 4, 0)

def clipping_ips(propensity_scores, clip_param):
    '''IPSをclipする'''
    return np.clip(propensity_scores, clip_param, 1)


REPLICATION_NUM = 50
def _estimate_value_on_n(estimate_method, b_policy, t_policy, n, r=REPLICATION_NUM):
    '''nが指定されたときの推定値
    '''
    ope_estimands = []
    for seed in range(r):
        X, T, Y = gen(n, seed=seed + 10)
        piX = t_policy(X)  # target policy
        Q = b_policy(T, X)  # behavior policy
        
        estimated_value = estimate_method(n=n, X=X, Y=Y, T=T, Q=Q, piX=piX)
        ope_estimands.append(estimated_value)
    ope_estimands = np.array(ope_estimands)
    return ope_estimands