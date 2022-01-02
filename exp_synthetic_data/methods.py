import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from utils import epanechnikov_kernel, clipping_ips

def true_outcomes(n, X, Y, T, Q, piX):
    np.random.seed(10)
    return 2 * np.power(X - piX, 3/2) + 0.2 * np.random.randn(n).reshape(-1, 1)


def continuous_ips(n, X, Y, T, Q, piX, n0_optimal_bandwidth=0.23):
    # COPE
    h = n0_optimal_bandwidth * np.power(10/n, 1/5)

    # clipping propensity score
    Q_clipped = clipping_ips(Q, clip_param=0.1)

    estimated_value = np.mean(epanechnikov_kernel((piX - T) / h) * Y / Q_clipped) / h
    return estimated_value


def discretized_ips(n, X, Y, T, Q, piX):
    clipped_Q = clipping_ips(Q, clip_param=0.1)
    
    bins = np.linspace(T.min(), T.max(), 11)
    discretized_T = np.argmin(np.abs(T - bins), axis=1)
    discretized_piX = np.argmin(np.abs(piX - bins), axis=1)
    rej_mask = (discretized_piX == discretized_T)  # sample REJection MASK
    estimated_value = np.mean(Y[rej_mask] / clipped_Q[rej_mask])
    return estimated_value


def direct_method_poly(n, X, Y, T, Q, piX):
    '''
    多項式回帰モデルを用いたDirect Method
    '''
    pre = PolynomialFeatures(degree=3)
    reg = LinearRegression()
    behavior_features = pre.fit_transform(np.hstack([X, T]))
    reg.fit(behavior_features, Y[:, 0])
    
    evaluation_features = pre.fit_transform(np.hstack([X, piX]))
    estimated_reward = reg.predict(evaluation_features)
    return np.mean(estimated_reward)


def direct_method_rf(n, X, Y, T, Q, piX):
    '''
    Random Forestを用いたDirect Method
    '''
    reg = RandomForestRegressor(n_estimators=10)
    behavior_features = np.hstack([X, T])
    reg.fit(behavior_features, Y[:, 0])
    evaluation_features = np.hstack([X, piX])
    estimated_reward = reg.predict(evaluation_features)
    return estimated_reward