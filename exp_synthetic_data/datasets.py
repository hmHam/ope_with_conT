import numpy as np

# Section5.1のOPE用データ生成
# (x, t, y)
def gen(n, seed=0, is_treat_randomly=False):
    '''
    n: データ数
    '''
    # NOTE: 縦ベクトルで返す
    np.random.seed(seed)
    x = np.random.uniform(low=0, high=1, size=(n, 1))
    # optimal bandwidth in np.linspace(0.23, 0.27, 10)
    if is_treat_randomly:
        t = np.random.uniform(low=-0.5, high=1.3, size=(n, 1))
    else:
        t = x + 0.1 + 0.5 * np.random.randn(n, 1)
    epsilon = np.random.randn(n, 1)
    y = 2 * np.power(np.abs(x - t), 1.5) + 0.2 * epsilon
    return (x, t, y)


def gen_high_dimension(n, seed=0):
    '''Xが10次元'''
    pass