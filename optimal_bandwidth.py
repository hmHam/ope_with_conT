import numpy as np
import pandas as pd
from scipy import integrate
from tqdm import tqdm
# epanechnikov kernelの2次モーメントとRoughness
R_K = 3/20
KAPPA2_K = 4/5

def _get_numerator(gen, target_policy, behavior_policy):
    # E[Y|\pi(X), X]を近似 10万回のreplicationで期待値を近似
    n = 10
    samples = np.zeros(3).reshape(1, -1)
    print('aprroximating E[Y\]...')
    for i in tqdm(range(pow(10, 5))):
        X, _, Y = gen(n, seed=i)
        piX = target_policy(X)
        samples_i = np.hstack([X, Y, piX])
        samples = np.vstack([samples, samples_i])
    samples = samples[1:, :]

    df = pd.DataFrame(samples, columns=['X', 'Y', 'piX'])
    E_Y2 = df.groupby(['X', 'piX'])['Y'].apply(lambda x: (x**2).mean()).to_numpy()
    print('done')
    f_T_on_X = behavior_policy(piX, X)

    numerator = R_K * np.mean(E_Y2 / f_T_on_X)
    return numerator

# FIXME: Y, T, Xは生成しなおす。
def _get_denominator():
    # f_{Y|T, X}を求める。
    # 分子 pdf
    YTX = np.hstack([Y, T, X])
    joint_density_YTX = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(YTX)
    # 分母 pdf
    TX = np.hstack([T, X])
    density_TX = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(TX)

    # Y|T, X pdf
    YpiXX = np.hstack([Y, piX, X])
    piXX = np.hstack([piX, X])
    conditional_density_Y_on_piXX = joint_density_YTX.score_samples(YpiXX) / density_TX.score_samples(piXX)

    # 数値微分
    dt = 1e-7 # 適当
    partial_f_partial_T = np.gradient(conditional_density_Y_on_piXX, dt)
    partial_f_partial_T2 = np.gradient(partial_f_partial_T, dt)

    # 数値積分
    integral_y = integrate.simps(Y[:, 0] * partial_f_partial_T2 * KAPPA2_K / 2, Y[:, 0])

    denominator = 4 * integral_y**2 * n
    return denominator

def get_optimal_h(gen, target_policy, behavior_policy):
    numerator = _get_numerator(gen, target_policy, behavior_policy)
    print('completed to calculate numerator')
    denominator = _get_denominator()
    print('completed to calculate denominator')
    optimal_bandwidth = np.power(numerator / denominator, 1/5)
    print('completed to calculate optimal bandwidth')
    return optimal_bandwidth
