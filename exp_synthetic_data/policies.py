from scipy import stats

def behavior_policy1(t, x):
    # T|X ~ N(x + 0.1, 0.5)
    return stats.norm.pdf(t, loc=x + 0.1, scale=0.5)

def behavior_policy2(t, x):
    # T ~ Uniform[-0.5, 1.3]
    return stats.uniform.pdf(t, loc=-0.5, scale=1.8)