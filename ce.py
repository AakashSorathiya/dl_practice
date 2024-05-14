import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    ce = 0
    for yi, pi in zip(Y, P):
        if yi==1:
            ce += np.log(pi)
        else:
            ce += np.log(1-pi)
    return -ce