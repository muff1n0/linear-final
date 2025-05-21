import numpy as np
from sklearn.decomposition import NMF

def factorize(V, n):
    model = NMF(n_components=n, init="random", random_state=0)
    W = model.fit_transform(V)
    H = model.components_
    return W, H

