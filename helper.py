import numpy as np

in_dim = 100
tr_size = 1e5
batch_size = 200
step = 1e-3

W = np.random.normal(1,1,(in_dim,in_dim))
W = 5*W/np.linalg.norm(W)
np.linalg.norm(W)

x = np.random.normal(1,1,in_dim)
x = x/np.linalg.norm(x)
np.linalg.norm(x)h