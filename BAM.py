import numpy as np
import random
mem_vectors = [
    [1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 0]
]
q = len(mem_vectors) # number of vectors = 3
n = len(mem_vectors[0]) # dimensionality of vectors = 6
bip_mem_vecs = 2*np.array(mem_vectors) - 1
#Initialize and compute the weight matrix
zd_wt_mat = np.zeros((n,n))
for i in range(q):
  zd_wt_mat += np.outer(bip_mem_vecs[i, :], bip_mem_vecs[i, :])
 
zd_wt_mat -= q * np.eye(n) #Zero diagonal
probe = np.array([1, 0, 0, 0, 1, 1])
print(f'The input vector is: {probe}')
signal_vector = 2*probe-1
flag = 0 #Initialize flag
 
while flag != n:
    permindex = np.random.permutation(n)  # Randomize order
    old_signal_vector = np.copy(signal_vector)
    # Update all neurons once per epoch
    for j in range(n):
        act_vec = np.dot(signal_vector, zd_wt_mat)
        if act_vec[permindex[j]] > 0:
            signal_vector[permindex[j]] = 1
        elif act_vec[permindex[j]] < 0:
            signal_vector[permindex[j]] = -1
    flag = np.dot(signal_vector, old_signal_vector)
 
print(f'The recalled vector is: {0.5 * (signal_vector + 1)}')