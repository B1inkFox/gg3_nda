from HMM_models import *
from inference import *


def func(model, spike_data):

    #Compute expected states first with fwd-bwd algorithm
    ll = poisson_logpdf(spike_data, model.lambdas)
    init = model.initial_distribution
    trans = model.transition_matrix

    pd = np.empty(model.T)
    pd[0] = np.matmul(ll[0], init)

    for t in range(1, model.T):
        sum = 0
        for i in model.states: #state at time t-1
            for j in model.states: # state at time t
                sum += trans[i][j]*ll[t-1][i]*ll[t][j]
        pd[t] = sum
    return 

import numpy as np

# Example 2D probability distribution
prob_dist = np.array([
    [0.1, 0.2, 0.1],
    [0.3, 0.2, 0.1]
])

# Step 1: Normalize the distribution (if not already normalized)
prob_dist /= prob_dist.sum()

# Step 2: Flatten the array
flat_prob_dist = prob_dist.ravel()

# Step 3: Use numpy.random.choice to pick an index
flat_index = np.random.choice(len(flat_prob_dist), p=flat_prob_dist)

# Step 4: Convert the 1D index back to 2D
rows, cols = prob_dist.shape
row, col = divmod(flat_index, cols)

print(f"Selected index: ({row}, {col})")
print(prob_dist[row][col])