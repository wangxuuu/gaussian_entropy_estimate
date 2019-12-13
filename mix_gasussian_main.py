#%%

import  numpy as np
import torch
import matplotlib.pyplot as plt
import os
from mix_gaussian import Mix_Gaussian
import NNmodel

# use GPU if available
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)

# initialize random seed
np.random.seed(0)
torch.manual_seed(0)
rho1 = 0.9
rho2 = -0.9

for dim in range(2, 10):
    # generate Gaussian distributed samples

    # setting parameters
    sample_size = 1000
    # dim = 6
    mean = np.zeros(dim)
    cov1 = np.identity(dim)*(1-rho1) + np.ones([dim, dim])*rho1
    cov2 = np.identity(dim)*(1-rho2) + np.ones([dim, dim])*rho2

    # Generate samples from Gaussian distribution
    g = Mix_Gaussian(dim, mean, cov1, cov2, sample_size, mix=0.5)
    X = g.generate_data
    groundtruth = g.groundtruth_entropy
    ref_entropy = NNmodel._uniform_sample_entropy(X)

    # -------------------------- Training ----------------------------- #
    # Using Neural Network to estimate the entropy of the generated Gaussian distribution

    batch_size = 100
    lr = 1e-5

    #####################
    ref_batch_factor = 20 #
    ####################

    NN = NNmodel.NNE(torch.Tensor(X), batch_size=batch_size, ref_batch_factor=ref_batch_factor, lr=lr)

    num_iteration = 100000

    X_entropy = []

    for i in range(num_iteration):
        NN.step()
        X_entropy.append(ref_entropy - NN.forward())
    plt.figure()
    plt.plot(X_entropy, label='entropy')
    plt.axhline(groundtruth, label='ground truth', linestyle='--', color='red')
    plt.xlabel('Iteration')
    plt.title('dim=%d' %dim)
    plt.legend()
    plt.show()
    plt.savefig('results/mixgaussian_dim%d.png' %dim)





