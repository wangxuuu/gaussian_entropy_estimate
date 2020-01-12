# crosstrain : still use uniform distribution to retrain the model; loss = self.forward()
# crosstrain2: use product of marginal distribution to retrain the model
# crosstrain3: use product of marginal distribution to retrain P(XY)

import numpy as np
import torch
import matplotlib.pyplot as plt
from data.mix_gaussian import MixedGaussian
from models import combine_train
import os

os.makedirs('./results/crosstrain4/', exist_ok=True)

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)

for d in range(1,8):
    # d = 1  # d is the dimension of X and Y. The dimension of joint mix-gaussian distribution is 2*d
    rho = 0.9
    sample_size = 400

    X = np.zeros((sample_size, d))
    Y = np.zeros((sample_size, d))

    mg = MixedGaussian(sample_size=sample_size, rho1=rho, rho2=-rho)

    _, _, _, mi = mg.ground_truth

    # ground truth of entropy of mixed gaussian distribution (X,Y)
    MI = mi * d

    # (X,Y) is a mixed gaussian distribution; but X and Y are not.
    for i in range(d):
        data = mg.data
        X[:, i] = data[:, 0]
        Y[:, i] = data[:, 1]

    data = np.append(X, Y, axis=1)

    # -------------------------- Training ----------------------------- #
    # Using Neural Network to estimate the entropy of the generated Gaussian distribution
    batch_size = 100
    lr = 1e-4

    #####################
    ref_batch_factor = 10  #
    ####################

    NN = combine_train.CrossTrain(torch.Tensor(X), torch.Tensor(Y), batch_size=batch_size, ref_batch_factor=ref_batch_factor, lr=lr)

    num_iteration = 100000
    MutualInfo = []

    for i in range(num_iteration):
        NN.step()
        mi_estimate = NN.forward().cpu().item()
        MutualInfo.append(mi_estimate)

        if i % 100 ==0:
            print("Iteration:{0}({1:.2%}) MutualInfo:{2} GroundTruth:{3} dim:{4}".format(i, i/num_iteration, MutualInfo[-1], MI, d))

    plt.figure()
    plt.plot(MutualInfo, label='Mutual Information')
    plt.axhline(MI, label='ground truth', linestyle='--', color='red')
    plt.xlabel('Iteration')
    plt.title('XY dim=%d' % (2 * d))
    plt.legend()
    plt.savefig("./results/crosstrain4/dim=%d.png" % (2 * d))
    plt.show()


    ma_rate = 0.01                 # rate of moving average
    MI_list = MutualInfo.copy()    # see also the estimate() member function of MINE
    for i in range(1,len(MI_list)):
        MI_list[i] = (1-ma_rate) * MI_list[i-1] + ma_rate * MI_list[i]

    plt.figure()
    plt.plot(MI_list, label='Mutual Information')
    plt.axhline(MI, label='ground truth', linestyle='--', color='red')
    plt.xlabel('Iteration')
    plt.title('XY dim=%d' % (2 * d))
    plt.legend()
    plt.savefig("./results/crosstrain4/moveavg_dim=%d.png" % (2 * d))
    plt.show()







