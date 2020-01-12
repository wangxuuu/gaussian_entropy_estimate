# crosstrain : still use uniform distribution to retrain the model; loss = self.forward()
# crosstrain2: use product of marginal distribution to retrain the model
# crosstrain3: use product of marginal distribution to retrain P(XY)

import numpy as np
import torch
import matplotlib.pyplot as plt
from data.mix_gaussian import MixedGaussian
from models import combine_train2
import os
from tqdm import tqdm

np.random.seed(0)
torch.manual_seed(0)

os.makedirs('./results/crosstrain-test/', exist_ok=True)

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)

for d in range(1,8):
#for d in range(2,3):
    # d = 1  # d is the dimension of X and Y. The dimension of joint mix-gaussian distribution is 2*d
    rho = 0.9
    sample_size = 400

    X = np.zeros((sample_size, d))
    Y = np.zeros((sample_size, d))

    mg = MixedGaussian(sample_size=sample_size, rho1=rho, rho2=-rho, mix=.5)

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
    ref_batch_factor = 1  #
    ####################

    NN = combine_train2.CrossTrain(torch.Tensor(X), torch.Tensor(Y), batch_size=batch_size, ref_batch_factor=ref_batch_factor, lr=lr)

    num_iteration = 100000
    num_iterationA = int(num_iteration/10)
    num_iterationB = num_iteration-num_iterationA
    MutualInfo = []
    dXY = []
    dX = []
    dY = []

    #for i in range(num_iteration):
    for i in tqdm(range(num_iterationA)):
        NN.stepA()
        mi_estimate = NN.forwardA().cpu().item()
        MutualInfo.append(mi_estimate)
#        mi_estimate = NN.forward()[0].cpu().item()
#        MutualInfo.append(mi_estimate)
#        dXY.append(NN.forward()[1].cpu().item())
#        dX.append(NN.forward()[2].cpu().item())
#        dY.append(NN.forward()[3].cpu().item())

        if i % 100 ==0:
            print("Iteration:{0}({1:.2%}) MutualInfo:{2} GroundTruth:{3} dim:{4}".format(i, i/num_iteration, MutualInfo[-1], MI, d))

    for i in tqdm(range(num_iterationB)):
        NN.stepB()
        mi_estimate = NN.forwardB().cpu().item()
        MutualInfo.append(mi_estimate)
        if i % 100 ==0:
            print("Iteration:{0}({1:.2%}) MutualInfo:{2} GroundTruth:{3} dim:{4}".format(i, i/num_iteration, MutualInfo[-1], MI, d))

    plt.figure()
    plt.plot(MutualInfo, label='Mutual Information')
    plt.axhline(MI, label='ground truth', linestyle='--', color='red')
    plt.xlabel('Iteration')
    plt.title('XY dim=%d' % (2 * d))
    plt.legend()
    plt.savefig("./results/crosstrain-test/dim=%d.png" % (2 * d))
    plt.show()


    ma_rate = 0.01                 # rate of moving average
    MI_list = MutualInfo.copy()    # see also the estimate() member function of MINE
    for i in range(1,len(MI_list)):
        MI_list[i] = (1-ma_rate) * MI_list[i-1] + ma_rate * MI_list[i]
#        dXY[i] = (1-ma_rate) * dXY[i-1] + ma_rate * dXY[i]
#        dX[i] = (1-ma_rate) * dX[i-1] + ma_rate * dX[i]
#        dY[i] = (1-ma_rate) * dY[i-1] + ma_rate * dY[i]

    plt.figure()
    plt.plot(MI_list, label='Mutual Information')
    plt.axhline(MI, label='ground truth', linestyle='--', color='red')
    plt.xlabel('Iteration')
    plt.title('XY dim=%d' % (2 * d))
    plt.legend()
    plt.savefig("./results/crosstrain-test/moveavg_dim=%d.png" % (2 * d))
    plt.show()


#    plt.figure()
#    plt.plot(dXY, label='dXY')
#    plt.plot(dX, label='dX')
#    plt.plot(dY, label='dY')
#    plt.xlabel('Iteration')
#    plt.title('XY dim=%d' % (2 * d))
#    plt.legend()
#    plt.savefig("./results/crosstrain-test/KLmoveavg_dim=%d.png" % (2 * d))
#    plt.show()







