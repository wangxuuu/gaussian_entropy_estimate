import numpy as np
import torch
import matplotlib.pyplot as plt
from data.mix_gaussian import MixedGaussian
from models import crosstraining
import os

os.makedirs('./results/cross_training', exist_ok=True)

for d in range(1, 4):
    # d is the dimension of X and Y. The dimension of joint mix-gaussian distribution is 2*d
    rho = 0.9
    sample_size = 400

    X = np.zeros((sample_size, d))
    Y = np.zeros((sample_size, d))

    mg = MixedGaussian(sample_size=sample_size, rho1=rho, rho2=-rho)

    hx, hy, hxy, mi = mg.ground_truth
    # ground truth of entropy of mixed gaussian distribution (X,Y)
    h_xy = hxy * d
    hx = hx * d
    hy = hy * d
    # (X,Y) is a mixed gaussian distribution; but X and Y are not.
    for i in range(d):
        data = mg.data
        X[:, i] = data[:, 0]
        Y[:, i] = data[:, 1]

    import scipy.stats as st
    pdf = st.multivariate_normal(
        mean=np.zeros(d),
        cov=np.identity(d))
    density_x = pdf.pdf(X)  # p(x)
    density_y = pdf.pdf(Y)  # p(y)
    density_xy = density_x * density_y
    # The cross entropy of the reference distribution, i.e. product of marginal distribution
    ref_entropy_XY_MINE = -np.log(density_xy).mean()

    data = np.append(X, Y, axis=1)

    # TODO cross entropy of reference distribution
    ref_entropy_XY_MINEE = crosstraining._uniform_sample_entropy(data)
    # -------------------------- Training ----------------------------- #
    # Using Neural Network to estimate the entropy of the generated Gaussian distribution
    batch_size = 100
    lr = 1e-4
    ref_batch_factor = 10

    NN = crosstraining.cross_train(torch.Tensor(X), torch.Tensor(Y), batch_size=batch_size,
                                   ref_batch_factor=ref_batch_factor, lr=lr, ma_rate=0.1, hidden_size=100, ma_ef=1)

    num_iteration = 100000
    entropy_XY = []
    dXY_list = []

    k = np.floor(num_iteration/5)

    for i in range(num_iteration):
        if  i < k:
            NN.step_MINEE()
            dXY = NN.forward_MINEE()
            dXY_list.append(dXY)
            entropy_XY.append(ref_entropy_XY_MINEE - dXY)
        else:
            NN.step_MINE()
            dXY = NN.forward_MINE()
            dXY_list.append(dXY)
            entropy_XY.append(ref_entropy_XY_MINE - dXY)

        if i % 100 ==0:
            print("iteration:{0} entropy:{1} Ground truth:{2}".format(i, entropy_XY[-1], h_xy))

    ma_rate = 0.01  # rate of moving average
    entropy_list = entropy_XY.copy()  # see also the estimate() member function of MINE
    for i in range(1, len(entropy_list)):
        entropy_list[i] = (1 - ma_rate) * entropy_list[i - 1] + ma_rate * entropy_list[i]

    plt.figure()
    plt.plot(entropy_list, label='XY entropy')
    plt.axhline(h_xy, label='ground truth', linestyle='--', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('entropy')
    plt.title('XY dim=%d' % (2 * d))
    plt.legend()
    plt.savefig("./results/cross_training/dim=%d learnrate=%f.png" % (2 * d, lr))
    plt.show()
