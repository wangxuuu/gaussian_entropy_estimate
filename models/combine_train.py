"""
This is to train neural network to estimate Mutual Information by combining MINEE and MINE.
MINE has problem of undershooting;
MINEE has problem of overshooting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import itertools

def _resample(data, batch_size, replace=False):
    # Resample the given data sample.
    index = np.random.choice(
        range(data.shape[0]), size=batch_size, replace=replace)
    batch = data[index]
    return batch


def _uniform_sample(data, batch_size):
    # Sample the reference uniform distribution
    data_min = data.min(dim=0)[0]
    data_max = data.max(dim=0)[0]
    return (data_max - data_min) * torch.rand((batch_size, data_min.shape[0])) + data_min


def _div(net, data, ref):
    """
    Calculate the divergence estimate using a neural network
    Args:
        net: Neural Network
        data: samples form P(Z)
        ref:  samples from P(Z') reference distribution
    Returns:
        E[f(Z)] - ln{E[e^f(Z')]}
    """
    mean_f = net(data).mean()
    log_mean_ef_ref = torch.logsumexp(net(ref), 0) - np.log(ref.shape[0])
    return mean_f - log_mean_ef_ref

class CrossTrain():

    class Net(nn.Module):
        # Inner class that defines the neural network architecture
        def __init__(self, input_size=2, hidden_size=100, sigma=0.02):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, 1)
            nn.init.normal_(self.fc1.weight, std=sigma)
            nn.init.constant_(self.fc1.bias, 0)
            nn.init.normal_(self.fc2.weight, std=sigma)
            nn.init.constant_(self.fc2.bias, 0)
            nn.init.normal_(self.fc3.weight, std=sigma)
            nn.init.constant_(self.fc3.bias, 0)

        def forward(self, input):
            output = F.elu(self.fc1(input))
            output = F.elu(self.fc2(output))
            output = self.fc3(output)
            return output

    def __init__(self, X, Y, batch_size=32, ref_batch_factor=1, lr=1e-3, hidden_size=100):
        self.lr = lr
        self.batch_size = batch_size
        self.ref_batch_factor = ref_batch_factor
        self.X = X
        self.Y = Y
        self.XY = torch.cat((self.X, self.Y), dim=1)

        self.X_ref = _uniform_sample(X, batch_size=int(
            self.ref_batch_factor * X.shape[0]))
        self.Y_ref = _uniform_sample(Y, batch_size=int(
            self.ref_batch_factor * Y.shape[0]))

        self.XY_net = CrossTrain.Net(
            input_size=X.shape[1]+Y.shape[1], hidden_size=100)
        self.X_net = CrossTrain.Net(input_size=X.shape[1], hidden_size=100)
        self.Y_net = CrossTrain.Net(input_size=Y.shape[1], hidden_size=100)

        self.XY_optimizer = optim.Adam(self.XY_net.parameters(), lr=lr)
        self.X_optimizer = optim.Adam(self.X_net.parameters(), lr=lr)
        self.Y_optimizer = optim.Adam(self.Y_net.parameters(), lr=lr)

        self.total_optimizer = optim.Adam(
            itertools.chain(self.XY_net.parameters(), self.X_net.parameters(), self.Y_net.parameters()), lr=0.1*lr
        )

    def step(self):

        self.XY_optimizer.zero_grad()
        self.X_optimizer.zero_grad()
        self.Y_optimizer.zero_grad()

        batch_XY = _resample(self.XY, batch_size=self.batch_size)
        batch_X = _resample(self.X, batch_size=self.batch_size)
        batch_Y = _resample(self.Y, batch_size=self.batch_size)

        batch_XY_ref_mine = torch.cat((batch_X, batch_Y), dim=1)

        batch_X_ref = _uniform_sample(self.X, batch_size=int(
            self.ref_batch_factor * self.batch_size))
        batch_Y_ref = _uniform_sample(self.Y, batch_size=int(
            self.ref_batch_factor * self.batch_size))
        batch_XY_ref = torch.cat((batch_X_ref, batch_Y_ref), dim=1)

        batch_loss_XY = -_div(self.XY_net, batch_XY, batch_XY_ref)
        batch_loss_XY.backward()
        self.XY_optimizer.step()

        batch_loss_X = -_div(self.X_net, batch_X, batch_X_ref)
        batch_loss_X.backward()
        self.X_optimizer.step()

        batch_loss_Y = -_div(self.Y_net, batch_Y, batch_Y_ref)
        batch_loss_Y.backward()
        self.Y_optimizer.step()

        self.total_optimizer.zero_grad()
        # mine_loss = self.forward()

        # dXY = _div(self.XY_net, batch_XY, batch_XY_ref_mine)
        # dX = _div(self.X_net, self.X, batch_X)
        # dY = _div(self.Y_net, self.Y, batch_Y)
        # mine_loss = -(dXY - dX - dY)

        log_mean_ef_ref = - np.log(batch_XY.shape[0]) + \
                            torch.logsumexp((self.XY_net(batch_XY_ref_mine)-self.X_net(batch_X)-self.Y_net(batch_Y)), 0)
        mine_loss = -(self.XY_net(batch_XY).mean() -
                     self.X_net(batch_X).mean() -
                     self.Y_net(batch_Y).mean() - log_mean_ef_ref)

        mine_loss.backward()
        self.total_optimizer.step()


    def forward(self, X=None, Y=None):
        """
        Outputs:
            dXY: divergence of sample joint distribution of (X,Y)
                to the uniform reference
            dX: divergence of sample marginal distribution of X
                to the uniform reference
            dY: divergence of sample marginal distribution of Y
                to the uniform reference
            return dXY - dX - dY
        Arguments:
            X (tensor, optional): samples of X.
            Y (tensor, optional): samples of Y.
        By default, X and Y for training is used.
        The arguments are useful for testing/validation with a separate data set.
        """
        XY = None
        if X is None or Y is None:
            XY, X, Y = self.XY, self.X, self.Y
        else:
            XY = torch.cat((X, Y), dim=1)
        X_ref = _uniform_sample(X, batch_size=int(
            self.ref_batch_factor * X.shape[0]))
        Y_ref = _uniform_sample(Y, batch_size=int(
            self.ref_batch_factor * Y.shape[0]))
        XY_ref = torch.cat((X_ref, Y_ref), dim=1)

        # dXY = _div(self.XY_net, XY, XY_ref).cpu().item()
        # dX = _div(self.X_net, X, X_ref).cpu().item()
        # dY = _div(self.Y_net, Y, Y_ref).cpu().item()

        dXY = _div(self.XY_net, XY, XY_ref)
        dX = _div(self.X_net, X, X_ref)
        dY = _div(self.Y_net, Y, Y_ref)

        return dXY - dX - dY

