"""
This is used to cross-training the neural network to estimate entropy by switching the reference distribution:
uniform distribution and marginal product distribution. The aim is to take the advantage of
faster convergence speed of using uniform distribution and avoid overshooting at the same time.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

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

def _uniform_sample_entropy(data):
    """
    :param data: reference uniform distribution points
    :return: entropy of uniform distribution
    """
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    dist = (data_max - data_min)
    return np.sum(np.log(dist))

def _div(net, data, ref):
    # Calculate the divergence estimate using a neural network
    mean_f = net(data).mean()
    log_mean_ef_ref = torch.logsumexp(net(ref), 0) - np.log(ref.shape[0])
    return mean_f - log_mean_ef_ref

class cross_train():
    """
    Class for cross-training

    The entropy is estimated using neural estimation of the divergence
    between p(z) and a custom reference distribution p(z'), where p(z')
    can be the the product of marginal distribution or uniform distribution.

    Arguments:
    X (tensor): samples of X
        dim 0: different samples
        dim 1: different components
    Y (tensor): samples of Y
        dim 0: different samples
        dim 1: different components
    ma_rate (float, optional): rate of moving average in the gradient estimate
    ma_ef (float, optional): initial value used in the moving average
    lr (float, optional): learning rate
    hidden_size (int, optional): size of the hidden layers
    """
    class Net(nn.Module):
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

    def __init__(self, X, Y, batch_size=32, ref_batch_factor=1, lr=1e-3, ma_rate=0.1, hidden_size=100, ma_ef=1):
        self.lr = lr
        self.batch_size = batch_size
        self.ma_rate = ma_rate
        self.ma_ef = ma_ef
        self.ref_batch_factor = ref_batch_factor

        self.X = X
        self.Y = Y
        self.XY = torch.cat((self.X, self.Y), dim=1)

        self.X_ref_MINE = _resample(self.X, batch_size=self.X.shape[0])
        self.Y_ref_MINE = _resample(self.Y, batch_size=self.Y.shape[0])

        self.X_ref_MINEE = _uniform_sample(X, batch_size=int(
            self.ref_batch_factor * X.shape[0]))
        self.Y_ref_MINEE = _uniform_sample(Y, batch_size=int(
            self.ref_batch_factor * Y.shape[0]))

        self.XY_net = cross_train.Net(
            input_size=X.shape[1] + Y.shape[1], hidden_size=300)
        self.XY_optimizer = optim.Adam(self.XY_net.parameters(), lr=lr)

    def step_MINEE(self, iter=1):
        """
        Train the neural networks for one or more steps.
        :param iter:  (int, optional) number of steps to train.
        """
        for i in range(iter):
            self.XY_optimizer.zero_grad()
            batch_XY = _resample(self.XY, batch_size=self.batch_size)
            batch_X_ref = _uniform_sample(self.X, batch_size=int(
                self.ref_batch_factor * self.batch_size))
            batch_Y_ref = _uniform_sample(self.Y, batch_size=int(
                self.ref_batch_factor * self.batch_size))
            batch_XY_ref = torch.cat((batch_X_ref, batch_Y_ref), dim=1)

            batch_loss_XY = -_div(self.XY_net, batch_XY, batch_XY_ref)
            batch_loss_XY.backward()
            self.XY_optimizer.step()

    def step_MINE(self, iter=1):
        r"""Train the neural networks for one or more steps.

        Argument:
        iter (int, optional): number of steps to train.
        """
        for i in range(iter):
            self.XY_optimizer.zero_grad()
            batch_XY = _resample(self.XY, batch_size=self.batch_size)
            batch_XY_ref = torch.cat((_resample(self.X, batch_size=self.batch_size),
                                      _resample(self.Y, batch_size=self.batch_size)), dim=1)
            # define the loss function with moving average in the gradient estimate
            mean_fXY = self.XY_net(batch_XY).mean()
            mean_efXY_ref = torch.exp(self.XY_net(batch_XY_ref)).mean()
            self.ma_ef = (1-self.ma_rate)*self.ma_ef + \
                self.ma_rate*mean_efXY_ref
            batch_loss_XY = - mean_fXY + \
                (1 / self.ma_ef.mean()).detach() * mean_efXY_ref
            batch_loss_XY.backward()
            self.XY_optimizer.step()

    def forward_MINE(self, X=None, Y=None):
        """
        Evaluate the neural network on (X,Y). - Using product of marginal distribution
        :param X: (tensor, optional) samples of X.
        :param Y: (tensor, optional) samples of Y.
        :return: entropy estimation
        """
        XY = None
        if X is None or Y is None:
            XY, X, Y = self.XY, self.X, self.Y
        else:
            XY = torch.cat((X, Y), dim=1)
        X_ref = _resample(X, batch_size=X.shape[0])
        Y_ref = _resample(Y, batch_size=Y.shape[0])
        XY_ref = torch.cat((X_ref, Y_ref), dim=1)
        return _div(self.XY_net, XY, XY_ref).cpu().item()

    def forward_MINEE(self, X=None, Y=None):
        """
        Evaluate the neural network on (X,Y). - Using uniform distribution
        :param X: (tensor, optional) samples of X.
        :param Y: (tensor, optional) samples of Y.
        :return: entropy estimation
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
        dXY = _div(self.XY_net, XY, XY_ref).cpu().item()
        return dXY

    def state_dict(self):
        r"""Return a dictionary storing the state of the estimator.
        """
        return {
            'XY_net': self.XY_net.state_dict(),
            'XY_optimizer': self.XY_optimizer.state_dict(),
            'X': self.X,
            'Y': self.Y,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'ma_rate': self.ma_rate,
            'ma_ef': self.ma_ef,
            'ref_batch_factor': self.ref_batch_factor
        }

    def load_state_dict(self, state_dict):
        r"""Load the dictionary of state state_dict.
        """
        self.XY_net.load_state_dict(state_dict['XY_net'])
        self.XY_optimizer.load_state_dict(state_dict['XY_optimizer'])
        self.X = state_dict['X']
        self.Y = state_dict['Y']
        self.ma_rate = state_dict['ma_rate']
        self.ma_ef = state_dict['ma_ef']
        if 'lr' in state_dict:
            self.lr = state_dict['lr']
        if 'batch_size' in state_dict:
            self.batch_size = state_dict['batch_size']
        if 'ref_batch_factor' in state_dict:
            self.ref_batch_factor = state_dict['ref_batch_factor']