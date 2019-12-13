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


class NNE():

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

    def __init__(self, X, batch_size=32, ref_batch_factor=1, lr=1e-3, hidden_size=100):
        self.lr = lr
        self.batch_size = batch_size
        self.ref_batch_factor = ref_batch_factor
        self.X = X

        self.X_ref = _uniform_sample(X, batch_size=int(
            self.ref_batch_factor * X.shape[0]))
        self.X_net = NNE.Net(input_size=X.shape[1], hidden_size=100)
        self.X_optimizer = optim.Adam(self.X_net.parameters(), lr=lr)

    def step(self, iter=1):
        r"""Train the neural networks for one or more steps.

        Argument:
        iter (int, optional): number of steps to train.
        """
        for i in range(iter):
            self.X_optimizer.zero_grad()
            batch_X = _resample(self.X, batch_size=self.batch_size)
            batch_X_ref = _uniform_sample(self.X, batch_size=int(
                self.ref_batch_factor * self.batch_size))

            batch_loss_X = -_div(self.X_net, batch_X, batch_X_ref)

            batch_loss_X.backward()
            self.X_optimizer.step()


    def forward(self, X=None):
        r"""Evaluate the neural networks to return an array of 3 divergences estimates 
        (dX).

        Outputs:
            dX: divergence of sample marginal distribution of X 
                to the uniform reference
        Arguments:
            X (tensor, optional): samples of X.
        By default, X for training is used.
        The arguments are useful for testing/validation with a separate data set.
        """
        X_ref = _uniform_sample(self.X, batch_size=int(
            self.ref_batch_factor * self.X.shape[0]))
        dX = _div(self.X_net, self.X, X_ref).cpu().item()
        return dX



