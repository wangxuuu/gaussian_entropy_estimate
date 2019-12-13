import numpy as np
import math


class Gaussian():
    def __init__(self, dim=2, mean=np.zeros(2), cov=np.identity(2), sample_size=400):
        super().__init__()
        self.dim = dim
        self.samplesize = sample_size
        self.mean = mean
        self.cov = cov

    @property
    def generate_data(self):
        """
        Generate gaussian distribution samples
        :return: sample points
        """
        return np.random.multivariate_normal(
            mean=self.mean,
            cov=self.cov,
            size=self.samplesize)

    @property
    def groundtruth_entropy(self):
        """
        entropy for high-dimension gaussian : http://www.limoncc.com/%E6%A6%82%E7%8E%87%E8%AE%BA/2017-01-10-%E5%A4%9A%E5%85%83%E9%AB%98%E6%96%AF%E5%88%86%E5%B8%83%E7%9A%84%E7%86%B5/
        :return: entropy of the high dimension gaussian distribution
        """
        H = self.dim/2 * (np.log(2*math.pi) + 1) + \
                    np.log(np.linalg.det(self.cov))/2
        return H


