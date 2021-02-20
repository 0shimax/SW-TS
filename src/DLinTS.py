import numpy as np
from math import log
from numpy.linalg import pinv
# from numpy import linalg as LA
# from scipy.stats import truncnorm
import scipy
from dataclasses import dataclass
# from arm import ArmGaussian


@dataclass
class DLinTS(object):
    """
    Implementation of the class for the Discounted Follow-The-Gaussian-Perturbed-Leader
    param:
        - d: dimension of the action vectors (feature dimension)
        - delta: probability of theta in the confidence bound
        - alpha: tuning the exploration parameter
        - lambda_: regularization parameter
        - s: constant such that L2 norm of theta smaller than s
        - gamma: discount parameter
        - name: additional suffix when comparing several policies (optional)
        - sm: Should Sherman-Morisson formula be used for inverting matrices ?
        - sigma_noise: square root of the variance of the noise
        - verbose: To print information
        - omniscient: Does the policy knows when the breakpoints happen ?
    ACTION NORMS ARE SUPPOSED TO BE BOUNDED BE 1
    """
    dim: int
    delta: float
    alpha: float
    lambda_: float
    # s: float
    # l: float
    gamma: float
    sigma_noise: float
    verbose: bool  # = True
    # S-M cannot be used with this model for the moment
    sm: bool  # = False
    # omniscient: bool  # = False
    t: int = None
    gamma2_t: float = None

    def __post_init__(self):
        ''' build attributes '''
        # first term in square root
        self.c_delta = 2 * log(1 / self.delta)

        ''' attributes for the re-init '''
        # model parameter
        self.hat_theta = np.zeros(self.dim)
        # Design Matrix
        self.cov = self.lambda_ * np.identity(self.dim)
        # Design Square Matrix
        self.cov_squared = self.lambda_ * np.identity(self.dim)
        self.invcov = 1 / self.lambda_ * np.identity(self.dim)
        self.b = np.zeros(self.dim)
        self.t = 0
        self.gamma2_t = 1.0

    def update_state(self, features:np.ndarray, reward:float):
        """
        Updating the main parameters for the model
        param:
            - features: Feature used for updating
            - reward: Reward used for updating
        Output:
        -------
        Nothing, but the class instances are updated
        """
        assert isinstance(features, np.ndarray), 'np.array required'
        aat = np.outer(features, features.T)
        self.gamma2_t *= self.gamma ** 2
        self.cov = self.gamma * self.cov + aat + (1-self.gamma) * self.lambda_ * np.identity(self.dim)
        self.cov_squared = self.gamma ** 2 * self.cov_squared + aat + (1 - self.gamma ** 2) * self.lambda_ * np.identity(self.dim)
        self.b = self.gamma * self.b + reward * features

        # const1 = np.sqrt(self.lambda_) * self.s
        # beta_t = const1 + self.sigma_noise *\
        #     np.sqrt(self.c_delta + self.dim * np.log(1 + self.l**2 *(1-self.gamma2_t)/(self.dim * self.lambda_*(1 - self.gamma**2))))
        self.gamma2_t *= self.gamma ** 2

        if not self.sm:
            self.invcov = pinv(self.cov)
        else:
            raise NotImplementedError("Method SM is not implemented for D-LinTS")
        z = np.random.multivariate_normal(np.zeros(self.dim), np.eye(self.dim))
        self.hat_theta = np.inner(self.invcov, self.b + self.alpha * np.dot(scipy.linalg.sqrtm(self.cov_squared).real, z))

        if self.verbose:
            print('AAt:', aat)
            print('Policy was updated at time t= ' + str(self.t))
            print('Reward received  =', reward)
        self.t += 1

    def calculate_win_rate(self, features:np.ndarray):
        assert isinstance(features, np.ndarray), 'np.array required'
        return features.dot(self.hat_theta)[0][0]

    def __str__(self):
        return 'D-LinTS'

    @staticmethod
    def id():
        return 'D-LinTS'