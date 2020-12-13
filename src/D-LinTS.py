import numpy as np
from math import log
from numpy.linalg import pinv
# from numpy import linalg as LA
# from scipy.stats import truncnorm
import scipy
from dataclasses import dataclass
from arm import ArmGaussian


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
    d: int
    delta: float
    alpha: float
    lambda_: float
    s: float
    l: float
    gamma: float
    name: str
    sigma_noise: float
    verbose: bool = True
    # S-M cannot be used with this model for the moment
    sm: bool = False
    omniscient: bool = False
    t: int = 0
    gamma2_t: float = 1.

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
        self.invcov = pinv(self.cov)  # 1 / self.lambda_ * np.identity(self.dim)
        self.b = np.zeros(self.dim)

    def select_arm(self, arms: list[ArmGaussian]) -> int:
        """
        Selecting an arm according to the D-FTGPL policy
        param:
            - arms : list of objects Arm with contextualized features
        Output:
        -------
        chosen_arm : index of the pulled arm
        """
        assert type(arms) == list, 'List of arms as input required'
        k_t = len(arms)  # available actions at time t
        ucb_s = np.zeros(k_t)  # upper-confidence bounds for every action
        const1 = np.sqrt(self.lambda_) * self.s
        beta_t = const1 + self.sigma_noise * \
            np.sqrt(self.c_delta + self.dim *
                    np.log(1 + self.l**2 * (1 - self.gamma2_t) / (self.dim *
                           self.lambda_*(1 - self.gamma**2))))
        for (i, arm) in enumerate(arms):
            features = arm.features
            if self.t < self.dim:
                ucb_s[i] = np.Inf
            else:
                ucb_s[i] = np.dot(self.hat_theta, features)

        # Shuffle to avoid always pulling the same arm when ties
        mixer = np.random.random(ucb_s.size)
        # # Sort the indices
        # ucb_indices = list(np.lexsort((mixer, ucb_s)))
        # # Reverse list
        # output = ucb_indices[::-1]
        # chosen_arm = output[0]

        # Sort the indices
        ucb_indices = np.lexsort((mixer, ucb_s))
        chosen_arm = ucb_indices[-1]
        if self.verbose:
            # Sanity checks
            print('-- lambda:', self.lambda_)
            print('--- beta_t:', beta_t)
            print('--- theta_hat: ', self.hat_theta)
            print('--- Design Matrix:', self.cov)
            print('--- Design Square Matrix:', self.cov_squared)
            print('--- UCBs:', ucb_s)
            print('--- Chosen arm:', chosen_arm)
        return chosen_arm

    def update_state(self, features: np.ndarray, reward: float):
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

    def re_init(self):
        """
        Re-init function to reinitialize the statistics while keeping the same hyperparameters
        """
        self.t = 0
        self.hat_theta = np.zeros(self.dim)
        self.cov = self.lambda_ * np.identity(self.dim)
        self.invcov = 1 / self.lambda_ * np.identity(self.dim)
        self.cov_squared = self.lambda_ * np.identity(self.dim)
        self.b = np.zeros(self.dim)
        self.gamma2_t = 1

        if self.verbose:
            print('Parameters of the policy reinitialized')
            print('Design Matrix after init: ', self.cov)

    def __str__(self):
        return 'D-LinTS' + self.name

    @staticmethod
    def id():
        return 'D-LinTS'