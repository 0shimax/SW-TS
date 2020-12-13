import numpy as np
from dataclasses import dataclass


class Arm(object):
    def pull(self, theta, sigma_noise):
        print('pulling from the parent class')
        pass

    def get_expected_reward(self, theta):
        print('Receiving reward from the parent class')


@dataclass
class ArmGaussian(Arm):
    """
    Arm vector with gaussian noise
    """
    features: np.ndarray

    def __post_init__(self):
        """
        Constructor
        """
        # assert isinstance(vector, np.ndarray), 'np.array required'
        # self.features = vector  # action for the arm, numpy-array
        self.dim = self.features.shape[0]

    def get_expected_reward(self, theta: np.ndarray) -> float:
        """
        Return dot(A_t,theta)
        """
        assert isinstance(theta, np.ndarray), 'np.array required for the theta vector'
        return np.dot(self.features, theta)

    def pull(self, theta: np.ndarray, sigma_noise: float) -> float:
        """
        We are in the stochastic setting.
        The reward is sampled according to Normal(dot(A_t,theta),sigma_noise**2)
        """
        return np.random.normal(self.get_expected_reward(theta), sigma_noise)