import colorednoise as cn
import numpy as np
import copy

# This module provides noise injection mechanisms for mutation in Genetic Algorithms, supporting three types of noise:
# Pink noise (time-correlated), Ornstein-Uhlenbeck noise (temporal noise with mean reversion), and Gaussian noise (independent samples). 
# These noise types are used to perturb neural network weights during evolutionary processes in reinforcement learning environments.
# The module includes functions for adjusting noise characteristics (e.g., decay factor) and ensures the generated noise is suitable for 
# injecting into the weight sets of individuals in the population.
# Pink noise class includes an extra function for injecting noise into the continuous input space.

class PinkNoiseInjection:
    @staticmethod
    def pink_noise(individual, decay_factor): # for weight mutation
        return cn.powerlaw_psd_gaussian(exponent = 1, size = individual.shape)*decay_factor
    @staticmethod
    def pink_noise_generate_samples(size, decay_factor): # for continuous input space noise injection
        # new, completely random pink noise samples each time, as random_state not set constant.
        return cn.powerlaw_psd_gaussian(exponent = 1, size = size)*decay_factor

class OUNoiseInjectionParameterSpace:
    """Ornstein-Uhlenbeck process."""
    """ This class was derived from the repository https://github.com/soeren-kirchner/project2-reacher """

    def __init__(self, size, mu=0, theta=0.15): # theta is set as 0.15, common practice according to Core Paper B, CW 1
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = 1
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def alter_decay_factor(self, decay_factor): # after every generation, the decay factor must be altered.
        self.sigma = decay_factor

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state

class GaussianNoiseParameterSpace:
    # since Gaussian noise consists of independently taken samples of noise, there is no need for objects per individual
    # all individuals can simply acquire a noise array.
    @staticmethod
    def gaussian_noise(shape, decay_factor):
        """
        Generates Gaussian noise

        Parameters:
            shape (int): dimensionality of space
            decay_factor: controls the scale of Gaussian noise
            loc: mean of the distribution
        Returns:
            np.ndarray: The updated noise array.
        """
        return np.random.normal(loc = 0.0, scale = decay_factor, size = shape)