# -*- coding: utf-8 -*-
"""

.. moduleauthor:: Valentin Emiya
"""

import abc
import numpy as np


class BernoulliMultiArmedBandits:
    """
    Bandit problem with Bernoulli distributions

    Parameters
    ----------
    true_values : array-like
        True values (expectation of reward) for each arm
    """
    def __init__(self, true_values):
        true_values = np.array(true_values)
        assert np.all(0 <= true_values)
        assert np.all(true_values <= 1)
        self._true_values = true_values

    @property
    def n_arms(self):
        """
        Number of arms

        Returns
        -------
        int
        """
        return self._true_values.size

    def step(self, a):
        """
        Play an arm and return reward

        Parameters
        ----------
        a : int
            Index of arm to be played

        Returns
        -------
        bool
            Reward obtained from playing arm `a` (true if win, false otherwise)
        """
        assert 0 <= a
        assert a < self.n_arms
        return np.random.rand() < self._true_values[a]

    def __str__(self):
        return '{}-arms bandit problem with Bernoulli distributions'.format(
            self.n_arms)


class NormalMultiArmedBandits:
    """
    Bandit problem with normal distributions

    Parameters
    ----------
    n_arms : int
        Number of arms or actions
    """

    def __init__(self, n_arms):
        self._true_values = np.random.randn(n_arms)

    @property
    def n_arms(self):
        """
        Number of arms

        Returns
        -------
        int
        """
        return self._true_values.size

    def step(self, a):
        """
        Play an arm and return reward

        Parameters
        ----------
        a : int
            Index of arm to be played

        Returns
        -------
        float
            Reward obtained from playing arm `a`
        """
        assert 0 <= a
        assert a < self.n_arms
        return np.random.randn() + self._true_values[a]

    def __str__(self):
        return '{}-arms bandit problem with Normal distributions'.format(
            self.n_arms)


class BanditAlgorithm(abc.ABC):
    """
    A generic abstract class for Bandit Algorithms

    Parameters
    ----------
    n_arms : int
        Number of arms
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, n_arms=10):
        self.n_arms = n_arms

    @abc.abstractmethod
    def get_action(self):
        """
        Choose an action (abstract)

        Returns
        -------
        int
            The chosen action
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit_step(self, action, reward):
        """
        Update current value estimates with an (action, reward) pair (abstract)

        Parameters
        ----------
        action : int
        reward : float

        """
        raise NotImplementedError


class RandomBanditAlgorithm(BanditAlgorithm):
    """
    A generic class for Bandit Algorithms

    Parameters
    ----------
    n_arms : int
        Number of arms
    """
    def __init__(self, n_arms=10):
        BanditAlgorithm.__init__(self, n_arms=n_arms)
        # Estimation of the value of each arm
        self._value_estimates = np.zeros(n_arms)
        # Number of times each arm has been chosen
        self._n_estimates = np.zeros(n_arms)

    def get_action(self):
        """
        Choose an action at random uniformly among the available arms

        Returns
        -------
        int
            The chosen action
        """
        return np.random.randint(self.n_arms)

    def fit_step(self, action, reward):
        """
        Do nothing since actions are chosen at random

        Parameters
        ----------
        action : int
        reward : float

        """
        pass


class GreedyBanditAlgorithm(BanditAlgorithm):
    """
    Greedy Bandit Algorithm

    Parameters
    ----------
    n_arms : int
        Number of arms
    """
    def __init__(self, n_arms=10):
        BanditAlgorithm.__init__(self, n_arms=n_arms)

        # Estimation of the value of each arm
        self._value_estimates = np.zeros(n_arms)

        # Number of times each arm has been chosen
        self._n_estimates = np.zeros(n_arms, dtype=int)

        # Used to not iterate over each arm when using `get_action`
        self._exploration_done = False

    def get_action(self):
        """
        Choose the action with maximum estimated value

        Returns
        -------
        int
            The chosen action
        """
        if self._exploration_done:
            return np.argmax(self._value_estimates) 

        # return the first non explored arm
        for idx_arm in range(self.n_arms):
            if self._n_estimates[idx_arm] == 0:
                # if it's the last arm then exploration is done
                if idx_arm == (self.n_arms-1):
                    self._exploration_done = True

                return idx_arm

    def fit_step(self, action, reward):
        """
        Update current value estimates with an (action, reward) pair

        Parameters
        ----------
        action : int
        reward : float
        """
        self._n_estimates[action] += 1
        self._value_estimates[action] += 1 / self._n_estimates[action] * (reward - self._value_estimates[action])
        


class EpsilonGreedyBanditAlgorithm(GreedyBanditAlgorithm,
                                   RandomBanditAlgorithm):
    """
    Epsilon-greedy Bandit Algorithm

    Parameters
    ----------
    n_arms : int
        Number of arms
    epsilon : float
        Probability to choose an action at random
    """
    def __init__(self, n_arms=10, epsilon=0.1):
        GreedyBanditAlgorithm.__init__(self, n_arms=n_arms)
        self._epsilon = epsilon

    def get_action(self):
        """
        Get Epsilon-greedy action

        Choose an action at random with probability epsilon and a greedy
        action otherwise.

        Returns
        -------
        int
            The chosen action
        """
        """
        Choose the action with maximum estimated value

        Returns
        -------
        int
            The chosen action
        """
        if self._exploration_done:
            # exploitation
            if np.random.rand() > self._epsilon:
                return np.argmax(self._value_estimates)
            # exploration
            else:
                return np.random.randint(self.n_arms)

        # return the first non explored arm
        for idx_arm in range(self.n_arms):
            if self._n_estimates[idx_arm] == 0:
                # if it's the last arm then exploration is done
                if idx_arm == (self.n_arms-1):
                    self._exploration_done = True

                return idx_arm


class UcbBanditAlgorithm(GreedyBanditAlgorithm):
    """

    Parameters
    ----------
    n_arms : int
        Number of arms
    c : float
        Positive parameter to adjust exploration/explotation UCB criterion
    """
    def __init__(self, n_arms, c):
        GreedyBanditAlgorithm.__init__(self, n_arms=n_arms)
        self.c = c
        self.current_iter = 0
        self._upper_confidence_bounds = np.empty(n_arms)

    def get_action(self):
        """
        Get UCB action

        Returns
        -------
        int
            The chosen action
        """

        if self._exploration_done:
            self.current_iter += 1
            promising_values = np.zeros(self.n_arms)

            for idx_arm in range(self.n_arms):
                self.compute_upper_confidence_bound(idx_arm)

                promising_values[idx_arm] = self._value_estimates[idx_arm]
                promising_values[idx_arm] += self._upper_confidence_bounds[idx_arm]

            return np.argmax(promising_values)

        # return the first non explored arm
        for idx_arm in range(self.n_arms):
            if self._n_estimates[idx_arm] == 0:
                # if it's the last arm then exploration is done
                if idx_arm == (self.n_arms-1):
                    self._exploration_done = True

                return idx_arm

    def compute_upper_confidence_bound(self, idx_arm):
        self._upper_confidence_bounds[idx_arm] = self.c * np.sqrt(np.log(self.current_iter) / self._n_estimates[idx_arm])


class ThompsonSamplingAlgorithm(BanditAlgorithm):
    """

    Parameters
    ----------
    n_arms : int
        Number of arms
    """
    def __init__(self, n_arms):
        BanditAlgorithm.__init__(self, n_arms=n_arms)
        self._alphas = np.ones(n_arms)
        self._betas = np.ones(n_arms)

    def get_action(self):
        thetas = np.zeros(self.n_arms)
        for idx_arm in range(self.n_arms):
            thetas[idx_arm] = np.random.beta(self._alphas[idx_arm],
                                             self._betas[idx_arm])
        
        return np.argmax(thetas)

    def fit_step(self, action, reward):
        self._alphas[action] += reward
        self._betas[action] += (1-reward)
