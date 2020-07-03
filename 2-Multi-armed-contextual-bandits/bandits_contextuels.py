# -*- coding: utf-8 -*-
"""

.. moduleauthor:: Valentin Emiya
"""

import numpy as np


class LinearBandits:
    """
    Linear bandit problem

    Parameters
    ----------
    n_arms : int
        Number of arms or actions
    n_features : int
        Number of features
    """

    def __init__(self, n_arms, n_features):
        self._theta = np.random.randn(n_features, n_arms)

    @property
    def n_arms(self):
        return self._theta.shape[1]

    @property
    def n_features(self):
        return self._theta.shape[0]

    def step(self, a, x):
        """
        Parameters
        ----------
        a : int
            Index of action/arm
        x : ndarray
            Context (1D array)

        Returns
        -------
        float
            Reward
        """

        assert 0 <= a
        assert a < self.n_arms
        return np.vdot(x, self._theta[:, a]) + np.random.randn()

    def get_context(self):
        """
        Returns
        -------
        ndarray
            Context (1D array)
        """
        return np.random.randn(self.n_features)

    def __str__(self):
        return '{}-arms linear bandit in dimension {}'.format(self.n_arms,
                                                              self.n_features)


class LinUCBAlgorithm():
    """

    Parameters
    ----------
    n_arms : int
        Number of arms
    n_features : int
        Number of features
    delta : float
        Confidence level in [0, 1]
    """

    def __init__(self, n_arms, n_features, delta):
        self.delta = delta
        self.alpha = 1 + np.sqrt(1/2 * np.log(2/delta))

        self.A = np.empty((n_features, n_features, n_arms))
        self.B = np.empty((n_features, n_arms))

        for idx_arm in range(n_arms):
            self.A[:, :, idx_arm] = np.identity(n_features)
            self.B[:, idx_arm] = np.zeros(n_features)

    @property
    def n_arms(self):
        return self.A.shape[2]

    @property
    def n_features(self):
        return self.A.shape[0]

    def get_action(self, x):
        """
        Choose an action

        Parameters
        ----------
        x : ndarray
            Context

        Returns
        -------
        int
            The chosen action
        """
        thetas = np.empty((self.n_arms, self.n_features))
        us = np.empty(self.n_arms)

        for idx_arm in range(self.n_arms):
            # on calcule naïvement l'inverse de A
            A_inv = np.linalg.inv(self.A[:, :, idx_arm])
            
            # calcul du theta du bras
            thetas[idx_arm] = A_inv @ self.B[:, idx_arm]

            # calcul du u du bras
            part1 = x.T @ thetas[idx_arm]
            part2 = self.alpha * np.sqrt(x.T @ A_inv @ x)
            us[idx_arm] = part1 + part2

        return np.argmax(us)

    def fit_step(self, action, reward, x):
        """
        Update current value estimates with an (action, reward) pair

        Parameters
        ----------
        action : int
        reward : float
        x : ndarray

        """
        self.A[:, :, action] += x @ x.T
        self.B[:, action] += reward * x


class LinUCBAlgorithmOptimized(LinUCBAlgorithm):
    """

    Parameters
    ----------
    n_arms : int
        Number of arms
    n_features : int
        Number of features
    delta : float
        Confidence level in [0, 1]
    """

    def __init__(self, n_arms, n_features, delta):
        LinUCBAlgorithm.__init__(self, n_arms, n_features, delta)

    def get_action(self, x):
        """
        Choose an action

        Parameters
        ----------
        x : ndarray
            Context

        Returns
        -------
        int
            The chosen action
        """
        thetas = np.empty((self.n_arms, self.n_features))
        us = np.empty(self.n_arms)

        for idx_arm in range(self.n_arms):
            A = self.A[:, :, idx_arm]
            B = self.B[:, idx_arm]
            # calcul du theta du bras
            thetas[idx_arm] = A @ B

            # calcul du u du bras
            part1 = x.T @ thetas[idx_arm]
            part2 = self.alpha * np.sqrt(x.T @ A @ x)
            us[idx_arm] = part1 + part2

        return np.argmax(us)

    def fit_step(self, action, reward, x):
        """
        Update current value estimates with an (action, reward) pair

        Parameters
        ----------
        action : int
        reward : float
        x : ndarray

        """
        # https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
        u = np.expand_dims(x, axis=1)

        """
        import scipy.linalg as la
        A = self.A[:, :, action]
        Ahat = A + np.outer(u, u.T)
        b = np.random.randn(u.shape[0])
        xhat = la.solve(Ahat, b)
        xhat2 = (Ainv @ (np.kron(u, u.T) @ Ainv)) / (1 + u.T @ (Ainv @ u))
        """

        Ainv = self.A[:, :, action]
        Ainv -= (Ainv @ u @ u.T @ Ainv) / (1 + u.T @ Ainv @ u)

        self.A[:, :, action] = Ainv
        self.B[:, action] += reward * x