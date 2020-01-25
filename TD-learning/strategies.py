# This file contains very simple implementation of some Reinforcement Learning (RL) strategies.
# It aims to be the most generic and simplest possible code.
#
# Current available strategies :
# - EpsilonGreedy
# - EpsilonGreedyDecay
# - UCB
#
# Written by :
# Hippolyte L. DEBERNARDI @ Aix-Marseille University, 2020
# Jérémy FERSULA          @ Aix-Marseille University, 2020
#
# Licence is granted to freely use and distribute for any sensible/legal
# purpose so long as this comment remains in any distributed code.

import abc
import random
import numpy as np


class Strategy(abc.ABC):
    """
    A generic abstract class for a strategy.

    A strategy is here considered as a mechanism to balance exploitation/exploration.
    A strategy should only be responsible for the selection of the next action.
    """
    __metaclass__ = abc.ABCMeta

    def reset(self):
        pass

    @abc.abstractmethod
    def choose_action(self, state, actions, policies):
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError


class Random(Strategy):
    """
    The most simple strategy you can imagine. Used as a baseline for experiments.
    """
    def choose_action(self, state, actions, policies):
        return random.choice(actions)

    def __str__(self):
        return "Random"


class EpsilonGreedy(Strategy):
    """
    TODO
    """
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def choose_action(self, state, actions, policies):
        if random.random() < self.epsilon:
            return random.choice(actions)

        policy_max = max(policies)

        # if we have several best actions, randomly choose one among them
        if policies.count(policy_max) > 1:
            best_actions = [action for action, q in zip(actions, policies) if q == policy_max]
            return random.choice(best_actions)

        # if we have only one best action, choose it
        return actions[policies.index(policy_max)]

    def __str__(self):
        return fr"EpsilonGreedy($\epsilon$={self.epsilon})"


class EpsilonGreedyDecay(EpsilonGreedy):
    """
    TODO
    """
    def __init__(self, epsilon=0.1, decay_factor=0.99999):
        super().__init__(epsilon=epsilon)
        self.decay_factor = decay_factor
        self.initial_epsilon = epsilon
        
    def reset(self):
        self.epsilon = self.initial_epsilon

    def choose_action(self, state, actions, policies):
        self.epsilon *= self.decay_factor
        return super().choose_action(state, actions, policies)

    def __str__(self):
        return fr"EpsilonGreedyDecay($\epsilon$={self.epsilon},decay={self.decay_factor})"


class UCB(Strategy):
    """
    fixme : on peut inférer self.T à partir de self.N
    """
    def __init__(self, c):
        self.c = c

        # T holds the number of visits each (state) has been visited,
        # e.g. `self.T(0, 8)` means we visited 8 times the state 0
        self.T = {}

        # N holds the number of visits each (state, action) has been visited,
        # e.g. `self.N((0, 3), 8)` means we took 8 times the action 3 on state 0
        self.N = {}

        # B holds the UpperConfidenceBounds of all (state, action) tuples,
        # e.g. `self.B((0, 3), 8)` means the upper confidence bounds of the action 3 on state 0 is 8
        self.B = {}

    def reset(self):
        self.T = {}
        self.N = {}
        self.B = {}

    def _get_bound(self, state, action):
        return self.B.get((state, action), 0.0)

    def _get_n(self, state, action):
        return self.N.get((state, action), 0)

    def _get_t(self, state):
        return self.T.get(state, 0)

    def fit_action(self, state, action, actions):
        # increments the visits counters
        self.N[(state, action)] = self._get_n(state, action) + 1
        self.T[state] = self._get_t(state) + 1

        # the bounds for each action for that given state
        for action in actions:
            self.B[(state, action)] = self.c * np.sqrt(np.log(self._get_t(state)) /
                                                       self._get_n(state, action))

    def choose_action(self, state, actions, policies):
        # ----- exploration -----
        # ensure each action for that given state has been already chosen
        if self._get_t(state) < len(actions):
            best_action = actions[self._get_t(state)]

        # ----- exploitation -----
        # the best action is the value that max the UCB condition
        else:
            bounds = [self._get_bound(state, action) for action in actions]
            best_action = np.argmax([policy + bound for policy, bound in zip(policies, bounds)])

        self.fit_action(state, best_action, actions)
        return best_action
    
    def __str__(self):
        return f"UCB(C={self.c})"
