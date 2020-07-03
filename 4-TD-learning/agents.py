# This file contains very simple implementation of some reinforcement learning agents.
# It aims to be the most generic and simplest possible code.
#
# Current available agents :
# - QLearn
# - SARSA
#
# Written by :
# Hippolyte L. DEBERNARDI @ Aix-Marseille University, 2020
# Jérémy FERSULA          @ Aix-Marseille University, 2020
#
# Licence is granted to freely use and distribute for any sensible/legal
# purpose so long as this comment remains in any distributed code.

import abc
import pickle


class Agent(abc.ABC):
    """
    A generic abstract class for an Agent.

    Parameters
    ----------
    actions: array-like
        Actions available for the agent.

    strategy: Strategy
        The exploitation/exploration strategy to use, e.g. EpsilonGreedy, UCB.

    alpha: float
        The learning rate. Called alpha for consistency with academic equations.

    gamma: float
        The discount factor. Called gamma for consistency with academic equations.
        Generally in [0, 1] :
            gamma=1 means no decay over time for the decision (i.e any decision at any given
            time is considered equal for the agent).
            gamma=0 leads the agent to only learn 1 step ahead (i.e any decision is independent).

    initial_policy: float
        The numeric value of a newly visited (state, action) tuple. Default is 0.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, actions, strategy, alpha, gamma, initial_policy=0.0):
        self.actions = actions
        self.strategy = strategy
        self.alpha = alpha
        self.gamma = gamma
        self.initial_policy = initial_policy

        # Q holds the policy, e.g. `self.Q((0, 3), 8)`
        # means taking the action 3 on state 0 will lead to a final cumulative reward of 8
        self.Q = {}

    def save(self, filename, pathname="resources/policies"):
        with open(f"{pathname}/{filename}.pkl", "wb") as f:
            pickle.dump(self.Q, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename, pathname="resources/policies"):
        with open(f"{pathname}/{filename}.pkl", "rb") as f:
            self.Q = pickle.load(f)

    def reset(self):
        self.strategy.reset()
        self.Q = {}

    def get_policy(self, state, action):
        return self.Q.get((state, action), self.initial_policy)

    def choose_action(self, state):
        # we pass directly the policy values to the strategy responsible to choose an action
        # to avoid circular dependencies on the Q dictionary and avoid None value when a given
        # (state, action) tuple isn't visited yet
        q = [self.get_policy(state, action) for action in self.actions]
        return self.strategy.choose_action(state, self.actions, q)

    @abc.abstractmethod
    def fit_step(self, state1, action1, reward, state2):
        raise NotImplementedError


class QLearn(Agent):
    """
    A generic Q-Learning Agent that can use various strategies.

    Policy update equation :
        Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s', a') - Q(s,a))

    It uses the best next choice of utility in later state to update the former state.

    Parameters
    ----------
    actions: array-like
        Actions available for the agent.

    strategy: Strategy
        The exploitation/exploration strategy to use, e.g. EpsilonGreedy, UCB.

    alpha: float
        The learning rate. Called alpha for consistency with academic equations.

    gamma: float
        The discount factor. Called gamma for consistency with academic equations.
        Generally in [0, 1] :
            gamma=1 means no decay over time for the decision (i.e any decision at any given
            time is considered equal for the agent).
            gamma=0 leads the agent to only learn 1 step ahead (i.e any decision is independent).

    initial_policy: float
        The numeric value of a newly visited (state, action) tuple. Default is 0.
    """
    def __init__(self, actions, strategy, alpha, gamma, initial_policy=0.0):
        super().__init__(actions=actions,
                         strategy=strategy,
                         alpha=alpha,
                         gamma=gamma,
                         initial_policy=initial_policy)

    def fit_step(self, state1, action1, reward, state2):
        q1 = self.Q.get((state1, action1), None)

        # if we never visited (state1, action1) before, we'll just initialize it
        if not q1:
            self.Q[(state1, action1)] = reward

        # we take another action and update our policy based on it
        else:
            q2 = max([self.get_policy(state2, action2) for action2 in self.actions])
            self.Q[(state1, action1)] += self.alpha * (reward + self.gamma * q2 - q1)

    def __str__(self):
        return fr"QLearn($\alpha$={self.alpha},$\gamma$={self.gamma})-{self.strategy}"


class SARSA(Agent):
    """
    A generic SARSA Agent that can use various strategies.

    Policy update equation :
        Q(s, a) += alpha * (reward(s, a) + gamma * (Q(s', a') - Q(s,a))

    It uses the next choice of policy in later state to update the former state.

    Parameters
    ----------
    actions: array-like
        Actions available for the agent.

    strategy: Strategy
        The exploitation/exploration strategy to use, e.g. EpsilonGreedy, UCB.

    alpha: float
        The learning rate. Called alpha for consistency with academic equations.

    gamma: float
        The discount factor. Called gamma for consistency with academic equations.
        Generally in [0, 1] :
            gamma=1 means no decay over time for the decision (i.e any decision at any given
            time is considered equal for the agent).
            gamma=0 leads the agent to only learn 1 step ahead (i.e any decision is independent).

    initial_policy: float
        The numeric value of a newly visited (state, action) tuple. Default is 0.
    """
    def __init__(self, actions, strategy, alpha, gamma, initial_policy=0.0):
        super().__init__(actions=actions,
                         strategy=strategy,
                         alpha=alpha,
                         gamma=gamma,
                         initial_policy=initial_policy)

    def fit_step(self, state1, action1, reward, state2):
        q1 = self.Q.get((state1, action1), None)

        # if we never visited (state1, action1) before, we'll just initialize it
        if not q1:
            self.Q[(state1, action1)] = reward

        # we take another action and update our policy based on it
        else:
            action2 = self.choose_action(state2)
            q2 = self.get_policy(state2, action2)
            self.Q[(state1, action1)] += self.alpha * (reward + self.gamma * q2 - q1)

    def __str__(self):
        return fr"SARSA($\alpha$={self.alpha},$\gamma$={self.gamma})-{self.strategy}"
