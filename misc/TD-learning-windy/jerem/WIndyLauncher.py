import matplotlib.pyplot as plt

from WindyMapping import WindyProblem
from agents import EpsilonGreedySARSA, EpsilonGreedyQLearn

x = 10
y = 7
start = (0, 3)
end = (7, 3)

wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

actions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

sarsa = EpsilonGreedySARSA(
            actions=actions, alpha=0.7, gamma=1, initial_policy=0, epsilon=0
        )

qlearn = EpsilonGreedyQLearn(
            actions=actions, alpha=0.7, gamma=1, initial_policy=0, epsilon=0
        )

# pb = WindyProblem(qlearn, x, y, start, end, wind, 1, show=True, nb_win=100)
pb = WindyProblem(sarsa, x, y, start, end, wind, 1, show=True, nb_win=100)

plt.plot(pb.steps)
plt.show()
