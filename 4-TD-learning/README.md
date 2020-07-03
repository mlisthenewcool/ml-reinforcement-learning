# Various reinforcement learning strategies on openAI Cartpole environment

* Written with Python 3.
* Relies on : numpy, gym, [matplotlib, ]
* Authors : Jérémy FERSULA & Hippolyte L. DEBERNARDI

## Agents available

* QLearn
* SARSA

## Strategies available

* EpsilonGreedy
* EpsilonGreedyDecay
* UpperConfidenceBounds - UCB

## How to use the code

* Instantiate a QLearn agent with an EpsilonGreedy strategy

```
from agents import QLearn
from strategies import EpsilonGreedy

# initialize strategy
epsilon = 0.1
eps_greedy_strategy = EpsilonGreedy(epsilon)

# initialize agent
actions = [0, 1]
alpha = 0.7
gamma = 1
initial_policy = 1
qlearn_eps_greedy = QLearn(actions, eps_greedy_strategy, alpha, gamma, initial_policy)
```

## Analyse efficiency for different strategies

* Run learning : Explore the code of launcher_ucb.py
* Run analysis : Explore the Jupyter Notebook 

## Performance analysis under environment modification

![Angle_Disc](resources/plots/Angle_Disc.png?raw=true "Angle_Disc")
![AngVel_Disc](resources/plots/AngVel_Disc.png?raw=true "AngVel_Disc")
![Pole_Disc](resources/plots/Pole_Size.png?raw=true "AngVel_Disc")
