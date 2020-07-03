# extra modules
import numpy as np
import matplotlib.pyplot as plt

# own modules
from games import WindyEnv
from agents import EpsilonGreedyQLearn, EpsilonGreedySARSA


def run_one_simulation(game, agent, state1):
    steps = 0
    is_done = False

    while not is_done:
        steps += 1
        # ----- CHOOSE AN ACTION (AGENT) -----
        action = agent.choose_action(state1)

        # ----- UPDATE STATE -----
        reward, state2, is_done = game.do_action(state1, action)

        # ----- UPDATE AGENT -----
        agent.fit_step(state1, action, reward, state2)

        # ----- WE'RE MOVING ONTO THE NEXT STATE -----
        state1 = state2

    return steps, agent


if __name__ == "__main__":
    # ----- GAME CONFIGURATION -----
    grid = np.zeros((7, 10))
    winds = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
    assert winds.shape[0] == grid.shape[1]

    source = (3, 0)
    destination = (3, 7)

    game = WindyEnv(grid=grid, winds=winds, winning_cell=destination)
    actions = game.get_actions()

    # ----- AGENT PARAMETERS -----
    alpha = 0.7
    gamma = 1
    initial_policy = 0
    epsilon = 0.1

    agents = {
        "EpsilonGreedyQLearn": EpsilonGreedyQLearn(
            actions=actions,
            alpha=alpha,
            gamma=gamma,
            initial_policy=initial_policy,
            epsilon=epsilon
        ),
        "EpsilonGreedySARSA": EpsilonGreedySARSA(
            actions=actions,
            alpha=alpha,
            gamma=gamma,
            initial_policy=initial_policy,
            epsilon=epsilon
        )
    }

    # simulation parameters
    simulations = 200
    results = {
        # "Random": np.zeros(simulations),
        "EpsilonGreedyQLearn": np.zeros(simulations),
        "EpsilonGreedySARSA": np.zeros(simulations),
    }

    for agent_name, agent in agents.items():
        for simulation_idx in range(simulations):
            steps, agent = run_one_simulation(game, agent, source)
            results[agent_name][simulation_idx] = steps

        # for (state, action), policy in agent.Q.items():
        #     print(state, action, policy)

    # print(game.to_string(source))

    # ----- PLOT RESULT -----
    for agent_name, result in results.items():
        plt.semilogy(result, label=f"{agent_name} - {np.mean(result)}")
    plt.legend()
    plt.title(f"Nombre de coups joués par un agent avant victoire.\n"
              f"$epsilon={epsilon}$, $alpha={alpha}$")
    plt.savefig(f"windy_epsilon={epsilon}_alpha={alpha}.png")
    plt.show()
