import gym
import numpy as np
import matplotlib.pyplot as plt

# own modules
from discretisations import obs_to_state_1
from agents import QLearn, SARSA
from strategies import EpsilonGreedy, EpsilonGreedyDecay, UCB


if __name__ == "__main__":
    # ----- STRATEGIES PARAMETERS -----
    epsilon = 0.1
    decay_factor = 0.9999
    c = 20

    strategies = [
        EpsilonGreedy(epsilon),
        EpsilonGreedyDecay(epsilon, decay_factor),
        UCB(c)
    ]

    # ----- AGENTS PARAMETERS -----
    actions = [0, 1]
    alpha = 0.7
    gamma = 1
    initial_policy = 0

    agents = {
        "Random": SARSA(
            actions=actions,
            strategy=EpsilonGreedy(epsilon=1),
            alpha=alpha,
            gamma=gamma,
            initial_policy=initial_policy
        ),

        "QLearn-EpsilonGreedy": QLearn(
            actions=actions,
            strategy=strategies[0],
            alpha=alpha,
            gamma=gamma,
            initial_policy=initial_policy,
        ),

        "SARSA-EpsilonGreedy": SARSA(
            actions=actions,
            strategy=strategies[0],
            alpha=alpha,
            gamma=gamma,
            initial_policy=initial_policy,
        ),

        "QLearn-EpsilonGreedyDecay": QLearn(
            actions=actions,
            strategy=strategies[1],
            alpha=alpha,
            gamma=gamma,
            initial_policy=initial_policy,
        ),

        "SARSA-EpsilonGreedyDecay": SARSA(
            actions=actions,
            strategy=strategies[1],
            alpha=alpha,
            gamma=gamma,
            initial_policy=initial_policy,
        ),

        "QLearn-UCB": QLearn(
            actions=actions,
            strategy=strategies[2],
            alpha=alpha,
            gamma=gamma,
            initial_policy=initial_policy,
        ),

        "SARSA-UCB": SARSA(
            actions=actions,
            strategy=strategies[2],
            alpha=alpha,
            gamma=gamma,
            initial_policy=initial_policy,
        ),
    }

    # ----- GAME INITIALIZATION -----
    env = gym.make('CartPole-v0')

    # ----- SIMULATIONS -----
    simulations = 1_000
    results = {
        "Random": np.zeros(simulations),
        "QLearn-EpsilonGreedy": np.zeros(simulations),
        "SARSA-EpsilonGreedy": np.zeros(simulations),
        "QLearn-EpsilonGreedyDecay": np.zeros(simulations),
        "SARSA-EpsilonGreedyDecay": np.zeros(simulations),
        "QLearn-UCB": np.zeros(simulations),
        "SARSA-UCB": np.zeros(simulations)
    }

    observations = list()

    for agent_name, agent in agents.items():
        for simulation_idx in range(simulations):
            steps = 0
            done = False
            observation1 = env.reset()

            observations = list()
            while not done:
                # ----- SHOULD WE DISPLAY THE GRAPHIC INTERFACE ? -----
                # env.render()

                # ----- THE AGENT CHOOSE AN ACTION -----
                state1 = obs_to_state_1(observation1)
                action = agent.choose_action(state1)

                # ----- UPDATE ENV -----
                observation2, reward, done, info = env.step(action)
                state2 = obs_to_state_1(observation2)

                # ----- UPDATE AGENT -----
                agent.fit_step(state1, action, reward, state2)

                # fixme, saving observations for later purposes
                observations.append(observation1)
                observations.append(observation2)

                # ----- WE'RE MOVING ONTO THE NEXT STEP -----
                observation1 = observation2
                steps += 1

            # ----- UPDATE SIMULATION STEPS COUNTER -----
            results[agent_name][simulation_idx] = steps

        print(agent_name, " -> ", np.mean(results[agent_name]))

    # ----- PLOT RESULT -----
    for agent_name, result in results.items():
        # result_smooth = [result[i:i+10].mean() for i in range(len(result)) if i % 10 == 0]
        result_sum = result.cumsum(axis=0)
        plt.plot(result_sum, label=f"{agent_name} - {np.mean(result)}")
    plt.legend()
    plt.title("évolution du résultat cumulé moyen par simulation")

    config_params = {
        "simulations": simulations,
        "alpha": alpha,
        "gamma": gamma,
        "initial_policy": initial_policy,
        "epsilon": epsilon,
        "decay_factor": decay_factor,
        "c": c
    }

    filename = "results_"
    filename += "_".join([f"{k}={v}" for k, v in config_params.items()])
    plt.savefig("./resources/" + filename + ".png")
    plt.show()

    observations = np.array(observations)
    print()
    print("    ", "cart_position, cart_velocity, pole_angle, pole_velocity")
    print("---------------------------------------------------------------")
    print("min ", [round(val, 5) for val in np.min(observations, axis=0)])
    print("max ", [round(val, 5) for val in np.max(observations, axis=0)])
    print("var ", [round(val, 5) for val in np.var(observations, axis=0)])
    print("mean", [round(val, 5) for val in np.mean(observations, axis=0)])

    # saving files
    """
    for agent_name, agent in agents.items():
        if agent_name is not "Random":
            if agents_reload[agent_name] is True:
                import os
                os.remove(f"resources/{agent_name}_{old_simulations}.pkl")
                simulations_tmp = simulations + old_simulations
                agent.save(f"{agent_name}_{simulations_tmp}")
            else:
                agent.save(f"{agent_name}_{simulations}")
    """

    """
    # ----- SHOW TIME -----
    for agent_name, agent in agents.items():
        print("SHOW TIME ", agent_name)
        for simulation_idx in range(2):
            observation = env.reset()
            steps = 0
            done = False

            while not done:
                env.render()
                state1 = obs_to_state_1(observation)
                action = agent.choose_action(state1)
                observation, reward, done, info = env.step(action)
    """

    env.close()
