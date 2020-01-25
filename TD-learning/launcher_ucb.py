import gym
import numpy as np
import matplotlib.pyplot as plt

# own modules
from discretisations import obs_to_state_1
from agents import QLearn, SARSA
from strategies import EpsilonGreedy, EpsilonGreedyDecay, UCB


def do(size=(9, 5), sim=1000):
    # ----- STRATEGIES PARAMETERS -----
    c_all = [0.1, 1, 5, 10, 50, 100]

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
    }

    for c in c_all:
        agents[f"QLearn-UCB-{c}"] = QLearn(
            actions=actions,
            strategy=UCB(c),
            alpha=alpha,
            gamma=gamma,
            initial_policy=initial_policy,
        )

        agents[f"SARSA-UCB-{c}"] = QLearn(
            actions=actions,
            strategy=UCB(c),
            alpha=alpha,
            gamma=gamma,
            initial_policy=initial_policy,
        )

    # ----- GAME INITIALIZATION -----
    env = gym.make('CartPole-v0')

    # ----- SIMULATIONS -----
    simulations = sim
    results = {agent_name: np.zeros(simulations) for agent_name in agents.keys()}

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
        result_smooth = [result[i-30:i+30].mean() for i in range(30, len(result) - 30)]
        # result_sum = result.cumsum(axis=0)
        plt.plot(result_smooth, label=f"{agent_name} - {np.mean(result)}")
    plt.legend()
    plt.title("évolution de la récompense en faisant varier le paramètre c de la stratégie UCB")
    fig = plt.gcf()
    fig.set_size_inches(size[0], size[1])
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
    # for agent_name, agent in agents.items():
    #     if agent_name is not "Random":
    #         agent.save(f"{agent_name}_{simulations}")

    env.close()


if __name__ == "__main__":
    do()
