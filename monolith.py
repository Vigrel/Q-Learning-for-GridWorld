###########
# IMPORTS #
###########
import matplotlib.pyplot as plt
import numpy as np
import randomname
from gridworld import GridWorld

#####################
# GRIDWORD EXAMPLES #
#####################
random_generated = """ 
    wwwwwwwwwwwwwwwwwwww
    w             g    w
    w      w           w
    w               o ww
    w  w    w   o o    w
    w w  wow   w o     w
    ww     o     w o   w
    w     w   w    o   w
    w     ww           w
    w                  w
    ww    ww           w
    w           ww   w w
    w  ow              w
    w                w w
    w       o  o    w  w
    w w      w   w w w w
    w  ow  w           w
    w            o    ow
    w    a wo          w
    wwwwwwwwwwwwwwwwwwww
    """
simple = """ 
    wwwwwwwwwwwwwwwwwwww
    wa                ow
    w wwwwwwwwwwwww o ow
    w  wwwwwwwwwwwwwo ow
    w   wwwwwwwwwwwwo ow
    w    wwwwwwwwwwwo ow
    w     wwwwwwwwwwo ow
    w      wwwwwwwwwo ow
    w       wwwwwwwwo ow
    w        wwwwwwwo ow
    w         wwwwwwo ow
    w          wwwwwo ow
    w           wwwwo ow
    w            wwwo ow
    w             wwo ow
    w              wo ow
    w               o ow
    w               o ow
    w               ogow
    wwwwwwwwwwwwwwwwwwww
    """


########################
# ENVIROMENT GENERATOR #
########################
class Cell:
    WALL = "w"
    EMPTY = " "
    AGENT = "a"
    HOLE = "o"
    PRIZE = "g"


def generate_map(size, hole_percentage, prize_percentage, wall_percentage):
    game_map = np.full((size, size), Cell.EMPTY, dtype=str)

    game_map[0, :] = game_map[:, -1] = game_map[:, 0] = game_map[-1, :] = Cell.WALL

    agent_row, agent_col = np.random.randint(1, size - 1, size=2)
    game_map[agent_row, agent_col] = Cell.AGENT

    def place_items(item, percentage):
        num_items = int((size - 2) * (size - 2) * percentage)
        indices = np.random.choice(
            np.arange(1, size - 1), size=(num_items, 2), replace=True
        )
        game_map[indices[:, 0], indices[:, 1]] = item

    place_items(Cell.HOLE, hole_percentage)
    place_items(Cell.PRIZE, prize_percentage)
    place_items(Cell.WALL, wall_percentage)

    return game_map


def save_map(game_map):
    map_name = randomname.get_name().replace("-", "_").upper()
    with open("worlds.py", "a") as file:
        file.write(f"\n{map_name} = ''' \n")
        file.writelines("    " + "".join(row) + "\n" for row in game_map)
        file.write("    '''\n")


game_map = generate_map(20, 0.1, 0.01, 0.25)
save_map(game_map)

#####################
# N_STEP Q_LEARNING #
#####################
n_values = [1, 2, 3]
epsilon_values = [0.1, 0.2, 0.3]
env = GridWorld(random_generated, random_state=42)


def epsilon_greedy_policy(q_values, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(env.action_values)
    return np.argmax(q_values[state])


def plot_learning_curve(mean_costs, std_errors):
    x = np.arange(5, len(mean_costs[0]) * 5 + 1, 5)

    for i, (mean, std_err) in enumerate(zip(mean_costs, std_errors), start=1):
        mean = np.array(mean)
        std_err = np.array(std_err)

        plt.plot(x, mean, label=f"N={i}, ε={0.1 * i}")
        plt.fill_between(x, mean - std_err, mean + std_err, alpha=0.2)

    plt.title("Learning Curve")
    plt.xlabel("Episode Count")
    plt.ylabel("Cumulative Cost")
    plt.legend()
    plt.show()


def run_experiments(
    env, n_values, epsilon_values, repeat=10, alpha=0.1, gamma=0.99, num_episodes=5000
):
    mean_costs = []
    std_errors = []

    for n in n_values:
        print(n)
        n_step_mean_costs = []
        n_step_std_errors = []

        for epsilon in epsilon_values:
            print(epsilon)
            replications_costs = []

            for _ in range(repeat):
                q_values = np.full((env.state_count, env.action_size), 0)
                cumulative_cost = 0

                for _ in range(1, num_episodes + 1):
                    _epsilon = max(0.1, epsilon)

                    state = env.reset()
                    done = False
                    n_step_buffer = []

                    while not done:
                        action = epsilon_greedy_policy(q_values, state, _epsilon)
                        next_state, reward, done, _ = env.step(action)
                        n_step_buffer.append((state, action, reward))

                        if len(n_step_buffer) >= n:
                            n_step_return = sum(
                                [
                                    (gamma**i) * r
                                    for i, (_, _, r) in enumerate(n_step_buffer)
                                ]
                            )
                            first_state, first_action, _ = n_step_buffer.pop(0)

                            q_values[first_state][first_action] = (
                                1 - alpha
                            ) * q_values[first_state][first_action] + alpha * (
                                n_step_return
                                + (gamma**n) * np.max(q_values[next_state])
                            )

                        state = next_state
                        cumulative_cost += reward

                    replications_costs.append(cumulative_cost)

                n_step_mean = np.mean(replications_costs)
                n_step_std_err = np.std(replications_costs, ddof=1) / np.sqrt(repeat)

                n_step_mean_costs.append(n_step_mean)
                n_step_std_errors.append(n_step_std_err)

            mean_costs.append(n_step_mean_costs)
            std_errors.append(n_step_std_errors)

    plot_learning_curve(mean_costs, std_errors)
    return mean_costs, std_errors


run_experiments(env, n_values, epsilon_values)
