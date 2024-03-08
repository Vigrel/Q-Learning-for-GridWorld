import numpy as np
from gridworld import GridWorld
import matplotlib.pyplot as plt

np.random.seed(42)


def epsilon_greedy_policy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(env.action_values)
    return np.argmax(Q[state])


def train_q_learning(env, epsilon, alpha=0.01, gamma=0.99, num_episodes=10000):
    q_values = np.full((env.state_count, env.action_size), 0)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        cumulative_cost = 0

        while not done:
            action = epsilon_greedy_policy(q_values, state, epsilon)
            next_state, reward, done, _ = env.step(action)

            q_values[state][action] = (1 - alpha) * q_values[state][action] + alpha * (
                reward + gamma * np.max(q_values[next_state])
            )

            state = next_state
            cumulative_cost += reward

    return q_values


world = """
    wwwwwwwwwwwwwwwww
    wa   o   w     gw
    w               w
    www  o   www  www
    w               w
    wwwww    o    www
    w     ww        w
    wwwwwwwwwwwwwwwww
    """

env = GridWorld(world, random_state=42)
epsilon_values = [0.1, 0.2, 0.3]

for epsilon in epsilon_values:
    Q = train_q_learning(env, epsilon)
    pi = np.argmax(Q, axis=1)
    print(f"Epsilon={epsilon}")
    env.show(pi)
