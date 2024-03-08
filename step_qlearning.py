from gridworld import GridWorld
import numpy as np
from worlds import AGED_AIRPORT

np.random.seed(42)
env = GridWorld(AGED_AIRPORT, random_state=42)

epsilon_values = [0.1, 0.2, 0.3]
n_values = [1, 2, 3]
min_epsilon = 0.01
decay_rate = 0.995


def epsilon_greedy_policy(q_values, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(env.action_values)
    return np.argmax(q_values[state])


def train_n_step_q_learning(
    env, epsilon, n, repeat=10, step=5, alpha=0.1, gamma=0.99, num_episodes=1000
):
    q_values = np.full((env.state_count, env.action_size), 0)

    for _ in range(repeat):
        for _ in range(num_episodes):
            _epsilon = max(min_epsilon, epsilon * decay_rate)

            state = env.reset()
            done = False

            n_step_buffer = []

            while not done:
                action = epsilon_greedy_policy(q_values, state, _epsilon)
                next_state, reward, done, _ = env.step(action)

                n_step_buffer.append((state, action, reward))

                if len(n_step_buffer) >= n:
                    n_step_return = sum(
                        [(gamma**i) * r for i, (_, _, r) in enumerate(n_step_buffer)]
                    )
                    first_state, first_action, _ = n_step_buffer.pop(0)

                    q_values[first_state][first_action] = (1 - alpha) * q_values[
                        first_state
                    ][first_action] + alpha * (
                        n_step_return + (gamma**n) * np.max(q_values[next_state])
                    )

                state = next_state

            for i, (s, a, r) in enumerate(n_step_buffer):
                n_step_return = sum(
                    [(gamma**j) * r for j, (_, _, r) in enumerate(n_step_buffer[i:])]
                )
                q_values[s][a] = (1 - alpha) * Q[s][a] + alpha * (
                    n_step_return
                    + (gamma ** len(n_step_buffer[i:])) * np.max(Q[next_state])
                )

    return Q


for n in n_values:
    for epsilon in epsilon_values:
        Q = train_n_step_q_learning(env, epsilon, n)
        pi = np.argmax(Q, axis=1)
        print(f"Epsilon={epsilon}", pi)
        env.show(pi)
