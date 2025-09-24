import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# --- Environment: deterministic 3x3 Gridworld ---
class GridWorld3x3:
    def __init__(self, start=(0,0), goal=(2,2)):
        self.rows = 3
        self.cols = 3
        self.start = start
        self.goal = goal
        self.actions = [0,1,2,3]  # Up, Right, Down, Left
        self.delta = {
            0: (-1, 0),  # Up
            1: (0, +1),  # Right
            2: (+1, 0),  # Down
            3: (0, -1),  # Left
        }
    def step(self, state, action):
        if state == self.goal:
            return state, 0.0, True
        dr, dc = self.delta[action]
        nr, nc = state[0] + dr, state[1] + dc
        if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
            next_state = state
        else:
            next_state = (nr, nc)
        if next_state == self.goal:
            return next_state, 10.0, True
        else:
            return next_state, -1.0, False
    def is_terminal(self, state):
        return state == self.goal
    def all_states(self):
        for r in range(self.rows):
            for c in range(self.cols):
                yield (r,c)

# --- ε-greedy policy factory ---
def epsilon_greedy_policy_factory(Q, env, epsilon=0.1):
    def policy(state):
        if env.is_terminal(state):
            return None
        if random.random() < epsilon:
            return random.choice(env.actions)
        else:
            q_values = [Q[(state,a)] for a in env.actions]
            max_q = max(q_values)
            best_actions = [a for a,q in zip(env.actions, q_values) if q == max_q]
            return random.choice(best_actions)
    return policy

# --- Episode generation ---
def generate_episode(env, policy, start_state):
    episode = []
    state = start_state
    done = False
    while not done:
        action = policy(state)
        next_state, reward, done = env.step(state, action)
        episode.append((state, action, reward))
        state = next_state
    return episode

# --- Every-Visit MC Control ---
def every_visit_mc_control(env, num_episodes=100000, gamma=0.99, epsilon=0.1, seed=0):
    random.seed(seed); np.random.seed(seed)
    Q = defaultdict(float)
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)

    for ep in range(num_episodes):
        policy = epsilon_greedy_policy_factory(Q, env, epsilon)
        episode = generate_episode(env, policy, start_state=env.start)

        rewards = [r for (_,_,r) in episode]
        T = len(rewards)
        for t,(s,a,r) in enumerate(episode):
            Gt, discount = 0.0, 1.0
            for k in range(t,T):
                Gt += discount * rewards[k]
                discount *= gamma
            returns_sum[(s,a)] += Gt
            returns_count[(s,a)] += 1
            Q[(s,a)] = returns_sum[(s,a)] / returns_count[(s,a)]

    # --- Derive greedy policy and V(s) ---
    policy_grid = np.full((env.rows, env.cols), " ", dtype=object)
    V_grid = np.zeros((env.rows, env.cols))
    arrows = {0:"↑", 1:"→", 2:"↓", 3:"←"}

    for r in range(env.rows):
        for c in range(env.cols):
            s = (r,c)
            if env.is_terminal(s):
                policy_grid[r,c] = "G"
                V_grid[r,c] = 0.0
            else:
                q_vals = [Q[(s,a)] for a in env.actions]
                best = np.argmax(q_vals)
                policy_grid[r,c] = arrows[best]
                V_grid[r,c] = max(q_vals)

    return Q, policy_grid, V_grid

# --- Heatmap plot ---
def plot_value_heatmap(V_grid, title="Value Function"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4,4))
    plt.imshow(V_grid, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="V(s)")
    plt.title(title)
    rows, cols = V_grid.shape
    for r in range(rows):
        for c in range(cols):
            plt.text(c, r, f"{V_grid[r,c]:.2f}", ha="center", va="center", color="black")
    plt.gca().invert_yaxis()
    plt.show()

# --- Main ---
if __name__ == "__main__":
    env = GridWorld3x3()
    Q, policy_grid, V_grid = every_visit_mc_control(env, num_episodes=100000, gamma=0.99, epsilon=0.1, seed=42)

    print("\nGreedy Policy derived from Every-Visit MC Control (100k episodes):")
    for r in range(env.rows):
        print(" ".join(policy_grid[r]))

    print("\nEstimated V(s) grid (rows=0..2, cols=0..2):")
    print(np.round(V_grid,2))

    plot_value_heatmap(V_grid, title="Value Function (MC Control)")
