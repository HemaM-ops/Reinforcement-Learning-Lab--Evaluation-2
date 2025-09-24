import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

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
        """Return (next_state, reward, done)."""
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

# --- Uniform random policy ---
def uniform_random_policy_factory(env):
    def policy(state):
        return random.choice(env.actions)
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

# --- First-Visit MC Prediction ---
def first_visit_mc_prediction(env, policy, num_episodes=10000, gamma=0.99, seed=0):
    random.seed(seed); np.random.seed(seed)
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    V = defaultdict(float)

    for ep in range(num_episodes):
        episode = generate_episode(env, policy, start_state=env.start)
        rewards = [r for (_,_,r) in episode]
        T = len(rewards)
        first_visits = set()
        for t,(s,a,r) in enumerate(episode):
            if s not in first_visits:
                first_visits.add(s)
                Gt, discount = 0.0, 1.0
                for k in range(t,T):
                    Gt += discount * rewards[k]
                    discount *= gamma
                returns_sum[s] += Gt
                returns_count[s] += 1
                V[s] = returns_sum[s] / returns_count[s]

    V_grid = np.zeros((env.rows, env.cols))
    for r in range(env.rows):
        for c in range(env.cols):
            s = (r,c)
            V_grid[r,c] = 0.0 if env.is_terminal(s) else V.get(s,0.0)
    return V_grid, V

# --- Plotting ---
def plot_value_heatmap(V_grid, title="Value Function"):
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
    policy = uniform_random_policy_factory(env)

    V_grid, V_dict = first_visit_mc_prediction(env, policy, num_episodes=10000, gamma=0.99, seed=42)
    print("\nFirst-Visit MC Prediction (10k episodes) Value Grid:\n", np.round(V_grid,2))
    plot_value_heatmap(V_grid, title="MC Prediction (Uniform Random Policy)")
