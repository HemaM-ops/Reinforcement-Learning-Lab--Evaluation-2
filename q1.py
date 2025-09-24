# monte_carlo_gridworld.py
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Environment definition: simple deterministic 5x5 gridworld ---
class GridWorld3x3:
    def __init__(self, start=(0,0), goal=(2,2)):
        self.rows = 3
        self.cols = 3
        self.start = start
        self.goal = goal
        self.actions = [0,1,2,3]  # Up, Right, Down, Left
        # action to delta mapping
        self.delta = {
            0: (-1, 0),  # Up
            1: (0, +1),  # Right
            2: (+1, 0),  # Down
            3: (0, -1),  # Left
        }

    def step(self, state, action):
        """Deterministic step. Return (next_state, reward, done)."""
        if state == self.goal:
            return state, 0, True
        r_delta = self.delta[action]
        next_r = state[0] + r_delta[0]
        next_c = state[1] + r_delta[1]
        # if out of bounds, stay in same cell
        if next_r < 0 or next_r >= self.rows or next_c < 0 or next_c >= self.cols:
            next_state = state
        else:
            next_state = (next_r, next_c)
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


# --- Monte Carlo First-Visit prediction under fixed (uniform random) policy ---
def generate_episode(env, policy, start_state):
    """Generates one episode following policy. Returns list of (S, A, R)."""
    episode = []
    state = start_state
    done = False
    while not done:
        action = policy(state)  # action chosen by policy
        next_state, reward, done = env.step(state, action)
        episode.append((state, action, reward))
        state = next_state
    return episode

def uniform_random_policy_factory(env):
    def policy(state):
        # uniform over all 4 actions (including when some moves are invalid)
        return random.choice(env.actions)
    return policy

def first_visit_mc_prediction(env, policy, num_episodes=10000, gamma=0.99, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    returns_sum = defaultdict(float)   # sum of returns for each state
    returns_count = defaultdict(int)   # visit count for first visits
    V = defaultdict(float)

    for ep in range(1, num_episodes+1):
        episode = generate_episode(env, policy, start_state=env.start)
        # compute returns G_t for each time step (discounted)
        G = 0.0
        # Precompute rewards for easier G computation in reverse
        rewards = [step[2] for step in episode]
        T = len(rewards)
        # find first-visit states in episode
        first_visit_states = set()
        for t, (s, a, r) in enumerate(episode):
            if s not in first_visit_states:
                first_visit_states.add(s)
                # compute return G_t as sum_{k=t}^{T-1} gamma^{k-t} * r_k
                G_t = 0.0
                pow_g = 1.0
                for k in range(t, T):
                    G_t += (pow_g * rewards[k])
                    pow_g *= gamma
                # accumulate
                returns_sum[s] += G_t
                returns_count[s] += 1
                V[s] = returns_sum[s] / returns_count[s]

        # optional: progress print
        if ep % (num_episodes//10) == 0:
            print(f"Episode {ep}/{num_episodes} completed.")

    # build a 2D array for easy plotting
    V_grid = np.zeros((env.rows, env.cols))
    for r in range(env.rows):
        for c in range(env.cols):
            s = (r,c)
            if env.is_terminal(s):
                V_grid[r,c] = 0.0  # terminal
            else:
                V_grid[r,c] = V.get(s, 0.0)
    return V_grid, V, returns_count

# --- Plotting helpers ---
def plot_value_heatmap(V_grid, title="Value function"):
    plt.figure(figsize=(6,5))
    plt.imshow(V_grid, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='V(s)')
    plt.title(title)
    # annotate cells
    rows, cols = V_grid.shape
    for r in range(rows):
        for c in range(cols):
            plt.text(c, r, f"{V_grid[r,c]:.2f}", ha='center', va='center', color='black')
    plt.gca().invert_yaxis()
    plt.show()

# --- Main block to run as script ---
if __name__ == "__main__":
    env = GridWorld3x3(start=(0,0), goal=(2,2))
    policy = uniform_random_policy_factory(env)
    # run first-visit MC prediction
    V_grid, V_dict, counts = first_visit_mc_prediction(env, policy, num_episodes=10000, gamma=0.99, seed=42)
    print("\nEstimated V(s) grid (rows are 0..2):\n", np.round(V_grid,2))
    plot_value_heatmap(V_grid, title="First-Visit MC: V(s) under Uniform Random Policy (10k episodes)")
