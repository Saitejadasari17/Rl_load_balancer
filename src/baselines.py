import numpy as np
try:
    from .environment import LoadBalancerEnv
except ImportError:
    # For direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from environment import LoadBalancerEnv

class BaselineAgent:
    """Base class for baseline load balancing algorithms."""
    def __init__(self, n_servers):
        self.n_servers = n_servers
        self.reset()

    def reset(self):
        self.last_action = -1

    def act(self, state):
        """Choose action based on current state."""
        raise NotImplementedError

class RoundRobinAgent(BaselineAgent):
    """Round Robin: Cycle through servers sequentially."""
    def act(self, state):
        self.last_action = (self.last_action + 1) % self.n_servers
        return self.last_action

class LeastConnectionsAgent(BaselineAgent):
    """Least Connections: Route to server with fewest queued requests."""
    def act(self, state):
        # State format per server: [cpu, queue, latency]
        features_per_server = 3
        queues = [state[i*features_per_server + 1] for i in range(self.n_servers)]  # Extract queue values
        return np.argmin(queues)  # Choose server with smallest queue

class RandomAgent(BaselineAgent):
    """Random: Choose server randomly."""
    def act(self, state):
        return np.random.randint(0, self.n_servers)

class WeightedRoundRobinAgent(BaselineAgent):
    """Weighted Round Robin: Consider server capacities."""
    def __init__(self, n_servers, weights=None):
        super().__init__(n_servers)
        self.weights = weights or [1.0] * n_servers  # Default equal weights
        self.cumulative_weights = np.cumsum(self.weights)
        self.total_weight = self.cumulative_weights[-1]
        self.current_weight = 0

    def act(self, state):
        self.current_weight = (self.current_weight + 1) % self.total_weight
        for i, cw in enumerate(self.cumulative_weights):
            if self.current_weight <= cw:
                return i
        return 0  # Fallback

def evaluate_agent(env, agent, n_episodes=10, max_steps=200):
    """Evaluate an agent over multiple episodes."""
    total_rewards = []
    latencies = []
    utilizations = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        agent.reset()
        episode_reward = 0
        episode_latencies = []
        episode_utilizations = []

        for step in range(max_steps):
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            # Convert reward back to latency (since reward = -latency)
            latency = -reward
            episode_latencies.append(latency)

            # Track server utilizations
            features_per_server = getattr(env, 'FEATURES_PER_SERVER', 3)
            server_cpus = [next_obs[i*features_per_server] for i in range(env.n_servers)]
            episode_utilizations.append(np.mean(server_cpus))

            obs = next_obs

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)
        latencies.append(np.mean(episode_latencies))
        utilizations.append(np.mean(episode_utilizations))

    return {
        'avg_reward': np.mean(total_rewards),
        'avg_latency': np.mean(latencies),
        'avg_utilization': np.mean(utilizations),
        'std_latency': np.std(latencies),
        'std_utilization': np.std(utilizations)
    }

def run_baseline_comparison(env, n_episodes=10):
    """Compare all baseline algorithms."""
    agents = {
        'Round Robin': RoundRobinAgent(env.n_servers),
        'Least Connections': LeastConnectionsAgent(env.n_servers),
        'Random': RandomAgent(env.n_servers),
        'Weighted RR': WeightedRoundRobinAgent(env.n_servers)
    }

    results = {}
    for name, agent in agents.items():
        print(f"Evaluating {name}...")
        results[name] = evaluate_agent(env, agent, n_episodes)
        print(f"  Avg Latency: {results[name]['avg_latency']:.2f} ms")
        print(f"  Avg Utilization: {results[name]['avg_utilization']:.3f}")

    return results