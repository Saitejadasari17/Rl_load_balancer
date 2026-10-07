import gymnasium as gym
import numpy as np

class LoadBalancerEnv(gym.Env):
    """
    Custom Gym environment for load balancing simulation.
    Matches the paper "Adaptive Load Balancing in Distributed Systems...":
    State features per server:
        1. CPU utilization [0, 1]
        2. Queue length (normalized by Q_max)
        3. Processing latency (exponential moving average, normalized)
    """

    FEATURES_PER_SERVER = 3

    def __init__(self, n_servers=3, max_queue=20, max_steps=200, trace_data=None):
        super(LoadBalancerEnv, self).__init__()

        self.n_servers = n_servers
        self.max_queue = max_queue
        self.max_steps = max_steps
        self.trace_data = trace_data  # Real traffic traces from Azure dataset
        self.trace_index = 0

        # Action space: choose which server (0 to n_servers-1)
        self.action_space = gym.spaces.Discrete(n_servers)

        # Observation space: 3 normalized features per server (all in [0, 1])
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_servers * self.FEATURES_PER_SERVER,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Initialize servers
        self.servers = [{
            'cpu': 0.0,        # CPU utilization 0-1
            'queue': 0,        # Number of pending requests
            'latency': 0.0,    # Average response time (normalized)
        } for _ in range(self.n_servers)]

        self.current_step = 0
        self.total_requests = 0

        return self._get_state(), {}

    def _get_state(self):
        """Get current state as numpy array."""
        state = []
        for server in self.servers:
            state.extend([
                server['cpu'],
                min(server['queue'] / self.max_queue, 1.0),  # Normalize queue
                server['latency'],
            ])
        return np.array(state, dtype=np.float32)

    def step(self, action):
        """
        Execute one step in the environment.
        """
        if not (0 <= action < self.n_servers):
            raise ValueError(f"Invalid action {action}. Must be 0-{self.n_servers-1}")

        # Get request load from traces or generate randomly
        if self.trace_data is not None:
            # Use real Azure dataset trace
            request_load = self.trace_data[self.trace_index % len(self.trace_data)]
            self.trace_index += 1
        else:
            # Synthetic random traffic
            request_load = np.random.uniform(0.05, 0.3)  # Random request size

        # Update chosen server
        server = self.servers[action]
        server['queue'] = min(server['queue'] + 1, self.max_queue)  # Cap true queue length
        server['cpu'] = min(server['cpu'] + request_load, 1.0)

        # Calculate response time based on server load (Equation 1)
        base_latency = 50  # Base response time in ms
        cpu_penalty = server['cpu'] * 100  # CPU load penalty
        queue_penalty = server['queue'] * 10  # Queue length penalty
        latency = base_latency + cpu_penalty + queue_penalty

        # Update server latency (moving average)
        server['latency'] = 0.8 * server['latency'] + 0.2 * (latency / 200.0)  # Normalize for state representation

        # Natural decay for other servers (simulating processing)
        for i, s in enumerate(self.servers):
            if i != action:
                s['cpu'] = max(s['cpu'] - 0.02, 0.0)  # Gradual CPU decrease
                s['queue'] = max(s['queue'] - np.random.randint(0, 2), 0)  # Process some requests

        # Reward: latency-based (negative latency) as per paper
        reward = -latency

        self.current_step += 1
        self.total_requests += 1

        # Episode ends after max_steps
        terminated = self.current_step >= self.max_steps
        truncated = False

        return self._get_state(), reward, terminated, truncated, {"latency": latency}

    def render(self, mode='human'):
        """Render the current state."""
        print(f"Step: {self.current_step}")
        for i, server in enumerate(self.servers):
            print(f"Server {i}: CPU={server['cpu']:.2f}, Queue={server['queue']}, "
                  f"Latency={server['latency']:.2f}")
        print()

    def close(self):
        """Clean up resources."""
        pass