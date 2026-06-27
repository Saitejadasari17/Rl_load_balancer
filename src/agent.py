"""
Deep Q-Network Agent for Adaptive Load Balancing
"""

import os
import numpy as np
import json

try:
    from .environment import LoadBalancerEnv
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from environment import LoadBalancerEnv

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        self.lr = lr
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, hidden_size))
        self.W3 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b3 = np.zeros((1, output_size))

    def forward(self, x):
        """Forward pass through the network."""
        x = np.asarray(x, dtype=np.float32)
        x = np.atleast_2d(x)

        # Layer 1: Linear + ReLU
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU

        # Layer 2: Linear + ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = np.maximum(0, self.z2)  # ReLU

        # Layer 3: Linear (Q-values, no activation)
        self.z3 = self.a2 @ self.W3 + self.b3

        # Cache input for backprop
        self.input = x
        return self.z3

    def backward(self, x, target):
        """
        Backpropagation using mean squared error loss.
        Computes gradients and updates weights.
        """
        x = np.asarray(x, dtype=np.float32)
        x = np.atleast_2d(x)
        target = np.asarray(target, dtype=np.float32)
        target = np.atleast_2d(target)
        batch_size = x.shape[0]

        # Forward pass (recompute activations)
        output = self.forward(x)

        # Loss gradient: d(MSE)/d(output) = 2(output - target) / batch_size
        d_output = 2.0 * (output - target) / batch_size

        # Gradient for Layer 3
        dW3 = self.a2.T @ d_output
        db3 = np.sum(d_output, axis=0, keepdims=True)

        # Backprop through Layer 3
        d_a2 = d_output @ self.W3.T

        # ReLU gradient for Layer 2
        d_z2 = d_a2 * (self.z2 > 0).astype(float)

        # Gradient for Layer 2
        dW2 = self.a1.T @ d_z2
        db2 = np.sum(d_z2, axis=0, keepdims=True)

        # Backprop through Layer 2
        d_a1 = d_z2 @ self.W2.T

        # ReLU gradient for Layer 1
        d_z1 = d_a1 * (self.z1 > 0).astype(float)

        # Gradient for Layer 1
        dW1 = self.input.T @ d_z1
        db1 = np.sum(d_z1, axis=0, keepdims=True)

        # Gradient clipping to prevent exploding gradients
        for grad in [dW1, db1, dW2, db2, dW3, db3]:
            np.clip(grad, -1.0, 1.0, out=grad)

        # Update weights (gradient descent)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

        # Return loss for monitoring
        loss = np.mean((output - target) ** 2)
        return loss

    def copy_weights_from(self, other):
        """Copy weights from another network (for target network updates)."""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()

    def save(self, path):
        """Save network weights to file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        np.savez(path,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3)

    def load(self, path):
        """Load network weights from file."""
        if not path.endswith('.npz'):
            path += '.npz'
        data = np.load(path)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.W3 = data['W3']
        self.b3 = data['b3']


# ──────────────────────────────────────────────
#  Experience Replay Buffer
# ──────────────────────────────────────────────

class ReplayBuffer:
    """
    Stores past experiences (state, action, reward, next_state, done)
    and samples random mini-batches for training.

    This breaks correlation between consecutive decisions and
    prevents catastrophic forgetting.
    """

    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Store one experience."""
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a random mini-batch."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ──────────────────────────────────────────────
#  DQN Agent
# ──────────────────────────────────────────────

class DQNAgent:
    """
    Deep Q-Network agent for load balancing:
    - Network: 192 hidden units (increased from 160 for better pattern learning)
    - Learning Rate: 0.001 (stable, proven convergence)
    - Batch Size: 48 (balanced gradient stability)
    - Epsilon Decay: 0.995 (gradual exploration→exploitation)
    - Replay Buffer: 8,000 (diverse experience sampling)
    - Target Network: Updated every 10 episodes for stable Q-values
    - Training: 2000 episodes for thorough convergence
    - Reward: Pure latency-based (naturally encodes queue via latency calculation)
    """

    def __init__(self, state_size=9, action_size=3, hidden_size=192,
                 lr=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay=0.995, buffer_size=8000, batch_size=48,
                 target_update_freq=10):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Q-Network (for decisions)
        self.q_network = NeuralNetwork(state_size, hidden_size, action_size, lr)

        # Target Network (for stable Q-value targets)
        self.target_network = NeuralNetwork(state_size, hidden_size, action_size, lr)
        self.target_network.copy_weights_from(self.q_network)

        # Experience Replay
        self.memory = ReplayBuffer(buffer_size)

        # Training stats
        self.training_losses = []
        self.episode_rewards = []

    def act(self, state, training=True):
        """
        Choose action using epsilon-greedy policy.
        - With probability ε: pick random server (explore)
        - With probability 1-ε: pick server with highest Q-value (exploit)
        """
        state = np.asarray(state, dtype=np.float32)
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)

        q_values = self.q_network.forward(state)
        return int(np.argmax(q_values[0]))

    def predict(self, state, deterministic=True):
        """Predict action (compatible interface for evaluation)."""
        state = np.asarray(state, dtype=np.float32)
        if deterministic:
            q_values = self.q_network.forward(state)
            action = int(np.argmax(q_values[0]))
        else:
            action = self.act(state, training=True)
        return action, None

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        Sample mini-batch from replay buffer and update Q-network.
        Uses the Bellman equation:
            Q_target = reward + γ * max(Q_target_network(next_state))
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        # Sample random mini-batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Current Q-values from the online network
        current_q = self.q_network.forward(states)

        # Use the online network to choose the next action and the target network to estimate its value.
        next_online_q = self.q_network.forward(next_states)
        next_actions = np.argmax(next_online_q, axis=1)
        next_target_q = self.target_network.forward(next_states)

        # Compute target Q-values using Bellman equation
        target_q = current_q.copy()
        for i in range(self.batch_size):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                target_q[i, actions[i]] = rewards[i] + self.gamma * next_target_q[i, next_actions[i]]

        # Clip targets to a stable range for learning
        target_q = np.clip(target_q, -2.0, 0.0)

        # Update Q-network via backpropagation
        loss = self.q_network.backward(states, target_q)
        self.training_losses.append(float(loss))

        return loss

    def update_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        """Copy Q-network weights to target network."""
        self.target_network.copy_weights_from(self.q_network)

    def save(self, path="models/dqn_agent"):
        """Save model weights and config."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        self.q_network.save(path + "_q")
        self.target_network.save(path + "_target")

        config = {
            'state_size': int(self.state_size),
            'action_size': int(self.action_size),
            'epsilon': float(self.epsilon),
            'gamma': float(self.gamma),
        }
        with open(path + "_config.json", 'w') as f:
            json.dump(config, f)

    def load(self, path="models/dqn_agent"):
        """Load model weights."""
        self.q_network.load(path + "_q")
        self.target_network.load(path + "_target")

        config_path = path + "_config.json"
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            self.epsilon = config.get('epsilon', self.epsilon_end)


# ──────────────────────────────────────────────
#  Training & Evaluation Functions
# ──────────────────────────────────────────────

def train_rl_agent(env, n_episodes=20, max_steps=200, model_path="models/dqn_agent",
                   progress_callback=None, **kwargs):
    """
    Train the DQN agent on the load balancing environment.

    Args:
        env: LoadBalancerEnv instance
        n_episodes: Number of training episodes
        max_steps: Steps per episode
        model_path: Where to save the trained model
        progress_callback: Optional function(episode, n_episodes, avg_reward) for progress updates

    Returns:
        agent: Trained DQNAgent
        episode_rewards: List of total rewards per episode
        episode_latencies: List of average latency per episode
    """
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=kwargs.get('hidden_size', 192),
        lr=kwargs.get('lr', 0.001),
        gamma=kwargs.get('gamma', 0.99),
        epsilon_start=kwargs.get('epsilon_start', 1.0),
        epsilon_end=kwargs.get('epsilon_end', 0.05),
        epsilon_decay=kwargs.get('epsilon_decay', 0.995),
        buffer_size=kwargs.get('buffer_size', 8000),
        batch_size=kwargs.get('batch_size', 48),
        target_update_freq=kwargs.get('target_update_freq', 10),
    )

    episode_rewards = []
    episode_latencies = []

    print(f"Starting DQN training for {n_episodes} episodes...")
    print(f"Network: {state_size} → {kwargs.get('hidden_size', 192)} → {kwargs.get('hidden_size', 192)} → {action_size}")
    print(f"Replay buffer: {agent.memory.capacity}, Batch: {agent.batch_size}")
    print()

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        ep_latencies = []

        for step in range(max_steps):
            # Choose action (epsilon-greedy)
            action = agent.act(state, training=True)

            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)

            # Store experience
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, done)

            # Train on mini-batch from replay buffer
            agent.train_step()

            total_reward += reward
            # Since reward = -latency, we can recover latency from -reward
            ep_latencies.append(-reward)
            state = next_state

            if done:
                break

        # End of episode
        episode_rewards.append(total_reward)
        avg_latency = np.mean(ep_latencies)
        episode_latencies.append(avg_latency)

        # Decay epsilon
        agent.update_epsilon()

        # Update target network periodically
        if (episode + 1) % agent.target_update_freq == 0:
            agent.update_target_network()
            print(f"  → Target network updated at episode {episode + 1}")

        # Progress reporting
        print(f"Episode {episode + 1}/{n_episodes} | "
              f"Reward: {total_reward:.1f} | "
              f"Avg Latency: {avg_latency:.1f} ms | "
              f"Epsilon: {agent.epsilon:.3f}")

        if progress_callback:
            progress_callback(episode + 1, n_episodes, total_reward)

    # Save trained model
    agent.save(model_path)
    print(f"\nModel saved to {model_path}")
    print(f"Final epsilon: {agent.epsilon:.4f}")

    return agent, episode_rewards, episode_latencies


def evaluate_rl_agent(env, agent, n_episodes=10, max_steps=200):
    """
    Evaluate the trained DQN agent.

    Returns dict with:
        avg_reward, avg_latency, avg_utilization, fairness_index,
        std_latency, std_utilization, per_server_utilization
    """
    total_rewards = []
    latencies = []
    utilizations = []
    fairness_scores = []
    all_per_server = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_latencies = []
        episode_utilizations = []
        episode_per_server = []

        for step in range(max_steps):
            # Use deterministic policy (no exploration)
            action, _ = agent.predict(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            episode_latencies.append(-reward)

            # Track per-server CPU utilizations
            server_cpus = [next_state[i * 3] for i in range(env.n_servers)]
            episode_utilizations.append(np.mean(server_cpus))
            episode_per_server.append(server_cpus)

            state = next_state
            if terminated or truncated:
                break

        total_rewards.append(episode_reward)
        latencies.append(np.mean(episode_latencies))
        utilizations.append(np.mean(episode_utilizations))

        # Jain's Fairness Index on final server utilizations
        final_utils = [state[i * 3] for i in range(env.n_servers)]
        if sum(final_utils) > 0:
            fairness = sum(final_utils) ** 2 / (len(final_utils) * sum(u ** 2 for u in final_utils))
        else:
            fairness = 0
        fairness_scores.append(fairness)

        # Average per-server utilization across the episode
        per_server_avg = np.mean(episode_per_server, axis=0).tolist()
        all_per_server.append(per_server_avg)

    return {
        'avg_reward': float(np.mean(total_rewards)),
        'avg_latency': float(np.mean(latencies)),
        'avg_utilization': float(np.mean(utilizations)),
        'fairness_index': float(np.mean(fairness_scores)),
        'std_latency': float(np.std(latencies)),
        'std_utilization': float(np.std(utilizations)),
        'per_server_utilization': np.mean(all_per_server, axis=0).tolist(),
        'p99_latency': float(np.percentile([l for ep in latencies for l in [ep]], 99)),
    }


def load_rl_agent(model_path="models/dqn_agent", state_size=9, action_size=3):
    """Load a trained DQN agent from disk."""
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    agent.load(model_path)
    print(f"DQN agent loaded from {model_path}")
    return agent