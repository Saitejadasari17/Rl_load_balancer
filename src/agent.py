"""
Deep Q-Network Agent for Adaptive Load Balancing (PyTorch implementation)

Implements the enhanced DQN framework described in the project's research
problem statement (Section 11 - "Evolution of the Learning Algorithm"):

  - PyTorch-based Deep Q-Networks for efficient neural network training
  - Automatic GPU acceleration (CUDA), with Apple MPS and CPU fallback
  - Double DQN to reduce Q-value overestimation (decoupled action
    selection / evaluation)
  - Dueling Network Architecture (state-value + advantage streams)
  - Prioritized Experience Replay (PER)
  - Soft (Polyak) target network updates for smoother convergence
  - AdamW optimization with weight decay for improved generalization
  - Huber (Smooth L1) loss for robustness against large TD errors
  - Gradient clipping for optimization stability
  - Learning-rate scheduling for improved convergence
  - Mixed precision training for faster GPU execution
  - Checkpointing and early stopping for training efficiency/recoverability
  - TensorBoard logging of rewards, losses, and training progress
  - Deterministic seeding for reproducibility
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # tensorboard is optional at runtime
    SummaryWriter = None

try:
    from .environment import LoadBalancerEnv
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from environment import LoadBalancerEnv


# ──────────────────────────────────────────────
#  Device selection & reproducibility
# ──────────────────────────────────────────────

def get_device():
    """Select CUDA > Apple MPS > CPU, in that order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed=42):
    """Seed all sources of randomness for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────
#  Dueling Q-Network
# ──────────────────────────────────────────────

class DuelingQNetwork(nn.Module):
    """
    Dueling architecture: a shared feature extractor feeding two heads —
    a scalar state-value V(s) and a per-action advantage A(s, a).

        Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))

    Separating "how good is this state" from "how much better is this
    action than the others" improves value estimation, especially when
    many actions have similar value (as with near-identical servers).
    """

    def __init__(self, state_size, hidden_size, action_size):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


# ──────────────────────────────────────────────
#  Prioritized Experience Replay
# ──────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """
    Proportional Prioritized Experience Replay (Schaul et al., 2016).

    Transitions with larger TD-error are sampled more often, focusing
    learning on the most informative experiences. Importance-sampling
    weights correct the resulting bias.
    """

    def __init__(self, capacity=8000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Store one experience with max priority so it gets sampled at least once."""
        max_priority = self.priorities[:len(self.buffer)].max() if self.buffer else 1.0
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """Sample a prioritized mini-batch and return importance-sampling weights."""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]

        scaled = priorities ** self.alpha
        probs = scaled / scaled.sum()

        indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=probs)
        batch = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])

        return states, actions, rewards, next_states, dones, indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        """Update priorities for sampled transitions using their fresh TD-errors."""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(float(td_error)) + 1e-5

    def __len__(self):
        return len(self.buffer)


# ──────────────────────────────────────────────
#  DQN Agent
# ──────────────────────────────────────────────

class DQNAgent:
    """
    PyTorch Deep Q-Network agent for adaptive load balancing.

    Combines Double DQN + Dueling architecture + Prioritized Experience
    Replay + soft (Polyak) target updates for stable, sample-efficient
    learning, trained with AdamW, Huber loss, gradient clipping, LR
    scheduling, and (on CUDA) automatic mixed precision.
    """

    def __init__(self, state_size=9, action_size=3, hidden_size=128,
                 lr=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay=0.995, buffer_size=20000, batch_size=64,
                 target_update_freq=10, tau=0.01, weight_decay=1e-5,
                 per_alpha=0.6, per_beta_start=0.4, grad_clip_norm=1.0,
                 lr_step_size=200, lr_gamma=0.9, seed=42,
                 log_dir=None, use_amp=None):

        seed_everything(seed)

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.grad_clip_norm = grad_clip_norm
        self.per_beta = per_beta_start

        self.device = get_device()

        # Online (Q) network and target network (Dueling architecture)
        self.q_network = DuelingQNetwork(state_size, hidden_size, action_size).to(self.device)
        self.target_network = DuelingQNetwork(state_size, hidden_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # AdamW optimizer + step LR scheduler
        self.optimizer = torch.optim.AdamW(
            self.q_network.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=lr_step_size, gamma=lr_gamma
        )

        # Huber loss (per-sample, so PER importance weights can be applied)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

        # Mixed precision only makes sense on CUDA
        self.use_amp = use_amp if use_amp is not None else (self.device.type == 'cuda')
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.use_amp)

        # Prioritized Experience Replay
        self.memory = PrioritizedReplayBuffer(buffer_size, alpha=per_alpha)

        # Training stats / bookkeeping
        self.training_losses = []
        self.episode_rewards = []
        self.train_step_count = 0

        self.writer = SummaryWriter(log_dir) if (log_dir and SummaryWriter is not None) else None

    def act(self, state, training=True):
        """
        Choose action using epsilon-greedy policy with sticky exploration.
        """
        if not hasattr(self, 'sticky_burst_remaining'):
            self.sticky_burst_remaining = 0
            self.sticky_action = 0

        if training:
            if self.sticky_burst_remaining > 0:
                self.sticky_burst_remaining -= 1
                return self.sticky_action

            if np.random.random() < self.epsilon:
                self.sticky_action = np.random.randint(0, self.action_size)
                self.sticky_burst_remaining = np.random.randint(5, 16) - 1
                return self.sticky_action

        state_t = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_t)
        self.q_network.train()
        return int(torch.argmax(q_values, dim=1).item())

    def predict(self, state, deterministic=True):
        """Predict action (compatible interface for evaluation/baselines code)."""
        if deterministic:
            state_t = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state_t)
            action = int(torch.argmax(q_values, dim=1).item())
        else:
            action = self.act(state, training=True)
        return action, None

    def remember(self, state, action, reward, next_state, done):
        """Store experience in prioritized replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        Sample a prioritized mini-batch and update the online Q-network.

        Double DQN target:
            a* = argmax_a Q_online(s', a)
            y  = r + gamma * Q_target(s', a*)
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones, indices, is_weights = self.memory.sample(
            self.batch_size, beta=self.per_beta
        )

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        weights_t = torch.as_tensor(is_weights, dtype=torch.float32, device=self.device)

        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                # Double DQN: online network selects the action, target network evaluates it
                next_online_q = self.q_network(next_states_t)
                next_actions = torch.argmax(next_online_q, dim=1)
                next_target_q = self.target_network(next_states_t).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
                target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_target_q
                # Clip targets to a stable range for learning
                target_q = torch.clamp(target_q, -2.0, 0.0)

            elementwise_loss = self.loss_fn(current_q, target_q)
            loss = (elementwise_loss * weights_t).mean()

        td_errors = (current_q - target_q).detach()

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Refresh priorities based on the fresh TD-errors
        self.memory.update_priorities(indices, td_errors.cpu().numpy())

        self.train_step_count += 1
        loss_value = float(loss.item())
        self.training_losses.append(loss_value)

        if self.writer is not None:
            self.writer.add_scalar('train/loss', loss_value, self.train_step_count)
            self.writer.add_scalar('train/epsilon', self.epsilon, self.train_step_count)
            self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.train_step_count)

        # Soft-update target network every step for smoother convergence
        self._soft_update_target()

        return loss_value

    def _soft_update_target(self):
        """Polyak averaging: target = tau * online + (1 - tau) * target."""
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_network.parameters(), self.q_network.parameters()
            ):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * online_param.data)

    def update_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_per_beta(self, episode, total_episodes, beta_end=1.0):
        """Anneal PER importance-sampling beta towards 1.0 over training."""
        progress = min(episode / max(total_episodes, 1), 1.0)
        self.per_beta = self.per_beta + progress * (beta_end - self.per_beta)

    def update_target_network(self):
        """Hard sync of target network (kept for periodic full-copy compatibility)."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def step_scheduler(self):
        """Advance the learning-rate scheduler by one step."""
        self.scheduler.step()

    def save(self, path="models/dqn_agent"):
        """Save model weights, optimizer state, and config as a checkpoint."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path + ".pt")

        config = {
            'state_size': int(self.state_size),
            'action_size': int(self.action_size),
            'epsilon': float(self.epsilon),
            'gamma': float(self.gamma),
        }
        with open(path + "_config.json", 'w') as f:
            json.dump(config, f)

    def load(self, path="models/dqn_agent"):
        """Load model weights (and optimizer state, if present) from a checkpoint."""
        checkpoint_path = path if path.endswith(".pt") else path + ".pt"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        if 'optimizer' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except ValueError:
                pass  # optimizer shape mismatch (e.g. different hidden size) — skip

        config_path = (path[:-3] if path.endswith(".pt") else path) + "_config.json"
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            self.epsilon = config.get('epsilon', self.epsilon_end)


# ──────────────────────────────────────────────
#  Training & Evaluation Functions
# ──────────────────────────────────────────────

def train_rl_agent(env, n_episodes=20, max_steps=200, model_path="models/dqn_agent",
                    progress_callback=None, patience=40, log_dir="logs", **kwargs):
    """
    Train the DQN agent on the load balancing environment.

    Includes checkpointing (best model by episode reward is saved as we go)
    and early stopping (training halts once `patience` episodes pass without
    improvement).

    Args:
        env: LoadBalancerEnv instance
        n_episodes: Number of training episodes
        max_steps: Steps per episode
        model_path: Where to save the trained model
        progress_callback: Optional function(episode, n_episodes, avg_reward) for progress updates
        patience: Episodes without improvement before early stopping
        log_dir: TensorBoard log directory (None disables logging)

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
        tau=kwargs.get('tau', 0.01),
        weight_decay=kwargs.get('weight_decay', 1e-5),
        per_alpha=kwargs.get('per_alpha', 0.6),
        per_beta_start=kwargs.get('per_beta_start', 0.4),
        grad_clip_norm=kwargs.get('grad_clip_norm', 1.0),
        lr_step_size=kwargs.get('lr_step_size', 200),
        lr_gamma=kwargs.get('lr_gamma', 0.9),
        seed=kwargs.get('seed', 42),
        log_dir=log_dir,
    )

    episode_rewards = []
    episode_latencies = []
    best_latency = float('inf')
    episodes_without_improvement = 0
    ever_saved = False

    print(f"Training on device: {agent.device}")
    print(f"Starting DQN training for {n_episodes} episodes...")
    print(f"Network: {state_size} -> {kwargs.get('hidden_size', 192)} (dueling) -> {action_size}")
    print(f"Replay buffer: {agent.memory.capacity} (prioritized), Batch: {agent.batch_size}")
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

            # Train on a prioritized mini-batch from replay buffer
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

        # Decay epsilon, anneal PER beta, step LR scheduler
        agent.update_epsilon()
        agent.update_per_beta(episode, n_episodes)
        agent.step_scheduler()

        # Periodic hard sync on top of the continuous soft updates
        if (episode + 1) % agent.target_update_freq == 0:
            agent.update_target_network()

        if agent.writer is not None:
            agent.writer.add_scalar('episode/reward', total_reward, episode + 1)
            agent.writer.add_scalar('episode/avg_latency', avg_latency, episode + 1)

        # Progress reporting
        print(f"Episode {episode + 1}/{n_episodes} | "
              f"Reward: {total_reward:.1f} | "
              f"Avg Latency: {avg_latency:.1f} ms | "
              f"Epsilon: {agent.epsilon:.3f} | "
              f"LR: {agent.optimizer.param_groups[0]['lr']:.6f}")

        if progress_callback:
            progress_callback(episode + 1, n_episodes, total_reward)

        # Periodic deterministic evaluation for checkpointing and early stopping
        if (episode + 1) % 10 == 0:
            eval_results = evaluate_rl_agent(env, agent, n_episodes=3, max_steps=max_steps)
            eval_latency = eval_results['avg_latency']

            print(f"  --> Evaluation Latency: {eval_latency:.2f} ms")

            # Save if improved by more than 0.5 ms
            if eval_latency < best_latency - 0.5:
                best_latency = eval_latency
                episodes_without_improvement = 0
                ever_saved = True
                agent.save(model_path)
                print(f"  --> New best model saved (Latency: {best_latency:.2f} ms)")
            elif ever_saved:
                # Only count stagnation after we've saved at least one checkpoint
                episodes_without_improvement += 10

            if ever_saved and episodes_without_improvement >= patience:
                print(f"\nEarly stopping at episode {episode + 1} "
                      f"(no improvement for {patience} episodes)")
                # Reload the best checkpoint found
                try:
                    agent.load(model_path)
                except Exception:
                    pass
                break

    if agent.writer is not None:
        agent.writer.close()

    best_label = f"{best_latency:.2f} ms" if best_latency < float('inf') else "N/A"
    print(f"\nBest model saved to {model_path}.pt (latency={best_label})")
    print(f"Final epsilon: {agent.epsilon:.4f}")

    return agent, episode_rewards, episode_latencies


def evaluate_rl_agent(env, agent, n_episodes=10, max_steps=200):
    """
    Evaluate the trained DQN agent.

    Returns dict with:
        avg_reward, avg_latency, avg_utilization, fairness_index,
        std_latency, std_utilization, per_server_utilization, p99_latency
    """
    features_per_server = getattr(env, 'FEATURES_PER_SERVER', 3)

    total_rewards = []
    latencies = []
    utilizations = []
    fairness_scores = []
    all_per_server = []
    all_latencies_flat = []

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
            all_latencies_flat.append(-reward)

            # Track per-server CPU utilizations
            server_cpus = [next_state[i * features_per_server] for i in range(env.n_servers)]
            episode_utilizations.append(np.mean(server_cpus))
            episode_per_server.append(server_cpus)

            state = next_state
            if terminated or truncated:
                break

        total_rewards.append(episode_reward)
        latencies.append(np.mean(episode_latencies))
        utilizations.append(np.mean(episode_utilizations))

        # Jain's Fairness Index on final server utilizations
        final_utils = [state[i * features_per_server] for i in range(env.n_servers)]
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
        'p99_latency': float(np.percentile(all_latencies_flat, 99)) if all_latencies_flat else 0.0,
    }


def load_rl_agent(model_path="models/dqn_agent", state_size=9, action_size=3):
    """Load a trained DQN agent from disk."""
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    agent.load(model_path)
    print(f"DQN agent loaded from {model_path} (device: {agent.device})")
    return agent
