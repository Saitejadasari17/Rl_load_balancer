import numpy as np
import torch

from src.agent import DQNAgent, load_rl_agent
from src.environment import LoadBalancerEnv


def test_agent_uses_pytorch_and_runs_training_smoke():
    env = LoadBalancerEnv(n_servers=3, max_steps=8)
    agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, hidden_size=16, lr=0.001, batch_size=4, buffer_size=32)

    assert isinstance(agent.q_network, torch.nn.Module)
    assert isinstance(agent.target_network, torch.nn.Module)

    for _ in range(8):
        state, _ = env.reset()
        for _ in range(4):
            action = agent.act(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.remember(state, action, reward, next_state, terminated or truncated)
            agent.train_step()
            state = next_state
            if terminated or truncated:
                break

    assert len(agent.training_losses) > 0


def test_load_rl_agent_returns_agent():
    agent = load_rl_agent(state_size=9, action_size=3)
    assert isinstance(agent, DQNAgent)
