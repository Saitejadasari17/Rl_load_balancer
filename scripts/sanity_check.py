"""
Quick sanity check script — verifies the environment, agent, and baselines
can all be imported and run a single step without errors.

Run from the project root:
    python scripts/sanity_check.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from src.environment import LoadBalancerEnv
from src.baselines import RoundRobinAgent, LeastConnectionsAgent, RandomAgent
from src.agent import DQNAgent


def check_environment():
    print("Checking environment...", end=" ")
    env = LoadBalancerEnv(n_servers=3, max_steps=10)
    state, _ = env.reset()
    assert state.shape == (9,), f"Expected shape (9,), got {state.shape}"
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    assert next_state.shape == (9,)
    assert isinstance(reward, float)
    assert "latency" in info
    print("OK")


def check_baselines():
    print("Checking baselines...", end=" ")
    env = LoadBalancerEnv(n_servers=3, max_steps=10)
    agents = [
        RoundRobinAgent(3),
        LeastConnectionsAgent(3),
        RandomAgent(3),
    ]
    state, _ = env.reset()
    for agent in agents:
        action = agent.act(state)
        assert 0 <= action < 3
    print("OK")


def check_agent():
    print("Checking DQN agent...", end=" ")
    agent = DQNAgent(state_size=9, action_size=3, hidden_size=64, buffer_size=100, batch_size=4)
    state = np.zeros(9, dtype=np.float32)
    action = agent.act(state, training=False)
    assert 0 <= action < 3

    # Fill buffer and do one train step
    for _ in range(10):
        ns = np.random.rand(9).astype(np.float32)
        agent.remember(state, action, -100.0, ns, False)
        state = ns
    loss = agent.train_step()
    assert loss >= 0.0
    print("OK")


if __name__ == "__main__":
    print("Running sanity checks...\n")
    check_environment()
    check_baselines()
    check_agent()
    print("\nAll checks passed!")
