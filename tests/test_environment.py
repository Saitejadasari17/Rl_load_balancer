"""
Unit tests for the LoadBalancerEnv Gymnasium environment.
"""

import pytest
import numpy as np
from src.environment import LoadBalancerEnv


def test_env_initialization():
    env = LoadBalancerEnv(n_servers=3, max_queue=20, max_steps=100)
    assert env.n_servers == 3
    assert env.max_queue == 20
    assert env.max_steps == 100
    assert env.action_space.n == 3
    assert env.observation_space.shape == (9,)


def test_env_reset():
    env = LoadBalancerEnv(n_servers=3)
    state, info = env.reset()
    assert isinstance(state, np.ndarray)
    assert state.shape == (9,)
    assert state.dtype == np.float32
    assert isinstance(info, dict)
    # At start all servers should be idle
    assert np.all(state == 0.0)


def test_env_step_valid_action():
    env = LoadBalancerEnv(n_servers=3, max_steps=50)
    env.reset()
    next_state, reward, terminated, truncated, info = env.step(0)

    assert next_state.shape == (9,)
    assert isinstance(reward, (float, np.floating))
    assert reward <= 0.0  # Reward is negative latency
    assert not terminated
    assert not truncated
    assert "latency" in info
    assert info["latency"] >= 50.0  # Base latency is 50


def test_env_step_invalid_action():
    env = LoadBalancerEnv(n_servers=3)
    env.reset()
    with pytest.raises(ValueError):
        env.step(5)  # only actions 0, 1, 2 are valid


def test_env_episode_termination():
    max_steps = 10
    env = LoadBalancerEnv(n_servers=3, max_steps=max_steps)
    env.reset()
    done = False
    step_count = 0
    for _ in range(max_steps + 5):
        _, _, terminated, truncated, _ = env.step(0)
        step_count += 1
        if terminated or truncated:
            done = True
            break

    assert done
    assert step_count == max_steps
