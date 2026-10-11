"""
Unit tests for baseline load balancing algorithms (Round Robin, Least Connections, Random, Weighted RR).
"""

import numpy as np
from src.environment import LoadBalancerEnv
from src.baselines import (
    RoundRobinAgent,
    LeastConnectionsAgent,
    RandomAgent,
    WeightedRoundRobinAgent,
    evaluate_agent
)


def test_round_robin_cycles_correctly():
    n_servers = 3
    agent = RoundRobinAgent(n_servers)
    state = np.zeros(9, dtype=np.float32)

    actions = [agent.act(state) for _ in range(6)]
    assert actions == [0, 1, 2, 0, 1, 2]


def test_least_connections_selects_minimum_queue():
    n_servers = 3
    agent = LeastConnectionsAgent(n_servers)
    # State format: [cpu, queue, latency] * 3
    # Server 0: queue=0.8, Server 1: queue=0.2, Server 2: queue=0.5
    state = np.array([
        0.5, 0.8, 0.5,  # S0
        0.5, 0.2, 0.5,  # S1
        0.5, 0.5, 0.5   # S2
    ], dtype=np.float32)

    action = agent.act(state)
    assert action == 1  # Server 1 has lowest queue


def test_random_agent_valid_range():
    n_servers = 4
    agent = RandomAgent(n_servers)
    state = np.zeros(12, dtype=np.float32)

    actions = [agent.act(state) for _ in range(100)]
    assert all(0 <= a < n_servers for a in actions)
    assert len(set(actions)) > 1  # Should hit multiple servers


def test_weighted_round_robin():
    n_servers = 3
    weights = [2, 1, 1]
    agent = WeightedRoundRobinAgent(n_servers, weights=weights)
    state = np.zeros(9, dtype=np.float32)

    actions = [agent.act(state) for _ in range(8)]
    assert all(0 <= a < n_servers for a in actions)


def test_evaluate_agent_returns_valid_metrics():
    env = LoadBalancerEnv(n_servers=3, max_steps=20)
    agent = RoundRobinAgent(3)
    results = evaluate_agent(env, agent, n_episodes=2, max_steps=20)

    assert "avg_latency" in results
    assert "avg_utilization" in results
    assert "std_latency" in results
    assert results["avg_latency"] > 0
    assert 0 <= results["avg_utilization"] <= 1.0
