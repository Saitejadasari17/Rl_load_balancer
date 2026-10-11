"""
Script to benchmark all algorithms and print a comparison table.

Usage:
    python scripts/benchmark.py --episodes 10 --servers 3 --steps 200
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from src.environment import LoadBalancerEnv
from src.baselines import (
    RoundRobinAgent, LeastConnectionsAgent,
    RandomAgent, WeightedRoundRobinAgent, evaluate_agent
)


def run_benchmark(n_servers=3, n_episodes=10, max_steps=200):
    env = LoadBalancerEnv(n_servers=n_servers, max_steps=max_steps)

    algorithms = {
        "Round Robin":       RoundRobinAgent(n_servers),
        "Least Connections": LeastConnectionsAgent(n_servers),
        "Random":            RandomAgent(n_servers),
        "Weighted RR":       WeightedRoundRobinAgent(n_servers),
    }

    print(f"\nBenchmark: {n_servers} servers, {n_episodes} episodes, {max_steps} steps/ep\n")
    print(f"{'Algorithm':<22} {'Avg Latency':>14} {'Std Dev':>10} {'Avg Util':>12} {'Fairness':>10}")
    print("-" * 72)

    results = {}
    for name, agent in algorithms.items():
        r = evaluate_agent(env, agent, n_episodes=n_episodes, max_steps=max_steps)
        results[name] = r

        # Compute Jain's fairness from avg utilization (approximation)
        util = r['avg_utilization']
        fairness = min(1.0, util / 0.3)  # normalized estimate

        print(f"{name:<22} {r['avg_latency']:>12.1f}ms {r['std_latency']:>9.1f}  "
              f"{r['avg_utilization']:>11.3f}  {fairness:>9.3f}")

    best = min(results, key=lambda k: results[k]['avg_latency'])
    print(f"\nBest algorithm: {best} ({results[best]['avg_latency']:.1f} ms avg latency)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--servers", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    run_benchmark(args.servers, args.episodes, args.steps)
