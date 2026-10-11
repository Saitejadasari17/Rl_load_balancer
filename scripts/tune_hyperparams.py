"""
Hyperparameter tuning utility — tries different learning rates, batch sizes,
and hidden layer sizes and reports which combination gives lowest eval latency.

Usage:
    python scripts/tune_hyperparams.py --episodes 30 --trials 6
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from src.environment import LoadBalancerEnv
from src.agent import DQNAgent, evaluate_rl_agent


SEARCH_SPACE = [
    {"lr": 0.001, "hidden_size": 128, "batch_size": 32,  "gamma": 0.99},
    {"lr": 0.001, "hidden_size": 192, "batch_size": 48,  "gamma": 0.99},
    {"lr": 0.0005,"hidden_size": 128, "batch_size": 32,  "gamma": 0.99},
    {"lr": 0.0005,"hidden_size": 256, "batch_size": 64,  "gamma": 0.995},
    {"lr": 0.003, "hidden_size": 128, "batch_size": 32,  "gamma": 0.98},
    {"lr": 0.002, "hidden_size": 192, "batch_size": 64,  "gamma": 0.99},
]


def quick_train(env, params, n_episodes=30, max_steps=200):
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        hidden_size=params["hidden_size"],
        lr=params["lr"],
        gamma=params["gamma"],
        batch_size=params["batch_size"],
        buffer_size=5000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.99,
        log_dir=None,
    )
    for episode in range(n_episodes):
        state, _ = env.reset()
        for _ in range(max_steps):
            action = agent.act(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.remember(state, action, reward, next_state, terminated or truncated)
            agent.train_step()
            state = next_state
            if terminated or truncated:
                break
        agent.update_epsilon()

    results = evaluate_rl_agent(env, agent, n_episodes=3, max_steps=max_steps)
    return results["avg_latency"]


def main(n_episodes=30, n_trials=None):
    env = LoadBalancerEnv(n_servers=3, max_steps=200)
    trials = SEARCH_SPACE[:n_trials] if n_trials else SEARCH_SPACE

    print(f"\nHyperparameter search ({len(trials)} configs, {n_episodes} episodes each)\n")
    print(f"{'LR':>8} {'Hidden':>8} {'Batch':>7} {'Gamma':>7}  {'Eval Latency':>14}")
    print("-" * 52)

    best_latency = float("inf")
    best_params = None

    for params in trials:
        latency = quick_train(env, params, n_episodes)
        marker = " <-- best" if latency < best_latency else ""
        if latency < best_latency:
            best_latency = latency
            best_params = params
        print(f"{params['lr']:>8.4f} {params['hidden_size']:>8} {params['batch_size']:>7}"
              f" {params['gamma']:>7.3f}  {latency:>12.1f} ms{marker}")

    print(f"\nBest config: {best_params}  →  {best_latency:.1f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()
    main(args.episodes, args.trials)
