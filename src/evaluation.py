"""
Evaluation module for Adaptive Load Balancing.
Compares DQN agent against baseline algorithms with publication-quality plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

try:
    from .environment import LoadBalancerEnv
    from .baselines import run_baseline_comparison
    from .agent import DQNAgent, evaluate_rl_agent, load_rl_agent
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from environment import LoadBalancerEnv
    from baselines import run_baseline_comparison
    from agent import DQNAgent, evaluate_rl_agent, load_rl_agent

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def run_full_evaluation(model_path="models/dqn_agent", n_episodes=20):
    """Run complete evaluation comparing DQN agent with baselines."""
    print("Starting full evaluation...")

    # Create environment
    env = LoadBalancerEnv(n_servers=3, max_steps=200)

    # Evaluate baselines
    print("Evaluating baseline algorithms...")
    baseline_results = run_baseline_comparison(env, n_episodes)

    # Evaluate DQN agent
    print("Evaluating DQN agent...")
    try:
        agent = load_rl_agent(model_path)
        rl_results = evaluate_rl_agent(env, agent, n_episodes)
        baseline_results['RL Agent (DQN)'] = rl_results
    except (FileNotFoundError, Exception) as e:
        print(f"Warning: DQN model not found at {model_path}: {e}")
        print("Skipping RL evaluation.")
        rl_results = None

    return baseline_results


def plot_latency_comparison(results, save_path="results/latency_comparison.png"):
    """Plot average latency comparison."""
    algorithms = list(results.keys())
    latencies = [results[alg]['avg_latency'] for alg in algorithms]
    errors = [results[alg]['std_latency'] for alg in algorithms]

    colors = ['#e74c3c', '#e67e22', '#f39c12', '#27ae60', '#3498db']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, latencies, yerr=errors, capsize=5, color=colors[:len(algorithms)])
    plt.bar_label(bars, fmt='%.1f ms')

    plt.title('Average Response Time Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Average Latency (ms)', fontsize=12)
    plt.xlabel('Algorithm', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_utilization_fairness(results, save_path="results/utilization_fairness.png"):
    """Plot server utilization and fairness comparison."""
    algorithms = list(results.keys())
    utilizations = [results[alg]['avg_utilization'] for alg in algorithms]
    fairness_scores = [results[alg].get('fairness_index', 0) for alg in algorithms]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors = ['#e74c3c', '#e67e22', '#f39c12', '#27ae60', '#3498db']

    # Utilization plot
    bars1 = ax1.bar(algorithms, utilizations, color=colors[:len(algorithms)])
    ax1.bar_label(bars1, fmt='%.3f')
    ax1.set_title('Average Server Utilization', fontweight='bold')
    ax1.set_ylabel('Average CPU Utilization')
    ax1.set_xlabel('Algorithm')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    # Fairness plot
    bars2 = ax2.bar(algorithms, fairness_scores, color=colors[:len(algorithms)])
    ax2.bar_label(bars2, fmt='%.3f')
    ax2.set_title("Server Utilization Fairness (Jain's Index)", fontweight='bold')
    ax2.set_ylabel('Fairness Index (0-1)')
    ax2.set_xlabel('Algorithm')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_learning_curve(episode_rewards, episode_latencies=None,
                        save_path="results/learning_curve.png"):
    """Plot the DQN learning curve from training."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    episodes = range(1, len(episode_rewards) + 1)

    # Reward curve
    axes[0].plot(episodes, episode_rewards, color='#3498db', linewidth=2, marker='o', markersize=4)
    axes[0].set_title('Training Reward per Episode', fontweight='bold')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].grid(True, alpha=0.3)

    # Latency curve
    if episode_latencies:
        axes[1].plot(episodes, episode_latencies, color='#e74c3c', linewidth=2, marker='o', markersize=4)
        axes[1].set_title('Average Latency per Episode', fontweight='bold')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Average Latency (ms)')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_summary_report(results, save_path="results/summary_report.txt"):
    """Generate a text summary of results."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        f.write("ADAPTIVE LOAD BALANCING EVALUATION REPORT\n")
        f.write("=" * 50 + "\n")
        f.write("Agent: Deep Q-Network (DQN) — PyTorch implementation\n")
        f.write("        (Double DQN + Dueling architecture + Prioritized Experience Replay)\n\n")

        f.write("ALGORITHMS COMPARED:\n")
        for alg in results.keys():
            f.write(f"- {alg}\n")
        f.write("\n")

        f.write("PERFORMANCE METRICS:\n\n")

        header = "| Algorithm | Avg Latency (ms) | Std Dev | Avg Utilization | Fairness Index |\n"
        separator = "|-----------|------------------|---------|-----------------|----------------|\n"
        f.write(header)
        f.write(separator)

        for alg, metrics in results.items():
            latency = metrics['avg_latency']
            std_lat = metrics['std_latency']
            util = metrics['avg_utilization']
            fairness = metrics.get('fairness_index', 'N/A')
            if isinstance(fairness, float):
                fairness_str = f"{fairness:.3f}"
            else:
                fairness_str = str(fairness)
            f.write(f"| {alg:<10} | {latency:>15.2f} | {std_lat:>7.2f} | {util:>15.3f} | {fairness_str:>14} |\n")

        f.write("\n")

        # Find best performers
        best_latency = min(results.items(), key=lambda x: x[1]['avg_latency'])
        best_fairness = max(results.items(), key=lambda x: x[1].get('fairness_index', 0))

        f.write("KEY FINDINGS:\n")
        f.write(f"- Best latency: {best_latency[0]} ({best_latency[1]['avg_latency']:.2f} ms)\n")
        f.write(f"- Best fairness: {best_fairness[0]} (Jain's Index: {best_fairness[1].get('fairness_index', 'N/A')})\n")

        if 'RL Agent (DQN)' in results:
            rl_latency = results['RL Agent (DQN)']['avg_latency']
            rr_latency = results.get('Round Robin', {}).get('avg_latency', rl_latency)
            if rr_latency > rl_latency:
                improvement = ((rr_latency - rl_latency) / rr_latency) * 100
                f.write(f"- DQN improvement over Round Robin: {improvement:.1f}%\n")

    print(f"Summary report saved to {save_path}")


def run_all_evaluations():
    """Run all evaluations and generate plots."""
    results = run_full_evaluation()

    plot_latency_comparison(results)
    plot_utilization_fairness(results)

    generate_summary_report(results)

    print("All evaluations completed!")
    return results