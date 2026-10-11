"""
Model Inspection Tool
Inspects a saved PyTorch DQN checkpoint, printing parameter count,
layer shapes, and configuration metadata.

Usage:
    python scripts/inspect_model.py --model models/rl_agent_demo.pt
"""

import argparse
import os
import torch


def inspect_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"Error: checkpoint file '{checkpoint_path}' does not exist.")
        return

    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        print(f"Keys present in checkpoint: {list(checkpoint.keys())}")

        q_state = checkpoint.get("q_network", checkpoint)
        total_params = 0

        print("\n--- Q-Network Parameter Breakdown ---")
        for name, param in q_state.items():
            param_count = param.numel()
            total_params += param_count
            print(f"  {name:<40} {str(list(param.shape)):<20} {param_count:>10,} params")

        print("-" * 74)
        print(f"Total Trainable Parameters: {total_params:,}\n")

    else:
        print("Checkpoint is raw model or unrecognized dictionary format.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a trained model checkpoint")
    parser.add_argument("--model", type=str, default="models/rl_agent_demo.pt", help="Path to .pt checkpoint")
    args = parser.parse_args()
    inspect_model(args.model)
