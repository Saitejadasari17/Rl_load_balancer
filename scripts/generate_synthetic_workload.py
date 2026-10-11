"""
Synthetic Workload Generator
Generates realistic traffic traces (bursty, diurnal, sinusoidal)
and saves them to CSV or NumPy format for load balancer simulation testing.

Usage:
    python scripts/generate_synthetic_workload.py --pattern bursty --steps 1440 --output data/synthetic_trace.csv
"""

import argparse
import os
import numpy as np
import pandas as pd


def generate_trace(pattern="bursty", steps=1440, min_load=0.05, max_load=0.3):
    t = np.arange(steps)

    if pattern == "sinusoidal":
        # Simulates standard diurnal cycle (e.g. 24h peak and trough)
        cycle = 0.5 * (1 + np.sin(2 * np.pi * t / 1440))
        noise = np.random.normal(0, 0.05, size=steps)
        raw = min_load + (max_load - min_load) * cycle + noise

    elif pattern == "bursty":
        # Base steady load with occasional intense spikes
        base = np.random.uniform(min_load, min_load + 0.08, size=steps)
        spikes = (np.random.random(size=steps) < 0.03).astype(float) * np.random.uniform(0.1, 0.2, size=steps)
        raw = base + spikes

    elif pattern == "step":
        # Step changes in load intensity
        raw = np.full(steps, min_load + 0.05)
        raw[steps // 4 : steps // 2] = max_load * 0.8
        raw[3 * steps // 4 :] = max_load

    else:
        # Uniform random default
        raw = np.random.uniform(min_load, max_load, size=steps)

    clipped = np.clip(raw, min_load, max_load)
    return clipped


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic workload traces")
    parser.add_argument("--pattern", choices=["bursty", "sinusoidal", "step", "uniform"], default="bursty")
    parser.add_argument("--steps", type=int, default=1440, help="Number of timesteps (default: 1440 mins)")
    parser.add_argument("--output", type=str, default="data/synthetic_trace.csv", help="Output path")
    args = parser.parse_args()

    trace = generate_trace(pattern=args.pattern, steps=args.steps)
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    df = pd.DataFrame({"step": np.arange(len(trace)), "load": trace})
    df.to_csv(args.output, index=False)
    print(f"Generated {len(trace)} steps of '{args.pattern}' workload -> {args.output}")


if __name__ == "__main__":
    main()
