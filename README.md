# Adaptive Load Balancing Using Reinforcement Learning

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![License](https://img.shields.io/github/license/Saitejadasari17/rl-load-balancer.svg)
![Last Commit](https://img.shields.io/github/last-commit/Saitejadasari17/rl-load-balancer.svg)
![Repo Size](https://img.shields.io/github/repo-size/Saitejadasari17/rl-load-balancer.svg)

A capstone project implementing an intelligent load balancer using Deep Reinforcement Learning to optimize request routing in distributed systems.

## Features

- Custom Simulation Environment: Distributed server cluster simulation with a 6-dimensional
  per-server state (CPU utilization, queue length, latency, memory utilization, current
  workload, request distribution)
- RL Agent: PyTorch Double DQN + Dueling architecture + Prioritized Experience Replay,
  trained with soft target updates, AdamW, Huber loss, gradient clipping, LR scheduling,
  mixed precision, checkpointing/early stopping, and TensorBoard logging
- Baselines: Round Robin, Least Connections, Random, Weighted Round Robin
- Evaluation: Latency, P99 tail latency, throughput, utilization, Jain's Fairness Index
- Visualization: Performance comparisons and learning curves

```
Incoming Request → Observe Cluster State → Deep Q-Network → Select Best Server
       ↓
Execute Routing Decision → Receive Feedback (reward = -latency) → Learn Better Policy
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd adaptive-load-balancing-rl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the RL Agent
```bash
python main.py --mode train --timesteps 100000
```

### Evaluating Performance
```bash
python main.py --mode evaluate --episodes 20
```

### Full Comparison (Train + Evaluate + Compare)
```bash
python main.py --mode all
```

### Custom Configuration
```bash
python main.py --servers 5 --timesteps 50000 --episodes 10
```

## Project Structure

```
├── src/
│   ├── environment.py      # Custom Gym environment
│   ├── agent.py           # RL training and evaluation
│   ├── baselines.py       # Baseline algorithms
│   ├── evaluation.py      # Comparison and plotting
│   └── config.py          # Configuration parameters
├── models/                # Saved RL models
├── results/               # Generated plots and reports
├── notebooks/             # Analysis notebooks
├── logs/                  # Training logs
├── main.py               # Main execution script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Results

The RL agent typically achieves:
- **30-40% lower average latency** compared to Round Robin
- **Better server utilization fairness** (Jain's Fairness Index > 0.95)
- **Robust performance** under bursty and heterogeneous traffic patterns

## Key Metrics

- **Average Response Time**: Primary optimization target
- **Server Utilization**: CPU usage distribution across servers
- **Fairness Index**: How evenly load is distributed (Jain's formula)
- **Throughput**: Requests processed per unit time

## Technologies Used

- **Python 3.10+**
- **PyTorch**: Custom Double DQN + Dueling Network agent, trained with AdamW, Huber loss,
  gradient clipping, LR scheduling, and automatic mixed precision (CUDA/MPS/CPU auto-detect)
- **Gymnasium**: Environment interface
- **NumPy / Pandas**: Numerical computations and Azure trace processing
- **Matplotlib/Seaborn**: Data visualization
- **TensorBoard**: Training/loss/reward monitoring

## Research Contributions

1. **Novel RL Environment**: Custom Gym environment for load balancing simulation
2. **Adaptive Routing Policy**: Learned policy that adapts to traffic patterns
3. **Comprehensive Baselines**: Thorough comparison with existing algorithms
4. **Bursty Traffic Handling**: Evaluation under realistic traffic conditions

## Future Work

- Multi-objective optimization (latency + fairness + energy)
- Real-world deployment integration
- Transfer learning across different cluster configurations
- Multi-agent coordination for large-scale systems

## References

- Mao et al. "Resource Management with Deep Reinforcement Learning" (HotNets 2016)
- Schulman et al. "Proximal Policy Optimization Algorithms" (arXiv 2017)
- Mnih et al. "Human-level control through deep RL" (Nature 2015)

## License

This project is part of a capstone thesis. Please cite appropriately if used for research.