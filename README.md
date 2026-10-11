# Adaptive Load Balancing Using Reinforcement Learning

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-green.svg)
![License](https://img.shields.io/github/license/Saitejadasari17/Rl_load_balancer.svg)
![Last Commit](https://img.shields.io/github/last-commit/Saitejadasari17/Rl_load_balancer.svg)
![Repo Size](https://img.shields.io/github/repo-size/Saitejadasari17/Rl_load_balancer.svg)
![Issues](https://img.shields.io/github/issues/Saitejadasari17/Rl_load_balancer.svg)
![Stars](https://img.shields.io/github/stars/Saitejadasari17/Rl_load_balancer.svg?style=social)

A capstone project that builds an intelligent load balancer using Deep Reinforcement Learning. The system learns to route incoming requests across servers to minimize latency and improve fairness — outperforming traditional static algorithms like Round Robin and Least Connections.

---

## How it works

```
Incoming Request → Observe Cluster State → Dueling Double DQN → Select Best Server
       ↓
Execute Routing → Receive Reward (-latency) → Update Q-Network → Better Policy
```

The agent observes CPU utilization, queue depth, and response latency per server, then selects the optimal target using a trained neural network.

---

## Features

- **Custom Gym Environment** — Simulates a 3-server cluster with realistic CPU/queue/latency dynamics and Azure Functions traffic traces
- **Dueling Double DQN** — PyTorch implementation with Prioritized Experience Replay, soft target network updates (Polyak averaging), AdamW optimizer, Huber loss, and gradient clipping
- **Sticky Exploration** — Prevents policy collapse during training; agent commits to a server for multiple steps before re-evaluating
- **Checkpointing + Early Stopping** — Saves best model by evaluation latency every 10 episodes
- **Baseline Comparison** — Round Robin, Least Connections, Random, Weighted Round Robin
- **Web Dashboard** — React + Vite frontend for uploading data, running training, and visualizing results
- **REST API** — FastAPI backend exposing `/train`, `/results`, `/metrics` endpoints

---

## Results

After sufficient training (300+ episodes):

| Algorithm | Avg Latency | Fairness Index |
|-----------|-------------|----------------|
| Random | ~290 ms | ~0.60 |
| Round Robin | ~250 ms | ~0.90 |
| Least Connections | ~230 ms | ~0.92 |
| Weighted RR | ~240 ms | ~0.88 |
| **DQN (ours)** | **~160 ms** | **~0.97** |

- ~30–40% lower latency vs Round Robin
- Jain's Fairness Index > 0.95

---

## Project Structure

```
├── src/
│   ├── environment.py      # Custom Gymnasium environment (state, action, reward)
│   ├── agent.py            # Dueling Double DQN + PER (PyTorch)
│   ├── baselines.py        # Round Robin, Least Connections, Random, Weighted RR
│   ├── evaluation.py       # Metrics, plots, comparison reports
│   ├── azure_dataset.py    # Microsoft Azure Functions trace loader (multi-file)
│   └── config.py           # Hyperparameters
├── backend/
│   └── main.py             # FastAPI server
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   └── components/     # MetricCards, ResultCharts, UploadSection, TrainingProgress
│   └── package.json
├── requirements.txt
└── README.md
```

---

## Installation

### Backend

```bash
git clone https://github.com/Saitejadasari17/Rl_load_balancer.git
cd Rl_load_balancer
pip install -r requirements.txt
python backend/main.py
```

Backend runs at `http://localhost:8000`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`

---

## Usage

### Quick start via web UI

1. Start backend and frontend (see above)
2. Open `http://localhost:5173`
3. Set **Training Episodes** to `300`
4. Click **Start Training**
5. Results appear automatically when done

### Command line

```bash
# Train only
python main.py --mode train --episodes 300

# Evaluate a saved model
python main.py --mode evaluate

# Full pipeline
python main.py --mode all
```

---

## Azure Dataset

This project supports the [Microsoft Azure Functions Invocation Dataset (2019)](https://github.com/Azure/AzurePublicDataset).

Place extracted CSV files in `data/extracted/`. The loader reads `invocations_per_function_md.anon.d01.csv` by default (Day 1 only).

---

## Technologies

- **Python 3.10+**
- **PyTorch** — Neural network, backprop, CUDA/MPS/CPU auto-detection
- **Gymnasium** — RL environment interface
- **FastAPI + Uvicorn** — REST API backend
- **React + Vite + TailwindCSS** — Frontend dashboard
- **Recharts** — Data visualization
- **NumPy / Pandas** — Numerical computation and data loading

---

## References

- Mnih et al. "Human-level control through deep reinforcement learning" — Nature (2015)
- Wang et al. "Dueling Network Architectures for Deep Reinforcement Learning" — ICML (2016)
- Schaul et al. "Prioritized Experience Replay" — ICLR (2016)
- Mao et al. "Resource Management with Deep Reinforcement Learning" — HotNets (2016)
- Shahid et al. "Machine Learning for Load Balancing in Cloud" — IEEE Access (2020)

---

## License

MIT License — see [LICENSE](LICENSE) file for details.