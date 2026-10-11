# System Architecture

This document describes the technical architecture of the Adaptive Load Balancer using Dueling Double Deep Q-Networks (DQN) with Prioritized Experience Replay (PER).

---

## 1. System Overview

```
                      +-----------------------------+
                      |   Incoming Client Request   |
                      +--------------+--------------+
                                     |
                                     v
                       [ Observe Cluster State s_t ]
                                     |
                 +-------------------+-------------------+
                 |                                       |
                 v                                       v
    +-------------------------+             +-------------------------+
    |   Feature Extraction    |             |   Heuristic Baselines   |
    |  (CPU, Queue, Latency)  |             | (RR, LeastConn, Random) |
    +------------+------------+             +-------------------------+
                 |
                 v
    +-------------------------+
    |   Dueling Q-Network     |
    |  V(s) + A(s, a) streams |
    +------------+------------+
                 |
                 v
      [ Select Server a_t ]
                 |
                 v
    +-------------------------+
    | Simulated / Live Cluster|
    | Server a_t receives job |
    +------------+------------+
                 |
                 v
    [ Instantaneous Delay L_t ]
                 |
                 v
      [ Reward r_t = -L_t ]
                 |
                 v
    +-------------------------+
    | Prioritized Buffer (PER)|
    | TD-error weight update  |
    +-------------------------+
```

---

## 2. Markov Decision Process (MDP)

### State Space $S \in \mathbb{R}^{3N}$
For $N$ backend servers, the state vector captures:
1. **CPU Utilization** $c_i \in [0, 1]$
2. **Normalized Queue Occupancy** $q_i = \min(Q_i / Q_{\max}, 1.0)$
3. **Smoothed Latency Estimate** $\ell_i \in [0, 1]$ (EMA normalized)

For $N = 3$, state dimension is $3 \times 3 = 9$.

### Action Space $A$
Discrete routing choice $a_t \in \{0, 1, \dots, N-1\}$.

### Reward Function $r_t$
$$r_t = -L_t$$
where $L_t = L_{\text{base}} + 100 \cdot c_{a_t} + 10 \cdot Q_{a_t}$ (with $L_{\text{base}} = 50\text{ ms}$).

Maximizing cumulative discounted return directly minimizes total system response latency.

---

## 3. Neural Network Architecture

- **Shared Trunk**: 2 fully connected layers (128 units, ReLU).
- **State-Value Stream $V(s)$**: Linear(128, 64) $\to$ ReLU $\to$ Linear(64, 1).
- **Advantage Stream $A(s, a)$**: Linear(128, 64) $\to$ ReLU $\to$ Linear(64, $|A|$).
- **Aggregation**:
  $$Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|A|} \sum_{a'} A(s, a') \right)$$

---

## 4. Training Stability Enhancements

1. **Double DQN Target Decoupling**: Selects next greedy action using online network, evaluates Q-value using target network.
2. **Prioritized Experience Replay**: Transitions sampled according to $| \delta_i |^\alpha$; importance sampling weights $w_i = (N \cdot P(i))^{-\beta}$ correct non-uniform sampling bias.
3. **Sticky Exploration**: In $\epsilon$-greedy exploration, an exploratory decision commits for a burst of $L \sim \text{Uniform}(5, 15)$ steps to prevent policy collapse into a single server.
4. **Periodic Deterministic Checkpointing**: Checkpoints evaluated deterministically ($\epsilon = 0$) every 10 episodes to prevent noisy exploration rewards from misleading model selection.
