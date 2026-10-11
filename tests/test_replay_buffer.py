"""
Unit tests for the Prioritized Experience Replay (PER) buffer.
"""

import numpy as np
from src.agent import PrioritizedReplayBuffer


def test_buffer_initialization():
    buf = PrioritizedReplayBuffer(capacity=100, alpha=0.6)
    assert len(buf) == 0
    assert buf.capacity == 100
    assert buf.alpha == 0.6


def test_buffer_push_and_length():
    buf = PrioritizedReplayBuffer(capacity=10)
    s = np.zeros(9, dtype=np.float32)
    for i in range(5):
        buf.push(s, i % 3, -100.0, s, False)
    assert len(buf) == 5


def test_buffer_capacity_rollover():
    buf = PrioritizedReplayBuffer(capacity=5)
    s = np.zeros(9, dtype=np.float32)
    for i in range(12):
        buf.push(s, i % 3, float(i), s, False)
    assert len(buf) == 5


def test_buffer_sampling_shapes_and_types():
    buf = PrioritizedReplayBuffer(capacity=50, alpha=0.6)
    s = np.ones(9, dtype=np.float32)

    for i in range(20):
        buf.push(s * i, i % 3, -float(i * 10), s * (i + 1), i == 19)

    batch_size = 8
    states, actions, rewards, next_states, dones, indices, is_weights = buf.sample(batch_size, beta=0.4)

    assert states.shape == (batch_size, 9)
    assert actions.shape == (batch_size,)
    assert rewards.shape == (batch_size,)
    assert next_states.shape == (batch_size, 9)
    assert dones.shape == (batch_size,)
    assert len(indices) == batch_size
    assert len(is_weights) == batch_size
    assert np.all(is_weights >= 0.0)
    assert np.all(is_weights <= 1.0)


def test_buffer_priority_update():
    buf = PrioritizedReplayBuffer(capacity=20)
    s = np.zeros(9, dtype=np.float32)
    for i in range(10):
        buf.push(s, 0, -1.0, s, False)

    _, _, _, _, _, indices, _ = buf.sample(4, beta=0.4)
    fresh_td_errors = np.array([2.5, 0.1, 4.0, 1.2])
    buf.update_priorities(indices, fresh_td_errors)

    for idx, err in zip(indices, fresh_td_errors):
        assert np.isclose(buf.priorities[idx], err + 1e-5)
