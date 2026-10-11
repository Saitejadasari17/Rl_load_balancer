# Changelog

All notable changes to this project are documented here.

## [1.1.0] - 2024-10-11

### Added
- Dueling Double DQN with Prioritized Experience Replay
- Sticky exploration to prevent policy collapse
- Deterministic evaluation-based checkpointing
- Azure Functions dataset loader (multi-file support)
- FastAPI REST backend
- React + Vite frontend dashboard
- GitHub Actions badges in README

### Changed
- Replaced PPO with custom DQN implementation
- Aligned state space with paper (3 features per server)
- Fixed early stopping counter logic
- Removed infinite polling loop in frontend

### Fixed
- NameError in train_rl_agent after early stopping
- Baseline agent state indexing for new feature count
- Frontend polling infinite loop on completed training

## [1.0.0] - 2024-10-07

### Added
- Initial DQN agent with PyTorch
- Custom Gymnasium environment
- Round Robin, Least Connections, Random, Weighted RR baselines
- Jain's Fairness Index evaluation
- Basic React frontend
