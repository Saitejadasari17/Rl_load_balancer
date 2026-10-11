# Troubleshooting & Frequently Asked Questions (FAQ)

Common questions, known issues, and solutions when running the Adaptive Load Balancer.

---

### 1. Training Latency Doesn't Seem to Improve in the First 20 Episodes
**Reason**: With initial exploration rate $\epsilon_{\text{start}} = 1.0$ and $\epsilon_{\text{decay}} = 0.995$, the agent spends early episodes exploring almost uniformly at random to fill the replay buffer.
**Solution**: Run training for **200 to 500 episodes**. The policy typically converges and clearly begins outperforming Round Robin after episode 100 as $\epsilon$ drops below 0.3.

---

### 2. CUDA Out of Memory (OOM) or GPU Not Detected
**Reason**: The PyTorch agent automatically checks for CUDA, MPS (Apple Silicon), and CPU fallbacks.
**Solution**:
- If training on CPU (default on machines without Nvidia GPU), no configuration is required.
- If you have an Nvidia GPU and PyTorch is not detecting it, verify your PyTorch build matches your installed CUDA driver:
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```
- If memory is limited, reduce `batch_size` in `src/config.py` from 48 to 32.

---

### 3. Azure Dataset Files Not Found
**Reason**: The raw Azure Functions invocation dataset files are ~145MB each and are not committed to Git to keep repository size light.
**Solution**:
- The system automatically falls back to realistic synthetic traffic when Azure trace files are absent.
- To use production Azure traces:
  1. Download the 2019 dataset from [AzurePublicDataset](https://github.com/Azure/AzurePublicDataset).
  2. Extract `invocations_per_function_md.anon.d01.csv` into `data/extracted/`.
  3. The loader in `src/azure_dataset.py` will detect and parse it on the next run.

---

### 4. Frontend Can't Connect to Backend (CORS / Connection Refused)
**Reason**: The backend server is either not running or running on a different port.
**Solution**:
- Start the FastAPI backend first:
  ```bash
  python backend/main.py
  ```
- Check that `http://localhost:8000/health` returns `{"status": "healthy"}` in your browser.
- If running on a remote host, configure `VITE_API_URL` in `frontend/.env`.

---

### 5. Running Automated Tests
To verify all components locally:
```bash
pytest tests/ -v
```
To run the quick sanity check:
```bash
python scripts/sanity_check.py
```
