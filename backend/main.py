from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import sys
import traceback
import numpy as np
import pandas as pd
import json

try:
    from ..src.environment import LoadBalancerEnv
    from ..src.baselines import (
        RoundRobinAgent, LeastConnectionsAgent, RandomAgent,
        WeightedRoundRobinAgent, evaluate_agent
    )
    from ..src.agent import train_rl_agent, evaluate_rl_agent
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__) + '/..')
    from src.environment import LoadBalancerEnv
    from src.baselines import (
        RoundRobinAgent, LeastConnectionsAgent, RandomAgent,
        WeightedRoundRobinAgent, evaluate_agent
    )
    from src.agent import train_rl_agent, evaluate_rl_agent
app = FastAPI(title="Adaptive Load Balancing - RL System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
training_state = {
    "status": "idle",
    "progress": 0,
    "message": "Ready for training"
}

class TrainingConfig(BaseModel):
    timesteps: int = 2000  
    episodes: int = 20
    servers: int = 3

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "RL Load Balancer API",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/training-status")
async def get_status():
    return training_state

@app.post("/upload-dataset")
async def upload_csv(file: UploadFile = File(...)):
    """Upload and validate CSV"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files supported")
        
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        return {
            "status": "success",
            "filename": file.filename,
            "rows": int(len(df)),
            "columns": list(df.columns)[:10]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")

@app.post("/train")
async def start_training(config: TrainingConfig):
    """Start RL training"""
    try:
        training_state["status"] = "training"
        training_state["progress"] = 0
        training_state["message"] = "Initializing..."
        

        env = LoadBalancerEnv(n_servers=config.servers)
        training_state["progress"] = 10
        training_state["message"] = "Environment created"
        
     
        training_state["progress"] = 25
        training_state["message"] = "Evaluating baselines..."
        
        baselines = {
            'Round Robin': RoundRobinAgent(config.servers),
            'Least Connections': LeastConnectionsAgent(config.servers),
            'Random': RandomAgent(config.servers),
            'Weighted RR': WeightedRoundRobinAgent(config.servers),
        }
        
        baseline_results = {}
        for name, agent in baselines.items():
            result = evaluate_agent(env, agent, n_episodes=5, max_steps=500)
            baseline_results[name] = {
                'avg_latency': float(result['avg_latency']),
                'avg_utilization': float(result['avg_utilization']),
                'std_latency': float(result['std_latency'])
            }
        
        best_baseline = min(r['avg_latency'] for r in baseline_results.values())
        
        # Train RL
        training_state["progress"] = 50
        training_state["message"] = f"Training RL for {config.timesteps} episodes..."
        
        os.makedirs('models', exist_ok=True)
        model, training_rewards, episode_latencies = train_rl_agent(
            env=env,
            n_episodes=config.episodes,
            max_steps=200,
            model_path='models/rl_agent_demo'
        )

        # Evaluate RL
        training_state["progress"] = 85
        training_state["message"] = "Evaluating RL agent..."

        rl_results = evaluate_rl_agent(env, model, n_episodes=5, max_steps=200)
        rl_latency = rl_results['avg_latency']

        # Calculate improvement
        improvement = ((best_baseline - rl_latency) / best_baseline) * 100 if best_baseline > 0 else 0.0

        # Build learning curve (one point every episode)
        learning_curve = [
            {"episode": i + 1, "latency": float(lat), "reward": float(training_rewards[i])}
            for i, lat in enumerate(episode_latencies)
        ]

        # Prepare results
        results = {
            "status": "completed",
            "baselines": baseline_results,
            "rl_agent": {
                "avg_latency": float(rl_latency),
                "avg_utilization": float(rl_results['avg_utilization']),
                "fairness_index": float(rl_results['fairness_index'])
            },
            "best_baseline_ms": float(best_baseline),
            "rl_agent_ms": float(rl_latency),
            "improvement": float(improvement),
            "learning_curve": learning_curve,
            "per_server_utilization": rl_results.get('per_server_utilization', [])
        }

        # Save results
        os.makedirs('results', exist_ok=True)
        with open('results/api_training_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        training_state["status"] = "completed"
        training_state["progress"] = 100
        training_state["message"] = "Training complete!"

        return results
    
    except Exception as e:
        training_state["status"] = "failed"
        training_state["progress"] = 0
        training_state["message"] = f"Error: {str(e)}"
        print(f"Training error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results")
async def get_results():
    """Get last training results"""
    try:
        with open('results/api_training_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No results yet")

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    return {
        "max_servers": 10,
        "max_episodes": 100,
        "max_timesteps": 500000,
        "algorithms": ["Round Robin", "Least Connections", "Random", "Weighted RR", "DQN"],
        "rl_config": {
            "algorithm": "Double DQN + Dueling + Prioritized Experience Replay",
            "learning_rate": 0.001,
            "gamma": 0.99,
            "batch_size": 48,
            "target_update_freq": 10
        }
    }

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)