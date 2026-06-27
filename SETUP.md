# Running the Project Locally

## Prerequisites
- Python 3.12+ 
- Node.js 18+
- Git

## Setup Backend

1. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

2. **Start the FastAPI backend** (Terminal 1)
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

3. **Test the backend** (Terminal 2)
```bash
curl http://127.0.0.1:8000/
```

Should return:
```json
{"status":"online","service":"RL Load Balancer API","version":"1.0.0"}
```

## Setup Frontend

1. **Install dependencies** (Terminal 3)
```bash
cd frontend
npm install
```

2. **Start development server**
```bash
npm run dev
```

Expected output:
```
  ➜  Local:   http://localhost:5173/
```

3. **Open in browser**
- Visit http://localhost:5173/
- Upload a CSV dataset
- Click "Start Training"
- View results and metrics

## Project Structure

```
.
├── backend/
│   ├── main.py           # FastAPI server
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx       # Main React component
│   │   └── components/
│   ├── vite.config.js
│   └── package.json
├── src/
│   ├── agent.py          # DQN agent
│   ├── environment.py    # RL environment
│   ├── baselines.py      # Baseline algorithms
│   └── ...
└── requirements.txt      # Python dependencies
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Health check |
| `/health` | GET | Detailed status |
| `/training-status` | GET | Get training progress |
| `/upload-dataset` | POST | Upload CSV file |
| `/train` | POST | Start RL training |
| `/results` | GET | Get training results |
| `/metrics` | GET | Get performance metrics |

## Sample Dataset Format

CSV with columns:
- `arrival_time`: Request arrival time
- `cpu_time`: CPU time required
- `region`: Geographic region

Example:
```
arrival_time,cpu_time,region
0.1,0.5,us-east
0.2,0.3,us-west
```

## Troubleshooting

**Backend fails to start:**
- Check if port 8000 is in use: `netstat -ano | findstr :8000`
- Ensure Python dependencies are installed
- Verify Python version: `python --version`

**Frontend won't build:**
- Clear node_modules: `rm -r node_modules && npm install`
- Clear cache: `npm cache clean --force`
- Check Node version: `node --version`

**Training errors:**
- Verify CSV format
- Check that data rows exist
- Review backend logs

## Deployment

**Render (Backend):** Already configured in render.yaml
**Vercel (Frontend):** Already configured in vercel.json

See README.md for deployment URLs.
