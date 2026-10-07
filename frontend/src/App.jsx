import React, { useState, useRef } from 'react';
import { Activity, TrendingDown, BarChart3, Zap } from 'lucide-react';
import UploadSection from './components/UploadSection';
import TrainingProgress from './components/TrainingProgress';
import MetricCards from './components/MetricCards';
import ResultCharts from './components/ResultCharts';

export default function App() {
  const [apiUrl] = useState(import.meta.env.VITE_API_URL || 'http://localhost:8000');
  const [trainingStatus, setTrainingStatus] = useState('idle');
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [trainConfig, setTrainConfig] = useState({
    timesteps: 50000,
    episodes: 100,
    servers: 3
  });

  // /train is synchronous — it blocks until training is done and returns results inline.
  // No polling needed; we just await the response.
  const handleStartTraining = async () => {
    setTrainingStatus('training');
    setResults(null);
    setError(null);

    try {
      const res = await fetch(`${apiUrl}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(trainConfig)
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setResults(data);
      setTrainingStatus('completed');
    } catch (err) {
      console.error('Training error:', err);
      setError(err.message);
      setTrainingStatus('idle');
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center gap-3 mb-2">
            <Zap className="w-8 h-8 text-blue-500" />
            <h1 className="text-3xl font-bold">RL Load Balancer</h1>
          </div>
          <p className="text-slate-400">Adaptive load balancing using Deep Reinforcement Learning</p>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8 space-y-8">

        {/* Upload & Config Section */}
        <section className="grid md:grid-cols-3 gap-6">
          <div className="md:col-span-2">
            <UploadSection />
          </div>
          <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5 text-green-500" />
              Configuration
            </h3>
            <div className="space-y-4">
              <div>
                <label className="text-sm text-slate-400">Training Episodes</label>
                <input
                  type="number"
                  min={10}
                  max={2000}
                  value={trainConfig.episodes}
                  onChange={e => setTrainConfig({ ...trainConfig, episodes: parseInt(e.target.value) || 100 })}
                  className="w-full mt-1 bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white"
                />
                <p className="text-xs text-slate-500 mt-1">Recommended: 200+ for meaningful learning</p>
              </div>
              <div>
                <label className="text-sm text-slate-400">Servers</label>
                <input
                  type="number"
                  min={2}
                  max={10}
                  value={trainConfig.servers}
                  onChange={e => setTrainConfig({ ...trainConfig, servers: parseInt(e.target.value) || 3 })}
                  className="w-full mt-1 bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white"
                />
              </div>
              <button
                onClick={handleStartTraining}
                disabled={trainingStatus === 'training'}
                className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 disabled:cursor-not-allowed px-4 py-2 rounded font-semibold transition"
              >
                {trainingStatus === 'training' ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
                    </svg>
                    Training... (check terminal)
                  </span>
                ) : 'Start Training'}
              </button>

              {trainingStatus === 'completed' && (
                <button
                  onClick={() => { setResults(null); setTrainingStatus('idle'); }}
                  className="w-full bg-slate-700 hover:bg-slate-600 px-4 py-2 rounded font-semibold transition text-sm"
                >
                  Reset & Train Again
                </button>
              )}
            </div>
          </div>
        </section>

        {/* Error banner */}
        {error && (
          <div className="bg-red-900/30 border border-red-700 rounded-lg p-4 text-red-300">
            <strong>Training failed:</strong> {error}
          </div>
        )}

        {/* Training in-progress notice */}
        {trainingStatus === 'training' && (
          <TrainingProgress status={trainingStatus} />
        )}

        {/* Results */}
        {results && (
          <>
            <MetricCards results={results} />
            <ResultCharts results={results} />
          </>
        )}

        {/* Empty State */}
        {!results && trainingStatus === 'idle' && (
          <div className="text-center py-12 bg-slate-800/30 rounded-lg border border-slate-700">
            <BarChart3 className="w-12 h-12 text-slate-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-slate-400 mb-2">Ready to train</h3>
            <p className="text-slate-500">Configure episodes above and click Start Training</p>
            <p className="text-slate-600 text-sm mt-2">For demo purposes use 50 episodes (~30 seconds). For real results use 500+.</p>
          </div>
        )}
      </main>
    </div>
  );
}
