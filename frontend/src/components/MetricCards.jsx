import React from 'react';
import { TrendingDown, Zap, Award } from 'lucide-react';

export default function MetricCards({ results }) {
  const improvement = results.improvement;
  const isPositive = improvement > 0;

  return (
    <div className="grid md:grid-cols-4 gap-4">
      {/* Best Baseline */}
      <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
        <div className="text-sm text-slate-400 mb-1">Best Baseline</div>
        <div className="text-2xl font-bold text-amber-400">{results.best_baseline_ms.toFixed(2)} ms</div>
        <div className="text-xs text-slate-500 mt-2">Least Connections Algorithm</div>
      </div>

      {/* RL Agent */}
      <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
        <div className="text-sm text-slate-400 mb-1">RL Agent</div>
        <div className="text-2xl font-bold text-blue-400">{results.rl_agent_ms.toFixed(2)} ms</div>
        <div className="text-xs text-slate-500 mt-2">Dueling Double DQN</div>
      </div>

      {/* Improvement */}
      <div className={`rounded-lg p-4 border ${
        isPositive 
          ? 'bg-green-900/20 border-green-700' 
          : 'bg-orange-900/20 border-orange-700'
      }`}>
        <div className="text-sm text-slate-400 mb-1">Improvement</div>
        <div className={`text-2xl font-bold ${isPositive ? 'text-green-400' : 'text-orange-400'}`}>
          {isPositive ? '+' : ''}{improvement.toFixed(1)}%
        </div>
        <div className="text-xs text-slate-500 mt-2 flex items-center gap-1">
          <TrendingDown className="w-3 h-3" /> vs Best Baseline
        </div>
      </div>

      {/* Fairness */}
      <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
        <div className="text-sm text-slate-400 mb-1">Fairness Index</div>
        <div className="text-2xl font-bold text-purple-400">{results.rl_agent.fairness_index.toFixed(3)}</div>
        <div className="text-xs text-slate-500 mt-2">Server Load Balance</div>
      </div>
    </div>
  );
}
