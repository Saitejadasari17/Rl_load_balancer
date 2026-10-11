import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts';

export default function ResultCharts({ results }) {
  // Prepare baseline comparison data
  const baselineData = Object.entries(results.baselines).map(([name, data]) => ({
    name: name.split(' ').slice(0, 2).join(' '),
    latency: parseFloat(data.avg_latency.toFixed(2)),
    utilization: parseFloat((data.avg_utilization * 100).toFixed(1))
  }));

  // Add RL agent to comparison
  const comparisonData = [
    ...baselineData,
    {
      name: 'RL Agent',
      latency: parseFloat(results.rl_agent_ms.toFixed(2)),
      utilization: parseFloat((results.rl_agent.avg_utilization * 100).toFixed(1))
    }
  ];

  return (
    <div className="grid md:grid-cols-2 gap-6">
      {/* Latency Comparison */}
      <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
        <h3 className="font-semibold mb-4">Latency Comparison (ms)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={comparisonData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="name" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip 
              contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
              labelStyle={{ color: '#f1f5f9' }}
            />
            <Bar dataKey="latency" fill="#3b82f6" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Utilization Comparison */}
      <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
        <h3 className="font-semibold mb-4">Server Utilization (%)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={comparisonData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="name" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip 
              contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
              labelStyle={{ color: '#f1f5f9' }}
            />
            <Bar dataKey="utilization" fill="#10b981" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Baseline Algorithms Detail */}
      <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
        <h3 className="font-semibold mb-4">Baseline Algorithms</h3>
        <div className="space-y-3">
          {Object.entries(results.baselines).map(([name, data]) => (
            <div key={name} className="flex justify-between items-center p-2 bg-slate-700/30 rounded">
              <span className="text-sm">{name}</span>
              <span className="font-semibold text-blue-400">{parseFloat(data.avg_latency.toFixed(2))} ms</span>
            </div>
          ))}
        </div>
      </div>

      {/* RL Agent Summary */}
      <div className="bg-slate-800/50 rounded-lg p-6 border border-slate-700">
        <h3 className="font-semibold mb-4">RL Agent Performance</h3>
        <div className="space-y-3">
          <div className="flex justify-between p-2 bg-blue-900/20 rounded">
            <span className="text-sm">Avg Latency</span>
            <span className="font-semibold text-blue-400">{results.rl_agent.avg_latency.toFixed(2)} ms</span>
          </div>
          <div className="flex justify-between p-2 bg-green-900/20 rounded">
            <span className="text-sm">Utilization</span>
            <span className="font-semibold text-green-400">{(results.rl_agent.avg_utilization * 100).toFixed(1)} %</span>
          </div>
          <div className="flex justify-between p-2 bg-purple-900/20 rounded">
            <span className="text-sm">Fairness Index</span>
            <span className="font-semibold text-purple-400">{results.rl_agent.fairness_index.toFixed(3)}</span>
          </div>
        </div>
      </div>
    </div>
  );
}