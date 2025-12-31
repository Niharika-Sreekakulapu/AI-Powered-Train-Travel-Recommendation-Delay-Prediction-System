import React, { useState } from 'react';
import LoadingSpinner from './LoadingSpinner';
import { apiService } from '../services/api';

export default function PropagationPanel() {
  const [date, setDate] = useState(new Date().toISOString().split('T')[0]);
  const [station, setStation] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const runBacktest = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const payload = { date, station: station || undefined, max_transfer_minutes: 240, recovery_margin: 5 };
      const res = await apiService.propagateHistorical(payload);
      setResult(res);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Failed to run propagation');
    }
    setLoading(false);
  };

  return (
    <div className="bg-white p-6 rounded-xl shadow-sm">
      <h3 className="text-lg font-semibold mb-3">Propagation Backtest (Historical Day)</h3>
      <div className="flex space-x-3 items-center mb-4">
        <input type="date" value={date} onChange={(e)=>setDate(e.target.value)} className="border p-2 rounded" />
        <input placeholder="Station (optional) e.g., HYB" value={station} onChange={(e)=>setStation(e.target.value)} className="border p-2 rounded" />
        <button onClick={runBacktest} className="px-4 py-2 bg-primary-600 text-white rounded hover:bg-primary-700">Run</button>
      </div>

      {loading && <LoadingSpinner message="Running propagation backtest..." />}

      {error && <div className="text-red-600">{error}</div>}

      {result && (
        <div className="mt-4 space-y-4">
          <div className="flex space-x-6">
            <div>
              <h4 className="font-medium">Metrics</h4>
              <pre className="text-sm bg-gray-50 p-3 rounded">{JSON.stringify(result.metrics, null, 2)}</pre>
            </div>
            <div>
              <h4 className="font-medium">Top affected trains</h4>
              <ol className="list-decimal list-inside text-sm">
                {result.top_affected.map(t => (
                  <li key={t.train_id}>{t.train_id}: simulated {t.simulated.toFixed(1)}m (delta {t.delta.toFixed(1)}m)</li>
                ))}
              </ol>
            </div>
          </div>

          <div>
            <h4 className="font-medium">Visualization</h4>
            <div className="border rounded p-4">
              <img alt="Propagation visualization" src={`data:image/png;base64,${result.viz_base64_png}`} style={{ maxWidth: '100%' }} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
