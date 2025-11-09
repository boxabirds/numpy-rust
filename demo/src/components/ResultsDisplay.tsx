import { BenchmarkResult } from './BenchmarkRunner';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { CheckCircle, XCircle } from 'lucide-react';

interface Props {
  results: BenchmarkResult[];
}

export default function ResultsDisplay({ results }: Props) {
  if (results.length === 0) {
    return (
      <div className="results-display empty">
        <p className="empty-message">
          Run a benchmark to see results here
        </p>
      </div>
    );
  }

  // Group results by operation type
  const matmulResults = results.filter(r => r.operation === 'matmul');
  const sinResults = results.filter(r => r.operation === 'sin');
  const expResults = results.filter(r => r.operation === 'exp');

  // Prepare chart data
  const chartData = results.map(r => ({
    size: r.operation === 'matmul' ? `${r.matrixSize}×${r.matrixSize}` : r.matrixSize.toLocaleString(),
    'CPU Time (ms)': parseFloat(r.cpuTime.toFixed(2)),
    'GPU Time (ms)': parseFloat(r.gpuTime.toFixed(2)),
    operation: r.operation,
  }));

  const formatTime = (ms: number) => {
    if (ms < 1) return `${(ms * 1000).toFixed(2)} µs`;
    if (ms < 1000) return `${ms.toFixed(2)} ms`;
    return `${(ms / 1000).toFixed(2)} s`;
  };

  const renderResultsTable = (opResults: BenchmarkResult[], title: string) => {
    if (opResults.length === 0) return null;

    return (
      <div className="operation-results">
        <h3>{title}</h3>
        <table className="results-table">
          <thead>
            <tr>
              <th>Size</th>
              <th>CPU Time</th>
              <th>GPU Time</th>
              <th>Speedup</th>
              <th>Correct</th>
            </tr>
          </thead>
          <tbody>
            {opResults.map((result, idx) => (
              <tr key={idx}>
                <td>
                  {result.operation === 'matmul'
                    ? `${result.matrixSize}×${result.matrixSize}`
                    : result.matrixSize.toLocaleString()}
                </td>
                <td>{formatTime(result.cpuTime)}</td>
                <td>{formatTime(result.gpuTime)}</td>
                <td className={result.speedup > 1 ? 'speedup-positive' : 'speedup-negative'}>
                  {result.speedup.toFixed(2)}×
                </td>
                <td>
                  {result.correct ? (
                    <CheckCircle className="icon-success" size={20} />
                  ) : (
                    <XCircle className="icon-error" size={20} />
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div className="results-display">
      <h2>Benchmark Results</h2>

      <div className="results-summary">
        <div className="summary-card">
          <span className="summary-label">Total Runs</span>
          <span className="summary-value">{results.length}</span>
        </div>
        <div className="summary-card">
          <span className="summary-label">Avg Speedup</span>
          <span className="summary-value">
            {(results.reduce((sum, r) => sum + r.speedup, 0) / results.length).toFixed(2)}×
          </span>
        </div>
        <div className="summary-card">
          <span className="summary-label">Max Speedup</span>
          <span className="summary-value">
            {Math.max(...results.map(r => r.speedup)).toFixed(2)}×
          </span>
        </div>
        <div className="summary-card">
          <span className="summary-label">All Correct</span>
          <span className="summary-value">
            {results.every(r => r.correct) ? (
              <CheckCircle className="icon-success" size={24} />
            ) : (
              <XCircle className="icon-error" size={24} />
            )}
          </span>
        </div>
      </div>

      <div className="chart-container">
        <h3>Performance Comparison</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="size" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" label={{ value: 'Time (ms)', angle: -90, position: 'insideLeft' }} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1e293b',
                border: '1px solid #334155',
                borderRadius: '0.5rem',
                color: '#f1f5f9'
              }}
            />
            <Legend />
            <Bar dataKey="CPU Time (ms)" fill="#ef4444" />
            <Bar dataKey="GPU Time (ms)" fill="#10b981" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="results-tables">
        {renderResultsTable(matmulResults, 'Matrix Multiplication')}
        {renderResultsTable(sinResults, 'Sin (Element-wise)')}
        {renderResultsTable(expResults, 'Exp (Element-wise)')}
      </div>
    </div>
  );
}
