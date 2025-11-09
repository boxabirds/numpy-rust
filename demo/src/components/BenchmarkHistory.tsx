import { Download, Trash2 } from 'lucide-react';
import type { BenchmarkResult } from '../types/benchmark';

interface Props {
  results: BenchmarkResult[];
  onClear?: () => void;
}

/**
 * Export single result to CSV
 */
function exportResultToCSV(result: BenchmarkResult) {
  const csv = generateCSV([result]);
  const timestamp = new Date(result.timestamp).toISOString().replace(/[:.]/g, '-');
  const filename = `benchmark-${result.testName.replace(/\s+/g, '_')}-${timestamp}.csv`;
  downloadCSV(csv, filename);
}

/**
 * Export all results to CSV
 */
function exportAllResultsToCSV(results: BenchmarkResult[]) {
  const csv = generateCSV(results);
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const filename = `benchmark-all-${timestamp}.csv`;
  downloadCSV(csv, filename);
}

/**
 * Generate CSV from results
 */
function generateCSV(results: BenchmarkResult[]): string {
  if (results.length === 0) {
    return '';
  }

  // CSV header
  const headers = [
    'Test Name',
    'Operation',
    'Timestamp',
    'Dimensions',
    'CPU Time (ms)',
    'GPU Time (ms)',
    'Speedup',
    'Correct',
  ];

  // CSV rows
  const rows = results.map(result => {
    const dimensionsStr = Object.entries(result.dimensions)
      .map(([key, value]) => `${key}:${value}`)
      .join(';');

    return [
      escapeCSV(result.testName),
      escapeCSV(result.operation),
      new Date(result.timestamp).toISOString(),
      escapeCSV(dimensionsStr),
      result.cpuTime.toFixed(2),
      result.gpuTime.toFixed(2),
      result.speedup.toFixed(2),
      result.correct ? 'Yes' : 'No',
    ];
  });

  // Combine header and rows
  const csvLines = [headers, ...rows];
  return csvLines.map(row => row.join(',')).join('\n');
}

/**
 * Escape CSV field
 */
function escapeCSV(value: string): string {
  if (value.includes(',') || value.includes('"') || value.includes('\n')) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}

/**
 * Download CSV file
 */
function downloadCSV(csv: string, filename: string) {
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Format speedup with color coding
 */
function formatSpeedup(speedup: number): { text: string; className: string } {
  const text = speedup >= 1 ? `${speedup.toFixed(2)}×` : `${speedup.toFixed(2)}×`;

  let className = 'speedup-neutral';
  if (speedup >= 100) {
    className = 'speedup-excellent';
  } else if (speedup >= 10) {
    className = 'speedup-great';
  } else if (speedup >= 2) {
    className = 'speedup-good';
  } else if (speedup < 1) {
    className = 'speedup-slow';
  }

  return { text, className };
}

/**
 * Format dimensions for display
 */
function formatDimensions(dimensions: Record<string, number>): string {
  return Object.entries(dimensions)
    .map(([key, value]) => {
      // Format large numbers with locale
      const formattedValue = value >= 1000 ? value.toLocaleString() : value;
      return `${key}: ${formattedValue}`;
    })
    .join(', ');
}

export default function BenchmarkHistory({ results, onClear }: Props) {
  if (results.length === 0) {
    return (
      <div className="benchmark-history empty">
        <p className="empty-message">No benchmark results yet. Run a test to see results here.</p>
      </div>
    );
  }

  return (
    <div className="benchmark-history">
      <div className="history-header">
        <h3>Test History ({results.length})</h3>
        <div className="history-actions">
          <button
            className="btn-download-all"
            onClick={() => exportAllResultsToCSV(results)}
            title="Download all results as CSV"
          >
            <Download size={16} />
            Download All CSV
          </button>
          {onClear && (
            <button
              className="btn-clear"
              onClick={onClear}
              title="Clear all results"
            >
              <Trash2 size={16} />
              Clear
            </button>
          )}
        </div>
      </div>

      <div className="history-table-container">
        <table className="history-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Test</th>
              <th>Dimensions</th>
              <th>CPU (ms)</th>
              <th>GPU (ms)</th>
              <th>Speedup</th>
              <th>✓</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {results.map((result, index) => {
              const speedup = formatSpeedup(result.speedup);
              const time = new Date(result.timestamp).toLocaleTimeString();
              const dimensions = formatDimensions(result.dimensions);

              return (
                <tr key={index} className={result.correct ? '' : 'error'}>
                  <td className="time-cell">{time}</td>
                  <td className="test-cell" title={result.operation}>
                    {result.testName}
                  </td>
                  <td className="dimensions-cell" title={dimensions}>
                    {dimensions}
                  </td>
                  <td className="cpu-time-cell">{result.cpuTime.toFixed(2)}</td>
                  <td className="gpu-time-cell">{result.gpuTime.toFixed(2)}</td>
                  <td className={`speedup-cell ${speedup.className}`}>
                    {speedup.text}
                  </td>
                  <td className="correct-cell">
                    {result.correct ? (
                      <span className="correct-icon" title="Results match">✓</span>
                    ) : (
                      <span className="error-icon" title="Results mismatch">✗</span>
                    )}
                  </td>
                  <td className="actions-cell">
                    <button
                      className="btn-download-row"
                      onClick={() => exportResultToCSV(result)}
                      title="Download as CSV"
                    >
                      <Download size={14} />
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
