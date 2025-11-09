import { useState, useEffect } from 'react';
import { AlertCircle, Zap } from 'lucide-react';
import BenchmarkRunner, { BenchmarkResult } from './components/BenchmarkRunner';
import ResultsDisplay from './components/ResultsDisplay';
import GPUInfo from './components/GPUInfo';
import type { WorkerMessage, WorkerResponse } from './workers/benchmark.worker';
import './styles/app.css';

export default function App() {
  const [worker, setWorker] = useState<Worker | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [gpuAvailable, setGpuAvailable] = useState(false);
  const [initProgress, setInitProgress] = useState('Initializing...');
  const [results, setResults] = useState<BenchmarkResult[]>([]);

  useEffect(() => {
    // Check WebGPU support
    if (!navigator.gpu) {
      setError('WebGPU is not supported in this browser. Please use Chrome 113+ or Edge 113+.');
      setLoading(false);
      return;
    }

    // Create Web Worker
    const benchmarkWorker = new Worker(
      new URL('./workers/benchmark.worker.ts', import.meta.url),
      { type: 'module' }
    );

    // Listen for messages from worker
    benchmarkWorker.addEventListener('message', (event: MessageEvent<WorkerResponse>) => {
      const response = event.data;

      switch (response.type) {
        case 'init_success':
          setGpuAvailable(response.gpuAvailable);
          setLoading(false);
          setWorker(benchmarkWorker);
          break;

        case 'init_error':
          setError(`Worker initialization failed: ${response.error}`);
          setLoading(false);
          break;

        case 'progress':
          setInitProgress(response.message);
          break;

        case 'error':
          // Errors during benchmark are handled by BenchmarkRunner
          break;
      }
    });

    // Initialize the worker
    benchmarkWorker.postMessage({ type: 'init' } as WorkerMessage);

    // Cleanup on unmount
    return () => {
      benchmarkWorker.terminate();
    };
  }, []);

  function handleResults(result: BenchmarkResult) {
    setResults(prev => [...prev, result]);
  }

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="spinner"></div>
        <p>{initProgress}</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-screen">
        <AlertCircle size={48} />
        <h2>Initialization Failed</h2>
        <p>{error}</p>
        <div className="error-help">
          <h3>Requirements:</h3>
          <ul>
            <li>Chrome 113+ or Edge 113+ (WebGPU support)</li>
            <li>GPU with WebGPU-compatible driver</li>
            <li>Enable chrome://flags/#enable-unsafe-webgpu (if needed)</li>
          </ul>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1>
            <Zap className="logo" />
            numpy-rust WebGPU Demo
          </h1>
          <p className="subtitle">
            GPU-accelerated matrix operations in your browser
          </p>
        </div>
        <GPUInfo gpuAvailable={gpuAvailable} />
      </header>

      <main className="app-main">
        <section className="info-section">
          <h2>Interactive GPU Benchmarks</h2>
          <p>
            Compare CPU vs GPU performance for matrix multiplication and element-wise operations.
            Run benchmarks below to see the speedup achieved by GPU acceleration.
            {!gpuAvailable && ' (Running in CPU-only mode)'}
          </p>
        </section>

        <section className="benchmark-section">
          <BenchmarkRunner worker={worker} onResults={handleResults} />
        </section>

        <section className="results-section">
          <ResultsDisplay results={results} />
        </section>

        <section className="features">
          <div className="feature-card">
            <h3>Matrix Multiplication</h3>
            <p>GPU-accelerated matmul with 100-300x speedup for large matrices (≥512×512)</p>
          </div>
          <div className="feature-card">
            <h3>Element-wise Operations</h3>
            <p>GPU sin, cos, exp, log with 50-200x speedup for arrays ≥100K elements</p>
          </div>
          <div className="feature-card">
            <h3>Non-blocking Execution</h3>
            <p>Benchmarks run in a Web Worker, keeping the UI responsive</p>
          </div>
        </section>
      </main>

      <footer className="app-footer">
        <p>Built with Bun, React, Vite, and Rust 🦀</p>
        <p>
          <a href="https://github.com/boxabirds/numpy-rust" target="_blank" rel="noopener">
            View on GitHub
          </a>
        </p>
      </footer>
    </div>
  );
}
