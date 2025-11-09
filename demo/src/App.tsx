import { useState, useEffect } from 'react';
import { AlertCircle, Zap } from 'lucide-react';
import BenchmarkSidebar from './components/BenchmarkSidebar';
import BenchmarkControls from './components/BenchmarkControls';
import BenchmarkHistory from './components/BenchmarkHistory';
import GPUInfo from './components/GPUInfo';
import type { BenchmarkResult } from './types/benchmark';
import type { WorkerMessage, WorkerResponse } from './workers/benchmark.worker';
import './styles/app.css';

export default function App() {
  const [worker, setWorker] = useState<Worker | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [gpuAvailable, setGpuAvailable] = useState(false);
  const [initProgress, setInitProgress] = useState('Initializing...');
  const [results, setResults] = useState<BenchmarkResult[]>([]);
  const [selectedTestId, setSelectedTestId] = useState<string | null>('matmul');

  useEffect(() => {
    console.log('[Main] Starting worker initialization...');

    // Check WebGPU support
    if (!navigator.gpu) {
      console.error('[Main] navigator.gpu not available');
      setError('WebGPU is not supported in this browser. Please use Chrome 113+ or Edge 113+.');
      setLoading(false);
      return;
    }

    console.log('[Main] WebGPU available, creating worker...');

    // Create Web Worker
    const benchmarkWorker = new Worker(
      new URL('./workers/benchmark.worker.ts', import.meta.url),
      { type: 'module' }
    );

    console.log('[Main] Worker created, setting up message handler...');

    // Listen for messages from worker
    benchmarkWorker.addEventListener('message', (event: MessageEvent<WorkerResponse>) => {
      const response = event.data;
      console.log('[Main] Received message from worker:', response.type, response);

      switch (response.type) {
        case 'init_success':
          console.log('[Main] Worker initialized successfully, GPU available:', response.gpuAvailable);
          setGpuAvailable(response.gpuAvailable);
          setLoading(false);
          setWorker(benchmarkWorker);
          break;

        case 'init_error':
          console.error('[Main] Worker initialization error:', response.error);
          setError(`Worker initialization failed: ${response.error}`);
          setLoading(false);
          break;

        case 'progress':
          console.log('[Main] Progress update:', response.message);
          setInitProgress(response.message);
          break;

        case 'error':
          console.error('[Main] Worker error during operation:', response.error);
          // Errors during benchmark are handled by BenchmarkRunner
          break;
      }
    });

    console.log('[Main] Sending init message to worker...');

    // Initialize the worker
    benchmarkWorker.postMessage({ type: 'init' } as WorkerMessage);

    // Cleanup on unmount
    return () => {
      console.log('[Main] Cleaning up worker...');
      benchmarkWorker.terminate();
    };
  }, []);

  function handleResults(result: BenchmarkResult) {
    setResults(prev => [...prev, result]);
  }

  function clearResults() {
    setResults([]);
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
            numpy-rust WebGPU Benchmarks
          </h1>
          <p className="subtitle">
            Comprehensive CPU vs GPU performance comparison
            {!gpuAvailable && ' (Running in CPU-only mode)'}
          </p>
        </div>
        <GPUInfo gpuAvailable={gpuAvailable} />
      </header>

      <main className="app-main">
        <div className="benchmark-layout">
          {/* Left sidebar - Test navigation */}
          <aside className="sidebar-column">
            <BenchmarkSidebar
              selectedTestId={selectedTestId}
              onSelectTest={setSelectedTestId}
            />
          </aside>

          {/* Right content - Controls and History */}
          <div className="content-column">
            {/* Top - Dimension controls and run button */}
            <section className="controls-section">
              <BenchmarkControls
                worker={worker}
                selectedTestId={selectedTestId}
                onResults={handleResults}
              />
            </section>

            {/* Bottom - Test run history */}
            <section className="history-section">
              <BenchmarkHistory
                results={results}
                onClear={clearResults}
              />
            </section>
          </div>
        </div>
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
