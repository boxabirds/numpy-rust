import { useState, useEffect } from 'react';
import { AlertCircle, Zap } from 'lucide-react';
import BenchmarkRunner, { BenchmarkResult } from './components/BenchmarkRunner';
import ResultsDisplay from './components/ResultsDisplay';
import GPUInfo from './components/GPUInfo';
import './styles/app.css';

export default function App() {
  const [wasmModule, setWasmModule] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<BenchmarkResult[]>([]);

  useEffect(() => {
    async function initWasm() {
      try {
        if (!navigator.gpu) {
          setError('WebGPU is not supported in this browser. Please use Chrome 113+ or Edge 113+.');
          setLoading(false);
          return;
        }

        // Load WASM module
        const wasmModule = await import('../pkg/numpy_rust');

        // Initialize WASM (required before calling any functions)
        await wasmModule.default();

        // Initialize GPU
        await wasmModule.init_gpu();

        setWasmModule(wasmModule);
        setLoading(false);
      } catch (err) {
        console.error('Failed to initialize WASM:', err);
        setError(`WASM initialization failed: ${err}`);
        setLoading(false);
      }
    }

    initWasm();
  }, []);

  function handleResults(result: BenchmarkResult) {
    setResults(prev => [...prev, result]);
  }

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="spinner"></div>
        <p>Checking WebGPU support...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-screen">
        <AlertCircle size={48} />
        <h2>WebGPU Not Available</h2>
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
        {wasmModule && <GPUInfo wasmModule={wasmModule} />}
      </header>

      <main className="app-main">
        <section className="info-section">
          <h2>Interactive GPU Benchmarks</h2>
          <p>
            Compare CPU vs GPU performance for matrix multiplication and element-wise operations.
            Run benchmarks below to see the speedup achieved by GPU acceleration.
          </p>
        </section>

        <section className="benchmark-section">
          <BenchmarkRunner wasmModule={wasmModule} onResults={handleResults} />
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
            <h3>Cross-platform</h3>
            <p>Works on Vulkan, Metal, DX12, and WebGPU backends</p>
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
