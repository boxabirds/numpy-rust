import { useState } from 'react';
import { Play, Loader } from 'lucide-react';

interface Props {
  wasmModule: any;
  onResults: (results: BenchmarkResult) => void;
}

export interface BenchmarkResult {
  matrixSize: number;
  cpuTime: number;
  gpuTime: number;
  speedup: number;
  correct: boolean;
  operation: 'matmul' | 'sin' | 'exp';
}

const MATRIX_SIZES = [128, 256, 512, 1024, 2048];
const ARRAY_SIZES = [10000, 50000, 100000, 500000, 1000000];

export default function BenchmarkRunner({ wasmModule, onResults }: Props) {
  const [operation, setOperation] = useState<'matmul' | 'sin' | 'exp'>('matmul');
  const [size, setSize] = useState(512);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState('');

  const sizes = operation === 'matmul' ? MATRIX_SIZES : ARRAY_SIZES;

  async function runBenchmark() {
    setRunning(true);
    setProgress('Generating test data...');

    try {
      if (operation === 'matmul') {
        await runMatmulBenchmark();
      } else if (operation === 'sin') {
        await runSinBenchmark();
      } else if (operation === 'exp') {
        await runExpBenchmark();
      }

      setProgress('');
    } catch (err) {
      console.error('Benchmark failed:', err);
      setProgress(`Error: ${err}`);
    } finally {
      setRunning(false);
    }
  }

  async function runMatmulBenchmark() {
    // Generate random matrices
    const a = wasmModule.generate_random_matrix(size, size);
    const b = wasmModule.generate_random_matrix(size, size);

    // Run CPU benchmark
    setProgress('Running CPU matrix multiplication...');
    const cpuStart = performance.now();
    const cpuResult = wasmModule.matmul_cpu(a, b, size);
    const cpuTime = performance.now() - cpuStart;

    // Small delay to ensure GPU is ready
    await new Promise(resolve => setTimeout(resolve, 100));

    // Run GPU benchmark
    setProgress('Running GPU matrix multiplication...');
    const gpuStart = performance.now();
    const gpuResult = wasmModule.matmul_gpu(a, b, size);
    const gpuTime = performance.now() - gpuStart;

    // Verify correctness (check subset)
    setProgress('Verifying results...');
    let correct = true;
    const checkCount = Math.min(100, size * size);
    for (let i = 0; i < checkCount; i++) {
      const diff = Math.abs(cpuResult[i] - gpuResult[i]);
      if (diff > 0.01) {
        correct = false;
        console.warn(`Mismatch at index ${i}: CPU=${cpuResult[i]}, GPU=${gpuResult[i]}`);
        break;
      }
    }

    const speedup = cpuTime / gpuTime;

    onResults({
      matrixSize: size,
      cpuTime,
      gpuTime,
      speedup,
      correct,
      operation: 'matmul',
    });
  }

  async function runSinBenchmark() {
    // Generate random array
    const input = wasmModule.generate_random_array(size);

    // Run CPU benchmark
    setProgress('Running CPU sin...');
    const cpuStart = performance.now();
    const cpuResult = wasmModule.sin_cpu(input);
    const cpuTime = performance.now() - cpuStart;

    await new Promise(resolve => setTimeout(resolve, 100));

    // Run GPU benchmark
    setProgress('Running GPU sin...');
    const gpuStart = performance.now();
    const gpuResult = wasmModule.sin_gpu(input);
    const gpuTime = performance.now() - gpuStart;

    // Verify correctness
    setProgress('Verifying results...');
    let correct = true;
    const checkCount = Math.min(1000, size);
    for (let i = 0; i < checkCount; i++) {
      const diff = Math.abs(cpuResult[i] - gpuResult[i]);
      if (diff > 1e-5) {
        correct = false;
        break;
      }
    }

    const speedup = cpuTime / gpuTime;

    onResults({
      matrixSize: size,
      cpuTime,
      gpuTime,
      speedup,
      correct,
      operation: 'sin',
    });
  }

  async function runExpBenchmark() {
    // Generate random array (smaller values for exp)
    const input = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      input[i] = Math.random() * 2; // Keep values small for exp
    }

    // Run CPU benchmark
    setProgress('Running CPU exp...');
    const cpuStart = performance.now();
    const cpuResult = wasmModule.exp_cpu(input);
    const cpuTime = performance.now() - cpuStart;

    await new Promise(resolve => setTimeout(resolve, 100));

    // Run GPU benchmark
    setProgress('Running GPU exp...');
    const gpuStart = performance.now();
    const gpuResult = wasmModule.exp_gpu(input);
    const gpuTime = performance.now() - gpuStart;

    // Verify correctness
    setProgress('Verifying results...');
    let correct = true;
    const checkCount = Math.min(1000, size);
    for (let i = 0; i < checkCount; i++) {
      const diff = Math.abs(cpuResult[i] - gpuResult[i]);
      const relativeError = diff / Math.max(cpuResult[i], 1e-6);
      if (relativeError > 1e-4) {
        correct = false;
        break;
      }
    }

    const speedup = cpuTime / gpuTime;

    onResults({
      matrixSize: size,
      cpuTime,
      gpuTime,
      speedup,
      correct,
      operation: 'exp',
    });
  }

  return (
    <div className="benchmark-runner">
      <div className="operation-selector">
        <label>Operation:</label>
        <div className="operation-buttons">
          <button
            className={operation === 'matmul' ? 'active' : ''}
            onClick={() => setOperation('matmul')}
            disabled={running}
          >
            Matrix Multiply
          </button>
          <button
            className={operation === 'sin' ? 'active' : ''}
            onClick={() => setOperation('sin')}
            disabled={running}
          >
            Sin (element-wise)
          </button>
          <button
            className={operation === 'exp' ? 'active' : ''}
            onClick={() => setOperation('exp')}
            disabled={running}
          >
            Exp (element-wise)
          </button>
        </div>
      </div>

      <div className="size-selector">
        <label>
          {operation === 'matmul' ? 'Matrix Size:' : 'Array Size:'}
        </label>
        <div className="size-buttons">
          {sizes.map(s => (
            <button
              key={s}
              className={size === s ? 'active' : ''}
              onClick={() => setSize(s)}
              disabled={running}
            >
              {operation === 'matmul' ? `${s}×${s}` : `${s.toLocaleString()}`}
            </button>
          ))}
        </div>
        <input
          type="range"
          min={sizes[0]}
          max={sizes[sizes.length - 1]}
          step={operation === 'matmul' ? 128 : 10000}
          value={size}
          onChange={(e) => setSize(Number(e.target.value))}
          disabled={running}
        />
        <span className="size-display">
          {operation === 'matmul' ? `${size}×${size}` : size.toLocaleString()}
        </span>
      </div>

      <button
        className="run-button"
        onClick={runBenchmark}
        disabled={running}
      >
        {running ? (
          <>
            <Loader className="spinning" size={20} />
            Running...
          </>
        ) : (
          <>
            <Play size={20} />
            Run Benchmark
          </>
        )}
      </button>

      {progress && (
        <div className="progress-indicator">
          {progress}
        </div>
      )}
    </div>
  );
}
