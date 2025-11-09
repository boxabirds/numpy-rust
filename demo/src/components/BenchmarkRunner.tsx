import { useState, useEffect } from 'react';
import { Play, Loader } from 'lucide-react';
import type { WorkerMessage, WorkerResponse } from '../workers/benchmark.worker';

interface Props {
  worker: Worker | null;
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

export default function BenchmarkRunner({ worker, onResults }: Props) {
  const [operation, setOperation] = useState<'matmul' | 'sin' | 'exp'>('matmul');
  const [size, setSize] = useState(512);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState('');

  const sizes = operation === 'matmul' ? MATRIX_SIZES : ARRAY_SIZES;

  useEffect(() => {
    if (!worker) return;

    // Listen for messages from worker
    const handleMessage = (event: MessageEvent<WorkerResponse>) => {
      const response = event.data;

      switch (response.type) {
        case 'progress':
          setProgress(response.message);
          break;

        case 'result':
          onResults({
            matrixSize: response.size,
            cpuTime: response.cpuTime,
            gpuTime: response.gpuTime,
            speedup: response.speedup,
            correct: response.correct,
            operation: response.operation,
          });
          setProgress('');
          setRunning(false);
          break;

        case 'error':
          console.error('Worker error:', response.error);
          setProgress(`Error: ${response.error}`);
          setRunning(false);
          break;
      }
    };

    worker.addEventListener('message', handleMessage);

    return () => {
      worker.removeEventListener('message', handleMessage);
    };
  }, [worker, onResults]);

  function runBenchmark() {
    if (!worker) {
      setProgress('Worker not initialized');
      return;
    }

    setRunning(true);
    setProgress('Starting benchmark...');

    // Send benchmark request to worker
    worker.postMessage({
      type: 'benchmark',
      operation,
      size,
    } as WorkerMessage);
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
        disabled={running || !worker}
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
