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

const MATRIX_SIZES = [128, 256, 512, 1024, 2048, 4096];
const ARRAY_SIZES = [1000000, 10000000, 100000000]; // 1M, 10M, 100M

export default function BenchmarkRunner({ worker, onResults }: Props) {
  const [operation, setOperation] = useState<'matmul' | 'sin' | 'exp'>('matmul');
  const [size, setSize] = useState(512);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState('');

  const sizes = operation === 'matmul' ? MATRIX_SIZES : ARRAY_SIZES;

  // Update size when operation changes to use appropriate default
  useEffect(() => {
    if (operation === 'matmul') {
      setSize(1024); // Good default for matmul
    } else {
      setSize(10000000); // 10M - large enough to see GPU benefit
    }
  }, [operation]);

  useEffect(() => {
    if (!worker) {
      console.log('[BenchmarkRunner] No worker available');
      return;
    }

    console.log('[BenchmarkRunner] Setting up message handler');

    // Listen for messages from worker
    const handleMessage = (event: MessageEvent<WorkerResponse>) => {
      const response = event.data;
      console.log('[BenchmarkRunner] Received from worker:', response.type, response);

      // Log main thread responsiveness
      const now = performance.now();
      console.log('[BenchmarkRunner] Main thread timestamp:', now, 'ms');

      switch (response.type) {
        case 'progress':
          console.log('[BenchmarkRunner] Progress:', response.message);
          setProgress(response.message);
          break;

        case 'result':
          console.log('[BenchmarkRunner] Benchmark complete:', response);
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
          console.error('[BenchmarkRunner] Worker error:', response.error);
          setProgress(`Error: ${response.error}`);
          setRunning(false);
          break;
      }
    };

    worker.addEventListener('message', handleMessage);

    return () => {
      console.log('[BenchmarkRunner] Cleaning up message handler');
      worker.removeEventListener('message', handleMessage);
    };
  }, [worker, onResults]);

  function runBenchmark() {
    console.log('[BenchmarkRunner] runBenchmark called');
    console.log('[BenchmarkRunner] Main thread before benchmark:', performance.now());

    if (!worker) {
      console.error('[BenchmarkRunner] Worker not initialized!');
      setProgress('Worker not initialized');
      return;
    }

    console.log('[BenchmarkRunner] Starting benchmark:', { operation, size });
    setRunning(true);
    setProgress('Starting benchmark...');

    // Send benchmark request to worker
    console.log('[BenchmarkRunner] Posting message to worker...');
    worker.postMessage({
      type: 'benchmark',
      operation,
      size,
    } as WorkerMessage);
    console.log('[BenchmarkRunner] Message posted, main thread should be free');
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
