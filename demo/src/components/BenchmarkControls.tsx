import { useState, useEffect } from 'react';
import { Play, Loader } from 'lucide-react';
import { findTestById } from '../data/testHierarchy';
import type { BenchmarkTest } from '../types/benchmark';
import type { BenchmarkResult } from '../types/benchmark';
import type { WorkerMessage, WorkerResponse } from '../workers/benchmark.worker';

interface Props {
  worker: Worker | null;
  selectedTestId: string | null;
  onResults: (results: BenchmarkResult) => void;
  cpuTimeout: number;
}

export default function BenchmarkControls({ worker, selectedTestId, onResults, cpuTimeout }: Props) {
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState('');
  const [dimensions, setDimensions] = useState<Record<string, number>>({});

  const testInfo = selectedTestId ? findTestById(selectedTestId) : null;
  const test = testInfo?.test as BenchmarkTest | undefined;

  // Initialize dimensions when test changes
  useEffect(() => {
    if (test) {
      const defaultDimensions: Record<string, number> = {};
      for (const dim of test.dimensions) {
        defaultDimensions[dim.label] = dim.default;
      }
      setDimensions(defaultDimensions);
    }
  }, [test]);

  // Listen for messages from worker
  useEffect(() => {
    if (!worker) {
      return;
    }

    const handleMessage = (event: MessageEvent<WorkerResponse>) => {
      const response = event.data;

      switch (response.type) {
        case 'progress':
          setProgress(response.message);
          break;

        case 'result':
          if (test) {
            onResults({
              testId: test.id,
              testName: test.name,
              timestamp: Date.now(),
              dimensions: { ...dimensions },
              cpuTime: response.cpuTime,
              gpuTime: response.gpuTime,
              speedup: response.speedup,
              correct: response.correct,
              operation: test.operation,
            });
          }
          setProgress('');
          setRunning(false);
          break;

        case 'error':
          setProgress(`Error: ${response.error}`);
          setRunning(false);
          break;
      }
    };

    worker.addEventListener('message', handleMessage);

    return () => {
      worker.removeEventListener('message', handleMessage);
    };
  }, [worker, test, dimensions, onResults]);

  function runBenchmark() {
    if (!worker || !test) {
      setProgress('Worker or test not available');
      return;
    }

    setRunning(true);
    setProgress('Starting benchmark...');

    // Determine size based on dimensions
    // For now, use first dimension as "size" for worker compatibility
    const primaryDimension = test.dimensions[0];
    const size = dimensions[primaryDimension.label];

    worker.postMessage({
      type: 'benchmark',
      operation: test.operation as any,
      size,
      cpuTimeout: cpuTimeout > 0 ? cpuTimeout : undefined,
    } as WorkerMessage);
  }

  function updateDimension(label: string, value: number) {
    setDimensions(prev => ({
      ...prev,
      [label]: value,
    }));
  }

  if (!test) {
    return (
      <div className="benchmark-controls">
        <div className="no-test-selected">
          <p>Select a test from the sidebar to begin</p>
        </div>
      </div>
    );
  }

  return (
    <div className="benchmark-controls">
      <div className="test-info">
        <h2>{test.name}</h2>
        <p className="test-description">{test.description}</p>
        {testInfo && (
          <p className="test-category">
            {testInfo.groupName} → {testInfo.categoryName}
          </p>
        )}
        {test.minGpuSize && (
          <p className="gpu-hint">
            ⚡ GPU acceleration beneficial for sizes ≥ {test.minGpuSize.toLocaleString()}
          </p>
        )}
      </div>

      <div className="dimension-controls">
        <h3>Dimensions</h3>
        {test.dimensions.map(dim => {
          const currentValue = dimensions[dim.label] ?? dim.default;
          const formattedValue = dim.format ? dim.format(currentValue) : currentValue.toLocaleString();

          return (
            <div key={dim.label} className="dimension-control">
              <label>{dim.label}:</label>

              {/* Preset buttons */}
              <div className="dimension-buttons">
                {dim.values.map(value => {
                  const formatted = dim.format ? dim.format(value) : value.toLocaleString();
                  return (
                    <button
                      key={value}
                      className={currentValue === value ? 'active' : ''}
                      onClick={() => updateDimension(dim.label, value)}
                      disabled={running}
                    >
                      {formatted}
                    </button>
                  );
                })}
              </div>

              {/* Slider */}
              <div className="dimension-slider">
                <input
                  type="range"
                  min={dim.values[0]}
                  max={dim.values[dim.values.length - 1]}
                  step={dim.values.length > 1 ? (dim.values[1] - dim.values[0]) : 1}
                  value={currentValue}
                  onChange={(e) => updateDimension(dim.label, Number(e.target.value))}
                  disabled={running}
                />
                <span className="dimension-value">{formattedValue}</span>
              </div>
            </div>
          );
        })}
      </div>

      <div className="run-section">
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
    </div>
  );
}
