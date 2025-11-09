/**
 * Web Worker for running GPU benchmarks without blocking the main thread
 */

// Message types
export type WorkerMessage =
  | { type: 'init' }
  | {
      type: 'benchmark';
      operation: 'matmul' | 'sin' | 'exp';
      size: number;
    };

export type WorkerResponse =
  | { type: 'init_success'; gpuAvailable: boolean }
  | { type: 'init_error'; error: string }
  | { type: 'progress'; message: string }
  | {
      type: 'result';
      operation: 'matmul' | 'sin' | 'exp';
      size: number;
      cpuTime: number;
      gpuTime: number;
      speedup: number;
      correct: boolean;
    }
  | { type: 'error'; error: string };

let wasmModule: any = null;
let gpuAvailable = false;

// Handle messages from main thread
self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const message = event.data;

  try {
    switch (message.type) {
      case 'init':
        await initWasm();
        break;
      case 'benchmark':
        await runBenchmark(message.operation, message.size);
        break;
    }
  } catch (error) {
    postMessage({
      type: 'error',
      error: String(error),
    } as WorkerResponse);
  }
};

async function initWasm() {
  try {
    postMessage({ type: 'progress', message: 'Loading WASM module...' } as WorkerResponse);

    // Load WASM module
    const module = await import('../../pkg/numpy_rust');

    postMessage({ type: 'progress', message: 'Initializing WASM...' } as WorkerResponse);

    // Initialize WASM
    await module.default();

    postMessage({ type: 'progress', message: 'Checking WebGPU support...' } as WorkerResponse);

    // Try to initialize GPU
    try {
      await module.init_gpu();
      gpuAvailable = true;
      postMessage({ type: 'progress', message: 'GPU initialized successfully!' } as WorkerResponse);
    } catch (err) {
      gpuAvailable = false;
      console.warn('GPU initialization failed, will use CPU only:', err);
      postMessage({ type: 'progress', message: 'GPU not available, using CPU only' } as WorkerResponse);
    }

    wasmModule = module;

    postMessage({
      type: 'init_success',
      gpuAvailable,
    } as WorkerResponse);
  } catch (error) {
    postMessage({
      type: 'init_error',
      error: String(error),
    } as WorkerResponse);
  }
}

async function runBenchmark(operation: 'matmul' | 'sin' | 'exp', size: number) {
  if (!wasmModule) {
    throw new Error('WASM module not initialized');
  }

  switch (operation) {
    case 'matmul':
      await runMatmulBenchmark(size);
      break;
    case 'sin':
      await runSinBenchmark(size);
      break;
    case 'exp':
      await runExpBenchmark(size);
      break;
  }
}

async function runMatmulBenchmark(size: number) {
  postMessage({ type: 'progress', message: 'Generating test matrices...' } as WorkerResponse);

  // Generate random matrices
  const a = wasmModule.generate_random_matrix(size, size);
  const b = wasmModule.generate_random_matrix(size, size);

  // Run CPU benchmark
  postMessage({ type: 'progress', message: `Running CPU matrix multiplication (${size}×${size})...` } as WorkerResponse);
  const cpuStart = performance.now();
  const cpuResult = wasmModule.matmul_cpu(a, b, size);
  const cpuTime = performance.now() - cpuStart;

  // Small delay to ensure GPU is ready
  await new Promise((resolve) => setTimeout(resolve, 100));

  // Run GPU benchmark
  postMessage({ type: 'progress', message: `Running GPU matrix multiplication (${size}×${size})...` } as WorkerResponse);
  const gpuStart = performance.now();
  const gpuResult = wasmModule.matmul_gpu(a, b, size);
  const gpuTime = performance.now() - gpuStart;

  // Verify correctness (check subset)
  postMessage({ type: 'progress', message: 'Verifying results...' } as WorkerResponse);
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

  postMessage({
    type: 'result',
    operation: 'matmul',
    size,
    cpuTime,
    gpuTime,
    speedup,
    correct,
  } as WorkerResponse);
}

async function runSinBenchmark(size: number) {
  postMessage({ type: 'progress', message: 'Generating test array...' } as WorkerResponse);

  // Generate random array
  const input = wasmModule.generate_random_array(size);

  // Run CPU benchmark
  postMessage({ type: 'progress', message: `Running CPU sin (${size.toLocaleString()} elements)...` } as WorkerResponse);
  const cpuStart = performance.now();
  const cpuResult = wasmModule.sin_cpu(input);
  const cpuTime = performance.now() - cpuStart;

  await new Promise((resolve) => setTimeout(resolve, 100));

  // Run GPU benchmark
  postMessage({ type: 'progress', message: `Running GPU sin (${size.toLocaleString()} elements)...` } as WorkerResponse);
  const gpuStart = performance.now();
  const gpuResult = wasmModule.sin_gpu(input);
  const gpuTime = performance.now() - gpuStart;

  // Verify correctness
  postMessage({ type: 'progress', message: 'Verifying results...' } as WorkerResponse);
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

  postMessage({
    type: 'result',
    operation: 'sin',
    size,
    cpuTime,
    gpuTime,
    speedup,
    correct,
  } as WorkerResponse);
}

async function runExpBenchmark(size: number) {
  postMessage({ type: 'progress', message: 'Generating test array...' } as WorkerResponse);

  // Generate random array (smaller values for exp)
  const input = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    input[i] = Math.random() * 2; // Keep values small for exp
  }

  // Run CPU benchmark
  postMessage({ type: 'progress', message: `Running CPU exp (${size.toLocaleString()} elements)...` } as WorkerResponse);
  const cpuStart = performance.now();
  const cpuResult = wasmModule.exp_cpu(input);
  const cpuTime = performance.now() - cpuStart;

  await new Promise((resolve) => setTimeout(resolve, 100));

  // Run GPU benchmark
  postMessage({ type: 'progress', message: `Running GPU exp (${size.toLocaleString()} elements)...` } as WorkerResponse);
  const gpuStart = performance.now();
  const gpuResult = wasmModule.exp_gpu(input);
  const gpuTime = performance.now() - gpuStart;

  // Verify correctness
  postMessage({ type: 'progress', message: 'Verifying results...' } as WorkerResponse);
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

  postMessage({
    type: 'result',
    operation: 'exp',
    size,
    cpuTime,
    gpuTime,
    speedup,
    correct,
  } as WorkerResponse);
}
