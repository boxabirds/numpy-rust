/**
 * Benchmark test category hierarchy and types
 */

export interface TestDimension {
  label: string;
  values: number[];
  default: number;
  format?: (val: number) => string;
}

export interface BenchmarkTest {
  id: string;
  name: string;
  description: string;
  operation: string;
  dimensions: TestDimension[];
  minGpuSize?: number; // Minimum size where GPU is beneficial
}

export interface TestCategory {
  id: string;
  name: string;
  description: string;
  tests: BenchmarkTest[];
}

export interface TestGroup {
  id: string;
  name: string;
  icon?: string;
  categories: TestCategory[];
}

export interface BenchmarkResult {
  testId: string;
  testName: string;
  timestamp: number;
  dimensions: Record<string, number>;
  cpuTime: number;
  gpuTime: number;
  speedup: number;
  correct: boolean;
  operation: string;
}

export interface BenchmarkHistory {
  results: BenchmarkResult[];
}
