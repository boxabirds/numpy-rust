/**
 * Complete test hierarchy for GPU vs CPU benchmarks
 */

import type { TestGroup } from '../types/benchmark';

const formatMatrix = (n: number) => `${n}×${n}`;
const formatArray = (n: number) => n.toLocaleString();

export const TEST_HIERARCHY: TestGroup[] = [
  {
    id: 'matrix-ops',
    name: 'Matrix Operations',
    icon: '⊗',
    categories: [
      {
        id: 'basic-linalg',
        name: 'Basic Linear Algebra',
        description: 'Core matrix operations',
        tests: [
          {
            id: 'matmul',
            name: 'Matrix Multiplication',
            description: 'C = A × B',
            operation: 'matmul',
            dimensions: [
              {
                label: 'Matrix Size',
                values: [128, 256, 512, 1024, 2048, 4096],
                default: 1024,
                format: formatMatrix,
              },
            ],
            minGpuSize: 512,
          },
          {
            id: 'matmul-rectangular',
            name: 'Rectangular Matrix Multiply',
            description: 'Non-square matrix multiplication',
            operation: 'matmul_rect',
            dimensions: [
              {
                label: 'M (rows)',
                values: [256, 512, 1024, 2048, 4096],
                default: 1024,
              },
              {
                label: 'N (cols)',
                values: [256, 512, 1024, 2048, 4096],
                default: 1024,
              },
              {
                label: 'K (inner)',
                values: [256, 512, 1024, 2048],
                default: 512,
              },
            ],
            minGpuSize: 512,
          },
          {
            id: 'dot-product',
            name: 'Dot Product',
            description: 'Vector dot product',
            operation: 'dot',
            dimensions: [
              {
                label: 'Vector Size',
                values: [10000, 100000, 1000000, 10000000],
                default: 1000000,
                format: formatArray,
              },
            ],
            minGpuSize: 100000,
          },
          {
            id: 'transpose',
            name: 'Matrix Transpose',
            description: 'In-place and out-of-place transpose',
            operation: 'transpose',
            dimensions: [
              {
                label: 'Matrix Size',
                values: [512, 1024, 2048, 4096, 8192],
                default: 2048,
                format: formatMatrix,
              },
            ],
            minGpuSize: 1024,
          },
        ],
      },
      {
        id: 'advanced-linalg',
        name: 'Advanced Linear Algebra',
        description: 'Decompositions and factorizations',
        tests: [
          {
            id: 'qr-decomp',
            name: 'QR Decomposition',
            description: 'QR factorization',
            operation: 'qr',
            dimensions: [
              {
                label: 'Matrix Size',
                values: [256, 512, 1024, 2048],
                default: 512,
                format: formatMatrix,
              },
            ],
          },
          {
            id: 'svd',
            name: 'SVD',
            description: 'Singular Value Decomposition',
            operation: 'svd',
            dimensions: [
              {
                label: 'Matrix Size',
                values: [256, 512, 1024],
                default: 512,
                format: formatMatrix,
              },
            ],
          },
          {
            id: 'eigenvalue',
            name: 'Eigenvalue Decomposition',
            description: 'Eigenvalues and eigenvectors',
            operation: 'eig',
            dimensions: [
              {
                label: 'Matrix Size',
                values: [256, 512, 1024, 2048],
                default: 512,
                format: formatMatrix,
              },
            ],
          },
        ],
      },
    ],
  },
  {
    id: 'elementwise',
    name: 'Element-wise Operations',
    icon: '∀',
    categories: [
      {
        id: 'trigonometric',
        name: 'Trigonometric Functions',
        description: 'Sin, cos, tan operations',
        tests: [
          {
            id: 'sin',
            name: 'Sine',
            description: 'Element-wise sine',
            operation: 'sin',
            dimensions: [
              {
                label: 'Array Size',
                values: [1000000, 10000000, 100000000],
                default: 10000000,
                format: formatArray,
              },
            ],
            minGpuSize: 1000000,
          },
          {
            id: 'cos',
            name: 'Cosine',
            description: 'Element-wise cosine',
            operation: 'cos',
            dimensions: [
              {
                label: 'Array Size',
                values: [1000000, 10000000, 100000000],
                default: 10000000,
                format: formatArray,
              },
            ],
            minGpuSize: 1000000,
          },
          {
            id: 'tan',
            name: 'Tangent',
            description: 'Element-wise tangent',
            operation: 'tan',
            dimensions: [
              {
                label: 'Array Size',
                values: [1000000, 10000000, 100000000],
                default: 10000000,
                format: formatArray,
              },
            ],
            minGpuSize: 1000000,
          },
        ],
      },
      {
        id: 'exponential',
        name: 'Exponential & Logarithmic',
        description: 'Exp, log, power operations',
        tests: [
          {
            id: 'exp',
            name: 'Exponential',
            description: 'Element-wise exp(x)',
            operation: 'exp',
            dimensions: [
              {
                label: 'Array Size',
                values: [1000000, 10000000, 100000000],
                default: 10000000,
                format: formatArray,
              },
            ],
            minGpuSize: 1000000,
          },
          {
            id: 'log',
            name: 'Natural Logarithm',
            description: 'Element-wise ln(x)',
            operation: 'log',
            dimensions: [
              {
                label: 'Array Size',
                values: [1000000, 10000000, 100000000],
                default: 10000000,
                format: formatArray,
              },
            ],
            minGpuSize: 1000000,
          },
          {
            id: 'sqrt',
            name: 'Square Root',
            description: 'Element-wise sqrt(x)',
            operation: 'sqrt',
            dimensions: [
              {
                label: 'Array Size',
                values: [1000000, 10000000, 100000000],
                default: 10000000,
                format: formatArray,
              },
            ],
            minGpuSize: 1000000,
          },
          {
            id: 'square',
            name: 'Square',
            description: 'Element-wise x²',
            operation: 'square',
            dimensions: [
              {
                label: 'Array Size',
                values: [1000000, 10000000, 100000000],
                default: 10000000,
                format: formatArray,
              },
            ],
            minGpuSize: 1000000,
          },
        ],
      },
      {
        id: 'arithmetic',
        name: 'Arithmetic Operations',
        description: 'Basic arithmetic',
        tests: [
          {
            id: 'add',
            name: 'Addition',
            description: 'Element-wise A + B',
            operation: 'add',
            dimensions: [
              {
                label: 'Array Size',
                values: [1000000, 10000000, 100000000],
                default: 10000000,
                format: formatArray,
              },
            ],
            minGpuSize: 10000000,
          },
          {
            id: 'multiply',
            name: 'Multiplication',
            description: 'Element-wise A × B',
            operation: 'multiply',
            dimensions: [
              {
                label: 'Array Size',
                values: [1000000, 10000000, 100000000],
                default: 10000000,
                format: formatArray,
              },
            ],
            minGpuSize: 10000000,
          },
          {
            id: 'abs',
            name: 'Absolute Value',
            description: 'Element-wise |x|',
            operation: 'abs',
            dimensions: [
              {
                label: 'Array Size',
                values: [1000000, 10000000, 100000000],
                default: 10000000,
                format: formatArray,
              },
            ],
            minGpuSize: 10000000,
          },
        ],
      },
    ],
  },
  {
    id: 'reductions',
    name: 'Reduction Operations',
    icon: '∑',
    categories: [
      {
        id: 'basic-reductions',
        name: 'Basic Reductions',
        description: 'Sum, mean, min, max',
        tests: [
          {
            id: 'sum',
            name: 'Sum',
            description: 'Sum all elements',
            operation: 'sum',
            dimensions: [
              {
                label: 'Array Size',
                values: [1000000, 10000000, 100000000],
                default: 10000000,
                format: formatArray,
              },
            ],
            minGpuSize: 1000000,
          },
          {
            id: 'mean',
            name: 'Mean',
            description: 'Average of all elements',
            operation: 'mean',
            dimensions: [
              {
                label: 'Array Size',
                values: [1000000, 10000000, 100000000],
                default: 10000000,
                format: formatArray,
              },
            ],
            minGpuSize: 1000000,
          },
          {
            id: 'max',
            name: 'Maximum',
            description: 'Find maximum element',
            operation: 'max',
            dimensions: [
              {
                label: 'Array Size',
                values: [1000000, 10000000, 100000000],
                default: 10000000,
                format: formatArray,
              },
            ],
            minGpuSize: 1000000,
          },
          {
            id: 'min',
            name: 'Minimum',
            description: 'Find minimum element',
            operation: 'min',
            dimensions: [
              {
                label: 'Array Size',
                values: [1000000, 10000000, 100000000],
                default: 10000000,
                format: formatArray,
              },
            ],
            minGpuSize: 1000000,
          },
        ],
      },
      {
        id: 'axis-reductions',
        name: 'Axis-wise Reductions',
        description: 'Reductions along specific axes',
        tests: [
          {
            id: 'sum-axis',
            name: 'Sum Along Axis',
            description: 'Sum rows or columns',
            operation: 'sum_axis',
            dimensions: [
              {
                label: 'Matrix Size',
                values: [512, 1024, 2048, 4096],
                default: 1024,
                format: formatMatrix,
              },
              {
                label: 'Axis',
                values: [0, 1],
                default: 0,
              },
            ],
            minGpuSize: 1024,
          },
          {
            id: 'mean-axis',
            name: 'Mean Along Axis',
            description: 'Average rows or columns',
            operation: 'mean_axis',
            dimensions: [
              {
                label: 'Matrix Size',
                values: [512, 1024, 2048, 4096],
                default: 1024,
                format: formatMatrix,
              },
              {
                label: 'Axis',
                values: [0, 1],
                default: 0,
              },
            ],
            minGpuSize: 1024,
          },
        ],
      },
      {
        id: 'statistical',
        name: 'Statistical Reductions',
        description: 'Variance, standard deviation',
        tests: [
          {
            id: 'variance',
            name: 'Variance',
            description: 'Sample variance',
            operation: 'variance',
            dimensions: [
              {
                label: 'Array Size',
                values: [1000000, 10000000, 100000000],
                default: 10000000,
                format: formatArray,
              },
            ],
            minGpuSize: 1000000,
          },
          {
            id: 'std',
            name: 'Standard Deviation',
            description: 'Sample standard deviation',
            operation: 'std',
            dimensions: [
              {
                label: 'Array Size',
                values: [1000000, 10000000, 100000000],
                default: 10000000,
                format: formatArray,
              },
            ],
            minGpuSize: 1000000,
          },
        ],
      },
    ],
  },
  {
    id: 'transforms',
    name: 'Transforms',
    icon: '⇄',
    categories: [
      {
        id: 'fft',
        name: 'FFT Operations',
        description: 'Fast Fourier Transforms',
        tests: [
          {
            id: 'fft-1d',
            name: '1D FFT',
            description: 'One-dimensional FFT',
            operation: 'fft',
            dimensions: [
              {
                label: 'Array Size',
                values: [1024, 4096, 16384, 65536, 262144],
                default: 16384,
                format: formatArray,
              },
            ],
            minGpuSize: 4096,
          },
          {
            id: 'ifft-1d',
            name: '1D IFFT',
            description: 'Inverse FFT',
            operation: 'ifft',
            dimensions: [
              {
                label: 'Array Size',
                values: [1024, 4096, 16384, 65536, 262144],
                default: 16384,
                format: formatArray,
              },
            ],
            minGpuSize: 4096,
          },
          {
            id: 'fft-2d',
            name: '2D FFT',
            description: 'Two-dimensional FFT',
            operation: 'fft2',
            dimensions: [
              {
                label: 'Image Size',
                values: [128, 256, 512, 1024, 2048],
                default: 512,
                format: formatMatrix,
              },
            ],
            minGpuSize: 256,
          },
        ],
      },
      {
        id: 'convolution',
        name: 'Convolution',
        description: 'Convolution operations',
        tests: [
          {
            id: 'conv1d',
            name: '1D Convolution',
            description: 'Signal convolution',
            operation: 'conv1d',
            dimensions: [
              {
                label: 'Signal Length',
                values: [10000, 100000, 1000000],
                default: 100000,
                format: formatArray,
              },
              {
                label: 'Kernel Size',
                values: [3, 5, 11, 21, 51],
                default: 11,
              },
            ],
            minGpuSize: 10000,
          },
          {
            id: 'conv2d',
            name: '2D Convolution',
            description: 'Image convolution',
            operation: 'conv2d',
            dimensions: [
              {
                label: 'Image Size',
                values: [256, 512, 1024, 2048],
                default: 512,
                format: formatMatrix,
              },
              {
                label: 'Kernel Size',
                values: [3, 5, 7, 11],
                default: 5,
                format: formatMatrix,
              },
            ],
            minGpuSize: 256,
          },
        ],
      },
    ],
  },
  {
    id: 'sorting',
    name: 'Sorting & Search',
    icon: '⇅',
    categories: [
      {
        id: 'sorting',
        name: 'Sorting Algorithms',
        description: 'Sort operations',
        tests: [
          {
            id: 'sort',
            name: 'Sort',
            description: 'Full array sort',
            operation: 'sort',
            dimensions: [
              {
                label: 'Array Size',
                values: [10000, 100000, 1000000, 10000000],
                default: 1000000,
                format: formatArray,
              },
            ],
            minGpuSize: 100000,
          },
          {
            id: 'argsort',
            name: 'Argsort',
            description: 'Return sorting indices',
            operation: 'argsort',
            dimensions: [
              {
                label: 'Array Size',
                values: [10000, 100000, 1000000, 10000000],
                default: 1000000,
                format: formatArray,
              },
            ],
            minGpuSize: 100000,
          },
        ],
      },
      {
        id: 'search',
        name: 'Search Operations',
        description: 'Find elements',
        tests: [
          {
            id: 'argmax',
            name: 'Argmax',
            description: 'Index of maximum',
            operation: 'argmax',
            dimensions: [
              {
                label: 'Array Size',
                values: [1000000, 10000000, 100000000],
                default: 10000000,
                format: formatArray,
              },
            ],
            minGpuSize: 1000000,
          },
          {
            id: 'argmin',
            name: 'Argmin',
            description: 'Index of minimum',
            operation: 'argmin',
            dimensions: [
              {
                label: 'Array Size',
                values: [1000000, 10000000, 100000000],
                default: 10000000,
                format: formatArray,
              },
            ],
            minGpuSize: 1000000,
          },
        ],
      },
    ],
  },
  {
    id: 'image',
    name: 'Image Processing',
    icon: '🖼',
    categories: [
      {
        id: 'filters',
        name: 'Image Filters',
        description: 'Blur, edge detection',
        tests: [
          {
            id: 'gaussian-blur',
            name: 'Gaussian Blur',
            description: 'Gaussian blur filter',
            operation: 'gaussian_blur',
            dimensions: [
              {
                label: 'Image Size',
                values: [256, 512, 1024, 2048, 4096],
                default: 1024,
                format: formatMatrix,
              },
              {
                label: 'Sigma',
                values: [1, 2, 3, 5],
                default: 2,
              },
            ],
            minGpuSize: 512,
          },
          {
            id: 'sobel',
            name: 'Sobel Edge Detection',
            description: 'Edge detection filter',
            operation: 'sobel',
            dimensions: [
              {
                label: 'Image Size',
                values: [256, 512, 1024, 2048, 4096],
                default: 1024,
                format: formatMatrix,
              },
            ],
            minGpuSize: 512,
          },
        ],
      },
      {
        id: 'morphology',
        name: 'Morphological Operations',
        description: 'Erosion, dilation',
        tests: [
          {
            id: 'erosion',
            name: 'Erosion',
            description: 'Morphological erosion',
            operation: 'erosion',
            dimensions: [
              {
                label: 'Image Size',
                values: [256, 512, 1024, 2048],
                default: 512,
                format: formatMatrix,
              },
              {
                label: 'Kernel Size',
                values: [3, 5, 7],
                default: 3,
              },
            ],
            minGpuSize: 256,
          },
          {
            id: 'dilation',
            name: 'Dilation',
            description: 'Morphological dilation',
            operation: 'dilation',
            dimensions: [
              {
                label: 'Image Size',
                values: [256, 512, 1024, 2048],
                default: 512,
                format: formatMatrix,
              },
              {
                label: 'Kernel Size',
                values: [3, 5, 7],
                default: 3,
              },
            ],
            minGpuSize: 256,
          },
        ],
      },
    ],
  },
  {
    id: 'batch',
    name: 'Batch Operations',
    icon: '📦',
    categories: [
      {
        id: 'batch-matmul',
        name: 'Batch Matrix Operations',
        description: 'Multiple matrix operations',
        tests: [
          {
            id: 'batch-matmul',
            name: 'Batch MatMul',
            description: 'Multiple matrix multiplications',
            operation: 'batch_matmul',
            dimensions: [
              {
                label: 'Batch Size',
                values: [8, 16, 32, 64, 128],
                default: 32,
              },
              {
                label: 'Matrix Size',
                values: [64, 128, 256, 512],
                default: 128,
                format: formatMatrix,
              },
            ],
            minGpuSize: 128,
          },
          {
            id: 'batch-dot',
            name: 'Batch Dot Product',
            description: 'Multiple dot products',
            operation: 'batch_dot',
            dimensions: [
              {
                label: 'Batch Size',
                values: [100, 1000, 10000],
                default: 1000,
              },
              {
                label: 'Vector Size',
                values: [1000, 10000, 100000],
                default: 10000,
                format: formatArray,
              },
            ],
            minGpuSize: 1000,
          },
        ],
      },
    ],
  },
];

/**
 * Flatten hierarchy for search
 */
export function getAllTests() {
  const tests: Array<{
    groupId: string;
    groupName: string;
    categoryId: string;
    categoryName: string;
    test: any;
  }> = [];

  for (const group of TEST_HIERARCHY) {
    for (const category of group.categories) {
      for (const test of category.tests) {
        tests.push({
          groupId: group.id,
          groupName: group.name,
          categoryId: category.id,
          categoryName: category.name,
          test,
        });
      }
    }
  }

  return tests;
}

/**
 * Find test by ID
 */
export function findTestById(testId: string) {
  const allTests = getAllTests();
  return allTests.find(t => t.test.id === testId);
}
