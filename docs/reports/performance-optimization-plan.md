# NumPy-Rust Performance Optimization Plan

**Date:** November 9, 2025
**Author:** Performance Analysis Team
**Target:** 2-10x performance improvements beyond current benchmarks

## Executive Summary

This document outlines a comprehensive plan to achieve significant performance improvements in the numpy-rust library. Current benchmarks show we're already **2-100x faster** than Python NumPy (via PyO3), but there are substantial opportunities to improve pure Rust performance even further through:

1. **SIMD vectorization** (2-4x potential speedup)
2. **Parallel processing with Rayon** (2-8x on multi-core systems)
3. **Memory optimization and cache locality** (1.5-3x improvement)
4. **Specialized fast paths** (2-5x for common cases)
5. **Algorithm improvements** (varies by operation)

**Estimated aggregate improvement:** 5-20x faster for hot-path operations on modern hardware.

---

## Current Performance Baseline

### Internal Rust Benchmarks (Pure Rust Performance)

| Operation | Size | Time | Notes |
|-----------|------|------|-------|
| zeros | 100 | 80 ns | Allocation-bound |
| ones | 100 | 65 ns | Allocation-bound |
| linspace | 100 | 71 ns | Compute-bound |
| sin | 10K | ~93 µs | Element-wise, SIMD opportunity |
| sum | 10K | ~1 µs | Reduction, parallelizable |
| matmul | 50x50 | ~15 µs | BLAS opportunity |

### Key Performance Characteristics

1. **Small arrays (< 1000)**: Dominated by allocation overhead
2. **Medium arrays (1K-10K)**: Balanced compute and memory
3. **Large arrays (> 10K)**: Compute-bound, best SIMD/parallel targets

---

## Optimization Strategy

### Phase 1: Low-Hanging Fruit (Estimated 2-5x improvement)

#### 1.1 Explicit SIMD for Math Operations
**Priority:** HIGH | **Effort:** Medium | **Impact:** 2-4x

**Current State:**
```rust
pub fn sin<S, D>(arr: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    arr.mapv(|x| x.sin())  // Relies on LLVM auto-vectorization
}
```

**Problems:**
- `mapv()` may not always vectorize
- No explicit SIMD intrinsics
- No architecture-specific optimizations

**Solution:**
- Use `packed_simd` or `std::simd` for explicit vectorization
- Implement AVX2/AVX-512 fast paths for f32/f64
- Fall back to scalar for unsupported types/platforms

**Implementation Plan:**
```rust
// Example optimized sin with explicit SIMD
#[cfg(target_feature = "avx2")]
pub fn sin_f64_avx2(arr: &Array1<f64>) -> Array1<f64> {
    // Process 4 f64s at a time with AVX2
    // Remainder handled by scalar code
}

pub fn sin<S, D>(arr: &ArrayBase<S, D>) -> Array<S::Elem, D> {
    #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
    if TypeId::of::<S::Elem>() == TypeId::of::<f64>() {
        // Use SIMD fast path
    }
    // Fallback to current implementation
}
```

**Affected Operations:**
- All trigonometric: sin, cos, tan, arcsin, arccos, arctan
- Exponential: exp, log, sqrt, power
- Hyperbolic: sinh, cosh, tanh

**Benchmark Target:** 2-4x faster for arrays > 1000 elements

---

#### 1.2 Parallel Processing with Rayon
**Priority:** HIGH | **Effort:** Low-Medium | **Impact:** 2-8x (CPU-dependent)

**Current State:**
- Rayon is available in dependencies
- Not used in most operations
- No parallel versions of key functions

**Solution:**
Implement parallel versions for operations on large arrays:

```rust
use rayon::prelude::*;

pub fn sum_parallel<S, D>(arr: &ArrayBase<S, D>) -> S::Elem
where
    S: Data + Sync,
    S::Elem: Float + Send + Sync,
    D: Dimension,
{
    arr.par_iter()
        .cloned()
        .reduce(|| Zero::zero(), |a, b| a + b)
}

// Threshold-based dispatch
pub fn sum<S, D>(arr: &ArrayBase<S, D>) -> S::Elem {
    if arr.len() > 10_000 {
        sum_parallel(arr)
    } else {
        sum_sequential(arr)
    }
}
```

**Affected Operations:**
- Reductions: sum, prod, mean, var, std
- Element-wise: sin, cos, exp, log (large arrays)
- Linear algebra: matrix multiplication (already uses matrixmultiply with parallelism)

**Benchmark Target:** 4-8x on 8-core CPU for large arrays

---

#### 1.3 In-Place Operations
**Priority:** MEDIUM | **Effort:** Medium | **Impact:** 1.5-2x

**Current State:**
- All operations allocate new arrays
- No in-place variants available
- Unnecessary copies in chained operations

**Solution:**
Add `_mut` variants that operate in-place:

```rust
pub fn sin_mut<S, D>(arr: &mut ArrayBase<S, D>)
where
    S: DataMut,
    S::Elem: Float,
    D: Dimension,
{
    arr.mapv_inplace(|x| x.sin())
}

// Usage eliminates allocation
let mut arr = Array1::linspace(0.0, 100.0, 10000);
sin_mut(&mut arr);  // No allocation
```

**Affected Operations:**
- All math operations
- Clipping, rounding
- Element-wise transformations

**Benchmark Target:** 1.5-2x for operations in hot loops

---

### Phase 2: Algorithm Optimizations (Estimated 2-5x improvement)

#### 2.1 Faster Statistical Functions
**Priority:** MEDIUM | **Effort:** Medium | **Impact:** 2-5x

**Current Median Implementation:**
```rust
pub fn median<S>(arr: &ArrayBase<S, ndarray::Ix1>) -> Result<S::Elem>
{
    let mut sorted = arr.to_vec();  // O(n) allocation
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());  // O(n log n)
    // ...
}
```

**Problems:**
- Full sort for median is O(n log n)
- Unnecessary allocation
- No quickselect algorithm

**Solution:**
```rust
// Use quickselect for O(n) average case
pub fn median_quickselect<S>(arr: &ArrayBase<S, ndarray::Ix1>) -> Result<S::Elem>
{
    // Quickselect algorithm - O(n) average, O(n²) worst
    // Much faster than full sort for single percentile
}
```

**Also Optimize:**
- `percentile`: Use interpolated quickselect
- `var`/`std`: Single-pass Welford's algorithm (currently may be two-pass)
- `corrcoef`: Optimize with blocked computation

**Benchmark Target:** 3-5x faster median, 1.5-2x faster var/std

---

#### 2.2 Specialized Fast Paths
**Priority:** MEDIUM | **Effort:** Low-Medium | **Impact:** 2-10x (specific cases)

**Opportunities:**

1. **Power-of-2 FFT sizes:**
   ```rust
   // Current: works for all sizes
   // Optimized: ultra-fast for 2^n sizes
   pub fn fft_pow2<T>(arr: &Array1<Complex<T>>) -> Array1<Complex<T>>
   ```

2. **Small matrix operations:**
   ```rust
   // Unrolled 2x2, 3x3, 4x4 matrix multiply
   #[inline(always)]
   pub fn matmul_2x2(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
       // Fully unrolled, no loops
   }
   ```

3. **Aligned array operations:**
   ```rust
   // Fast path for 32-byte aligned arrays
   // Use aligned SIMD loads/stores
   ```

**Benchmark Target:** 5-10x for specialized cases

---

#### 2.3 Cache-Friendly Memory Layout
**Priority:** MEDIUM | **Effort:** High | **Impact:** 1.5-3x

**Current Issues:**
- Matrix operations may have poor cache locality
- Transpose creates temporary copies
- No explicit cache blocking

**Solutions:**

1. **Blocked Matrix Multiplication:**
   ```rust
   // Cache-blocked matmul for large matrices
   const BLOCK_SIZE: usize = 64;  // Tune to L1 cache

   pub fn matmul_blocked<T>(a: &Array2<T>, b: &Array2<T>) -> Array2<T> {
       // Multiply in blocks to keep data in L1/L2 cache
   }
   ```

2. **Strided Access Optimization:**
   ```rust
   // Optimize operations on non-contiguous arrays
   // Consider copying to contiguous if beneficial
   ```

**Benchmark Target:** 2-3x for large matrix operations

---

### Phase 3: Advanced Optimizations (Estimated 1.5-3x improvement)

#### 3.1 Custom Allocators
**Priority:** LOW | **Effort:** High | **Impact:** 1.2-2x

**Opportunities:**
- Arena allocators for temporary arrays
- Memory pools for common sizes
- Aligned allocation for SIMD

**Implementation:**
```rust
// Custom allocator for SIMD-aligned arrays
pub struct AlignedAllocator<const ALIGN: usize>;

// Usage
let arr: Array1<f64, AlignedAllocator<32>> = aligned_zeros(1000);
```

---

#### 3.2 Compile-Time Specialization
**Priority:** LOW | **Effort:** Medium | **Impact:** 1.5-2x

**Opportunities:**
- Const generics for array sizes
- Type-specialized implementations
- Compile-time shape validation

```rust
// Zero-cost array operations at compile time
pub fn matmul_static<const M: usize, const N: usize, const P: usize>(
    a: &Array2<f64>,
    b: &Array2<f64>,
) -> Array2<f64> {
    // Compiler knows sizes, can optimize fully
}
```

---

#### 3.3 Lazy Evaluation / Expression Templates
**Priority:** LOW | **Effort:** Very High | **Impact:** 2-5x (for chained ops)

**Goal:** Eliminate intermediate allocations in chained operations

**Current:**
```rust
let result = sin(&exp(&(a + b)));  // 3 allocations
```

**Optimized:**
```rust
let result = (a + b).exp().sin().eval();  // 1 allocation
```

**Implementation:** Would require significant library redesign

---

## Benchmark Expansion Plan

### New Benchmarks Needed

1. **SIMD-specific benchmarks:**
   - Compare scalar vs SIMD implementations
   - Test different array sizes (16, 64, 256, 1024, 4096, 16384)
   - Measure vectorization efficiency

2. **Parallel benchmarks:**
   - Single-thread vs multi-thread
   - Scaling with core count (1, 2, 4, 8, 16 cores)
   - Overhead measurement for small arrays

3. **Memory benchmarks:**
   - Allocation overhead
   - Cache miss rates (using perf counters)
   - Memory bandwidth utilization

4. **Real-world workloads:**
   - Image processing pipeline
   - Machine learning forward pass
   - Signal processing chain
   - Monte Carlo simulation

---

## Implementation Roadmap

### Sprint 1: Quick Wins (1-2 weeks)
- [ ] Add parallel implementations for sum, mean, var, std
- [ ] Implement threshold-based dispatch (parallel vs sequential)
- [ ] Add comprehensive benchmarks for parallel operations
- [ ] **Expected gain:** 3-5x on multi-core for large arrays

### Sprint 2: SIMD Math Operations (2-3 weeks)
- [ ] Implement AVX2 fast paths for sin, cos, exp, log
- [ ] Add runtime CPU feature detection
- [ ] Comprehensive SIMD benchmarks
- [ ] **Expected gain:** 2-4x for vectorizable operations

### Sprint 3: In-Place Operations (1-2 weeks)
- [ ] Add `_mut` variants for all math operations
- [ ] Update API documentation
- [ ] Benchmarks comparing allocation vs in-place
- [ ] **Expected gain:** 1.5-2x for in-place use cases

### Sprint 4: Algorithm Improvements (2-3 weeks)
- [ ] Quickselect for median and percentile
- [ ] Welford's algorithm for var/std (if not already)
- [ ] Blocked matrix multiplication
- [ ] **Expected gain:** 2-5x for statistical operations

### Sprint 5: Specialized Fast Paths (1-2 weeks)
- [ ] Unrolled small matrix operations
- [ ] Power-of-2 FFT optimization
- [ ] Aligned array fast paths
- [ ] **Expected gain:** 5-10x for specialized cases

---

## Measurement and Validation

### Performance Testing Infrastructure

1. **Automated benchmarking:**
   ```bash
   cargo bench --bench comprehensive_perf > results.txt
   ```

2. **Regression detection:**
   - Store baseline results
   - Compare each commit against baseline
   - Alert on >5% performance degradation

3. **Hardware diversity:**
   - Test on: Intel x86_64, AMD x86_64, ARM64
   - Various core counts: 2, 4, 8, 16
   - Different cache sizes

4. **Profiling tools:**
   - `perf stat` for CPU counters
   - `flamegraph` for hotspot identification
   - `valgrind --tool=cachegrind` for cache analysis

---

## Risk Assessment

### Low Risk
- Parallel implementations (Rayon well-tested)
- In-place operations (straightforward)
- Additional benchmarks (no production impact)

### Medium Risk
- SIMD implementations (platform-specific, need testing)
- Algorithm changes (need correctness validation)
- Threshold tuning (may need adjustment per workload)

### High Risk
- Custom allocators (can introduce bugs)
- Lazy evaluation (major API redesign)
- Breaking changes to public API

---

## Success Metrics

### Primary Metrics
1. **Throughput:** Operations per second (↑ 5-20x target)
2. **Latency:** Time per operation (↓ 5-20x target)
3. **Memory:** Peak allocation (↓ 30-50% for in-place)
4. **Scalability:** Speedup vs core count (linear to 8 cores)

### Secondary Metrics
1. **Compilation time:** Should not increase >20%
2. **Binary size:** Should not increase >30%
3. **Code complexity:** Maintain readability
4. **Test coverage:** Maintain >90%

---

## Conclusion

This optimization plan provides a clear path to making numpy-rust **5-20x faster** than current implementation through systematic application of:
- SIMD vectorization
- Parallel processing
- Memory optimization
- Algorithm improvements

The plan is structured in phases with increasing complexity, allowing for incremental improvements and validation at each step.

**Next Steps:**
1. Review and approve this plan
2. Begin Sprint 1 (parallel operations)
3. Establish performance tracking infrastructure
4. Execute optimization sprints iteratively

**Estimated total effort:** 10-15 weeks for full implementation
**Expected performance gain:** 5-20x for hot-path operations
**Compatibility:** Maintain full API compatibility (except additions)

---

## Appendix: Reference Implementations

### A. Competitive Analysis

**NumPy (Python + C):**
- Uses BLAS/LAPACK for linear algebra
- SIMD via NumPy's internal vectorization
- Parallel via OpenMP in some operations
- Limited by GIL for pure-Python code

**Eigen (C++):**
- Extensive SIMD (SSE, AVX, AVX-512, NEON)
- Expression templates for zero-cost abstraction
- Excellent cache blocking
- Our performance target for pure-Rust code

**Julia:**
- LLVM-based optimization
- Excellent SIMD generation
- Parallel primitives
- Good reference for algorithm selection

### B. Useful Crates

- `packed_simd` - Portable SIMD
- `rayon` - Data parallelism (already included)
- `ndarray-parallel` - Parallel ndarray operations
- `criterion` - Benchmarking (already included)
- `perf-event` - Hardware performance counters

---

**Document Version:** 1.0
**Last Updated:** November 9, 2025
**Status:** Draft for Review
