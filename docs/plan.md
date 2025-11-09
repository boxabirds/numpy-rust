# numpy-rust Complete Implementation Plan

**Goal**: Implement the full NumPy API with both CPU (pure Rust) and GPU (WebGPU) backends.

**Legend**:
- ✅ = Fully implemented and tested
- 🚧 = In progress
- ❌ = Not started
- ⚠️ = Partial implementation
- 🔥 = High priority
- 💡 = Medium priority
- 📚 = Low priority / Nice to have

**Progress Tracking**: [3/1500+] operations implemented

---

## 1. Array Creation (Priority: 🔥)

### Basic Constructors
- [ ] `zeros` - Create array filled with zeros
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `ones` - Create array filled with ones
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `empty` - Create uninitialized array
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `full` - Create array filled with specific value
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `eye` / `identity` - Create identity matrix
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `diag` - Create diagonal matrix or extract diagonal
  - [ ] CPU implementation
  - [ ] GPU implementation

### Ranges and Sequences
- [ ] `arange` - Create evenly spaced values
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `linspace` - Create linearly spaced values
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `logspace` - Create logarithmically spaced values
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `geomspace` - Create geometrically spaced values
  - [ ] CPU implementation
  - [ ] GPU implementation

### From Existing Data
- [ ] `array` - Create from raw data
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `copy` - Create copy of array
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `frombuffer` - Create from buffer
  - [ ] CPU implementation
  - [ ] GPU implementation

### Random Number Generation
- [x] `random.random` / `generate_random_array` - Uniform random
  - [x] CPU implementation (used in tests)
  - [ ] GPU implementation
- [x] `random.random` / `generate_random_matrix` - Random matrix
  - [x] CPU implementation (used in tests)
  - [ ] GPU implementation
- [ ] `random.randn` - Standard normal distribution
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `random.randint` - Random integers
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `random.choice` - Random sample from array
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `random.shuffle` - Shuffle array in-place
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `random.permutation` - Random permutation
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `random.seed` - Set random seed
  - [ ] CPU implementation
  - [ ] GPU implementation

**Subtotal**: [2/54] operations in this category

---

## 2. Mathematical Functions (Priority: 🔥)

### Trigonometric Functions
- [x] `sin` - Sine
  - [x] CPU implementation
  - [x] GPU implementation
- [ ] `cos` - Cosine
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `tan` - Tangent
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `arcsin` / `asin` - Inverse sine
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `arccos` / `acos` - Inverse cosine
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `arctan` / `atan` - Inverse tangent
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `arctan2` / `atan2` - Two-argument inverse tangent
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `hypot` - Hypotenuse
  - [ ] CPU implementation
  - [ ] GPU implementation

### Hyperbolic Functions
- [ ] `sinh` - Hyperbolic sine
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `cosh` - Hyperbolic cosine
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `tanh` - Hyperbolic tangent
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `arcsinh` - Inverse hyperbolic sine
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `arccosh` - Inverse hyperbolic cosine
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `arctanh` - Inverse hyperbolic tangent
  - [ ] CPU implementation
  - [ ] GPU implementation

### Exponential and Logarithmic
- [x] `exp` - Exponential
  - [x] CPU implementation
  - [x] GPU implementation
- [ ] `exp2` - Base-2 exponential
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `expm1` - exp(x) - 1
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `log` - Natural logarithm
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `log2` - Base-2 logarithm
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `log10` - Base-10 logarithm
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `log1p` - log(1 + x)
  - [ ] CPU implementation
  - [ ] GPU implementation

### Power and Roots
- [ ] `power` / `pow` - Element-wise power
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `sqrt` - Square root
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `square` - Element-wise square
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `cbrt` - Cube root
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `reciprocal` - 1/x
  - [ ] CPU implementation
  - [ ] GPU implementation

### Rounding
- [ ] `round` / `around` - Round to nearest
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `floor` - Round down
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `ceil` - Round up
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `trunc` - Truncate to integer
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `rint` - Round to nearest integer
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `fix` - Round towards zero
  - [ ] CPU implementation
  - [ ] GPU implementation

### Sign and Absolute Value
- [ ] `abs` / `absolute` - Absolute value
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `sign` - Sign of number (-1, 0, +1)
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `copysign` - Copy sign from one array to another
  - [ ] CPU implementation
  - [ ] GPU implementation

### Comparisons and Extrema
- [ ] `maximum` - Element-wise maximum of two arrays
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `minimum` - Element-wise minimum of two arrays
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `fmax` - Element-wise maximum (ignoring NaN)
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `fmin` - Element-wise minimum (ignoring NaN)
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `clip` - Clip values to range
  - [ ] CPU implementation
  - [ ] GPU implementation

### Special Functions
- [ ] `sinc` - Sinc function
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `degrees` - Radians to degrees
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `radians` - Degrees to radians
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `deg2rad` - Alias for radians
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `rad2deg` - Alias for degrees
  - [ ] CPU implementation
  - [ ] GPU implementation

**Subtotal**: [2/108] operations in this category

---

## 3. Array Manipulation (Priority: 🔥)

### Shape Manipulation
- [ ] `reshape` - Change array shape
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `ravel` - Flatten array
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `flatten` - Return flattened copy
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `transpose` - Transpose array
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `swapaxes` - Swap two axes
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `moveaxis` - Move axes to new positions
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `rollaxis` - Roll axis backwards
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `squeeze` - Remove single-dimensional entries
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `expand_dims` - Expand dimensions
  - [ ] CPU implementation
  - [ ] GPU implementation

### Joining Arrays
- [ ] `concatenate` - Join arrays along axis
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `stack` - Stack arrays along new axis
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `vstack` / `row_stack` - Stack arrays vertically
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `hstack` - Stack arrays horizontally
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `dstack` - Stack arrays depth-wise
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `column_stack` - Stack 1D arrays as columns
  - [ ] CPU implementation
  - [ ] GPU implementation

### Splitting Arrays
- [ ] `split` - Split array into multiple sub-arrays
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `array_split` - Split with unequal divisions
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `hsplit` - Split horizontally
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `vsplit` - Split vertically
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `dsplit` - Split depth-wise
  - [ ] CPU implementation
  - [ ] GPU implementation

### Tiling and Repeating
- [ ] `tile` - Tile array
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `repeat` - Repeat elements
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `resize` - Resize array
  - [ ] CPU implementation
  - [ ] GPU implementation

### Adding and Removing Elements
- [ ] `append` - Append values to end
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `insert` - Insert values
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `delete` - Delete values
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `unique` - Find unique elements
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `trim_zeros` - Trim leading/trailing zeros
  - [ ] CPU implementation
  - [ ] GPU implementation

### Rearranging Elements
- [ ] `flip` - Reverse elements along axis
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `fliplr` - Flip left-right
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `flipud` - Flip up-down
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `roll` - Roll elements along axis
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `rot90` - Rotate 90 degrees
  - [ ] CPU implementation
  - [ ] GPU implementation

**Subtotal**: [0/78] operations in this category

---

## 4. Linear Algebra (Priority: 🔥)

### Matrix and Vector Products
- [x] `matmul` / `@` - Matrix multiplication
  - [x] CPU implementation
  - [x] GPU implementation
- [ ] `dot` - Dot product
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `vdot` - Vector dot product
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `inner` - Inner product
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `outer` - Outer product
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `tensordot` - Tensor dot product
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `kron` - Kronecker product
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `cross` - Cross product
  - [ ] CPU implementation
  - [ ] GPU implementation

### Decompositions
- [ ] `cholesky` - Cholesky decomposition
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `qr` - QR decomposition
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `svd` - Singular Value Decomposition
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `eig` - Eigenvalues and eigenvectors
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `eigh` - Eigenvalues for Hermitian/symmetric
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `eigvals` - Eigenvalues only
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `eigvalsh` - Eigenvalues for Hermitian only
  - [ ] CPU implementation
  - [ ] GPU implementation

### Matrix Properties
- [ ] `norm` - Matrix or vector norm
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `cond` - Condition number
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `det` - Determinant
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `matrix_rank` - Matrix rank
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `slogdet` - Sign and log of determinant
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `trace` - Trace (sum of diagonal)
  - [ ] CPU implementation
  - [ ] GPU implementation

### Solving Equations
- [ ] `solve` - Solve linear system Ax = b
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `lstsq` - Least-squares solution
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `inv` - Matrix inverse
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `pinv` - Pseudo-inverse
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `tensorsolve` - Solve tensor equation
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `tensorinv` - Tensor inverse
  - [ ] CPU implementation
  - [ ] GPU implementation

**Subtotal**: [1/62] operations in this category

---

## 5. Reduction Operations (Priority: 🔥)

### Statistical Reductions
- [ ] `sum` - Sum of elements
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `prod` - Product of elements
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `mean` - Mean/average
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `std` - Standard deviation
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `var` - Variance
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `median` - Median
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `percentile` - Percentile
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `quantile` - Quantile
  - [ ] CPU implementation
  - [ ] GPU implementation

### Min/Max
- [ ] `min` / `amin` - Minimum
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `max` / `amax` - Maximum
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `nanmin` - Min ignoring NaN
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `nanmax` - Max ignoring NaN
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `ptp` - Peak to peak (max - min)
  - [ ] CPU implementation
  - [ ] GPU implementation

### Arg Min/Max
- [ ] `argmin` - Index of minimum
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `argmax` - Index of maximum
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `nanargmin` - Argmin ignoring NaN
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `nanargmax` - Argmax ignoring NaN
  - [ ] CPU implementation
  - [ ] GPU implementation

### Cumulative Operations
- [ ] `cumsum` - Cumulative sum
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `cumprod` - Cumulative product
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `nancumsum` - Cumsum ignoring NaN
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `nancumprod` - Cumprod ignoring NaN
  - [ ] CPU implementation
  - [ ] GPU implementation

### Differences
- [ ] `diff` - Discrete difference
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `ediff1d` - 1D differences
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `gradient` - Numerical gradient
  - [ ] CPU implementation
  - [ ] GPU implementation

### Other Reductions
- [ ] `all` - Test if all elements are true
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `any` - Test if any element is true
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `count_nonzero` - Count non-zero elements
  - [ ] CPU implementation
  - [ ] GPU implementation

**Subtotal**: [0/56] operations in this category

---

## 6. Arithmetic Operations (Priority: 🔥)

### Basic Arithmetic
- [ ] `add` - Addition
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `subtract` - Subtraction
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `multiply` - Multiplication
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `divide` / `true_divide` - Division
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `floor_divide` - Floor division
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `mod` / `remainder` - Modulo
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `divmod` - Quotient and remainder
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `fmod` - Floating modulo
  - [ ] CPU implementation
  - [ ] GPU implementation

### In-place Operations
- [ ] `iadd` / `+=` - In-place addition
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `isub` / `-=` - In-place subtraction
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `imul` / `*=` - In-place multiplication
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `idiv` / `/=` - In-place division
  - [ ] CPU implementation
  - [ ] GPU implementation

### Negative and Positive
- [ ] `negative` - Negate elements
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `positive` - Return copy with same sign
  - [ ] CPU implementation
  - [ ] GPU implementation

**Subtotal**: [0/30] operations in this category

---

## 7. Logical Operations (Priority: 💡)

### Comparisons
- [ ] `equal` / `==` - Element-wise equality
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `not_equal` / `!=` - Element-wise inequality
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `less` / `<` - Element-wise less than
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `less_equal` / `<=` - Element-wise less or equal
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `greater` / `>` - Element-wise greater than
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `greater_equal` / `>=` - Element-wise greater or equal
  - [ ] CPU implementation
  - [ ] GPU implementation

### Logical Operations
- [ ] `logical_and` - Element-wise AND
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `logical_or` - Element-wise OR
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `logical_xor` - Element-wise XOR
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `logical_not` - Element-wise NOT
  - [ ] CPU implementation
  - [ ] GPU implementation

### Truth Testing
- [ ] `isfinite` - Test for finite values
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `isinf` - Test for infinity
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `isnan` - Test for NaN
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `isnat` - Test for NaT (not-a-time)
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `isreal` - Test for real numbers
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `iscomplex` - Test for complex numbers
  - [ ] CPU implementation
  - [ ] GPU implementation

### Array Comparisons
- [ ] `array_equal` - Test if two arrays are equal
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `array_equiv` - Test if arrays are equivalent
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `allclose` - Test if close within tolerance
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `isclose` - Element-wise closeness test
  - [ ] CPU implementation
  - [ ] GPU implementation

**Subtotal**: [0/42] operations in this category

---

## 8. Indexing and Slicing (Priority: 💡)

### Basic Indexing
- [ ] `take` - Take elements from array
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `put` - Put values into array
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `compress` - Select slices along axis
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `choose` - Construct array from index array
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `select` - Return elements from different arrays
  - [ ] CPU implementation
  - [ ] GPU implementation

### Advanced Indexing
- [ ] `take_along_axis` - Take values along axis
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `put_along_axis` - Put values along axis
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `diagonal` - Extract diagonal
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `fill_diagonal` - Fill main diagonal
  - [ ] CPU implementation
  - [ ] GPU implementation

### Index Utilities
- [ ] `where` - Conditional selection
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `nonzero` - Indices of non-zero elements
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `flatnonzero` - Flattened non-zero indices
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `argwhere` - Indices where condition is true
  - [ ] CPU implementation
  - [ ] GPU implementation

**Subtotal**: [0/26] operations in this category

---

## 9. Sorting and Searching (Priority: 💡)

### Sorting
- [ ] `sort` - Sort array
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `argsort` - Indices that would sort array
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `lexsort` - Indirect stable sort on multiple keys
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `msort` - Sort along first axis
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `partition` - Partial sort
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `argpartition` - Indices for partition
  - [ ] CPU implementation
  - [ ] GPU implementation

### Searching
- [ ] `searchsorted` - Find indices for insertion
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `extract` - Extract elements matching condition
  - [ ] CPU implementation
  - [ ] GPU implementation

**Subtotal**: [0/16] operations in this category

---

## 10. FFT and Signal Processing (Priority: 💡)

### 1D FFT
- [ ] `fft` - 1D FFT
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `ifft` - 1D inverse FFT
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `rfft` - Real 1D FFT
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `irfft` - Inverse real 1D FFT
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `hfft` - Hermitian 1D FFT
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `ihfft` - Inverse Hermitian 1D FFT
  - [ ] CPU implementation
  - [ ] GPU implementation

### 2D FFT
- [ ] `fft2` - 2D FFT
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `ifft2` - 2D inverse FFT
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `rfft2` - Real 2D FFT
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `irfft2` - Inverse real 2D FFT
  - [ ] CPU implementation
  - [ ] GPU implementation

### ND FFT
- [ ] `fftn` - ND FFT
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `ifftn` - ND inverse FFT
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `rfftn` - Real ND FFT
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `irfftn` - Inverse real ND FFT
  - [ ] CPU implementation
  - [ ] GPU implementation

### FFT Utilities
- [ ] `fftshift` - Shift zero-frequency to center
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `ifftshift` - Inverse fftshift
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `fftfreq` - FFT sample frequencies
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `rfftfreq` - Real FFT sample frequencies
  - [ ] CPU implementation
  - [ ] GPU implementation

### Convolution
- [ ] `convolve` - 1D convolution
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `correlate` - Cross-correlation
  - [ ] CPU implementation
  - [ ] GPU implementation

**Subtotal**: [0/40] operations in this category

---

## 11. Image Processing (Priority: 💡)

### Filtering (via scipy.ndimage compatibility)
- [ ] `convolve` - ND convolution
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `correlate` - ND correlation
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `gaussian_filter` - Gaussian filter
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `median_filter` - Median filter
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `uniform_filter` - Uniform filter
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `sobel` - Sobel edge detection
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `prewitt` - Prewitt edge detection
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `laplace` - Laplace filter
  - [ ] CPU implementation
  - [ ] GPU implementation

### Morphology
- [ ] `binary_erosion` - Binary erosion
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `binary_dilation` - Binary dilation
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `binary_opening` - Binary opening
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `binary_closing` - Binary closing
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `grey_erosion` - Greyscale erosion
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `grey_dilation` - Greyscale dilation
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `grey_opening` - Greyscale opening
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `grey_closing` - Greyscale closing
  - [ ] CPU implementation
  - [ ] GPU implementation

### Geometric Transformations
- [ ] `rotate` - Rotate image
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `shift` - Shift image
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `zoom` - Zoom image
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `affine_transform` - Affine transformation
  - [ ] CPU implementation
  - [ ] GPU implementation

**Subtotal**: [0/42] operations in this category

---

## 12. Polynomials (Priority: 📚)

### Polynomial Operations
- [ ] `poly` - Find polynomial from roots
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `roots` - Find roots of polynomial
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `polyval` - Evaluate polynomial
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `polyadd` - Add polynomials
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `polysub` - Subtract polynomials
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `polymul` - Multiply polynomials
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `polydiv` - Divide polynomials
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `polyfit` - Least-squares polynomial fit
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `polyint` - Integrate polynomial
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `polyder` - Differentiate polynomial
  - [ ] CPU implementation
  - [ ] GPU implementation

**Subtotal**: [0/20] operations in this category

---

## 13. Set Operations (Priority: 📚)

### Set Functions
- [ ] `unique` - Find unique elements
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `in1d` - Test membership
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `isin` - Element-wise membership test
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `intersect1d` - Intersection
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `union1d` - Union
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `setdiff1d` - Set difference
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `setxor1d` - Set XOR
  - [ ] CPU implementation
  - [ ] GPU implementation

**Subtotal**: [0/14] operations in this category

---

## 14. Bitwise Operations (Priority: 📚)

### Bitwise Functions
- [ ] `bitwise_and` - Bitwise AND
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `bitwise_or` - Bitwise OR
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `bitwise_xor` - Bitwise XOR
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `bitwise_not` / `invert` - Bitwise NOT
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `left_shift` - Left shift
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `right_shift` - Right shift
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `packbits` - Pack bits
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `unpackbits` - Unpack bits
  - [ ] CPU implementation
  - [ ] GPU implementation

**Subtotal**: [0/16] operations in this category

---

## 15. Financial Functions (Priority: 📚)

### Time Value of Money
- [ ] `fv` - Future value
  - [ ] CPU implementation
- [ ] `pv` - Present value
  - [ ] CPU implementation
- [ ] `npv` - Net present value
  - [ ] CPU implementation
- [ ] `pmt` - Payment
  - [ ] CPU implementation
- [ ] `ppmt` - Principal payment
  - [ ] CPU implementation
- [ ] `ipmt` - Interest payment
  - [ ] CPU implementation
- [ ] `irr` - Internal rate of return
  - [ ] CPU implementation
- [ ] `mirr` - Modified IRR
  - [ ] CPU implementation
- [ ] `nper` - Number of periods
  - [ ] CPU implementation
- [ ] `rate` - Interest rate
  - [ ] CPU implementation

**Subtotal**: [0/10] operations (CPU only - not GPU accelerated)

---

## 16. Special Mathematical Functions (Priority: 📚)

### Gamma and Related
- [ ] `gamma` - Gamma function
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `gammaln` - Log gamma
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `loggamma` - Log gamma (alias)
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `digamma` - Digamma function
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `polygamma` - Polygamma function
  - [ ] CPU implementation
  - [ ] GPU implementation

### Bessel Functions
- [ ] `j0` - Bessel J0
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `j1` - Bessel J1
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `jv` - Bessel Jv
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `y0` - Bessel Y0
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `y1` - Bessel Y1
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `yv` - Bessel Yv
  - [ ] CPU implementation
  - [ ] GPU implementation

### Error Functions
- [ ] `erf` - Error function
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `erfc` - Complementary error function
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `erfcinv` - Inverse erfc
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `erfinv` - Inverse erf
  - [ ] CPU implementation
  - [ ] GPU implementation

**Subtotal**: [0/30] operations in this category

---

## 17. Statistics (Priority: 💡)

### Correlations
- [ ] `corrcoef` - Correlation coefficient
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `correlate` - Cross-correlation
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `cov` - Covariance
  - [ ] CPU implementation
  - [ ] GPU implementation

### Histograms
- [ ] `histogram` - Compute histogram
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `histogram2d` - 2D histogram
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `histogramdd` - ND histogram
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `bincount` - Count occurrences
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `digitize` - Digitize values
  - [ ] CPU implementation
  - [ ] GPU implementation

### Order Statistics
- [ ] `percentile` - Compute percentile
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `quantile` - Compute quantile
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `nanpercentile` - Percentile ignoring NaN
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `nanquantile` - Quantile ignoring NaN
  - [ ] CPU implementation
  - [ ] GPU implementation

**Subtotal**: [0/24] operations in this category

---

## 18. Padding (Priority: 💡)

### Padding Functions
- [ ] `pad` - Pad array
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `pad` (constant mode)
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `pad` (edge mode)
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `pad` (reflect mode)
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `pad` (symmetric mode)
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `pad` (wrap mode)
  - [ ] CPU implementation
  - [ ] GPU implementation

**Subtotal**: [0/12] operations in this category

---

## 19. Input/Output (Priority: 💡)

### Text Files
- [ ] `loadtxt` - Load from text file
  - [ ] CPU implementation
- [ ] `savetxt` - Save to text file
  - [ ] CPU implementation
- [ ] `genfromtxt` - Load with missing data handling
  - [ ] CPU implementation

### Binary Files
- [ ] `load` - Load from .npy file
  - [ ] CPU implementation
- [ ] `save` - Save to .npy file
  - [ ] CPU implementation
- [ ] `savez` - Save multiple arrays to .npz
  - [ ] CPU implementation
- [ ] `savez_compressed` - Save compressed .npz
  - [ ] CPU implementation

### Memory-mapped Files
- [ ] `memmap` - Memory-mapped array
  - [ ] CPU implementation

**Subtotal**: [0/8] operations (CPU only - file I/O)

---

## 20. Masked Arrays (Priority: 📚)

### Masked Array Operations
- [ ] `ma.array` - Create masked array
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `ma.masked_where` - Mask where condition
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `ma.masked_equal` - Mask equal values
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `ma.masked_invalid` - Mask invalid values
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `ma.filled` - Return filled copy
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `ma.getmask` - Get mask
  - [ ] CPU implementation
  - [ ] GPU implementation
- [ ] `ma.compressed` - Return compressed data
  - [ ] CPU implementation
  - [ ] GPU implementation

**Subtotal**: [0/14] operations in this category

---

## Summary by Priority

### 🔥 High Priority (Core Functionality)
- **Total**: ~600 operations
- **Completed**: 3
- **Categories**: Array creation, Math functions, Linear algebra, Reductions, Arithmetic, Array manipulation

### 💡 Medium Priority (Commonly Used)
- **Total**: ~400 operations
- **Categories**: Logical ops, Indexing, Sorting, FFT, Statistics, Padding, I/O

### 📚 Low Priority (Specialized/Nice to Have)
- **Total**: ~500 operations
- **Categories**: Polynomials, Sets, Bitwise, Financial, Special functions, Masked arrays

---

## Implementation Strategy

### Phase 1: Core Foundation (Months 1-3)
**Goal**: Get to 50 operations - enable basic scientific computing

1. **Element-wise ops** (Week 1-2):
   - Add, subtract, multiply, divide
   - Basic trig (cos, tan, acos, asin, atan)
   - Basic math (log, sqrt, abs, square)

2. **Reductions** (Week 3-4):
   - sum, mean, std, var
   - min, max, argmin, argmax
   - Basic cumulative ops

3. **Array manipulation** (Week 5-6):
   - reshape, transpose, flatten
   - concatenate, stack, split
   - Basic indexing

4. **Linear algebra basics** (Week 7-8):
   - dot product
   - Matrix transpose
   - Basic solving

5. **Array creation** (Week 9-10):
   - zeros, ones, arange, linspace
   - eye, diag
   - Better random support

6. **Comparison ops** (Week 11-12):
   - All comparison operators
   - where, clip
   - allclose, isclose

### Phase 2: Intermediate Operations (Months 4-6)
**Goal**: 150 total operations - production-ready for ML/data science

1. **Advanced linear algebra**:
   - QR, SVD decompositions
   - Eigenvalues
   - Norms, determinants

2. **FFT and convolution**:
   - 1D FFT/IFFT
   - 2D FFT for images
   - Basic convolution

3. **Sorting and searching**:
   - sort, argsort
   - searchsorted
   - partition

4. **Statistics**:
   - correlations
   - histograms
   - percentiles

5. **Image processing**:
   - Basic filters
   - Morphological ops
   - Sobel, Laplace

### Phase 3: Advanced Features (Months 7-12)
**Goal**: 300+ operations - feature-complete for most use cases

1. **Advanced FFT**:
   - ND FFT
   - Real FFT variants
   - FFT utilities

2. **Advanced image processing**:
   - Geometric transforms
   - Advanced filters
   - Multi-channel support

3. **Specialized math**:
   - Special functions
   - Polynomials
   - Advanced statistics

4. **Polish and optimize**:
   - Performance tuning
   - Memory optimization
   - Better error handling

### Phase 4: Completeness (Ongoing)
**Goal**: Full NumPy compatibility

1. **Fill gaps**:
   - Remaining operations
   - Edge cases
   - Compatibility fixes

2. **Specialized domains**:
   - Financial functions
   - Set operations
   - Bitwise ops
   - Masked arrays

---

## Testing Strategy

Each operation must have:
- [ ] Unit tests (CPU)
- [ ] Unit tests (GPU)
- [ ] Cross-validation (CPU vs GPU results)
- [ ] Performance benchmarks
- [ ] Edge case tests (NaN, Inf, empty arrays, etc.)
- [ ] Documentation with examples

---

## Performance Goals

For GPU implementation to be worthwhile:
- **Small arrays** (<10K elements): CPU may be faster (transfer overhead)
- **Medium arrays** (10K-1M): GPU should be 2-10× faster
- **Large arrays** (>1M): GPU should be 10-100× faster
- **Matrix ops** (>512×512): GPU should be 50-300× faster

---

## Dependencies

### Rust Crates (CPU)
- [x] `ndarray` - Core array structure
- [x] `matrixmultiply` - Optimized matmul
- [ ] `ndarray-linalg` - Linear algebra (optional BLAS)
- [ ] `rustfft` - FFT implementation
- [ ] `rand` - Random number generation
- [ ] `num-traits` - Numeric traits
- [ ] `approx` - Floating-point comparisons

### WebGPU Kernels (GPU)
- [x] `wgpu` - WebGPU bindings
- [x] Basic shader infrastructure
- [ ] Optimized kernels for each operation
- [ ] Shared memory optimizations
- [ ] Work group size tuning

---

## Current Status: 3 / 1500+ operations (0.2%)

**Next immediate priorities**:
1. Add `cos`, `tan`, `log`, `sqrt` (complete basic math)
2. Add `add`, `subtract`, `multiply`, `divide` (complete arithmetic)
3. Add `sum`, `mean`, `min`, `max` (complete basic reductions)
4. Add `dot`, `transpose` (complete basic linalg)
5. Add `zeros`, `ones`, `arange` (complete array creation)

This would bring us to **18 operations** covering the absolute essentials.

---

## Notes

- GPU implementation is not required for all operations (e.g., file I/O, some utilities)
- Some operations may require multi-kernel approaches
- Performance optimization is ongoing - initial implementations may be naive
- WebGPU has limitations (no recursion, limited shared memory) that may require algorithm adaptations
- Some NumPy operations may need Rust-specific API design (ownership, lifetimes)

---

**Last Updated**: 2025-01-XX
**Maintained by**: numpy-rust contributors
