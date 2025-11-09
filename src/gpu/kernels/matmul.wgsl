// Tiled matrix multiplication shader
// Computes C = A × B where A is MxK, B is KxN, C is MxN
// Uses shared memory tiling for optimal cache performance

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec3<u32>; // m, n, k

// Shared memory tiles (16×16 = 256 elements each)
var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

const TILE_SIZE: u32 = 16u;

@compute @workgroup_size(16, 16, 1)
fn matmul_tiled(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let m = dims.x;
    let n = dims.y;
    let k = dims.z;

    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;

    // Check bounds but don't exit early (to maintain uniform control flow)
    let valid_thread = row < m && col < n;

    var sum = 0.0;

    // Number of tiles needed to cover K dimension
    let num_tiles = (k + TILE_SIZE - 1u) / TILE_SIZE;

    // Process tiles
    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let k_offset = t * TILE_SIZE;

        // Load tile of A into shared memory
        let a_col = k_offset + local_col;
        if (row < m && a_col < k) {
            tile_a[local_row * TILE_SIZE + local_col] = a[row * k + a_col];
        } else {
            tile_a[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // Load tile of B into shared memory
        let b_row = k_offset + local_row;
        if (b_row < k && col < n) {
            tile_b[local_row * TILE_SIZE + local_col] = b[b_row * n + col];
        } else {
            tile_b[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // Synchronize workgroup - all threads must reach here
        workgroupBarrier();

        // Compute partial dot product using shared memory
        if (valid_thread) {
            for (var i = 0u; i < TILE_SIZE; i = i + 1u) {
                sum = sum + tile_a[local_row * TILE_SIZE + i] *
                           tile_b[i * TILE_SIZE + local_col];
            }
        }

        // Synchronize before loading next tile - all threads must reach here
        workgroupBarrier();
    }

    // Write result only if thread is valid
    if (valid_thread) {
        c[row * n + col] = sum;
    }
}
