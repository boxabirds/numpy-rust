// Parallel reduction shader
// Efficiently computes sum/max/min using shared memory

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    size: u32,
    op_type: u32,  // 0=sum, 1=max, 2=min, 3=product
    _padding1: u32,
    _padding2: u32,
}

// Shared memory for workgroup reduction
var<workgroup> shared_data: array<f32, 256>;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256, 1, 1)
fn reduce(@builtin(global_invocation_id) global_id: vec3<u32>,
          @builtin(local_invocation_id) local_id: vec3<u32>,
          @builtin(workgroup_id) group_id: vec3<u32>) {
    
    let tid = local_id.x;
    let gid = global_id.x;
    
    // Load data into shared memory
    var value: f32;
    if (gid < params.size) {
        value = input[gid];
    } else {
        // Initialize with identity element
        switch params.op_type {
            case 0u: { value = 0.0; }  // sum
            case 1u: { value = -3.40282e38; }  // max (f32::MIN)
            case 2u: { value = 3.40282e38; }   // min (f32::MAX)
            case 3u: { value = 1.0; }  // product
            default: { value = 0.0; }
        }
    }
    shared_data[tid] = value;
    
    workgroupBarrier();
    
    // Parallel reduction in shared memory
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (tid < stride && (gid + stride) < params.size) {
            let a = shared_data[tid];
            let b = shared_data[tid + stride];
            
            switch params.op_type {
                case 0u: { shared_data[tid] = a + b; }  // sum
                case 1u: { shared_data[tid] = max(a, b); }  // max
                case 2u: { shared_data[tid] = min(a, b); }  // min
                case 3u: { shared_data[tid] = a * b; }  // product
                default: { shared_data[tid] = a + b; }
            }
        }
        workgroupBarrier();
    }
    
    // Write workgroup result
    if (tid == 0u) {
        output[group_id.x] = shared_data[0];
    }
}
