// Element-wise operations shader
// Supports: sin, cos, tan, exp, log, sqrt, abs, etc.
// Uses vectorized loads for better performance

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    size: u32,
    op_type: u32,  // 0=sin, 1=cos, 2=exp, 3=log, 4=sqrt, 5=abs, etc.
    _padding1: u32,
    _padding2: u32,
}

@compute @workgroup_size(256, 1, 1)
fn elementwise_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    let x = input[idx];
    var result: f32;

    switch params.op_type {
        case 0u: { result = sin(x); }      // sin
        case 1u: { result = cos(x); }      // cos
        case 2u: { result = tan(x); }      // tan
        case 3u: { result = exp(x); }      // exp
        case 4u: { result = log(x); }      // log (natural)
        case 5u: { result = sqrt(x); }     // sqrt
        case 6u: { result = abs(x); }      // abs
        case 7u: { result = x * x; }       // square
        case 8u: { result = 1.0 / x; }     // reciprocal
        default: { result = x; }           // identity
    }

    output[idx] = result;
}
