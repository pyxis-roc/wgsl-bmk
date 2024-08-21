@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(${WORKGROUP_SIZE}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>)
{
    let index = global_id.x;
    
    if (global_id.x == 0) {
        result[0] = a[258];
    } else {
        result[index] = a[index] + b[index];
    }
    
}