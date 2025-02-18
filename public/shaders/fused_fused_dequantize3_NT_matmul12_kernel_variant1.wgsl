@group(0) @binding(0) var<storage, read_write> NT_matmul : array<f32>;
@group(0) @binding(1) var<storage, read> lv1995 : array<u32>;
@group(0) @binding(2) var<storage, read> lv1996 : array<f32>;
@group(0) @binding(3) var<storage, read> rms_norm46 : array<f32>;

struct PODArgs {
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

var<workgroup> red_buf0 : array<f32, 64>;
@compute @workgroup_size(64, 1, 1)
fn fused_fused_dequantize3_NT_matmul12_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  // check all indices
  var bounds_check_pass: bool = true;
  if (((i32(threadIdx.x) * 8i) + 1543i) < 0 || u32((i32(threadIdx.x) * 8i) + 1543i) >= arrayLength(&rms_norm46)) {
    bounds_check_pass = false;
  }
  if ((v__1 * 256i) + i32(threadIdx.x) < 0 || u32((v__1 * 256i) + i32(threadIdx.x)) >= arrayLength(&lv1995)) {
    bounds_check_pass = false;
  }
  if (((v__1 * 64i) + (i32(threadIdx.x)>>2u)) + 48i < 0 || u32(((v__1 * 64i) + (i32(threadIdx.x)>>2u)) + 48i) >= arrayLength(&lv1996)) {
    bounds_check_pass = false;
  }
  if (v__1 < 0 || u32(v__1) >= arrayLength(&NT_matmul)) {
    bounds_check_pass = false;
  }
  // can't use arrayLength for red_buf0 as it has a constant size.
  if ((i32(threadIdx.x) + 32i) < 0 || u32(i32(threadIdx.x) + 32i) >= 64) {
    bounds_check_pass = false;
  }
  var NT_matmul_rf_local : array<f32, 1>;
  var lv1995_local : array<u32, 1>;
  var NT_matmul_rf_local_1 : array<f32, 1>;
  NT_matmul_rf_local[0i] = 0.000000e+00f;
  if (bounds_check_pass) {
  lv1995_local[0i] = lv1995[((v__1 * 256i) + i32(threadIdx.x))];
  NT_matmul_rf_local[0i] = fma(rms_norm46[(i32(threadIdx.x) * 8i)], ((f32(((lv1995_local[0i]>>0u) & 15u)) - 7.000000e+00f) * lv1996[((v__1 * 64i) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 1i)], ((f32(((lv1995_local[0i]>>4u) & 15u)) - 7.000000e+00f) * lv1996[((v__1 * 64i) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 2i)], ((f32(((lv1995_local[0i]>>8u) & 15u)) - 7.000000e+00f) * lv1996[((v__1 * 64i) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 3i)], ((f32(((lv1995_local[0i]>>12u) & 15u)) - 7.000000e+00f) * lv1996[((v__1 * 64i) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 4i)], ((f32(((lv1995_local[0i]>>16u) & 15u)) - 7.000000e+00f) * lv1996[((v__1 * 64i) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 5i)], ((f32(((lv1995_local[0i]>>20u) & 15u)) - 7.000000e+00f) * lv1996[((v__1 * 64i) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 6i)], ((f32(((lv1995_local[0i]>>24u) & 15u)) - 7.000000e+00f) * lv1996[((v__1 * 64i) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 7i)], ((f32(((lv1995_local[0i]>>28u) & 15u)) - 7.000000e+00f) * lv1996[((v__1 * 64i) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
  lv1995_local[0i] = lv1995[(((v__1 * 256i) + i32(threadIdx.x)) + 64i)];
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 512i)], ((f32(((lv1995_local[0i]>>0u) & 15u)) - 7.000000e+00f) * lv1996[(((v__1 * 64i) + (i32(threadIdx.x)>>2u)) + 16i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 513i)], ((f32(((lv1995_local[0i]>>4u) & 15u)) - 7.000000e+00f) * lv1996[(((v__1 * 64i) + (i32(threadIdx.x)>>2u)) + 16i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 514i)], ((f32(((lv1995_local[0i]>>8u) & 15u)) - 7.000000e+00f) * lv1996[(((v__1 * 64i) + (i32(threadIdx.x)>>2u)) + 16i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 515i)], ((f32(((lv1995_local[0i]>>12u) & 15u)) - 7.000000e+00f) * lv1996[(((v__1 * 64i) + (i32(threadIdx.x)>>2u)) + 16i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 516i)], ((f32(((lv1995_local[0i]>>16u) & 15u)) - 7.000000e+00f) * lv1996[(((v__1 * 64i) + (i32(threadIdx.x)>>2u)) + 16i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 517i)], ((f32(((lv1995_local[0i]>>20u) & 15u)) - 7.000000e+00f) * lv1996[(((v__1 * 64i) + (i32(threadIdx.x)>>2u)) + 16i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 518i)], ((f32(((lv1995_local[0i]>>24u) & 15u)) - 7.000000e+00f) * lv1996[(((v__1 * 64i) + (i32(threadIdx.x)>>2u)) + 16i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 519i)], ((f32(((lv1995_local[0i]>>28u) & 15u)) - 7.000000e+00f) * lv1996[(((v__1 * 64i) + (i32(threadIdx.x)>>2u)) + 16i)]), NT_matmul_rf_local[0i]);
  lv1995_local[0i] = lv1995[(((v__1 * 256i) + i32(threadIdx.x)) + 128i)];
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 1024i)], ((f32(((lv1995_local[0i]>>0u) & 15u)) - 7.000000e+00f) * lv1996[(((v__1 * 64i) + (i32(threadIdx.x)>>2u)) + 32i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 1025i)], ((f32(((lv1995_local[0i]>>4u) & 15u)) - 7.000000e+00f) * lv1996[(((v__1 * 64i) + (i32(threadIdx.x)>>2u)) + 32i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 1026i)], ((f32(((lv1995_local[0i]>>8u) & 15u)) - 7.000000e+00f) * lv1996[(((v__1 * 64i) + (i32(threadIdx.x)>>2u)) + 32i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 1027i)], ((f32(((lv1995_local[0i]>>12u) & 15u)) - 7.000000e+00f) * lv1996[(((v__1 * 64i) + (i32(threadIdx.x)>>2u)) + 32i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 1028i)], ((f32(((lv1995_local[0i]>>16u) & 15u)) - 7.000000e+00f) * lv1996[(((v__1 * 64i) + (i32(threadIdx.x)>>2u)) + 32i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 1029i)], ((f32(((lv1995_local[0i]>>20u) & 15u)) - 7.000000e+00f) * lv1996[(((v__1 * 64i) + (i32(threadIdx.x)>>2u)) + 32i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 1030i)], ((f32(((lv1995_local[0i]>>24u) & 15u)) - 7.000000e+00f) * lv1996[(((v__1 * 64i) + (i32(threadIdx.x)>>2u)) + 32i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 1031i)], ((f32(((lv1995_local[0i]>>28u) & 15u)) - 7.000000e+00f) * lv1996[(((v__1 * 64i) + (i32(threadIdx.x)>>2u)) + 32i)]), NT_matmul_rf_local[0i]);
  lv1995_local[0i] = lv1995[(((v__1 * 256i) + i32(threadIdx.x)) + 192i)];
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 1536i)], ((f32(((lv1995_local[0i]>>0u) & 15u)) - 7.000000e+00f) * lv1996[(((v__1 * 64i) + (i32(threadIdx.x)>>2u)) + 48i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 1537i)], ((f32(((lv1995_local[0i]>>4u) & 15u)) - 7.000000e+00f) * lv1996[(((v__1 * 64i) + (i32(threadIdx.x)>>2u)) + 48i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 1538i)], ((f32(((lv1995_local[0i]>>8u) & 15u)) - 7.000000e+00f) * lv1996[(((v__1 * 64i) + (i32(threadIdx.x)>>2u)) + 48i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 1539i)], ((f32(((lv1995_local[0i]>>12u) & 15u)) - 7.000000e+00f) * lv1996[(((v__1 * 64i) + (i32(threadIdx.x)>>2u)) + 48i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 1540i)], ((f32(((lv1995_local[0i]>>16u) & 15u)) - 7.000000e+00f) * lv1996[(((v__1 * 64i) + (i32(threadIdx.x)>>2u)) + 48i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 1541i)], ((f32(((lv1995_local[0i]>>20u) & 15u)) - 7.000000e+00f) * lv1996[(((v__1 * 64i) + (i32(threadIdx.x)>>2u)) + 48i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 1542i)], ((f32(((lv1995_local[0i]>>24u) & 15u)) - 7.000000e+00f) * lv1996[(((v__1 * 64i) + (i32(threadIdx.x)>>2u)) + 48i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm46[((i32(threadIdx.x) * 8i) + 1543i)], ((f32(((lv1995_local[0i]>>28u) & 15u)) - 7.000000e+00f) * lv1996[(((v__1 * 64i) + (i32(threadIdx.x)>>2u)) + 48i)]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local_1[0i] = 0.000000e+00f;
  NT_matmul_rf_local_1[0i] = (NT_matmul_rf_local_1[0i] + NT_matmul_rf_local[0i]);
  }
  workgroupBarrier();
  red_buf0[i32(threadIdx.x)] = NT_matmul_rf_local_1[0i];
  workgroupBarrier();
  if (i32(threadIdx.x) < 32i && bounds_check_pass) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 32i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 16i && bounds_check_pass) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 16i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 8i && bounds_check_pass) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 8i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 4i && bounds_check_pass) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 4i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 2i && bounds_check_pass) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 2i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 1i && bounds_check_pass) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 1i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) == 0i && bounds_check_pass) {
    NT_matmul[v__1] = red_buf0[0i];
  }
}