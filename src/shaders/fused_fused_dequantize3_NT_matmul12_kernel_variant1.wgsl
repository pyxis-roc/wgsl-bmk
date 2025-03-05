@group(0) @binding(0) var<storage, read_write> NT_matmul : array<f32>;
@group(0) @binding(1) var<storage, read> lv1995 : array<u32>;
@group(0) @binding(2) var<storage, read> lv1996 : array<f32>;
@group(0) @binding(3) var<storage, read> rms_norm46 : array<f32>;

struct PODArgs {
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

const WORKGROUP_SIZE_X = 64;
const WORKGROUP_SIZE_Y = 1;
const WORKGROUP_SIZE_Z = 1;

var<workgroup> red_buf0 : array<f32, 64>;
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn dequantize_with_checks(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  // First off, here I'm placing the checks after the if {} return statement.
  // This is an optimization. In reality, each of the checks emitted by the solver below
  // would first have needed to check the negation of this condition as well,
  // as the checks would actually be placed above it.
  // For simplicity, though, I will be omitting them and placing it below.
  // Checks for rms_norms.
  // threadIdx.x is replaced by 63, as the @workgroup_size given says threadIdx.x is in [0 : 63].
  // Solver would eliminate the min check as it knows it is always true from the constants involved.
  if (u32(i32(63) * 8i) >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 1 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 2 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 3 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 4 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 5 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 6 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 7 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 512 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 513 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 514 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 515 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 516 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 517 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 518 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 519 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 1024 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 1025 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 1026 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 1027 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 1028 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 1029 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 1030 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 1031 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 1536 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 1537 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 1538 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 1539 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 1540 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 1541 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 1542 >= arrayLength(&rms_norm46)) {
    return;
  }
  if (u32(i32(63) * 8i) + 1543 >= arrayLength(&rms_norm46)) {
    return;
  }

  // Checks for lv1995. Expanding v__1 from blockIdx.z * gridDim.x + blockIdx.x.
  // Min value of blockDim.x and blockDim.z is 0
  // Max value of blockDim.z and blockDim.x is gridDim.x - 1 and gridDim.z - 1 respectively
  // This is how the solver would expand out the formula.
  // But the solver would eliminate the < 0 check as it solves constants and knows it is always true.
  if ((min((gridDim.z - 1) * gridDim.x + (gridDim.x - 1), podArgs.packGridDimX - 1) * 256 + 63 >= arrayLength(&lv1995))) {
    return;
  }
  if ((min((gridDim.z - 1) * gridDim.x + (gridDim.x - 1), podArgs.packGridDimX) * 256 + 63) + 64 >= arrayLength(&lv1995)) {
    return;
  }
  if ((min((gridDim.z - 1) * gridDim.x + (gridDim.x - 1), podArgs.packGridDimX) * 256 + 63) + 128 >= arrayLength(&lv1995)) {
    return;
  }
  if ((min((gridDim.z - 1) * gridDim.x + (gridDim.x - 1), podArgs.packGridDimX) * 256 + 63) + 192 >= arrayLength(&lv1995)) {
    return;
  }

  // Checks for lv1996. Expanding v__1 as before.
  if ((min((gridDim.z - 1) * gridDim.x + (gridDim.x - 1), podArgs.packGridDimX - 1) * 64 + ((63 >> 2u)) >= arrayLength(&lv1996))) {
    return;
  }
  if ((min((gridDim.z - 1) * gridDim.x + (gridDim.x - 1), podArgs.packGridDimX - 1) * 64 + ((63 >> 2u)) + 16 >= arrayLength(&lv1996))) {
    return;
  }
  if ((min((gridDim.z - 1) * gridDim.x + (gridDim.x - 1), podArgs.packGridDimX - 1) * 64 + ((63 >> 2u)) + 32 >= arrayLength(&lv1996))) {
    return;
  }
  if ((min((gridDim.z - 1) * gridDim.x + (gridDim.x - 1), podArgs.packGridDimX - 1) * 64 + ((63 >> 2u)) + 48 >= arrayLength(&lv1996))) {
    return;
  }
  
  // If we're basing this on what the solver would do, then the writes down below to red_buf0 would actually have been eliminated, as we know the size to redbuf0.
  // So we actually can remove ALL of the checks against redbuf0. This is quite common. All arrays that reside in workgroup memory have their sizes known,
  // and so the solver would be able to eliminate *all* bounds checks that are indexing into them based on some value of threadIdx.
  // See below... This is what the constraint solver would essentially be receving at first.
  // All of the checks against NT_matmul_rf_local are eliminated as the index is always 0, and we have a global assumption that the minimum of any array length is 1.
  // (wgsl does not permit 0 length arrays!)
  // The accesses to redbuf0 that occur inside conditions are similarly eliminated. The if check would just further constrain the threadIdx.x.
  // For instance, the if (i32(threadIdx.x) < 32i ) would cause the bounds check inside the formula to be generated as min(31, 31) < 0 and max(31, 31) >= 64.
  // min(31, 31) 
  // regardless of whether the condition could ever be true, the check inside could always be eliminated
  /*
  if ((min(63i, 32i-1i) < 0 || max(63i, 32i-1i)) < 0 && (min(63i, 32i-1i) < 0 || max(63i, 32i-1i) >= 64) >= 64) {
    return;
  }
  */

  // The final condition for NT_matmul. This time, v__1 is computed slightly differently as threadIdx.x is now known to be concretized to 0
  if ((min((gridDim.z - 1) * gridDim.x + (gridDim.x - 1), podArgs.packGridDimX - 1) * 64 + 0 >= arrayLength(&NT_matmul))) {
    return;
  }

  var NT_matmul_rf_local : array<f32, 1>;
  var lv1995_local : array<u32, 1>;
  var NT_matmul_rf_local_1 : array<f32, 1>;
  NT_matmul_rf_local[0i] = 0.000000e+00f;
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
  workgroupBarrier();
  red_buf0[i32(threadIdx.x)] = NT_matmul_rf_local_1[0i];
  workgroupBarrier();
  if (i32(threadIdx.x) < 32i ) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 32i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 16i ) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 16i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 8i ) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 8i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 4i) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 4i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 2i) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 2i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 1i) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 1i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) == 0i) {
    NT_matmul[v__1] = red_buf0[0i];
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    NT_matmul[0] = f32(15);
  }
}