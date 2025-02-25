import { TimingHelper } from "../types.mjs";
import { adapter, safeRequestDevice, adapter_limits } from "../gpu_setup.js";

const NUM_ELEMENTS = 1024;
const LV_1995_SIZE = NUM_ELEMENTS;
const LV_1996_SIZE = NUM_ELEMENTS / 4;

/** How many elements are being processed per block */
const packGridDimX = NUM_ELEMENTS / 256;
/** How many blocks should there be */
const NUM_BLOCKS = packGridDimX;

const WORKGROUP_SIZE_X = 64; // Workgroup size has been hardcoded to 64 in the shader
const RMS_NORM_SIZE = 2048; // Max index is threadIdx * 8 + 1563. 63 * 8 + 1563 = 2047

const NT_MATMUL_SIZE = packGridDimX;
/** How many iterations to run by default */
const DEFAULT_ITERATION = 100;
/// The maximum workgroup size we need is..
/// WORKGROUP_SIZE_X * WORK_PER_THREAD_X * 4
/** Number of decimal digits to display for the timestamp */
const TIMESTAMP_PRECISION = 3;
/** Number of decimal digits to display for the result */
const RESULT_PRECISION = 3;
/// Set a max iterations to stop browsers from crashing
const MAX_ITERATIONS = 10000;
const MIN_ITERATIONS = 1;
var computePipeline: GPUComputePipeline;
var bindGroup: GPUBindGroup;
var gpuReadBuffer: GPUBuffer;
var lv1995_Host, lv1996_Host, rms_norm46_Host;
var iteration = DEFAULT_ITERATION;
var commandQueue = [];
var timingEncoder: TimingHelper;
const device = await safeRequestDevice(adapter, ["timestamp-query"], {
    maxComputeWorkgroupStorageSize:
        adapter_limits.maxComputeWorkgroupStorageSize,
});

timingEncoder = new TimingHelper(device);
// Check if we can get timestamp support here.
function recordCommands() {
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = timingEncoder.beginComputePass(commandEncoder);
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(NUM_BLOCKS);
    passEncoder.end();
    commandQueue.push(commandEncoder);
}

function submitQueue() {
    // We need the total duration of all kernels in the queue...
    // If we submit each one, then
    device.queue.submit(commandQueue.map((enc) => enc.finish()));
    commandQueue = [];
}

document
    .getElementById("it")
    .setAttribute("value", DEFAULT_ITERATION.toString());
document.getElementById("it").setAttribute("max", MAX_ITERATIONS.toString());
document.getElementById("it").setAttribute("min", MIN_ITERATIONS.toString());
(document.getElementById("it") as HTMLInputElement).value =
    DEFAULT_ITERATION.toString();
(async () => {
    if (!navigator.gpu) {
        console.log(
            "WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag."
        );
        return;
    }
    timingEncoder = new TimingHelper(device);
    // Uniform Buffer
    const podArgs_Host = new Uint32Array([packGridDimX]);
    const podArgs_Device = device.createBuffer({
        mappedAtCreation: true,
        size: podArgs_Host.byteLength,
        usage: GPUBufferUsage.UNIFORM,
    });
    new Uint32Array(podArgs_Device.getMappedRange()).set(podArgs_Host);
    podArgs_Device.unmap();

    // Second Matrix
    lv1995_Host = new Uint32Array(LV_1995_SIZE);
    for (var i = 0; i < LV_1995_SIZE; i++) {
        lv1995_Host[i] = Math.floor(Math.random() * 4294967295);
    }
    const lv1995_Device = device.createBuffer({
        mappedAtCreation: true,
        size: lv1995_Host.byteLength,
        usage: GPUBufferUsage.STORAGE,
    });
    new Uint32Array(lv1995_Device.getMappedRange()).set(lv1995_Host);
    lv1995_Device.unmap();

    lv1996_Host = new Uint32Array(LV_1996_SIZE);
    for (var i = 0; i < LV_1996_SIZE; i++) {
        lv1996_Host[i] = Math.floor(Math.random() * 4294967295);
    }
    const lv1996_Device = device.createBuffer({
        mappedAtCreation: true,
        size: lv1995_Host.byteLength,
        usage: GPUBufferUsage.STORAGE,
    });
    new Uint32Array(lv1996_Device.getMappedRange()).set(lv1995_Host);
    lv1996_Device.unmap();

    rms_norm46_Host = new Float32Array(RMS_NORM_SIZE);
    for (var i = 0; i < RMS_NORM_SIZE; i++) {
        rms_norm46_Host[i] = Math.random();
    }
    const rms_norm46_Device = device.createBuffer({
        mappedAtCreation: true,
        size: rms_norm46_Host.byteLength,
        usage: GPUBufferUsage.STORAGE,
    });
    new Float32Array(rms_norm46_Device.getMappedRange()).set(rms_norm46_Host);
    rms_norm46_Device.unmap();

    // Result Matrix
    const NT_matmul_BufferSize =
        Float32Array.BYTES_PER_ELEMENT * NT_MATMUL_SIZE;
    const NT_matmul_Device = device.createBuffer({
        size: NT_matmul_BufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    // Bind group layout and bind group
    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "uniform" },
            },
        ],
    });
    bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: NT_matmul_Device } },
            { binding: 1, resource: { buffer: lv1995_Device } },
            { binding: 2, resource: { buffer: lv1996_Device } },
            { binding: 3, resource: { buffer: rms_norm46_Device } },
            { binding: 4, resource: { buffer: podArgs_Device } },
        ],
    });

    const shaderCode = `@group(0) @binding(0) var<storage, read_write> NT_matmul : array<f32>;
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
  if (i32(threadIdx.x) < 32i) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 32i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 16i) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 16i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 8i) {
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
}
`;
    // Pipeline setup
    computePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout],
        }),
        compute: {
            module: device.createShaderModule({
                code: shaderCode,
            }),
            entryPoint: "fused_fused_dequantize3_NT_matmul12_kernel",
        },
    });
    recordCommands();
    // Get a GPU buffer for reading in an unmapped state.
    gpuReadBuffer = device.createBuffer({
        size: NT_matmul_BufferSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    // Encode commands for copying buffer to buffer.
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
        NT_matmul_Device /* source buffer */,
        0 /* source offset */,
        gpuReadBuffer /* destination buffer */,
        0 /* destination offset */,
        NT_matmul_BufferSize /* size */
    );
    commandQueue.push(commandEncoder);
    // Submit GPU commands.
    submitQueue();
    // Read buffer.
    await gpuReadBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = new Float32Array(gpuReadBuffer.getMappedRange());
    // On warmup finished, remove the warmup div.
    document.getElementById("warmup").remove();
    /*
    for (var i = 0; i < dimAOuter; i++)
    for (var j = 0; j< dimBOuter; j++)
    {
      let test = 0;
      for (var k =0; k < dimInner; k++)
      {
        test += firstMatrix[i * dimInner + k] * secondMatrix[k * dimBOuter + j];
      }
      console.log(`result[${i}, ${j}] = ${arrayBuffer[i * dimBOuter + j]}, expectedResult = ${test}`);
    }
    */
    gpuReadBuffer.unmap();
    initializeResultTable();
})();
/// Add event listenner that validates the input and updates iteration
export function handleChange(e: { target: { value: string } }) {
    // Only allow whole numbers
    const match = e.target.value.match(/[0]*(\d+)([^\d]|$)/);
    e.target.value = match ? match[1] : iteration.toString();
    iteration = Math.min(parseInt(e.target.value), MAX_ITERATIONS);
}
function initializeResultTable() {
    // Don't initialize if it already exists.
    if (document.getElementById("dequantize-result-table") !== null) return;
    // Make the headers
    const resultTable = document.createElement("table");
    resultTable.id = "dequantize-result-table";
    resultTable.classList.add("result-table", "dequantize");
    resultTable.setAttribute("hidden", "");
    var header = document.createElement("thead");
    var headerRow = document.createElement("tr");
    [
        { text: "Iterations", tooltip: "Number of iterations" },
        {
            text: "Shader time (ms)",
            tooltip: "Amount of time passed measured using timestamp queries",
        },
        { text: "Js time (ms)", tooltip: "" },
        {
            text: "Result",
            tooltip:
                "The computed value of some random element in the result matrix",
        },
    ].forEach((entry) => {
        var th = document.createElement("th");
        th.textContent = entry.text;
        th.title = entry.tooltip;
        headerRow.appendChild(th);
    });
    header.appendChild(headerRow);
    resultTable.appendChild(header);
    var tbody = document.createElement("tbody");
    tbody.id = "dequantize-result-table-body";
    resultTable.appendChild(tbody);
    document.getElementById("dequantize-result-div").appendChild(resultTable);
}
function addResultRow(
    timestamp_time: number,
    js_time: number,
    result: number,
    iterations: number
) {
    document
        .getElementById("dequantize-result-table")
        .removeAttribute("hidden");
    var tbody = document.getElementById("dequantize-result-table-body");
    var row = document.createElement("tr");
    [
        iterations,
        timestamp_time.toFixed(TIMESTAMP_PRECISION),
        js_time.toFixed(TIMESTAMP_PRECISION),
        result.toFixed(RESULT_PRECISION),
    ].forEach((value) => {
        var td = document.createElement("td");
        td.innerText = `${value}`;
        row.appendChild(td);
    });
    tbody.appendChild(row);
}
export async function run() {
    //   const computeFence = device.queue.createFence();
    // iteration = parseInt((document.getElementById("it") as HTMLInputElement).value , 10);
    var start = performance.now();
    for (var i = 0; i < iteration; i++) {
        recordCommands();
    }
    device.queue.submit(commandQueue.map((enc) => enc.finish()));
    commandQueue.length = 0;
    const perf_now_ttl_time = await device.queue
        .onSubmittedWorkDone()
        .then(() => {
            return performance.now() - start;
        });
    const total_time = await timingEncoder.getResult();
    // Read buffer.
    await gpuReadBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = new Float32Array(gpuReadBuffer.getMappedRange());

    const meanTime = total_time / 1000000 / iteration;
    const meanTimePNow = perf_now_ttl_time / iteration;
    addResultRow(
        meanTime,
        meanTimePNow,
        arrayBuffer[0],
        iteration
    );
    gpuReadBuffer.unmap();
}
