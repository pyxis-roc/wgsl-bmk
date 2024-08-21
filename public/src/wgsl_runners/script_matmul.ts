import { NumericLiteral } from "../../../node_modules/typescript/lib/typescript";
import { TimingHelper } from "../types.mjs";
import { adapter, safeRequestDevice, adapter_limits } from "../gpu_setup.js";

// The maximum memory we need
const dimAOuter = 1024;
const dimInner = 1024;
const dimBOuter = 1024;
const WORKGROUP_SIZE_X = 16;
const WORKGROUP_SIZE_Y = 16;

/// Adjust the work per thread based on the adapter limits.
/// Adjusting this is OK since we dispatch enough workgroups to cover the entire matrix
/// based on these values.
const WORK_PER_THREAD_X =
  adapter_limits.maxComputeWorkgroupStorageSize < 32768 ? 2 : 4;
const WORK_PER_THREAD_Y = WORK_PER_THREAD_X;
const DEFAULT_ITERATION = 100;

/// The maximum workgroup size we need is..
/// WORKGROUP_SIZE_X * WORK_PER_THREAD_X * 4

/// Number of decimal digits to display for the timestamp
const TIMESTAMP_PRECISION = 3;
/// Number of decimal digits to display for the result
const RESULT_PRECISION = 3;

/// Set a max iterations to stop browsers from crashing
const MAX_ITERATIONS = 10000;
const MIN_ITERATIONS = 1;
var computePipeline: GPUComputePipeline;
var bindGroup: GPUBindGroup;
var gpuReadBuffer: GPUBuffer;
var firstMatrix: Float32Array, secondMatrix: Float32Array;

var iteration = DEFAULT_ITERATION;
var commandQueue: Array<GPUCommandEncoder> = [];
var timingEncoder: TimingHelper;

const device = await safeRequestDevice(adapter, ["timestamp-query"], {
  maxComputeWorkgroupStorageSize: adapter_limits.maxComputeWorkgroupStorageSize,
});

timingEncoder = new TimingHelper(device);

// Check if we can get timestamp support here.

function recordCommands() {
  const commandEncoder = device.createCommandEncoder();

  const passEncoder = timingEncoder.beginComputePass(commandEncoder);
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(
    dimBOuter / (WORKGROUP_SIZE_X * WORK_PER_THREAD_X) /* x */,
    dimAOuter / (WORKGROUP_SIZE_Y * WORK_PER_THREAD_Y) /* y */
  );
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
  const uniformData = new Int32Array([
    dimAOuter /* A rows */,
    dimInner /* A columns */,
    dimInner /* B rows */,
    dimBOuter /* B columns */,
  ]);

  const uniformBuffer = device.createBuffer({
    mappedAtCreation: true,
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM,
  });
  new Int32Array(uniformBuffer.getMappedRange()).set(uniformData);
  uniformBuffer.unmap();

  // First Matrix
  firstMatrix = new Float32Array(dimAOuter * dimInner);
  for (var i = 0; i < dimAOuter * dimInner; i++) {
    firstMatrix[i] = Math.random();
  }

  const gpuBufferFirstMatrix = device.createBuffer({
    mappedAtCreation: true,
    size: firstMatrix.byteLength,
    usage: GPUBufferUsage.STORAGE,
  });
  new Float32Array(gpuBufferFirstMatrix.getMappedRange()).set(firstMatrix);
  gpuBufferFirstMatrix.unmap();

  // Second Matrix
  secondMatrix = new Float32Array(dimInner * dimBOuter);
  for (var i = 0; i < dimInner * dimBOuter; i++) {
    secondMatrix[i] = Math.random();
  }

  const gpuBufferSecondMatrix = device.createBuffer({
    mappedAtCreation: true,
    size: secondMatrix.byteLength,
    usage: GPUBufferUsage.STORAGE,
  });
  new Float32Array(gpuBufferSecondMatrix.getMappedRange()).set(secondMatrix);
  gpuBufferSecondMatrix.unmap();

  // Result Matrix
  const resultMatrixBufferSize =
    Float32Array.BYTES_PER_ELEMENT * (uniformData[0] * uniformData[3]);
  const resultMatrixBuffer = device.createBuffer({
    size: resultMatrixBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // Bind group layout and bind group

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
    ],
  });

  bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: gpuBufferFirstMatrix,
        },
      },
      {
        binding: 1,
        resource: {
          buffer: gpuBufferSecondMatrix,
        },
      },
      {
        binding: 2,
        resource: {
          buffer: resultMatrixBuffer,
        },
      },
      {
        binding: 3,
        resource: {
          buffer: uniformBuffer,
        },
      },
    ],
  });

  // Compute shader code (manually translated from GLSL)
  const shaderCode = `const WORKGROUP_SIZE_X = ${WORKGROUP_SIZE_X};
const WORKGROUP_SIZE_Y = ${WORKGROUP_SIZE_Y};
const WORKGROUP_SIZE_Z = 1;
const WORK_PER_THREAD: vec2<i32> = vec2<i32>(${WORK_PER_THREAD_X}, ${WORK_PER_THREAD_Y});

const RowPerThread: i32 = WORK_PER_THREAD.x;
const ColPerThread: i32 = WORK_PER_THREAD.y;

const TileAOuter: i32 = WORKGROUP_SIZE_X * RowPerThread;
const TileBOuter: i32 = WORKGROUP_SIZE_Y * ColPerThread;
const TileInner: i32 = max(TileAOuter, TileBOuter);


@group(0) @binding(0) var<storage> firstMatrix: array<f32>;
@group(0) @binding(1) var<storage> secondMatrix: array<f32>;
@group(0) @binding(2) var<storage, read_write> resultMatrix: array<f32>;
@group(0) @binding(3) var<uniform> iShape: vec4<i32>;

var<workgroup> mm_Asub: array<array<f32, TileAOuter>, TileInner>;
var<workgroup> mm_Bsub: array<array<f32, TileInner>, TileBOuter>;
var<private> gl_LocalInvocationID_1: vec3<u32>;
var<private> gl_GlobalInvocationID_1: vec3<u32>;

fn mm_readA(row: i32, col: i32) -> f32 {
    return firstMatrix[row * iShape[1] + col];
}

fn mm_readB(row: i32, col: i32) -> f32 {
    return secondMatrix[row *  iShape[3] + col];
}

fn mm_write(row: i32, col: i32, value: f32) {
    resultMatrix[row * iShape[3] + col] = value;
}

fn mm_matMul(local_invocation_id: vec3<u32>, global_invocation_id: vec3<u32>) {
  var ACached: f32;
  var BCached: array<f32, ColPerThread>;
  let dimAOuter = iShape[0];
  let dimInner = iShape[1];
  let dimBOuter = iShape[3];
  let tileRow = i32(local_invocation_id.y) * RowPerThread;
  let tileCol = i32(local_invocation_id.x) * ColPerThread;
  let globalRow = i32(global_invocation_id.y) * RowPerThread;
  let globalCol = i32(global_invocation_id.x) * ColPerThread;
  let numTiles = (dimInner - 1) / TileInner + 1;
  var acc: array<array<f32, 64>, 64>;
  for(var innerRow = i32(0); innerRow < RowPerThread; innerRow++) {
    for(var innerCol = i32(0); innerCol < ColPerThread; innerCol++) {
      acc[innerRow][innerCol] = 0.0;
    }
  }
  let ColPerThreadA: i32 = (TileInner / WORKGROUP_SIZE_X);
  var tileColA = i32(local_invocation_id.x) * ColPerThreadA;
  let RowPerThreadB: i32 = TileInner / WORKGROUP_SIZE_Y;
  var tileRowB = i32(local_invocation_id.y) * RowPerThreadB;
  // loop over shared dimension
  for (var t = i32(0); t < numTiles; t++) {
    // Load one tile of A into workgroup memory
    for (var innerRow = i32(0); innerRow < RowPerThread; innerRow++) {
        for (var innerCol = i32(0); innerCol < ColPerThreadA; innerCol++) {
            let inputRow = tileRow + innerRow;
            let inputCol = tileColA + innerCol;
            mm_Asub[inputRow][inputCol] = mm_readA(globalRow + innerRow, t * TileInner + inputCol);
        }
    }
    // Load one tile of B into workgroup memory
    for (var innerRow = i32(0); innerRow < RowPerThreadB; innerRow++) {
        for (var innerCol = i32(0); innerCol < ColPerThread; innerCol++) {
            let inputRow = tileRowB + innerRow;
            let inputCol = tileCol + innerCol;
            mm_Bsub[inputRow][inputCol] = mm_readB(t * TileInner + inputRow, globalCol + innerCol);
        }
    }
    workgroupBarrier();
    // Compute acc values for a single thread.
    for (var k = i32(0); k < TileInner; k++) {
        for (var inner = i32(0); inner < ColPerThread; inner++) {
            BCached[inner] = mm_Bsub[k][tileCol + inner];
        }
        for (var innerRow = i32(0); innerRow < RowPerThread; innerRow++) {
            ACached = mm_Asub[tileRow + innerRow][k];
            for (var innerCol = i32(0); innerCol < ColPerThread; innerCol++) {
                acc[innerRow][innerCol] += ACached * BCached[innerCol];
            }
        }
    }
    workgroupBarrier();
    for (var innerRow = i32(0); innerRow < RowPerThread; innerRow++) {
        for (var innerCol = i32(0); innerCol < ColPerThread; innerCol++) {
            if ((globalCol + innerCol) < dimBOuter && (globalRow + innerRow) < dimAOuter) {
                mm_write(globalRow + innerRow, globalCol + innerCol, acc[innerRow][innerCol]);
            }
        }
    }
  }
}
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>) {
    mm_matMul(local_id, gid);
    return;
}`;

  // Pipeline setup

  computePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    }),
    compute: {
      module: device.createShaderModule({
        code: shaderCode,
      }),
      entryPoint: "main",
    },
  });

  recordCommands();

  // Get a GPU buffer for reading in an unmapped state.
  gpuReadBuffer = device.createBuffer({
    size: resultMatrixBufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  // Encode commands for copying buffer to buffer.
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(
    resultMatrixBuffer /* source buffer */,
    0 /* source offset */,
    gpuReadBuffer /* destination buffer */,
    0 /* destination offset */,
    resultMatrixBufferSize /* size */
  );
  commandQueue.push(commandEncoder);

  // Submit GPU commands.
  submitQueue();

  // Read buffer.
  await gpuReadBuffer.mapAsync(GPUMapMode.READ);
  const arrayBuffer = new Float32Array(gpuReadBuffer.getMappedRange());

  let acc = 0,
    m = Math.floor(dimAOuter * Math.random()),
    n = Math.floor(dimBOuter * Math.random());
  for (let k = 0; k < dimInner; k++)
    acc += firstMatrix[m * dimInner + k] * secondMatrix[k * dimBOuter + n];
  const result = arrayBuffer[m * dimBOuter + n];

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
export function handleChange(e) {
  // Only allow whole numbers
  const match = e.target.value.match(/[0]*(\d+)([^\d]|$)/);
  e.target.value = match ? match[1] : iteration.toString();

  iteration = Math.min(parseInt(e.target.value), MAX_ITERATIONS);
}

function initializeResultTable() {
  // Don't initialize if it already exists.
  if (document.getElementById("tfjs-result-table") !== null) return;

  // Make the headers

  const resultTable = document.createElement("table");
  resultTable.id = "tfjs-result-table";
  resultTable.classList.add("result-table");
  resultTable.classList.add("tfjs");
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
    { text: "GFLOPS", tooltip: "The calculated number of GFLOPS" },
    {
      text: "Expected",
      tooltip: "The expected value of a random element in the result matrix",
    },
    {
      text: "Result",
      tooltip:
        "The computed value of the same random element in the result matrix",
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
  tbody.id = "tfjs-result-table-body";
  resultTable.appendChild(tbody);

  document.getElementById("tfjs-result-div").appendChild(resultTable);
}

function addResultRow(
  timestamp_time: number,
  js_time: number,
  gflops: number,
  expected: number,
  result: number,
  iterations: number
) {
  document.getElementById("tfjs-result-table").removeAttribute("hidden");
  var tbody = document.getElementById("tfjs-result-table-body");

  var row = document.createElement("tr");
  [
    iterations,
    timestamp_time.toFixed(TIMESTAMP_PRECISION),
    js_time.toFixed(TIMESTAMP_PRECISION),
    gflops,
    expected.toFixed(RESULT_PRECISION),
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

  let acc = 0,
    m = Math.floor(dimAOuter * Math.random()),
    n = Math.floor(dimBOuter * Math.random());
  for (let k = 0; k < dimInner; k++)
    acc += firstMatrix[m * dimInner + k] * secondMatrix[k * dimBOuter + n];

  const meanTime = total_time / 1000000 / iteration;
  const meanTimePNow = perf_now_ttl_time / iteration;
  addResultRow(
    meanTime,
    meanTimePNow,
    Math.round((2 * dimAOuter * dimBOuter * dimInner) / meanTime / 10000) / 100,
    acc,
    arrayBuffer[m * dimBOuter + n],
    iteration
  );
  gpuReadBuffer.unmap();
}
