import { TimingHelper } from "./types.mjs";
import { adapter, default_device as device } from "./gpu_setup.js";

const WORKGROUP_SIZE = 64;

// Check if we can get timestamp support here.
const timeSupport = adapter.features.has("timestamp-query");
if (!timeSupport) {
  throw new Error("Timestamp query not supported");
}

const resolveBuffer = device.createBuffer({
  size: 8,
  usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
});

// So we have timestamp query support

// Sample data

const shaderModule = device.createShaderModule({
  code: `@group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<storage, read> b: array<f32>;
    @group(0) @binding(2) var<storage, read> c: array<f32>;
    @group(0) @binding(3) var<storage, read_write> result: array<f32>;
    @group(0) @binding(4) var<uniform> sz: u32;

    const zeroVec: vec3<u32> = vec3<u32>(0, 0, 0);

    @compute @workgroup_size(${WORKGROUP_SIZE})
    fn main(@builtin(local_invocation_index) local_id: u32, @builtin(workgroup_id) blockIdx: vec3<u32>)
    {
      if (all(blockIdx != zeroVec)) {return;}

      for(var i = local_id; i < sz; i = i + ${WORKGROUP_SIZE}) {
          result[i] = a[i] * b[i] + c[i];
        }
    }`,
});

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
      buffer: { type: "read-only-storage" },
    },
    {
      binding: 3,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    },
    {
      binding: 4,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "uniform" },
    },
  ],
});

const computePipeline = device.createComputePipeline({
  layout: device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  }),
  compute: {
    module: shaderModule,
    entryPoint: "main",
  },
});

const timingHelper = new TimingHelper(device);

function makeNewBuffer(
  device: GPUDevice,
  sz: number,
  usage: GPUBufferUsageFlags,
  mappedAtCreation: boolean = false
) {
  return device.createBuffer({
    size: sz * Float32Array.BYTES_PER_ELEMENT,
    usage: usage,
    mappedAtCreation: mappedAtCreation,
  });
}

function makeAndInitBuffer(device: GPUDevice, sz: number, init_val?: number) {
  const buffer = makeNewBuffer(device, sz, GPUBufferUsage.STORAGE, true);
  const data = new Float32Array(buffer.getMappedRange());
  for (let i = 0; i < sz; i++) {
    data[i] = init_val ? init_val : Math.random();
  }
  buffer.unmap();
  return buffer;
}

function makeAndInitIntValue(device: GPUDevice, val: number) {
  const buffer = makeNewBuffer(device, 1, GPUBufferUsage.UNIFORM, true);
  const data = new Uint32Array(buffer.getMappedRange());
  device.queue.writeBuffer(buffer, 0, new Uint32Array([val]));
  buffer.unmap();
  return buffer;
}

export async function executeKernel(
  device: GPUDevice,
  computePipeline: GPUComputePipeline,
  sz: number = 8192
) {
  const a_input = makeAndInitBuffer(device, sz);
  const b_input = makeAndInitBuffer(device, sz);
  const c_input = makeAndInitBuffer(device, sz);
  const output_device = makeNewBuffer(
    device,
    sz,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  );
  const staging_buffer = makeNewBuffer(
    device,
    sz,
    GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  );
  const sz_buffer = makeAndInitIntValue(device, sz);

  const bindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: a_input } },
      { binding: 1, resource: { buffer: b_input } },
      { binding: 2, resource: { buffer: c_input } },
      { binding: 3, resource: { buffer: output_device } },
      { binding: 4, resource: { buffer: sz_buffer } },
    ],
  });

  const commandEncoder = device.createCommandEncoder();

  const timingEncoder = timingHelper.beginComputePass(commandEncoder, {
    label: "Hello-WGPU",
  });
  timingEncoder.setPipeline(computePipeline);
  timingEncoder.setBindGroup(0, bindGroup);
  timingEncoder.dispatchWorkgroups(32);
  timingEncoder.end();

  // copy the data to the output...
  commandEncoder.copyBufferToBuffer(
    output_device,
    0,
    staging_buffer,
    0,
    staging_buffer.size
  );

  device.queue.submit([commandEncoder.finish()]);

  await staging_buffer.mapAsync(GPUMapMode.READ, 0, staging_buffer.size);
  const duration = await timingHelper.getResult();
  const copyArrayBuffer = staging_buffer.getMappedRange(0, staging_buffer.size);
  const outputData = copyArrayBuffer.slice(0);
  staging_buffer.unmap();

  a_input.destroy();
  b_input.destroy();
  c_input.destroy();
  output_device.destroy();
  staging_buffer.destroy();
  sz_buffer.destroy();

  return { outputData, duration };
}

async function run() {
  document.getElementById("outputTable").innerText = "RUNNING...";
  let iterations = parseInt(
    (document.getElementById("iterations") as HTMLInputElement).value,
    10
  );
  var output: ArrayBuffer;
  var duration;
  let durations = [];
  for (let i = 0; i < iterations; i++) {
    let { outputData, duration } = await executeKernel(device, computePipeline);
    durations.push(duration);
    output = outputData;
  }
  const total = durations.reduce(
    (accumulator, currentValue) => accumulator + currentValue,
    0
  );
  console.log(durations);
  console.log(output);
  createTable(output);
  return { averageTime: total / iterations, totalTime: total };
}

export function createTable(output) {
  const table = document.createElement("table");
  const headerRow = document.createElement("tr");
  ["Index", "Output"].forEach((text) => {
    const th = document.createElement("th");
    th.textContent = text;
    headerRow.appendChild(th);
  });
  table.appendChild(headerRow);

  for (let i = 0; i < output.length; i++) {
    const row = document.createElement("tr");
    [i, output[i]].forEach((value) => {
      const td = document.createElement("td");
      td.textContent = value;
      row.appendChild(td);
    });
    table.appendChild(row);
  }

  const outputTableDiv = document.getElementById("outputTable");
  outputTableDiv.innerHTML = ""; // Clear previous table if any
  outputTableDiv.appendChild(table);
}

document.getElementById("runButton").addEventListener("click", async () => {
  const { averageTime, totalTime } = await run();
  console.log(`Average time: ${averageTime} ms`);
  console.log(`Total time: ${totalTime} ms`);
});

export {};
