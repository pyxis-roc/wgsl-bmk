import { TimingHelper } from "scripts/classes/TimingHelper.mjs";
import { adapter, adapter_limits, safeRequestDevice } from "scripts/gpu_setup";

/** The GPU device */
const device = await safeRequestDevice(adapter, ["timestamp-query"], {
	maxComputeWorkgroupStorageSize: adapter_limits.maxComputeWorkgroupStorageSize,
});

/**
 * Initialize buffer data
 * @param sizeInBytes The size of the buffer, in bytes
 * @param arrayTy: The type of the array to initialize
 * @param initMethod: The method to use to initialize the buffer
 */
function initHostBuffer<
	T extends Float32ArrayConstructor | Uint32ArrayConstructor,
>(
	sizeInBytes: number,
	arrayConstructor: T,
	initMethod: InstanceType<T> | "random" | number,
): InstanceType<T> {
	if (typeof initMethod === "number") {
		return new arrayConstructor(
			sizeInBytes / arrayConstructor.BYTES_PER_ELEMENT,
		).fill(initMethod) as InstanceType<T>;
	}
	if (initMethod !== "random") {
		return initMethod;
	}

	const elemFunc: () => number =
		arrayConstructor instanceof Uint32Array
			? () => Math.floor(Math.random() * 4294967295)
			: Math.random;

	const hostArray = new arrayConstructor(
		sizeInBytes / arrayConstructor.BYTES_PER_ELEMENT,
	);

	for (let i = 0; i < hostArray.length; i++) {
		hostArray[i] = elemFunc();
	}

	return hostArray as InstanceType<T>;
}

export interface kernelExecutionResults {
	meanTime: number;
	meanTimePNow: number;
	result: number;
	iterations: number;
}

export class Dequantize {
	commandQueue: GPUCommandEncoder[] = [];
	timingEncoder: TimingHelper;
	gpu: GPUDevice;
	/** The device buffer used to read the contents of the kernel output */
	gpuReadBuffer: GPUBuffer;

	/**The name of the entry point for the kernel */
	entryPointName: string;

	/** The comppute pipeline attached to this dequantize instance.*/
	private pipeline: GPUComputePipeline;

	/** Holds the promise o */
	private computePipelinePromiseResult: Promise<GPUError | null>;
	private shaderCreationPromiseResult: Promise<GPUError | null>;

	private initialized = false;
	private NT_matmul: GPUBuffer;
	private bindGroup: GPUBindGroup;
	/** Set to true once the instance has been constructed to prevent init methods from being called after setup */
	private readonly constructed: boolean;
	static readonly NUM_WORKGROUPS_X: GPUSize32 = 16384;
	static readonly NUM_WORKGROUPS_Y: GPUSize32 = 1;
	static readonly NUM_WORKGROUPS_Z: GPUSize32 = 1;
	static readonly ENTRY_POINT = "fused_fused_dequantize3_NT_matmul12_kernel";
	static readonly NT_MATMUL_BYTES = 65536;
	static readonly LV1995_BYTES = 4194304;
	static readonly LV1996_BYTES = 16777216;
	static readonly RMS_NORM_BYTES = 32768;
	static readonly PACK_GRID_DIM_X = (Dequantize.NT_MATMUL_BYTES / Float32Array.BYTES_PER_ELEMENT) / 64;
	static readonly BIND_GROUP_LAYOUT: GPUBindGroupLayoutDescriptor = {
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
	};

	initializePipeline(
		layout: GPUPipelineLayout,
		module: GPUShaderModule,
		entryPointName: string,
	): GPUComputePipeline {
		return this.gpu.createComputePipeline({
			layout,
			compute: { module, entryPoint: entryPointName },
		});
	}

	/**
	 * Construct the dequantize instance.
	 * You must call {@linkcode Dequantize.init init } before running the kernel.
	 * The constructor cannot do this, as it is an async operation.
	 */
	constructor(moduleSource: string, entryPointName: string) {
		if (!navigator.gpu) {
			throw new Error(
				"WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag.",
			);
		}

		this.gpu = device;
		this.timingEncoder = new TimingHelper(device);

		const bindGroupLayout = device.createBindGroupLayout(
			Dequantize.BIND_GROUP_LAYOUT,
		);
		this.bindGroup = this.doBindBuffers(bindGroupLayout);

		const pipelineLayout = device.createPipelineLayout({
			bindGroupLayouts: [bindGroupLayout],
		});

		device.pushErrorScope("validation");
		const module = device.createShaderModule({ code: moduleSource });
		this.shaderCreationPromiseResult = device.popErrorScope();
		/* Create the pipeline and shader module */

		this.entryPointName = entryPointName;

		device.pushErrorScope("validation");
		this.pipeline = this.initializePipeline(
			pipelineLayout,
			module,
			entryPointName,
		);

		// Store the promise result of initializing the computePipeline.
		this.computePipelinePromiseResult = device.popErrorScope();

		// Get a GPU buffer for reading in an unmapped state.
		this.gpuReadBuffer = device.createBuffer({
			size: Dequantize.NT_MATMUL_BYTES,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		this.constructed = true;
	}

	/**
	 * Initialize the dequantize instance.
	 * Throws a GPUError if there was an error creating the shader module or compute Pipeline.
	 * If this throws an error, the instance is not usable and should be discarded.
	 */
	async init() {
		if (!this.constructed) {
			throw new Error("Dequantize instance was not constructed properly.");
		}
		// Catch the shaderCreation error first, if it's null there's no error, so get the compute pipeline error.
		const error =
			(await this.shaderCreationPromiseResult) ||
			(await this.computePipelinePromiseResult);
		if (error) {
			throw error;
		}
		this.recordCommands();
		const commandEncoder = this.gpu.createCommandEncoder();
		commandEncoder.copyBufferToBuffer(
			this.NT_matmul /* source buffer */,
			0 /* source offset */,
			this.gpuReadBuffer /* destination buffer */,
			0 /* destination offset */,
			Dequantize.NT_MATMUL_BYTES /* size */,
		);
		this.commandQueue.push(commandEncoder);

		// Wait for the GPU to finish...
		await this.gpu.queue.onSubmittedWorkDone();

		this.initialized = true;
	}

	/** Create a new commandEncoder and have it just copy the contents of NT_matmul to GPUReadBuffer */
	async readGPUBuffer(idx = 0): Promise<number> {
		// Unmap in case it was already mapped.
		if (this.gpuReadBuffer.mapState === "mapped") {
			this.gpuReadBuffer.unmap();
		}
		const commandEncoder = device.createCommandEncoder();
		commandEncoder.copyBufferToBuffer(
			this.NT_matmul /* source buffer */,
			0 /* source offset */,
			this.gpuReadBuffer /* destination buffer */,
			0 /* destination offset */,
			Dequantize.NT_MATMUL_BYTES /* size */,
		);
		this.commandQueue.push(commandEncoder);
		await this.gpu.queue.onSubmittedWorkDone();
		// Read buffer.

		await this.gpuReadBuffer.mapAsync(GPUMapMode.READ);
		const result = new Float32Array(this.gpuReadBuffer.getMappedRange())[idx];
		this.gpuReadBuffer.unmap();
		return result;
	}

	/**
	 * Run this dequantize instance.
	 * @returns
	 * @param numIterations The number of iterations to run.
	 * @throws Error if the initialization is still running.
	 */
	public async run(
		numIterations: number,
		result_index = 0,
	): Promise<kernelExecutionResults> {
		if (!this.initialized) {
			console.error("Initialization is still running...");
			throw new Error("Initialization is still running...");
		}
		// const computeFence = device.queue.createFence();
		// iteration = parseInt((document.getElementById("it") as HTMLInputElement).value , 10);

		const start = performance.now();
		for (let i = 0; i < numIterations; i++) {
			this.recordCommands();
		}

		device.queue.submit(this.commandQueue.map((enc) => enc.finish()));
		this.commandQueue.length = 0;

		const perf_now_ttl_time = await device.queue
			.onSubmittedWorkDone()
			.then(() => {
				return performance.now() - start;
			});
		const total_time = await this.timingEncoder.getResult();
		const meanTime = total_time / 1000000 / numIterations;
		const meanTimePNow = perf_now_ttl_time / numIterations;

		const result = await this.readGPUBuffer();

		// Read buffer

		return {
			meanTime,
			meanTimePNow,
			result: result,
			iterations: numIterations,
		};
	}

	/**
	 * Record the commands to the command queue.
	 */
	recordCommands() {
		const commandEncoder = this.gpu.createCommandEncoder();
		const passEncoder = this.timingEncoder.beginComputePass(commandEncoder);
		passEncoder.setPipeline(this.pipeline);
		passEncoder.setBindGroup(0, this.bindGroup);
		passEncoder.dispatchWorkgroups(
			Dequantize.NUM_WORKGROUPS_X,
			Dequantize.NUM_WORKGROUPS_Y,
			Dequantize.NUM_WORKGROUPS_Z,
		);
		passEncoder.end();
		this.commandQueue.push(commandEncoder);
	}

	clearCommandQueue() {
		this.commandQueue.length = 0;
	}

	createGPUBuffer(
		sizeInBytes: number,
		usage: GPUBufferUsageFlags,
		arrayType: Uint32ArrayConstructor,
		initialize?: "random" | Uint32Array,
	): GPUBuffer;
	createGPUBuffer(
		sizeInBytes: number,
		usage: GPUBufferUsageFlags,
		arrayType: Float32ArrayConstructor,
		initialize?: "random" | Float32Array,
	): GPUBuffer;

	/**
	 * Create a GPU buffer with the given parameters.
	 * @param device The GPU device
	 * @param sizeInBytes The size of the buffer, in bytes.
	 * @param usage The usage flags
	 * @param setData Whether to set the data in the buffer
	 * @param arrayType The constructor for the host buffer.
	 * @returns The GPU buffer in an unmapped state.
	 */
	createGPUBuffer<T extends Float32ArrayConstructor | Uint32ArrayConstructor>(
		sizeInBytes: number,
		usage: GPUBufferUsageFlags,
		arrayType: T,
		initialize?: InstanceType<T> | "random",
	): GPUBuffer {
		const buffer = this.gpu.createBuffer({
			mappedAtCreation: !!initialize,
			size: sizeInBytes,
			usage,
		});

		if (!initialize) return buffer;

		new arrayType(buffer.getMappedRange()).set(
			initHostBuffer(sizeInBytes, arrayType, initialize),
		);
		buffer.unmap();

		return buffer;
	}

	/** Create and initialize the buffers (with random data) to be used by the dequantize kernel.
	 * Sets `this.NT_matmul` to reference the GPU buffer that is provided as the NT_matmul buffer to the kernel.
	 */
	doBindBuffers(bindGroupLayout: GPUBindGroupLayout): GPUBindGroup {
		if (this.constructed) {
			return this.bindGroup;
		}
		const STORAGE = GPUBufferUsage.STORAGE;
		const lv1995 = this.createGPUBuffer(
			Dequantize.LV1995_BYTES,
			STORAGE,
			Uint32Array,
			"random",
		);
		const lv1996 = this.createGPUBuffer(
			Dequantize.LV1996_BYTES,
			STORAGE,
			Float32Array,
			"random",
		);
		const rms_norm46 = this.createGPUBuffer(
			Dequantize.RMS_NORM_BYTES,
			STORAGE,
			Float32Array,
			"random",
		);
		this.NT_matmul = this.createGPUBuffer(
			Dequantize.NT_MATMUL_BYTES,
			STORAGE | GPUBufferUsage.COPY_SRC,
			Float32Array,
		);

		const podArgs = this.createGPUBuffer(
			Uint32Array.BYTES_PER_ELEMENT,
			GPUBufferUsage.UNIFORM,
			Uint32Array,
			new Uint32Array([Dequantize.PACK_GRID_DIM_X]),
		);

		return this.gpu.createBindGroup({
			layout: bindGroupLayout,
			entries: [
				{ binding: 0, resource: { buffer: this.NT_matmul } },
				{ binding: 1, resource: { buffer: lv1995 } },
				{ binding: 2, resource: { buffer: lv1996 } },
				{ binding: 3, resource: { buffer: rms_norm46 } },
				{ binding: 4, resource: { buffer: podArgs } },
			],
		});
	}
}
