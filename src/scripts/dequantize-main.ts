import { Dequantize } from "scripts/runners/dequantize";
import shader from "src/shaders/fused_fused_dequantize3_NT_matmul12_kernel.wgsl?raw";
import variantShader from "src/shaders/fused_fused_dequantize3_NT_matmul12_kernel_variant1.wgsl?raw";
import { createResultTable } from "./element-builders";
import { BooleanHolder, NumberHolder } from "./utils/valueHolders";

const RESULT_DIV_ID = "dequantize-result-div";
const VARIANT_RESULT_DIV_ID = "dequantize-result-div-variant";

const DEQUANTIZE_CARD_ID = "dequantize-bmk";
const DEQUANTIZE_VARIANT_CARD_ID = "dequantize-bmk-variant";

const MAX_ITERATIONS = 10000;
const MIN_ITERATIONS = 1;

/** How many elements are being processed per block */

/** How many iterations to run by default */
const DEFAULT_ITERATIONS = 100;
/// The maximum workgroup size we need is..
/// WORKGROUP_SIZE_X * WORK_PER_THREAD_X * 4
/** Number of decimal digits to display for the timestamp */
const TIMESTAMP_PRECISION = 3;
/** Number of decimal digits to display for the result */
const RESULT_PRECISION = 3;

let dequantize: Benchmark;
let dequantizeVariant: Benchmark;

/** Whether there is an experiment currently running. */
const isRunning = new BooleanHolder(false);

interface BmkConstructorParams {
	/** The name of the entry point in the shader */
	entryPoint: string;
	/** The shader code */
	shader: string;
	/** Whether this is the variant version of the benchmark */
	variant: boolean;
	iterationInput: HTMLInputElement;
	runButtonId: string;
	results_div_id: string;
	card_div_id: string;
}

class Benchmark {
	/** The kernel attached to the benchmark */
	readonly kernel: Dequantize;
	/** The table element attached to the benchmark */
	readonly resultsTable: HTMLTableElement;
	readonly warmup_div_id: string;
	readonly iterationHolder: NumberHolder;
	readonly initialization: Promise<void>;
	protected isInitialized = false;
	/** The id of the card div in the DOM */
	readonly card_id: string;

	constructor(params: BmkConstructorParams) {
		this.resultsTable = createResultTable(params.results_div_id + "-table");
		document
			.getElementById(params.results_div_id)
			?.appendChild(this.resultsTable);
		this.kernel = new Dequantize(params.shader, params.entryPoint);
		this.warmup_div_id = params.variant
			? "dequantize-warmup-variant"
			: "dequantize-warmup";

		this.iterationHolder = new NumberHolder(DEFAULT_ITERATIONS);

		this.card_id = params.card_div_id;
		document
			.getElementById(params.runButtonId)
			?.addEventListener("click", async () => {
				await this.run(this.iterationHolder.value);
			});

		params.iterationInput.addEventListener("input", () => {
			updateIterations(this.iterationHolder, params.iterationInput);
		});

		document.getElementById(params.runButtonId);

		this.initialization = this.doInit();
	}

	async init() {
		if (this.isInitialized) {
			return;
		}
		await this.initialization;
		this.isInitialized = true;
		document.getElementById(this.card_id)?.classList.remove("invisible");
	}

	async doInit() {
		if (this.isInitialized) {
			return;
		}
		await this.kernel.init();
	}

	addResultRow(
		timestamp_time: number,
		js_time: number,
		result: number,
		iterations: number,
	) {
		const tbody = this.resultsTable.querySelector("tbody");
		if (!tbody) {
			throw new Error("No tbody found in results table");
		}

		const row = document.createElement("tr");
		for (const value of [
			iterations,
			timestamp_time.toFixed(TIMESTAMP_PRECISION),
			js_time.toFixed(TIMESTAMP_PRECISION),
			result.toFixed(RESULT_PRECISION),
		]) {
			const td = document.createElement("td");
			td.classList.add("table-data");

			td.innerText = value.toString();
			row.appendChild(td);
		}
		tbody?.appendChild(row);
	}

	async run(iterations: number) {
		if (isRunning.testAndSet()) {
			throw new Error("Refusing to run while a test is already running.");
		}
		await this.kernel.run(iterations).then((results) => {
			this.addResultRow(
				results.meanTime,
				results.meanTimePNow,
				results.result,
				results.iterations,
			);
		});
		isRunning.value = false;
	}
}

function initializeDOM() {
	for (const id of ["it", "it-variant"]) {
		const elem = document.getElementById(id);
		if (!elem) {
			return;
		}
		elem.setAttribute("max", MAX_ITERATIONS.toString());
		elem.setAttribute("min", MIN_ITERATIONS.toString());
		elem.setAttribute("value", DEFAULT_ITERATIONS.toString());
	}
}

/// Add event listenner that validates the input and updates iteration
function updateIterations(counter: NumberHolder, target: HTMLInputElement) {
	const value = Number.parseInt(target.value);
	if (Number.isNaN(value)) {
		target.value = counter.value.toString();
		return;
	}
	if (value < MIN_ITERATIONS) {
		target.value = MIN_ITERATIONS.toString();
		counter.value = MIN_ITERATIONS;
		return;
	}
	if (value > MAX_ITERATIONS) {
		target.value = MAX_ITERATIONS.toString();
		counter.value = MAX_ITERATIONS;
		return;
	}
}

async function onLoad() {
	initializeDOM();
	dequantize = new Benchmark({
		entryPoint: "dequantize_plain",
		shader: shader,
		variant: false,
		iterationInput: document.getElementById("it") as HTMLInputElement,
		runButtonId: "dequantize-button",
		card_div_id: DEQUANTIZE_CARD_ID,
		results_div_id: RESULT_DIV_ID,
	});
	dequantizeVariant = new Benchmark({
		entryPoint: "dequantize_with_checks",
		shader: variantShader,
		variant: true,
		iterationInput: document.getElementById("it-variant") as HTMLInputElement,
		runButtonId: "dequantize-button-variant",
		card_div_id: DEQUANTIZE_VARIANT_CARD_ID,
		results_div_id: VARIANT_RESULT_DIV_ID,
	});

	const promises = [dequantize.init(), dequantizeVariant.init()];
	await Promise.all(promises);
}

await onLoad();
