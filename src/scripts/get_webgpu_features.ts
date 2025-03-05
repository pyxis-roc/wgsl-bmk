import { adapter } from "./gpu_setup.js";

type HeaderLevel = 1 | 2 | 3 | 4 | 5 | 6;
type GPUAdapterInfoStringMap = Record<
	keyof Omit<GPUAdapterInfo, "__brand">,
	string
>;

type GPUSupportedLimitStringMap = Record<
	keyof Omit<GPUSupportedLimits, "__brand">,
	number
>;

interface KeyValueMapping {
	[key: string]: { toString(): string };
}

interface AddNewHeaderOptions {
	parent: HTMLElement;
	text: string;
	level?: HeaderLevel;
	classList?: Iterable<string>;
}

const addNewHeader = ({
	parent,
	text,
	level = 2,
	classList,
}: AddNewHeaderOptions) => {
	const header = document.createElement(`h${level}`);
	if (classList) header.classList.add(...classList, "mb-3");
	header.innerText = text;
	parent.appendChild(header);
};

interface AddNewDivOptions {
	parent: HTMLElement;
	text: { toString(): string };
	classList?: Iterable<string>;
}

const addNewDiv = ({ parent, text, classList }: AddNewDivOptions) => {
	const div = document.createElement("div");
	if (classList) div.classList.add(...classList, "mb-2");
	div.innerText = text.toString();
	parent.appendChild(div);
	return div;
};

interface AddFeatureListOptions {
	parent: HTMLElement;
	features?: Iterable<{ toString(): string }>;
	classList?: Iterable<string>;
	unsupported_content?: string;
	empty_content?: string;
}

const addFeatureList = ({
	parent,
	features,
	classList = [],
	unsupported_content = "⚠️ Querying this is unsupported by your browser",
	empty_content = "⚠️ None reported.",
}: AddFeatureListOptions) => {
	if (features) {
		for (const feature of features) {
			addNewDiv({
				parent,
				text: feature,
				classList: [...classList, "list-group-item"],
			});
		}
	} else if (features === undefined || features === null) {
		addNewDiv({
			parent,
			text: unsupported_content,
			classList: [...classList, "list-group-item", "list-group-item-warning"],
		});
	} else {
		addNewDiv({
			parent,
			text: empty_content,
			classList: [...classList, "list-group-item", "list-group-item-danger"],
		});
	}
};

interface AddKeyValueListOptions {
	parent: HTMLElement;
	kvList: KeyValueMapping;
	key_classList?: Iterable<string>;
	value_classList?: Iterable<string>;
	unsupported_content?: string;
	empty_content?: string;
}

const addKeyValueList = ({
	parent,
	kvList,
	key_classList = [],
	unsupported_content = "⚠️ Querying this is unsupported by your browser",
	empty_content = "⚠️ None reported.",
}: AddKeyValueListOptions) => {
	let empty = true;
	for (const key in kvList) {
		const featureDiv = document.createElement("div");
		featureDiv.className =
			"d-flex justify-content-between align-items-center list-group-item";
		const infoKey = document.createElement("span");
		infoKey.classList.add("fw-bold", ...key_classList);
		infoKey.innerText = key;
		featureDiv.appendChild(infoKey);
		const infoValue = document.createElement("span");
		infoValue.classList.add(...key_classList);
		infoValue.innerText =
			kvList[key]?.toString() !== "" ? kvList[key]?.toString() : "-";
		featureDiv.appendChild(infoValue);
		parent.appendChild(featureDiv);
		empty = false;
	}
	if (!empty) return;
	addNewDiv({
		parent,
		text: kvList ? empty_content : unsupported_content,
		classList: [
			...key_classList,
			kvList != null ? "list-group-item-danger" : "list-group-item-warning",
		],
	});
};

await (async () => {
	const webgpu_feature_div = document.getElementById("webgpu-features");
	if (!webgpu_feature_div) {
		console.warn("No div found with id 'webgpu-features'");
		return;
	}

	if (!navigator.gpu) {
		webgpu_feature_div.innerText = "WebGPU not supported";
		return;
	}
	// Otherwise, list the features
	const wgslFeatures = navigator.gpu.wgslLanguageFeatures;

	addNewHeader({
		parent: webgpu_feature_div,
		text: "WGSL Language Features",
		classList: ["feature-header-2"],
	});
	addFeatureList({
		parent: webgpu_feature_div,
		features: wgslFeatures,
		classList: ["list-group"],
		unsupported_content:
			"⚠️ Querying WGSL features is unsupported by your browser.",
	});

	// Now add adapter features
	addNewHeader({
		parent: webgpu_feature_div,
		text: "Adapter Properties",
		level: 2,
		classList: ["card"],
	});
	addNewHeader({
		parent: webgpu_feature_div,
		text: "Info",
		level: 3,
		classList: ["feature-header-3"],
	});

	let adapter_info = adapter.info;
	if (
		adapter.info === undefined ||
		// @ts-ignore
		(adapter.info === null && adapter.requestAdapterInfo)
	) {
		// @ts-ignore
		adapter_info = await adapter.requestAdapterInfo();
	}
	addKeyValueList({
		parent: webgpu_feature_div,
		kvList: adapter_info as GPUAdapterInfoStringMap,
		key_classList: ["feature-key"],
		value_classList: ["feature-value"],
		unsupported_content:
			"⚠️ Querying adapter info is unsupported by your browser.",
	});

	addNewHeader({
		parent: webgpu_feature_div,
		text: "Features",
		level: 3,
		classList: ["feature-header-3"],
	});

	const adapterFeatures = adapter.features;
	addFeatureList({
		parent: webgpu_feature_div,
		features: adapterFeatures,
		classList: ["list-group"],
	});

	const adapterLimits = adapter.limits;
	addNewHeader({
		parent: webgpu_feature_div,
		text: "Limits",
		level: 3,
		classList: ["feature-header-3"],
	});
	addKeyValueList({
		parent: webgpu_feature_div,
		kvList: adapterLimits as GPUSupportedLimitStringMap,
		key_classList: ["feature-key"],
		value_classList: ["feature-value"],
	});

	// Now add adapter limits
	const limitsHeader = document.createElement("h3");
	limitsHeader.innerText = "Adapter Limits";
})();
