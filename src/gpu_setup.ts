let adapter: GPUAdapter;
let adapter_info: GPUAdapterInfo;
let adapter_limits: GPUSupportedLimits;
let isInitialized = false;

type GPULimitsRequest = Partial<Omit<GPUSupportedLimits, "__brand">>;

let defaultDesiredFeatures: Iterable<GPUFeatureName> = ["timestamp-query"];
let defaultDesiredLimits: GPULimitsRequest = {
  maxComputeWorkgroupStorageSize: 32768,
};
let default_device: GPUDevice;
let timeSupport: boolean;

/// Requests a device with the desired features and limits. If the adapter does not support a feature,
/// then it is not requested. If the adatper supports a limit that is lower than the desired limit, then
/// the maximum supported limit is requested.
async function safeRequestDevice(
  adapter: GPUAdapter,
  desiredFeatures: Iterable<GPUFeatureName>,
  desiredLimits: GPULimitsRequest
): Promise<GPUDevice> {
  let device: GPUDevice;
  // for each feature, check if it is supported by the adapter, and if it is not, then do not
  // request it.
  const requestableFeatures = Array.from(desiredFeatures).filter((feature) =>
    adapter.features.has(feature)
  );
  const requestableLimits: GPULimitsRequest = {};
  for (const key in desiredLimits) {
    requestableLimits[key] = Math.min(desiredLimits[key], adapter.limits[key]);
  }
  try {
    device = await adapter.requestDevice({
      requiredFeatures: requestableFeatures,
      requiredLimits: requestableLimits,
    });
  } catch (e) {
    console.error("Error requesting device: ", e);
    throw e;
  }
  return device;
}

await (async () => {
  if (isInitialized) return;
  isInitialized = true;
  if (!navigator.gpu) {
    console.warn(
      "WebGPU is not supported. \
      For chrome, enable chrome://flags/#enable-unsafe-webgpu flag. " +
        "For firefox, ensure you are using firefox nightly and enable it \
      in about:config. Other browsers are untested."
    );
  }

  adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No appropriate GPUAdapter found.");
  }
  adapter_info = adapter.info;

  /// Use deprecated api if adapter.info is not available
  if (adapter.info === undefined || adapter.info === null) {
    adapter_info = await adapter.requestAdapterInfo();
  }

  /// get adapter limits.
  adapter_limits = adapter.limits;
  default_device = await safeRequestDevice(
    adapter,
    defaultDesiredFeatures,
    defaultDesiredLimits
  );
  // check if the browser supports timestamp query.
  // The adapter must support timestamp query AND the browser must implement createQuerySet.
  timeSupport =
    adapter.features.has("timestamp-query") &&
    default_device.createQuerySet != null;
})();

export {
  adapter,
  default_device,
  safeRequestDevice,
  timeSupport,
  adapter_info,
  adapter_limits,
};
