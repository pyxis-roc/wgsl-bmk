// Adapted from https://webgpufundamentals.org/webgpu/lessons/webgpu-timing.html
export class TimingHelper {
  private canTimeStamp: boolean;
  private device: GPUDevice;
  private querySet: GPUQuerySet;
  private resolveBuffer: GPUBuffer;
  private resultBuffers: Array<GPUBuffer> = [];
  // state can be 'free', 'need resolve', 'wait for result'
  private state = "free";

  constructor(device: GPUDevice) {
    this.device = device;
    this.canTimeStamp =
      device.features.has("timestamp-query") &&
      device.createQuerySet !== undefined;
    if (this.canTimeStamp) {
      this.querySet = device.createQuerySet({
        type: "timestamp",
        count: 2,
      });
      this.resolveBuffer = device.createBuffer({
        size: this.querySet.count * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
    }
  }

  private beginTimestampPass(
    encoder: GPUCommandEncoder,
    fnName: string,
    descriptor: GPUComputePassDescriptor
  ) {
    if (this.canTimeStamp) {
      //   assert(this.#state === "free", "state not free");
      this.state = "need resolve";

      const pass = encoder.beginComputePass({
        ...descriptor,
        ...{
          timestampWrites: {
            querySet: this.querySet,
            beginningOfPassWriteIndex: 0,
            endOfPassWriteIndex: 1,
          },
        },
      });

      const resolve = () => this.resolveTiming(encoder);
      pass.end = (function (origFn) {
        return function () {
          origFn.call(this);
          resolve();
        };
      })(pass.end);

      return pass;
    } else {
      return encoder.beginComputePass(descriptor);
    }
  }

  beginComputePass(
    encoder: GPUCommandEncoder,
    descriptor: GPUComputePassDescriptor = {}
  ) {
    return this.beginTimestampPass(encoder, "beginComputePass", descriptor);
  }

  private resolveTiming(encoder: GPUCommandEncoder) {
    if (!this.canTimeStamp) {
      return;
    }
    // assert(this.#state === "need resolve", "must call addTimestampToPass");
    this.state = "wait for result";

    const resultBuffer = this.device.createBuffer({
      size: this.resolveBuffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    encoder.resolveQuerySet(
      this.querySet,
      0,
      this.querySet.count,
      this.resolveBuffer,
      0
    );
    encoder.copyBufferToBuffer(
      this.resolveBuffer,
      0,
      resultBuffer,
      0,
      resultBuffer.size
    );
    // push to the list of result buffers so that we can use it!
    this.resultBuffers.push(resultBuffer);
  }

  async getResult() {
    if (!this.canTimeStamp) {
      return -1;
    }
    // assert(this.#state === "wait for result", "must call resolveTiming");
    this.state = "free";
    // wait for the device to finish the work!
    await this.device.queue.onSubmittedWorkDone();
    let duration = (
      await Promise.all(
        this.resultBuffers.map(async (buffer) => {
          await buffer.mapAsync(GPUMapMode.READ);
          const times = new BigInt64Array(buffer.getMappedRange());
          const duration = Number(times[1] - times[0]);
          buffer.unmap();
          buffer.destroy();
          return duration;
        })
      )
    ).reduce((a, b) => a + b, 0);

    /// Get the total duration

    /// clear the result buffer
    this.resultBuffers.length = 0;
    return duration;
  }
}
