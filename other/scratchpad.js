class Instruction {
  constructor(device) {
    this.device = device;
    this.bufferDeletionStack = [];
    this.unloadDeletionStack = [];

    this.initBindGroups();
  }

  initBindGroup(layout, buffers, label = "") {
    return this.device.createBindGroup({
      layout,
      entries: buffers.map((buffer, i) => ({
        binding: i,
        resource: { buffer },
      })),
      label,
    });
  }

  initBuffer(ops, row, col = 1, noDelete = false) {
    const buffer = this.device.createBuffer({
      size: this.bufferSize(row, col),
      usage: ops.map((u) => bufferUsageDict[u]).reduce((a, b) => a | b),
    });
    if (!noDelete) this.bufferDeletionStack.push(buffer);
    else this.unloadDeletionStack.push(buffer);
    return buffer;
  }

  bufferSize(dimA, dimB = 1) {
    return Math.ceil((dimA * dimB * Float32Array.BYTES_PER_ELEMENT) / 1) * 1;
  }

  initBindGroups() {
    const bg = (types) =>
      this.device.createBindGroupLayout({
        entries: types.map((entry, i) => ({
          binding: i,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: entry },
        })),
      });

    this.r_r_r_Layout = bg(["read-only-storage", "read-only-storage", "read-only-storage"]);
    this.r_r_Layout = bg(["read-only-storage", "read-only-storage"]);
    this.r_Layout = bg(["read-only-storage"]);
    this.u_s_Layout = bg(["uniform", "storage"]);
    this.u_s_s_s_Layout = bg(["uniform", "storage", "storage", "storage"]);
  }

  initPipeline(code, bindGroupLayouts, label = "", constants = {}) {
    return this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts }),
      compute: {
        module: this.device.createShaderModule({ code }),
        entryPoint: "main",
        constants,
      },
      label,
    });
  }

  unloadBuffers() {
    this.unloadDeletionStack.map((buffer) => buffer.destroy());
    this.unloadDeletionStack = [];
  }

  destroyBuffers() {
    this.bufferDeletionStack.map((buffer) => buffer.destroy());
    this.bufferDeletionStack = [];
  }
}

class FastMatMul extends Instruction {
  constructor(device) {
    super(device);
    this.name = "fastMatMul";
    this.pipelineCache = new Map();
  }

  getPipeline(rows) {
    const div4 = rows % 4 === 0;
    const pipelineCacheKey = div4 ? "fastMatMulNoCheck" : "fastMatMul";
    if (this.pipelineCache.has(pipelineCacheKey)) {
      return this.pipelineCache.get(pipelineCacheKey);
    }
    const kernel = div4 ? this.fastMatMulNoCheck : this.fastMatMul;
    const pipeline = this.initPipeline(kernel, [this.u_s_Layout, this.r_r_Layout], pipelineCacheKey);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  newInstance(rows, cols, shared, bufA, bufB) {
    const pipeline = this.getPipeline(rows);
    const uniformBuffer = this.initBuffer(["uniform", "copy_to"], 4);
    const resultBuf = this.initBuffer(["storage", "copy_from"], rows, cols);
    const opBindGroup = this.initBindGroup(this.u_s_Layout, [uniformBuffer, resultBuf], "opBindGroup");
    const inputBindGroup = this.initBindGroup(this.r_r_Layout, [bufA, bufB], "inputBindGroup");
    const workgroups = { x: wgSize(cols, 64), y: wgSize(rows, 32) };
    this.device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([rows, cols, Math.ceil(cols / 4), Math.ceil(shared / 4)]));

    return {
      resultBuf,
      pass: {
        pipeline,
        groups: [opBindGroup, inputBindGroup],
        workgroups,
      },
    };
  }

  fastMatMul = `
    struct CMeta {
      M: u32,
      N: u32,
      ND4: u32,
      KD4: u32,
    }

    @group(1) @binding(0) var<storage,read> array_a: array<vec4<f32>>;
    @group(1) @binding(1) var<storage,read> array_b: array<vec4<f32>>;

    @group(0) @binding(0) var<uniform> cmeta: CMeta;
    @group(0) @binding(1) var<storage,read_write> array_c: array<vec4<f32>>;

    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      var M: u32 = cmeta.M;
      var N: u32 = cmeta.N;
      var ND4: u32 = cmeta.ND4;
      var KD4: u32 = cmeta.KD4;
      var x: u32 = global_id.x;
      var y: u32 = global_id.y;

      if (x * 8 >= N || y * 4 >= M) {
        return;
      }

      var sum00: vec4<f32> = vec4<f32>();
      var sum01: vec4<f32> = vec4<f32>();
      var sum02: vec4<f32> = vec4<f32>();
      var sum03: vec4<f32> = vec4<f32>();
      var sum10: vec4<f32> = vec4<f32>();
      var sum11: vec4<f32> = vec4<f32>();
      var sum12: vec4<f32> = vec4<f32>();
      var sum13: vec4<f32> = vec4<f32>();

      for(var k: u32 = 0u; k < KD4; k = k + 1u) {
        var arow0: vec4<f32> = array_a[(y * 4u + 0u) * KD4 + k];
        var arow1: vec4<f32> = array_a[(y * 4u + 1u) * KD4 + k];
        var arow2: vec4<f32> = array_a[(y * 4u + 2u) * KD4 + k];
        var arow3: vec4<f32> = array_a[(y * 4u + 3u) * KD4 + k];
        var brow: vec4<f32>;

        brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.x) * brow + sum00;
        sum01 = vec4<f32>(arow1.x) * brow + sum01;
        sum02 = vec4<f32>(arow2.x) * brow + sum02;
        sum03 = vec4<f32>(arow3.x) * brow + sum03;

        brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.x) * brow + sum10;
        sum11 = vec4<f32>(arow1.x) * brow + sum11;
        sum12 = vec4<f32>(arow2.x) * brow + sum12;
        sum13 = vec4<f32>(arow3.x) * brow + sum13;

        brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.y) * brow + sum00;
        sum01 = vec4<f32>(arow1.y) * brow + sum01;
        sum02 = vec4<f32>(arow2.y) * brow + sum02;
        sum03 = vec4<f32>(arow3.y) * brow + sum03;

        brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.y) * brow + sum10;
        sum11 = vec4<f32>(arow1.y) * brow + sum11;
        sum12 = vec4<f32>(arow2.y) * brow + sum12;
        sum13 = vec4<f32>(arow3.y) * brow + sum13;

        brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.z) * brow + sum00;
        sum01 = vec4<f32>(arow1.z) * brow + sum01;
        sum02 = vec4<f32>(arow2.z) * brow + sum02;
        sum03 = vec4<f32>(arow3.z) * brow + sum03;

        brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.z) * brow + sum10;
        sum11 = vec4<f32>(arow1.z) * brow + sum11;
        sum12 = vec4<f32>(arow2.z) * brow + sum12;
        sum13 = vec4<f32>(arow3.z) * brow + sum13;

        brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.w) * brow + sum00;
        sum01 = vec4<f32>(arow1.w) * brow + sum01;
        sum02 = vec4<f32>(arow2.w) * brow + sum02;
        sum03 = vec4<f32>(arow3.w) * brow + sum03;

        brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.w) * brow + sum10;
        sum11 = vec4<f32>(arow1.w) * brow + sum11;
        sum12 = vec4<f32>(arow2.w) * brow + sum12;
        sum13 = vec4<f32>(arow3.w) * brow + sum13;
      }

      if (y * 4u + 0u < M) {
        array_c[x * 2u + 0u + (y * 4u + 0u) * ND4] = sum00;
        array_c[x * 2u + 1u + (y * 4u + 0u) * ND4] = sum10;
      }
      if (y * 4u + 1u < M) {
        array_c[x * 2u + 0u + (y * 4u + 1u) * ND4] = sum01;
        array_c[x * 2u + 1u + (y * 4u + 1u) * ND4] = sum11;
      }
      if (y * 4u + 2u < M) {
        array_c[x * 2u + 0u + (y * 4u + 2u) * ND4] = sum02;
        array_c[x * 2u + 1u + (y * 4u + 2u) * ND4] = sum12;
      }
      if (y * 4u + 3u < M) {
        array_c[x * 2u + 0u + (y * 4u + 3u) * ND4] = sum03;
        array_c[x * 2u + 1u + (y * 4u + 3u) * ND4] = sum13;
      }
    }
  `;

  fastMatMulNoCheck = `
    struct CMeta {
      M: u32,
      N: u32,
      ND4: u32,
      KD4: u32,
    }

    @group(1) @binding(0) var<storage,read> array_a: array<vec4<f32>>;
    @group(1) @binding(1) var<storage,read> array_b: array<vec4<f32>>;

    @group(0) @binding(0) var<uniform> cmeta: CMeta;
    @group(0) @binding(1) var<storage,read_write> array_c: array<vec4<f32>>;

    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      var M: u32 = cmeta.M;
      var N: u32 = cmeta.N;
      var ND4: u32 = cmeta.ND4;
      var KD4: u32 = cmeta.KD4;
      var x: u32 = global_id.x;
      var y: u32 = global_id.y;

      if (x * 8 >= N || y * 4 >= M) {
        return;
      }

      var sum00: vec4<f32> = vec4<f32>();
      var sum01: vec4<f32> = vec4<f32>();
      var sum02: vec4<f32> = vec4<f32>();
      var sum03: vec4<f32> = vec4<f32>();
      var sum10: vec4<f32> = vec4<f32>();
      var sum11: vec4<f32> = vec4<f32>();
      var sum12: vec4<f32> = vec4<f32>();
      var sum13: vec4<f32> = vec4<f32>();

      for(var k: u32 = 0u; k < KD4; k = k + 1u) {
        var arow0: vec4<f32> = array_a[(y * 4u + 0u) * KD4 + k];
        var arow1: vec4<f32> = array_a[(y * 4u + 1u) * KD4 + k];
        var arow2: vec4<f32> = array_a[(y * 4u + 2u) * KD4 + k];
        var arow3: vec4<f32> = array_a[(y * 4u + 3u) * KD4 + k];
        var brow: vec4<f32>;

        brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.x) * brow + sum00;
        sum01 = vec4<f32>(arow1.x) * brow + sum01;
        sum02 = vec4<f32>(arow2.x) * brow + sum02;
        sum03 = vec4<f32>(arow3.x) * brow + sum03;

        brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.x) * brow + sum10;
        sum11 = vec4<f32>(arow1.x) * brow + sum11;
        sum12 = vec4<f32>(arow2.x) * brow + sum12;
        sum13 = vec4<f32>(arow3.x) * brow + sum13;

        brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.y) * brow + sum00;
        sum01 = vec4<f32>(arow1.y) * brow + sum01;
        sum02 = vec4<f32>(arow2.y) * brow + sum02;
        sum03 = vec4<f32>(arow3.y) * brow + sum03;

        brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.y) * brow + sum10;
        sum11 = vec4<f32>(arow1.y) * brow + sum11;
        sum12 = vec4<f32>(arow2.y) * brow + sum12;
        sum13 = vec4<f32>(arow3.y) * brow + sum13;

        brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.z) * brow + sum00;
        sum01 = vec4<f32>(arow1.z) * brow + sum01;
        sum02 = vec4<f32>(arow2.z) * brow + sum02;
        sum03 = vec4<f32>(arow3.z) * brow + sum03;

        brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.z) * brow + sum10;
        sum11 = vec4<f32>(arow1.z) * brow + sum11;
        sum12 = vec4<f32>(arow2.z) * brow + sum12;
        sum13 = vec4<f32>(arow3.z) * brow + sum13;

        brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.w) * brow + sum00;
        sum01 = vec4<f32>(arow1.w) * brow + sum01;
        sum02 = vec4<f32>(arow2.w) * brow + sum02;
        sum03 = vec4<f32>(arow3.w) * brow + sum03;

        brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.w) * brow + sum10;
        sum11 = vec4<f32>(arow1.w) * brow + sum11;
        sum12 = vec4<f32>(arow2.w) * brow + sum12;
        sum13 = vec4<f32>(arow3.w) * brow + sum13;
      }

      array_c[x * 2u + 0u + (y * 4u + 0u) * ND4] = sum00;
      array_c[x * 2u + 1u + (y * 4u + 0u) * ND4] = sum10;
      array_c[x * 2u + 0u + (y * 4u + 1u) * ND4] = sum01;
      array_c[x * 2u + 1u + (y * 4u + 1u) * ND4] = sum11;
      array_c[x * 2u + 0u + (y * 4u + 2u) * ND4] = sum02;
      array_c[x * 2u + 1u + (y * 4u + 2u) * ND4] = sum12;
      array_c[x * 2u + 0u + (y * 4u + 3u) * ND4] = sum03;
      array_c[x * 2u + 1u + (y * 4u + 3u) * ND4] = sum13;
    }
  `;
}

class TestGPT {
  constructor(folder, type, doAttentionCache = false) {
    this.folder = folder;
    this.tokenizerType = type;
    this.initialized = false;

    this.device;
    this.model;
    this.tokenizer;
    this.params;
    this.minBufferOffset = 1;
    this.doAttentionCache = doAttentionCache;

    this.defaultPrompt;
    this.defaultTopK;
    this.defaultTemperature;
    this.defaultTokens;

    this.bufferDeletionStack = [];
    this.unloadDeletionStack = [];
  }

  async initialize() {
    if (this.initialized) return console.error("Model already initialized");
    if (!navigator.gpu) throw new Error("WebGPU is not supported");

    const adapter = await navigator.gpu.requestAdapter();
    this.device = await adapter.requestDevice();

    this.matMulOperation = new FastMatMul(this.device);

    const dimM = 10;
    const dimN = 10;
    const demo = new Float32Array(dimM * dimN);
    for (let i = 0; i < dimM * dimN; i++) demo[i] = 1;
    const weights1 = this.initTensor(demo, [dimM, dimN], ["storage", "copy_from"]);
    // const weights2 = this.initTensor(demo, [dimM, dimN], ["storage", "copy_from"]);
    this.inputBuffer = this.initBuffer(["storage", "copy_from", "copy_to"], dimM, dimN);

    this.computePasses = [];
    let intermediateBuffer = this.inputBuffer;
    for (let i = 0; i < 10; i++) {
      let { pass, resultBuf } = this.matMulOperation.newInstance(10, 10, 10, intermediateBuffer, weights1);
      intermediateBuffer = resultBuf;
      this.computePasses.push(pass);
    }
    this.resultBuffer = intermediateBuffer;
    this.outputBuffer = this.initBuffer(["map_read", "copy_to"], dimM, dimN);

    this.initialized = true;
  }

  initBindGroup(layout, buffers) {
    return this.device.createBindGroup({
      layout,
      entries: buffers.map((buffer, i) => ({
        binding: i,
        resource: { buffer },
      })),
    });
  }

  async fetchAndSplitQKVWeightTensors(url, dims, ops) {
    const data = transpose(await fetchBin(url), dims[0], dims[1]);

    const qWeights = transpose(data.subarray(0, dims[0] * dims[0]), dims[0], dims[0]);
    const kWeights = transpose(data.subarray(dims[0] * dims[0], dims[0] * dims[0] * 2), dims[0], dims[0]);
    const vWeights = transpose(data.subarray(dims[0] * dims[0] * 2, dims[0] * dims[0] * 3), dims[0], dims[0]);

    const qWeightsBuffer = this.initTensor(qWeights, [dims[0], dims[0]], ops);
    const kWeightsBuffer = this.initTensor(kWeights, [dims[0], dims[0]], ops);
    const vWeightsBuffer = this.initTensor(vWeights, [dims[0], dims[0]], ops);

    return [qWeightsBuffer, kWeightsBuffer, vWeightsBuffer];
  }

  async fetchAndSplitQKVBiasTensors(url, dims, ops) {
    const data = await fetchBin(url);

    const qBias = data.subarray(0, dims[0]);
    const kBias = data.subarray(dims[0], dims[0] * 2);
    const vBias = data.subarray(dims[0] * 2, dims[0] * 3);

    const qBiasBuffer = this.initTensor(qBias, [dims[0]], ops);
    const kBiasBuffer = this.initTensor(kBias, [dims[0]], ops);
    const vBiasBuffer = this.initTensor(vBias, [dims[0]], ops);

    return [qBiasBuffer, kBiasBuffer, vBiasBuffer];
  }

  async fetchAndInitTensor(url, dims, ops, constructor) {
    console.log("Fetching and initializing tensor...", url);
    const data = await fetchBin(url);
    return this.initTensor(data, dims, ops, constructor ?? Float32Array);
  }

  initOutputBuffer(commandEncoder, buffer, row, col) {
    const outputBuffer = this.initBuffer(["map_read", "copy_to"], row, col);
    commandEncoder.copyBufferToBuffer(buffer, 0, outputBuffer, 0, this.bufferSize(row, col));
    return outputBuffer;
  }

  initBuffer(ops, row, col = 1, noDelete = false) {
    const buffer = this.device.createBuffer({
      size: this.bufferSize(row, col),
      usage: ops.map((u) => bufferUsageDict[u]).reduce((a, b) => a | b),
    });
    if (!noDelete) this.bufferDeletionStack.push(buffer);
    else this.unloadDeletionStack.push(buffer);
    return buffer;
  }

  initTensor(data, dims, ops, constructor = Float32Array) {
    const buffer = this.device.createBuffer({
      size: this.bufferSize(dims[0], dims[1], dims[2] || 1),
      usage: ops.map((u) => bufferUsageDict[u]).reduce((a, b) => a | b),
      mappedAtCreation: true,
    });
    const array = new constructor(buffer.getMappedRange());
    array.set(data);
    buffer.unmap();
    this.unloadDeletionStack.push(buffer);
    return buffer;
  }

  bufferSize(dimX, dimY = 1, dimZ = 1) {
    return Math.ceil((dimX * dimY * dimZ * Float32Array.BYTES_PER_ELEMENT) / this.minBufferOffset) * this.minBufferOffset;
  }

  unloadBuffers() {
    this.unloadDeletionStack.map((buffer) => buffer.destroy());
    this.unloadDeletionStack = [];
  }

  destroyBuffers() {
    this.bufferDeletionStack.map((buffer) => buffer.destroy());
    this.bufferDeletionStack = [];
  }

  initBindGroups() {
    const bg = (types) =>
      this.device.createBindGroupLayout({
        entries: types.map((entry, i) => ({
          binding: i,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: entry },
        })),
      });

    this.r_r_r_Layout = bg(["read-only-storage", "read-only-storage", "read-only-storage"]);
    this.r_r_Layout = bg(["read-only-storage", "read-only-storage"]);
    this.r_Layout = bg(["read-only-storage"]);
    this.u_s_Layout = bg(["uniform", "storage"]);
    this.u_s_s_s_Layout = bg(["uniform", "storage", "storage", "storage"]);
  }

  async initPipelines() {
    const p = (code, bindGroupLayouts) => {
      return this.device.createComputePipelineAsync({
        layout: this.device.createPipelineLayout({ bindGroupLayouts }),
        compute: {
          module: this.device.createShaderModule({ code }),
          entryPoint: "main",
        },
      });
    };
  }

  async testAttn() {
    initializeOperations(this.device);
    const [
      qkvWeightArray,
      qkvBiasArray,
      linearWeightsBuffer,
      linearBiasBuffer,
      inputBuffer,
      dOutputBuffer,
    ] = await Promise.all([
      this.fetchAndSplitQKVWeightTensors(`weights/test/c_attn_w.bin`, [1536, 3 * 1536], ["storage", "copy_from"]),
      this.fetchAndSplitQKVBiasTensors(`weights/test/c_attn_b.bin`, [1536], ["storage"]),
      this.fetchAndInitTensor(`weights/test/c_proj_w.bin`, [1536, 1536], ["storage"]),
      this.fetchAndInitTensor(`weights/test/c_proj_b.bin`, [1536], ["storage"]),
      this.fetchAndInitTensor(`weights/test/inp.bin`, [160, 1536], ["storage"]),
      this.fetchAndInitTensor(`weights/test/grad_res.bin`, [160, 1536], ["storage"]),
    ]);
    const { resultBuffer, passes, caches } = AttentionBlock.newFusedInstance(
      160,
      1536,
      0.0883883476,
      12,
      128,
      inputBuffer,
      qkvWeightArray[0],
      qkvBiasArray[0],
      qkvWeightArray[1],
      qkvBiasArray[1],
      qkvWeightArray[2],
      qkvBiasArray[2],
      linearWeightsBuffer,
      linearBiasBuffer,
      FastMatMulBlock,
      SoftmaxBlock,
    );
    await runComputePasses(this.device, passes);
    console.log(formatAsMatrix(
      (await serializeBuffer(this.device, resultBuffer)).float32ArrayBuffer,
      160, 1536
    ));
    const {
      dInputBuffer,
      dQWeightsBuffer,
      dKWeightsBuffer,
      dVWeightsBuffer,
      dAttentionSoftmaxOutputsBufferMasked,
      dAttentionSoftmaxOutputsBuffer,
      passes: passes2
    } = AttentionBackwards.newInstance(
      dOutputBuffer,
      caches,
      160,
      1536,
      0.0883883476,
      12,
      128,
      inputBuffer,
      qkvWeightArray[0],
      qkvBiasArray[0],
      qkvWeightArray[1],
      qkvBiasArray[1],
      qkvWeightArray[2],
      qkvBiasArray[2],
      linearWeightsBuffer,
      linearBiasBuffer,
      FastMatMulBlock,
      SoftmaxBlock,
    )
    await runComputePasses(this.device, passes2);
    console.log(formatAsMatrix(
      (await serializeBuffer(this.device, dAttentionSoftmaxOutputsBufferMasked)).float32ArrayBuffer,
      12 * 160, 160
    ));
    console.log(formatAsMatrix(
      (await serializeBuffer(this.device, dAttentionSoftmaxOutputsBuffer)).float32ArrayBuffer,
      12 * 160, 160
    ));
    console.log("dInput", formatAsMatrix(
      (await serializeBuffer(this.device, dInputBuffer)).float32ArrayBuffer,
      160, 1536
    ));
    console.log("dKWeights", formatAsMatrix(
      (await serializeBuffer(this.device, dQWeightsBuffer)).float32ArrayBuffer,
      1536, 1536
    ));
  }

  async testSoftmax() {
    initializeOperations(this.device);
    const A = new Float32Array([1.9269150495529175, 1.4872841835021973, 0.9007171988487244, -2.1055214405059814, 0.6784184575080872, -1.2345449924468994, -0.043067481368780136, -1.6046669483184814, -0.7521361708641052, 1.6487228870391846, -0.3924786448478699, -1.4036067724227905, -0.7278812527656555, -0.5594298839569092, -0.7688389420509338, 0.7624453902244568, 1.6423169374465942, -0.15959732234477997, -0.4973974823951721, 0.4395892322063446, -0.7581311464309692, 1.078317642211914, 0.8008005023002625, 1.680620551109314, 1.27912437915802, 1.2964228391647339, 0.610466480255127, 1.334737777709961, -0.2316243201494217, 0.041759490966796875, -0.2515752613544464, 0.859858512878418, -1.3846741914749146, -0.8712361454963684, -0.2233659327030182, 1.7173610925674438, 0.31887972354888916, -0.42451897263526917, 0.30572032928466797, -0.7745925188064575, -1.5575722455978394, 0.9956361055374146, -0.8797858357429504, -0.601142942905426, -1.2741514444351196, 2.1227850914001465, -1.234653353691101, -0.4879138767719269, -0.9138230085372925, -0.6581372618675232, 0.07802387326955795, 0.5258087515830994, -0.48799172043800354, 1.1913691759109497, -0.8140076398849487, -0.7359928488731384, -1.4032478332519531, 0.036003824323415756, -0.06347727030515671, 0.6756148934364319, -0.0978068932890892, 1.8445940017700195, -1.184537410736084, 1.3835493326187134, 1.4451336860656738, 0.8564125895500183, 2.218075752258301, 0.5231655240058899, 0.34664666652679443, -0.19733144342899323, -1.054589867591858, 1.2779951095581055, -0.1721908152103424, 0.5237884521484375, 0.056621816009283066, 0.4262961447238922, 0.575005054473877, -0.641724169254303, -2.2063980102539062, -0.7508036494255066, 0.01086814422160387, -0.33874261379241943, -1.3406798839569092, -0.5853705406188965, 0.5361881852149963, 0.5246226191520691, 1.1412016153335571, 0.051643650978803635, 0.7439519762992859, -0.4815842807292938, -1.0494657754898071, 0.603898823261261, -1.7222950458526611, -0.827768862247467, 1.3347028493881226, 0.48353925347328186, -2.5095443725585938, 0.4880010485649109, 0.7845868468284607, 0.028647208586335182, 0.640755295753479, 0.5832474231719971, 1.0669267177581787, -0.4501534402370453, -0.18526746332645416, 0.7527588605880737, 0.4047577977180481, 0.17846599221229553, 0.2649095058441162, 1.2731683254241943, -0.0013108636485412717, -0.3036036491394043, -1.457029104232788, -0.10233523696660995, -0.5991530418395996, 0.4770564138889313, 0.7261772155761719, 0.09115186333656311, -0.3890652060508728, 0.5279164910316467, -0.012685478664934635, 0.24083632230758667, 0.13253536820411682, 0.7642406225204468, 1.095009684562683, 0.3398909568786621, 0.7199674248695374, 0.41140761971473694, 1.931160569190979, 1.0118638277053833, -1.4364064931869507, -1.1298598051071167, -0.1360345333814621, 1.6354097127914429, 0.6547404527664185, 0.5760046243667603, 1.1415079832077026, 0.018564576283097267, -1.8058050870895386, 0.9254347681999207, -0.3753443658351898, 1.0330872535705566, -0.6866511702537537, 0.6368135809898376, -0.9726738929748535, 0.9584577679634094, 1.6192004680633545, 1.450609803199768, 0.2694803476333618, -0.21037596464157104, -0.7328027486801147, 0.1042979285120964, 0.3487516939640045, 0.9675940275192261, -0.46568843722343445, 1.6047972440719604, -2.4801204204559326, -0.4175437390804291, -1.1954537630081177, 0.8123368620872498, -1.9005532264709473, 0.22857652604579926, 0.02485940419137478, -0.3459502160549164, 0.2868320941925049, -0.7308424115180969, 0.17482034862041473, -1.0939288139343262, -1.6021603345870972, 1.3528969287872314, 1.288827657699585, 0.05229555815458298, -1.546850562095642, 0.7567060589790344, 0.7755194902420044, 2.026535749435425, 0.03581761196255684, 0.12058872729539871, -0.8056638240814209, -0.20757682621479034, -0.9319477677345276, -1.5909663438796997, -1.1359758377075195, -0.5225975513458252, -0.5187733173370361, -1.5012763738632202, -1.9266544580459595, 0.1278512328863144, 1.0229133367538452, -0.5557947754859924, 0.7042727470397949, 0.7098760008811951, 1.7743884325027466, -0.9215506911277771, 0.9624499082565308, -0.3370155692100525, -1.1753336191177368, 0.35805708169937134, 0.47876808047294617, 1.353700041770935, 0.5260620713233948, 2.1120381355285645, -0.5207571387290955, -0.9320058822631836, 0.18516133725643158, 1.0686918497085571, 1.3065342903137207, 0.4598345160484314, -0.8146268725395203, -1.0212390422821045, -0.49492350220680237, -0.5922514796257019, 0.1543159782886505, 0.4407668709754944, -0.1482921838760376, -2.3184430599212646, -0.39799532294273376, 1.080486536026001, -1.7808643579483032, 1.5080455541610718, 0.30942851305007935, -0.5003092288970947, 1.0350031852722168, 1.6896475553512573, -0.004505051765590906, 1.666792392730713, 0.15392017364501953, -1.0602532625198364, -0.572657585144043, 0.0835680440068245, 0.39990535378456116, 1.989207148551941, -0.07198750972747803, -0.9060945510864258, -2.0487122535705566, -1.0810552835464478, 0.017623215913772583, 0.07822597771883011, 0.19315828382968903, 0.40967339277267456, -0.9291301965713501, 0.2761908769607544, -0.5388751029968262, 0.4625823497772217, -0.8718891739845276, -0.027118293568491936, -0.3532457649707794, 1.4638570547103882, 1.255434274673462, -0.7149558067321777, 0.8539194464683533, 0.5129911303520203, 0.5397310256958008, 0.5655050277709961, 0.5057917237281799, 0.22245368361473083, -0.6854815483093262, 0.5635589957237244, -1.507175087928772, -1.610666036605835, -1.4790465831756592, 0.43227431178092957, -0.1250254064798355, 0.7821183800697327, -1.598767638206482, -0.10912995785474777, 0.7151994705200195, 0.039139606058597565, 1.305860161781311, 0.246592715382576, -1.9775909185409546, 0.01789604313671589, -1.379301905632019, 0.625802755355835, -2.5849504470825195, -0.023999439552426338, -0.12219284474849701, -0.746995210647583, 1.7093087434768677, 0.05792269483208656, 1.1929808855056763, 1.9372931718826294, 0.7287134528160095, 0.9808937907218933, 0.41459232568740845, 1.156563401222229, 0.2690545618534088, -0.036629438400268555, 0.9732940793037415, -1.0150787830352783, -0.5419175624847412, -0.44102486968040466, -0.3136177957057953, -0.12925422191619873, -0.7149620652198792, -0.04756207764148712, 2.0207436084747314, 0.25391900539398193, 0.9364385008811951, 0.7122364044189453, -0.031765542924404144, 0.10164086520671844, 1.34330415725708, 0.7132695913314819, 0.4038029611110687, -0.7139783501625061, 0.8337290287017822, -0.9585452675819397, 0.45363426208496094, 1.2460919618606567, -2.3065085411071777, -1.286892056465149, 0.17988650500774384, -2.126762628555298, -0.13408353924751282, -1.0407686233520508, -0.7647228837013245, -0.05528254434466362, 1.204850673675537, -0.9824733138084412, 0.4334380030632019, -0.7171904444694519, 1.055368423461914, -1.4533969163894653, 0.46515071392059326, 0.37139150500297546, -0.004656785633414984, 0.07954943925142288, 0.3781784772872925, 0.7051143050193787, -1.7236979007720947, -0.8434811234474182, 0.4351435601711273, 0.26588720083236694, -0.5870985388755798, 0.0826888456940651, 0.885380744934082, 0.1824439913034439, 0.7863810062408447, -0.05792016535997391, 0.5666652917861938, -0.7097625136375427, -0.4875054359436035, 0.05009583756327629, 0.6084084510803223, 1.6308681964874268, -0.08472305536270142, 1.0844124555587769, 0.9477656483650208, -0.6766292452812195, -0.5730168223381042, -0.3303174376487732, -0.7939430475234985, 0.3752319812774658, 0.08790969103574753, -1.241483449935913, -0.3202543258666992, -0.8443779945373535, -0.5513466000556946, 1.9889612197875977, 1.900311827659607, 1.6950805187225342, 0.028089528903365135, -0.17536965012550354, -1.7734957933425903, -0.7046411633491516, -0.3946518898010254, 1.8868111371994019, -0.21844324469566345, 0.1662992537021637, 2.1441681385040283, 1.7045671939849854, 0.3459012508392334, 0.6424751281738281, -0.20395448803901672, 0.6853673458099365, -0.1396879255771637, -1.1807503700256348, -1.282929539680481, 0.448485791683197, -0.590737521648407, 0.8540631532669067, -0.4900679290294647, -0.35945725440979004, 0.6663737893104553, -0.0742657482624054, -0.20960482954978943, 0.16632141172885895, 1.4703037738800049, -0.9390866756439209, -0.6013189554214478, -0.09964022785425186, -0.9851518273353577, -2.488459348678589, -0.33131900429725647, 0.8435799479484558, 0.9874473810195923, -0.33197471499443054, -0.8076189756393433, 0.824364185333252, 0.024699924513697624, -1.0641485452651978, -0.7601934671401978, -0.40750598907470703, 0.9623646140098572, -0.14264193177223206, 0.15271379053592682, -0.0388023778796196, 0.9446058869361877, -1.5824053287506104, 0.9871290922164917, 1.1456739902496338, -0.14181147515773773, -0.2763414680957794, -0.19321373105049133, 0.7767809629440308, 0.6838752627372742, -1.324589490890503, -0.5160817503929138, 0.6001842617988586, -0.4702207148075104, -0.608643651008606, -0.04619227349758148, -1.6457397937774658, -0.4833274185657501, -0.7402940392494202, 0.31428107619285583, 0.1415552943944931, 1.0348176956176758, -0.6264376044273376, -0.5150921940803528, 0.6902899742126465, -0.4939991533756256, 1.1366126537322998, -0.46184006333351135, 1.419979453086853, 0.848518967628479, -0.047891248017549515, 0.668560266494751, 1.0429801940917969, 0.6899018287658691, -1.3129348754882812, 0.037803765386343, -1.1702114343643188, -0.10318558663129807, 1.1894739866256714, 0.7606943845748901, -0.7463049292564392, -1.3838841915130615, 0.4868715703487396, -1.0020296573638916, 0.0329488143324852, -0.4291958808898926, -0.9817978739738464, -0.6420586109161377, 0.8265888094902039, 1.5913959741592407, -0.1208132952451706, -0.48302069306373596, 0.11329790204763412, 0.07715096324682236, -0.9228128790855408, -1.2619991302490234, 1.0860532522201538, 1.096641182899475, -0.6836934685707092, 0.06604336202144623, -0.0007741411682218313, 0.1620604544878006, 1.1959582567214966, -1.3061535358428955, -1.4039719104766846, -1.0597201585769653, 0.3057299852371216, 0.4150581359863281, -0.7174144983291626, 2.833967924118042, 1.9534740447998047, 2.0486814975738525, -1.0880383253097534, 1.621694564819336, 0.8512656688690186, -0.40046998858451843, -0.6088272333145142, -0.5080955028533936, -0.6184902191162109, -1.647040605545044, -1.0362098217010498, -0.4503057301044464, -0.07296605408191681, -0.5479547381401062, -1.1425533294677734, -0.4487519860267639, -0.03045438788831234, 0.3830311596393585, -0.04476971551775932, 1.1799415349960327, -0.33142781257629395, 0.6495042443275452, 0.09495851397514343, -0.7525873184204102, -0.6472296714782715, -1.282261610031128, 1.96529221534729, -0.9638491272926331, -2.5667941570281982, 0.7096128463745117, 0.8198426961898804, 0.6214461326599121, 0.4231860339641571, -0.33889803290367126, 0.5179733037948608]);
    const gradRes = new Float32Array([-1.36376953125, 0.19295759499073029, -0.6103342771530151, 0.16323445737361908, 1.51017165184021, 0.21230429410934448, -0.7252010703086853, -0.9527732729911804, 0.5216946601867676, -0.4638674259185791, 0.18237744271755219, -0.38666075468063354, -1.7906768321990967, 0.09329313784837723, -1.9152568578720093, -0.6421753168106079, 1.3438509702682495, -1.2922308444976807, 0.7662441730499268, 0.6454001665115356, 0.353316068649292, -2.6474881172180176, -1.4575366973876953, -0.9712379574775696, 0.25403109192848206, -0.17905889451503754, 1.199284553527832, -0.42921745777130127, 1.010284185409546, 0.6110401153564453, 1.2208385467529297, -0.6076440811157227, -1.7376023530960083, -0.12535223364830017, -1.3658145666122437, 1.1117461919784546, -0.6227965950965881, -0.7891808748245239, -0.1678244024515152, 1.6433145999908447, 2.0070886611938477, -1.2531018257141113, 1.118869662284851, 1.7732776403427124, -2.071660280227661, -0.4125255346298218, -0.9769554734230042, -0.03363388776779175, 1.8594977855682373, 2.6221468448638916, 0.36905255913734436, 0.3802972435951233, 0.19898030161857605, -0.23609091341495514, 0.30340856313705444, -0.45007672905921936, 0.47390419244766235, 0.6503364443778992, 1.1662380695343018, 0.01693599671125412, 0.5325868129730225, -0.6035351157188416, -0.1742597371339798, 0.6092063188552856, -0.8032152652740479, -1.1209008693695068, 0.19564072787761688, -0.7815181016921997, -1.7898790836334229, -0.26157355308532715, -0.44025033712387085, 2.1848294734954834, -0.4800971448421478, -1.2871731519699097, 0.7388824224472046, 0.03389474004507065, -0.3122936189174652, -0.2541753649711609, -1.205536127090454, -0.9542103409767151, 0.061276569962501526, 0.0852610319852829, 0.7481252551078796, -0.16356196999549866, -0.9085567593574524, 0.3129958212375641, 0.8050476908683777, -1.1133602857589722, 0.4981626570224762, -1.1999552249908447, 0.12711311876773834, 0.4403660297393799, 0.6377718448638916, 0.15978877246379852, 1.7697970867156982, 0.6268178820610046, -1.8736529350280762, 2.3259060382843018, -0.9203909635543823, 0.6661149263381958, -0.44026491045951843, -2.3179564476013184, 1.294582724571228, 0.22267311811447144, -0.8483412265777588, 1.6489418745040894, 1.6005686521530151, -0.07858924567699432, 0.4310458302497864, 0.3683530390262604, 0.7637977004051208, 1.1792222261428833, -0.4137862026691437, 0.5184086561203003, -0.7015367150306702, -0.4323408901691437, 0.1414770483970642, 0.07110362499952316, 0.5633530616760254, -0.5786358118057251, -1.083811640739441, -0.3889262080192566, 0.8126106858253479, 1.4981187582015991, 0.043896082788705826, 1.4443233013153076, 0.23202891647815704, 0.5064982771873474, -1.2786966562271118, -0.03842746838927269, 1.9138009548187256, 0.3378446102142334, 0.12505607306957245, -0.7621515393257141, -1.1905603408813477, 0.7756073474884033, 0.455719918012619, 0.2503303289413452, -1.3610970973968506, 1.8018341064453125, -0.07434200495481491, -0.1566413789987564, -0.8708453178405762, -0.6410972476005554, -0.41456282138824463, -0.6902380585670471, -0.22995668649673462, -2.172283887863159, 0.08768323808908463, 1.0937845706939697, -0.11772056668996811, -0.29864323139190674, -0.9536206126213074, -0.09247277677059174, -1.0166544914245605, -0.007675690110772848, -0.518220841884613, 0.83954256772995, 0.058522701263427734, -1.6682480573654175, 2.129624843597412, -1.518147349357605, 0.1387282758951187, -1.1797568798065186, -0.5297411680221558, 0.9625157713890076, 0.2794382572174072, -0.5718191266059875, -2.7936289310455322, -0.7111541628837585, 0.5235220193862915, -1.71055006980896, 0.8384853601455688, -0.2698453664779663, 0.12306156754493713, 0.8757511377334595, 0.1513299196958542, 0.7393931746482849, 0.27310311794281006, 2.7312309741973877, 0.43200522661209106, -0.309181809425354, -0.09658124297857285, 1.5419251918792725, -0.10874485224485397, -0.41890469193458557, 1.4384384155273438, -0.7068426609039307, -1.2519514560699463, 3.0250487327575684, 1.3462589979171753, 0.8556069731712341, 0.3220294117927551, 0.44605663418769836, 1.5229592323303223, 1.2804899215698242, -0.11616043001413345, 1.37053644657135, -0.4809381365776062, -0.9903622269630432, -1.3641812801361084, 0.008205652236938477, -0.40586018562316895, -0.7110859751701355, -0.3495793640613556, 0.3797488212585449, 0.9993038773536682, 1.2751853466033936, 0.9594927430152893, 0.10350999981164932, 0.8290343880653381, 2.0921294689178467, 0.7953095436096191, 0.2792840898036957, 0.1864478439092636, 0.35471320152282715, 0.09063850343227386, 1.7422553300857544, -1.266001582145691, 0.38916081190109253, 0.34287506341934204, -1.4590638875961304, -1.4936561584472656, -0.22138537466526031, 0.22523505985736847, -0.07724537700414658, 0.9856938123703003, 1.2783366441726685, 0.28815189003944397, 0.869049608707428, -0.8097057938575745, -1.4298604726791382, 0.45901596546173096, 0.5309328436851501, -1.3614805936813354, 1.9562491178512573, 1.7684898376464844, -0.9857985377311707, -1.237075924873352, -2.301875114440918, -0.0010086018592119217, -0.8494256734848022, -1.6593921184539795, 0.3062906563282013, 1.182044506072998, 0.32602694630622864, -0.3894469738006592, 2.8543806076049805, 0.8243650794029236, 0.7983470559120178, 1.8890222311019897, 0.5934627652168274, 0.0696544423699379, -1.6034338474273682, -0.42982181906700134, 0.5761587619781494, 0.34436318278312683, -3.1016061305999756, -1.4587225914001465, -1.4318257570266724, -0.6071268916130066, -0.25973790884017944, -0.7190187573432922, -0.38583093881607056, 0.5233524441719055, -0.8211760520935059, -0.47086891531944275, 0.6016423106193542, -0.28251126408576965, 0.7692679762840271, -0.7668923139572144, -0.9494866728782654, 0.01691739819943905, 0.08027740567922592, 0.7448412775993347, 1.345484972000122, 0.12682189047336578, -2.4520719051361084, 0.4159761965274811, 1.9025356769561768, -0.7346699833869934, 0.04465712979435921, -1.5211198329925537, 0.3478375971317291, 0.7401772737503052, 1.4161994457244873, 0.6833979487419128, -0.13825181126594543, 0.9212993383407593, 0.5282443761825562, -0.00822838768362999, -1.4493322372436523, -0.605182409286499, -0.17924532294273376, 0.1995580792427063, -1.2461947202682495, -0.41459983587265015, 1.4558700323104858, 0.3316534161567688, -1.0001006126403809, -0.6919524073600769, -0.4719906747341156, -1.2894344329833984, 1.0762810707092285, -1.0667427778244019, -1.9893426895141602, 0.29731306433677673, 0.4344586431980133, 0.0033933203667402267, -1.0240145921707153, 0.22404730319976807, -0.7554785013198853, 1.3675811290740967, -0.3197358250617981, -0.9130924344062805, 1.9192092418670654, -1.6514869928359985, 2.1477253437042236, -0.6604135036468506, 0.11352583765983582, -0.22056588530540466, 0.7118127346038818, 0.3415871858596802, 1.5885895490646362, -0.3488781750202179, -0.4579193592071533, -1.232207179069519, -0.598077118396759, -0.2815468907356262, 0.05281926319003105, 0.42497748136520386, 0.4825834035873413, 0.48813387751579285, 1.0082393884658813, -0.5950038433074951, 0.3926331400871277, 0.8229668736457825, -0.886031985282898, 1.480103850364685, 0.8391514420509338, -0.20004983246326447, 0.9949536919593811, 0.7201864719390869, -0.13413065671920776, -1.4067999124526978, -2.3609628677368164, -0.2904941141605377, -0.13345853984355927, -0.15693345665931702, 1.138344645500183, -0.2505214214324951, 1.6704555749893188, -0.545271098613739, -2.15816330909729, -1.6607974767684937, -0.6637442111968994, 0.3657907545566559, -0.39920157194137573, 0.49674081802368164, -2.369168758392334, -0.5614708065986633, -0.5949130654335022, 1.2687277793884277, 1.2904434204101562, -1.1755682229995728, -0.0783226415514946, -0.9705760478973389, 1.4723697900772095, 1.4108561277389526, -1.3143675327301025, -1.31621515750885, -1.2524477243423462, -1.5844100713729858, -2.5446670055389404, 1.3719074726104736, -0.5379465222358704, 0.7378400564193726, -0.8505350351333618, 0.03610055148601532, 1.3406710624694824, 0.9199972748756409, -0.3787555396556854, -1.5597758293151855, -0.80095374584198, -0.7111088037490845, -0.3866667151451111, 0.9578314423561096, -0.8225309252738953, -2.3908050060272217, 0.322247713804245, 1.875388741493225, 1.1042989492416382, -0.522375762462616, -0.7401803731918335, 0.16235657036304474, -0.23699739575386047, 0.5099347233772278, 1.670624852180481, 1.5921050310134888, -0.41619211435317993, 1.861944556236267, -1.077892780303955, 0.8848567605018616, -0.8342104554176331, 1.0300743579864502, -0.8680981993675232, -0.5701602697372437, 0.323322057723999, 1.1284750699996948, -1.2123126983642578, 2.602391004562378, -0.09572362899780273, -0.08114802837371826, 1.2586978673934937, 0.8691263794898987, -0.9609367251396179, 0.05182275176048279, -0.3284812867641449, -2.247206687927246, -0.4478967487812042, 0.4234687089920044, -0.38745859265327454, -0.22963790595531464, -0.40709349513053894, 0.8702965974807739, -1.0552809238433838, -1.3284008502960205, 0.7060741782188416, 0.35730114579200745, 0.5892837643623352, 0.9187757968902588, 0.6662830114364624, 0.24650610983371735, 0.1328691840171814, 0.12191437929868698, 0.47808775305747986, 0.276134192943573, -0.5895729064941406, 0.569182813167572, -0.7911049723625183, -0.19896702468395233, -1.3615713119506836, -0.5193603038787842, 0.07648162543773651, 0.34005022048950195, 1.4557304382324219, -0.3461014926433563, -0.2633814215660095, -0.447700172662735, -0.7288169264793396, -0.16066236793994904, -0.32063713669776917, -0.6307737827301025, -0.7887667417526245, 1.3061572313308716, -0.9275762438774109, -0.26273950934410095, 0.9314952492713928, -0.4593467116355896, -0.9419456720352173, -0.7089186310768127, 2.1860759258270264, -0.6493165493011475, 0.45214059948921204, 0.8520749807357788, -1.6946725845336914, 1.1805996894836426, -2.8929238319396973, -0.387578547000885, -0.7124031782150269, -1.6171332597732544, -0.3589920103549957, 0.051366694271564484, 0.6950237154960632, 1.835181474685669, -1.9180361032485962, -1.3923954963684082, 0.540465772151947, 0.4350730776786804, -2.2717032432556152, -0.13386189937591553, -0.058557309210300446, 0.12574495375156403, -0.5525767803192139, 0.07448001205921173, -0.14928652346134186, -0.5522536635398865, -0.09342005103826523, -1.0284309387207031, 0.40444278717041016, 2.1425962448120117, -0.5153722763061523, 1.0827196836471558, 1.2498642206192017, 0.9821352958679199, 0.22690093517303467, 0.49279212951660156, -0.5128253102302551, 0.3006223440170288, 0.077346570789814, 0.6477669477462769, -0.43242430686950684, 1.1740481853485107, 0.7011352777481079, 0.6674330234527588, -0.8035953640937805, -1.3776048421859741, -0.4410470724105835, 0.1417587250471115, 1.1084681749343872, 0.5544233322143555, 1.5817502737045288]);
    const bufferA = this.initBuffer(["storage", "copy_from", "copy_to"], 32, 16);
    this.device.queue.writeBuffer(bufferA, 0, A);
    const { resultBuffer, passes } = SoftmaxBlock.newInstance(
      32,
      16,
      bufferA,
    );
    await runComputePasses(this.device, passes);
    console.log(formatAsMatrix(
      (await serializeBuffer(this.device, resultBuffer)).float32ArrayBuffer,
      32, 16
    ));
    const bufferD = this.initBuffer(['storage', 'copy_from', 'copy_to'], 32, 16);
    this.device.queue.writeBuffer(bufferD, 0, gradRes);
    const { dInputBuffer, passes: passes2 } = SoftmaxBackwards.newInstance(
      bufferD, 32, 16, resultBuffer
    );
    await runComputePasses(this.device, passes2);
    console.log(formatAsMatrix(
      (await serializeBuffer(this.device, dInputBuffer)).float32ArrayBuffer,
      32, 16
    ));
  }

  async test() {
    initializeOperations(this.device);
    const pA = ([-1.1258398294448853, -1.152360200881958, -0.2505785822868347, -0.4338788390159607, 0.8487103581428528, 0.6920092105865479, -0.31601276993751526, -2.1152195930480957, 0.32227492332458496, -1.2633347511291504, 0.34998318552970886, 0.30813390016555786, 0.11984150856733322, 1.237657904624939, 1.1167771816253662, -0.2472776472568512, -1.3526537418365479, -1.695931315422058, 0.5666505098342896, 0.7935084104537964, 0.5988394618034363, -1.5550950765609741, -0.34136030077934265, 1.85300612449646, 0.7501894235610962, -0.5854971408843994, -0.1733970195055008, 0.18347792327404022, 1.3893661499023438, 1.5863343477249146, 0.946298360824585, -0.8436768054962158, -0.6135830879211426, 0.03159274160861969, -0.4926770329475403, 0.2484147548675537, 0.4396958649158478, 0.11241118609905243, 0.6407923698425293, 0.441156268119812, -0.10230965167284012, 0.7924439907073975, -0.2896675765514374, 0.05250748619437218, 0.5228604078292847, 2.3022053241729736, -1.4688938856124878, -1.586688756942749, -0.6730899214744568, 0.8728312253952026, 1.055357575416565, 0.1778441220521927, -0.23033547401428223, -0.3917543888092041, 0.5432946681976318, -0.39515751600265503, -0.44621720910072327, 0.7440207004547119, 1.5209795236587524, 3.4105026721954346, -1.5311843156814575, -1.2341350317001343, 1.8197252750396729, -0.5515286922454834, -0.5692480206489563, 0.9199712872505188, 1.1108161211013794, 1.2898738384246826, -1.4781743288040161, 2.567232847213745, -0.4731197953224182, 0.3355507254600525, -1.6293259859085083, -0.54974365234375, -0.47983425855636597, -0.49968215823173523, -1.066980004310608, 1.114939570426941, -0.14067143201828003, 0.8057536482810974, -0.09334822744131088, 0.6870502233505249, -0.8383153676986694, 0.0008918217499740422, 0.8418940901756287, -0.4000345468521118, 1.0394619703292847, 0.3581531047821045, -0.24600094556808472, 2.302516460418701, -1.881689190864563, -0.049727022647857666, -1.0449786186218262, -0.9565005302429199, 0.03353185951709747, 0.7100865840911865, 1.6458669900894165, -1.3601689338684082, 0.34456542134284973, 0.5198677182197571, -2.6133224964141846, -1.6964746713638306, -0.2282416820526123, 0.2799549996852875, 0.24692639708518982, 0.07688700407743454, 0.33800581097602844, 0.45440176129341125, 0.45694077014923096, -0.8653709888458252, 0.7813079357147217, -0.926789402961731, -0.21883368492126465, -2.435065269470215, -0.07291452586650848, -0.03398660197854042, 0.9625182747840881, 0.34916824102401733, -0.9214619994163513, -0.05619478225708008, -0.6226984858512878, -0.4637221693992615, 1.921782374382019, -0.4025455117225647, 0.12390247732400894, 1.1647834777832031, 0.9233735203742981, 1.3872952461242676, -0.8833757638931274, -0.41891345381736755, -0.8048266768455505, 0.5656090974807739, 0.6103646159172058, 0.4668835401535034, 1.9506571292877197, -1.0630985498428345, -0.07732575386762619, 0.1163986548781395, -0.5939906239509583, -1.2439285516738892, -0.10209263116121292, -1.033548355102539, -0.3126388490200043, 0.24578548967838287, -0.25964149832725525, 0.11833705008029938, 0.24395832419395447, 1.1646006107330322, 0.2885758578777313, 0.38659775257110596, -0.20106391608715057, -0.11792698502540588, 0.19219909608364105, -0.7721568942070007, -1.9003455638885498, 0.13067743182182312, -0.7042941451072693, 0.3147209584712982, 0.15739288926124573, 0.3853627145290375, 0.9671456813812256, -0.9910828471183777, 0.3016054630279541, -0.10731688141822815, 0.9984563589096069, -0.49871477484703064, 0.7611109018325806, 0.6183008551597595, 0.31404855847358704, 0.21333301067352295, -0.1200508326292038, 0.36045974493026733, -0.3140355050563812, -1.0787080526351929, 0.24081051349639893, -1.3962273597717285, -0.06614456325769424, -0.3583550751209259, -1.5615617036819458, -0.3546432852745056, 1.0810725688934326, 0.13147805631160736, 1.5735375881195068, 0.7814293503761292, -1.0786579847335815, -0.720909833908081, 1.470792531967163, 0.2756350040435791, 0.6667810678482056, -0.9943896532058716, -1.1893646717071533, -1.1959497928619385, -0.5596300959587097, 0.5334718227386475, 0.40688663721084595, 0.3945865333080292, 0.1715109646320343, 0.876044750213623, -0.28708741068840027, 1.0216400623321533, -0.07439491152763367, -1.0922236442565918, 0.3920263946056366, 0.5945261120796204, 0.6622740626335144, -1.2063024044036865, 0.6074396967887878, -0.547156810760498, 1.1710891723632812, 0.09749628603458405, 0.9633742570877075, 0.8403232097625732, -1.2536547183990479, 0.9868361353874207, -0.494655042886734, -1.2830430269241333, 0.9552218317985535, 1.2835776805877686, -0.6658616065979004, 0.5651336908340454, 0.2877037227153778, -0.03337525576353073, -1.0618884563446045, -0.11442617326974869, -0.3433358371257782, 1.5712648630142212, 0.19161488115787506, 0.3799419403076172, -0.1447574645280838, 0.6376178860664368, -0.28129008412361145, -1.3298759460449219, -0.14201070368289948, -0.5341457724571228, -0.5233790278434753, 0.8615042567253113, -0.8869631290435791, 0.8387746810913086, 1.152895212173462, -1.7610975503921509, -1.4777427911758423, -1.7556711435317993, 0.07616619765758514, -1.0786035060882568, 1.4403417110443115, -0.11059419065713882, 0.576860249042511, -0.16917410492897034, -0.06402487307786942, 1.0384255647659302, 0.9068235158920288, -0.4755135476589203, -0.8707441687583923, 0.1447429060935974, 1.902856469154358, 0.39039579033851624, -0.039372581988573074, -0.8014718294143677, -0.49554431438446045, -0.36151406168937683, 0.5851132273674011, -1.1560068130493164, -0.14336487650871277, -0.19474057853221893, -0.08556339144706726, 1.3945202827453613, 0.5969001054763794, -0.4828483462333679, -0.3660986125469208, -1.3270522356033325, 1.695279598236084, 2.0654995441436768, -0.2339576780796051, 0.7073183059692383, 0.5800480842590332, 0.2683020830154419, -2.05893874168396, 0.5340204834938049, -0.5353949069976807, -0.8636655807495117, -0.023494349792599678, 1.171669840812683, 0.3986872136592865, -0.19871552288532257, -1.1559408903121948, -0.316669762134552, 0.9402980804443359, -1.1469541788101196, 0.5588033199310303, 0.7917585372924805, -0.18467576801776886, -0.7317724823951721, -0.0806519091129303, -0.9800607562065125, 0.06049158424139023, -0.48895469307899475, -0.8137312531471252, 0.8199948072433472, -0.633173406124115, 1.2947547435760498, 1.4628292322158813, -0.6204336881637573, 0.9883891940116882, -0.4321780204772949, -0.6232194304466248, -0.21624533832073212, -0.4886760413646698, 0.7869552969932556, 0.10759072750806808, -1.0714776515960693, -0.11664776504039764, -1.0169707536697388, -1.197978138923645, 0.47843775153160095, -1.2295241355895996, -1.3700367212295532, 1.5435417890548706, -0.0332069993019104, -0.4186263680458069, -0.25559648871421814, -0.1292310208082199, -0.05459475517272949, 0.4083467423915863, 1.1263659000396729, 1.9350574016571045, 1.0076850652694702, 1.0046416521072388, -0.43351972103118896, -1.2425976991653442, 1.2845500707626343, 0.24377228319644928, 0.5303685665130615, -0.014530729502439499, -2.235718011856079, 1.4660301208496094, -1.2190581560134888, 0.6442300081253052, 3.9300038814544678, -0.1244242787361145, 0.2953416705131531, 0.38265419006347656, -0.549721360206604, -0.9940357804298401, 1.345936894416809, 1.9456682205200195, -1.2903639078140259, -2.3494760990142822, -2.068861961364746, 0.9094210863113403, -0.6946200728416443, 1.9594571590423584, -1.1038278341293335, 0.5411418080329895, 1.5389583110809326, 1.08604896068573, 1.246405005455017, 0.11507505923509598, 1.619307279586792, 0.46369341015815735, 1.3007354736328125, 0.873230516910553, 0.0651267021894455, 0.7732410430908203, -0.970138430595398, -0.8876768946647644, -0.3183170557022095, -0.3344038724899292, 0.4542836546897888, 0.49895304441452026, 0.8779974579811096, 0.38944435119628906, 1.462517499923706, 0.4795060157775879, -0.5333998799324036, -0.03465134650468826, 0.6572969555854797, -0.3112243115901947, -0.5620035529136658, -0.4834926128387451, -1.2721126079559326, -0.17401844263076782, 0.5541168451309204, -0.18165524303913116, -0.23447339236736298, 0.29420149326324463, 0.7973228693008423, 1.2642154693603516, 0.935491681098938, 0.5454632043838501, -1.5373886823654175, 0.31243863701820374, 0.7400596737861633, 1.4502208232879639, 4.1014933586120605, 1.1182256937026978, -1.566847801208496, -0.6989754438400269, 0.574386715888977, 1.2380632162094116, -0.6405401229858398, -0.7644728422164917, 0.2408405840396881, 0.16642573475837708, -2.23181414604187, 1.3892109394073486, -0.5023325681686401, 1.6796929836273193, -1.0239529609680176, 1.6859242916107178, -1.2176920175552368, 0.7649633288383484, 1.1971186399459839, -0.712786853313446, -0.06557541340589523, 2.204970359802246, 1.7851710319519043, -0.011840680614113808, 0.9796671271324158, -1.0660514831542969, 1.7719635963439941, -0.27926018834114075, -0.2769016921520233, 0.7489254474639893, -0.6434551477432251, -0.9517593383789062, 0.2715212404727936, 0.6715788841247559, 1.8499820232391357, 1.1909500360488892, -0.5898563861846924, 0.9646937847137451, -1.5093512535095215, 2.255730628967285, 1.2287956476211548, -0.4854608178138733, 0.4535709619522095, 1.3514447212219238, 0.4339268207550049, -0.5132520198822021, -0.18602684140205383, 0.27565816044807434, 0.10969064384698868, 0.35942408442497253, -0.7537387013435364, 0.22939957678318024, -0.2544441819190979, 1.5800135135650635, -0.24436436593532562, -1.1991091966629028, -0.025686170905828476, 1.802375078201294, -1.0596528053283691, 3.4028263092041016, -0.5686699748039246, -0.4754890501499176, 1.7431631088256836, -0.2044074535369873, -0.31641438603401184, 1.2937437295913696, 1.3452626466751099, 0.19394375383853912, 1.5716748237609863, -0.38273516297340393, 1.3950834274291992, 0.34274551272392273, -1.604492425918579, -0.5873053073883057, 0.6003884673118591, 0.4378035068511963, -0.0964546874165535, 0.33026883006095886, -0.1875188946723938, -1.4270646572113037, 0.59255450963974, -1.1581913232803345, 0.035760924220085144, 0.21601258218288422, -0.9160799980163574, 1.5598512887954712, -3.153724431991577, -0.5611025094985962, -0.4302971065044403, -0.33322516083717346, -1.54639732837677, -0.01471690647304058, 1.2251118421554565, 1.5936384201049805, -1.6314972639083862, -0.05687696114182472, 0.6296589970588684, 0.2711690366268158, -0.6859827041625977, -1.0917942523956299, 1.6796669960021973, -0.8808245062828064, 0.5800328254699707, 0.36422988772392273, 0.08813405781984329, -1.3069120645523071, -0.7063697576522827, -0.16421614587306976, -0.9714681506156921, -1.03084397315979, 0.6472792625427246, -0.19061511754989624, 0.7166509628295898, -2.0001885890960693, -2.4096577167510986, 0.21942289173603058, -1.6988605260849, 1.3094241619110107, -1.6612948179244995]);
    const A = new Float32Array(pA.concat(...pA).concat(...pA))
    const pB = ([-0.5460728406906128, -0.630177915096283, -0.6346500515937805, 0.9746645092964172, 0.2098425179719925, 0.02988985739648342, 1.7092351913452148, -0.7257586121559143, -0.7735371589660645, 0.5962143540382385, -1.250418782234192, 1.145593285560608, 0.7393418550491333, 1.2531980276107788, -0.4445230960845947, 0.8184519410133362, -0.8180161118507385, 0.36031603813171387, -1.6146106719970703, -2.4733614921569824, 0.036155786365270615, -0.3422180116176605, -0.3816922605037689, -0.05687851086258888, 0.8436174988746643, 0.6828738451004028, 3.3943958282470703, -1.6687780618667603, 0.5108585953712463, -0.28598499298095703, 0.33505186438560486, 1.1719372272491455, 1.2955021858215332, 0.8908616304397583, -0.4898480176925659, -1.1726573705673218, -0.6870454549789429, -2.3348581790924072, 0.09404075890779495, -0.20208217203617096, -0.059524137526750565, 2.0118417739868164, -0.3367916941642761, 0.3259803354740143, 0.5352019667625427, 1.9732894897460938, -0.20751099288463593, -0.0305744931101799, 0.1267281472682953, 0.005546563304960728, 0.7943360209465027, 0.4071536064147949, -0.36090219020843506, 1.3102548122406006, -0.9650526642799377, 0.8806111812591553, -0.10247437655925751, -0.6770100593566895, -0.4106607735157013, -1.6185554265975952, 0.5079081058502197, 2.3229541778564453, 0.2297753393650055, -0.5296528935432434, -0.8733066916465759, 0.004261419177055359, -1.2578867673873901, -1.0844677686691284, 0.7529793977737427, 0.32364773750305176, -0.27501001954078674, 1.3056118488311768, 0.2117518186569214, 0.27196231484413147, -0.9268431663513184, -2.732999801635742, -0.564173698425293, -0.2739996314048767, 0.13978058099746704, 0.5085619688034058, 0.27709972858428955, -0.9812496304512024, 0.8888465762138367, 1.5689952373504639, -0.08185258507728577, -0.3494001030921936, 0.20242652297019958, -0.2883831262588501, 0.14829760789871216, 2.418731212615967, 1.3278692960739136, -0.26386022567749023, 0.3644659221172333, 2.5440163612365723, -2.689467191696167, 2.4426093101501465, 0.01037450972944498, -0.9964888095855713, 0.9785000681877136, -0.44143953919410706, -0.26104333996772766, 0.7979768514633179, -1.1071447134017944, 2.330575466156006, -0.7650176286697388, -0.4749671518802643, -0.4952569901943207, -0.19836050271987915, 2.2148795127868652, -0.13668861985206604, -1.0181605815887451, 0.17840538918972015, -0.5135857462882996, -0.5644288659095764, -0.9183681011199951, -0.749563455581665, -0.09493295848369598, 1.1008661985397339, 1.3104606866836548, -0.2928459644317627, -0.7883442044258118, -0.1695212572813034, -2.1748616695404053, 0.7202475070953369, 0.2854461669921875, 0.22903010249137878, 1.2832691669464111, -1.3792049884796143, 0.5407565832138062, -0.9478074312210083, 0.20214395225048065, -0.35074812173843384, 0.5450052618980408, 1.5412107706069946, 0.6002392172813416, -0.3380149006843567, 0.40466293692588806, 0.8931286334991455, -1.45409095287323, 1.1874935626983643, -0.2995234429836273, 2.296318292617798, 0.3305456340312958, 2.1747679710388184, -1.245952844619751, 2.4966132640838623, -0.706882894039154, 1.1504485607147217, -0.5497670769691467, 0.8155179619789124, 2.000545024871826, -0.4393474757671356, -0.4318980276584625, -0.47292864322662354, 0.32564112544059753, -0.9736213088035583, 0.7978925108909607, -0.5733004212379456, 0.20604948699474335, 0.21373623609542847, 0.9543960690498352, 0.7330628037452698, -1.4552098512649536, -2.04579758644104, -0.17707549035549164, 0.6240473985671997, -1.6889100074768066, 0.8175789713859558, -0.3895973563194275, 1.2776107788085938, 0.5146814584732056, 0.34851741790771484, -1.0378724336624146, -0.9586871862411499, 0.9181745052337646, 0.4887973666191101, 0.9839283227920532, 0.9961360096931458, -1.1657720804214478, -0.5853711366653442, -0.18619273602962494, -1.237442135810852, 1.1838980913162231, -0.19545401632785797, -1.3366087675094604, 1.0511319637298584, -1.026934027671814, -0.28749531507492065, 1.8479732275009155, -2.394956111907959, 0.4076978266239166, -0.16988591849803925, 0.5841776132583618, 1.0504306554794312, 1.285578966140747, -1.6165199279785156, -0.7689623832702637, -1.2204617261886597, 0.5731281042098999, 0.6991968154907227, 0.2510557174682617, 0.27845174074172974, -0.09463465958833694, 1.610432744026184, -0.12166494131088257, -1.3941329717636108, -0.9047872424125671, -0.34669890999794006, 0.7049364447593689, 0.0305445808917284, -0.8542383313179016, 0.538818359375, -0.5264900922775269, -1.332040786743164, 1.545137882232666, 0.40862593054771423, -2.054642677307129, 0.5259053111076355, 0.5994642376899719, -0.40781933069229126, 0.4530232548713684, -0.39178934693336487, 2.1403496265411377, -0.20620287954807281, -0.09838856011629105, 0.48546716570854187, 0.7075708508491516, 0.04306506738066673, -0.4394379258155823, -0.6761147379875183, 1.7388859987258911, -0.9422927498817444, -1.0646462440490723, -0.2297092229127884, -1.256403923034668, 0.5570129752159119, 0.15761485695838928, 1.027068853378296, 1.2292814254760742, -0.0012231325963512063, -1.8095455169677734, 0.692562997341156, 1.1982135772705078, 1.3166900873184204, -0.3483226001262665, 1.2075036764144897, -0.6478288173675537, 0.5265247821807861, -0.9507911801338196, -2.145578622817993, 0.38271036744117737, -0.40974682569503784, -0.7386032938957214, 1.655275583267212, 0.5203718543052673, -0.23261602222919464, 0.4974166452884674, 0.2685222029685974, 1.4768786430358887, 0.35479506850242615, 1.6247085332870483, 0.593422532081604, -1.7254115343093872, -0.6220155954360962, -0.006249400787055492, 0.895197331905365, -0.007699869107455015, 0.04517161846160889, 0.5793444514274597, -1.5825188159942627, -0.5877922177314758, -0.11398255825042725, 0.7014375329017639, -0.5556063055992126, -0.38169318437576294, -0.04424409940838814, 0.5195826292037964, 0.7600806951522827, -1.0292669534683228, -1.2504048347473145, -0.11369367688894272, -0.6140205264091492, 0.8530076742172241, 0.15509827435016632, 1.1059221029281616, 1.0944527387619019, 0.5702730417251587, 0.7763661742210388, -1.3568336963653564, -0.7653952240943909, -0.2090262472629547, -0.26518332958221436, -0.7778550982475281, 1.4072339534759521, -2.9776642322540283, -0.14418236911296844, -0.8499265909194946, -0.29783177375793457, 2.1292507648468018, 0.5027324557304382, -0.8870898485183716, 1.997416377067566, -1.6983762979507446, -0.6720045804977417, -0.7617099285125732, -1.9145387411117554, -1.1217594146728516, -0.6035577654838562, -2.4189693927764893, -0.9170435070991516, 0.7018370032310486, -0.7560974955558777, 0.3763469457626343, 1.1279659271240234, 0.02189653366804123, 1.4937312602996826, -2.1925909519195557, -0.1090661808848381, -1.5689644813537598, 0.5581148862838745, -1.0654696226119995, 0.40257763862609863, -0.4052416682243347, -0.5833661556243896, -0.258523166179657, -0.6055594682693481, -0.18824270367622375, 0.9596066474914551, 0.6800092458724976, -0.7692208290100098, -0.07288359105587006, -1.4433460235595703, -1.2622580528259277, 0.5881808400154114, -0.876555860042572, -1.0352659225463867, -1.2152215242385864, -0.1696276217699051, 0.5822805166244507, 1.889708161354065, 0.27074846625328064, 0.730144739151001, -1.2801202535629272, -0.35817936062812805, -0.06125941127538681, 1.3390552997589111, -1.0588995218276978, 0.4159274995326996, -0.7801193594932556, -0.02442266047000885, -0.9445182085037231, 0.07608630508184433, 3.114617347717285, -0.8107677102088928, -1.105208158493042, -0.7233065366744995, 0.7882310748100281, 1.3645267486572266, 0.3736513555049896, -1.0004340410232544, 0.0015595387667417526, -0.13155868649482727, 0.10314879566431046, -0.369794100522995, -0.7115960717201233, -0.12029961496591568, 1.2383081912994385, 0.19852314889431, -0.7985085844993591, 0.27604880928993225, 0.09818912297487259, 2.4820919036865234, -1.200674057006836, 1.020957589149475, -0.38697153329849243, -0.05688869580626488, -0.27344438433647156, -0.061053838580846786, 0.5320888161659241, -0.8971832990646362, -1.3016557693481445, -1.372825264930725, 1.6909323930740356, -0.46219968795776367, 0.2035849541425705, -1.032828688621521, 1.1304807662963867, -0.5703012943267822, -2.100937843322754, 0.38992249965667725, 0.0873434990644455, -0.8573606014251709, -0.27461615204811096, 0.44321209192276, 0.9591097235679626, 1.3319531679153442, 0.7733116745948792, 3.056696891784668, -2.4878029823303223, -1.6961745023727417, -0.050967391580343246, 1.392568826675415, 2.2975423336029053, 0.5272547602653503, 0.15138177573680878, 0.5188243985176086, -0.5896209478378296, 1.0006802082061768, -1.3174312114715576, 0.005600185599178076, 0.5678830742835999, -1.419203281402588, -0.32263442873954773, 0.16012698411941528, -1.4835623502731323, 0.8233740925788879, 1.016634464263916, 1.286753535270691, 2.0820066928863525, 0.11674309521913528, -0.1482292115688324, 0.8930535912513733, -0.7618316411972046, -1.5387883186340332, -1.7845648527145386, 0.6954885721206665, -2.456195116043091, 0.020697416737675667, -0.12531797587871552, 1.6862393617630005, 2.2762389183044434, -0.9567826390266418, 0.7144834399223328, 0.844455361366272, 0.06993348151445389, 0.006232986226677895, -1.4055302143096924, 0.11945226043462753, -0.33122149109840393, -0.840735137462616, 0.4611181616783142, 0.6159672141075134, -0.21903789043426514, -0.2406650185585022, -0.6250956654548645, 0.8161492347717285, -0.571069061756134, -0.1194528117775917, -0.4273608326911926, 0.8143250346183777, -1.4120763540267944, -0.09977369755506516, 0.07493619620800018, 0.5903218388557434, 0.3985564112663269, -0.6358906030654907, -0.3538537919521332, -1.098997712135315, -2.054348945617676, -0.08066242933273315, -0.000535374041646719, -1.1454399824142456, 2.0060811042785645, -0.17704834043979645, -1.51466965675354, 1.0749554634094238, 0.05076493322849274, -1.655743956565857, -0.46148812770843506, 0.038077034056186676, -1.3240481615066528, 0.5548263192176819, -0.7227262854576111, 2.12211537361145, 0.049964603036642075, 0.4521275460720062, 0.8145699501037598, -0.139418825507164, -0.3676704168319702, -0.4573548436164856, -1.294549822807312, 0.7011868953704834, -1.9097658395767212, 1.0538135766983032, -1.047306776046753, 0.22511537373065948, 0.679832935333252, 0.5288675427436829, 0.35422441363334656, 0.5805200934410095, 0.03690203279256821, 0.6230257749557495, 0.8397579193115234, -0.8819846510887146, 0.965045154094696, 0.8688325881958008, 0.887984573841095, -0.9247846603393555, 0.3240966498851776, -0.8189603090286255, 0.3624175786972046, 0.22211487591266632, 0.27903175354003906, -0.3151742219924927, -0.40103623270988464, 0.21571780741214752, 1.2853788137435913, -1.0385949611663818, -1.6339619159698486, 0.5374267101287842, 1.0826129913330078, -1.7104681730270386]);
    const B = new Float32Array(pB.concat(...pB).concat(...pB))
    const bufferA = this.initBuffer(["storage", "copy_from", "copy_to"], 3*16, 32);
    const bufferB = this.initBuffer(["storage", "copy_from", "copy_to"], 3*32, 16);
    this.device.queue.writeBuffer(bufferA, 0, A);
    this.device.queue.writeBuffer(bufferB, 0, B);

    console.log(formatAsMatrix(A, 3, 512))

    const { resultBuffer, passes } = FastMatMulBatchedBlock.newInstance(
      16,
      16,
      32,
      bufferA,
      bufferB,
      undefined,
      //this.initBuffer(["storage", "copy_from", "copy_to"], 3*16),
      3,
    );
    await runComputePasses(this.device, passes);
    console.log((await serializeBuffer(this.device, resultBuffer)).float32ArrayBuffer)

    console.log(formatAsMatrix(
      (await serializeBuffer(this.device, resultBuffer)).float32ArrayBuffer,
      48, 16
    ));

    const C = new Float32Array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    const bufferC = this.initBuffer(['storage', 'copy_from', 'copy_to'], 16, 16);
    this.device.queue.writeBuffer(bufferC, 0, C);
    const { dInputBuffer, dWeightsBuffer, dBiasBuffer, passes: passes2 } = FastMatMulBackwards.newInstance(
      bufferC, 16, 16, 32, bufferA, bufferB,
    );
    await runComputePasses(this.device, passes2);
    console.log(formatAsMatrix(
      (await serializeBuffer(this.device, dInputBuffer)).float32ArrayBuffer,
      16, 32
    ));
    console.log(formatAsMatrix(
      (await serializeBuffer(this.device, dWeightsBuffer)).float32ArrayBuffer,
      32, 16
    ));
    console.log((await serializeBuffer(this.device, dBiasBuffer)));
  }

  async testLoss() {
    initializeOperations(this.device);
    const [
      aBuffer,
      bBuffer
    ] = await Promise.all([
      this.fetchAndInitTensor(`weights/test/a.bin`, [640, 7680], ["storage", "copy_from"]),
      this.fetchAndInitTensor(`weights/test/b.bin`, [640], ["storage"], Uint32Array),
    ]);

    const { resultBuffer, caches, passes } = CrossEntropyLoss.newInstance(
      640,
      7680,
      aBuffer,
      bBuffer,
    );
    await runComputePasses(this.device, passes);
    console.log((await serializeBuffer(this.device, resultBuffer)).float32ArrayBuffer);

    const dLosses = new Float32Array(new Array(640).fill(1/640));
    const dLossesBuffer = this.initBuffer(['storage', 'copy_from', 'copy_to'], 640);
    this.device.queue.writeBuffer(dLossesBuffer, 0, dLosses);
    
    // const { dLogitsBuffer, passes: passes2 } = CrossEntropyBackwards.newInstance(
    //   dLossesBuffer,
    //   caches,
    //   640,
    //   7680,
    //   aBuffer,
    //   bBuffer,
    // );
    // await runComputePasses(this.device, passes2);
    // console.log(formatAsMatrix(
    //   (await serializeBuffer(this.device, caches.probsBuffer)).float32ArrayBuffer,
    //   640,
    //   7680,
    // ));
    // console.log(formatAsMatrix(
    //   (await serializeBuffer(this.device, dLogitsBuffer)).float32ArrayBuffer,
    //   640,
    //   7680,
    // ));
  }

  async testLayerNorm() {
    initializeOperations(this.device);
    const [ inputBuffer, dOutputBuffer ] = await Promise.all([
      this.fetchAndInitTensor(`weights/test/ln_in.bin`, [16, 128], ["storage", "copy_from"]),
      this.fetchAndInitTensor(`weights/test/ln_grads.bin`, [16, 128], ["storage", "copy_from"]),
    ]);

    const gammaArray = new Float32Array(new Array(16).fill(1));
    const gammaBuffer = this.initBuffer(['storage', 'copy_from', 'copy_to'], 16);
    this.device.queue.writeBuffer(gammaBuffer, 0, gammaArray);

    const betaArray = new Float32Array(new Array(16).fill(0));
    const betaBuffer = this.initBuffer(['storage', 'copy_from', 'copy_to'], 16);
    this.device.queue.writeBuffer(betaBuffer, 0, betaArray);

    const { resultBuffer, caches, passes } = LayerNormBlock.newInstance(
      16, 128,
      inputBuffer,
      gammaBuffer,
      betaBuffer,
    )
    await runComputePasses(this.device, passes);
    console.log(formatAsMatrix(
      (await serializeBuffer(this.device, resultBuffer)).float32ArrayBuffer,
      16,
      128,
    ));

    // backprop
    const {
      dInputBuffer,
      dBetaBuffer,
      dGammaBuffer,
      passes: passes2,
    } = LayerNormBackwards.newInstance(
      dOutputBuffer, caches,
      16,
      128,
      inputBuffer, gammaBuffer, betaBuffer
    );
    await runComputePasses(this.device, passes2);
    console.log(formatAsMatrix(
      (await serializeBuffer(this.device, dInputBuffer)).float32ArrayBuffer,
      16,
      128,
    ));
    console.log(formatAsMatrix(
      (await serializeBuffer(this.device, dGammaBuffer)).float32ArrayBuffer,
      16,
      128,
    ));
    console.log(formatAsMatrix(
      (await serializeBuffer(this.device, dBetaBuffer)).float32ArrayBuffer,
      16,
      128,
    ));
  }

  async testMisc() {
    //initializeOperations(this.device);

    const resBuffer = await window.model.testLayer();
  }
}

async function test() {
  const GPU = new TestGPT();
  await GPU.initialize();
  await GPU.testSoftmax();
}

/*


fast row add shader for reference
struct BMeta {
      M: u32,
      N: u32,
      ND4: u32,
    }

    @group(1) @binding(0) var<storage,read> array_matrix: array<vec4<f32>>;
    @group(1) @binding(1) var<storage,read> array_bias: array<vec4<f32>>;
    @group(0) @binding(0) var<uniform> bmeta: BMeta;
    @group(0) @binding(1) var<storage,read_write> array_output: array<vec4<f32>>;

    @compute @workgroup_size(8,8)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      var col: u32 = global_id.x;
      var row: u32 = global_id.y;
      var ND4: u32 = bmeta.ND4;
      var M: u32 = bmeta.M;
      
      if (row >= M || col >= ND4) {
        return;
      }

      array_output[row * ND4 + col] = array_matrix[row * ND4 + col] + array_bias[col];
    }

    class FastMatMulBlockClass extends Block {
  constructor() {
    super();
    this.name = "fastMatMul";
    this.pipelineCache = new Map();
  }

  getPipeline(rows) {
    const div4 = rows % 4 === 0;
    const pipelineCacheKey = div4 ? "fastMatMulNoCheck" : "fastMatMul";
    if (this.pipelineCache.has(pipelineCacheKey)) return this.pipelineCache.get(pipelineCacheKey);
    const kernel = div4 ? this.fastMatMulNoCheck : this.fastMatMul;
    const pipeline = this.initPipeline(kernel, [this.u_s_Layout, this.r_r_Layout], `${this.name}_Pipeline_${pipelineCacheKey}`);
    this.pipelineCache.set(pipelineCacheKey, pipeline);
    return pipeline;
  }

  newInstance(rows, cols, shared, bufA, bufB) {
    const pipeline = this.getPipeline(rows);
    const uniformBuffer = this.initBuffer(["uniform", "copy_to"], [4]);
    const resultBuffer = this.initBuffer(["storage", "copy_from"], [rows, cols]);
    const opBindGroup = this.initBindGroup(this.u_s_Layout, [uniformBuffer, resultBuffer], `${this.name}_OpG`);
    const inputBindGroup = this.initBindGroup(this.r_r_Layout, [bufA, bufB], `${this.name}_InputG`);
    const workgroups = { x: wgSize(cols, 64), y: wgSize(rows, 32) };
    this.device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([rows, cols, Math.ceil(cols / 4), Math.ceil(shared / 4)]));

    return {
      resultBuffer,
      passes: [
        {
          flag: "compute",
          pipeline,
          groups: [opBindGroup, inputBindGroup],
          workgroups,
        },
      ],
    };
  }

  fastMatMul = `
    struct CMeta {
      M: u32,
      N: u32,
      ND4: u32,
      KD4: u32,
    }

    @group(1) @binding(0) var<storage,read> array_a: array<vec4<f32>>;
    @group(1) @binding(1) var<storage,read> array_b: array<vec4<f32>>;

    @group(0) @binding(0) var<uniform> cmeta: CMeta;
    @group(0) @binding(1) var<storage,read_write> array_c: array<vec4<f32>>;

    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      var M: u32 = cmeta.M;
      var N: u32 = cmeta.N;
      var ND4: u32 = cmeta.ND4;
      var KD4: u32 = cmeta.KD4;
      var x: u32 = global_id.x;
      var y: u32 = global_id.y;

      if (x * 8 >= N || y * 4 >= M) {
        return;
      }

      var sum00: vec4<f32> = vec4<f32>();
      var sum01: vec4<f32> = vec4<f32>();
      var sum02: vec4<f32> = vec4<f32>();
      var sum03: vec4<f32> = vec4<f32>();
      var sum10: vec4<f32> = vec4<f32>();
      var sum11: vec4<f32> = vec4<f32>();
      var sum12: vec4<f32> = vec4<f32>();
      var sum13: vec4<f32> = vec4<f32>();

      for(var k: u32 = 0u; k < KD4; k = k + 1u) {
        var arow0: vec4<f32> = array_a[(y * 4u + 0u) * KD4 + k];
        var arow1: vec4<f32> = array_a[(y * 4u + 1u) * KD4 + k];
        var arow2: vec4<f32> = array_a[(y * 4u + 2u) * KD4 + k];
        var arow3: vec4<f32> = array_a[(y * 4u + 3u) * KD4 + k];
        var brow: vec4<f32>;

        brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.x) * brow + sum00;
        sum01 = vec4<f32>(arow1.x) * brow + sum01;
        sum02 = vec4<f32>(arow2.x) * brow + sum02;
        sum03 = vec4<f32>(arow3.x) * brow + sum03;

        brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.x) * brow + sum10;
        sum11 = vec4<f32>(arow1.x) * brow + sum11;
        sum12 = vec4<f32>(arow2.x) * brow + sum12;
        sum13 = vec4<f32>(arow3.x) * brow + sum13;

        brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.y) * brow + sum00;
        sum01 = vec4<f32>(arow1.y) * brow + sum01;
        sum02 = vec4<f32>(arow2.y) * brow + sum02;
        sum03 = vec4<f32>(arow3.y) * brow + sum03;

        brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.y) * brow + sum10;
        sum11 = vec4<f32>(arow1.y) * brow + sum11;
        sum12 = vec4<f32>(arow2.y) * brow + sum12;
        sum13 = vec4<f32>(arow3.y) * brow + sum13;

        brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.z) * brow + sum00;
        sum01 = vec4<f32>(arow1.z) * brow + sum01;
        sum02 = vec4<f32>(arow2.z) * brow + sum02;
        sum03 = vec4<f32>(arow3.z) * brow + sum03;

        brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.z) * brow + sum10;
        sum11 = vec4<f32>(arow1.z) * brow + sum11;
        sum12 = vec4<f32>(arow2.z) * brow + sum12;
        sum13 = vec4<f32>(arow3.z) * brow + sum13;

        brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.w) * brow + sum00;
        sum01 = vec4<f32>(arow1.w) * brow + sum01;
        sum02 = vec4<f32>(arow2.w) * brow + sum02;
        sum03 = vec4<f32>(arow3.w) * brow + sum03;

        brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.w) * brow + sum10;
        sum11 = vec4<f32>(arow1.w) * brow + sum11;
        sum12 = vec4<f32>(arow2.w) * brow + sum12;
        sum13 = vec4<f32>(arow3.w) * brow + sum13;
      }

      if (y * 4u + 0u < M) {
        array_c[x * 2u + 0u + (y * 4u + 0u) * ND4] = sum00;
        array_c[x * 2u + 1u + (y * 4u + 0u) * ND4] = sum10;
      }
      if (y * 4u + 1u < M) {
        array_c[x * 2u + 0u + (y * 4u + 1u) * ND4] = sum01;
        array_c[x * 2u + 1u + (y * 4u + 1u) * ND4] = sum11;
      }
      if (y * 4u + 2u < M) {
        array_c[x * 2u + 0u + (y * 4u + 2u) * ND4] = sum02;
        array_c[x * 2u + 1u + (y * 4u + 2u) * ND4] = sum12;
      }
      if (y * 4u + 3u < M) {
        array_c[x * 2u + 0u + (y * 4u + 3u) * ND4] = sum03;
        array_c[x * 2u + 1u + (y * 4u + 3u) * ND4] = sum13;
      }
    }
  `;

  fastMatMulNoCheck = `
    struct CMeta {
      M: u32,
      N: u32,
      ND4: u32,
      KD4: u32,
    }

    @group(1) @binding(0) var<storage,read> array_a: array<vec4<f32>>;
    @group(1) @binding(1) var<storage,read> array_b: array<vec4<f32>>;

    @group(0) @binding(0) var<uniform> cmeta: CMeta;
    @group(0) @binding(1) var<storage,read_write> array_c: array<vec4<f32>>;

    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      var M: u32 = cmeta.M;
      var N: u32 = cmeta.N;
      var ND4: u32 = cmeta.ND4;
      var KD4: u32 = cmeta.KD4;
      var x: u32 = global_id.x;
      var y: u32 = global_id.y;

      if (x * 8 >= N || y * 4 >= M) {
        return;
      }

      var sum00: vec4<f32> = vec4<f32>();
      var sum01: vec4<f32> = vec4<f32>();
      var sum02: vec4<f32> = vec4<f32>();
      var sum03: vec4<f32> = vec4<f32>();
      var sum10: vec4<f32> = vec4<f32>();
      var sum11: vec4<f32> = vec4<f32>();
      var sum12: vec4<f32> = vec4<f32>();
      var sum13: vec4<f32> = vec4<f32>();

      for(var k: u32 = 0u; k < KD4; k = k + 1u) {
        var arow0: vec4<f32> = array_a[(y * 4u + 0u) * KD4 + k];
        var arow1: vec4<f32> = array_a[(y * 4u + 1u) * KD4 + k];
        var arow2: vec4<f32> = array_a[(y * 4u + 2u) * KD4 + k];
        var arow3: vec4<f32> = array_a[(y * 4u + 3u) * KD4 + k];
        var brow: vec4<f32>;

        brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.x) * brow + sum00;
        sum01 = vec4<f32>(arow1.x) * brow + sum01;
        sum02 = vec4<f32>(arow2.x) * brow + sum02;
        sum03 = vec4<f32>(arow3.x) * brow + sum03;

        brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.x) * brow + sum10;
        sum11 = vec4<f32>(arow1.x) * brow + sum11;
        sum12 = vec4<f32>(arow2.x) * brow + sum12;
        sum13 = vec4<f32>(arow3.x) * brow + sum13;

        brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.y) * brow + sum00;
        sum01 = vec4<f32>(arow1.y) * brow + sum01;
        sum02 = vec4<f32>(arow2.y) * brow + sum02;
        sum03 = vec4<f32>(arow3.y) * brow + sum03;

        brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.y) * brow + sum10;
        sum11 = vec4<f32>(arow1.y) * brow + sum11;
        sum12 = vec4<f32>(arow2.y) * brow + sum12;
        sum13 = vec4<f32>(arow3.y) * brow + sum13;

        brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.z) * brow + sum00;
        sum01 = vec4<f32>(arow1.z) * brow + sum01;
        sum02 = vec4<f32>(arow2.z) * brow + sum02;
        sum03 = vec4<f32>(arow3.z) * brow + sum03;

        brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.z) * brow + sum10;
        sum11 = vec4<f32>(arow1.z) * brow + sum11;
        sum12 = vec4<f32>(arow2.z) * brow + sum12;
        sum13 = vec4<f32>(arow3.z) * brow + sum13;

        brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.w) * brow + sum00;
        sum01 = vec4<f32>(arow1.w) * brow + sum01;
        sum02 = vec4<f32>(arow2.w) * brow + sum02;
        sum03 = vec4<f32>(arow3.w) * brow + sum03;

        brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.w) * brow + sum10;
        sum11 = vec4<f32>(arow1.w) * brow + sum11;
        sum12 = vec4<f32>(arow2.w) * brow + sum12;
        sum13 = vec4<f32>(arow3.w) * brow + sum13;
      }

      array_c[x * 2u + 0u + (y * 4u + 0u) * ND4] = sum00;
      array_c[x * 2u + 1u + (y * 4u + 0u) * ND4] = sum10;
      array_c[x * 2u + 0u + (y * 4u + 1u) * ND4] = sum01;
      array_c[x * 2u + 1u + (y * 4u + 1u) * ND4] = sum11;
      array_c[x * 2u + 0u + (y * 4u + 2u) * ND4] = sum02;
      array_c[x * 2u + 1u + (y * 4u + 2u) * ND4] = sum12;
      array_c[x * 2u + 0u + (y * 4u + 3u) * ND4] = sum03;
      array_c[x * 2u + 1u + (y * 4u + 3u) * ND4] = sum13;
    }
  `;
}


 fusedAttentionShaderNew = `
    struct Meta {
      M: u32,
      N: u32,
      ND4: u32,
      KD4: u32,
      attentionScale: f32,
    }

    @group(1) @binding(0) var<storage,read> query_array: array<vec4<f32>>;
    @group(1) @binding(1) var<storage,read> key_array: array<vec4<f32>>;

    @group(0) @binding(0) var<uniform> uniforms: Meta;
    @group(0) @binding(1) var<storage,read_write> array_c: array<vec4<f32>>;

    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      var M: u32 = uniforms.M;
      var N: u32 = uniforms.N;
      var ND4: u32 = uniforms.ND4;
      var KD4: u32 = uniforms.KD4;
      var x: u32 = global_id.x;
      var y: u32 = global_id.y;

      if (x * 8 >= N || y * 4 >= M) {
        return;
      }

      var sum00: vec4<f32> = vec4<f32>();
      var sum01: vec4<f32> = vec4<f32>();
      var sum02: vec4<f32> = vec4<f32>();
      var sum03: vec4<f32> = vec4<f32>();
      var sum10: vec4<f32> = vec4<f32>();
      var sum11: vec4<f32> = vec4<f32>();
      var sum12: vec4<f32> = vec4<f32>();
      var sum13: vec4<f32> = vec4<f32>();

      for(var k: u32 = 0u; k < KD4; k = k + 1u) {
        var arow0: vec4<f32> = query_array[(y * 4u + 0u) * KD4 + k];
        var arow1: vec4<f32> = query_array[(y * 4u + 1u) * KD4 + k];
        var arow2: vec4<f32> = query_array[(y * 4u + 2u) * KD4 + k];
        var arow3: vec4<f32> = query_array[(y * 4u + 3u) * KD4 + k];
        var brow: vec4<f32>;

        brow = key_array[(k * 4u + 0u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.x) * brow + sum00;
        sum01 = vec4<f32>(arow1.x) * brow + sum01;
        sum02 = vec4<f32>(arow2.x) * brow + sum02;
        sum03 = vec4<f32>(arow3.x) * brow + sum03;

        brow = key_array[(k * 4u + 0u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.x) * brow + sum10;
        sum11 = vec4<f32>(arow1.x) * brow + sum11;
        sum12 = vec4<f32>(arow2.x) * brow + sum12;
        sum13 = vec4<f32>(arow3.x) * brow + sum13;

        brow = key_array[(k * 4u + 1u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.y) * brow + sum00;
        sum01 = vec4<f32>(arow1.y) * brow + sum01;
        sum02 = vec4<f32>(arow2.y) * brow + sum02;
        sum03 = vec4<f32>(arow3.y) * brow + sum03;

        brow = key_array[(k * 4u + 1u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.y) * brow + sum10;
        sum11 = vec4<f32>(arow1.y) * brow + sum11;
        sum12 = vec4<f32>(arow2.y) * brow + sum12;
        sum13 = vec4<f32>(arow3.y) * brow + sum13;

        brow = key_array[(k * 4u + 2u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.z) * brow + sum00;
        sum01 = vec4<f32>(arow1.z) * brow + sum01;
        sum02 = vec4<f32>(arow2.z) * brow + sum02;
        sum03 = vec4<f32>(arow3.z) * brow + sum03;

        brow = key_array[(k * 4u + 2u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.z) * brow + sum10;
        sum11 = vec4<f32>(arow1.z) * brow + sum11;
        sum12 = vec4<f32>(arow2.z) * brow + sum12;
        sum13 = vec4<f32>(arow3.z) * brow + sum13;

        brow = key_array[(k * 4u + 3u) * ND4 + x * 2u + 0u];
        sum00 = vec4<f32>(arow0.w) * brow + sum00;
        sum01 = vec4<f32>(arow1.w) * brow + sum01;
        sum02 = vec4<f32>(arow2.w) * brow + sum02;
        sum03 = vec4<f32>(arow3.w) * brow + sum03;

        brow = key_array[(k * 4u + 3u) * ND4 + x * 2u + 1u];
        sum10 = vec4<f32>(arow0.w) * brow + sum10;
        sum11 = vec4<f32>(arow1.w) * brow + sum11;
        sum12 = vec4<f32>(arow2.w) * brow + sum12;
        sum13 = vec4<f32>(arow3.w) * brow + sum13;
      }

      if (y * 4u + 0u < M) {
        array_c[x * 2u + 0u + (y * 4u + 0u) * ND4] = sum00;
        array_c[x * 2u + 1u + (y * 4u + 0u) * ND4] = sum10;
      }
      if (y * 4u + 1u < M) {
        array_c[x * 2u + 0u + (y * 4u + 1u) * ND4] = sum01;
        array_c[x * 2u + 1u + (y * 4u + 1u) * ND4] = sum11;
      }
      if (y * 4u + 2u < M) {
        array_c[x * 2u + 0u + (y * 4u + 2u) * ND4] = sum02;
        array_c[x * 2u + 1u + (y * 4u + 2u) * ND4] = sum12;
      }
      if (y * 4u + 3u < M) {
        array_c[x * 2u + 0u + (y * 4u + 3u) * ND4] = sum03;
        array_c[x * 2u + 1u + (y * 4u + 3u) * ND4] = sum13;
      }
    `;


  // In progress.
  //    withCheckOffset: `
  //    var x1Offset: u32 = ((x * 2u + 0u) / uniforms.TOffset) * uniforms.TOffset * M;
  //    var x2Offset: u32 = ((x * 2u + 1u) / uniforms.TOffset) * uniforms.TOffset * M;
  //

  //    if (y * 4u + 0u < M) {
  //      array_c[xMod * 2u + 0u + x1Offset + (y * 4u + 0u) * uniforms.TOffset] = vec4<f32>(1.0);
  //      array_c[xMod * 2u + 1u + x2Offset + (y * 4u + 0u) * uniforms.TOffset] = vec4<f32>(f32(x1Offset));
  //    }
  //    if (y * 4u + 1u < M) {
  //      array_c[xMod * 2u + 0u + x1Offset + (y * 4u + 1u) * uniforms.TOffset] = vec4<f32>(2.0);
  //      array_c[xMod * 2u + 1u + x2Offset + (y * 4u + 1u) * uniforms.TOffset] = vec4<f32>(f32(x2Offset));
  //    }
  //    if (y * 4u + 2u < M) {
  //      array_c[xMod * 2u + 0u + x1Offset + (y * 4u + 2u) * uniforms.TOffset] = vec4<f32>(3.0);
  //      array_c[xMod * 2u + 1u + x2Offset + (y * 4u + 2u) * uniforms.TOffset] = vec4<f32>(3.0);
  //    }
  //    if (y * 4u + 3u < M) {
  //      array_c[xMod * 2u + 0u + x1Offset + (y * 4u + 3u) * uniforms.TOffset] = vec4<f32>(4.0);
  //      array_c[xMod * 2u + 1u + x2Offset + (y * 4u + 3u) * uniforms.TOffset] = vec4<f32>(4.0);
  //    }
  //  `,

    transposeShader = `
    struct Meta {
      M: u32,
      N: u32,
    }
    
    @group(1) @binding(0) var<storage, read> input_array: array<f32>;
    
    @group(0) @binding(0) var<uniform> uniforms: Meta;
    @group(0) @binding(1) var<storage, read_write> result_array: array<f32>;
    
    // Bank conflicts?
    var<workgroup> tile: array<array<f32, 8>, 8>;
    
    @compute @workgroup_size(8, 8)
    fn main (@builtin(workgroup_id) wg_id: vec3<u32>,  @builtin(local_invocation_id) local_id: vec3<u32>) {
      let col: u32 = wg_id.x;
      let row: u32 = wg_id.y;
      let N: u32 = uniforms.N;
      let M: u32 = uniforms.M;

      let tile_col = col * 8u + local_id.x;
      let tile_row = row * 8u + local_id.y;
    
      // Load a tile from input_array to shared memory tile
      if (tile_row < M && tile_col < N) {
        tile[local_id.y][local_id.x] = input_array[tile_row * N + tile_col];
      }
    
      workgroupBarrier(); // Ensure all threads have finished writing to the shared memory before proceeding
        
      // Write the transposed tile to result_array. Flips dims.
      if (tile_row < M && tile_col < N) {
        result_array[tile_col * M + tile_row] = tile[local_id.x][local_id.y]; 
      }
    }
  `;

*/
