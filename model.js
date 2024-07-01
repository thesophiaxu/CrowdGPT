// async function testTransferBuffer(device, obj) {
//   const recurser = async (localObj) => {
//     if (localObj.constructor.name === "GPUBuffer") {
//       await localObj.mapAsync(GPUMapMode.READ, 0, localObj.size);
//       const copyArrayBuffer = stagingBuffer.getMappedRange(0, localObj.size);
//       console.log(copyArrayBuffer);
//       const newBuffer = device.createBuffer({size: localObj.size, usage: localObj.usage});
//       device.queue.writeBuffer(newBuffer, 0, copyArrayBuffer, 0, copyArrayBuffer.length);

//     } else if (Array.isArray(localObj)) {
//       return Promise.all(localObj.map(recurser));
//     } else if (typeof localObj === "object") {
//       return Object.fromEntries(await Promise.all(Object.entries(localObj).map(async ([k, v]) => [k, await recurser(v)])));
//     } else {
//       return localObj;
//     }
//   }
//   const newObj = await recurser(obj);
//   await device.queue.onSubmittedWorkDone();
//   return newObj;
// }

async function serializeBuffer(device, buffer) {
  if (buffer.usage & GPUBufferUsage.COPY_SRC === 0 || buffer.usage & GPUBufferUsage.COPY_DST === 0) {
    throw new Error("Buffer is not copyable");
  }
  const stagingBuffer = device.createBuffer({
    size: buffer.size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  });
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, buffer.size);
  device.queue.submit([commandEncoder.finish()]);
  await stagingBuffer.mapAsync(GPUMapMode.READ, 0, buffer.size);

  const copyArrayBuffer = stagingBuffer.getMappedRange(0, buffer.size);
  return {
    _type: "GPUBuffer",
    size: buffer.size,
    usage: buffer.usage,
    float32ArrayBuffer: new Float32Array(copyArrayBuffer).slice(0),
  }
}

async function fmt(buffer, x, y) {
  const serialized = await serializeBuffer(window.ddevice, buffer);
  console.log(formatAsMatrix(serialized.float32ArrayBuffer, x, y));
}

async function deserializeBuffer(device, buffer) {
  if (buffer._type !== "GPUBuffer") throw new Error("Unexpected buffer type");
  const resBuffer = device.createBuffer({
    size: buffer.size,
    usage: buffer.usage,
  });
  device.queue.writeBuffer(resBuffer, 0, buffer.float32ArrayBuffer, 0);
  await device.queue.onSubmittedWorkDone();
  return resBuffer;
}

async function runComputePasses(device, computePasses) {
  const commandEncoder = device.createCommandEncoder();
  for (const pass of computePasses) {
    if (pass.flag === "compute") {
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pass.pipeline);
      for (let i = 0; i < pass.groups.length; i++) passEncoder.setBindGroup(i, pass.groups[i]);
      passEncoder.dispatchWorkgroups(pass.workgroups.x, pass.workgroups.y, pass.workgroups.z || undefined);
      passEncoder.end();
    } else if (pass.flag === "copy") {
      commandEncoder.copyBufferToBuffer(pass.src, pass.srcOffset, pass.dst, pass.dstOffset, pass.size);
    }
  }
  device.queue.submit([commandEncoder.finish()]);
  return await device.queue.onSubmittedWorkDone();
}

class GPT {
  constructor(folder, type) {
    this.folder = folder;
    this.tokenizerType = type;
    this.initialized = false;

    this.device;
    this.model;
    this.tokenizer;
    this.params;
    this.minBufferOffset = 1;

    this.defaultPrompt;
    this.defaultTopK;
    this.defaultTemperature;
    this.defaultTokens;

    this.externalBuffer;

    this.unloadDeletionStack = [];
    this.paramsDict = {};
    this.memoryDict = {};
    this.gradientsDict = {};
  }

  async initialize() {
    if (this.initialized) return console.error("Model already initialized");
    if (!navigator.gpu) throw new Error("WebGPU is not supported");

    const adapter = await navigator.gpu.requestAdapter();
    this.device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: 1_000_000_000,
        maxStorageBufferBindingSize: 1_000_000_000
      },
    });
    window.ddevice = this.device;

    initializeOperations(this.device);

    [this.model, this.params] = await this.loadModel(this.folder);
    this.tokenizer = this.tokenizerType.startsWith("bpe") ? new GPT2Tokenizer() : new SimpleTokenizer();
    await this.tokenizer.load(this.tokenizerType === "bpe10k" ? "10k": null);

    if (this.tokenizerType.startsWith("bpe")) {
      this.defaultPrompt = `What is the answer to life, the universe, and everything?\n`;
      this.defaultTopK = 3;
      this.defaultTemperature = 1;
      this.defaultTokens = 30;
    } else {
      this.defaultPrompt = `WILL:\nAh, how dare you challenge me?\nHave you forgotten I built WebGPT?\n`;
      this.defaultTopK = 2;
      this.defaultTemperature = 1;
      this.defaultTokens = 80;
    }

    this.initialized = true;

    console.log("Model initialized");

    this.model.layer_caches = [];
    for (let i=0; i<this.params.n_layer; ++i) {
      this.model.layer_caches.push({});
    }
    this.training = false;
  }

  initBuffer(ops, row, col = 1) {
    const buffer = this.device.createBuffer({
      size: this.bufferSize(row, col),
      usage: ops.map((u) => bufferUsageDict[u]).reduce((a, b) => a | b),
    });
    return buffer;
  }

  async *generate(prompt, max_new_tokens, top_k, temperature) {
    if (!this.initialized) {
      console.error("Model not loaded yet");
      return;
    }

    // Buffer size (321644800) exceeds the max buffer size limit (268435456).
    //  - While calling [Device].CreateBuffer([BufferDescriptor]).

    let history = this.tokenizer.encode(prompt);
    console.log(`Prompt (${history.length} tokens):\n${prompt}`);

    const warmupRuns = 3;
    let totalTime = 0;

    for (let i = 0; i < max_new_tokens; i++) {
      const idx_cond = history.slice(-this.params.n_ctx);
      const useAttCache = i !== 0 && history.length <= this.params.n_ctx;

      const startTime = performance.now();
      const logits = await this.run(idx_cond);
      const endTime = performance.now();

      // console.log(`\nIteration ${i + 1} of ${max_new_tokens}`);
      const lapsedTime = endTime - startTime;
      console.log(`Kernel execution time: ${lapsedTime} ms`);
      i >= warmupRuns && (totalTime += lapsedTime);

      const { topKIndices, topKProbs } = selectTopK(logits, top_k);
      const probs = cpuSoftmax(topKProbs, temperature);
      const idx_next = topKIndices[sampleFromDistribution(probs)];

      history = history.concat(idx_next);

      // console.log(`Output:\n${this.tokenizer.decode(history)}`);

      // const totalProbs = cpuSoftmax(logits, temperature);
      // const tokenProbsString = Array.from(totalProbs)
      //   .map((value, index) => ({ value, index }))
      //   .sort((a, b) => b.value - a.value)
      //   .slice(0, 8)
      //   .map((prob) => `{ ${this.tokenizer.decode([prob.index]).replace(/(\r\n|\n|\r)/gm, "newline")} } : ${prob.value.toPrecision(3)}`)
      //   .join(" | ");
      // console.log("Top 8 token probs:", tokenProbsString);

      yield this.tokenizer.decode([idx_next]);
    }

    console.log(`Average kernel execution time: ${totalTime / (max_new_tokens - warmupRuns)} ms`);
  }

  async run(idx, train = false) {
    const { posEmbdBuffer, normGammaBuffer, normBetaBuffer, embeddingsBuffers, deEmbeddingsBuffers } = this.model;
    const { n_embd, n_layer, vocab_size, vocab_chunk_size, vocab_chunk_instances } = this.params;
    if (idx.length % 16 !== 0) {
      idx = [...new Array(Math.ceil(idx.length / 16) * 16 - idx.length).fill(0), ...idx]
    }
    const seq_length = idx.length;
    //const seq_length = 1024;

    // ---------------- Create Passes ---------------- //
    // Note: These are re-initialized because everytime seq_length changes buffers are different sizes.

    // Pipeline creation is major bottleneck to spin up speed! Also buffer re-use.

    let computePasses = [];
    let intermediateBuffer;
    let residualBuffer;
    {
      const { passes, resultBuffer } = EmbedBlock.newInstance(idx, seq_length, n_embd, vocab_chunk_size, embeddingsBuffers, posEmbdBuffer, ResidualBlock);
      intermediateBuffer = resultBuffer;
      residualBuffer = resultBuffer;
      computePasses.push(...passes);
    }
    const midPoint = Math.floor(n_layer / 2);

    if (false) {
      // TEST: 1/2 Run locally
      for (let i = 0; i < midPoint; i++) {
        const resultBuffers = await this.runLayer({
          layer: i,
          seq_length,
          intermediateBuffer,
          intermediateBuffer,
        })
        intermediateBuffer = resultBuffers.intermediateBuffer;
        residualBuffer = resultBuffers.intermediateBuffer;
      }
      await runComputePasses(this.device, computePasses);
      computePasses = [];

      // TEST 2/2 Run remotely
      const remoteResults = await this.runLayersOnOthers({
        from: midPoint,
        to: n_layer,
        seq_length,
        intermediateBuffer,
        intermediateBuffer,
      });
      intermediateBuffer = remoteResults.intermediateBuffer;
      residualBuffer = remoteResults.intermediateBuffer;
    } else {
      for (let i = 0; i < n_layer; i++) {
        const resultBuffers = await this.runLayer({
          layer: i,
          seq_length,
          intermediateBuffer,
          residualBuffer,
        })
        intermediateBuffer = resultBuffers.intermediateBuffer;
        residualBuffer = resultBuffers.intermediateBuffer;
        computePasses.push(...resultBuffers.passes);
      }
    }

    {
      if (this.externalBuffer) {
        computePasses.push({
          flag: "copy",
          src: intermediateBuffer,
          srcOffset: 0,
          dst: this.externalBuffer,
          dstOffset: 0,
          size: this.bufferSize(seq_length, n_embd),
        });
      }
    }
    {
      this.finalLayerNormInput = intermediateBuffer;
      const { passes, caches, resultBuffer } = LayerNormBlock.newInstance(seq_length, n_embd, intermediateBuffer, normGammaBuffer, normBetaBuffer);
      this.finalLayerNormOutput = resultBuffer;
      intermediateBuffer = resultBuffer;
      computePasses.push(...passes);
      this.finalLayerNormStats = caches.statsResultBuffer;
    }
    
    // ---------------- Compute Passes ----------------

    if (!train) {
      {
        const { passes, resultBuffer } = DeEmbedBlock.newInstance(
          n_embd,
          vocab_size,
          vocab_chunk_size * vocab_chunk_instances,
          seq_length,
          vocab_chunk_size,
          intermediateBuffer,
          deEmbeddingsBuffers
        );
        intermediateBuffer = resultBuffer;
        computePasses.push(...passes);
      }

      const resultBuffer = intermediateBuffer;

      await runComputePasses(this.device, computePasses);

      // ---------------- Read Results ----------------

      await resultBuffer.mapAsync(GPUMapMode.READ);
      const output = resultBuffer.getMappedRange();
      const outputArray = new Float32Array(output).slice(0); // Copy the array, otherwise it'll be destroyed.

      clearOperationCache();

      return outputArray;
    } else {
      const { passes, resultBuffer } = FastMatMulBatchedBlock.newInstance(
        seq_length,
        vocab_size,
        n_embd,
        intermediateBuffer,
        deEmbeddingsBuffers[0],
        undefined,
        1,
      )
      computePasses.push(...passes);
      //clearOperationCache();
      return { resultBuffer, passes: computePasses, caches: {
        finalLayerNormInput: this.finalLayerNormInput,
        finalLayerNormOutput: this.finalLayerNormOutput,
        finalLayerNormStats: this.finalLayerNormStats,
      } };
    }
  }

  clearGradients() {
    for (const key in this.paramsDict) {
      if (key.startsWith("lm_head")) continue;
      this.gradientsDict[key] = [];
    }
  }

  async runBackwards(
    outputsBuffer, // seq_length x vocab_size
    targets, // seq_length array
    idx,
    caches,
    oneHotBuffer,
    batchSize,
    params_dict,
  ) {
    const { posEmbdBuffer, normGammaBuffer, normBetaBuffer, embeddingsBuffers, deEmbeddingsBuffers } = this.model;
    const { n_embd, n_layer, vocab_size, vocab_chunk_size, vocab_chunk_instances } = this.params;
    if (idx.length % 16 !== 0) {
      idx = [...new Array(Math.ceil(idx.length / 16) * 16 - idx.length).fill(0), ...idx]
    }
    const seq_length = idx.length;

    const targetsBuffer = this.initBuffer(['storage', 'copy_from', 'copy_to'], seq_length);
    this.device.queue.writeBuffer(targetsBuffer, 0, new Uint32Array(targets));
    const {
      resultBuffer,
      caches: crossEntropyCaches,
      passes: crossEntropyForwardPasses,
    } = CrossEntropyLoss.newInstance(
      seq_length,
      vocab_size,
      outputsBuffer,
      targetsBuffer,
    );
    
    const getLosses = async () => (await serializeBuffer(this.device, resultBuffer)).float32ArrayBuffer;
    

    // now backwards
    const dLosses = new Float32Array(new Array(seq_length).fill(1/(seq_length * batchSize)));
    const dLossesBuffer = this.initBuffer(['storage', 'copy_from', 'copy_to'], seq_length);
    this.device.queue.writeBuffer(dLossesBuffer, 0, dLosses);

    const { 
      dLogitsBuffer, // seq_length x vocab_size
      passes: crossEntropyBackwardPasses
    } = CrossEntropyBackwards.newInstance(
      dLossesBuffer,
      crossEntropyCaches,
      seq_length,
      vocab_size,
      outputsBuffer,
      targetsBuffer,
    );

    const {
      passes: deEmbeddingsPasses,
      dInputBuffer: dDeEmbeddingsInputBuffer,
      dWeightsBuffer: dDeEmbeddingsBuffer,
    } = FastMatMulBackwards.newInstance(
      dLogitsBuffer,
      seq_length,
      vocab_size,
      n_embd,
      caches.finalLayerNormOutput,
      deEmbeddingsBuffers[0],
      undefined,
    )

    const {
      dBetaBuffer: dFinalBetaBuffer,
      dGammaBuffer: dFinalGammaBuffer,
      dInputBuffer: dFinalLayerOutputBuffer,
      passes: layerNormBackwardPasses,
    } = LayerNormBackwards.newInstance(
      dDeEmbeddingsInputBuffer,
      { statsResultBuffer: caches.finalLayerNormStats },
      seq_length,
      n_embd,
      caches.finalLayerNormInput,
      normGammaBuffer,
      normBetaBuffer,
    )

    let dLayerOutputBuffer = dFinalLayerOutputBuffer;
    let layersBackwards = [];
    let layersPasses = [];
    for (let i = n_layer-1; i >= 0; i--) {
      const layerBackwards = await this.layerBackwards({
        layer: i,
        dOutputBuffer: dLayerOutputBuffer,
        seq_length,
      })
      layersBackwards[i] = layerBackwards;
      layersPasses.push(...layerBackwards.passes);
      dLayerOutputBuffer = layerBackwards.dInputBuffer;
    }

    const buffersToDelete = [
      targetsBuffer,
      dLossesBuffer,
    ]
    this.unloadDeletionStack.push(...buffersToDelete);

    const {
      passes: embeddingsPasses,
      dWeightsBuffer: dEmbeddingsBuffer,
    } = FastMatMulBackwards.newInstance(
      dLayerOutputBuffer,
      seq_length,
      n_embd,
      vocab_size,
      oneHotBuffer,
      embeddingsBuffers[0],
      undefined,
    )

    // get joined gradients for embedding / deembedding
    const {
      resultBuffer: dDeEmbeddingsBufferTransposed,
      passes: dDeEmbeddingsTransposePasses
    } = TransposeBlock.newInstance(
      n_embd,
      vocab_size,
      dDeEmbeddingsBuffer,
    )

    const {
      resultBuffer: dJoinedEmbBuffer,
      passes: dJoinedEmbPasses
    } = ResidualBlock.newInstance(
        model.params.vocab_size,
        model.params.n_embd,
        dEmbeddingsBuffer,
        dDeEmbeddingsBufferTransposed,
    )

    // set gradientsDict
    this.gradientsDict[`wte`].push(dJoinedEmbBuffer);
    this.gradientsDict[`ln_f.weight`].push(dFinalGammaBuffer);
    // fmt(dFinalBetaBuffer, 72, 1);
    // fmt(dFinalGammaBuffer, 72, 1);
    this.gradientsDict[`ln_f.bias`].push(dFinalBetaBuffer);
    for (let i = n_layer-1; i >= 0; i--) {
      this.gradientsDict[`h.${i}.attn.q.weight`].push(layersBackwards[i].dAttentionQWeightBuffer);
      this.gradientsDict[`h.${i}.attn.q.bias`].push(layersBackwards[i].dAttentionQBiasBuffer);
      this.gradientsDict[`h.${i}.attn.k.weight`].push(layersBackwards[i].dAttentionKWeightBuffer);
      this.gradientsDict[`h.${i}.attn.k.bias`].push(layersBackwards[i].dAttentionKBiasBuffer);
      this.gradientsDict[`h.${i}.attn.v.weight`].push(layersBackwards[i].dAttentionVWeightBuffer);
      this.gradientsDict[`h.${i}.attn.v.bias`].push(layersBackwards[i].dAttentionVBiasBuffer);
      this.gradientsDict[`h.${i}.attn.c_proj.weight`].push(layersBackwards[i].dAttentionLinearWeightBuffer);
      this.gradientsDict[`h.${i}.attn.c_proj.bias`].push(layersBackwards[i].dAttentionLinearBiasBuffer);
      this.gradientsDict[`h.${i}.ln_1.weight`].push(layersBackwards[i].dLn1WeightBuffer);
      this.gradientsDict[`h.${i}.ln_1.bias`].push(layersBackwards[i].dLn1BiasBuffer);
      this.gradientsDict[`h.${i}.ln_2.weight`].push(layersBackwards[i].dLn2WeightBuffer);
      this.gradientsDict[`h.${i}.ln_2.bias`].push(layersBackwards[i].dLn2BiasBuffer);
      this.gradientsDict[`h.${i}.mlp.c_fc.weight`].push(layersBackwards[i].dMlp1WeightBuffer);
      this.gradientsDict[`h.${i}.mlp.c_fc.bias`].push(layersBackwards[i].dMlp1BiasBuffer);
      this.gradientsDict[`h.${i}.mlp.c_proj.weight`].push(layersBackwards[i].dMlp2WeightBuffer);
      this.gradientsDict[`h.${i}.mlp.c_proj.bias`].push(layersBackwards[i].dMlp2BiasBuffer);
    }


    return {
      getLosses,
      caches,
      dFinalBetaBuffer,
      dFinalGammaBuffer,
      dFinalLayerOutputBuffer,
      layersBackwards,
      dEmbeddingsBuffer,
      dDeEmbeddingsBuffer,
      passes: [
        ...crossEntropyForwardPasses,
        ...crossEntropyBackwardPasses,
        ...deEmbeddingsPasses,
        ...layerNormBackwardPasses,
        ...layersPasses,
        ...embeddingsPasses,
        ...dDeEmbeddingsTransposePasses,
        ...dJoinedEmbPasses,
      ]
    }
  }

  async runLayersForOthers({
    from, to, seq_length, intermediateBufferSerialized, residualBufferSerialized
  }) {
    this.computePasses = [];
    let intermediateBuffer = await deserializeBuffer(this.device, intermediateBufferSerialized);
    let residualBuffer = await deserializeBuffer(this.device, residualBufferSerialized);
    for (let i = from; i < to; i++) {
      // const t0 = Date.now();
      // const _buffers = layer_buffers[i];
      // const buffers = await testTransferBuffer(this.device, _buffers);
      const resultBuffers = await this.runLayer({
        layer: i,
        seq_length,
        intermediateBuffer,
        residualBuffer,
      })
      intermediateBuffer = resultBuffers.intermediateBuffer;
      residualBuffer = resultBuffers.residualBuffer;
    }
    await runComputePasses(this.device, this.computePasses);
    return {
      intermediateBufferSerialized: await serializeBuffer(this.device, intermediateBuffer),
      residualBufferSerialized: await serializeBuffer(this.device, residualBuffer),
    }
  }

  async runLayersOnOthers({
    from, to, seq_length, intermediateBuffer, residualBuffer
  }) {
    const {
      intermediateBufferSerialized,
      residualBufferSerialized
    } = await window.computeNode.runLayersOnAnyNode({
      from, to, seq_length,
      intermediateBufferSerialized: await serializeBuffer(this.device, intermediateBuffer),
      residualBufferSerialized: await serializeBuffer(this.device, residualBuffer),
    });
    return {
      intermediateBuffer: await deserializeBuffer(this.device, intermediateBufferSerialized),
      residualBuffer: await deserializeBuffer(this.device, residualBufferSerialized),
    }
  }

  async runLayer({
    layer, seq_length, intermediateBuffer
  }) {
    const { layer_buffers, layer_caches } = this.model;
    const { attention_scale, n_embd, n_head, head_size, hidden_size } = this.params;
    const i = layer;
    const buffers = layer_buffers[i];
    const caches = layer_caches[i];
    const computePasses = [];
    let residualBuffer = intermediateBuffer;
    if (this.training) {
      caches.inputBuffer = intermediateBuffer;
    }
    // console.log('Data transfer overhead: ', Date.now() - t0, ' ms');
    {
      const { passes, caches: lnCaches, resultBuffer } = LayerNormBlock.newInstance(
        seq_length,
        n_embd,
        intermediateBuffer,
        buffers.normAttentionGammaBuffer,
        buffers.normAttentionBetaBuffer
      );
      if (this.training) {
        caches.layerNorm1StatsResultBuffer = lnCaches.statsResultBuffer;
      }
      intermediateBuffer = resultBuffer;
      computePasses.push(...passes);
    }
    {
      const { passes, tmpBuffer, caches: attnCaches, resultBuffer } = AttentionBlock.newFusedInstance(
        seq_length,
        n_embd,
        attention_scale,
        n_head,
        head_size,
        intermediateBuffer,
        buffers.qkvWeightArray[0],
        buffers.qkvBiasArray[0],
        buffers.qkvWeightArray[1],
        buffers.qkvBiasArray[1],
        buffers.qkvWeightArray[2],
        buffers.qkvBiasArray[2],
        buffers.linearWeightsBuffer,
        buffers.linearBiasBuffer,
        FastMatMulBlock,
        SoftmaxBlock
      );
      if (this.training) {
        caches.attentionCaches = attnCaches;
        caches.attentionInputBuffer = intermediateBuffer;
      }
      intermediateBuffer = resultBuffer;
      computePasses.push(...passes);

      // await runComputePasses(this.device, computePasses);
      // computePasses = [];
      // console.log(i);
      // console.log(formatAsMatrix(
      //   (await serializeBuffer(this.device, tmpBuffer)).float32ArrayBuffer,
      //   seq_length,
      //   head_size * n_head,
      // ));
    }
    {
      const { passes, resultBuffer } = ResidualBlock.newInstance(seq_length, n_embd, intermediateBuffer, residualBuffer);
      intermediateBuffer = resultBuffer;
      residualBuffer = resultBuffer;
      computePasses.push(...passes);
    }
    {
      const { passes, caches: lnCaches, resultBuffer } = LayerNormBlock.newInstance(
        seq_length,
        n_embd,
        intermediateBuffer,
        buffers.normLinearGammaBuffer,
        buffers.normLinearBetaBuffer
      );
      if (this.training) {
        caches.layerNorm2StatsResultBuffer = lnCaches.statsResultBuffer;
        caches.layerNorm2InputBuffer = intermediateBuffer;
      }
      intermediateBuffer = resultBuffer;
      computePasses.push(...passes);
    }
    {
      const { resultBuffer, passes } = FastMatMulBlock.newInstance(
        seq_length,
        hidden_size,
        n_embd,
        intermediateBuffer,
        buffers.firstLayerWeightsBuffer,
        buffers.firstLayerBiasBuffer
      );
      if (this.training) {
        caches.mlp1OutputBuffer = resultBuffer;
        caches.mlp1InputBuffer = intermediateBuffer;
      }
      intermediateBuffer = resultBuffer;
      computePasses.push(...passes);
      
    }
    {
      const { resultBuffer, passes } = GeluBlock.newInstance(seq_length, hidden_size, intermediateBuffer);
      intermediateBuffer = resultBuffer;
      computePasses.push(...passes);
    }
    {
      const { resultBuffer, passes } = FastMatMulBlock.newInstance(
        seq_length,
        n_embd,
        hidden_size,
        intermediateBuffer,
        buffers.secondLayerWeightsBuffer,
        buffers.secondLayerBiasBuffer
      );
      if (this.training) caches.mlp2InputBuffer = intermediateBuffer;
      intermediateBuffer = resultBuffer;
      computePasses.push(...passes);
    }
    {
      const { passes, resultBuffer } = ResidualBlock.newInstance(seq_length, n_embd, intermediateBuffer, residualBuffer);
      intermediateBuffer = resultBuffer;
      residualBuffer = resultBuffer;
      computePasses.push(...passes);
    }
    // if (i % 2 === 0) {
    //   await runComputePasses(this.device, computePasses);
    //   computePasses = [];
    //   const t0 = Date.now();
    //   const sb1 = await serializeBuffer(this.device, intermediateBuffer);
    //   const sb2 = await serializeBuffer(this.device, residualBuffer);
    //   intermediateBuffer = await deserializeBuffer(this.device, sb1);
    //   residualBuffer = await deserializeBuffer(this.device, sb2);
    //   //console.log(sb);
    //   console.log('Data serialize overhead: ', Date.now() - t0, ' ms');
    // }
    return { intermediateBuffer, passes: computePasses };
  }

  async layerBackwards({
    layer,
    dOutputBuffer,
    seq_length,
  }) {
    const { layer_buffers, layer_caches } = this.model;
    const { attention_scale, n_embd, n_head, head_size, hidden_size } = this.params;

    const buffers = layer_buffers[layer];
    const cache = layer_caches[layer];

    const {
      mlp2InputBuffer,
      mlp1OutputBuffer,
      mlp1InputBuffer,
      layerNorm2StatsResultBuffer,
      layerNorm2InputBuffer,
      attentionInputBuffer,
      attentionCaches,
      layerNorm1StatsResultBuffer,
      inputBuffer,
    } = cache;

    const dMlp2OutputBuffer = dOutputBuffer;
    const dResidualOutputBuffer_1Component = dOutputBuffer;
    const {
      dInputBuffer: dMlp2InputBuffer,
      dWeightsBuffer: dMlp2WeightBuffer,
      dBiasBuffer: dMlp2BiasBuffer,
      passes: dMlp2Passes
    } = FastMatMulBackwards.newInstance(
      dMlp2OutputBuffer,
      seq_length,
      n_embd,
      hidden_size,
      mlp2InputBuffer,
      buffers.secondLayerWeightsBuffer,
      buffers.secondLayerBiasBuffer,
    )

    const { 
      dInputBuffer: dMlp1OutputBuffer,
      passes: dGeluPasses,
     } = GeluBackwards.newInstance(dMlp2InputBuffer, seq_length, hidden_size, mlp1OutputBuffer);

     const {
      dInputBuffer: dLn2OuptutBuffer,
      dWeightsBuffer: dMlp1WeightBuffer,
      dBiasBuffer: dMlp1BiasBuffer,
      passes: dMlp1Passes
    } = FastMatMulBackwards.newInstance(
      dMlp1OutputBuffer,
      seq_length,
      hidden_size,
      n_embd,
      mlp1InputBuffer,
      buffers.firstLayerWeightsBuffer,
      buffers.firstLayerBiasBuffer,
    );

    const {
      dInputBuffer: dResidualOutputBuffer_2Component,
      dGammaBuffer: dLn2WeightBuffer,
      dBetaBuffer: dLn2BiasBuffer,
      passes: dLayerNorm2Passes
    } = LayerNormBackwards.newInstance(
      dLn2OuptutBuffer,
      { statsResultBuffer: layerNorm2StatsResultBuffer },
      seq_length,
      n_embd,
      layerNorm2InputBuffer,
      buffers.normLinearGammaBuffer,
      buffers.normLinearBetaBuffer
    );

    const {
      resultBuffer: dResidualOutputBuffer,
      passes: combinePasses1,
    } = ResidualBlock.newInstance(
      seq_length,
      n_embd,
      dResidualOutputBuffer_1Component,
      dResidualOutputBuffer_2Component,
    );

    const dAttentionOutputBuffer = dResidualOutputBuffer;
    const dInputBuffer_1Component = dResidualOutputBuffer;

    const {
      dInputBuffer: dAttentionInputBuffer,
      dQWeightsBuffer: dAttentionQWeightBuffer,
      dKWeightsBuffer: dAttentionKWeightBuffer,
      dVWeightsBuffer: dAttentionVWeightBuffer,
      dQBiasBuffer: dAttentionQBiasBuffer,
      dKBiasBuffer: dAttentionKBiasBuffer,
      dVBiasBuffer: dAttentionVBiasBuffer,
      dLinearWeightsBuffer: dAttentionLinearWeightBuffer,
      dLinearBiasBuffer: dAttentionLinearBiasBuffer,
      passes: dAttentionPasses,
    } = AttentionBackwards.newInstance(
      dAttentionOutputBuffer,
      attentionCaches,
      seq_length,
      n_embd,
      attention_scale,
      n_head,
      head_size,
      attentionInputBuffer,
      buffers.qkvWeightArray[0],
      buffers.qkvBiasArray[0],
      buffers.qkvWeightArray[1],
      buffers.qkvBiasArray[1],
      buffers.qkvWeightArray[2],
      buffers.qkvBiasArray[2],
      buffers.linearWeightsBuffer,
      buffers.linearBiasBuffer,
      FastMatMulBlock,
      SoftmaxBlock
    )

    const {
      dInputBuffer: dInputBuffer_2Component,
      dGammaBuffer: dLn1WeightBuffer,
      dBetaBuffer: dLn1BiasBuffer,
      passes: dLayerNorm1Passes,
    } = LayerNormBackwards.newInstance(
      dAttentionInputBuffer,
      { statsResultBuffer: layerNorm1StatsResultBuffer },
      seq_length,
      n_embd,
      inputBuffer,
      buffers.normAttentionGammaBuffer,
      buffers.normAttentionBetaBuffer
    )

    const {
      resultBuffer: dInputBuffer,
      passes: combinePasses2,
    } = ResidualBlock.newInstance(
      seq_length,
      n_embd,
      dInputBuffer_1Component,
      dInputBuffer_2Component,
    );
    
    return {
      dInputBuffer,
      dLn1WeightBuffer,
      dLn1BiasBuffer,
      dAttentionQWeightBuffer,
      dAttentionQBiasBuffer,
      dAttentionKWeightBuffer,
      dAttentionKBiasBuffer,
      dAttentionVWeightBuffer,
      dAttentionVBiasBuffer,
      dAttentionLinearWeightBuffer,
      dAttentionLinearBiasBuffer,
      dLn2WeightBuffer,
      dLn2BiasBuffer,
      dMlp1WeightBuffer,
      dMlp1BiasBuffer,
      dMlp2WeightBuffer,
      dMlp2BiasBuffer,
      passes: [
        ...dMlp2Passes,
        ...dGeluPasses,
        ...dMlp1Passes,
        ...dLayerNorm2Passes,
        ...combinePasses1,
        ...dAttentionPasses,
        ...dLayerNorm1Passes,
        ...combinePasses2,
      ]
    }
  }

  async testLayer() {
    this.computePasses = [];
    this.training = true;
    const inputBuffer = await this.fetchAndInitTensor(`weights/test/1L_input.bin`, [16, 128], ["storage", "copy_from"]);
    const dOutputBuffer = await this.fetchAndInitTensor(`weights/test/1L_dOutput.bin`, [16, 128], ["storage", "copy_from"]);
    const emptyResidual = this.initTensor(new Array(16 * 128).fill(0), [16, 128], ["storage", "copy_from", "copy_to"])
    const { intermediateBuffer } = await this.runLayer({
      layer: 0, 
      seq_length: 16,
      residualBuffer: emptyResidual,
      intermediateBuffer: inputBuffer,
    });
    await runComputePasses(this.device, this.computePasses);
    console.log("OUTPUT:")
    console.log(formatAsMatrix(
      (await serializeBuffer(this.device, intermediateBuffer)).float32ArrayBuffer,
      16,
      128,
    ));
    console.log("DOUTPUT:")
    console.log(formatAsMatrix(
      (await serializeBuffer(this.device, dOutputBuffer)).float32ArrayBuffer,
      16,
      128,
    ));
    const {
      dInputBuffer,
      dLn1WeightBuffer,
      dLn1BiasBuffer,
      dAttentionQWeightBuffer,
      dAttentionQBiasBuffer,
      dAttentionKWeightBuffer,
      dAttentionKBiasBuffer,
      dAttentionVWeightBuffer,
      dAttentionVBiasBuffer,
      dAttentionLinearWeightBuffer,
      dAttentionLinearBiasBuffer,
      dLn2WeightBuffer,
      dLn2BiasBuffer,
      dMlp1WeightBuffer,
      dMlp1BiasBuffer,
      dMlp2WeightBuffer,
      dMlp2BiasBuffer,
      passes,
    } = await this.layerBackwards({
      layer: 0,
      seq_length: 16,
      dOutputBuffer,
      inputBuffer,
    })
    await runComputePasses(this.device, passes);
    console.log("dInput:")
    console.log(formatAsMatrix(
      (await serializeBuffer(this.device, dAttentionQWeightBuffer)).float32ArrayBuffer,
      128,
      128,
    ));
    console.log("dInput:")
    console.log(formatAsMatrix(
      (await serializeBuffer(this.device, dInputBuffer)).float32ArrayBuffer,
      16, 128,
      1,
    ));
    return;
  }

  async loadModel(folder) {
    if (this.initialized) return console.error("Model already loaded");

    console.log("Loading model from folder:", folder);
    const weightsFolder = `weights/${folder}/`;

    const params = await this.loadParameters(weightsFolder);
    const { embeddingsBuffers, deEmbeddingsBuffers } = await this.loadEmbeddings(params, weightsFolder);
    const { posEmbdBuffer } = await this.loadPositionalEmbeddings(params, weightsFolder);
    const layer_buffers = await this.loadLayers(params, weightsFolder);

    console.log("Loading final layer norm...");
    const { normGammaBuffer, normBetaBuffer } = await this.loadFinalLayerNorm(params, weightsFolder);

    const output = { layer_buffers, embeddingsBuffers, deEmbeddingsBuffers, posEmbdBuffer, normGammaBuffer, normBetaBuffer };
    console.log("Finished loading model.", output, params);
    return [output, params];
  }

  async loadParameters(weightsFolder) {
    console.log("Loading params...");
    const params = await (await fetch(`${weightsFolder}/params_gpt.json`)).json();

    // Did you enable GitHub LFS? Won't work without it.
    if (params.n_embd % 4 !== 0) throw new Error("Model load failed: n_embd must be divisible by 4.");
    if (params.n_embd % params.n_head !== 0) throw new Error("Model load failed: n_embd must be divisible by n_head.");
    // I'm unsure if this is a reasonable requirement here. At worst, I can figure out some padding method.
    if ((params.n_embd / params.n_head) % 4 !== 0) throw new Error("Model load failed: n_embd / n_head must be divisible by 4.");
    const tokenParam = this.bufferSize(params.vocab_size, params.n_embd);
    let minSplits = Math.ceil(tokenParam / this.device.limits.maxStorageBufferBindingSize);
    function vocabChunkSizeCalc(vocab_size, n_embd, splits, maxStorageBufferBindingSize) {
      // Possibly could be better? Needs actual benchmarking to know what approach is best.
      const optimisticSize = Math.ceil(vocab_size / splits / 4) * 4 * n_embd;
      const pessimiticSize = Math.floor(vocab_size / splits / 4) * 4 * n_embd;
      let vocab_chunk_size = optimisticSize;
      if (optimisticSize > maxStorageBufferBindingSize) {
        vocab_chunk_size = pessimiticSize;
        if (pessimiticSize * splits < tokenParam) {
          return vocabChunkSizeCalc(vocab_size, n_embd, splits + 1, maxStorageBufferBindingSize);
        }
      }
      return { vocab_chunk_size: vocab_chunk_size / n_embd, splits };
    }
    const { vocab_chunk_size, splits } = vocabChunkSizeCalc(params.vocab_size, params.n_embd, minSplits, this.device.limits.maxStorageBufferBindingSize);
    console.log("Splits: ", splits)
    if (splits !== 1) console.error("This model will be too big to train here.")
    if (splits > minSplits) console.warn(`Non-optimal number of vocab splits. Optimal: ${minSplits}, Selected: ${splits}`);

    // Set derived parameters
    params.vocab_chunk_size = vocab_chunk_size;
    params.vocab_chunk_instances = splits;
    params.head_size = params.n_embd / params.n_head;
    params.hidden_size = params.n_embd * 4;
    params.attention_scale = 1 / Math.sqrt(params.n_embd / params.n_head);
    params.bias = params.bias == undefined ? true : params.bias;

    // Check for overflow in buffers larger than maxStorageBufferBindingSize
    const maxBufferSize = this.device.limits.maxStorageBufferBindingSize / 4;
    if (params.n_embd * params.n_ctx > maxBufferSize) console.warn("Model load failed: n_embd * n_ctx must be less than maxStorageBufferBindingSize.");
    if (params.n_embd * params.hidden_size > maxBufferSize)
      console.warn("Model load failed: n_embd * hidden_size must be less than maxStorageBufferBindingSize.");
    if (params.n_ctx * params.n_ctx * params.n_head > maxBufferSize)
      console.warn("Model load failed: n_ctx * n_ctx must be less than maxStorageBufferBindingSize.");
    if (params.n_embd * params.n_embd * 3 > maxBufferSize)
      console.warn("Model load failed: n_embd * n_embd * 3 must be less than maxStorageBufferBindingSize.");

    console.log("Params:", params);

    return params;
  }

  async loadEmbeddings(params, weightsFolder) {
    console.log("Loading token embeddings...");
    const embeddingWeights = await fetchBin(`${weightsFolder}/transformer.wte.weight_gpt.bin`);

    // Chunks are stored in row-major order and are of dimensions n_embd x vocab_chunk_size.
    // Embedding weights are imported in column-major order and are of dimensions vocab_size x n_embd.
    // We pre-transpose the chunk for the deEmbedding process for the matmul. Could do this on GPU later.
    const embeddingsBuffers = [];
    const deEmbeddingsBuffers = [];
    for (let i = 0; i < params.vocab_chunk_instances; i++) {
      console.log(`Loading deEmbedding chunk ${i + 1}/${params.vocab_chunk_instances}...`);
      const offset = i * params.vocab_chunk_size;
      let size = params.vocab_chunk_size;

      const paddedArray = new Float32Array(params.vocab_chunk_size * params.n_embd);
      if (i === params.vocab_chunk_instances - 1) {
        size = params.vocab_size - offset;
        paddedArray.set(size * params.n_embd, zeros((params.vocab_chunk_size * params.vocab_chunk_instances - params.vocab_size) * params.n_embd));
      }
      paddedArray.set(embeddingWeights.subarray(offset * params.n_embd, offset * params.n_embd + size * params.n_embd));

      const embBuffer = this.initTensor(paddedArray, [params.vocab_chunk_size, params.n_embd], ["storage", "copy_from", "copy_to"]);
      embeddingsBuffers.push(embBuffer);

      //const chunk = transpose(paddedArray, params.vocab_chunk_size, params.n_embd); // Use GPU perhaps?
      const { resultBuffer: deembBuffer, passes } = TransposeBlock.newInstance(
        params.vocab_chunk_size,
        params.n_embd,
        embBuffer,
        undefined,
        true
      );
      this.unloadDeletionStack.push(deembBuffer);
      await runComputePasses(this.device, passes);
      deEmbeddingsBuffers.push(deembBuffer);
    }

    this.paramsDict[`wte`] = embeddingsBuffers[0];
    this.paramsDict[`lm_head.weight`] = deEmbeddingsBuffers[0];

    return { embeddingsBuffers, deEmbeddingsBuffers };
  }

  async loadPositionalEmbeddings(params, weightsFolder) {
    console.log("Loading positional embeddings...");
    const posEmbeddings = await fetchBin(`${weightsFolder}/transformer.wpe.weight_gpt.bin`);
    const posEmbdBuffer = this.initTensor(posEmbeddings, [params.n_ctx, params.n_embd], ["copy_from"]);

    this.paramsDict[`wpe`] = posEmbdBuffer;

    return { posEmbdBuffer };
  }

  async loadFinalLayerNorm(params, weightsFolder) {
    console.log("Loading final norm...");
    const prefix = `${weightsFolder}/transformer.ln_f.`;

    const tensorPromises = [
      this.fetchAndInitTensor(`${prefix}weight_gpt.bin`, [params.n_embd], ["storage", "copy_from", "copy_to"]),
      this.fetchAndInitTensor(`${prefix}bias_gpt.bin`, [params.n_embd], ["storage", "copy_from", "copy_to"]),
    ];

    const [normGammaBuffer, normBetaBuffer] = await Promise.all(tensorPromises);

    this.paramsDict[`ln_f.weight`] = normGammaBuffer;
    this.paramsDict[`ln_f.bias`] = normBetaBuffer;

    return { normGammaBuffer, normBetaBuffer };
  }

  async loadLayers(params, weightsFolder) {
    console.log("Loading layers...");
    const layerPromises = [];

    for (let i = 0; i < params.n_layer; i++) {
      layerPromises.push(this.loadLayer(params, weightsFolder, i));
    }

    const layer_buffers = await Promise.all(layerPromises);
    return layer_buffers;
  }

  async loadLayer(params, weightsFolder, layerIndex) {
    console.log("Starting to load layer...", layerIndex);
    const prefix = `${weightsFolder}transformer.h.${layerIndex}.`;

    // Create an array of promises for fetching and initializing the tensors
    const tensorPromises = [
      this.fetchAndInitTensor(`${prefix}ln_1.weight_gpt.bin`, [params.n_embd], ["storage", "copy_from", "copy_to"]),
      this.fetchAndInitTensor(`${prefix}ln_1.bias_gpt.bin`, [params.n_embd], ["storage", "copy_from", "copy_to"]),
      this.fetchAndSplitQKVWeightTensors(`${prefix}attn.c_attn.weight_gpt.bin`, [params.n_embd, 3 * params.n_embd], ["storage", "copy_from", "copy_to"]),
      this.fetchAndSplitQKVBiasTensors(`${prefix}attn.c_attn.bias_gpt.bin`, [params.n_embd], ["storage", "copy_from", "copy_to"]),
      this.fetchAndInitTensor(`${prefix}attn.c_proj.weight_gpt.bin`, [params.n_embd, params.n_embd], ["storage", "copy_from", "copy_to"]),
      this.fetchAndInitTensor(`${prefix}attn.c_proj.bias_gpt.bin`, [params.n_embd], ["storage", "copy_from", "copy_to"]),
      this.fetchAndInitTensor(`${prefix}ln_2.weight_gpt.bin`, [params.n_embd], ["storage", "copy_from", "copy_to"]),
      this.fetchAndInitTensor(`${prefix}ln_2.bias_gpt.bin`, [params.n_embd], ["storage", "copy_from", "copy_to"]),
      this.fetchAndInitTensor(`${prefix}mlp.c_fc.weight_gpt.bin`, [params.n_embd, params.hidden_size], ["storage", "copy_from", "copy_to"]),
      this.fetchAndInitTensor(`${prefix}mlp.c_fc.bias_gpt.bin`, [params.hidden_size], ["storage", "copy_from", "copy_to"]),
      this.fetchAndInitTensor(`${prefix}mlp.c_proj.weight_gpt.bin`, [params.hidden_size, params.n_embd], ["storage", "copy_from", "copy_to"]),
      this.fetchAndInitTensor(`${prefix}mlp.c_proj.bias_gpt.bin`, [params.n_embd], ["storage", "copy_from", "copy_to"]),
    ];

    // Wait for all tensors to be fetched and initialized
    const [
      normAttentionGammaBuffer,
      normAttentionBetaBuffer,
      qkvWeightArray,
      qkvBiasArray,
      linearWeightsBuffer,
      linearBiasBuffer,
      normLinearGammaBuffer,
      normLinearBetaBuffer,
      firstLayerWeightsBuffer,
      firstLayerBiasBuffer,
      secondLayerWeightsBuffer,
      secondLayerBiasBuffer,
    ] = await Promise.all(tensorPromises);

    this.paramsDict[`h.${layerIndex}.attn.q.weight`] = qkvWeightArray[0];
    this.paramsDict[`h.${layerIndex}.attn.q.bias`] = qkvBiasArray[0];
    this.paramsDict[`h.${layerIndex}.attn.k.weight`] = qkvWeightArray[1];
    this.paramsDict[`h.${layerIndex}.attn.k.bias`] = qkvBiasArray[1];
    this.paramsDict[`h.${layerIndex}.attn.v.weight`] = qkvWeightArray[2];
    this.paramsDict[`h.${layerIndex}.attn.v.bias`] = qkvBiasArray[2];
    this.paramsDict[`h.${layerIndex}.attn.c_proj.weight`] = linearWeightsBuffer;
    this.paramsDict[`h.${layerIndex}.attn.c_proj.bias`] = linearBiasBuffer;
    this.paramsDict[`h.${layerIndex}.ln_1.weight`] = normAttentionGammaBuffer;
    this.paramsDict[`h.${layerIndex}.ln_1.bias`] = normAttentionBetaBuffer;
    this.paramsDict[`h.${layerIndex}.ln_2.weight`] = normLinearGammaBuffer;
    this.paramsDict[`h.${layerIndex}.ln_2.bias`] = normLinearBetaBuffer;
    this.paramsDict[`h.${layerIndex}.mlp.c_fc.weight`] = firstLayerWeightsBuffer;
    this.paramsDict[`h.${layerIndex}.mlp.c_fc.bias`] = firstLayerBiasBuffer;
    this.paramsDict[`h.${layerIndex}.mlp.c_proj.weight`] = secondLayerWeightsBuffer;
    this.paramsDict[`h.${layerIndex}.mlp.c_proj.bias`] = secondLayerBiasBuffer;

    // Process the fetched data and return the layer buffers
    return {
      normAttentionGammaBuffer,
      normAttentionBetaBuffer,
      qkvWeightArray,
      qkvBiasArray,
      linearWeightsBuffer,
      linearBiasBuffer,
      normLinearGammaBuffer,
      normLinearBetaBuffer,
      firstLayerWeightsBuffer,
      firstLayerBiasBuffer,
      secondLayerWeightsBuffer,
      secondLayerBiasBuffer,
    };
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

  async fetchAndInitTensor(url, dims, ops) {
    console.log("Fetching and initializing tensor...", url);
    const data = await fetchBin(url);
    return this.initTensor(data, dims, ops);
  }

  initTensor(data, dims, ops) {
    const buffer = this.device.createBuffer({
      size: this.bufferSize(dims[0], dims[1] || 1, dims[2] || 1),
      usage: ops.map((u) => bufferUsageDict[u]).reduce((a, b) => a | b),
      mappedAtCreation: true,
    });
    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();
    this.unloadDeletionStack.push(buffer);
    return buffer;
  }

  unloadBuffers() {
    this.unloadDeletionStack.map((buffer) => buffer.destroy());
    this.unloadDeletionStack = [];
  }

  bufferSize(dimX, dimY = 1, dimZ = 1) {
    const size = Math.ceil((dimX * dimY * dimZ * Float32Array.BYTES_PER_ELEMENT) / this.minBufferOffset) * this.minBufferOffset;
    if (size > this.device.limits.maxStorageBufferBindingSize)
      console.warn("Warning: Buffer size calc result exceeds GPU limit, are you using this value for a tensor size?", dimX, dimY, dimZ, size);
    return size;
  }
}
