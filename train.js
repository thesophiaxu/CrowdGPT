async function prepareDatasets({
    datasetName,
    partition = 0,
    tokenizerName = "bpe10k",
    endOfTextToken = "<|endoftext|>"
}) {
    // 1. Prepare tokens
    const datasetRes = await fetch(`/datasets/${datasetName}/train_${partition}.txt`);
    const dataset = await datasetRes.text();
    const tokenizerClass = tokenizerName.startsWith("bpe") ? GPT2Tokenizer : SimpleTokenizer;
    const tokenizer = new tokenizerClass();
    await tokenizer.load(tokenizerName === "bpe10k" ? "10k": null);
    let tokens;
    if (tokenizerName === "bpe10k") {
        const datasetParts = dataset.split(endOfTextToken);
        const tokenParts = datasetParts.map(it => tokenizer.encode(it));
        const separatorToken = 14663; //tokenizer.encode(endOfTextToken);
        tokens = tokenParts.reduce(
            (acc, val, idx) => idx === 0 ? val : acc.push(separatorToken, ...val) && acc, []
        );
    } else {
        tokens = tokenizer.encode(dataset);
    }
    
    console.log(`Loaded training dataset with ${tokens.length} tokens.`);
    console.log(tokens);
    console.log(tokenizer);
    //return;

    // 2. Create input / output pairs
    const model = window.model;
    if (!model) throw new Error("Must load model first.");
    const params = 0
        + model.params.vocab_size * model.params.n_embd // em/deembedding weights
        + model.params.n_ctx * model.params.n_embd // positional embeddings
        + model.params.n_layer * ( 0
            + model.params.n_embd * 4 // layer norms
            + model.params.n_embd ** 2 * 3 // attention weights
            + model.params.n_embd * 3 // attention biases
            + model.params.n_embd ** 2 // attention proj weights
            + model.params.n_embd // attention proj biases
            + model.params.n_embd ** 2 * 4 * 2 // mlp weights
            + model.params.n_embd * 4 // mlp fc biases
            + model.params.n_embd // mlp proj biases
        )
        + model.params.n_embd * 2 // final layer norm
    console.log(`Training with ${params} parameters.`);

    // 3. Prepare input / output pairs
    

    // 4. Start training
    let local_grad_acc = 1;
    let max_step = 2000;
    let step = 0;
    let ioPairs = preparePairs(tokens, model.params.n_ctx, model.params.vocab_size, model);
    
    model.training = true;
    initTraining(model);

    while (step < max_step) {
        const times = [];
        let start_time = Date.now();
        if (ioPairs.length === 0) {
            ioPairs = preparePairs(tokens, model.params.n_ctx, model.params.vocab_size, model);
        }
        const runs = [];
        const inputs = [];
        const targets = [];
        const onehots = [];
        for (let i=0; i<local_grad_acc; i++) {
            const pair = ioPairs.shift();
            const input = pair.input;
            inputs.push(input);
            //console.log(input);
            const target = pair.output;
            targets.push(target);
            runs.push(model.run(input, true));
            onehots.push(pair.onehot);
        }
        const results = await Promise.all(runs);
        const passes = results.map(it => it.passes);
        //console.log(passes.flat().length)
        const outputs = results.map(it => it.resultBuffer);
        await runComputePasses(model.device, passes.flat());
        //console.log(targets, outputs);
        // console.log(formatAsMatrix(
        //     (await serializeBuffer(model.device, outputs[0])).float32ArrayBuffer,
        //     model.params.n_ctx,
        //     model.params.vocab_size,
        //   ))

        let end_time = Date.now();
        times.push(end_time - start_time);

        start_time = Date.now();
        const backwardRuns = [];
        for (let i=0; i<local_grad_acc; i++) {
            backwardRuns.push(model.runBackwards(
                outputs[i],
                targets[i],
                inputs[i],
                results[i].caches,
                onehots[i],
                local_grad_acc,
            ))
        }
        const backwardResults = await Promise.all(backwardRuns);
        const backwardPasses = backwardResults.map(it => it.passes).flat();
        //console.log(backwardPasses.length)
        // for (let i=0; i<backwardPasses.length; i++) {
        //     const t0 = Date.now();
        //     await runComputePasses(model.device, [backwardPasses[i]]);
        //     const t1 = Date.now();
        //     console.log(`Pass ${i} in ${t1 - t0}ms.`, backwardPasses[i]);
        // }
        await runComputePasses(model.device, backwardPasses);
        
        const losses = (await Promise.all(backwardResults.map(async it => [...await it.getLosses()]))).flat();
        const avgLoss = losses.reduce((acc, val) => acc + val, 0) / losses.length;
        
        //console.log(backwardResults);

        end_time = Date.now();
        times.push(end_time - start_time);

        // update weights
        start_time = Date.now();
        await updateWeights(model, step);
        end_time = Date.now();
        times.push(end_time - start_time);
        console.log(`Step ${step} - Avg loss: ${avgLoss} - ${times.join(" / ")}ms.`);

        clearOperationCache();

        step++;
    }
}

function preparePairs(tokens, block_size, vocab_size, model, num_pairs = 1) {
    const ioPairs = [];
    for (let i=0; i<num_pairs; i++) {
        // generate a random index between 0 and tokens.length - block_size
        const idx = i * 69 //Math.floor(Math.random() * (tokens.length - block_size - 1));

        const input = tokens.slice(idx, idx+block_size);
        const output = tokens.slice(idx+1, idx+block_size+1);

        const oneHotArrays = new Float32Array(block_size * vocab_size);
        input.forEach((target, i) => {
            oneHotArrays[i * vocab_size + target] = 1.0;
        });
        const oneHotBuffer = model.initBuffer(['storage', 'copy_from', 'copy_to'], block_size * vocab_size);
        model.device.queue.writeBuffer(oneHotBuffer, 0, oneHotArrays);
        
        ioPairs.push({
            input,
            output,
            onehot: oneHotBuffer,
        });
    }
    return ioPairs;
}

async function updateWeights(model, step) {
    const computePasses = [];

    const debugDisabled = [] //["ln_f.weight", "ln_f.bias", "wte"];
    Object.keys(model.paramsDict).forEach((key) => {
        if (key === "wpe" || key === "lm_head.weight"
            || (key.startsWith('h.') && key.includes(".attn.q.weight")) 
            || (key.startsWith('h.') && key.includes(".attn.k.weight")) 
            || (key.startsWith('h.') && key.includes(".attn.v")) 
            || (key.startsWith('h.') && key.includes(".mlp.") && key.includes(".c_fc."))
            || debugDisabled.includes(key)) return; // ignore positional embeddings
        
        const currBuffer = model.paramsDict[key];
        const currGradients = model.gradientsDict[key];
        const currMomentum = model.memoryDict[key][0];
        const currVelocity = model.memoryDict[key][1];
        const size = currBuffer.size / 4; // fp32

        const { passes } = AdamWBlock.newInstance({
            weightsBuffer: currBuffer,
            gradientsBuffer: currGradients,
            mMemoryBuffer: currMomentum,
            vMemoryBuffer: currVelocity,
            numParameters: size,
            learningRate: 0.0005,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1e-8,
            t: step,
        });
        computePasses.push(...passes);
    })

    const { passes: transposeLmHeadPasses } = TransposeBlock.newInstance(
        model.params.vocab_size,
        model.params.n_embd,
        model.paramsDict['wte'],
        undefined,
        false,
        model.paramsDict['lm_head.weight'],
    );
    computePasses.push(...transposeLmHeadPasses);

    await runComputePasses(model.device, computePasses);

    model.clearGradients();
}

function initTraining(model) {
    model.training = true;

    // Init momentum memories
    if (Object.keys(model.memoryDict).length === 0) {
        Object.entries(model.paramsDict).forEach(([key, buffer]) => {
            if (key === "wpe" || key === "lm_head.weight") return; // ignore positional embeddings
            model.memoryDict[key] = [];
            model.memoryDict[key].push(model.initBuffer(["storage", "copy_from", "copy_to"], buffer.size / 4, undefined));
            model.memoryDict[key].push(model.initBuffer(["storage", "copy_from", "copy_to"], buffer.size / 4, undefined));
        });
    }

    model.clearGradients();
}