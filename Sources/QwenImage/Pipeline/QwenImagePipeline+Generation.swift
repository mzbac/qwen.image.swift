import MLX
import MLXRandom

extension QwenImagePipeline {
  public func generatePixels(
    parameters: GenerationParameters,
    model: QwenModelConfiguration,
    maxPromptLength: Int? = nil,
    seed: UInt64? = nil,
    progress: QwenProgressHandler? = nil
  ) async throws -> MLXArray {
    let (scheduler, runtime) = QwenSchedulerFactory.flowMatchSchedulerRuntime(
      model: model,
      generation: parameters
    )

    let promptLength = min(maxPromptLength ?? model.maxSequenceLength, model.maxSequenceLength)

    try ensureTokenizer()
    try ensureTextEncoder()
    let guidanceEncoding = try encodeGuidancePrompts(
      prompt: parameters.prompt,
      negativePrompt: parameters.negativePrompt,
      maxLength: promptLength
    )

    let stacked = guidanceEncoding.stackedEmbeddings()
    let embeddings = stacked.embeddings.asType(preferredWeightDType())
    let attentionMask = stacked.attentionMask.asType(.int32)

    MLX.eval(embeddings, attentionMask)

    releaseTextEncoder()

    try ensureUNetAndVAE(model: model)

    var latents = try makeInitialLatents(
      height: runtime.height,
      width: runtime.width,
      sigmas: scheduler.sigmas,
      seed: seed
    )

    for step in 0..<parameters.steps {
      try Task.checkCancellation()
      let stackedLatents = GuidanceUtilities.stackLatentsForGuidance(latents)
      let modelInput = scheduler.scaleModelInput(stackedLatents, timestep: step).asType(latents.dtype)
      let noisePred = try denoiseLatents(
        timestepIndex: step,
        runtimeConfig: runtime,
        latentImages: modelInput,
        encoderHiddenStates: embeddings,
        encoderHiddenStatesMask: attentionMask
      )
      let (unconditionalNoise, conditionalNoise) = GuidanceUtilities.splitGuidanceLatents(noisePred)

      let guided: MLXArray
      if let trueCFGScale = parameters.trueCFGScale, trueCFGScale > 1 {
        var combined = unconditionalNoise + trueCFGScale * (conditionalNoise - unconditionalNoise)
        let axis = combined.ndim - 1
        let condSquared = MLX.sum(conditionalNoise * conditionalNoise, axes: [axis], keepDims: true)
        let combSquared = MLX.sum(combined * combined, axes: [axis], keepDims: true)
        let epsilon = MLX.ones(condSquared.shape, dtype: condSquared.dtype) * Float32(1e-6)
        let conditionalNorm = MLX.sqrt(MLX.maximum(condSquared, epsilon))
        let combinedNorm = MLX.sqrt(MLX.maximum(combSquared, epsilon))
        let ratio = conditionalNorm / combinedNorm
        combined = combined * ratio
        guided = combined.asType(latents.dtype)
      } else {
        guided = GuidanceUtilities.applyClassifierFreeGuidance(
          unconditional: unconditionalNoise,
          conditional: conditionalNoise,
          guidanceScale: parameters.guidanceScale
        ).asType(latents.dtype)
      }

      latents = scheduler.step(modelOutput: guided, timestep: step, sample: latents)
      if progress != nil {
        MLX.eval(latents)
      }
      progress?(ProgressInfo(step: step + 1, total: parameters.steps))
      await Task.yield()
    }

    let decoded = try decodeLatents(latents)
    return MLX.clip(decoded, min: 0, max: 1).asType(preferredWeightDType())
  }

  // MARK: - Policy-Free Generation API

  public func generatePixels(
    parameters: GenerationParameters,
    model: QwenModelConfiguration,
    guidanceEncoding: QwenGuidanceEncoding,
    seed: UInt64? = nil,
    progress: QwenProgressHandler? = nil
  ) async throws -> MLXArray {
    let (scheduler, runtime) = QwenSchedulerFactory.flowMatchSchedulerRuntime(
      model: model,
      generation: parameters
    )

    let stacked = guidanceEncoding.stackedEmbeddings()
    let embeddings = stacked.embeddings.asType(preferredWeightDType())
    let attentionMask = stacked.attentionMask.asType(.int32)

    MLX.eval(embeddings, attentionMask)

    releaseTextEncoder()

    try ensureUNetAndVAE(model: model)

    var latents = try makeInitialLatents(
      height: runtime.height,
      width: runtime.width,
      sigmas: scheduler.sigmas,
      seed: seed
    )

    for step in 0..<parameters.steps {
      try Task.checkCancellation()
      let stackedLatents = GuidanceUtilities.stackLatentsForGuidance(latents)
      let modelInput = scheduler.scaleModelInput(stackedLatents, timestep: step).asType(latents.dtype)
      let noisePred = try denoiseLatents(
        timestepIndex: step,
        runtimeConfig: runtime,
        latentImages: modelInput,
        encoderHiddenStates: embeddings,
        encoderHiddenStatesMask: attentionMask
      )
      let (unconditionalNoise, conditionalNoise) = GuidanceUtilities.splitGuidanceLatents(noisePred)

      let guided: MLXArray
      if let trueCFGScale = parameters.trueCFGScale, trueCFGScale > 1 {
        var combined = unconditionalNoise + trueCFGScale * (conditionalNoise - unconditionalNoise)
        let axis = combined.ndim - 1
        let condSquared = MLX.sum(conditionalNoise * conditionalNoise, axes: [axis], keepDims: true)
        let combSquared = MLX.sum(combined * combined, axes: [axis], keepDims: true)
        let epsilon = MLX.ones(condSquared.shape, dtype: condSquared.dtype) * Float32(1e-6)
        let conditionalNorm = MLX.sqrt(MLX.maximum(condSquared, epsilon))
        let combinedNorm = MLX.sqrt(MLX.maximum(combSquared, epsilon))
        let ratio = conditionalNorm / combinedNorm
        combined = combined * ratio
        guided = combined.asType(latents.dtype)
      } else {
        guided = GuidanceUtilities.applyClassifierFreeGuidance(
          unconditional: unconditionalNoise,
          conditional: conditionalNoise,
          guidanceScale: parameters.guidanceScale
        ).asType(latents.dtype)
      }

      latents = scheduler.step(modelOutput: guided, timestep: step, sample: latents)
      if progress != nil {
        MLX.eval(latents)
      }
      progress?(ProgressInfo(step: step + 1, total: parameters.steps))
      await Task.yield()
    }

    let decoded = try decodeLatents(latents)
    return MLX.clip(decoded, min: 0, max: 1).asType(preferredWeightDType())
  }

  func makeInitialLatents(
    height: Int,
    width: Int,
    sigmas: MLXArray,
    seed: UInt64?
  ) throws -> MLXArray {
    let latentHeight = max(1, height / 8)
    let latentWidth = max(1, width / 8)
    let dtype = preferredWeightDType()
    let latents: MLXArray
    if let seed {
      let total = 1 * 16 * latentHeight * latentWidth
      var generator = PhiloxNormalGenerator(seed: seed)
      let values = generator.generate(count: total)
      latents = MLXArray(values, [1, 16, latentHeight, latentWidth]).asType(dtype)
    } else {
      latents = MLXRandom.normal(
        [1, 16, latentHeight, latentWidth],
        dtype: dtype
      )
    }
    let sigmaValue = sigmas[0].asType(dtype)
    return latents * sigmaValue
  }
}
