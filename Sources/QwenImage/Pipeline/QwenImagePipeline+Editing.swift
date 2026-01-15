import Foundation
import Logging
import MLX

#if canImport(CoreGraphics)
import CoreGraphics
#endif

private struct EditReferenceContext {
  let referenceTokens: MLXArray
  let imageSegments: [(Int, Int, Int)]
  let targetSize: (width: Int, height: Int)
#if canImport(CoreGraphics)
  let visionConditionImages: [CGImage]
#endif
  let windowSequenceLengths: MLXArray?
  let cumulativeSequenceLengths: MLXArray?
  let rotaryCos: MLXArray?
  let rotarySin: MLXArray?
}

extension QwenImagePipeline {
#if canImport(CoreGraphics)
  public func generateEditedPixels(
    parameters: GenerationParameters,
    model: QwenModelConfiguration,
    referenceImage: CGImage,
    maxPromptLength: Int? = nil,
    seed: UInt64? = nil,
    progress: QwenProgressHandler? = nil
  ) async throws -> MLXArray {
    guard config == .imageEditing else {
      throw PipelineError.invalidConfiguration("generateEditedPixels requires QwenImagePipeline(config: .imageEditing)")
    }
    try ensureTokenizer()
    try ensureTextEncoder()

    let (scheduler, runtime) = QwenSchedulerFactory.flowMatchSchedulerRuntime(
      model: model,
      generation: parameters
    )
    let promptLength = min(maxPromptLength ?? model.maxSequenceLength, model.maxSequenceLength)
    let preferredDType = preferredWeightDType()
    let referenceContext = try prepareEditReferenceContext(
      referenceImage: referenceImage,
      runtime: runtime,
      dtype: preferredDType,
      referenceTargetArea: 1_048_576
    )
    let visionPromptContext = try prepareVisionPromptContext(
      referenceImages: [referenceImage],
      conditionImages: referenceContext.visionConditionImages
    )
    let guidanceEncoding = try encodeGuidancePromptsInternal(
      prompt: parameters.prompt,
      negativePrompt: parameters.negativePrompt,
      maxLength: promptLength,
      visionContext: visionPromptContext
    )
    let finalReferenceDType = preferredWeightDType()
    let referenceTokens = referenceContext.referenceTokens.asType(finalReferenceDType)
    pipelineLogger.debug("edit: reference tokens shape=\(referenceTokens.shape) segments=\(referenceContext.imageSegments)")

    MLX.eval(
      guidanceEncoding.unconditionalEmbeddings,
      guidanceEncoding.conditionalEmbeddings,
      guidanceEncoding.unconditionalMask,
      guidanceEncoding.conditionalMask
    )

    releaseEncoders()

    try ensureUNetAndVAE(model: model)

    let latents = try makeInitialLatents(
      height: runtime.height,
      width: runtime.width,
      sigmas: scheduler.sigmas,
      seed: seed
    )

    let finalLatents = try await runEditDenoiseLoop(
      latents: latents,
      parameters: parameters,
      runtime: runtime,
      scheduler: scheduler,
      unconditionalEmbeddings: guidanceEncoding.unconditionalEmbeddings,
      conditionalEmbeddings: guidanceEncoding.conditionalEmbeddings,
      unconditionalMask: guidanceEncoding.unconditionalMask,
      conditionalMask: guidanceEncoding.conditionalMask,
      referenceTokens: referenceTokens,
      imageSegments: referenceContext.imageSegments.isEmpty ? nil : referenceContext.imageSegments,
      progress: progress
    )

    var decoded = try decodeLatents(finalLatents)
    if decoded.ndim == 4 {
      decoded = decoded[0, 0..., 0..., 0...]
    }

    let targetWidth = parameters.width
    let targetHeight = parameters.height
    if (targetWidth != runtime.width || targetHeight != runtime.height) &&
       (targetWidth > 0 && targetHeight > 0) {
      decoded = QwenMLXImageResizer.resizeCHW(
        decoded,
        targetHeight: targetHeight,
        targetWidth: targetWidth
      )
    }

    return MLX.clip(decoded, min: 0, max: 1).asType(preferredWeightDType())
  }

  public func generateEditedPixels(
    parameters: GenerationParameters,
    model: QwenModelConfiguration,
    referenceImages: [CGImage],
    maxPromptLength: Int? = nil,
    seed: UInt64? = nil,
    progress: QwenProgressHandler? = nil
  ) async throws -> MLXArray {
    guard config == .imageEditing else {
      throw PipelineError.invalidConfiguration("generateEditedPixels requires QwenImagePipeline(config: .imageEditing)")
    }
    try ensureTokenizer()
    try ensureTextEncoder()

    precondition(!referenceImages.isEmpty, "At least one reference image required.")

    let (scheduler, runtime) = QwenSchedulerFactory.flowMatchSchedulerRuntime(
      model: model,
      generation: parameters
    )
    let promptLength = min(maxPromptLength ?? model.maxSequenceLength, model.maxSequenceLength)
    let limitedReferences = Array(referenceImages.prefix(2))
    let preferredDType = preferredWeightDType()
    let referenceContext = try prepareEditReferenceContext(
      referenceImages: limitedReferences,
      runtime: runtime,
      dtype: preferredDType,
      referenceTargetArea: 1_048_576
    )
    let visionPromptContext = try prepareVisionPromptContext(
      referenceImages: limitedReferences,
      conditionImages: referenceContext.visionConditionImages
    )
    let guidanceEncoding = try encodeGuidancePromptsInternal(
      prompt: parameters.prompt,
      negativePrompt: parameters.negativePrompt,
      maxLength: promptLength,
      visionContext: visionPromptContext
    )
    let finalReferenceDType = preferredWeightDType()
    let referenceTokens = referenceContext.referenceTokens.asType(finalReferenceDType)
    pipelineLogger.debug("edit: reference tokens shape=\(referenceTokens.shape) segments=\(referenceContext.imageSegments)")
    MLX.eval(
      guidanceEncoding.unconditionalEmbeddings,
      guidanceEncoding.conditionalEmbeddings,
      guidanceEncoding.unconditionalMask,
      guidanceEncoding.conditionalMask
    )

    releaseEncoders()

    try ensureUNetAndVAE(model: model)

    let latents = try makeInitialLatents(
      height: runtime.height,
      width: runtime.width,
      sigmas: scheduler.sigmas,
      seed: seed
    )

    let finalLatents = try await runEditDenoiseLoop(
      latents: latents,
      parameters: parameters,
      runtime: runtime,
      scheduler: scheduler,
      unconditionalEmbeddings: guidanceEncoding.unconditionalEmbeddings,
      conditionalEmbeddings: guidanceEncoding.conditionalEmbeddings,
      unconditionalMask: guidanceEncoding.unconditionalMask,
      conditionalMask: guidanceEncoding.conditionalMask,
      referenceTokens: referenceTokens,
      imageSegments: referenceContext.imageSegments,
      progress: progress
    )

    var decoded = try decodeLatents(finalLatents)
    if decoded.ndim == 4 {
      decoded = decoded[0, 0..., 0..., 0...]
    }

    let targetWidth = parameters.width
    let targetHeight = parameters.height
    if (targetWidth != runtime.width || targetHeight != runtime.height) &&
       (targetWidth > 0 && targetHeight > 0) {
      decoded = QwenMLXImageResizer.resizeCHW(
        decoded,
        targetHeight: targetHeight,
        targetWidth: targetWidth
      )
    }

    return MLX.clip(decoded, min: 0, max: 1).asType(preferredWeightDType())
  }

  func prepareVisionPromptContext(
    referenceImages: [CGImage],
    conditionImages: [CGImage]? = nil
  ) throws -> VisionPromptContext? {
    guard !referenceImages.isEmpty else {
      return nil
    }
    let inputs = conditionImages ?? referenceImages
    guard inputs.count == referenceImages.count else {
      throw PipelineError.invalidTensorShape("Condition image count \(inputs.count) does not match references \(referenceImages.count).")
    }
    visionLogger.debug("building context for \(inputs.count) reference image(s)")
    let mergeSize = visionConfiguration.spatialMergeSize
    let conditionMultiple = visionConfiguration.patchSize * visionConfiguration.spatialMergeSize
    let conditionTargetArea = 147_456
    var patchInputs: [MLXArray] = []
    var grids: [QwenVisionGrid] = []
    var tokenCounts: [Int] = []
    var gridTHWList: [(Int, Int, Int)] = []

    for (index, image) in inputs.enumerated() {
      visionLogger.debug("image[\(index)] condition size=\(image.width)x\(image.height)")
      let intermediateSize = EditSizing.computeVisionConditionDimensions(
        referenceWidth: image.width,
        referenceHeight: image.height,
        targetArea: conditionTargetArea,
        multiple: 32
      )
      let finalSize = try QwenVisionUtils.smartResize(
        height: intermediateSize.height,
        width: intermediateSize.width,
        factor: conditionMultiple,
        minPixels: visionPreprocessor.config.minPixels,
        maxPixels: visionPreprocessor.config.maxPixels
      )
      let processed = try visionPreprocessor.preprocess(
        cgImage: image,
        targetSize: (height: finalSize.height, width: finalSize.width),
        intermediateSize: intermediateSize
      )
      visionLogger.debug("image[\(index)] resized to \(processed.resizedSize.width)x\(processed.resizedSize.height)")
      visionLogger.debug("image[\(index)] patches shape=\(processed.patches.shape) grid=\(processed.grid)")
      guard processed.patches.ndim == 2 else {
        throw PipelineError.invalidTensorShape("Expected vision patches with rank 2 for reference image \(index).")
      }
      let tokens = processed.patches.dim(0)
      let patchVolume = processed.patches.dim(1)
      let batched = processed.patches.reshaped(1, tokens, patchVolume)
      patchInputs.append(batched)
      grids.append(processed.grid)

      guard processed.grid.height % mergeSize == 0, processed.grid.width % mergeSize == 0 else {
        throw PipelineError.invalidTensorShape("Vision grid dimensions must be divisible by merge size.")
      }
      let spatialH = processed.grid.height / mergeSize
      let spatialW = processed.grid.width / mergeSize
      let count = processed.grid.temporal * spatialH * spatialW
      tokenCounts.append(count)
      gridTHWList.append((processed.grid.temporal, processed.grid.height, processed.grid.width))
    }

    let patchVolume = patchInputs.first?.dim(2) ?? 0
    if patchInputs.contains(where: { $0.dim(2) != patchVolume }) {
      throw PipelineError.invalidTensorShape("Mismatched patch volumes across reference images.")
    }

    let maxTokens = patchInputs.map { $0.dim(1) }.max() ?? 0
    let paddedInputs: [MLXArray]
    if maxTokens > 0 && patchInputs.contains(where: { $0.dim(1) != maxTokens }) {
      paddedInputs = patchInputs.map { patches in
        let tokens = patches.dim(1)
        guard tokens < maxTokens else { return patches }
        let pad = MLX.zeros([1, maxTokens - tokens, patchVolume], dtype: patches.dtype)
        return MLX.concatenated([patches, pad], axis: 1)
      }
    } else {
      paddedInputs = patchInputs
    }

    let stackedPatches = paddedInputs.count == 1
      ? paddedInputs[0]
      : MLX.concatenated(paddedInputs, axis: 0)
    visionLogger.debug("stacked patch input shape=\(stackedPatches.shape)")
    return VisionPromptContext(
      patchInputs: stackedPatches,
      grids: grids,
      tokenCounts: tokenCounts,
      gridTHW: gridTHWList
    )
  }

  private func prepareEditReferenceContext(
    referenceImage: CGImage,
    runtime: QwenRuntimeConfig,
    dtype: DType,
    referenceTargetArea: Int
  ) throws -> EditReferenceContext {
    pipelineLogger.debug("edit: reference input size=\(referenceImage.width)x\(referenceImage.height)")
    let target = EditSizing.computeVAEDimensions(
      referenceWidth: referenceImage.width,
      referenceHeight: referenceImage.height,
      targetArea: referenceTargetArea
    )
    pipelineLogger.debug("edit: resized reference to \(target.width)x\(target.height)")
    let pixels = try QwenImageIO.resizedPixelArray(
      from: referenceImage,
      width: target.width,
      height: target.height
    )
    pipelineLogger.debug("edit: reference pixel array shape=\(pixels.shape) dtype=\(pixels.dtype)")
    let (latents, _, _) = try encodePixelsWithIntermediates(pixels)
    pipelineLogger.debug("edit: encoded latents shape=\(latents.shape)")
    let packed = LatentUtilities.packLatents(
      latents,
      height: target.height,
      width: target.width
    ).asType(dtype)
    let packedBatch = packed.reshaped(1, packed.dim(1), packed.dim(2))
    let referenceTokens = MLX.concatenated([packedBatch, packedBatch], axis: 0)
    let latentPatchHeight = max(1, runtime.height / 16)
    let latentPatchWidth = max(1, runtime.width / 16)
    let referencePatchHeight = max(1, target.height / 16)
    let referencePatchWidth = max(1, target.width / 16)
    let segments: [(Int, Int, Int)] = [
      (1, latentPatchHeight, latentPatchWidth),
      (1, referencePatchHeight, referencePatchWidth)
    ]
    pipelineLogger.debug("edit: image segments=\(segments)")
    return EditReferenceContext(
      referenceTokens: referenceTokens,
      imageSegments: segments,
      targetSize: (target.width, target.height),
      visionConditionImages: [referenceImage],
      windowSequenceLengths: nil,
      cumulativeSequenceLengths: nil,
      rotaryCos: nil,
      rotarySin: nil
    )
  }

  private func prepareEditReferenceContext(
    referenceImages: [CGImage],
    runtime: QwenRuntimeConfig,
    dtype: DType,
    referenceTargetArea: Int
  ) throws -> EditReferenceContext {
    precondition(!referenceImages.isEmpty, "At least one reference image required.")

    let referenceTargetArea = 147_456
    var segments: [(Int, Int, Int)] = [
      (1, max(1, runtime.height / 16), max(1, runtime.width / 16))
    ]
    var perImagePacked: [MLXArray] = []
    perImagePacked.reserveCapacity(referenceImages.count)
    var lastTargetSize: (width: Int, height: Int) = (runtime.width, runtime.height)
    var visionConditionImages: [CGImage] = []
    visionConditionImages.reserveCapacity(referenceImages.count)

    for (idx, image) in referenceImages.prefix(2).enumerated() {
      pipelineLogger.debug("edit: reference[\(idx)] input size=\(image.width)x\(image.height)")
      visionConditionImages.append(image)
      let target = EditSizing.computeVAEDimensions(
        referenceWidth: image.width,
        referenceHeight: image.height,
        targetArea: referenceTargetArea
      )
      lastTargetSize = target
      let packedHeight = max(1, target.height / 16)
      let packedWidth = max(1, target.width / 16)
      let pixels = try QwenImageIO.resizedPixelArray(
        from: image,
        width: target.width,
        height: target.height
      )
      let (latents, _, _) = try encodePixelsWithIntermediates(pixels)
      let packed = LatentUtilities.packLatents(
        latents,
        height: target.height,
        width: target.width
      ).asType(dtype)
      let packedBatch = packed.reshaped(1, packed.dim(1), packed.dim(2))
      perImagePacked.append(packedBatch)
      segments.append((1, packedHeight, packedWidth))
    }

    let combinedSingle = MLX.concatenated(perImagePacked, axis: 1)
    let referenceTokens = MLX.concatenated([combinedSingle, combinedSingle], axis: 0)
    pipelineLogger.debug("edit: image segments=\(segments)")
    return EditReferenceContext(
      referenceTokens: referenceTokens,
      imageSegments: segments,
      targetSize: (lastTargetSize.width, lastTargetSize.height),
      visionConditionImages: visionConditionImages,
      windowSequenceLengths: nil,
      cumulativeSequenceLengths: nil,
      rotaryCos: nil,
      rotarySin: nil
    )
  }
#endif

  func runEditDenoiseLoop(
    latents initialLatents: MLXArray,
    parameters: GenerationParameters,
    runtime: QwenRuntimeConfig,
    scheduler: QwenFlowMatchScheduler,
    unconditionalEmbeddings rawUncondEmbeddings: MLXArray,
    conditionalEmbeddings rawCondEmbeddings: MLXArray,
    unconditionalMask rawUncondMask: MLXArray,
    conditionalMask rawCondMask: MLXArray,
    referenceTokens rawReferenceTokens: MLXArray,
    imageSegments: [(Int, Int, Int)]?,
    progress: QwenProgressHandler?
  ) async throws -> MLXArray {
    guard let unet else {
      throw PipelineError.componentNotLoaded("UNet")
    }
    var latents = initialLatents
    let unconditionalEmbeddings = rawUncondEmbeddings.asType(latents.dtype)
    let conditionalEmbeddings = rawCondEmbeddings.asType(latents.dtype)
    let unconditionalMask = rawUncondMask.asType(.int32)
    let conditionalMask = rawCondMask.asType(.int32)
    var referenceTokens = rawReferenceTokens.asType(latents.dtype)

    if referenceTokens.dim(0) == 1 {
      referenceTokens = MLX.concatenated([referenceTokens, referenceTokens], axis: 0)
    }
    precondition(referenceTokens.dim(0) >= 2, "Reference tokens must include unconditional and conditional rows.")

    let referenceTokensUncond = referenceTokens[0..<1, 0..., 0...]
    let referenceTokensCond = referenceTokens[1..<2, 0..., 0...]
    let extraSegments = imageSegments ?? []
    if !extraSegments.isEmpty {
      pipelineLogger.debug("edit: img_segments=\(extraSegments)")
    }

    let latentHeight = runtime.height / 16
    let latentWidth = runtime.width / 16
    let latentSegment = (1, latentHeight, latentWidth)
    let segments: [(Int, Int, Int)]
    if let extra = imageSegments, !extra.isEmpty {
      if let first = extra.first, first == latentSegment {
        segments = extra
      } else {
        segments = [latentSegment] + extra
      }
    } else {
      segments = [latentSegment]
    }
    let textLen = unconditionalEmbeddings.dim(1)
    let textLengths = Array(repeating: textLen, count: unconditionalEmbeddings.dim(0))
    let rope = QwenEmbedRope(theta: 10_000, axesDimensions: [16, 56, 56], scaleRope: true)
    let precomputedRoPE = rope(videoSegments: segments, textSequenceLengths: textLengths)

    var tokenCountCached: Int? = nil
    for step in 0..<parameters.steps {
      try Task.checkCancellation()
      let scaledLatents = scheduler.scaleModelInput(latents, timestep: step).asType(latents.dtype)
      let latentTokensSingle = LatentUtilities.packLatents(
        scaledLatents,
        height: runtime.height,
        width: runtime.width
      ).asType(referenceTokens.dtype)
      if step == 0 {
        pipelineLogger.debug("edit: latents tokens=\(latentTokensSingle.shape) ref(uncond)=\(referenceTokensUncond.shape) ref(cond)=\(referenceTokensCond.shape)")
        if pipelineLogger.logLevel <= .debug {
          let cMaskI32 = conditionalMask.asType(.int32)
          let cLensArr = MLX.sum(cMaskI32, axis: 1)
          MLX.eval(cLensArr)
          let uLens = textLengths
          let cLens = (0..<cLensArr.dim(0)).map { cLensArr[$0].item(Int.self) }
          pipelineLogger.debug("encode(uncond) shape=\(unconditionalEmbeddings.shape) mask=\(unconditionalMask.shape) txt_seq_lens=\(uLens)")
          pipelineLogger.debug("encode(cond) shape=\(conditionalEmbeddings.shape) mask=\(conditionalMask.shape) txt_seq_lens=\(cLens)")
        }
      }

      let segmentsArgument = extraSegments.isEmpty ? nil : extraSegments

      let tokenCount: Int
      if let cached = tokenCountCached {
        tokenCount = cached
      } else {
        tokenCount = latentTokensSingle.dim(1)
        tokenCountCached = tokenCount
      }

      let combinedCond = MLX.concatenated([latentTokensSingle, referenceTokensCond], axis: 1)
      if step == 0 {
        let latTok = latentTokensSingle.dim(1)
        let refTok = referenceTokensCond.dim(1)
        pipelineLogger.debug("combined(cond)=\(combinedCond.shape) lat_tokens=\(latTok) ref_tokens=\(refTok)")
      }
      pipelineLogger.debug("edit: forward(cond) step=\(step)")
      let noiseTokensCond = try unet.forwardTokens(
        timestepIndex: step,
        runtimeConfig: runtime,
        latentTokens: combinedCond,
        encoderHiddenStates: conditionalEmbeddings,
        encoderHiddenStatesMask: conditionalMask,
        imageSegments: segmentsArgument,
        precomputedImageRotaryEmbeddings: precomputedRoPE
      )
      pipelineLogger.debug("edit: forward(cond) done step=\(step)")
      let truncatedCond = noiseTokensCond[0..., 0..<tokenCount, 0...].asType(.float32)
      let conditionalNoise = LatentUtilities.unpackLatents(
        truncatedCond,
        height: runtime.height,
        width: runtime.width
      )

      let combinedUncond = MLX.concatenated([latentTokensSingle, referenceTokensUncond], axis: 1)
      pipelineLogger.debug("edit: forward(uncond) step=\(step)")
      let noiseTokensUncond = try unet.forwardTokens(
        timestepIndex: step,
        runtimeConfig: runtime,
        latentTokens: combinedUncond,
        encoderHiddenStates: unconditionalEmbeddings,
        encoderHiddenStatesMask: unconditionalMask,
        imageSegments: segmentsArgument,
        precomputedImageRotaryEmbeddings: precomputedRoPE
      )
      pipelineLogger.debug("edit: forward(uncond) done step=\(step)")
      let truncatedUncond = noiseTokensUncond[0..., 0..<tokenCount, 0...].asType(.float32)
      let unconditionalNoise = LatentUtilities.unpackLatents(
        truncatedUncond,
        height: runtime.height,
        width: runtime.width
      )

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

    return latents
  }

  func ensureVisionTower() throws -> QwenVisionTower {
    if let tower = visionTower {
      return tower
    }
    let config = visionConfiguration
    let tower = QwenVisionTower(configuration: config)
    let dtype = preferredWeightDType()
    guard let directory = transformerDirectory ?? baseWeightsDirectory else {
      throw PipelineError.componentNotLoaded("TransformerWeightsDirectory")
    }
    if textEncoderQuantization == nil {
      let textConfigURL = directory.appending(path: "text_encoder").appending(path: "config.json")
      textEncoderQuantization = QwenQuantizationPlan.load(from: textConfigURL)
    }
    try weightsLoader.loadVisionTower(
      fromDirectory: directory,
      into: tower,
      dtype: dtype,
      quantization: textEncoderQuantization
    )
    visionTower = tower
    textEncoder?.setVisionTower(tower)
    applyRuntimeQuantizationIfNeeded(to: tower, flag: &visionRuntimeQuantized)
    return tower
  }
}

