import Foundation
import Logging
import MLX
import MLXNN
import MLXRandom

#if canImport(CoreGraphics)
import CoreGraphics
#endif

// MARK: - Layered Pipeline Error

public enum LayeredPipelineError: Error {
  case componentsNotLoaded
  case textEncoderNotLoaded
  case tokenizerNotLoaded
  case invalidImageShape
  case invalidParameters(String)
}

// MARK: - Layered Prompt Encoding

public struct LayeredPromptEncoding {
  public let embeddings: MLXArray

  public let mask: MLXArray

  public var sequenceLength: Int {
    embeddings.dim(1)
  }

  public init(embeddings: MLXArray, mask: MLXArray) {
    self.embeddings = embeddings
    self.mask = mask
  }
}

// MARK: - Layered Generation Pipeline

public class QwenLayeredPipeline {
  private var logger = Logger(label: "qwen.image.layered")

  public var vae: QwenVAE?
  public var transformer: QwenLayeredTransformerV2?
  public var textEncoder: QwenTextEncoder?
  public var tokenizer: QwenTokenizer?

  public let vaeScaleFactor: Int = 8
  public let latentChannels: Int = 16
  public let patchSize: Int = 2

  // MARK: - Weights Metadata (for lazy reloads)

  private var baseWeightsDirectory: URL?
  private var weightsDType: DType = .bfloat16
  private var flowMatchConfiguration: QwenFlowMatchConfig?
  private var textEncoderQuantization: QwenQuantizationPlan?

  public init() {
    logger.info("QwenLayeredPipeline initialized")
  }

  public static func load(from path: URL, dtypeOverride: DType? = nil) async throws -> QwenLayeredPipeline {
    let pipeline = QwenLayeredPipeline()
    let loader = QwenWeightsLoader()

    pipeline.logger.info("Loading layered pipeline from \(path.path)")
    pipeline.baseWeightsDirectory = path
    pipeline.flowMatchConfiguration = try QwenFlowMatchConfig.load(fromSchedulerDirectory: path.appending(path: "scheduler"))
    var textEncoderConfiguration = try QwenTextEncoderConfiguration.load(from: path.appending(path: "text_encoder"))
    let resolvedDType = dtypeOverride ?? textEncoderConfiguration.outputDType
    textEncoderConfiguration.outputDType = resolvedDType
    pipeline.weightsDType = resolvedDType

    pipeline.logger.info("Loading VAE...")
    pipeline.vae = QwenVAE()
    try loader.loadVAE(fromDirectory: path, into: pipeline.vae!, dtype: resolvedDType)

    pipeline.logger.info("Loading transformer...")
    let transformerPath = path.appending(path: "transformer")
    let transformerConfig = try QwenLayeredTransformerConfiguration.load(from: transformerPath)
    let transformerQuantization = QwenQuantizationPlanResolver.resolve(
      root: path,
      configRelativePath: "transformer/config.json",
      componentName: "transformer",
      logger: pipeline.logger
    )
    pipeline.transformer = QwenLayeredTransformerV2(transformerConfig)
    try loader.loadTransformerForLayered(
      fromDirectory: path,
      into: pipeline.transformer!,
      dtype: resolvedDType,
      quantization: transformerQuantization
    )
    if let transformerQuantization, transformerQuantization.isEnabled {
      let spec = transformerQuantization.defaultSpec ?? transformerQuantization.prepackedLayers.values.first?.spec
      if let spec {
        pipeline.logger.info(
          "Applied quantization to layered transformer (bits=\(spec.bits), group=\(spec.groupSize), mode=\(spec.mode))."
        )
      } else {
        pipeline.logger.info("Applied quantization to layered transformer using provided manifest.")
      }
    }

    pipeline.logger.info("Loading text encoder...")
    let textEncoderQuantization = QwenQuantizationPlanResolver.resolve(
      root: path,
      configRelativePath: "text_encoder/config.json",
      componentName: "text_encoder",
      logger: pipeline.logger
    )
    pipeline.textEncoderQuantization = textEncoderQuantization
    pipeline.textEncoder = QwenTextEncoder(configuration: textEncoderConfiguration)
    try loader.loadTextEncoder(
      fromDirectory: path,
      into: pipeline.textEncoder!,
      dtype: resolvedDType,
      quantization: textEncoderQuantization
    )
    if let textEncoderQuantization, textEncoderQuantization.isEnabled,
       let spec = textEncoderQuantization.defaultSpec ?? textEncoderQuantization.prepackedLayers.values.first?.spec {
      pipeline.logger.info(
        "Applied quantization to layered text encoder (bits=\(spec.bits), group=\(spec.groupSize), mode=\(spec.mode))."
      )
    }

    pipeline.logger.info("Loading tokenizer...")
    let tokenizerPath = path.appending(path: "tokenizer")
    let textEncoderPath = path.appending(path: "text_encoder")

    if FileManager.default.fileExists(atPath: tokenizerPath.path) {
      pipeline.tokenizer = try QwenTokenizer.load(from: tokenizerPath)
    } else if FileManager.default.fileExists(atPath: textEncoderPath.path) {
      pipeline.tokenizer = try QwenTokenizer.load(from: textEncoderPath)
    } else {
      pipeline.tokenizer = try QwenTokenizer.load(from: path)
    }

    pipeline.logger.info("Layered pipeline loaded successfully")
    return pipeline
  }

  // MARK: - LoRA Support

  public func applyLora(from fileURL: URL, scale: Float = 1.0) throws {
    guard let transformer = transformer else {
      throw LayeredPipelineError.componentsNotLoaded
    }
    _ = try QwenLoRAApplier.apply(
      from: fileURL,
      scale: scale,
      computeDType: weightsDType,
      targets: [("layered transformer", transformer)],
      logger: logger,
      emptyError: nil
    )
  }

  public func generate(
    image: MLXArray,
    parameters: LayeredGenerationParameters,
    progress: ((Int, Int, Float) -> Void)? = nil
  ) async throws -> [MLXArray] {
    guard let vae = vae, let transformer = transformer else {
      throw LayeredPipelineError.componentsNotLoaded
    }

    logger.info("Starting layer generation with \(parameters.layers) layers")

    if let seed = parameters.seed {
      MLXRandom.seed(seed)
    }

    let batchSize = image.dim(0)
    let layers = parameters.layers

    let inputHeight = image.dim(-2)
    let inputWidth = image.dim(-1)
    let aspectRatio = Float(inputWidth) / Float(inputHeight)
    let (targetWidth, targetHeight) = calculateLayeredDimensions(
      resolution: parameters.resolution,
      aspectRatio: aspectRatio
    )

    logger.info("Input: \(inputWidth)x\(inputHeight), target: \(targetWidth)x\(targetHeight)")

    var resizedImage = image
    logger.debug("Input image shape: \(image.shape), ndim: \(image.ndim)")
    if image.ndim == 5 {
      resizedImage = image.squeezed(axis: 2)
      logger.debug("After squeeze: \(resizedImage.shape)")
    }
    if inputWidth != targetWidth || inputHeight != targetHeight {
      resizedImage = QwenMLXImageResizer.resizeNCHW(resizedImage, targetHeight: targetHeight, targetWidth: targetWidth)
      logger.debug("After resize: \(resizedImage.shape)")
    }

    let height = targetHeight
    let width = targetWidth

    logger.debug("Encoding image to latent space, shape: \(resizedImage.shape)")
    let (rawLatent, _, _) = vae.encodeRaw(resizedImage)
    let imageLatent = QwenVAE.normalizeLatent(rawLatent)

    let latentHeight = height / vaeScaleFactor
    let latentWidth = width / vaeScaleFactor

    let halfH = latentHeight / patchSize
    let halfW = latentWidth / patchSize
    let flowMatchConfig = flowMatchConfiguration ?? .init()
    let scheduler = QwenSchedulerFactory.flowMatchSchedulerForLayered(
      numInferenceSteps: parameters.numInferenceSteps,
      width: width,
      height: height,
      flowMatchConfig: flowMatchConfig
    )

    let noiseShape = [batchSize, layers + 1, latentChannels, latentHeight, latentWidth]
    let layerNoise = MLXRandom.normal(noiseShape).asType(imageLatent.dtype)
    var latents = LatentPacking.pack(layerNoise)
    logger.debug("Noise latents shape: \(latents.shape)")

    let imageLatent4D: MLXArray
    if imageLatent.ndim == 5 {
      imageLatent4D = imageLatent.squeezed(axis: 2)
    } else {
      imageLatent4D = imageLatent
    }
    logger.debug("Image latent 4D shape: \(imageLatent4D.shape)")
    let packedImageLatent = LatentPacking.packSingle(imageLatent4D)
    logger.debug("Packed image latent shape: \(packedImageLatent.shape)")

    let promptText = parameters.prompt ?? ""
    let encoded = try encodePrompt(promptText, dtype: imageLatent.dtype)
    let promptEmbeds = encoded.embeddings
    let promptMask = encoded.mask
    let txtSeqLens = [Int](repeating: promptEmbeds.dim(1), count: batchSize)

    var imgShapes: [[(Int, Int, Int)]] = []
    for _ in 0..<batchSize {
      var shapes: [(Int, Int, Int)] = []
      for _ in 0..<(layers + 2) {
        shapes.append((1, halfH, halfW))
      }
      imgShapes.append(shapes)
    }

    let additionalTCond = MLXArray.zeros([batchSize]).asType(.int32)

    let doTrueCFG = parameters.trueCFGScale > 1.0 && parameters.negativePrompt != nil
    var negativePromptEmbeds: MLXArray? = nil
    var negativePromptMask: MLXArray? = nil
    var negativeTxtSeqLens: [Int]? = nil

    if doTrueCFG && textEncoder != nil && tokenizer != nil {
      let negPrompt = parameters.negativePrompt ?? ""
      let negEncoded = try encodePrompt(negPrompt, dtype: imageLatent.dtype)
      negativePromptEmbeds = negEncoded.embeddings
      negativePromptMask = negEncoded.mask
      negativeTxtSeqLens = [Int](repeating: negEncoded.embeddings.dim(1), count: batchSize)
    }

    if let negEmbeds = negativePromptEmbeds, let negMask = negativePromptMask {
      MLX.eval(promptEmbeds, promptMask, negEmbeds, negMask)
    } else {
      MLX.eval(promptEmbeds, promptMask)
    }

    logger.info("Starting denoising with \(parameters.numInferenceSteps) steps")

    for i in 0..<parameters.numInferenceSteps {
      try Task.checkCancellation()
      let latentInput = MLX.concatenated([latents, packedImageLatent], axis: 1)
      let sigma = scheduler.sigmas[i]
      let timestepExpanded = MLX.full([batchSize], values: sigma).asType(latents.dtype)

      var noisePred = transformer(
        hiddenStates: latentInput,
        encoderHiddenStates: promptEmbeds,
        encoderHiddenStatesMask: promptMask,
        timestep: timestepExpanded,
        imgShapes: imgShapes,
        txtSeqLens: txtSeqLens,
        additionalTCond: additionalTCond
      )

      noisePred = noisePred[0..., 0..<latents.dim(1), 0...]

      if doTrueCFG, let negEmbeds = negativePromptEmbeds, let negMask = negativePromptMask, let negTxtSeqLens = negativeTxtSeqLens {
        var negNoisePred = transformer(
          hiddenStates: latentInput,
          encoderHiddenStates: negEmbeds,
          encoderHiddenStatesMask: negMask,
          timestep: timestepExpanded,
          imgShapes: imgShapes,
          txtSeqLens: negTxtSeqLens,
          additionalTCond: additionalTCond
        )
        negNoisePred = negNoisePred[0..., 0..<latents.dim(1), 0...]

        let scale = parameters.trueCFGScale
        var combPred = negNoisePred + scale * (noisePred - negNoisePred)

        if parameters.cfgNormalize {
          let condNorm = MLX.sqrt(MLX.sum(MLX.multiply(noisePred, noisePred), axis: -1, keepDims: true))
          let noiseNorm = MLX.sqrt(MLX.sum(MLX.multiply(combPred, combPred), axis: -1, keepDims: true))
          let eps = MLXArray(Float(1e-8))
          combPred = MLX.multiply(combPred, MLX.divide(condNorm, noiseNorm + eps))
        }

        noisePred = combPred
      }

      latents = scheduler.step(modelOutput: noisePred, timestep: i, sample: latents)
      if progress != nil {
        MLX.eval(latents)
      }

      let progressFraction = Float(i + 1) / Float(parameters.numInferenceSteps)
      progress?(i, parameters.numInferenceSteps, progressFraction)
      await Task.yield()
    }

    logger.info("Decoding layers")
    var unpackedLatents = LatentPacking.unpack(
      latents,
      layers: layers,
      height: latentHeight,
      width: latentWidth
    )

    unpackedLatents = QwenVAE.denormalizeLatent(unpackedLatents)

    var decodedLayers: [MLXArray] = []
    for l in 1...layers {
      let layerLatent = unpackedLatents[0..., l, 0..., 0..., 0...]
      let decoded = vae.decode(layerLatent)
      decodedLayers.append(decoded)
    }

    logger.info("Generation complete, produced \(decodedLayers.count) layers")
    return decodedLayers
  }

  // MARK: - Policy-Free Generation API

  public func encodePromptToEncoding(_ prompt: String, dtype: DType) throws -> LayeredPromptEncoding {
    let encoded = try encodePrompt(prompt, dtype: dtype)
    return LayeredPromptEncoding(embeddings: encoded.embeddings, mask: encoded.mask)
  }

  public func generate(
    image: MLXArray,
    parameters: LayeredGenerationParameters,
    promptEncoding: LayeredPromptEncoding,
    negativePromptEncoding: LayeredPromptEncoding? = nil,
    progress: ((Int, Int, Float) -> Void)? = nil
  ) async throws -> [MLXArray] {
    guard let vae = vae, let transformer = transformer else {
      throw LayeredPipelineError.componentsNotLoaded
    }

    logger.info("Starting layer generation with \(parameters.layers) layers (using precomputed encodings)")

    if let seed = parameters.seed {
      MLXRandom.seed(seed)
    }

    let batchSize = image.dim(0)
    let layers = parameters.layers

    let inputHeight = image.dim(-2)
    let inputWidth = image.dim(-1)
    let aspectRatio = Float(inputWidth) / Float(inputHeight)
    let (targetWidth, targetHeight) = calculateLayeredDimensions(
      resolution: parameters.resolution,
      aspectRatio: aspectRatio
    )

    logger.info("Input: \(inputWidth)x\(inputHeight), target: \(targetWidth)x\(targetHeight)")

    var resizedImage = image
    if image.ndim == 5 {
      resizedImage = image.squeezed(axis: 2)
    }
    if inputWidth != targetWidth || inputHeight != targetHeight {
      resizedImage = QwenMLXImageResizer.resizeNCHW(resizedImage, targetHeight: targetHeight, targetWidth: targetWidth)
    }

    let height = targetHeight
    let width = targetWidth

    logger.debug("Encoding image to latent space, shape: \(resizedImage.shape)")
    let (rawLatent, _, _) = vae.encodeRaw(resizedImage)
    let imageLatent = QwenVAE.normalizeLatent(rawLatent)

    let latentHeight = height / vaeScaleFactor
    let latentWidth = width / vaeScaleFactor

    let halfH = latentHeight / patchSize
    let halfW = latentWidth / patchSize
    let flowMatchConfig = flowMatchConfiguration ?? .init()
    let scheduler = QwenSchedulerFactory.flowMatchSchedulerForLayered(
      numInferenceSteps: parameters.numInferenceSteps,
      width: width,
      height: height,
      flowMatchConfig: flowMatchConfig
    )

    let noiseShape = [batchSize, layers + 1, latentChannels, latentHeight, latentWidth]
    let layerNoise = MLXRandom.normal(noiseShape).asType(imageLatent.dtype)
    var latents = LatentPacking.pack(layerNoise)

    let imageLatent4D: MLXArray
    if imageLatent.ndim == 5 {
      imageLatent4D = imageLatent.squeezed(axis: 2)
    } else {
      imageLatent4D = imageLatent
    }
    let packedImageLatent = LatentPacking.packSingle(imageLatent4D)

    let promptEmbeds = promptEncoding.embeddings
    let promptMask = promptEncoding.mask
    let txtSeqLens = [Int](repeating: promptEncoding.sequenceLength, count: batchSize)

    var imgShapes: [[(Int, Int, Int)]] = []
    for _ in 0..<batchSize {
      var shapes: [(Int, Int, Int)] = []
      for _ in 0..<(layers + 2) {
        shapes.append((1, halfH, halfW))
      }
      imgShapes.append(shapes)
    }

    let additionalTCond = MLXArray.zeros([batchSize]).asType(.int32)

    let doTrueCFG = parameters.trueCFGScale > 1.0 && negativePromptEncoding != nil
    let negativePromptEmbeds = negativePromptEncoding?.embeddings
    let negativePromptMask = negativePromptEncoding?.mask
    let negativeTxtSeqLens: [Int]? = negativePromptEncoding.map {
      [Int](repeating: $0.sequenceLength, count: batchSize)
    }

    if let negEmbeds = negativePromptEmbeds, let negMask = negativePromptMask {
      MLX.eval(promptEmbeds, promptMask, negEmbeds, negMask)
    } else {
      MLX.eval(promptEmbeds, promptMask)
    }

    logger.info("Starting denoising with \(parameters.numInferenceSteps) steps")

    for i in 0..<parameters.numInferenceSteps {
      try Task.checkCancellation()
      let latentInput = MLX.concatenated([latents, packedImageLatent], axis: 1)
      let sigma = scheduler.sigmas[i]
      let timestepExpanded = MLX.full([batchSize], values: sigma).asType(latents.dtype)

      var noisePred = transformer(
        hiddenStates: latentInput,
        encoderHiddenStates: promptEmbeds,
        encoderHiddenStatesMask: promptMask,
        timestep: timestepExpanded,
        imgShapes: imgShapes,
        txtSeqLens: txtSeqLens,
        additionalTCond: additionalTCond
      )

      noisePred = noisePred[0..., 0..<latents.dim(1), 0...]

      if doTrueCFG, let negEmbeds = negativePromptEmbeds, let negMask = negativePromptMask, let negTxtSeqLens = negativeTxtSeqLens {
        var negNoisePred = transformer(
          hiddenStates: latentInput,
          encoderHiddenStates: negEmbeds,
          encoderHiddenStatesMask: negMask,
          timestep: timestepExpanded,
          imgShapes: imgShapes,
          txtSeqLens: negTxtSeqLens,
          additionalTCond: additionalTCond
        )
        negNoisePred = negNoisePred[0..., 0..<latents.dim(1), 0...]

        let scale = parameters.trueCFGScale
        var combPred = negNoisePred + scale * (noisePred - negNoisePred)

        if parameters.cfgNormalize {
          let condNorm = MLX.sqrt(MLX.sum(MLX.multiply(noisePred, noisePred), axis: -1, keepDims: true))
          let noiseNorm = MLX.sqrt(MLX.sum(MLX.multiply(combPred, combPred), axis: -1, keepDims: true))
          let eps = MLXArray(Float(1e-8))
          combPred = MLX.multiply(combPred, MLX.divide(condNorm, noiseNorm + eps))
        }

        noisePred = combPred
      }

      latents = scheduler.step(modelOutput: noisePred, timestep: i, sample: latents)
      if progress != nil {
        MLX.eval(latents)
      }

      let progressFraction = Float(i + 1) / Float(parameters.numInferenceSteps)
      progress?(i, parameters.numInferenceSteps, progressFraction)
      await Task.yield()
    }

    logger.info("Decoding layers")
    var unpackedLatents = LatentPacking.unpack(
      latents,
      layers: layers,
      height: latentHeight,
      width: latentWidth
    )

    unpackedLatents = QwenVAE.denormalizeLatent(unpackedLatents)

    var decodedLayers: [MLXArray] = []
    for l in 1...layers {
      let layerLatent = unpackedLatents[0..., l, 0..., 0..., 0...]
      let decoded = vae.decode(layerLatent)
      decodedLayers.append(decoded)
    }

    logger.info("Generation complete, produced \(decodedLayers.count) layers")
    return decodedLayers
  }

  // MARK: - Explicit Lifecycle Hooks

  public var isTextEncoderLoaded: Bool {
    textEncoder != nil
  }

  public var isTokenizerLoaded: Bool {
    tokenizer != nil
  }

  public func releaseTextEncoder() {
    textEncoder = nil
    logger.debug("Text encoder released")
  }

  public func releaseTokenizer() {
    tokenizer = nil
    logger.debug("Tokenizer released")
  }

  public func reloadTextEncoder() throws {
    try ensureTextEncoder()
    logger.debug("Text encoder reloaded")
  }

  public func reloadTokenizer() throws {
    try ensureTokenizer()
    logger.debug("Tokenizer reloaded")
  }

  // MARK: - Private Helpers

  private func calculateLayeredDimensions(resolution: Int, aspectRatio: Float) -> (width: Int, height: Int) {
    let targetArea = Float(resolution * resolution)
    var width = sqrt(targetArea * aspectRatio)
    var height = width / aspectRatio

    width = (width / 32).rounded() * 32
    height = (height / 32).rounded() * 32

    return (Int(width), Int(height))
  }

  private func encodePrompt(_ prompt: String, dtype: DType) throws -> (embeddings: MLXArray, mask: MLXArray) {
    try ensureTokenizer()
    try ensureTextEncoder()
    guard let tokenizer else {
      throw LayeredPipelineError.tokenizerNotLoaded
    }
    guard let textEncoder = textEncoder else {
      throw LayeredPipelineError.textEncoderNotLoaded
    }

    let maxLength = 256
    let tokens = tokenizer.encode(prompts: [prompt], maxLength: maxLength)
    let inputIds = tokens.inputIds
    let attentionMask = tokens.attentionMask

    let (embeddings, mask) = textEncoder.encode(inputIds: inputIds, attentionMask: attentionMask)

    return (embeddings.asType(dtype), mask)
  }

  private func ensureTokenizer() throws {
    if tokenizer != nil { return }
    guard let root = baseWeightsDirectory else {
      throw LayeredPipelineError.tokenizerNotLoaded
    }
    let tokenizerPath = root.appending(path: "tokenizer")
    let textEncoderPath = root.appending(path: "text_encoder")

    if FileManager.default.fileExists(atPath: tokenizerPath.path) {
      tokenizer = try QwenTokenizer.load(from: tokenizerPath)
    } else if FileManager.default.fileExists(atPath: textEncoderPath.path) {
      tokenizer = try QwenTokenizer.load(from: textEncoderPath)
    } else {
      tokenizer = try QwenTokenizer.load(from: root)
    }
  }

  private func ensureTextEncoder() throws {
    if textEncoder != nil { return }
    guard let root = baseWeightsDirectory else {
      throw LayeredPipelineError.textEncoderNotLoaded
    }

    let loader = QwenWeightsLoader()
    let quantization = textEncoderQuantization ?? QwenQuantizationPlanResolver.resolve(
      root: root,
      configRelativePath: "text_encoder/config.json",
      componentName: "text_encoder",
      logger: logger
    )

    var configuration = try QwenTextEncoderConfiguration.load(from: root.appending(path: "text_encoder"))
    configuration.outputDType = weightsDType
    let encoder = QwenTextEncoder(configuration: configuration)
    try loader.loadTextEncoder(
      fromDirectory: root,
      into: encoder,
      dtype: weightsDType,
      quantization: quantization
    )
    textEncoder = encoder
    textEncoderQuantization = quantization

    if let quantization, quantization.isEnabled,
       let spec = quantization.defaultSpec ?? quantization.prepackedLayers.values.first?.spec {
      logger.info(
        "Applied quantization to layered text encoder (bits=\(spec.bits), group=\(spec.groupSize), mode=\(spec.mode))."
      )
    }
  }
}

extension QwenWeightsLoader {
  func loadTransformerForLayered(
    fromDirectory directory: URL,
    into transformer: QwenLayeredTransformerV2,
    dtype: DType,
    quantization: QwenQuantizationPlan? = nil
  ) throws {
    let transformerPath = directory.appending(path: "transformer")

    let fileManager = FileManager.default
    let contents = try fileManager.contentsOfDirectory(
      at: transformerPath,
      includingPropertiesForKeys: nil,
      options: [.skipsHiddenFiles]
    )
    let safetensorsFiles = contents.filter { $0.pathExtension == "safetensors" }.sorted { $0.path < $1.path }

    guard !safetensorsFiles.isEmpty else {
      throw QwenWeightsLoaderError.noSafetensorsFound(transformerPath)
    }

    let readers = try safetensorsFiles.map { try SafeTensorsReader(fileURL: $0) }
    let merged = try WeightsMapping.merge(readers: readers, dtype: nil)
    let availableKeys = Set(merged.keys)

    applyQuantization(
      plan: quantization,
      to: transformer,
      availableKeys: availableKeys,
      tensorNameTransform: Self.layeredTransformerTensorName
    )

    let parameters = try WeightsMapping.layeredTransformerParameters(
      from: merged,
      configuration: transformer.configuration,
      dtype: dtype,
      quantization: quantization
    )

    transformer.update(parameters: parameters)
  }
}
