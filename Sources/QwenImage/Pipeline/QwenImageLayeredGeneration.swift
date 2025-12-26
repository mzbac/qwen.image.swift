import Foundation
import Logging
import MLX
import MLXNN
import MLXRandom

#if canImport(CoreGraphics)
import CoreGraphics
#endif

// MARK: - Image Resizing Utilities for Layered Generation

private func resizeImageLayered(_ image: MLXArray, targetHeight: Int, targetWidth: Int) -> MLXArray {
  let srcHeight = image.dim(2)
  let srcWidth = image.dim(3)

  if srcHeight == targetHeight && srcWidth == targetWidth {
    return image
  }

  return bilinearResizeLayered(image, targetHeight: targetHeight, targetWidth: targetWidth)
}

private func bilinearResizeLayered(_ image: MLXArray, targetHeight: Int, targetWidth: Int) -> MLXArray {
  let srcHeight = image.dim(2)
  let srcWidth = image.dim(3)

  let scaleH = Float(srcHeight) / Float(targetHeight)
  let scaleW = Float(srcWidth) / Float(targetWidth)

  // Vectorized coordinate generation using MLXArray.linspace
  let yCoords = MLX.clip(
    MLXArray.linspace(Float32(0.5), Float32(targetHeight) - 0.5, count: targetHeight) * scaleH - 0.5,
    min: 0,
    max: Float32(srcHeight - 1)
  )
  let xCoords = MLX.clip(
    MLXArray.linspace(Float32(0.5), Float32(targetWidth) - 0.5, count: targetWidth) * scaleW - 0.5,
    min: 0,
    max: Float32(srcWidth - 1)
  )

  // Compute floor indices and weights vectorized
  let y0Arr = MLX.floor(yCoords).asType(.int32)
  let y1Arr = MLX.minimum(y0Arr + 1, MLXArray(Int32(srcHeight - 1)))
  let wyArr = (yCoords - y0Arr.asType(.float32)).reshaped([1, 1, targetHeight, 1])

  let x0Arr = MLX.floor(xCoords).asType(.int32)
  let x1Arr = MLX.minimum(x0Arr + 1, MLXArray(Int32(srcWidth - 1)))
  let wxArr = (xCoords - x0Arr.asType(.float32)).reshaped([1, 1, 1, targetWidth])

  // Bilinear interpolation
  let transposed = image.transposed(0, 1, 3, 2)

  let rowsY0 = MLX.take(transposed, y0Arr, axis: 3).transposed(0, 1, 3, 2)
  let rowsY1 = MLX.take(transposed, y1Arr, axis: 3).transposed(0, 1, 3, 2)

  let interpY = rowsY0 * (1 - wyArr) + rowsY1 * wyArr

  let colsX0 = MLX.take(interpY, x0Arr, axis: 3)
  let colsX1 = MLX.take(interpY, x1Arr, axis: 3)

  let result = colsX0 * (1 - wxArr) + colsX1 * wxArr

  return result.asType(image.dtype)
}

// MARK: - Layered Pipeline Error

public enum LayeredPipelineError: Error {
  case componentsNotLoaded
  case textEncoderNotLoaded
  case tokenizerNotLoaded
  case invalidImageShape
  case invalidParameters(String)
}

// MARK: - Layered Generation Pipeline

/// Standalone pipeline for layered image generation.
/// Can be used independently or alongside QwenImagePipeline.
public class QwenLayeredPipeline {
  private var logger = Logger(label: "qwen.image.layered")

  public var vae: QwenVAE?
  public var transformer: QwenLayeredTransformerV2?
  public var textEncoder: QwenTextEncoder?
  public var tokenizer: QwenTokenizer?

  public let vaeScaleFactor: Int = 8
  public let latentChannels: Int = 16
  public let patchSize: Int = 2

  public init() {
    logger.info("QwenLayeredPipeline initialized")
  }

  /// Load pipeline components from a model directory
  public static func load(from path: URL, dtype: DType = .bfloat16) async throws -> QwenLayeredPipeline {
    let pipeline = QwenLayeredPipeline()
    let loader = QwenWeightsLoader()

    pipeline.logger.info("Loading layered pipeline from \(path.path)")

    // Load VAE
    pipeline.logger.info("Loading VAE...")
    pipeline.vae = QwenVAE()
    try loader.loadVAE(fromDirectory: path, into: pipeline.vae!, dtype: dtype)

    // Load transformer with layered config (using working QwenTransformer)
    pipeline.logger.info("Loading transformer...")
    let transformerPath = path.appending(path: "transformer")
    let transformerConfig = try QwenLayeredTransformerConfiguration.load(from: transformerPath)
    let transformerQuantization = quantizationPlan(
      forLayeredComponentAt: path,
      configRelativePath: "transformer/config.json"
    )
    pipeline.transformer = QwenLayeredTransformerV2(transformerConfig)
    try loader.loadTransformerForLayered(
      fromDirectory: path,
      into: pipeline.transformer!,
      dtype: dtype,
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

    // Load text encoder and tokenizer (always required for proper generation)
    pipeline.logger.info("Loading text encoder...")
    let textEncoderQuantization = quantizationPlan(
      forLayeredComponentAt: path,
      configRelativePath: "text_encoder/config.json"
    )
    pipeline.textEncoder = QwenTextEncoder()
    try loader.loadTextEncoder(
      fromDirectory: path,
      into: pipeline.textEncoder!,
      dtype: dtype,
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

  /// Apply a LoRA adapter to the layered transformer.
  /// Compatible with Qwen-Image Lightning adapters.
  public func applyLora(from fileURL: URL, scale: Float = 1.0) throws {
    guard let transformer = transformer else {
      throw LayeredPipelineError.componentsNotLoaded
    }

    let layers = try Self.loadLoraLayers(from: fileURL)
    if layers.isEmpty {
      logger.warning("LoRA: no adapter layers found in \(fileURL.path)")
      return
    }

    logger.info("LoRA: loaded \(layers.count) adapter bases from \(fileURL.lastPathComponent)")

    let applied = Self.applyLoraLayers(layers, to: transformer, globalScale: scale, logger: logger)
    logger.info("LoRA: applied to layered transformer (\(applied) layers updated).")
  }

  private typealias LoraLayerMap = [String: (down: MLXArray, up: MLXArray, alpha: Float)]

  private static func loadLoraLayers(from fileURL: URL) throws -> LoraLayerMap {
    let reader = try SafeTensorsReader(fileURL: fileURL)
    var alphaTensors: [String: MLXArray] = [:]
    var downTensors: [String: MLXArray] = [:]
    var upTensors: [String: MLXArray] = [:]

    for name in reader.tensorNames {
      if name.hasSuffix(".alpha") {
        let base = String(name.dropLast(".alpha".count))
        alphaTensors[base] = try reader.tensor(named: name)
      } else if name.hasSuffix(".lora_down.weight") {
        let base = String(name.dropLast(".lora_down.weight".count))
        downTensors[base] = try reader.tensor(named: name)
      } else if name.hasSuffix(".lora_up.weight") {
        let base = String(name.dropLast(".lora_up.weight".count))
        upTensors[base] = try reader.tensor(named: name)
      }
    }

    var layers: LoraLayerMap = [:]
    for (base, down) in downTensors {
      guard let up = upTensors[base] else { continue }
      let rank = down.dim(0)
      guard rank > 0 else { continue }

      let alphaValue: Float
      if let alphaTensor = alphaTensors[base] {
        let scalar = alphaTensor.asType(.float32)
        MLX.eval(scalar)
        alphaValue = scalar.item(Float.self)
      } else {
        alphaValue = Float(rank)
      }

      layers[base] = (down: down, up: up, alpha: alphaValue)
    }

    return layers
  }

  /// Map internal transformer module paths to HF-style base keys used by LoRA.
  private static func transformerLoraBaseName(for path: String) -> String {
    var name = path
    name = name.replacingOccurrences(of: ".img_ff.mlp_in", with: ".img_mlp.net.0.proj")
    name = name.replacingOccurrences(of: ".img_ff.mlp_out", with: ".img_mlp.net.2")
    name = name.replacingOccurrences(of: ".txt_ff.mlp_in", with: ".txt_mlp.net.0.proj")
    name = name.replacingOccurrences(of: ".txt_ff.mlp_out", with: ".txt_mlp.net.2")
    name = name.replacingOccurrences(of: ".attn.attn_to_out", with: ".attn.to_out.0")
    name = name.replacingOccurrences(of: ".img_norm1.mod_linear", with: ".img_mod.1")
    name = name.replacingOccurrences(of: ".txt_norm1.mod_linear", with: ".txt_mod.1")
    return name
  }

  private static func applyLoraLayers(
    _ layers: LoraLayerMap,
    to transformer: QwenLayeredTransformerV2,
    globalScale: Float,
    logger: Logger
  ) -> Int {
    var layerUpdates: [String: MLXArray] = [:]
    var appliedCount = 0

    for (path, module) in transformer.namedModules() {
      let base = transformerLoraBaseName(for: path)
      guard let layer = layers[base] else { continue }

      let rank = layer.down.dim(0)
      guard rank > 0 else { continue }

      let effectiveScale = globalScale * layer.alpha / Float(rank)
      let loraDown = layer.down.asType(.bfloat16)
      let loraUp = layer.up.asType(.bfloat16)
      var delta = matmul(loraUp, loraDown)
      let scaleArray = MLXArray(Float32(effectiveScale))
      delta = (delta * scaleArray).asType(.bfloat16)

      if let quantizedLinear = module as? QuantizedLinear {
        logger.debug("LoRA: applying to quantized layer \(path)")
        let baseWeightFP32 = dequantized(
          quantizedLinear.weight,
          scales: quantizedLinear.scales,
          biases: quantizedLinear.biases,
          groupSize: quantizedLinear.groupSize,
          bits: quantizedLinear.bits
        )
        let baseWeight = baseWeightFP32.asType(.bfloat16)
        let fusedWeight = baseWeight + delta

        let fusedLinear = Linear(
          weight: fusedWeight,
          bias: quantizedLinear.bias
        )
        let requantized = QuantizedLinear(
          fusedLinear,
          groupSize: quantizedLinear.groupSize,
          bits: quantizedLinear.bits
        )

        layerUpdates["\(path).weight"] = requantized.weight
        layerUpdates["\(path).scales"] = requantized.scales
        if let biases = requantized.biases {
          layerUpdates["\(path).biases"] = biases
        }
        appliedCount += 1
      } else if let linear = module as? Linear {
        logger.debug("LoRA: fusing into linear layer \(path)")
        let currentWeight = linear.weight.asType(.float32)
        let fusedWeight = currentWeight + delta
        let finalWeight = fusedWeight.asType(linear.weight.dtype)
        layerUpdates["\(path).weight"] = finalWeight
        appliedCount += 1
      }
    }

    if !layerUpdates.isEmpty {
      transformer.update(parameters: ModuleParameters.unflattened(layerUpdates))
    }
    return appliedCount
  }

  /// Generate layer images from an input image
  public func generate(
    image: MLXArray,
    parameters: LayeredGenerationParameters,
    progress: ((Int, Int, Float) -> Void)? = nil
  ) throws -> [MLXArray] {
    guard let vae = vae, let transformer = transformer else {
      throw LayeredPipelineError.componentsNotLoaded
    }

    logger.info("Starting layer generation with \(parameters.layers) layers")

    if let seed = parameters.seed {
      MLXRandom.seed(seed)
    }

    let batchSize = image.dim(0)
    let layers = parameters.layers

    // Calculate target dimensions
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
      resizedImage = resizeImageLayered(resizedImage, targetHeight: targetHeight, targetWidth: targetWidth)
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
    let imageSeqLen = Float(halfH * halfW)
    let baseSeqLen: Float = 256.0
    let mu = sqrt(imageSeqLen / baseSeqLen)

    let scheduler = QwenFlowMatchScheduler(
      numInferenceSteps: parameters.numInferenceSteps,
      width: width,
      height: height,
      requiresSigmaShift: false,
      mu: mu
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

    guard textEncoder != nil, tokenizer != nil else {
      throw LayeredPipelineError.componentsNotLoaded
    }
    let prompt = parameters.prompt ?? ""
    let encoded = try encodePrompt(prompt, dtype: imageLatent.dtype)
    let promptEmbeds = encoded.embeddings
    let promptMask = encoded.mask

    var imgShapes: [[(Int, Int, Int)]] = []
    for _ in 0..<batchSize {
      var shapes: [(Int, Int, Int)] = []
      for _ in 0..<(layers + 2) {
        shapes.append((1, halfH, halfW))
      }
      imgShapes.append(shapes)
    }

    let txtSeqLens = [Int](repeating: promptEmbeds.dim(1), count: batchSize)
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
    textEncoder = nil

    logger.info("Starting denoising with \(parameters.numInferenceSteps) steps")

    for i in 0..<parameters.numInferenceSteps {
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

      let progressFraction = Float(i + 1) / Float(parameters.numInferenceSteps)
      progress?(i, parameters.numInferenceSteps, progressFraction)
    }

    logger.info("Decoding layers")
    var unpackedLatents = LatentPacking.unpack(
      latents,
      layers: layers,
      height: latentHeight,
      width: latentWidth
    )

    // Denormalize before decoding (matches Python pipeline behavior)
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
    guard let tokenizer = tokenizer, let textEncoder = textEncoder else {
      throw LayeredPipelineError.textEncoderNotLoaded
    }

    let maxLength = 256
    let tokens = tokenizer.encode(prompts: [prompt], maxLength: maxLength)
    let inputIds = tokens.inputIds
    let attentionMask = tokens.attentionMask

    // encode returns (embeddings, mask) tuple with dropIndex already applied
    let (embeddings, mask) = textEncoder.encode(inputIds: inputIds, attentionMask: attentionMask)

    return (embeddings.asType(dtype), mask)
  }
}

// MARK: - Weight Loader Extension

extension QwenLayeredPipeline {
  /// Build a quantization plan for layered components by merging per-component
  /// quantization config with any prepacked manifest in the snapshot root.
  static func quantizationPlan(
    forLayeredComponentAt root: URL,
    configRelativePath: String
  ) -> QwenQuantizationPlan? {
    let configURL = root.appending(path: configRelativePath)
    var plan = QwenQuantizationPlan.load(from: configURL)
    if let manifest = QwenQuantizedSnapshotManifest.load(from: root) {
      var workingPlan = plan ?? QwenQuantizationPlan()
      workingPlan.registerPrepackedLayers(from: manifest)
      plan = workingPlan
    }
    return plan
  }
}

extension QwenWeightsLoader {
  /// Load weights for the layered transformer using the working transformer implementation
  func loadTransformerForLayered(
    fromDirectory directory: URL,
    into transformer: QwenLayeredTransformerV2,
    dtype: DType,
    quantization: QwenQuantizationPlan? = nil
  ) throws {
    let transformerPath = directory.appending(path: "transformer")

    // Collect all safetensors files in the transformer directory
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

    // Read all weights and merge
    let readers = try safetensorsFiles.map { try SafeTensorsReader(fileURL: $0) }
    let merged = try WeightsMapping.merge(readers: readers, dtype: nil)
    let availableKeys = Set(merged.keys)

    applyQuantization(
      plan: quantization,
      to: transformer,
      availableKeys: availableKeys,
      tensorNameTransform: Self.layeredTransformerTensorName
    )

    // Use the layeredTransformerParameters function with correct key mappings for V2 types
    let parameters = try WeightsMapping.layeredTransformerParameters(
      from: merged,
      configuration: transformer.configuration,
      dtype: dtype,
      quantization: quantization
    )

    transformer.update(parameters: parameters)
  }
}
