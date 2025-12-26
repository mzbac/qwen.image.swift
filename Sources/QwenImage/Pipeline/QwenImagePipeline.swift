import Combine
import Foundation
import Logging
import MLX
import MLXNN
import MLXRandom

#if canImport(CoreGraphics)
import CoreGraphics
#endif

#if canImport(AppKit)
import AppKit
#elseif canImport(UIKit)
import UIKit
#endif

public struct QwenPromptEncodingResult {
  public let tokenBatch: QwenTokenBatch
  public let promptEmbeddings: MLXArray
  public let encoderAttentionMask: MLXArray

  public init(
    tokenBatch: QwenTokenBatch,
    promptEmbeddings: MLXArray,
    encoderAttentionMask: MLXArray
  ) {
    self.tokenBatch = tokenBatch
    self.promptEmbeddings = promptEmbeddings
    self.encoderAttentionMask = encoderAttentionMask
  }
}

public enum QwenPromptEncodingError: Error {
  case emptyBatch
  case missingConditional
}

public struct QwenGuidanceEncoding {
  public let unconditionalEmbeddings: MLXArray
  public let conditionalEmbeddings: MLXArray
  public let unconditionalMask: MLXArray
  public let conditionalMask: MLXArray
  public let tokenBatch: QwenTokenBatch

  public init(
    unconditionalEmbeddings: MLXArray,
    conditionalEmbeddings: MLXArray,
    unconditionalMask: MLXArray,
    conditionalMask: MLXArray,
    tokenBatch: QwenTokenBatch
  ) {
    self.unconditionalEmbeddings = unconditionalEmbeddings
    self.conditionalEmbeddings = conditionalEmbeddings
    self.unconditionalMask = unconditionalMask
    self.conditionalMask = conditionalMask
    self.tokenBatch = tokenBatch
  }
}

extension QwenPromptEncodingResult {
  public func guidanceEncoding() throws -> QwenGuidanceEncoding {
    let batchSize = promptEmbeddings.dim(0)
    guard batchSize > 0 else {
      throw QwenPromptEncodingError.emptyBatch
    }
    guard batchSize >= 2 else {
      throw QwenPromptEncodingError.missingConditional
    }

    let unconditionalEmbeddings = promptEmbeddings[0 ..< 1, 0..., 0...]
    let conditionalEmbeddings = promptEmbeddings[1..., 0..., 0...]

    let unconditionalMask = encoderAttentionMask[0 ..< 1, 0...]
    let conditionalMask = encoderAttentionMask[1..., 0...]

    return QwenGuidanceEncoding(
      unconditionalEmbeddings: unconditionalEmbeddings,
      conditionalEmbeddings: conditionalEmbeddings,
      unconditionalMask: unconditionalMask,
      conditionalMask: conditionalMask,
      tokenBatch: tokenBatch
    )
  }
}

extension QwenGuidanceEncoding {
  public func stackedEmbeddings() -> (embeddings: MLXArray, attentionMask: MLXArray) {
    let unconditional = unconditionalEmbeddings
    let conditional = conditionalEmbeddings
    let embeddings = MLX.concatenated([unconditional, conditional], axis: 0)

    let unconditionalMask = self.unconditionalMask.asType(.int32)
    let conditionalMask = self.conditionalMask.asType(.int32)
    let attentionMask = MLX.concatenated([unconditionalMask, conditionalMask], axis: 0)
    return (embeddings, attentionMask)
  }
}

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

final class VisionPromptContext {
  let patchInputs: MLXArray
  let grids: [QwenVisionGrid]
  let tokenCounts: [Int]
  let gridTHW: [(Int, Int, Int)]
  private var cachedEmbeddings: [MLXArray]?

  var placeholderCount: Int {
    tokenCounts.count
  }

  init(patchInputs: MLXArray, grids: [QwenVisionGrid], tokenCounts: [Int], gridTHW: [(Int, Int, Int)]) {
    precondition(grids.count == tokenCounts.count, "Grid metadata must align with token counts.")
    precondition(gridTHW.count == tokenCounts.count, "gridTHW and token counts must align.")
    self.patchInputs = patchInputs
    self.grids = grids
    self.tokenCounts = tokenCounts
    self.gridTHW = gridTHW
  }

  func visionEmbeddings(using tower: QwenVisionTower, dtype: DType?) throws -> [MLXArray] {
    if let cached = cachedEmbeddings {
      return cached
    }
    var hidden = try tower(patchInputs: patchInputs, grid: grids).hiddenStates
    if let dtype, hidden.dtype != dtype {
      hidden = hidden.asType(dtype)
    }
    var embeddings: [MLXArray] = []
    embeddings.reserveCapacity(tokenCounts.count)
    var offset = 0
    for (index, count) in tokenCounts.enumerated() {
      let end = offset + count
      precondition(end <= hidden.dim(0), "Vision tower output shorter than expected for placeholder \(index).")
      let slice = hidden[offset..<end, 0...]
      embeddings.append(slice)
      offset = end
    }
    precondition(offset == hidden.dim(0), "Unused vision tokens detected: \(hidden.dim(0) - offset).")
    cachedEmbeddings = embeddings
    return embeddings
  }
}

private let pipelineLogger = QwenLogger.pipeline
private let tokenizerLogger = QwenLogger.tokenizer
private let visionLogger = QwenLogger.vision

private typealias QwenLoraLayerMap = [String: (down: MLXArray, up: MLXArray, alpha: Float)]

public final class QwenImagePipeline {
  public enum PipelineError: Error {
    case componentNotLoaded(String)
    case generationNotImplemented
    case imageConversionUnavailable
    case imageConversionFailed
    case invalidTensorShape(String)
    case visionSpecialTokensUnavailable
    case visionPlaceholderMismatch
  }

  private let config: QwenImageConfig
  private let weightsLoader = QwenWeightsLoader()

  private var baseWeightsDirectory: URL?

  private var tokenizer: QwenTokenizer?
  private var textEncoder: QwenTextEncoder?
  private var visionTokensPerSecond: Int = 2
  private var transformer: QwenTransformer?
  private var transformerDirectory: URL?
  private var unet: QwenUNet?
  private var vae: QwenVAE?
  private let visionPreprocessor = QwenVisionPreprocessor()
  private var visionTower: QwenVisionTower?
  private let visionConfiguration = QwenVisionConfiguration()
  private var runtimeQuantizationSpec: QwenQuantizationSpec?
  private var runtimeQuantMinInputDim: Int = 4096
  private let runtimeQuantAllowedSuffixes: [String] = [
    ".attn.to_q",
    ".attn.to_k",
    ".attn.to_v",
    ".attn.attn_to_out",
    ".img_ff.mlp_in",
    ".img_ff.mlp_out",
    ".txt_ff.mlp_in",
    ".txt_ff.mlp_out"
  ]
  private var textEncoderRuntimeQuantized = false
  private var transformerRuntimeQuantized = false
  private var visionRuntimeQuantized = false
  private var textEncoderQuantization: QwenQuantizationPlan?
  private var transformerQuantization: QwenQuantizationPlan?
  private var attentionQuantizationSpec: QwenQuantizationSpec?

  private var progressSubject: PassthroughSubject<ProgressInfo, Never>?
  private var pendingLoraURL: URL?
  private var pendingLoraScale: Float = 1.0

  public init(config: QwenImageConfig) {
    self.config = config
  }

  /// Inform the pipeline of the snapshot root directory so it can lazily
  /// load components (UNet, VAE, vision tower) as needed.
  public func setBaseDirectory(_ directory: URL) {
    baseWeightsDirectory = directory
    if transformerDirectory == nil {
      transformerDirectory = directory
    }
  }

  /// Register a LoRA adapter to be applied lazily once the UNet is loaded.
  public func setPendingLora(from url: URL, scale: Float = 1.0) {
    pendingLoraURL = url
    pendingLoraScale = scale
  }

  public var progress: AnyPublisher<ProgressInfo, Never>? {
    progressSubject?.eraseToAnyPublisher()
  }

  private func combinedQuantizationPlan(
    configURL: URL,
    snapshotRoot: URL,
    componentName: String
  ) -> QwenQuantizationPlan? {
    var plan = QwenQuantizationPlan.load(from: configURL)
    if let manifest = QwenQuantizedSnapshotManifest.load(from: snapshotRoot) {
      var workingPlan = plan ?? QwenQuantizationPlan()
      workingPlan.registerPrepackedLayers(from: manifest)
      plan = workingPlan
      pipelineLogger.info(
        "Detected pre-packed quantization manifest for \(componentName) (\(manifest.layers.count) layers, bits=\(manifest.bits), group=\(manifest.groupSize))."
      )
    }
    applyAttentionQuantization()
    return plan
  }

  private func applyAttentionQuantization() {
    if let transformer {
      transformer.setAttentionQuantization(attentionQuantizationSpec)
    }
    if let unet {
      unet.transformer.setAttentionQuantization(attentionQuantizationSpec)
    }
  }

  private func runtimeQuantizationAllowed(path: String) -> Bool {
    for suffix in runtimeQuantAllowedSuffixes where path.hasSuffix(suffix) {
      return true
    }
    return false
  }

  @MainActor
  public func generateImageAsync(_ parameters: GenerationParameters) async throws -> PipelineImage {
    throw PipelineError.generationNotImplemented
  }

  public func prepareTokenizer(using snapshotOptions: HubSnapshotOptions) async throws {
    let snapshot = try HubSnapshot(options: snapshotOptions)
    try await loadTokenizer(from: snapshot)
  }

  public func prepareTokenizer(from directory: URL, maxLength: Int? = nil) throws {
    let tokenizer = try QwenTokenizer.load(from: directory, maxLengthOverride: maxLength)
    self.tokenizer = tokenizer
  }

  public func prepareTextEncoder(using snapshotOptions: HubSnapshotOptions) async throws {
    let snapshot = try HubSnapshot(options: snapshotOptions)
    try await loadTextEncoder(from: snapshot)
  }

  public func prepareTextEncoder(from directory: URL) throws {
    let encoder = QwenTextEncoder()
    let textEncoderDir = directory.appending(path: "text_encoder")
    let configURL = textEncoderDir.appending(path: "config.json")
    let quantPlan = combinedQuantizationPlan(
      configURL: configURL,
      snapshotRoot: directory,
      componentName: "text_encoder"
    )
    textEncoderQuantization = quantPlan
    try weightsLoader.loadTextEncoder(
      fromDirectory: directory,
      into: encoder,
      dtype: preferredWeightDType(),
      quantization: quantPlan
    )
    textEncoder = encoder
    applyRuntimeQuantizationIfNeeded(to: encoder, flag: &textEncoderRuntimeQuantized)
    if FileManager.default.fileExists(atPath: configURL.path) {
      do {
        let data = try Data(contentsOf: configURL)
        if let meta = try JSONSerialization.jsonObject(with: data) as? [String: Any],
           let vconf = meta["vision_config"] as? [String: Any] {
          if let tps = vconf["tokens_per_second"] as? Int {
            visionTokensPerSecond = tps
          } else if let tpsNum = vconf["tokens_per_second"] as? NSNumber {
            visionTokensPerSecond = tpsNum.intValue
          }
        }
      } catch {
        pipelineLogger.warning("prepareTextEncoder: failed to parse tokens_per_second: \(error)")
      }
    }
  }

  public func prepareTransformer(
    using snapshotOptions: HubSnapshotOptions,
    configuration: QwenTransformerConfiguration
  ) async throws {
    let snapshot = try HubSnapshot(options: snapshotOptions)
    try await loadTransformer(from: snapshot, configuration: configuration)
  }

  public func prepareTransformer(
    from directory: URL,
    configuration: QwenTransformerConfiguration
  ) throws {
    let transformer = QwenTransformer(configuration: configuration)
    let dtype = preferredWeightDType()
    let configURL = directory.appending(path: "transformer").appending(path: "config.json")
    let quantPlan = combinedQuantizationPlan(
      configURL: configURL,
      snapshotRoot: directory,
      componentName: "transformer"
    )
    transformerQuantization = quantPlan
    try weightsLoader.loadTransformer(
      fromDirectory: directory,
      into: transformer,
      dtype: dtype,
      quantization: quantPlan
    )
    transformerDirectory = directory
    self.transformer = transformer
    visionTower = nil
    applyRuntimeQuantizationIfNeeded(to: transformer, flag: &transformerRuntimeQuantized)
    transformer.setAttentionQuantization(attentionQuantizationSpec)
  }

  public func prepareUNet(
    using snapshotOptions: HubSnapshotOptions,
    configuration: QwenTransformerConfiguration
  ) async throws {
    let snapshot = try HubSnapshot(options: snapshotOptions)
    try await loadUNet(from: snapshot, configuration: configuration)
  }

  public func prepareUNet(
    from directory: URL,
    configuration: QwenTransformerConfiguration
  ) throws {
    let unet = QwenUNet(configuration: configuration)
    let dtype = preferredWeightDType()
    let configURL = directory.appending(path: "transformer").appending(path: "config.json")
    let quantPlan = combinedQuantizationPlan(
      configURL: configURL,
      snapshotRoot: directory,
      componentName: "transformer"
    )
    transformerQuantization = quantPlan
    try weightsLoader.loadUNet(
      fromDirectory: directory,
      into: unet,
      dtype: dtype,
      quantization: quantPlan
    )
    self.unet = unet
    transformerDirectory = directory
    visionTower = nil
    applyRuntimeQuantizationIfNeeded(to: unet.transformer, flag: &transformerRuntimeQuantized)
    unet.transformer.setAttentionQuantization(attentionQuantizationSpec)
  }

  public func prepareVAE(using snapshotOptions: HubSnapshotOptions) async throws {
    let snapshot = try HubSnapshot(options: snapshotOptions)
    try await loadVAE(from: snapshot)
  }

  public func prepareAll(
    using snapshotOptions: HubSnapshotOptions,
    configuration: QwenTransformerConfiguration,
    tokenizerMaxLength: Int? = nil
  ) async throws {
    let snapshot = try HubSnapshot(options: snapshotOptions)
    let root = try await snapshot.prepare()
    try prepareTokenizer(from: root, maxLength: tokenizerMaxLength)
    try prepareTextEncoder(from: root)
    try prepareUNet(from: root, configuration: configuration)
    try prepareVAE(from: root)
  }

  public func prepareVAE(from directory: URL) throws {
    let vae = QwenVAE()
    let dtype = preferredWeightDType()
    try weightsLoader.loadVAE(fromDirectory: directory, into: vae, dtype: dtype)
    self.vae = vae
  }

  public func denoiseTokens(
    timestepIndex: Int,
    runtimeConfig: QwenRuntimeConfig,
    latentTokens: MLXArray,
    encoderHiddenStates: MLXArray,
    encoderHiddenStatesMask: MLXArray,
    imageSegments: [(Int, Int, Int)]? = nil
  ) throws -> MLXArray {
    guard let unet else {
      throw PipelineError.componentNotLoaded("UNet")
    }
    return unet.forwardTokens(
      timestepIndex: timestepIndex,
      runtimeConfig: runtimeConfig,
      latentTokens: latentTokens,
      encoderHiddenStates: encoderHiddenStates,
      encoderHiddenStatesMask: encoderHiddenStatesMask,
      imageSegments: imageSegments
    )
  }

  public func denoiseLatents(
    timestepIndex: Int,
    runtimeConfig: QwenRuntimeConfig,
    latentImages: MLXArray,
    encoderHiddenStates: MLXArray,
    encoderHiddenStatesMask: MLXArray
  ) throws -> MLXArray {
    guard let unet else {
      throw PipelineError.componentNotLoaded("UNet")
    }
    return unet.forwardLatents(
      timestepIndex: timestepIndex,
      runtimeConfig: runtimeConfig,
      latentImages: latentImages,
      encoderHiddenStates: encoderHiddenStates,
      encoderHiddenStatesMask: encoderHiddenStatesMask
    )
  }

  public func encodePixels(_ pixels: MLXArray) throws -> MLXArray {
    return try encodePixelsWithIntermediates(pixels).latents
  }

  public func decodeLatents(_ latents: MLXArray) throws -> MLXArray {
    guard let vae else {
      throw PipelineError.componentNotLoaded("VAE")
    }
    guard latents.ndim == 4 else {
      throw PipelineError.invalidTensorShape("Expected latent tensor with four dimensions [batch, channels, height, width].")
    }
    guard latents.dim(1) == 16 else {
      throw PipelineError.invalidTensorShape("Latent tensor must have 16 channels.")
    }
    let decoded = vae.decodeWithDenormalization(latents.asType(preferredWeightDType() ?? latents.dtype))
    return denormalizeFromDecoder(decoded)
  }

#if canImport(CoreGraphics)
  public func encodeImage(_ image: PipelineImage) throws -> MLXArray {
    let cgImage = try Self.makeCGImage(from: image)
    return try encodeCGImage(cgImage)
  }

  public func encodeCGImage(_ image: CGImage) throws -> MLXArray {
    let array = try QwenImageIO.array(from: image)
    return try encodePixels(array)
  }

  public func decodeLatentsToCGImage(_ latents: MLXArray) throws -> CGImage {
    let pixels = try decodeLatents(latents)
    return try QwenImageIO.image(from: pixels)
  }

  public func decodeLatentsToImage(_ latents: MLXArray) throws -> PipelineImage {
    let cgImage = try decodeLatentsToCGImage(latents)
    return Self.makePipelineImage(from: cgImage)
  }
#endif

  public func setRuntimeQuantization(_ spec: QwenQuantizationSpec?, minInputDim: Int = 4096) {
    runtimeQuantizationSpec = spec
    runtimeQuantMinInputDim = minInputDim
    textEncoderRuntimeQuantized = false
    transformerRuntimeQuantized = false
    visionRuntimeQuantized = false
    if let spec {
      let layersDescription = runtimeQuantAllowedSuffixes.joined(separator: ", ")
      pipelineLogger.info("Runtime quantization enabled (bits: \(spec.bits), group: \(spec.groupSize), mode: \(spec.mode), min-dim: \(minInputDim), layers: \(layersDescription)).")
      attentionQuantizationSpec = spec
    } else {
      pipelineLogger.info("Runtime quantization disabled.")
      attentionQuantizationSpec = nil
    }
    if let encoder = textEncoder {
      applyRuntimeQuantizationIfNeeded(to: encoder, flag: &textEncoderRuntimeQuantized)
    }
    if let transformer = transformer {
      applyRuntimeQuantizationIfNeeded(to: transformer, flag: &transformerRuntimeQuantized)
    } else if let unet = unet {
      applyRuntimeQuantizationIfNeeded(to: unet.transformer, flag: &transformerRuntimeQuantized)
    }
    if let tower = visionTower {
      applyRuntimeQuantizationIfNeeded(to: tower, flag: &visionRuntimeQuantized)
    }
    applyAttentionQuantization()
  }

  public func setAttentionQuantization(_ spec: QwenQuantizationSpec?) {
    attentionQuantizationSpec = spec
    applyAttentionQuantization()
  }

  private func encodePixelsWithIntermediates(_ pixels: MLXArray) throws -> (latents: MLXArray, quantHidden: MLXArray, encoderHidden: MLXArray) {
    guard let vae else {
      throw PipelineError.componentNotLoaded("VAE")
    }
    guard pixels.ndim == 4, pixels.dim(1) == 3 else {
      throw PipelineError.invalidTensorShape("Expected encoder input shape [batch, 3, height, width].")
    }
    var normalized = pixels
    normalized = normalized.asType(.float32)
    normalized = normalizeForEncoder(normalized)
    pipelineLogger.debug("edit: encodePixels input shape=\(normalized.shape) dtype=\(normalized.dtype)")
    let (latents, encoderHidden, quantHidden) = vae.encodeWithIntermediates(normalized)
    pipelineLogger.debug("edit: encodePixels output shape=\(latents.shape) dtype=\(latents.dtype)")
    return (latents, quantHidden, encoderHidden)
  }

  private func loadExternalVAEPixels(from basePath: String) throws -> MLXArray {
    let baseURL = URL(fileURLWithPath: basePath, isDirectory: false)
    let metaURL = baseURL.appendingPathExtension("json")
    let binURL = baseURL.appendingPathExtension("bin")

    let metaData = try Data(contentsOf: metaURL)
    guard let meta = try JSONSerialization.jsonObject(with: metaData) as? [String: Any],
          let shape = meta["shape"] as? [Int],
          let dtype = meta["dtype"] as? String
    else {
      throw PipelineError.invalidTensorShape("Invalid VAE image metadata at \(metaURL.path)")
    }
    guard dtype == "float32" else {
      throw PipelineError.invalidTensorShape("Unsupported VAE image dtype: \(dtype)")
    }

    let data = try Data(contentsOf: binURL)
    let expected = shape.reduce(1, *)
    let values: [Float32] = data.withUnsafeBytes { raw in
      let ptr = raw.bindMemory(to: Float32.self)
      return Array(ptr.prefix(expected))
    }
    guard values.count == expected else {
      throw PipelineError.invalidTensorShape("VAE image count mismatch: expected \(expected) got \(values.count)")
    }

    var array = MLXArray(values, shape)
    if shape.count == 5 {
      guard shape[2] == 1 else {
        throw PipelineError.invalidTensorShape("Unsupported VAE image temporal dimension: \(shape)")
      }
      array = array.reshaped(shape[0], shape[1], shape[3], shape[4])
    } else if shape.count != 4 {
      throw PipelineError.invalidTensorShape("Unexpected VAE image rank: \(shape)")
    }
    var floatArray = array.asType(.float32)
    floatArray = (floatArray + MLXArray(Float32(1.0))) / MLXArray(Float32(2.0))
    return floatArray
  }

  public func tokenize(
    prompt: String,
    negativePrompt: String?,
    maxLength: Int
  ) throws -> QwenTokenBatch {
    guard let tokenizer else {
      throw PipelineError.componentNotLoaded("Tokenizer")
    }
    return tokenizer.encode(prompt: prompt, negativePrompt: negativePrompt, maxLength: maxLength)
  }

  public func tokenizeForGuidance(
    prompt: String,
    negativePrompt: String?,
    maxLength: Int
  ) throws -> QwenTokenBatch {
    let unconditionalPrompt = negativePrompt ?? ""
    return try tokenize(prompt: prompt, negativePrompt: unconditionalPrompt, maxLength: maxLength)
  }

  public func encodePrompts(
    prompt: String,
    negativePrompt: String?,
    maxLength: Int
  ) throws -> QwenPromptEncodingResult {
    let batch = try tokenize(prompt: prompt, negativePrompt: negativePrompt, maxLength: maxLength)
    let (promptEmbeddings, encoderMask) = try encode(inputIds: batch.inputIds, attentionMask: batch.attentionMask)
    return QwenPromptEncodingResult(
      tokenBatch: batch,
      promptEmbeddings: promptEmbeddings,
      encoderAttentionMask: encoderMask
    )
  }

  public func encodeGuidancePrompts(
    prompt: String,
    negativePrompt: String?,
    maxLength: Int
  ) throws -> QwenGuidanceEncoding {
    try encodeGuidancePromptsInternal(
      prompt: prompt,
      negativePrompt: negativePrompt,
      maxLength: maxLength,
      visionContext: nil
    )
  }

  func encodeGuidancePromptsInternal(
    prompt: String,
    negativePrompt: String?,
    maxLength: Int,
    visionContext: VisionPromptContext?
  ) throws -> QwenGuidanceEncoding {
    var promptSequence = prompt
    var negativeSequence = negativePrompt ?? ""
    if let visionContext, visionContext.placeholderCount > 0 {
      let count = visionContext.placeholderCount
      let segmentBody = (0..<count).reduce("") { partial, index in
        partial + "Picture \(index + 1): <|vision_start|><|image_pad|><|vision_end|>"
      }
      let segmentString = "\n" + segmentBody + "\n"
      promptSequence = segmentString + promptSequence
      negativeSequence = segmentString + negativeSequence
    }
    tokenizerLogger.debug("promptSequence=\(promptSequence.prefix(200))...")
    tokenizerLogger.debug("negativeSequence=\(negativeSequence.prefix(200))...")
    if let tokenizer {
      tokenizerLogger.debug("tokenizer templateTokenCount=\(tokenizer.templateTokenCount)")
    }

    var batch = try tokenize(
      prompt: promptSequence,
      negativePrompt: negativeSequence,
      maxLength: maxLength
    )
    tokenizerLogger.debug("tokenize: shape=\(batch.inputIds.shape) attention=\(batch.attentionMask.shape)")
    if tokenizerLogger.logLevel <= .debug {
      let maskInt = batch.attentionMask.asType(.int32)
      let attentionSum = MLX.sum(maskInt)
      MLX.eval(attentionSum)
      tokenizerLogger.debug("tokenize: attention sum=\(attentionSum.item(Int.self))")
      if batch.attentionMask.dim(0) == 2 {
        let row0 = MLX.sum(maskInt[0, 0...])
        let row1 = MLX.sum(maskInt[1, 0...])
        MLX.eval(row0)
        MLX.eval(row1)
        tokenizerLogger.debug("tokenize: per-row sums=\(row0.item(Int.self)), \(row1.item(Int.self))")
      }
    }
    var placeholderOffsets: [[Int]] = Array(repeating: [], count: batch.inputIds.dim(0))
    var placeholderSpanLengths: [[Int]] = Array(repeating: [], count: batch.inputIds.dim(0))
    if let visionContext, visionContext.placeholderCount > 0 {
      guard let tokenizer else {
        throw PipelineError.componentNotLoaded("Tokenizer")
      }
      guard let imageTokenId = tokenizer.imageTokenId else {
        throw PipelineError.visionSpecialTokensUnavailable
      }
      guard let visionStartTokenId = tokenizer.visionStartTokenId else {
        throw PipelineError.visionSpecialTokensUnavailable
      }
      let repeatCounts = visionContext.tokenCounts
      do {
        let expansion = try EditTokenUtils.expandVisionPlaceholders(
          batch: batch,
          padTokenId: tokenizer.padTokenId,
          imageTokenId: imageTokenId,
          visionStartTokenId: tokenizer.visionStartTokenId,
          visionEndTokenId: tokenizer.visionEndTokenId,
          repeatCounts: repeatCounts
        )
        batch = expansion.batch
        placeholderOffsets = expansion.startOffsets
        placeholderSpanLengths = expansion.spanLengths
      } catch {
        pipelineLogger.warning("expandVisionPlaceholders failed: \(error)")
        throw PipelineError.visionPlaceholderMismatch
      }
      tokenizerLogger.debug("post-expand tokenize: shape=\(batch.inputIds.shape)")
      if tokenizerLogger.logLevel <= .debug {
        let expandedMask = batch.attentionMask.asType(.int32)
        let expandedSum = MLX.sum(expandedMask)
        MLX.eval(expandedSum)
        tokenizerLogger.debug("post-expand attention sum=\(expandedSum.item(Int.self))")
        if batch.attentionMask.dim(0) == 2 {
          let row0 = MLX.sum(expandedMask[0, 0...])
          let row1 = MLX.sum(expandedMask[1, 0...])
          MLX.eval(row0)
          MLX.eval(row1)
          tokenizerLogger.debug("post-expand per-row sums=\(row0.item(Int.self)), \(row1.item(Int.self))")
        }
      }
      for (rowIndex, offsets) in placeholderOffsets.enumerated() {
        let lengths = placeholderSpanLengths[rowIndex]
        tokenizerLogger.debug("placeholderOffsets row=\(rowIndex) count=\(offsets.count) values=\(offsets) lengths=\(lengths)")
      }
  }

  let useJoint = (visionContext?.placeholderCount ?? 0) > 0
    var jointUsed = false
    var promptEmbeddings: MLXArray
    var encoderMask: MLXArray
    var templateDrop = textEncoder?.configuration.promptDropIndex ?? 0
    if let tokenizer {
      templateDrop = max(templateDrop, tokenizer.templateTokenCount + 1)
    }

    if useJoint, let textEncoder, let visionContext, visionContext.placeholderCount > 0 {
      guard let tokenizer else {
        throw PipelineError.componentNotLoaded("Tokenizer")
      }
      guard let imageTokenId = tokenizer.imageTokenId, let visionStartTokenId = tokenizer.visionStartTokenId else {
        throw PipelineError.visionSpecialTokensUnavailable
      }
      let tower = try ensureVisionTower()
      textEncoder.setVisionTower(tower)
      let joint = try textEncoder.encodeMultimodal(
        inputIds: batch.inputIds,
        attentionMask: batch.attentionMask,
        pixelValues: visionContext.patchInputs,
        grids: visionContext.grids,
        gridTHW: visionContext.gridTHW,
        tokenCounts: visionContext.tokenCounts,
        imageTokenId: imageTokenId,
        visionStartTokenId: visionStartTokenId,
        spatialMergeSize: visionConfiguration.spatialMergeSize,
        dropIndex: templateDrop
      )
      promptEmbeddings = joint.0
      encoderMask = joint.1
      jointUsed = true
      pipelineLogger.debug("encode: joint text+vision path enabled")
    } else {
      let out = try encode(
        inputIds: batch.inputIds,
        attentionMask: batch.attentionMask
      )
      promptEmbeddings = out.0
      encoderMask = out.1
    }
    if let textEncoder, !jointUsed {
      let configuredDrop = textEncoder.configuration.promptDropIndex
      var drop = configuredDrop
      drop = max(drop, templateDrop)
      pipelineLogger.debug("encode: pre-drop embeddings shape=\(promptEmbeddings.shape) dropIndex=\(drop)")
      if drop > 0 {
        placeholderOffsets = placeholderOffsets.map { row in
          row.map { max(0, $0 - drop) }
        }
      }
      if drop > configuredDrop {
        let extra = drop - configuredDrop
        var keepLengths: [Int] = []
        keepLengths.reserveCapacity(promptEmbeddings.dim(0))
        let lengthArray = MLX.sum(encoderMask.asType(.int32), axis: 1).asType(.int32)
        MLX.eval(lengthArray)
        let lengths = lengthArray.asArray(Int32.self)
        for row in 0..<promptEmbeddings.dim(0) {
          keepLengths.append(max(0, Int(lengths[row]) - extra))
        }
        let maxKeep = keepLengths.max() ?? 0
        if maxKeep >= 0 {
          var trimmedEmbeds: [MLXArray] = []
          var trimmedMasks: [MLXArray] = []
          trimmedEmbeds.reserveCapacity(promptEmbeddings.dim(0))
          trimmedMasks.reserveCapacity(promptEmbeddings.dim(0))
          let hiddenDim = promptEmbeddings.dim(2)
          for row in 0..<promptEmbeddings.dim(0) {
            let keep = keepLengths[row]
            var slice: MLXArray
            if keep > 0 {
              let upper = keep + extra
              slice = promptEmbeddings[row, extra..<upper, 0...]
            } else {
              slice = MLX.zeros([0, hiddenDim], dtype: promptEmbeddings.dtype)
            }
            if keep < maxKeep {
              let pad = MLX.zeros([maxKeep - keep, hiddenDim], dtype: promptEmbeddings.dtype)
              slice = MLX.concatenated([slice, pad], axis: 0)
            }
            trimmedEmbeds.append(slice)

            let maskRow: MLXArray
            if keep == 0 {
              maskRow = MLX.zeros([maxKeep], dtype: .int32)
            } else if keep == maxKeep {
              maskRow = MLX.ones([maxKeep], dtype: .int32)
            } else {
              let ones = MLX.ones([keep], dtype: .int32)
              let zeros = MLX.zeros([maxKeep - keep], dtype: .int32)
              maskRow = MLX.concatenated([ones, zeros], axis: 0)
            }
            trimmedMasks.append(maskRow)
          }
          promptEmbeddings = MLX.stacked(trimmedEmbeds, axis: 0)
          encoderMask = MLX.stacked(trimmedMasks, axis: 0)
        }
      }
    }

    if !jointUsed, let visionContext, visionContext.placeholderCount > 0 {
      let repeatCounts = visionContext.tokenCounts
      let replacements: [MLXArray]
      do {
        let tower = try ensureVisionTower()
        replacements = try visionContext.visionEmbeddings(using: tower, dtype: promptEmbeddings.dtype)
      } catch {
        throw PipelineError.invalidTensorShape("Failed to build vision embeddings for placeholder fallback: \(error)")
      }
      precondition(replacements.count == repeatCounts.count, "Mismatch between vision embeddings and repeat counts.")
      var updatedRows: [MLXArray] = []
      updatedRows.reserveCapacity(promptEmbeddings.dim(0))
      for row in 0..<promptEmbeddings.dim(0) {
        let offsets = placeholderOffsets[row]
        if offsets.count != repeatCounts.count {
          tokenizerLogger.warning("placeholder mismatch row \(row) offsets=\(offsets) repeatCounts=\(repeatCounts)")
        }
        let limit = min(offsets.count, repeatCounts.count)
        if limit == 0 {
          updatedRows.append(promptEmbeddings[row, 0..., 0...])
          continue
        }
        let seqLen = promptEmbeddings.dim(1)
        var cursor = 0
        let rowEmbeddings = promptEmbeddings[row, 0..., 0...]
        var segments: [MLXArray] = []
        segments.reserveCapacity(limit * 2 + 1)
        let spanLengths = placeholderSpanLengths[row]
        for index in 0..<limit {
          let start = offsets[index]
          guard start >= cursor else {
            tokenizerLogger.warning("placeholder start < cursor row=\(row) start=\(start) cursor=\(cursor)")
            throw PipelineError.visionPlaceholderMismatch
          }
          if start > cursor {
            let before = rowEmbeddings[cursor..<start, 0...]
            segments.append(before)
          }
          let removal = index < spanLengths.count ? spanLengths[index] : repeatCounts[index]
          let count = repeatCounts[index]
          guard count >= 0 else {
            tokenizerLogger.warning("negative repeat count row=\(row) index=\(index) count=\(count)")
            throw PipelineError.visionPlaceholderMismatch
          }
          let limit = start + removal
          if limit > seqLen {
            tokenizerLogger.warning("replacement extends past sequence row=\(row) limit=\(limit) seqLen=\(seqLen)")
            throw PipelineError.visionPlaceholderMismatch
          }
          let replacement = replacements[index]
          if replacement.dim(0) < count {
            tokenizerLogger.warning("replacement too short row=\(row) index=\(index) replacementRows=\(replacement.dim(0)) count=\(count)")
            throw PipelineError.visionPlaceholderMismatch
          }
          let replacementSlice = replacement[0..<count, 0...]
          segments.append(replacementSlice)
          cursor = limit
        }
        if cursor < seqLen {
          let tail = rowEmbeddings[cursor..<seqLen, 0...]
          segments.append(tail)
      }
      let updated = segments.count == 1 ? segments[0] : MLX.concatenated(segments, axis: 0)
      updatedRows.append(updated)
    }
      promptEmbeddings = MLX.stacked(updatedRows, axis: 0)
      tokenizerLogger.debug("placeholder replacement finished; new promptEmbeddings shape=\(promptEmbeddings.shape)")
    }

    let result = QwenPromptEncodingResult(
      tokenBatch: batch,
      promptEmbeddings: promptEmbeddings,
      encoderAttentionMask: encoderMask
    )
    do {
      return try result.guidanceEncoding()
    } catch {
      throw PipelineError.invalidTensorShape("Guidance encoding requires at least one unconditional and one conditional sequence.")
    }
  }

#if canImport(CoreGraphics)
  public func generateImage(
    parameters: GenerationParameters,
    model: QwenModelConfiguration,
    maxPromptLength: Int? = nil,
    seed: UInt64? = nil
  ) throws -> PipelineImage {
    let pixels = try generatePixels(
      parameters: parameters,
      model: model,
      maxPromptLength: maxPromptLength,
      seed: seed
    )
    let cgImage = try QwenImageIO.image(from: pixels)
    return Self.makePipelineImage(from: cgImage)
  }

  public func makeImage(from pixels: MLXArray) throws -> PipelineImage {
    let cgImage = try QwenImageIO.image(from: pixels)
    return Self.makePipelineImage(from: cgImage)
  }
#endif

  public func generatePixels(
    parameters: GenerationParameters,
    model: QwenModelConfiguration,
    maxPromptLength: Int? = nil,
    seed: UInt64? = nil
  ) throws -> MLXArray {
    guard let tokenizer else { throw PipelineError.componentNotLoaded("Tokenizer") }
    guard let textEncoder else { throw PipelineError.componentNotLoaded("TextEncoder") }

    let (scheduler, runtime) = QwenSchedulerFactory.flowMatchSchedulerRuntime(
      model: model,
      generation: parameters
    )

    let promptLength = min(maxPromptLength ?? model.maxSequenceLength, model.maxSequenceLength)
    let guidanceEncoding = try encodeGuidancePrompts(
      prompt: parameters.prompt,
      negativePrompt: parameters.negativePrompt,
      maxLength: promptLength
    )
    let stacked = guidanceEncoding.stackedEmbeddings()
    let embeddings = stacked.embeddings.asType(preferredWeightDType() ?? stacked.embeddings.dtype)
    let attentionMask = stacked.attentionMask.asType(.int32)

    MLX.eval(embeddings, attentionMask)
    offloadEncoderComponents()

    try ensureUNetAndVAE(model: model)

    var latents = try makeInitialLatents(
      height: runtime.height,
      width: runtime.width,
      sigmas: scheduler.sigmas,
      seed: seed
    )

    for step in 0..<parameters.steps {
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
      publish(step: step + 1, total: parameters.steps)
    }

    let decoded = try decodeLatents(latents)
    return MLX.clip(decoded, min: 0, max: 1)
  }

#if canImport(CoreGraphics)
  public func generateEditedPixels(
    parameters: GenerationParameters,
    model: QwenModelConfiguration,
    referenceImage: CGImage,
    maxPromptLength: Int? = nil,
    seed: UInt64? = nil
  ) throws -> MLXArray {
    guard let tokenizer else { throw PipelineError.componentNotLoaded("Tokenizer") }
    guard let textEncoder else { throw PipelineError.componentNotLoaded("TextEncoder") }

    let (scheduler, runtime) = QwenSchedulerFactory.flowMatchSchedulerRuntime(
      model: model,
      generation: parameters
    )
    let promptLength = min(maxPromptLength ?? model.maxSequenceLength, model.maxSequenceLength)
    let preferredDType = preferredWeightDType() ?? .float32
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
    let finalReferenceDType = preferredWeightDType() ?? guidanceEncoding.unconditionalEmbeddings.dtype
    let referenceTokens = referenceContext.referenceTokens.asType(finalReferenceDType)
    pipelineLogger.debug("edit: reference tokens shape=\(referenceTokens.shape) segments=\(referenceContext.imageSegments)")

    MLX.eval(
      guidanceEncoding.unconditionalEmbeddings,
      guidanceEncoding.conditionalEmbeddings,
      guidanceEncoding.unconditionalMask,
      guidanceEncoding.conditionalMask
    )
    offloadEncoderComponents()

    try ensureUNetAndVAE(model: model)

    let latents = try makeInitialLatents(
      height: runtime.height,
      width: runtime.width,
      sigmas: scheduler.sigmas,
      seed: seed
    )

    let finalLatents = try runEditDenoiseLoop(
      latents: latents,
      parameters: parameters,
      runtime: runtime,
      scheduler: scheduler,
      unconditionalEmbeddings: guidanceEncoding.unconditionalEmbeddings,
      conditionalEmbeddings: guidanceEncoding.conditionalEmbeddings,
      unconditionalMask: guidanceEncoding.unconditionalMask,
      conditionalMask: guidanceEncoding.conditionalMask,
      referenceTokens: referenceTokens,
      imageSegments: referenceContext.imageSegments.isEmpty ? nil : referenceContext.imageSegments
    )

    var decoded = try decodeLatents(finalLatents)
    // Strip batch dimension so callers see [3,H,W] pixels.
    if decoded.ndim == 4 {
      decoded = decoded[0, 0..., 0..., 0...]
    }

    if let editRes = parameters.editResolution {
      let targetWidth = parameters.width
      let targetHeight = parameters.height
      if (targetWidth != runtime.width || targetHeight != runtime.height) &&
         (targetWidth > 0 && targetHeight > 0) {
        decoded = try QwenImageIO.resize(
          rgbArray: decoded,
          targetHeight: targetHeight,
          targetWidth: targetWidth
        )
      }
    }

    return MLX.clip(decoded, min: 0, max: 1)
  }
#endif

#if canImport(CoreGraphics)
  /// Multi-image edit: accepts two reference images and concatenates their tokens.
  /// Both references are resized to the same canvas size computed from the first image
  /// (rounded to multiples of 16, targeting the output canvas area).
  public func generateEditedPixels(
    parameters: GenerationParameters,
    model: QwenModelConfiguration,
    referenceImages: [CGImage],
    maxPromptLength: Int? = nil,
    seed: UInt64? = nil
  ) throws -> MLXArray {
    guard let tokenizer else { throw PipelineError.componentNotLoaded("Tokenizer") }
    guard let textEncoder else { throw PipelineError.componentNotLoaded("TextEncoder") }

    precondition(!referenceImages.isEmpty, "At least one reference image required.")

    let (scheduler, runtime) = QwenSchedulerFactory.flowMatchSchedulerRuntime(
      model: model,
      generation: parameters
    )
    let promptLength = min(maxPromptLength ?? model.maxSequenceLength, model.maxSequenceLength)
    let limitedReferences = Array(referenceImages.prefix(2))
    let preferredDType = preferredWeightDType() ?? .float32
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
    let finalReferenceDType = preferredWeightDType() ?? guidanceEncoding.unconditionalEmbeddings.dtype
    let referenceTokens = referenceContext.referenceTokens.asType(finalReferenceDType)
    pipelineLogger.debug("edit: reference tokens shape=\(referenceTokens.shape) segments=\(referenceContext.imageSegments)")
    MLX.eval(
      guidanceEncoding.unconditionalEmbeddings,
      guidanceEncoding.conditionalEmbeddings,
      guidanceEncoding.unconditionalMask,
      guidanceEncoding.conditionalMask
    )
    offloadEncoderComponents()
    try ensureUNetAndVAE(model: model)

    var latents = try makeInitialLatents(
      height: runtime.height,
      width: runtime.width,
      sigmas: scheduler.sigmas,
      seed: seed
    )

    let finalLatents = try runEditDenoiseLoop(
      latents: latents,
      parameters: parameters,
      runtime: runtime,
      scheduler: scheduler,
      unconditionalEmbeddings: guidanceEncoding.unconditionalEmbeddings,
      conditionalEmbeddings: guidanceEncoding.conditionalEmbeddings,
      unconditionalMask: guidanceEncoding.unconditionalMask,
      conditionalMask: guidanceEncoding.conditionalMask,
      referenceTokens: referenceTokens,
      imageSegments: referenceContext.imageSegments
    )

    var decoded = try decodeLatents(finalLatents)
    if decoded.ndim == 4 {
      decoded = decoded[0, 0..., 0..., 0...]
    }

    if let editRes = parameters.editResolution {
      let targetWidth = parameters.width
      let targetHeight = parameters.height
      if (targetWidth != runtime.width || targetHeight != runtime.height) &&
         (targetWidth > 0 && targetHeight > 0) {
        decoded = try QwenImageIO.resize(
          rgbArray: decoded,
          targetHeight: targetHeight,
          targetWidth: targetWidth
        )
      }
    }

    return MLX.clip(decoded, min: 0, max: 1)
  }
#endif

#if canImport(CoreGraphics)
  private func prepareVisionPromptContext(
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

    let stackedPatches = patchInputs.count == 1
      ? patchInputs[0]
      : MLX.concatenated(patchInputs, axis: 0)
    visionLogger.debug("stacked patch input shape=\(stackedPatches.shape)")
    return VisionPromptContext(
      patchInputs: stackedPatches,
      grids: grids,
      tokenCounts: tokenCounts,
      gridTHW: gridTHWList
    )
  }
#endif

  private func prepareEditReferenceContext(
    referenceImage: CGImage,
    runtime: QwenRuntimeConfig,
    dtype: DType,
    referenceTargetArea: Int
  ) throws -> EditReferenceContext {
    pipelineLogger.debug("edit: reference input size=\(referenceImage.width)x\(referenceImage.height)")
    let visionImage = referenceImage
    var target = EditSizing.computeVAEDimensions(
      referenceWidth: referenceImage.width,
      referenceHeight: referenceImage.height,
      targetArea: referenceTargetArea
    )
    pipelineLogger.debug("edit: resized reference to \(target.width)x\(target.height)")
    var pixels = try QwenImageIO.resizedPixelArray(
      from: referenceImage,
      width: target.width,
      height: target.height
    )
    pipelineLogger.debug("edit: reference pixel array shape=\(pixels.shape) dtype=\(pixels.dtype)")
    let (latents, quantHidden, encoderHidden) = try encodePixelsWithIntermediates(pixels)
    pipelineLogger.debug("edit: encoded latents shape=\(latents.shape)")
    let packed = LatentUtilities.packLatents(
      latents,
      height: target.height,
      width: target.width
    ).asType(dtype)
    let packedBatch = packed.reshaped(1, packed.dim(1), packed.dim(2))
    var referenceTokens = MLX.concatenated([packedBatch, packedBatch], axis: 0)
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
      visionConditionImages: [visionImage],
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

  private func runEditDenoiseLoop(
    latents initialLatents: MLXArray,
    parameters: GenerationParameters,
    runtime: QwenRuntimeConfig,
    scheduler: QwenFlowMatchScheduler,
    unconditionalEmbeddings rawUncondEmbeddings: MLXArray,
    conditionalEmbeddings rawCondEmbeddings: MLXArray,
    unconditionalMask rawUncondMask: MLXArray,
    conditionalMask rawCondMask: MLXArray,
    referenceTokens rawReferenceTokens: MLXArray,
    imageSegments: [(Int, Int, Int)]?
  ) throws -> MLXArray {
    guard let unet else {
      throw PipelineError.componentNotLoaded("UNet")
    }
    var latents = initialLatents
    var unconditionalEmbeddings = rawUncondEmbeddings.asType(latents.dtype)
    var conditionalEmbeddings = rawCondEmbeddings.asType(latents.dtype)
    var unconditionalMask = rawUncondMask.asType(.int32)
    var conditionalMask = rawCondMask.asType(.int32)
    var referenceTokens = rawReferenceTokens.asType(latents.dtype)

    // Ensure we have per-branch reference tokens (uncond/cond). If only a single
    // row is provided, share it across both branches.
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
    var segments: [(Int, Int, Int)] = []
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

      // Conditional branch
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

      // Unconditional branch
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
      publish(step: step + 1, total: parameters.steps)
    }

    return latents
  }

  private func ensureVisionTower() throws -> QwenVisionTower {
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

  private func latentsStatistics(_ latents: MLXArray) -> (mean: Double, std: Double) {
    let flattened = latents.asType(.float32).flattened()
    if flattened.shape.contains(0) {
      return (0, 0)
    }
    let meanArray = MLX.mean(flattened)
    MLX.eval(meanArray)
    let mean = meanArray.item(Double.self)
    let centered = flattened - meanArray
    let varianceArray = MLX.mean(centered * centered)
    MLX.eval(varianceArray)
    let variance = max(varianceArray.item(Double.self), 0)
    return (mean, variance.squareRoot())
  }

  private func countNonFinite(_ array: MLXArray) -> Int {
    let finiteMask = MLX.isFinite(array)
    let nonFinite = MLX.logicalNot(finiteMask)
    let sumArray = MLX.sum(nonFinite.asType(.int32))
    MLX.eval(sumArray)
    return sumArray.item(Int.self)
  }

  /// Release large encoder-side components once prompt / reference encodings
  /// have been computed, so the UNet denoising loop can run with a smaller
  /// resident set. This primarily targets the text encoder and vision tower.
  private func offloadEncoderComponents() {
    textEncoder = nil
    textEncoderQuantization = nil
    textEncoderRuntimeQuantized = false
    visionTower = nil
    visionRuntimeQuantized = false
  }

  /// Lazily load UNet and VAE from the base weights directory (or existing
  /// transformerDirectory) after encoder components have been offloaded.
  private func ensureUNetAndVAE(model: QwenModelConfiguration) throws {
    if unet != nil && vae != nil { return }
    guard let directory = transformerDirectory ?? baseWeightsDirectory else {
      throw PipelineError.componentNotLoaded("UNet/VAE weights directory")
    }
    let transformerConfig = QwenTransformerConfiguration()
    if unet == nil {
      try prepareUNet(from: directory, configuration: transformerConfig)
    }
    if vae == nil {
      try prepareVAE(from: directory)
    }
    if let url = pendingLoraURL {
      try applyLora(from: url, scale: pendingLoraScale)
      pendingLoraURL = nil
    }
  }

  public func encode(
    inputIds: MLXArray,
    attentionMask: MLXArray?
  ) throws -> (promptEmbeddings: MLXArray, attentionMask: MLXArray) {
    guard let textEncoder else {
      throw PipelineError.componentNotLoaded("TextEncoder")
    }
    return textEncoder.encode(inputIds: inputIds, attentionMask: attentionMask)
  }

  private func loadTokenizer(from snapshot: HubSnapshot) async throws {
    let directory = try await snapshot.prepare { [weak self] progress in
      self?.publish(progress: progress)
    }
    let tokenizer = try QwenTokenizer.load(from: directory)
    self.tokenizer = tokenizer
  }

  private func loadTextEncoder(from snapshot: HubSnapshot) async throws {
    let directory = try await snapshot.prepare { [weak self] progress in
      self?.publish(progress: progress)
    }
    let encoder = QwenTextEncoder()
    let configURL = directory.appending(path: "text_encoder").appending(path: "config.json")
    let quantPlan = combinedQuantizationPlan(
      configURL: configURL,
      snapshotRoot: directory,
      componentName: "text_encoder"
    )
    textEncoderQuantization = quantPlan
    try weightsLoader.loadTextEncoder(
      fromDirectory: directory,
      into: encoder,
      dtype: preferredWeightDType(),
      quantization: quantPlan
    )
    textEncoder = encoder
    applyRuntimeQuantizationIfNeeded(to: encoder, flag: &textEncoderRuntimeQuantized)
  }

  private func loadTransformer(
    from snapshot: HubSnapshot,
    configuration: QwenTransformerConfiguration
  ) async throws {
    let directory = try await snapshot.prepare { [weak self] progress in
      self?.publish(progress: progress)
    }
    let transformer = QwenTransformer(configuration: configuration)
    let dtype = preferredWeightDType()
    let configURL = directory.appending(path: "transformer").appending(path: "config.json")
    let quantPlan = combinedQuantizationPlan(
      configURL: configURL,
      snapshotRoot: directory,
      componentName: "transformer"
    )
    transformerQuantization = quantPlan
    try weightsLoader.loadTransformer(
      fromDirectory: directory,
      into: transformer,
      dtype: dtype,
      quantization: quantPlan
    )
    transformerDirectory = directory
    self.transformer = transformer
    visionTower = nil
    applyRuntimeQuantizationIfNeeded(to: transformer, flag: &transformerRuntimeQuantized)
    transformer.setAttentionQuantization(attentionQuantizationSpec)
  }

  private func loadUNet(
    from snapshot: HubSnapshot,
    configuration: QwenTransformerConfiguration
  ) async throws {
    let directory = try await snapshot.prepare { [weak self] progress in
      self?.publish(progress: progress)
    }
    let unet = QwenUNet(configuration: configuration)
    let dtype = preferredWeightDType()
    let configURL = directory.appending(path: "transformer").appending(path: "config.json")
    let quantPlan = combinedQuantizationPlan(
      configURL: configURL,
      snapshotRoot: directory,
      componentName: "transformer"
    )
    transformerQuantization = quantPlan
    try weightsLoader.loadUNet(
      fromDirectory: directory,
      into: unet,
      dtype: dtype,
      quantization: quantPlan
    )
    self.unet = unet
    transformerDirectory = directory
    visionTower = nil
    applyRuntimeQuantizationIfNeeded(to: unet.transformer, flag: &transformerRuntimeQuantized)
    unet.transformer.setAttentionQuantization(attentionQuantizationSpec)
  }

  private func loadVAE(from snapshot: HubSnapshot) async throws {
    let vae = QwenVAE()
    let dtype = preferredWeightDType()
    try await weightsLoader.loadVAE(from: snapshot, into: vae, dtype: dtype) { [weak self] progress in
      self?.publish(progress: progress)
    }
    self.vae = vae
  }

  private func preferredWeightDType() -> DType? {
    return .bfloat16
  }

  private func publish(progress: HubSnapshotProgress) {
    if progressSubject == nil {
      progressSubject = .init()
    }
    let info = ProgressInfo(step: Int(progress.completedUnitCount), total: Int(progress.totalUnitCount))
    progressSubject?.send(info)
  }

  private func normalizeForEncoder(_ array: MLXArray) -> MLXArray {
    array * MLXArray(Float32(2.0)) - MLXArray(Float32(1.0))
  }

  private func denormalizeFromDecoder(_ array: MLXArray) -> MLXArray {
    (array + MLXArray(Float32(1.0))) / MLXArray(Float32(2.0))
  }

  private func applyRuntimeQuantizationIfNeeded(
    to module: Module?,
    flag: inout Bool
  ) {
    guard let spec = runtimeQuantizationSpec, !flag, let module else { return }
    guard module is QwenTransformer else {
      return
    }
    if quantizeModule(module, spec: spec) {
      flag = true
    }
  }

  private func quantizeModule(_ module: Module, spec: QwenQuantizationSpec) -> Bool {
    let supportedGroupSizes: Set<Int> = [32, 64, 128]
    guard supportedGroupSizes.contains(spec.groupSize) else {
      pipelineLogger.warning(
        "Runtime quantization skipped: group size \(spec.groupSize) is not supported (allowed: \(supportedGroupSizes.sorted())).")
      return false
    }
    var quantizedLayerCount = 0
    quantize(
      model: module,
      filter: { path, submodule in
        guard runtimeQuantizationAllowed(path: path) else { return nil }
        guard let tuple = runtimeQuantizationTuple(for: submodule, spec: spec, path: path) else { return nil }
        quantizedLayerCount += 1
        return tuple
      }
    )
    if quantizedLayerCount == 0 {
      pipelineLogger.warning(
        "Runtime quantization skipped: no layers in \(type(of: module)) satisfied the group size requirement.")
      return false
    }
    pipelineLogger.info(
      "Runtime quantized \(quantizedLayerCount) layers in \(type(of: module)) (\(spec.bits)-bit, group size \(spec.groupSize), mode \(spec.mode)).")
    return true
  }

  private func runtimeQuantizationTuple(
    for module: Module,
    spec: QwenQuantizationSpec,
    path: String
  ) -> (groupSize: Int, bits: Int, mode: QuantizationMode)? {
    if let linear = module as? Linear {
      let inputDims = linear.weight.dim(1)
      guard inputDims % spec.groupSize == 0 else { return nil }
      guard inputDims >= runtimeQuantMinInputDim else { return nil }
      return spec.tuple
    }
    if let embedding = module as? Embedding {
      let dim = embedding.weight.dim(1)
      guard dim % spec.groupSize == 0 else { return nil }
      guard dim >= runtimeQuantMinInputDim else { return nil }
      return spec.tuple
    }
    return nil
  }

  /// Apply a LoRA adapter stored in a safetensors file to the UNet transformer (and standalone
  /// transformer if loaded).
  ///
  /// The file is expected to use HF-style keys following the pattern:
  /// `transformer_blocks.N.*.{to_q,to_k,to_v,to_add_out,to_out.0,img_mlp.*,txt_mlp.*}.{alpha,lora_down.weight,lora_up.weight}`.
  ///
  /// This matches the naming used by Qwen-Image Lightning adapters such as:
  /// `Osrivers/Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors`.
  public func applyLora(from fileURL: URL, scale: Float = 1.0) throws {
    guard let unet else {
      throw PipelineError.componentNotLoaded("UNet")
    }
    let layers = try Self.loadLoraLayers(from: fileURL)
    if layers.isEmpty {
      pipelineLogger.warning("LoRA: no adapter layers found in \(fileURL.path)")
      throw PipelineError.invalidTensorShape("LoRA file \(fileURL.lastPathComponent) contained no recognized transformer blocks.")
    }

    pipelineLogger.info("LoRA: loaded \(layers.count) adapter bases from \(fileURL.lastPathComponent)")

    let appliedUNet = Self.applyLoraLayers(layers, to: unet.transformer, globalScale: scale)
    pipelineLogger.info("LoRA: applied to UNet transformer (\(appliedUNet) layers updated).")

    if let transformer {
      let appliedTransformer = Self.applyLoraLayers(layers, to: transformer, globalScale: scale)
      if appliedTransformer > 0 {
        pipelineLogger.info("LoRA: applied to standalone transformer (\(appliedTransformer) layers updated).")
      }
    }
  }

  private static func loadLoraLayers(from fileURL: URL) throws -> QwenLoraLayerMap {
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

    var layers: QwenLoraLayerMap = [:]
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

  /// Map internal transformer module paths to HF-style base keys used by LoRA and quantization.
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
    _ layers: QwenLoraLayerMap,
    to transformer: QwenTransformer,
    globalScale: Float
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
        pipelineLogger.debug("LoRA: applying to quantized layer \(path)")
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
        pipelineLogger.debug("LoRA: fusing into linear layer \(path)")
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

  private func makeInitialLatents(
    height: Int,
    width: Int,
    sigmas: MLXArray,
    seed: UInt64?
  ) throws -> MLXArray {
    let latentHeight = max(1, height / 8)
    let latentWidth = max(1, width / 8)
    let dtype = preferredWeightDType() ?? .float32
    var latents: MLXArray
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

  private func publish(step: Int, total: Int) {
    if progressSubject == nil {
      progressSubject = .init()
    }
    let info = ProgressInfo(step: step, total: total)
    progressSubject?.send(info)
  }

#if canImport(CoreGraphics)
  private static func makeCGImage(from image: PipelineImage) throws -> CGImage {
#if canImport(AppKit)
    guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
      throw PipelineError.imageConversionFailed
    }
    return cgImage
#elseif canImport(UIKit)
    guard let cgImage = image.cgImage else {
      throw PipelineError.imageConversionFailed
    }
    return cgImage
#else
    throw PipelineError.imageConversionUnavailable
#endif
  }

  private static func makePipelineImage(from cgImage: CGImage) -> PipelineImage {
#if canImport(AppKit)
    PipelineImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
#elseif canImport(UIKit)
    PipelineImage(cgImage: cgImage)
#else
    cgImage as AnyObject
#endif
  }
#endif

  func setTokenizerForTesting(_ tokenizer: QwenTokenizer) {
    self.tokenizer = tokenizer
  }

  func setVAEForTesting(_ vae: QwenVAE) {
    self.vae = vae
  }

  func setVisionTowerForTesting(_ tower: QwenVisionTower?) {
    self.visionTower = tower
  }

  public func tokenizerForDebug() -> QwenTokenizer? {
    tokenizer
  }
}
