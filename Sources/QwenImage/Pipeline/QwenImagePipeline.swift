import Foundation
import Logging
import MLX

let pipelineLogger = QwenLogger.pipeline
let tokenizerLogger = QwenLogger.tokenizer
let visionLogger = QwenLogger.vision

public final class QwenImagePipeline {
  public enum PipelineError: Error {
    case componentNotLoaded(String)
    case imageConversionUnavailable
    case imageConversionFailed
    case invalidTensorShape(String)
    case invalidConfiguration(String)
    case visionSpecialTokensUnavailable
    case visionPlaceholderMismatch
  }

  let config: QwenImageConfig
  let weightsLoader = QwenWeightsLoader()

  var baseWeightsDirectory: URL?

  var tokenizer: QwenTokenizer?
  var textEncoder: QwenTextEncoder?
  var visionTokensPerSecond: Int = 2
  var transformer: QwenTransformer?
  var transformerDirectory: URL?
  var unet: QwenUNet?
  var vae: QwenVAE?
  let visionPreprocessor = QwenVisionPreprocessor()
  var visionTower: QwenVisionTower?
  let visionConfiguration = QwenVisionConfiguration()
  var runtimeQuantizationSpec: QwenQuantizationSpec?
  var runtimeQuantMinInputDim: Int = 0
  let runtimeQuantAllowedSuffixes: [String] = [
    ".attn.to_q",
    ".attn.to_k",
    ".attn.to_v",
    ".attn.add_q_proj",
    ".attn.add_k_proj",
    ".attn.add_v_proj",
    ".attn.attn_to_out",
    ".attn.to_add_out",
    ".img_norm1.mod_linear",
    ".txt_norm1.mod_linear",
    ".img_ff.mlp_in",
    ".img_ff.mlp_out",
    ".txt_ff.mlp_in",
    ".txt_ff.mlp_out",
    ".norm_out.linear"
  ]
  var textEncoderRuntimeQuantized = false
  var transformerRuntimeQuantized = false
  var visionRuntimeQuantized = false
  var textEncoderQuantization: QwenQuantizationPlan?
  var transformerQuantization: QwenQuantizationPlan?
  var attentionQuantizationSpec: QwenQuantizationSpec?
  var dtypeOverride: DType?
  var textEncoderConfiguration: QwenTextEncoderConfiguration?
  var modelDescriptor: QwenModelDescriptor?

  var pendingLoraURL: URL?
  var pendingLoraScale: Float = 1.0

  public init(config: QwenImageConfig) {
    self.config = config
  }

  public func setDTypeOverride(_ dtype: DType?) {
    dtypeOverride = dtype
  }

  public func setBaseDirectory(_ directory: URL) {
    if baseWeightsDirectory != directory {
      modelDescriptor = nil
      textEncoderConfiguration = nil
      transformerQuantization = nil
      textEncoderQuantization = nil
    }
    baseWeightsDirectory = directory
    transformerDirectory = directory
  }

  public func loadModelDescriptor() throws -> QwenModelDescriptor {
    guard let directory = transformerDirectory ?? baseWeightsDirectory else {
      throw PipelineError.componentNotLoaded("ModelDirectory")
    }
    return try ensureModelDescriptor(from: directory)
  }

  public func resolvedDType() throws -> DType {
    let descriptor = try loadModelDescriptor()
    if let dtypeOverride {
      return dtypeOverride
    }
    if let textEncoderConfiguration {
      return textEncoderConfiguration.outputDType
    }
    return descriptor.textEncoderConfiguration.outputDType
  }

  public func promptEncodingQuantizationId() -> String {
    var parts: [String] = []
    if let plan = textEncoderQuantization, plan.isEnabled {
      if let spec = plan.defaultSpec ?? plan.prepackedLayers.values.first?.spec {
        parts.append("snap_q\(spec.bits)_\(spec.groupSize)_\(spec.mode)")
      } else {
        parts.append("snap_q")
      }
    }
    if textEncoderRuntimeQuantized, let spec = runtimeQuantizationSpec {
      parts.append("rt_q\(spec.bits)_\(spec.groupSize)_\(spec.mode)")
    }
    if parts.isEmpty {
      return "none"
    }
    return parts.joined(separator: "+")
  }

  func ensureModelDescriptor(from root: URL) throws -> QwenModelDescriptor {
    if let modelDescriptor {
      return modelDescriptor
    }
    let descriptor = try QwenModelDescriptor.load(from: root)
    modelDescriptor = descriptor
    if let tps = descriptor.visionTokensPerSecond {
      visionTokensPerSecond = tps
    }
    return descriptor
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
    let decoded = vae.decodeWithDenormalization(latents.asType(preferredWeightDType()))
    return denormalizeFromDecoder(decoded)
  }

  func encodePixelsWithIntermediates(_ pixels: MLXArray) throws -> (latents: MLXArray, quantHidden: MLXArray, encoderHidden: MLXArray) {
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

  public func encode(
    inputIds: MLXArray,
    attentionMask: MLXArray?
  ) throws -> (promptEmbeddings: MLXArray, attentionMask: MLXArray) {
    guard let textEncoder else {
      throw PipelineError.componentNotLoaded("TextEncoder")
    }
    return textEncoder.encode(inputIds: inputIds, attentionMask: attentionMask)
  }

  func preferredWeightDType() -> DType {
    if let dtypeOverride {
      return dtypeOverride
    }
    if let textEncoderConfiguration {
      return textEncoderConfiguration.outputDType
    }
    if let modelDescriptor {
      return modelDescriptor.textEncoderConfiguration.outputDType
    }
    return .bfloat16
  }

  private func normalizeForEncoder(_ array: MLXArray) -> MLXArray {
    let dtype = array.dtype
    let two = MLXArray(Float32(2.0)).asType(dtype)
    let one = MLXArray(Float32(1.0)).asType(dtype)
    return array * two - one
  }

  private func denormalizeFromDecoder(_ array: MLXArray) -> MLXArray {
    let dtype = array.dtype
    let one = MLXArray(Float32(1.0)).asType(dtype)
    let two = MLXArray(Float32(2.0)).asType(dtype)
    return (array + one) / two
  }

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
