import Foundation
import MLX
import MLXNN
import MLXFast

enum QwenTextEncoderError: Error {
  case visionTowerUnavailable
  case mismatchedVisionTokenCount
}

public struct QwenTextEncoderConfiguration {
  public var vocabSize: Int
  public var hiddenSize: Int
  public var numHiddenLayers: Int
  public var numAttentionHeads: Int
  public var numKeyValueHeads: Int
  public var intermediateSize: Int
  public var ropeTheta: Float
  public var maxPositionEmbeddings: Int
  public var rmsNormEps: Float
  public var promptDropIndex: Int
  public var outputDType: DType

  public init(
    vocabSize: Int = 152_064,
    hiddenSize: Int = 3_584,
    numHiddenLayers: Int = 28,
    numAttentionHeads: Int = 28,
    numKeyValueHeads: Int = 4,
    intermediateSize: Int = 18_944,
    ropeTheta: Float = 1_000_000.0,
    maxPositionEmbeddings: Int = 128_000,
    rmsNormEps: Float = 1e-6,
    promptDropIndex: Int = 34,
    outputDType: DType = .bfloat16
  ) {
    self.vocabSize = vocabSize
    self.hiddenSize = hiddenSize
    self.numHiddenLayers = numHiddenLayers
    self.numAttentionHeads = numAttentionHeads
    self.numKeyValueHeads = numKeyValueHeads
    self.intermediateSize = intermediateSize
    self.ropeTheta = ropeTheta
    self.maxPositionEmbeddings = maxPositionEmbeddings
    self.rmsNormEps = rmsNormEps
    self.promptDropIndex = promptDropIndex
    self.outputDType = outputDType
  }
}

public final class QwenTextEncoder: Module {

  public let configuration: QwenTextEncoderConfiguration
  @ModuleInfo(key: "encoder") var encoder: QwenEncoder
  private var visionTower: QwenVisionTower?

  public init(configuration: QwenTextEncoderConfiguration = .init()) {
    self.configuration = configuration
    self._encoder.wrappedValue = QwenEncoder(configuration: configuration)
  }

  func setVisionTower(_ tower: QwenVisionTower) {
    self.visionTower = tower
  }

  public func callAsFunction(
    inputIds: MLXArray,
    attentionMask: MLXArray? = nil
  ) -> (MLXArray, MLXArray) {
    encode(inputIds: inputIds, attentionMask: attentionMask)
  }

  public func encode(
    inputIds: MLXArray,
    attentionMask: MLXArray?
  ) -> (MLXArray, MLXArray) {
    let hiddenStates = encoder.forward(
      inputIds: inputIds,
      attentionMask: attentionMask
    )
    let processed = QwenTextEncoder.processTextEmbeddings(
      hiddenStates: hiddenStates,
      attentionMask: attentionMask,
      dropIndex: configuration.promptDropIndex,
      dtype: configuration.outputDType
    )
    return processed
  }

  static func processTextEmbeddings(
    hiddenStates: MLXArray,
    attentionMask: MLXArray?,
    dropIndex: Int,
    dtype: DType
  ) -> (MLXArray, MLXArray) {
    let batchSize = hiddenStates.dim(0)
    let seqLen = hiddenStates.dim(1)
    let hiddenDim = hiddenStates.dim(2)

    var mask: MLXArray
    if let attentionMask {
      mask = attentionMask
    } else {
      mask = MLX.ones([batchSize, seqLen], dtype: .int32)
    }
    if mask.dtype != .int32 {
      mask = mask.asType(.int32)
    }

    let trimmedStart = max(0, min(dropIndex, seqLen))
    let validLengthsArray = mask.sum(axis: 1).asType(.int32)
    MLX.eval(validLengthsArray)
    let validLengths = validLengthsArray.asArray(Int32.self)
    let trimmedLengths = validLengths.map { max(0, Int($0) - trimmedStart) }

    let maxTrimmedLength = trimmedLengths.max() ?? 0

    var paddedEmbeds: [MLXArray] = []
    paddedEmbeds.reserveCapacity(batchSize)
    var paddedMasks: [MLXArray] = []
    paddedMasks.reserveCapacity(batchSize)

    for batch in 0..<batchSize {
      let trimmedLength = trimmedLengths[batch]
      let sliceEnd = trimmedStart + trimmedLength

      var sampleEmbeds: MLXArray
      if trimmedLength > 0 {
        sampleEmbeds = hiddenStates[
          batch,
          trimmedStart ..< sliceEnd,
          0...
        ]
      } else {
        sampleEmbeds = MLX.zeros([0, hiddenDim], dtype: hiddenStates.dtype)
      }

      if trimmedLength < maxTrimmedLength {
        let pad = MLX.zeros([maxTrimmedLength - trimmedLength, hiddenDim], dtype: hiddenStates.dtype)
        sampleEmbeds = MLX.concatenated([sampleEmbeds, pad], axis: 0)
      }
      paddedEmbeds.append(sampleEmbeds)

      let sampleMask: MLXArray
      if trimmedLength == 0 {
        sampleMask = MLX.zeros([maxTrimmedLength], dtype: .int32)
      } else if trimmedLength == maxTrimmedLength {
        sampleMask = MLX.ones([maxTrimmedLength], dtype: .int32)
      } else {
        let leadZeros = MLX.zeros([maxTrimmedLength - trimmedLength], dtype: .int32)
        let tailOnes = MLX.ones([trimmedLength], dtype: .int32)
        sampleMask = MLX.concatenated([tailOnes, leadZeros], axis: 0)
      }
      paddedMasks.append(sampleMask)
    }

    let promptEmbeds = MLX.stacked(paddedEmbeds, axis: 0).asType(dtype)
    let encoderMask = MLX.stacked(paddedMasks, axis: 0)
    return (promptEmbeds, encoderMask)
  }

}

public final class QwenEncoder: Module {

  public let configuration: QwenTextEncoderConfiguration
  @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
  @ModuleInfo(key: "layers") var layers: [QwenEncoderLayer]
  @ModuleInfo(key: "norm") var norm: RMSNorm
  @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: QwenRotaryEmbedding

  public init(configuration: QwenTextEncoderConfiguration) {
    self.configuration = configuration
    self._embedTokens.wrappedValue = Embedding(
      embeddingCount: configuration.vocabSize, dimensions: configuration.hiddenSize)
    self._layers.wrappedValue = (0..<configuration.numHiddenLayers).map { layerIndex in
      QwenEncoderLayer(configuration: configuration, layerIndex: layerIndex)
    }
    self._norm.wrappedValue = RMSNorm(
      dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
    self._rotaryEmbedding.wrappedValue = QwenRotaryEmbedding(configuration: configuration)
  }

  public func callAsFunction(
    inputIds: MLXArray,
    attentionMask: MLXArray?
  ) -> MLXArray {
    forward(inputIds: inputIds, attentionMask: attentionMask)
  }

  public func forward(
    inputIds: MLXArray,
    attentionMask: MLXArray?
  ) -> MLXArray {
    let batchSize = inputIds.dim(0)
    let seqLen = inputIds.dim(1)

    var tokenIds = inputIds
    if tokenIds.dtype != .int32 {
      tokenIds = tokenIds.asType(.int32)
    }

    var hiddenStates = embedTokens(tokenIds)

    let cachePosition = MLXArray(0..<seqLen).asType(.int32)
    var positionIds = cachePosition[.newAxis, 0...]
    positionIds = MLX.repeated(positionIds, count: batchSize, axis: 0)
    positionIds = positionIds[.newAxis, .ellipsis]
    positionIds = MLX.repeated(positionIds, count: 3, axis: 0)

    let attentionMask4D = QwenEncoder.buildAttentionMask(
      mask: attentionMask,
      batchSize: batchSize,
      seqLen: seqLen
    )

    let positionEmbeddings = rotaryEmbedding(
      hiddenStates: hiddenStates,
      positionIds: positionIds
    )

    for (index, layer) in layers.enumerated() {
      hiddenStates = layer(
        hiddenStates: hiddenStates,
        attentionMask: attentionMask4D,
        positionEmbeddings: positionEmbeddings
      )
    }

    hiddenStates = norm(hiddenStates)
    return hiddenStates
  }

  static func buildAttentionMask(
    mask: MLXArray?,
    batchSize: Int,
    seqLen: Int
  ) -> MLXArray? {
    guard let mask else { return nil }
    let computeType = mask.dtype
    var attentionMask = mask
    if attentionMask.dtype != computeType {
      attentionMask = attentionMask.asType(computeType)
    }

    let zeros = MLX.zeros(attentionMask.shape).asType(computeType)
    let negInf = zeros + MLXArray(-Float.infinity)
    let keepMask = attentionMask .== MLXArray(1)
    var paddingMask = MLX.where(keepMask, zeros, negInf)
    paddingMask = MLX.expandedDimensions(paddingMask, axis: 1)
    paddingMask = MLX.expandedDimensions(paddingMask, axis: 1)

    let idx = MLXArray(0..<seqLen).asType(computeType)
    let j = idx[.newAxis, 0...]
    let i = idx[0..., .newAxis]
    let triBool = j .> i
    var causal = MLX.zeros([seqLen, seqLen]).asType(computeType)
    causal = MLX.where(triBool, causal + MLXArray(-Float.infinity), causal)
    causal = MLX.expandedDimensions(causal, axis: 0)
    causal = MLX.expandedDimensions(causal, axis: 0)
    causal = MLX.repeated(causal, count: batchSize, axis: 0)

    return causal + paddingMask
  }
}

public final class QwenEncoderLayer: Module {
  public let configuration: QwenTextEncoderConfiguration
  public let layerIndex: Int
  @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
  @ModuleInfo(key: "self_attn") var selfAttention: QwenAttention
  @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
  @ModuleInfo(key: "mlp") var mlp: QwenMLP

  init(configuration: QwenTextEncoderConfiguration, layerIndex: Int) {
    self.configuration = configuration
    self.layerIndex = layerIndex
    self._inputLayerNorm.wrappedValue = RMSNorm(
      dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
    self._selfAttention.wrappedValue = QwenAttention(configuration: configuration)
    self._postAttentionLayerNorm.wrappedValue = RMSNorm(
      dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
    self._mlp.wrappedValue = QwenMLP(configuration: configuration)
  }

  func callAsFunction(
    hiddenStates: MLXArray,
    attentionMask: MLXArray?,
    positionEmbeddings: (MLXArray, MLXArray)?
  ) -> MLXArray {
    var residual = hiddenStates
    var hiddenStates = inputLayerNorm(hiddenStates)
    let attnOut = selfAttention(
      hiddenStates: hiddenStates,
      attentionMask: attentionMask,
      positionEmbeddings: positionEmbeddings
    )
    let postAttention = attnOut + residual
    residual = postAttention
    let norm2Output = postAttentionLayerNorm(postAttention)
    let mlpOutput = mlp(norm2Output)
    let blockOutput = mlpOutput + residual
    return blockOutput
  }
}

public final class QwenAttention: Module {
  let hiddenSize: Int
  let numAttentionHeads: Int
  let numKeyValueHeads: Int
  let headDim: Int
  let numKeyValueGroups: Int
  let ropeSections: [Int]
  let scaling: Float

  @ModuleInfo(key: "q_proj") var qProj: Linear
  @ModuleInfo(key: "k_proj") var kProj: Linear
  @ModuleInfo(key: "v_proj") var vProj: Linear
  @ModuleInfo(key: "o_proj") var oProj: Linear

  init(configuration: QwenTextEncoderConfiguration) {
    self.hiddenSize = configuration.hiddenSize
    self.numAttentionHeads = configuration.numAttentionHeads
    self.numKeyValueHeads = configuration.numKeyValueHeads
    self.headDim = configuration.hiddenSize / configuration.numAttentionHeads
    self.numKeyValueGroups = configuration.numAttentionHeads / configuration.numKeyValueHeads
    self.ropeSections = [16, 24, 24]
    self.scaling = 1.0 / Float(headDim).squareRoot()

    self._qProj.wrappedValue = Linear(hiddenSize, numAttentionHeads * headDim)
    self._kProj.wrappedValue = Linear(hiddenSize, numKeyValueHeads * headDim)
    self._vProj.wrappedValue = Linear(hiddenSize, numKeyValueHeads * headDim)
    self._oProj.wrappedValue = Linear(numAttentionHeads * headDim, hiddenSize, bias: false)
  }

  func callAsFunction(
    hiddenStates: MLXArray,
    attentionMask: MLXArray?,
    positionEmbeddings: (MLXArray, MLXArray)?
  ) -> MLXArray {
    let batchSize = hiddenStates.dim(0)
    let seqLen = hiddenStates.dim(1)

    var queryStates = qProj(hiddenStates)
    let qLinear = queryStates
    queryStates = queryStates.reshaped(
      batchSize, seqLen, numAttentionHeads, headDim
    ).transposed(0, 2, 1, 3)

    var keyStates = kProj(hiddenStates)
    let kLinear = keyStates
    keyStates = keyStates.reshaped(
      batchSize, seqLen, numKeyValueHeads, headDim
    ).transposed(0, 2, 1, 3)

    var valueStates = vProj(hiddenStates)
    valueStates = valueStates.reshaped(
      batchSize, seqLen, numKeyValueHeads, headDim
    ).transposed(0, 2, 1, 3)

    if let positionEmbeddings {
      (queryStates, keyStates) = QwenAttention.applyRotary(
        q: queryStates,
        k: keyStates,
        positionEmbeddings: positionEmbeddings,
        sections: ropeSections
      )
    }

    if numKeyValueHeads != numAttentionHeads {
      keyStates = QwenAttention.repeatKeyValue(keyStates, repeats: numKeyValueGroups)
      valueStates = QwenAttention.repeatKeyValue(valueStates, repeats: numKeyValueGroups)
    }

    let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
    if let attentionMask {
      let prepared = attentionMask.dtype == queryStates.dtype ? attentionMask : attentionMask.asType(queryStates.dtype)
      maskMode = .array(prepared)
    } else {
      maskMode = .none
    }
    var attnOutput = MLXFast.scaledDotProductAttention(
      queries: queryStates,
      keys: keyStates,
      values: valueStates,
      scale: Float(scaling),
      mask: maskMode
    )
    attnOutput = attnOutput.transposed(0, 2, 1, 3)
      .reshaped(batchSize, seqLen, hiddenSize)
    attnOutput = attnOutput.asType(hiddenStates.dtype)
    let projected = oProj(attnOutput)
    return projected
  }

  private static func repeatKeyValue(_ tensor: MLXArray, repeats: Int) -> MLXArray {
    guard repeats > 1 else { return tensor }
    var expanded = MLX.expandedDimensions(tensor, axis: 2)
    expanded = MLX.repeated(expanded, count: repeats, axis: 2)
    let shape = tensor.shape
    return expanded.reshaped(shape[0], shape[1] * repeats, shape[2], shape[3])
  }

  private static func rotateHalf(_ tensor: MLXArray) -> MLXArray {
    let half = tensor.dim(tensor.ndim - 1) / 2
    let x1 = tensor[0..., 0..., 0..., ..<half]
    let x2 = tensor[0..., 0..., 0..., half...]
    return MLX.concatenated([-x2, x1], axis: -1)
  }

  private static func applyRotary(
    q: MLXArray,
    k: MLXArray,
    positionEmbeddings: (MLXArray, MLXArray),
    sections: [Int]
  ) -> (MLXArray, MLXArray) {
    let (cos, sin) = positionEmbeddings
    let doubledSections = sections.map { $0 * 2 }

    var cosChunks: [MLXArray] = []
    var sinChunks: [MLXArray] = []
    var start = 0

    for sectionSize in doubledSections {
      let end = start + sectionSize
      cosChunks.append(cos[0..., 0..., 0..., start..<end])
      sinChunks.append(sin[0..., 0..., 0..., start..<end])
      start = end
    }

    var cosSelected: [MLXArray] = []
    var sinSelected: [MLXArray] = []
    let modalOrder = [0, 1, 2]
    for (index, chunk) in cosChunks.enumerated() {
      let modality = modalOrder[index % modalOrder.count]
      cosSelected.append(chunk[modality, 0..., 0..., 0...])
      sinSelected.append(sinChunks[index][modality, 0..., 0..., 0...])
    }

    let cosCombined = MLX.concatenated(cosSelected, axis: -1)
    let sinCombined = MLX.concatenated(sinSelected, axis: -1)
    let cosExpanded = MLX.expandedDimensions(cosCombined, axis: 1)
    let sinExpanded = MLX.expandedDimensions(sinCombined, axis: 1)

    let qEmbed = (q * cosExpanded) + (rotateHalf(q) * sinExpanded)
    let kEmbed = (k * cosExpanded) + (rotateHalf(k) * sinExpanded)
    return (qEmbed, kEmbed)
  }

}

public final class QwenMLP: Module {
  let hiddenSize: Int
  let intermediateSize: Int

  @ModuleInfo(key: "gate_proj") var gateProj: Linear
  @ModuleInfo(key: "up_proj") var upProj: Linear
  @ModuleInfo(key: "down_proj") var downProj: Linear

  init(configuration: QwenTextEncoderConfiguration) {
    self.hiddenSize = configuration.hiddenSize
    self.intermediateSize = configuration.intermediateSize
    self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
    self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
    self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
  }

  public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
    let originalDType = hiddenStates.dtype
    let workingStates = hiddenStates

    let gateLinear = applyLinear(gateProj, to: workingStates)
    let gateOutput = MLXNN.silu(gateLinear)

    let upOutput = applyLinear(upProj, to: workingStates)

    let intermediateOutput = gateOutput * upOutput

    let output = applyLinear(downProj, to: intermediateOutput)
    return output.asType(originalDType)
  }

  private func applyLinear(_ layer: Linear, to input: MLXArray) -> MLXArray {
    return layer(input)
  }
}

public final class QwenRotaryEmbedding: Module {
  let invFreq: MLXArray
  let attentionScaling: Float
  let headDim: Int

  init(configuration: QwenTextEncoderConfiguration) {
    let computedHeadDim = configuration.hiddenSize / configuration.numAttentionHeads
    self.headDim = computedHeadDim
    let values = stride(from: 0, to: computedHeadDim, by: 2).map { Float($0) / Float(computedHeadDim) }
    let invValues = values.map { 1.0 / pow(configuration.ropeTheta, $0) }
    self.invFreq = MLXArray(invValues)
    self.attentionScaling = 1.0
  }

  public func callAsFunction(
    hiddenStates: MLXArray,
    positionIds: MLXArray
  ) -> (MLXArray, MLXArray) {
    var positionIds = positionIds
    if positionIds.ndim == 2 {
      positionIds = positionIds[.newAxis, .ellipsis]
      positionIds = MLX.repeated(positionIds, count: 3, axis: 0)
    }

    let dtype = hiddenStates.dtype
    var posFloat = positionIds.asType(.float32)
    posFloat = posFloat[0..., 0..., 0..., .newAxis]

    let invFreq = invFreq.asType(.float32)[.newAxis, .newAxis, .newAxis, 0...]
    let freqs = posFloat * invFreq
    let emb = MLX.concatenated([freqs, freqs], axis: -1)
    let cos = MLX.cos(emb) * attentionScaling
    let sin = MLX.sin(emb) * attentionScaling

    return (cos.asType(dtype), sin.asType(dtype))
  }
}

// MARK: - Joint text+vision helpers

extension QwenEncoder {
  // Expose token embedding lookup for constructing joint hidden states
  public func embed(inputIds: MLXArray) -> MLXArray {
    var tokenIds = inputIds
    if tokenIds.dtype != .int32 {
      tokenIds = tokenIds.asType(.int32)
    }
    return embedTokens(tokenIds)
  }

  // Start encoder from provided embeddings (skip token lookup)
  public func forward(
    embeddings: MLXArray,
    attentionMask: MLXArray?
  ) -> MLXArray {
    let batchSize = embeddings.dim(0)
    let seqLen = embeddings.dim(1)
    var hiddenStates = embeddings

    let cachePosition = MLXArray(0..<seqLen).asType(.int32)
    var positionIds = cachePosition[.newAxis, 0...]
    positionIds = MLX.repeated(positionIds, count: batchSize, axis: 0)
    positionIds = positionIds[.newAxis, .ellipsis]
    positionIds = MLX.repeated(positionIds, count: 3, axis: 0)

    let attentionMask4D = QwenEncoder.buildAttentionMask(
      mask: attentionMask,
      batchSize: batchSize,
      seqLen: seqLen
    )

    let positionEmbeddings = rotaryEmbedding(
      hiddenStates: hiddenStates,
      positionIds: positionIds
    )

    for layer in layers {
      hiddenStates = layer(
        hiddenStates: hiddenStates,
        attentionMask: attentionMask4D,
        positionEmbeddings: positionEmbeddings
      )
    }
    return norm(hiddenStates)
  }

  // Start encoder from provided embeddings and explicit position ids [3, batch, seq]
  public func forwardWithPositions(
    embeddings: MLXArray,
    attentionMask: MLXArray?,
    positionIds: MLXArray
  ) -> MLXArray {
    let batchSize = embeddings.dim(0)
    let seqLen = embeddings.dim(1)
    var hiddenStates = embeddings

    let attentionMask4D = QwenEncoder.buildAttentionMask(
      mask: attentionMask,
      batchSize: batchSize,
      seqLen: seqLen
    )

    let positionEmbeddings = rotaryEmbedding(
      hiddenStates: hiddenStates,
      positionIds: positionIds
    )

    for layer in layers {
      hiddenStates = layer(
        hiddenStates: hiddenStates,
        attentionMask: attentionMask4D,
        positionEmbeddings: positionEmbeddings
      )
    }
    return norm(hiddenStates)
  }
}

extension QwenTextEncoder {
  // Replace <|image_pad|> tokens with vision embeddings before running the transformer.
  public func encodeJoint(
    inputIds: MLXArray,
    attentionMask: MLXArray?,
    imageTokenId: Int,
    visionStartTokenId: Int,
    placeholderGridTHW: [(Int, Int, Int)],
    spatialMergeSize: Int,
    replacements: [MLXArray],
    dropIndex dropIndexOverride: Int? = nil
  ) -> (MLXArray, MLXArray) {
    var tokenIds = inputIds
    if tokenIds.dtype != .int32 {
      tokenIds = tokenIds.asType(.int32)
    }
    var hiddenStates = encoder.embed(inputIds: tokenIds)
    let dropIndexValue = dropIndexOverride ?? configuration.promptDropIndex

    if !replacements.isEmpty {
      hiddenStates = replaceVisionTokens(
        hiddenStates: hiddenStates,
        inputIds: tokenIds,
        imageTokenId: imageTokenId,
        replacements: replacements
      )
    }

    let attentionMaskUpdated: MLXArray
    if let attentionMask {
      attentionMaskUpdated = attentionMask.asType(.int32)
    } else {
      attentionMaskUpdated = MLX.ones([hiddenStates.dim(0), hiddenStates.dim(1)], dtype: .int32)
    }

    let positionIds = buildPositionIds(
      inputIds: tokenIds,
      attentionMask: attentionMaskUpdated,
      imageTokenId: imageTokenId,
      gridTHW: placeholderGridTHW,
      spatialMergeSize: spatialMergeSize
    )

    let encoded = encoder.forwardWithPositions(
      embeddings: hiddenStates,
      attentionMask: attentionMaskUpdated,
      positionIds: positionIds
    )
    let processed = QwenTextEncoder.processTextEmbeddings(
      hiddenStates: encoded,
      attentionMask: attentionMaskUpdated,
      dropIndex: dropIndexValue,
      dtype: configuration.outputDType
    )
    return processed
  }

  func encodeMultimodal(
    inputIds: MLXArray,
    attentionMask: MLXArray?,
    pixelValues: MLXArray,
    grids: [QwenVisionGrid],
    gridTHW: [(Int, Int, Int)],
    tokenCounts: [Int],
    imageTokenId: Int,
    visionStartTokenId: Int,
    spatialMergeSize: Int,
    dropIndex dropIndexOverride: Int? = nil
  ) throws -> (MLXArray, MLXArray) {
    guard let visionTower else {
      throw QwenTextEncoderError.visionTowerUnavailable
    }
    var towerOutput = try visionTower(patchInputs: pixelValues, grid: grids).hiddenStates
    towerOutput = towerOutput.asType(configuration.outputDType)
    let replacements = try QwenTextEncoder.splitVisionHiddenStates(
      hiddenStates: towerOutput,
      tokenCounts: tokenCounts
    )
    return encodeJoint(
      inputIds: inputIds,
      attentionMask: attentionMask,
      imageTokenId: imageTokenId,
      visionStartTokenId: visionStartTokenId,
      placeholderGridTHW: gridTHW,
      spatialMergeSize: spatialMergeSize,
      replacements: replacements,
      dropIndex: dropIndexOverride
    )
  }

  private func replaceVisionTokens(
    hiddenStates: MLXArray,
    inputIds: MLXArray,
    imageTokenId: Int,
    replacements: [MLXArray]
  ) -> MLXArray {
    let batch = hiddenStates.dim(0)
    let seqLen = hiddenStates.dim(1)
    let hiddenDim = hiddenStates.dim(2)

    guard let first = replacements.first else {
      return hiddenStates
    }
    var replacementTensor = replacements.count == 1 ? first : MLX.concatenated(replacements, axis: 0)
    if replacementTensor.dtype != .float32 {
      replacementTensor = replacementTensor.asType(.float32)
    }
    let replacementCount = replacementTensor.dim(0)

    let tokenMask = inputIds .== Int32(imageTokenId)
    let tokenMaskInt = tokenMask.asType(.int32)
    let counts = MLX.sum(tokenMaskInt, axis: 1).asType(.int32)
    MLX.eval(counts)
    let countValues = counts.asArray(Int32.self)
    for (row, count) in countValues.enumerated() {
      precondition(count == Int32(replacementCount), "[QwenTextEncoder] placeholder mismatch in row \(row)")
    }

    if replacementCount == 0 {
      return hiddenStates
    }

    let positions = tokenMaskInt.cumsum(axis: 1)
    let one = MLXArray(Int32(1))
    let zero = MLXArray(Int32(0))
    let replacementIndices = positions - one
    let safeIndices = MLX.maximum(replacementIndices, zero)
    let flatIndices = safeIndices.reshaped([-1])
    let gathered = MLX.take(replacementTensor, flatIndices, axis: 0)
    let gatheredReshaped = gathered.reshaped(batch, seqLen, hiddenDim).asType(hiddenStates.dtype)
    let maskExpanded = MLX.broadcast(tokenMask.reshaped(batch, seqLen, 1), to: [batch, seqLen, hiddenDim])
    return MLX.where(maskExpanded, gatheredReshaped, hiddenStates)
  }

  private func buildPositionIds(
    inputIds: MLXArray,
    attentionMask: MLXArray,
    imageTokenId: Int,
    gridTHW: [(Int, Int, Int)],
    spatialMergeSize: Int
  ) -> MLXArray {
    let batch = inputIds.dim(0)
    let seqLen = inputIds.dim(1)
    var ids = inputIds.asType(.int32)
    var mask = attentionMask.asType(.int32)
    MLX.eval(ids, mask)
    let idValues = ids.asArray(Int32.self)
    let maskValues = mask.asArray(Int32.self)

    var posT = [Int32](repeating: 0, count: batch * seqLen)
    var posH = [Int32](repeating: 0, count: batch * seqLen)
    var posW = [Int32](repeating: 0, count: batch * seqLen)

    for row in 0..<batch {
      var validIndices: [Int] = []
      validIndices.reserveCapacity(seqLen)
      var tokens: [Int32] = []
      tokens.reserveCapacity(seqLen)
      for idx in 0..<seqLen where maskValues[row * seqLen + idx] != 0 {
        validIndices.append(idx)
        tokens.append(idValues[row * seqLen + idx])
      }
      if tokens.isEmpty {
        continue
      }
      var rowPosT = [Int32](repeating: 0, count: tokens.count)
      var rowPosH = [Int32](repeating: 0, count: tokens.count)
      var rowPosW = [Int32](repeating: 0, count: tokens.count)
      var cursor = 0
      var placeholderIndex = 0
      var currentBase: Int32 = 0

      while cursor < tokens.count {
        var nextImage = cursor
        while nextImage < tokens.count && tokens[nextImage] != Int32(imageTokenId) {
          rowPosT[nextImage] = currentBase
          rowPosH[nextImage] = currentBase
          rowPosW[nextImage] = currentBase
          currentBase += 1
          nextImage += 1
        }
        if nextImage >= tokens.count {
          break
        }

        let grid = gridTHW.isEmpty ? (1, 1, 1) : gridTHW[min(placeholderIndex, gridTHW.count - 1)]
        let gridT = max(1, grid.0)
        let gridH = max(1, grid.1 / spatialMergeSize)
        let gridW = max(1, grid.2 / spatialMergeSize)
        let patchCount = gridT * gridH * gridW
        var local = 0
        while local < patchCount && nextImage + local < tokens.count {
          let tIndex = local / (gridH * gridW)
          let rem = local % (gridH * gridW)
          let hIndex = rem / gridW
          let wIndex = rem % gridW
          let dest = nextImage + local
          rowPosT[dest] = currentBase + Int32(tIndex)
          rowPosH[dest] = currentBase + Int32(hIndex)
          rowPosW[dest] = currentBase + Int32(wIndex)
          local += 1
        }
        currentBase += Int32(local)
        cursor = nextImage + local
        placeholderIndex += 1
      }

      for (localIdx, seqIdx) in validIndices.enumerated() {
        posT[row * seqLen + seqIdx] = rowPosT[localIdx]
        posH[row * seqLen + seqIdx] = rowPosH[localIdx]
        posW[row * seqLen + seqIdx] = rowPosW[localIdx]
      }
    }

    let posTextArr = MLXArray(posT.map(Float32.init), [batch, seqLen]).asType(.int32)
    let posHArr = MLXArray(posH.map(Float32.init), [batch, seqLen]).asType(.int32)
    let posWArr = MLXArray(posW.map(Float32.init), [batch, seqLen]).asType(.int32)
    return MLX.stacked([posTextArr, posHArr, posWArr], axis: 0)
  }

  private static func splitVisionHiddenStates(
    hiddenStates: MLXArray,
    tokenCounts: [Int]
  ) throws -> [MLXArray] {
    var embeddings: [MLXArray] = []
    embeddings.reserveCapacity(tokenCounts.count)
    var offset = 0
    for (index, count) in tokenCounts.enumerated() {
      let end = offset + count
      guard end <= hiddenStates.dim(0) else {
        throw QwenTextEncoderError.mismatchedVisionTokenCount
      }
      let slice = hiddenStates[offset..<end, 0...]
      embeddings.append(slice)
      offset = end
    }
    guard offset == hiddenStates.dim(0) else {
      throw QwenTextEncoderError.mismatchedVisionTokenCount
    }
    return embeddings
  }
}
