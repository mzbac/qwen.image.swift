import Foundation
import MLX
import MLXNN

public struct QwenTransformerConfiguration {
  public var inChannels: Int
  public var outChannels: Int
  public var numberOfLayers: Int
  public var attentionHeadDim: Int
  public var numberOfHeads: Int
  public var jointAttentionDim: Int
  public var patchSize: Int

  public init(
    inChannels: Int = 64,
    outChannels: Int = 16,
    numberOfLayers: Int = 60,
    attentionHeadDim: Int = 128,
    numberOfHeads: Int = 24,
    jointAttentionDim: Int = 3_584,
    patchSize: Int = 2
  ) {
    self.inChannels = inChannels
    self.outChannels = outChannels
    self.numberOfLayers = numberOfLayers
    self.attentionHeadDim = attentionHeadDim
    self.numberOfHeads = numberOfHeads
    self.jointAttentionDim = jointAttentionDim
    self.patchSize = patchSize
  }
}

public final class QwenTransformer: Module {

  public let configuration: QwenTransformerConfiguration
  let innerDimension: Int

  @ModuleInfo(key: "img_in") var imageProjection: Linear
  @ModuleInfo(key: "txt_norm") var textNorm: RMSNorm
  @ModuleInfo(key: "txt_in") var textProjection: Linear
  @ModuleInfo(key: "time_text_embed") var timeTextEmbed: QwenTimeTextEmbed
  @ModuleInfo(key: "pos_embed") var positionEmbedding: QwenEmbedRope
  @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [QwenTransformerBlock]
  @ModuleInfo(key: "norm_out") var normOut: QwenAdaLayerNormContinuous
  @ModuleInfo(key: "proj_out") var projectionOut: Linear

  public init(configuration: QwenTransformerConfiguration) {
    self.configuration = configuration
    self.innerDimension = configuration.numberOfHeads * configuration.attentionHeadDim

    self._imageProjection.wrappedValue = Linear(configuration.inChannels, innerDimension)
    self._textNorm.wrappedValue = RMSNorm(dimensions: configuration.jointAttentionDim, eps: 1e-6)
    self._textProjection.wrappedValue = Linear(configuration.jointAttentionDim, innerDimension)
    self._timeTextEmbed.wrappedValue = QwenTimeTextEmbed(
      timestepProjectionDim: 256,
      innerDim: innerDimension
    )
    self._positionEmbedding.wrappedValue = QwenEmbedRope(
      theta: 10_000,
      axesDimensions: [16, 56, 56],
      scaleRope: true
    )
    var blocks: [QwenTransformerBlock] = []
    for _ in 0..<configuration.numberOfLayers {
      blocks.append(
        QwenTransformerBlock(
          dimension: innerDimension,
          numberOfHeads: configuration.numberOfHeads,
          headDimension: configuration.attentionHeadDim
        )
      )
    }
    self._transformerBlocks.wrappedValue = blocks
    self._normOut.wrappedValue = QwenAdaLayerNormContinuous(
      embeddingDimension: innerDimension,
      conditioningDimension: innerDimension
    )
    self._projectionOut.wrappedValue = Linear(
      innerDimension,
      configuration.patchSize * configuration.patchSize * configuration.outChannels
    )
  }

  public func callAsFunction(
    timestepIndex: Int,
    runtimeConfig: QwenRuntimeConfig,
    hiddenStates: MLXArray,
    encoderHiddenStates: MLXArray,
    encoderHiddenStatesMask: MLXArray,
    imageSegments: [(Int, Int, Int)]? = nil
  ) -> MLXArray {
    forward(
      timestepIndex: timestepIndex,
      runtimeConfig: runtimeConfig,
      hiddenStates: hiddenStates,
      encoderHiddenStates: encoderHiddenStates,
      encoderHiddenStatesMask: encoderHiddenStatesMask,
      imageSegments: imageSegments
    )
  }

  public func forward(
    timestepIndex: Int,
    runtimeConfig: QwenRuntimeConfig,
    hiddenStates: MLXArray,
    encoderHiddenStates: MLXArray,
    encoderHiddenStatesMask: MLXArray,
    imageSegments: [(Int, Int, Int)]? = nil,
    precomputedImageRotaryEmbeddings: (MLXArray, MLXArray)? = nil
  ) -> MLXArray {
    var imageStates = imageProjection(hiddenStates)
    var textStates = textProjection(textNorm(encoderHiddenStates))

    let textEmbeddings = Self.computeTextEmbeddings(
      timestepIndex: timestepIndex,
      hiddenStates: imageStates,
      timeTextEmbed: timeTextEmbed,
      runtimeConfig: runtimeConfig
    )
    let positionalEmbeddings: (MLXArray, MLXArray)
    if let pre = precomputedImageRotaryEmbeddings {
      positionalEmbeddings = pre
    } else {
      let computedRotary = Self.computeRotaryEmbeddings(
        encoderHiddenStates: textStates,
        encoderHiddenStatesMask: encoderHiddenStatesMask,
        positionEmbedding: positionEmbedding,
        runtimeConfig: runtimeConfig,
        imageSegments: imageSegments
      )
      positionalEmbeddings = (computedRotary.0, computedRotary.1)
    }

    let textSequenceLength = textStates.dim(1)
    let imageSequenceLength = imageStates.dim(1)
    let additiveMask = AttentionUtils.convertKeyPaddingMaskToAdditiveMask(
      mask: encoderHiddenStatesMask,
      jointSequenceLength: textSequenceLength + imageSequenceLength,
      textSequenceLength: textSequenceLength,
      targetDType: textStates.dtype
    )

    for block in transformerBlocks {
      (textStates, imageStates) = block(
        hiddenStates: imageStates,
        encoderHiddenStates: textStates,
        encoderHiddenStatesMask: encoderHiddenStatesMask,
        textEmbeddings: textEmbeddings,
        imageRotaryEmbeddings: positionalEmbeddings,
        additiveMask: additiveMask
      )
    }

    var output = normOut(hiddenStates: imageStates, conditioning: textEmbeddings)
    output = projectionOut(output)
    return output
  }

  private static func computeTextEmbeddings(
    timestepIndex: Int,
    hiddenStates: MLXArray,
    timeTextEmbed: QwenTimeTextEmbed,
    runtimeConfig: QwenRuntimeConfig
  ) -> MLXArray {
    let batch = hiddenStates.dim(0)
    let sigma = runtimeConfig.scheduler.sigmas[timestepIndex].asType(.float32)
    let timesteps = MLX.broadcast(sigma, to: [batch])
    return timeTextEmbed(timestep: timesteps, hiddenStates: hiddenStates)
  }

  private static func computeRotaryEmbeddings(
    encoderHiddenStates: MLXArray,
    encoderHiddenStatesMask: MLXArray,
    positionEmbedding: QwenEmbedRope,
    runtimeConfig: QwenRuntimeConfig,
    imageSegments: [(Int, Int, Int)]?
  ) -> (MLXArray, MLXArray) {
    let latentHeight = runtimeConfig.height / 16
    let latentWidth = runtimeConfig.width / 16
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
    let textLen = encoderHiddenStates.dim(1)
    let batch = encoderHiddenStates.dim(0)
    let lengths = Array(repeating: textLen, count: batch)

    return positionEmbedding(
      videoSegments: segments,
      textSequenceLengths: lengths
    )
  }

  public func setAttentionQuantization(_ spec: QwenQuantizationSpec?) {
    for index in transformerBlocks.indices {
      transformerBlocks[index].setAttentionQuantization(spec)
    }
  }

}
