import Foundation
import MLX

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
