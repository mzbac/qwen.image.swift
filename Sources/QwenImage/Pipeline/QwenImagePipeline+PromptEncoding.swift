import Foundation
import Logging
import MLX

extension QwenImagePipeline {
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

  // MARK: - Internal Prompt Encoding

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
}
