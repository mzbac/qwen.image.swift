import Foundation
import MLX
import MLXNN

enum VisionTowerError: Error {
  case windowIndexOutOfBounds
}

struct QwenVisionTowerOutput {
  let hiddenStates: MLXArray
  let cumulativeWindowSequenceLengths: MLXArray
  let cumulativeSequenceLengths: MLXArray
  let windowIndex: MLXArray
  let rotaryCos: MLXArray
  let rotarySin: MLXArray
  let patchInputs: MLXArray
}

final class QwenVisionTower: Module {
  let configuration: QwenVisionConfiguration

  @ModuleInfo(key: "patch_embed") private var patchEmbed: QwenVisionPatchEmbed
  @ModuleInfo(key: "patch_merger") private var patchMerger: QwenVisionPatchMerger
  @ModuleInfo(key: "blocks") private var blocks: [QwenVisionBlock]

  private var mergeUnit: Int { configuration.spatialMergeSize * configuration.spatialMergeSize }
  private let headDim: Int
  private let rotaryInnerDim: Int
  private let rotaryHalfDim: Int
  private let ropeTheta: Float = 10_000.0
  private let invFrequencies: [Float32]

  init(configuration: QwenVisionConfiguration) {
    self.configuration = configuration
    self.headDim = configuration.embedDim / configuration.numHeads
    self.rotaryInnerDim = max(1, headDim / 2)
    self.rotaryHalfDim = max(1, rotaryInnerDim / 2)
    self.invFrequencies = QwenVisionTower.computeInverseFrequencies(
      dim: rotaryInnerDim,
      theta: ropeTheta
    )
    self._patchEmbed.wrappedValue = QwenVisionPatchEmbed(
      patchSize: configuration.patchSize,
      temporalPatchSize: configuration.temporalPatchSize,
      inChannels: configuration.inChannels,
      embedDim: configuration.embedDim
    )
    self._patchMerger.wrappedValue = QwenVisionPatchMerger(
      contextDim: configuration.embedDim,
      outputDim: configuration.outHiddenDim,
      spatialMergeSize: configuration.spatialMergeSize
    )
    let blocks = (0..<configuration.depth).map { index in
      QwenVisionBlock(configuration: configuration, blockIndex: index)
    }
    self._blocks.wrappedValue = blocks
  }

  func callAsFunction(patchInputs: MLXArray, grid: [QwenVisionGrid]) throws -> QwenVisionTowerOutput {
    precondition(patchInputs.ndim == 3, "Expected patch input shape [batch, tokens, patchVolume]")
    precondition(grid.count == patchInputs.dim(0), "Grid metadata must match batch size")

    QwenLogger.vision.debug("call: patchInputs shape=\(patchInputs.shape)")
    let batch = patchInputs.dim(0)
    let tokensPerSample = patchInputs.dim(1)
    let patchVolume = patchInputs.dim(2)

    var flattened = patchInputs.reshaped(batch * tokensPerSample, patchVolume)
    QwenLogger.vision.debug("flattened shape=\(flattened.shape)")
    flattened = patchEmbed(flattened)
    QwenLogger.vision.debug("after patchEmbed shape=\(flattened.shape)")
    let computeType = patchInputs.dtype

    var sampleOffset = 0
    var outputs: [MLXArray] = []
    var cuWindowLengths: [Int] = [0]
    var cuFullLengths: [Int] = [0]
    var windowIndicesAll: [Int32] = []
    var rotaryCosSlices: [MLXArray] = []
    var rotarySinSlices: [MLXArray] = []

    let maxGridSize = max(1, grid.map { max($0.height, $0.width) }.max() ?? 1)
    let rotaryTable = makeRotaryTable(maxSize: maxGridSize, dtype: computeType)

    var llmCellBase = 0
    for (index, entry) in grid.enumerated() {
      let tokenCount = entry.temporal * entry.height * entry.width
      let start = sampleOffset
      let end = sampleOffset + tokenCount
      let sampleTokens = flattened[start..<end, 0...]
      QwenLogger.vision.debug("sample \(index): sampleTokens shape=\(sampleTokens.shape)")
      sampleOffset = end

      let llmHeight = entry.height / configuration.spatialMergeSize
      let llmWidth = entry.width / configuration.spatialMergeSize
      precondition(llmHeight > 0 && llmWidth > 0, "Invalid spatial merge dimensions")

      let positions = makeOrderedPatchPositions(
        temporal: entry.temporal,
        height: entry.height,
        width: entry.width
      )
      var rotary = makeRotaryEmbeddings(
        rotaryTable: rotaryTable,
        orderedH: positions.h,
        orderedW: positions.w
      )

      let windowData = computeWindowIndexAndSeqLens(
        temporal: entry.temporal,
        height: entry.height,
        width: entry.width
      )
      let windowIndex = windowData.index
      let cuWindowSeqLens = windowData.cuWindowSeqLens
      let windowIndexValues = windowData.indexValues

      // Reorder hidden states and rotary embeddings by window index (group of 4 tokens per LLM cell)
      var hidden = sampleTokens
      hidden = hidden.reshaped(tokenCount / mergeUnit, mergeUnit, configuration.embedDim)
      hidden = MLX.take(hidden, windowIndex, axis: 0)
      hidden = hidden.reshaped(1, tokenCount, configuration.embedDim)
      let rotaryDim = rotary.cos.dim(1)
      rotary.cos = rotary.cos.reshaped(tokenCount / mergeUnit, mergeUnit, rotaryDim)
      rotary.cos = MLX.take(rotary.cos, windowIndex, axis: 0).reshaped(tokenCount, rotaryDim)
      rotary.sin = rotary.sin.reshaped(tokenCount / mergeUnit, mergeUnit, rotaryDim)
      rotary.sin = MLX.take(rotary.sin, windowIndex, axis: 0).reshaped(tokenCount, rotaryDim)

      // Build cuSeqlens for full attention (per temporal frame, unmerged grid)
      let fullPerFrame = Int32(entry.height * entry.width)
      let cuFullSeqLens = MLX.multiply(
        MLXArray(0..<(entry.temporal + 1)).asType(.int32),
        MLXArray(fullPerFrame, dtype: .int32)
      )

      let cosEmbed = rotary.cos.asType(hidden.dtype)
      let sinEmbed = rotary.sin.asType(hidden.dtype)
      let fullMask = buildAttentionMask(
        sequenceLength: tokenCount,
        cuSeqlens: cuFullSeqLens,
        dtype: computeType
      )
      let windowMask = buildAttentionMask(
        sequenceLength: tokenCount,
        cuSeqlens: cuWindowSeqLens,
        dtype: computeType
      )
      for (bidx, block) in blocks.enumerated() {
        let isFull = configuration.fullAttentionBlockIndices.contains(bidx)
        let attnMask = isFull ? fullMask : windowMask
        hidden = block(
          hidden,
          rotaryEmbedding: (cos: cosEmbed, sin: sinEmbed),
          attentionMask: attnMask,
          cuSeqlens: nil
        )
      }
      hidden = hidden.reshaped(tokenCount, configuration.embedDim)
      // Merge 2x2 tokens per LLM cell
      let merged = patchMerger(hidden)

      // Reverse window index to original LLM order
      let reverseIdx = argsortInt32(windowIndexValues)
      let restored = MLX.take(merged, reverseIdx, axis: 0)

      outputs.append(restored)
      rotaryCosSlices.append(rotary.cos)
      rotarySinSlices.append(rotary.sin)

      // Update cumulative lengths
      let mergedPerFrame = (entry.height / configuration.spatialMergeSize) * (entry.width / configuration.spatialMergeSize)
      var cumulativeWindow = cuWindowLengths.last ?? 0
      for _ in 0..<entry.temporal { cumulativeWindow += mergedPerFrame; cuWindowLengths.append(cumulativeWindow) }

      var cumulativeFull = cuFullLengths.last ?? 0
      let perFrame = entry.height * entry.width
      for _ in 0..<entry.temporal { cumulativeFull += perFrame; cuFullLengths.append(cumulativeFull) }

      // Aggregate window index (shifted by base across samples) for output/debug
      let base = Int32(llmCellBase)
      windowIndicesAll.append(contentsOf: windowIndexValues.map { $0 + base })

      llmCellBase += mergedPerFrame * entry.temporal
    }

    let hiddenStates = outputs.count == 1 ? outputs[0] : MLX.concatenated(outputs, axis: 0)
    let windowSeqLens = makeIndexArray(cuWindowLengths)
    let fullSeqLens = makeIndexArray(cuFullLengths)
    let windowIndicesArray = makeIndexArray(windowIndicesAll)
    let rotaryCos = rotaryCosSlices.count == 1 ? rotaryCosSlices[0] : MLX.concatenated(rotaryCosSlices, axis: 0)
    let rotarySin = rotarySinSlices.count == 1 ? rotarySinSlices[0] : MLX.concatenated(rotarySinSlices, axis: 0)

    return QwenVisionTowerOutput(
      hiddenStates: hiddenStates,
      cumulativeWindowSequenceLengths: windowSeqLens,
      cumulativeSequenceLengths: fullSeqLens,
      windowIndex: windowIndicesArray,
      rotaryCos: rotaryCos,
      rotarySin: rotarySin,
      patchInputs: patchInputs
    )
  }

  private func buildAttentionMask(
    sequenceLength: Int,
    cuSeqlens: MLXArray,
    dtype: DType
  ) -> MLXArray {
    let lensArr = cuSeqlens.asType(.int32)
    let count = lensArr.dim(0)
    guard count >= 2 else {
      return MLX.zeros([1, sequenceLength, sequenceLength], dtype: .bool)
    }
    let chunkCount = count - 1
    guard chunkCount > 0 else {
      return MLX.zeros([1, sequenceLength, sequenceLength], dtype: .bool)
    }
    let startValues = lensArr[0..<chunkCount]
    let endValues = lensArr[1..<count]
    let starts = startValues.reshaped(1, chunkCount)
    let ends = endValues.reshaped(1, chunkCount)
    let tokenPositions = MLXArray(0..<sequenceLength).asType(.int32).reshaped(sequenceLength, 1)
    let geStart = MLX.greaterEqual(tokenPositions, starts)
    let ltEnd = MLX.less(tokenPositions, ends)
    let membership = MLX.logicalAnd(geStart, ltEnd)
    let membershipTyped = membership.asType(dtype)
    let attn = MLX.matmul(membershipTyped, membershipTyped.transposed(1, 0))
    let hasOverlap = MLX.greater(attn, MLXArray(Float32(0), dtype: dtype))
    return hasOverlap.reshaped(1, sequenceLength, sequenceLength).asType(.bool)
  }

  // Build window index and cumulative lengths for attention chunking matching diffusers
  private func computeWindowIndexAndSeqLens(
    temporal: Int,
    height: Int,
    width: Int
  ) -> (index: MLXArray, cuWindowSeqLens: MLXArray, indexValues: [Int32]) {
    let llmH = height / configuration.spatialMergeSize
    let llmW = width / configuration.spatialMergeSize
    let vitWindow = configuration.windowSize / configuration.spatialMergeSize / configuration.patchSize
    let padH = (vitWindow - (llmH % vitWindow)) % vitWindow
    let padW = (vitWindow - (llmW % vitWindow)) % vitWindow
    let numWindowsH = (llmH + padH) / vitWindow
    let numWindowsW = (llmW + padW) / vitWindow

    // Build padded grid for a single frame.
    var padded: [Int32] = Array(repeating: -100, count: (llmH + padH) * (llmW + padW))
    for h in 0..<llmH {
      for w in 0..<llmW {
        padded[h * (llmW + padW) + w] = Int32(h * llmW + w)
      }
    }

    // Precompute window indices and counts for one frame.
    var frameWindowIndices: [Int32] = []
    var frameCuLens: [Int32] = [0]
    for wh in 0..<numWindowsH {
      for ww in 0..<numWindowsW {
        var count: Int32 = 0
        for ih in 0..<vitWindow {
          let h = wh * vitWindow + ih
          let rowBase = h * (llmW + padW)
          for iw in 0..<vitWindow {
            let w = ww * vitWindow + iw
            let val = padded[rowBase + w]
            if val != -100 {
              frameWindowIndices.append(val)
              count += 1
            }
          }
        }
        let last = frameCuLens.last ?? 0
        frameCuLens.append(last + count * Int32(mergeUnit))
      }
    }

    // Replicate per-frame windows across temporal dimension with base offsets.
    var windowIndexAll: [Int32] = []
    windowIndexAll.reserveCapacity(frameWindowIndices.count * temporal)
    var cuLens: [Int32] = [0]
    cuLens.reserveCapacity(frameCuLens.count * temporal)
    let perFrameBase = Int32(llmH * llmW)
    for t in 0..<temporal {
      let base = Int32(t) * perFrameBase
      for idx in frameWindowIndices {
        windowIndexAll.append(idx + base)
      }
      let offset = cuLens.last ?? 0
      for len in frameCuLens.dropFirst() {
        cuLens.append(offset + len)
      }
    }

    let indexArr = MLXArray(windowIndexAll.map(Float32.init), [windowIndexAll.count]).asType(.int32)
    let cuArr = MLXArray(cuLens.map(Float32.init), [cuLens.count]).asType(.int32)
    return (indexArr, cuArr, windowIndexAll)
  }

  // Argsort for int32 array (CPU path) → MLXArray[int32]
  private func argsortInt32(_ values: [Int32]) -> MLXArray {
    let indexed = values.enumerated().map { ($0.offset, $0.element) }
    let sorted = indexed.sorted { $0.1 < $1.1 }
    let result: [Int32] = sorted.map { Int32($0.0) }
    return MLXArray(result.map(Float32.init), [result.count]).asType(.int32)
  }

  private func makeOrderedPatchPositions(
    temporal: Int,
    height: Int,
    width: Int
  ) -> (h: MLXArray, w: MLXArray) {
    let merge = configuration.spatialMergeSize
    let llmHeight = height / merge
    let llmWidth = width / merge
    let llmCount = temporal * llmHeight * llmWidth

    var hValues: [Float32] = []
    var wValues: [Float32] = []
    hValues.reserveCapacity(llmCount * mergeUnit)
    wValues.reserveCapacity(llmCount * mergeUnit)

    for _ in 0..<temporal {
      for hBlock in 0..<llmHeight {
        for wBlock in 0..<llmWidth {
          for innerH in 0..<merge {
            for innerW in 0..<merge {
              hValues.append(Float32(hBlock * merge + innerH))
              wValues.append(Float32(wBlock * merge + innerW))
            }
          }
        }
      }
    }

    let llmShape = [llmCount, mergeUnit]
    let hArray = MLXArray(hValues, llmShape).asType(.int32)
    let wArray = MLXArray(wValues, llmShape).asType(.int32)

    return (
      hArray.reshaped(hArray.dim(0) * hArray.dim(1)),
      wArray.reshaped(wArray.dim(0) * wArray.dim(1))
    )
  }

  private func makeRotaryEmbeddings(
    rotaryTable: MLXArray,
    orderedH: MLXArray,
    orderedW: MLXArray
  ) -> (cos: MLXArray, sin: MLXArray) {
    let hAngles = MLX.take(rotaryTable, orderedH, axis: 0)
    let wAngles = MLX.take(rotaryTable, orderedW, axis: 0)
    let baseAngles = MLX.concatenated([hAngles, wAngles], axis: 1)
    let duplicated = MLX.concatenated([baseAngles, baseAngles], axis: 1)
    return (MLX.cos(duplicated), MLX.sin(duplicated))
  }

  private func makeIndexArray(_ values: [Int]) -> MLXArray {
    let floats = values.map(Float32.init)
    return MLXArray(floats, [values.count]).asType(.int32)
  }

  private func makeIndexArray(_ values: [Int32]) -> MLXArray {
    let floats = values.map(Float32.init)
    return MLXArray(floats, [values.count]).asType(.int32)
  }

  private func makeRotaryTable(maxSize: Int, dtype: DType) -> MLXArray {
    let count = max(1, rotaryHalfDim)
    var table: [Float32] = []
    table.reserveCapacity(maxSize * count)

    for position in 0..<maxSize {
      let pos = Float32(position)
      for freq in invFrequencies {
        table.append(pos * freq)
      }
    }

    return MLXArray(table, [maxSize, count]).asType(dtype)
  }

  private static func computeInverseFrequencies(dim: Int, theta: Float) -> [Float32] {
    var values: [Float32] = []
    let stepCount = max(1, dim / 2)
    let dimFloat = Float32(dim)
    for i in 0..<stepCount {
      let exponent = Float32(2 * i) / dimFloat
      let freq = powf(theta, -exponent)
      values.append(freq)
    }
    return values
  }

  var blockCount: Int {
    blocks.count
  }

  func updatePatchEmbed(parameters: ModuleParameters) {
    patchEmbed.update(parameters: parameters)
  }

  func updatePatchMerger(parameters: ModuleParameters) {
    patchMerger.update(parameters: parameters)
  }

  fileprivate func patchEmbedModule() -> QwenVisionPatchEmbed {
    patchEmbed
  }

  fileprivate func patchMergerModule() -> QwenVisionPatchMerger {
    patchMerger
  }

  func updateBlock(at index: Int, parameters: ModuleParameters) {
    precondition(index >= 0 && index < blocks.count, "Block index out of range")
    blocks[index].update(parameters: parameters)
  }
}
