import Foundation
import MLX

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
