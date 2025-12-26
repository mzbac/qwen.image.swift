import Foundation
import MLX

#if canImport(CoreGraphics)
import CoreGraphics
#endif

enum QwenVisionPreprocessorError: Error {
  case emptyInput
  case invalidChannelCount(expected: Int, got: Int)
  case invalidSpatialDimensions
#if canImport(CoreGraphics)
  case imageResizeFailed
#else
  case unsupportedPlatform
#endif
}

struct QwenVisionGrid {
  let temporal: Int
  let height: Int
  let width: Int

  func asArray(dtype: DType = .int32) -> MLXArray {
    MLXArray([temporal, height, width]).asType(dtype)
  }
}

struct QwenVisionPreprocessorConfig {
  var patchSize: Int = 14
  var temporalPatchSize: Int = 2
  var mergeSize: Int = 2
  var minPixels: Int = 56 * 56
  var maxPixels: Int = 28 * 28 * 1280
  var rescaleFactor: Float = 1.0
  var imageMean: [Float] = [0.48145466, 0.4578275, 0.40821073]
  var imageStd: [Float] = [0.26862954, 0.26130258, 0.27577711]
  var dtype: DType = .float32

  init() {}

  init(
    patchSize: Int,
    temporalPatchSize: Int,
    mergeSize: Int,
    minPixels: Int,
    maxPixels: Int,
    rescaleFactor: Float,
    imageMean: [Float],
    imageStd: [Float],
    dtype: DType
  ) {
    self.patchSize = patchSize
    self.temporalPatchSize = temporalPatchSize
    self.mergeSize = mergeSize
    self.minPixels = minPixels
    self.maxPixels = maxPixels
    self.rescaleFactor = rescaleFactor
    self.imageMean = imageMean
    self.imageStd = imageStd
    self.dtype = dtype
  }
}

struct QwenVisionPreprocessorOutput {
  let patches: MLXArray
  let grid: QwenVisionGrid
  let resizedSize: (height: Int, width: Int)
}

struct QwenVisionPreprocessor {
  let config: QwenVisionPreprocessorConfig

  init(config: QwenVisionPreprocessorConfig = .init()) {
    self.config = config
  }

#if canImport(CoreGraphics)
  func preprocess(
    cgImages: [CGImage],
    targetSize: (height: Int, width: Int)? = nil,
    intermediateSize: (height: Int, width: Int)? = nil
  ) throws -> QwenVisionPreprocessorOutput {
    guard !cgImages.isEmpty else { throw QwenVisionPreprocessorError.emptyInput }
    let height = cgImages[0].height
    let width = cgImages[0].width
    let factor = config.patchSize * config.mergeSize

    let (targetHeight, targetWidth): (Int, Int)
    if let targetSize {
      targetHeight = targetSize.height
      targetWidth = targetSize.width
    } else {
      (targetHeight, targetWidth) = try QwenVisionUtils.smartResize(
        height: height,
        width: width,
        factor: factor,
        minPixels: config.minPixels,
        maxPixels: config.maxPixels
      )
    }

    let arrays: [MLXArray] = try cgImages.map { image in
      try VisionImageProcessor.resizeAndNormalize(
        image: image,
        width: targetWidth,
        height: targetHeight,
        mean: config.imageMean,
        std: config.imageStd,
        rescaleFactor: config.rescaleFactor,
        addBatchDimension: false,
        dtype: .float32,
        intermediateWidth: intermediateSize?.width,
        intermediateHeight: intermediateSize?.height
      )
    }

    return try preprocess(pixelArrays: arrays, resizedHeight: targetHeight, resizedWidth: targetWidth)
  }

  func preprocess(
    cgImage: CGImage,
    targetSize: (height: Int, width: Int)? = nil,
    intermediateSize: (height: Int, width: Int)? = nil
  ) throws -> QwenVisionPreprocessorOutput {
    try preprocess(cgImages: [cgImage], targetSize: targetSize, intermediateSize: intermediateSize)
  }

#endif

  func preprocess(pixelArrays: [MLXArray]) throws -> QwenVisionPreprocessorOutput {
    guard let first = pixelArrays.first else { throw QwenVisionPreprocessorError.emptyInput }
    let height = first.dim(-2)
    let width = first.dim(-1)
    let meanValues = (0..<3).map { Float32(config.imageMean[min($0, config.imageMean.count - 1)]) }
    let stdValues = (0..<3).map { Float32(config.imageStd[min($0, config.imageStd.count - 1)]) }
    let mean = MLXArray(meanValues, [3, 1, 1]).asType(.float32)
    let std = MLXArray(stdValues, [3, 1, 1]).asType(.float32)
    let rescale = MLXArray(Float32(config.rescaleFactor))
    let normalized = try pixelArrays.map { frame -> MLXArray in
      let floatFrame = frame.asType(.float32)
      let channelCount = floatFrame.dim(0)
      guard channelCount == 3 else {
        throw QwenVisionPreprocessorError.invalidChannelCount(expected: 3, got: channelCount)
      }
      return (floatFrame * rescale - mean) / std
    }
    return try preprocess(pixelArrays: normalized, resizedHeight: height, resizedWidth: width)
  }

  private func preprocess(
    pixelArrays: [MLXArray],
    resizedHeight: Int,
    resizedWidth: Int
  ) throws -> QwenVisionPreprocessorOutput {
    guard let first = pixelArrays.first else { throw QwenVisionPreprocessorError.emptyInput }
    guard first.ndim == 3 else {
      throw QwenVisionPreprocessorError.invalidChannelCount(expected: 3, got: first.ndim)
    }
    guard first.dim(0) == 3 else {
      throw QwenVisionPreprocessorError.invalidChannelCount(expected: 3, got: first.dim(0))
    }

    let factor = config.patchSize * config.mergeSize
    guard resizedHeight % factor == 0 && resizedWidth % factor == 0 else {
      throw QwenVisionPreprocessorError.invalidSpatialDimensions
    }

    var frames: [MLXArray] = pixelArrays.map { $0.asType(.float32) }

    if frames.isEmpty {
      throw QwenVisionPreprocessorError.emptyInput
    }

    let remainder = frames.count % config.temporalPatchSize
    if remainder != 0, let last = frames.last {
      for _ in 0..<(config.temporalPatchSize - remainder) {
        frames.append(last)
      }
    }

    var patches = MLX.stacked(frames, axis: 0)
    let channel = patches.dim(1)
    let gridT = patches.dim(0) / config.temporalPatchSize
    let gridH = resizedHeight / config.patchSize
    let gridW = resizedWidth / config.patchSize

    guard gridH % config.mergeSize == 0 && gridW % config.mergeSize == 0 else {
      throw QwenVisionPreprocessorError.invalidSpatialDimensions
    }

    let mergedH = gridH / config.mergeSize
    let mergedW = gridW / config.mergeSize

    patches = patches.reshaped(
      gridT,
      config.temporalPatchSize,
      channel,
      mergedH,
      config.mergeSize,
      config.patchSize,
      mergedW,
      config.mergeSize,
      config.patchSize
    )

    patches = patches.transposed(0, 3, 6, 4, 7, 2, 1, 5, 8)

    let flattened = patches.reshaped(
      gridT * gridH * gridW,
      channel * config.temporalPatchSize * config.patchSize * config.patchSize
    ).asType(config.dtype)

    let grid = QwenVisionGrid(temporal: gridT, height: gridH, width: gridW)
    return QwenVisionPreprocessorOutput(
      patches: flattened,
      grid: grid,
      resizedSize: (height: resizedHeight, width: resizedWidth)
    )
  }

  // Optional: load precomputed patch inputs (e.g., from diffusers) for strict parity.
  // Expects a JSON alongside the .bin with fields: shape [tokens, patchVolume], image_grid_thw [[T,H,W]].
  static func loadExternalPatchInputs(basePath: String, dtype: DType = .float32) throws -> QwenVisionPreprocessorOutput {
    let jsonURL = URL(fileURLWithPath: basePath).appendingPathExtension("json")
    let binURL = URL(fileURLWithPath: basePath).appendingPathExtension("bin")
    let data = try Data(contentsOf: jsonURL)
    guard let meta = try JSONSerialization.jsonObject(with: data) as? [String: Any],
          let shape = meta["shape"] as? [Int],
          let grid = meta["image_grid_thw"] as? [[Int]], let thw = grid.first, thw.count == 3
    else { throw QwenVisionPreprocessorError.invalidSpatialDimensions }
    let values = try Data(contentsOf: binURL)
    let expectedCount = shape.reduce(1, *)
    let floats: [Float32] = values.withUnsafeBytes { raw in
      let ptr = raw.bindMemory(to: Float32.self)
      return Array(ptr.prefix(expectedCount))
    }
    guard floats.count == expectedCount else {
      throw QwenVisionPreprocessorError.invalidSpatialDimensions
    }
    let tokens: Int
    let patchVolume: Int
    switch shape.count {
    case 2:
      tokens = shape[0]
      patchVolume = shape[1]
    case 3:
      guard shape[0] == 1 else { throw QwenVisionPreprocessorError.invalidSpatialDimensions }
      tokens = shape[1]
      patchVolume = shape[2]
    default:
      throw QwenVisionPreprocessorError.invalidSpatialDimensions
    }
    var patches = MLXArray(floats, shape).asType(dtype)
    if shape.count == 3 {
      patches = patches.reshaped(tokens, patchVolume)
    }
    let gridObj = QwenVisionGrid(temporal: thw[0], height: thw[1], width: thw[2])
    // Use model patch size (14) for back-computing resized size from grid
    let resized = (height: thw[1] * 14, width: thw[2] * 14)
    return QwenVisionPreprocessorOutput(patches: patches, grid: gridObj, resizedSize: resized)
  }
}
