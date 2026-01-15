import Foundation
import MLX

enum QwenMLXImageResizer {
  static func resizeNCHW(
    _ image: MLXArray,
    targetHeight: Int,
    targetWidth: Int
  ) -> MLXArray {
    precondition(image.ndim == 4, "Expected NCHW image tensor [batch, channels, height, width].")
    precondition(targetHeight > 0 && targetWidth > 0, "Target size must be positive.")

    let srcHeight = image.dim(2)
    let srcWidth = image.dim(3)

    if srcHeight == targetHeight && srcWidth == targetWidth {
      return image
    }

    return bilinearResizeNCHW(image, targetHeight: targetHeight, targetWidth: targetWidth)
  }

  static func resizeCHW(
    _ image: MLXArray,
    targetHeight: Int,
    targetWidth: Int
  ) -> MLXArray {
    precondition(image.ndim == 3, "Expected CHW image tensor [channels, height, width].")
    let channels = image.dim(0)
    let srcHeight = image.dim(1)
    let srcWidth = image.dim(2)

    if srcHeight == targetHeight && srcWidth == targetWidth {
      return image
    }

    let batched = image.reshaped(1, channels, srcHeight, srcWidth)
    let resized = resizeNCHW(batched, targetHeight: targetHeight, targetWidth: targetWidth)
    return resized.squeezed(axis: 0)
  }

  private static func bilinearResizeNCHW(
    _ image: MLXArray,
    targetHeight: Int,
    targetWidth: Int
  ) -> MLXArray {
    let srcHeight = image.dim(2)
    let srcWidth = image.dim(3)

    let scaleH = Float32(srcHeight) / Float32(targetHeight)
    let scaleW = Float32(srcWidth) / Float32(targetWidth)

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

    let y0 = MLX.floor(yCoords).asType(.int32)
    let y1 = MLX.minimum(y0 + 1, MLXArray(Int32(srcHeight - 1)))
    let wy = (yCoords - y0.asType(.float32)).reshaped([1, 1, targetHeight, 1])

    let x0 = MLX.floor(xCoords).asType(.int32)
    let x1 = MLX.minimum(x0 + 1, MLXArray(Int32(srcWidth - 1)))
    let wx = (xCoords - x0.asType(.float32)).reshaped([1, 1, 1, targetWidth])

    let transposed = image.transposed(0, 1, 3, 2)
    let rowsY0 = MLX.take(transposed, y0, axis: 3).transposed(0, 1, 3, 2)
    let rowsY1 = MLX.take(transposed, y1, axis: 3).transposed(0, 1, 3, 2)

    let interpY = rowsY0 * (1 - wy) + rowsY1 * wy

    let colsX0 = MLX.take(interpY, x0, axis: 3)
    let colsX1 = MLX.take(interpY, x1, axis: 3)

    let result = colsX0 * (1 - wx) + colsX1 * wx
    return result.asType(image.dtype)
  }
}

