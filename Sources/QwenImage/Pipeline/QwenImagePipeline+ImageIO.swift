import Foundation
import MLX

#if canImport(CoreGraphics)
import CoreGraphics
#endif

#if canImport(AppKit)
import AppKit
#elseif canImport(UIKit)
import UIKit
#endif

extension QwenImagePipeline {
#if canImport(CoreGraphics)
  public func generateImage(
    parameters: GenerationParameters,
    model: QwenModelConfiguration,
    maxPromptLength: Int? = nil,
    seed: UInt64? = nil
  ) async throws -> PipelineImage {
    let pixels = try await generatePixels(
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
}
