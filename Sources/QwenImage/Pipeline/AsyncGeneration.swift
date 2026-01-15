import Foundation
import MLX

#if canImport(CoreGraphics)
import CoreGraphics
#endif

public enum QwenGenerationEvent {
  case progress(ProgressInfo)
  case output(MLXArray)
}

extension QwenImagePipeline {
  public func generatePixelsStream(
    parameters: GenerationParameters,
    model: QwenModelConfiguration,
    maxPromptLength: Int? = nil,
    seed: UInt64? = nil
  ) -> AsyncThrowingStream<QwenGenerationEvent, Error> {
    AsyncThrowingStream { continuation in
      let task = Task {
        do {
          let pixels = try await generatePixels(
            parameters: parameters,
            model: model,
            maxPromptLength: maxPromptLength,
            seed: seed,
            progress: { continuation.yield(.progress($0)) }
          )
          continuation.yield(.output(pixels))
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }
      continuation.onTermination = { _ in
        task.cancel()
      }
    }
  }

  public func generatePixelsStream(
    parameters: GenerationParameters,
    model: QwenModelConfiguration,
    guidanceEncoding: QwenGuidanceEncoding,
    seed: UInt64? = nil
  ) -> AsyncThrowingStream<QwenGenerationEvent, Error> {
    AsyncThrowingStream { continuation in
      let task = Task {
        do {
          let pixels = try await generatePixels(
            parameters: parameters,
            model: model,
            guidanceEncoding: guidanceEncoding,
            seed: seed,
            progress: { continuation.yield(.progress($0)) }
          )
          continuation.yield(.output(pixels))
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }
      continuation.onTermination = { _ in
        task.cancel()
      }
    }
  }

#if canImport(CoreGraphics)
  public func generateEditedPixelsStream(
    parameters: GenerationParameters,
    model: QwenModelConfiguration,
    referenceImage: CGImage,
    maxPromptLength: Int? = nil,
    seed: UInt64? = nil
  ) -> AsyncThrowingStream<QwenGenerationEvent, Error> {
    AsyncThrowingStream { continuation in
      let task = Task {
        do {
          let pixels = try await generateEditedPixels(
            parameters: parameters,
            model: model,
            referenceImage: referenceImage,
            maxPromptLength: maxPromptLength,
            seed: seed,
            progress: { continuation.yield(.progress($0)) }
          )
          continuation.yield(.output(pixels))
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }
      continuation.onTermination = { _ in
        task.cancel()
      }
    }
  }

  public func generateEditedPixelsStream(
    parameters: GenerationParameters,
    model: QwenModelConfiguration,
    referenceImages: [CGImage],
    maxPromptLength: Int? = nil,
    seed: UInt64? = nil
  ) -> AsyncThrowingStream<QwenGenerationEvent, Error> {
    AsyncThrowingStream { continuation in
      let task = Task {
        do {
          let pixels = try await generateEditedPixels(
            parameters: parameters,
            model: model,
            referenceImages: referenceImages,
            maxPromptLength: maxPromptLength,
            seed: seed,
            progress: { continuation.yield(.progress($0)) }
          )
          continuation.yield(.output(pixels))
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }
      continuation.onTermination = { _ in
        task.cancel()
      }
    }
  }
#endif
}

public struct QwenLayeredProgressInfo {
  public let step: Int
  public let total: Int
  public let fractionCompleted: Float

  public init(step: Int, total: Int, fractionCompleted: Float) {
    self.step = step
    self.total = total
    self.fractionCompleted = fractionCompleted
  }
}

public enum QwenLayeredGenerationEvent {
  case progress(QwenLayeredProgressInfo)
  case output([MLXArray])
}

extension QwenLayeredPipeline {
  public func generateStream(
    image: MLXArray,
    parameters: LayeredGenerationParameters
  ) -> AsyncThrowingStream<QwenLayeredGenerationEvent, Error> {
    AsyncThrowingStream { continuation in
      let task = Task {
        do {
          let layers = try await generate(
            image: image,
            parameters: parameters,
            progress: { step, total, fraction in
              continuation.yield(.progress(QwenLayeredProgressInfo(step: step, total: total, fractionCompleted: fraction)))
            }
          )
          continuation.yield(.output(layers))
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }
      continuation.onTermination = { _ in
        task.cancel()
      }
    }
  }

  public func generateStream(
    image: MLXArray,
    parameters: LayeredGenerationParameters,
    promptEncoding: LayeredPromptEncoding,
    negativePromptEncoding: LayeredPromptEncoding? = nil
  ) -> AsyncThrowingStream<QwenLayeredGenerationEvent, Error> {
    AsyncThrowingStream { continuation in
      let task = Task {
        do {
          let layers = try await generate(
            image: image,
            parameters: parameters,
            promptEncoding: promptEncoding,
            negativePromptEncoding: negativePromptEncoding,
            progress: { step, total, fraction in
              continuation.yield(.progress(QwenLayeredProgressInfo(step: step, total: total, fractionCompleted: fraction)))
            }
          )
          continuation.yield(.output(layers))
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }
      continuation.onTermination = { _ in
        task.cancel()
      }
    }
  }
}
