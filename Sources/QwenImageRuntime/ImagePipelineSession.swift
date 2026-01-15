import Foundation
import QwenImage
import MLX
import Logging

#if canImport(CoreGraphics)
import CoreGraphics
#endif

private actor ImagePipelineExecutor {
  private let pipeline: QwenImagePipeline

  init(pipeline: QwenImagePipeline) {
    self.pipeline = pipeline
  }

  func generatePixels(
    parameters: GenerationParameters,
    model: QwenModelConfiguration,
    guidanceEncoding: QwenGuidanceEncoding,
    seed: UInt64? = nil,
    eventEmitter: AsyncThrowingStreamEmitter<QwenGenerationEvent>? = nil
  ) async throws -> MLXArray {
    if let eventEmitter {
      return try await pipeline.generatePixels(
        parameters: parameters,
        model: model,
        guidanceEncoding: guidanceEncoding,
        seed: seed,
        progress: { info in eventEmitter.yield(.progress(info)) }
      )
    }

    return try await pipeline.generatePixels(
      parameters: parameters,
      model: model,
      guidanceEncoding: guidanceEncoding,
      seed: seed
    )
  }

#if canImport(CoreGraphics)
  func generateEditedPixels(
    parameters: GenerationParameters,
    model: QwenModelConfiguration,
    referenceImage: CGImage,
    maxPromptLength: Int? = nil,
    seed: UInt64? = nil,
    eventEmitter: AsyncThrowingStreamEmitter<QwenGenerationEvent>? = nil
  ) async throws -> MLXArray {
    if let eventEmitter {
      return try await pipeline.generateEditedPixels(
        parameters: parameters,
        model: model,
        referenceImage: referenceImage,
        maxPromptLength: maxPromptLength,
        seed: seed,
        progress: { info in eventEmitter.yield(.progress(info)) }
      )
    }

    return try await pipeline.generateEditedPixels(
      parameters: parameters,
      model: model,
      referenceImage: referenceImage,
      maxPromptLength: maxPromptLength,
      seed: seed
    )
  }

  func generateEditedPixels(
    parameters: GenerationParameters,
    model: QwenModelConfiguration,
    referenceImages: [CGImage],
    maxPromptLength: Int? = nil,
    seed: UInt64? = nil,
    eventEmitter: AsyncThrowingStreamEmitter<QwenGenerationEvent>? = nil
  ) async throws -> MLXArray {
    if let eventEmitter {
      return try await pipeline.generateEditedPixels(
        parameters: parameters,
        model: model,
        referenceImages: referenceImages,
        maxPromptLength: maxPromptLength,
        seed: seed,
        progress: { info in eventEmitter.yield(.progress(info)) }
      )
    }

    return try await pipeline.generateEditedPixels(
      parameters: parameters,
      model: model,
      referenceImages: referenceImages,
      maxPromptLength: maxPromptLength,
      seed: seed
    )
  }

  func makeImage(from pixels: MLXArray) throws -> PipelineImage {
    try pipeline.makeImage(from: pixels)
  }
#endif

  func encodeGuidancePrompts(
    prompt: String,
    negativePrompt: String?,
    maxLength: Int
  ) throws -> QwenGuidanceEncoding {
    try pipeline.encodeGuidancePrompts(
      prompt: prompt,
      negativePrompt: negativePrompt,
      maxLength: maxLength
    )
  }

  func loadModelDescriptor() throws -> QwenModelDescriptor {
    try pipeline.loadModelDescriptor()
  }

  func resolvedDType() throws -> DType {
    try pipeline.resolvedDType()
  }

  func promptEncodingQuantizationId() -> String {
    pipeline.promptEncodingQuantizationId()
  }

  func setPendingLora(from url: URL, scale: Float) {
    pipeline.setPendingLora(from: url, scale: scale)
  }

  func applyLora(from url: URL, scale: Float) throws {
    try pipeline.applyLora(from: url, scale: scale)
  }

  func releaseEncoders() {
    pipeline.releaseEncoders()
  }

  func releaseTextEncoder() {
    pipeline.releaseTextEncoder()
  }

  func releaseVisionTower() {
    pipeline.releaseVisionTower()
  }

  func reloadTextEncoder() throws {
    try pipeline.reloadTextEncoder()
  }

  func reloadTokenizer() throws {
    try pipeline.reloadTokenizer()
  }

  func isTextEncoderLoaded() -> Bool {
    pipeline.isTextEncoderLoaded
  }

  func isTokenizerLoaded() -> Bool {
    pipeline.isTokenizerLoaded
  }

  func isVisionTowerLoaded() -> Bool {
    pipeline.isVisionTowerLoaded
  }

  func isUNetLoaded() -> Bool {
    pipeline.isUNetLoaded
  }

  func isVAELoaded() -> Bool {
    pipeline.isVAELoaded
  }
}

// MARK: - Session Configuration

/// Configuration for ImagePipelineSession resource management policies.
public struct ImagePipelineSessionConfiguration: Sendable {
  /// Whether to release encoders after encoding prompts (before generation).
  /// Default is true to save memory during the denoising loop.
  public var releaseEncodersAfterEncoding: Bool

  /// Maximum number of cached prompt embeddings.
  public var maxCachedEmbeddings: Int

  /// GPU cache limit in bytes. If nil, no limit is set by the session.
  public var gpuCacheLimit: Int?

  /// Default configuration with memory-saving defaults.
  public static let `default` = ImagePipelineSessionConfiguration(
    releaseEncodersAfterEncoding: true,
    maxCachedEmbeddings: 10,
    gpuCacheLimit: nil
  )

  /// Configuration optimized for fast prompt iteration (keeps encoders loaded).
  public static let fastPromptIteration = ImagePipelineSessionConfiguration(
    releaseEncodersAfterEncoding: false,
    maxCachedEmbeddings: 20,
    gpuCacheLimit: nil
  )

  /// Configuration optimized for memory-constrained environments.
  public static let lowMemory = ImagePipelineSessionConfiguration(
    releaseEncodersAfterEncoding: true,
    maxCachedEmbeddings: 3,
    gpuCacheLimit: 2 * 1024 * 1024 * 1024  // 2GB
  )

  public init(
    releaseEncodersAfterEncoding: Bool = true,
    maxCachedEmbeddings: Int = 10,
    gpuCacheLimit: Int? = nil
  ) {
    self.releaseEncodersAfterEncoding = releaseEncodersAfterEncoding
    self.maxCachedEmbeddings = maxCachedEmbeddings
    self.gpuCacheLimit = gpuCacheLimit
  }
}

// MARK: - Session

/// Session wrapper for QwenImagePipeline with policy management.
///
/// This actor wraps a QwenImagePipeline and provides:
/// - Automatic prompt embedding caching with proper cache keys
/// - Configurable encoder release policy
/// - Thread-safe access to the pipeline
///
/// Usage:
/// ```swift
/// let pipeline = QwenImagePipeline(config: .textToImage)
/// pipeline.setBaseDirectory(modelPath)
/// try pipeline.prepareTokenizer(from: modelPath)
/// try pipeline.prepareTextEncoder(from: modelPath)
///
/// let session = ImagePipelineSession(
///   pipeline: pipeline,
///   modelId: "Qwen/Qwen-Image",
///   revision: "main"
/// )
///
/// let params = GenerationParameters(prompt: "A cat", width: 1024, height: 1024, steps: 30)
/// let pixels = try await session.generate(parameters: params, model: modelConfig)
/// ```
public actor ImagePipelineSession {
  private let pipelineExecutor: ImagePipelineExecutor
  private let embeddingsCache: PromptEmbeddingsCache
  private let modelId: String
  private let revision: String
  private let configuration: ImagePipelineSessionConfiguration
  private var logger = Logger(label: "qwen.image.session")
  private var appliedLora: (url: URL, scale: Float)?

  /// Create a new session wrapping a pipeline.
  /// - Parameters:
  ///   - pipeline: The pipeline to wrap. Should have tokenizer and text encoder loaded.
  ///   - modelId: Model identifier for cache keys.
  ///   - revision: Model revision for cache keys.
  ///   - configuration: Session configuration.
  public init(
    pipeline: QwenImagePipeline,
    modelId: String,
    revision: String = "main",
    configuration: ImagePipelineSessionConfiguration = .default
  ) {
    self.pipelineExecutor = ImagePipelineExecutor(pipeline: pipeline)
    self.modelId = modelId
    self.revision = revision
    self.configuration = configuration
    self.embeddingsCache = PromptEmbeddingsCache(maxEntries: configuration.maxCachedEmbeddings)

    // Apply GPU cache limit if specified
    if let limit = configuration.gpuCacheLimit {
      GPU.set(cacheLimit: limit)
    }
  }

  // MARK: - Generation

  /// Generate an image with automatic caching and resource management.
  ///
  /// This method:
  /// 1. Checks the cache for existing prompt embeddings
  /// 2. Encodes the prompt if not cached (requires tokenizer/textEncoder)
  /// 3. Caches the encoding for future use
  /// 4. Optionally releases encoders based on configuration
  /// 5. Runs generation with the cached/computed encoding
  ///
  /// - Parameters:
  ///   - parameters: Generation parameters.
  ///   - model: Model configuration.
  ///   - maxPromptLength: Maximum prompt length (defaults to model's max).
  ///   - seed: Random seed for reproducibility.
  /// - Returns: Generated pixel array.
  public func generate(
    parameters: GenerationParameters,
    model: QwenModelConfiguration,
    maxPromptLength: Int? = nil,
    seed: UInt64? = nil
  ) async throws -> MLXArray {
    let maxLength = maxPromptLength ?? model.maxSequenceLength

    // Get or compute guidance encoding
    let encoding = try await guidanceEncoding(
      prompt: parameters.prompt,
      negativePrompt: parameters.negativePrompt,
      maxLength: maxLength
    )

    // Release encoders if configured to do so
    if configuration.releaseEncodersAfterEncoding {
      await pipelineExecutor.releaseEncoders()
      logger.debug("Released encoders after encoding")
    }

    // Generate using the policy-free overload
    return try await pipelineExecutor.generatePixels(
      parameters: parameters,
      model: model,
      guidanceEncoding: encoding,
      seed: seed
    )
  }

  public func loadModelDescriptor() async throws -> QwenModelDescriptor {
    try await pipelineExecutor.loadModelDescriptor()
  }

#if canImport(CoreGraphics)
  public func makeImage(from pixels: MLXArray) async throws -> PipelineImage {
    try await pipelineExecutor.makeImage(from: pixels)
  }
#endif

  public func generateStream(
    parameters: GenerationParameters,
    model: QwenModelConfiguration,
    maxPromptLength: Int? = nil,
    seed: UInt64? = nil
  ) -> AsyncThrowingStream<QwenGenerationEvent, Error> {
    AsyncThrowingStream { continuation in
      let emitter = AsyncThrowingStreamEmitter<QwenGenerationEvent>(continuation)
      let task = Task {
        do {
          let maxLength = maxPromptLength ?? model.maxSequenceLength
          let encoding = try await guidanceEncoding(
            prompt: parameters.prompt,
            negativePrompt: parameters.negativePrompt,
            maxLength: maxLength
          )

          if configuration.releaseEncodersAfterEncoding {
            await pipelineExecutor.releaseEncoders()
          }

          let pixels = try await pipelineExecutor.generatePixels(
            parameters: parameters,
            model: model,
            guidanceEncoding: encoding,
            seed: seed,
            eventEmitter: emitter
          )
          emitter.yield(.output(pixels))
          emitter.finish()
        } catch {
          emitter.finish(throwing: error)
        }
      }

      continuation.onTermination = { _ in
        task.cancel()
        emitter.finish()
      }
    }
  }

#if canImport(CoreGraphics)
  public func generateEditedPixels(
    parameters: GenerationParameters,
    model: QwenModelConfiguration,
    referenceImage: CGImage,
    maxPromptLength: Int? = nil,
    seed: UInt64? = nil
  ) async throws -> MLXArray {
    try await pipelineExecutor.generateEditedPixels(
      parameters: parameters,
      model: model,
      referenceImage: referenceImage,
      maxPromptLength: maxPromptLength,
      seed: seed,
      eventEmitter: nil
    )
  }

  public func generateEditedPixels(
    parameters: GenerationParameters,
    model: QwenModelConfiguration,
    referenceImages: [CGImage],
    maxPromptLength: Int? = nil,
    seed: UInt64? = nil
  ) async throws -> MLXArray {
    try await pipelineExecutor.generateEditedPixels(
      parameters: parameters,
      model: model,
      referenceImages: referenceImages,
      maxPromptLength: maxPromptLength,
      seed: seed,
      eventEmitter: nil
    )
  }

  public func generateEditedPixelsStream(
    parameters: GenerationParameters,
    model: QwenModelConfiguration,
    referenceImage: CGImage,
    maxPromptLength: Int? = nil,
    seed: UInt64? = nil
  ) -> AsyncThrowingStream<QwenGenerationEvent, Error> {
    AsyncThrowingStream { continuation in
      let emitter = AsyncThrowingStreamEmitter<QwenGenerationEvent>(continuation)
      let task = Task {
        do {
          let pixels = try await pipelineExecutor.generateEditedPixels(
            parameters: parameters,
            model: model,
            referenceImage: referenceImage,
            maxPromptLength: maxPromptLength,
            seed: seed,
            eventEmitter: emitter
          )
          emitter.yield(.output(pixels))
          emitter.finish()
        } catch {
          emitter.finish(throwing: error)
        }
      }

      continuation.onTermination = { _ in
        task.cancel()
        emitter.finish()
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
      let emitter = AsyncThrowingStreamEmitter<QwenGenerationEvent>(continuation)
      let task = Task {
        do {
          let pixels = try await pipelineExecutor.generateEditedPixels(
            parameters: parameters,
            model: model,
            referenceImages: referenceImages,
            maxPromptLength: maxPromptLength,
            seed: seed,
            eventEmitter: emitter
          )
          emitter.yield(.output(pixels))
          emitter.finish()
        } catch {
          emitter.finish(throwing: error)
        }
      }

      continuation.onTermination = { _ in
        task.cancel()
        emitter.finish()
      }
    }
  }
#endif

  // MARK: - Encoding

  /// Get or compute guidance encoding with caching.
  ///
  /// - Parameters:
  ///   - prompt: The prompt text.
  ///   - negativePrompt: Optional negative prompt.
  ///   - maxLength: Maximum sequence length.
  /// - Returns: The guidance encoding (cached or freshly computed).
  public func guidanceEncoding(
    prompt: String,
    negativePrompt: String?,
    maxLength: Int
  ) async throws -> QwenGuidanceEncoding {
    let descriptor = try await pipelineExecutor.loadModelDescriptor()
    let dtype = try await pipelineExecutor.resolvedDType()
    let quantizationId = await pipelineExecutor.promptEncodingQuantizationId()

    let cacheKey = PromptEmbeddingsCacheKey(
      modelId: modelId,
      revision: revision,
      modelDescriptorId: descriptor.identity,
      quantizationId: quantizationId,
      dtype: dtype.stableName,
      maxLength: maxLength,
      prompt: prompt,
      negativePrompt: negativePrompt
    )

    // Check cache first
    if let cached = await embeddingsCache.get(key: cacheKey) {
      logger.debug("Cache hit for prompt encoding")
      return cached
    }

    logger.debug("Cache miss, encoding prompt")

    try await pipelineExecutor.reloadTokenizer()
    try await pipelineExecutor.reloadTextEncoder()

    let encoding = try await pipelineExecutor.encodeGuidancePrompts(
      prompt: prompt,
      negativePrompt: negativePrompt,
      maxLength: maxLength
    )

    await embeddingsCache.set(key: cacheKey, value: encoding)
    return encoding
  }

  // MARK: - Lifecycle Management

  /// Explicitly release encoder components to free memory.
  public func releaseEncoders() async {
    await pipelineExecutor.releaseEncoders()
    logger.debug("Encoders released explicitly")
  }

  /// Release only the text encoder.
  public func releaseTextEncoder() async {
    await pipelineExecutor.releaseTextEncoder()
  }

  /// Release only the vision tower.
  public func releaseVisionTower() async {
    await pipelineExecutor.releaseVisionTower()
  }

  /// Reload the text encoder (if weights directory is set).
  public func reloadTextEncoder() async throws {
    try await pipelineExecutor.reloadTextEncoder()
  }

  /// Reload the tokenizer (if weights directory is set).
  public func reloadTokenizer() async throws {
    try await pipelineExecutor.reloadTokenizer()
  }

  // MARK: - LoRA

  public func applyLora(from url: URL, scale: Float = 1.0) async throws {
    if let applied = appliedLora {
      guard applied.url == url, applied.scale == scale else {
        throw QwenImageRuntimeError.loraAlreadyApplied(
          appliedURL: applied.url,
          appliedScale: applied.scale,
          requestedURL: url,
          requestedScale: scale
        )
      }
      return
    }

    if await pipelineExecutor.isUNetLoaded() {
      try await pipelineExecutor.applyLora(from: url, scale: scale)
    } else {
      await pipelineExecutor.setPendingLora(from: url, scale: scale)
    }
    appliedLora = (url: url, scale: scale)
  }

  // MARK: - Cache Management

  /// Clear all cached prompt embeddings.
  public func clearCache() async {
    await embeddingsCache.invalidateAll()
    logger.debug("Embeddings cache cleared")
  }

  /// Invalidate cache entries for the current model.
  public func invalidateModelCache() async {
    await embeddingsCache.invalidateForModel(modelId, revision: revision)
  }

  /// The current number of cached embeddings.
  public var cacheCount: Int {
    get async {
      await embeddingsCache.count
    }
  }

  // MARK: - Status

  /// Check if the text encoder is currently loaded.
  public var isTextEncoderLoaded: Bool {
    get async {
      await pipelineExecutor.isTextEncoderLoaded()
    }
  }

  /// Check if the tokenizer is currently loaded.
  public var isTokenizerLoaded: Bool {
    get async {
      await pipelineExecutor.isTokenizerLoaded()
    }
  }

  /// Check if the vision tower is currently loaded.
  public var isVisionTowerLoaded: Bool {
    get async {
      await pipelineExecutor.isVisionTowerLoaded()
    }
  }

  /// Check if the UNet is currently loaded.
  public var isUNetLoaded: Bool {
    get async {
      await pipelineExecutor.isUNetLoaded()
    }
  }

  /// Check if the VAE is currently loaded.
  public var isVAELoaded: Bool {
    get async {
      await pipelineExecutor.isVAELoaded()
    }
  }
}

// MARK: - Sync Wrapper

extension ImagePipelineSession {
  /// Convenience wrapper for cache lookup.
  public func hasCachedEncoding(
    prompt: String,
    negativePrompt: String?,
    maxLength: Int
  ) async -> Bool {
    guard let descriptor = try? await pipelineExecutor.loadModelDescriptor() else {
      return false
    }
    guard let dtype = try? await pipelineExecutor.resolvedDType() else {
      return false
    }
    let quantizationId = await pipelineExecutor.promptEncodingQuantizationId()

    let cacheKey = PromptEmbeddingsCacheKey(
      modelId: modelId,
      revision: revision,
      modelDescriptorId: descriptor.identity,
      quantizationId: quantizationId,
      dtype: dtype.stableName,
      maxLength: maxLength,
      prompt: prompt,
      negativePrompt: negativePrompt
    )
    return await embeddingsCache.contains(key: cacheKey)
  }
}
