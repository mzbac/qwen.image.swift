import Foundation
import QwenImage
import MLX
import Logging

private actor LayeredPipelineExecutor {
  private let pipeline: QwenLayeredPipeline

  init(pipeline: QwenLayeredPipeline) {
    self.pipeline = pipeline
  }

  func encodePromptToEncoding(_ prompt: String, dtype: DType) throws -> LayeredPromptEncoding {
    try pipeline.encodePromptToEncoding(prompt, dtype: dtype)
  }

  func generate(
    image: MLXArray,
    parameters: LayeredGenerationParameters,
    promptEncoding: LayeredPromptEncoding,
    negativePromptEncoding: LayeredPromptEncoding?,
    progress: ((Int, Int, Float) -> Void)?
  ) async throws -> [MLXArray] {
    try await pipeline.generate(
      image: image,
      parameters: parameters,
      promptEncoding: promptEncoding,
      negativePromptEncoding: negativePromptEncoding,
      progress: progress
    )
  }

  func generate(
    image: MLXArray,
    parameters: LayeredGenerationParameters,
    promptEncoding: LayeredPromptEncoding,
    negativePromptEncoding: LayeredPromptEncoding?,
    eventEmitter: AsyncThrowingStreamEmitter<QwenLayeredGenerationEvent>
  ) async throws -> [MLXArray] {
    try await pipeline.generate(
      image: image,
      parameters: parameters,
      promptEncoding: promptEncoding,
      negativePromptEncoding: negativePromptEncoding,
      progress: { step, total, fraction in
        eventEmitter.yield(.progress(QwenLayeredProgressInfo(step: step, total: total, fractionCompleted: fraction)))
      }
    )
  }

  func applyLora(from url: URL, scale: Float) throws {
    try pipeline.applyLora(from: url, scale: scale)
  }

  func releaseTextEncoder() {
    pipeline.releaseTextEncoder()
  }

  func releaseTokenizer() {
    pipeline.releaseTokenizer()
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
}

// MARK: - Session Configuration

/// Configuration for LayeredPipelineSession resource management policies.
public struct LayeredPipelineSessionConfiguration: Sendable {
  /// Whether to release the text encoder after encoding prompts.
  /// Default is true to save memory during the denoising loop.
  public var releaseTextEncoderAfterEncoding: Bool

  /// Maximum number of cached caption embeddings.
  public var maxCachedCaptions: Int

  /// GPU cache limit in bytes. If nil, no limit is set by the session.
  public var gpuCacheLimit: Int?

  /// Whether to use fast (but less collision-resistant) image hashing.
  /// Fast hashing is suitable for most use cases.
  public var useFastImageHash: Bool

  /// Default configuration with memory-saving defaults.
  public static let `default` = LayeredPipelineSessionConfiguration(
    releaseTextEncoderAfterEncoding: true,
    maxCachedCaptions: 5,
    gpuCacheLimit: nil,
    useFastImageHash: true
  )

  /// Configuration optimized for fast prompt iteration.
  public static let fastPromptIteration = LayeredPipelineSessionConfiguration(
    releaseTextEncoderAfterEncoding: false,
    maxCachedCaptions: 10,
    gpuCacheLimit: nil,
    useFastImageHash: true
  )

  /// Configuration optimized for memory-constrained environments.
  public static let lowMemory = LayeredPipelineSessionConfiguration(
    releaseTextEncoderAfterEncoding: true,
    maxCachedCaptions: 2,
    gpuCacheLimit: 2 * 1024 * 1024 * 1024,  // 2GB
    useFastImageHash: true
  )

  public init(
    releaseTextEncoderAfterEncoding: Bool = true,
    maxCachedCaptions: Int = 5,
    gpuCacheLimit: Int? = nil,
    useFastImageHash: Bool = true
  ) {
    self.releaseTextEncoderAfterEncoding = releaseTextEncoderAfterEncoding
    self.maxCachedCaptions = maxCachedCaptions
    self.gpuCacheLimit = gpuCacheLimit
    self.useFastImageHash = useFastImageHash
  }
}

// MARK: - Session

/// Session wrapper for QwenLayeredPipeline with policy management.
///
/// This actor wraps a QwenLayeredPipeline and provides:
/// - Automatic caption embedding caching with proper cache keys
/// - Image hashing from original bytes (not MLX arrays)
/// - Configurable text encoder release policy
/// - Thread-safe access to the pipeline
///
/// Usage:
/// ```swift
/// let pipeline = try await QwenLayeredPipeline.load(from: modelPath)
///
/// let session = LayeredPipelineSession(
///   pipeline: pipeline,
///   modelId: "Qwen/Qwen-Image-Layered",
///   revision: "main"
/// )
///
/// // Load image data for cache key computation
/// let imageData = try Data(contentsOf: imageURL)
/// let imageArray = ... // Convert to MLXArray
///
/// let params = LayeredGenerationParameters(layers: 4, resolution: 640)
/// let layers = try await session.generate(
///   imageData: imageData,
///   image: imageArray,
///   parameters: params
/// )
/// ```
public actor LayeredPipelineSession {
  private let pipelineExecutor: LayeredPipelineExecutor
  private let captionCache: LayeredCaptionCache
  private let modelId: String
  private let revision: String
  private let configuration: LayeredPipelineSessionConfiguration
  private var logger = Logger(label: "qwen.layered.session")
  private var appliedLora: (url: URL, scale: Float)?

  /// Create a new session wrapping a pipeline.
  /// - Parameters:
  ///   - pipeline: The pipeline to wrap.
  ///   - modelId: Model identifier for cache keys.
  ///   - revision: Model revision for cache keys.
  ///   - configuration: Session configuration.
  public init(
    pipeline: QwenLayeredPipeline,
    modelId: String,
    revision: String = "main",
    configuration: LayeredPipelineSessionConfiguration = .default
  ) {
    self.pipelineExecutor = LayeredPipelineExecutor(pipeline: pipeline)
    self.modelId = modelId
    self.revision = revision
    self.configuration = configuration
    self.captionCache = LayeredCaptionCache(maxEntries: configuration.maxCachedCaptions)

    // Apply GPU cache limit if specified
    if let limit = configuration.gpuCacheLimit {
      GPU.set(cacheLimit: limit)
    }
  }

  // MARK: - Generation

  /// Generate layer images with automatic caching and resource management.
  ///
  /// This method:
  /// 1. Computes an image hash from the original bytes (no MLX.eval needed)
  /// 2. Checks the cache for existing caption embeddings
  /// 3. Encodes the caption if not cached
  /// 4. Caches the encoding for future use
  /// 5. Optionally releases text encoder based on configuration
  /// 6. Runs generation with the cached/computed encoding
  ///
  /// - Parameters:
  ///   - imageData: Original image file data (used for cache key hashing).
  ///   - image: The image as MLXArray.
  ///   - parameters: Generation parameters.
  ///   - progress: Optional progress callback.
  /// - Returns: Array of decoded layer images.
  public func generate(
    imageData: Data,
    image: MLXArray,
    parameters: LayeredGenerationParameters,
    progress: ((Int, Int, Float) -> Void)? = nil
  ) async throws -> [MLXArray] {
    // Compute image hash from original bytes (no MLX.eval needed!)
    let imageHash = configuration.useFastImageHash
      ? ImageHashUtility.fastHash(from: imageData)
      : ImageHashUtility.sha256Hash(from: imageData)

    let promptText = parameters.prompt ?? ""

    // Build cache key
    let cacheKey = LayeredCaptionCacheKey(
      modelId: modelId,
      revision: revision,
      prompt: promptText,
      negativePrompt: parameters.negativePrompt,
      imageHash: imageHash
    )

    // Get or compute encodings
    let (promptEncoding, negativeEncoding) = try await getOrComputeEncodings(
      cacheKey: cacheKey,
      prompt: promptText,
      negativePrompt: parameters.negativePrompt,
      dtype: image.dtype
    )

    // Release text encoder if configured
    if configuration.releaseTextEncoderAfterEncoding {
      await pipelineExecutor.releaseTextEncoder()
      logger.debug("Released text encoder after encoding")
    }

    // Generate using the policy-free overload
    return try await pipelineExecutor.generate(
      image: image,
      parameters: parameters,
      promptEncoding: promptEncoding,
      negativePromptEncoding: negativeEncoding,
      progress: progress
    )
  }

  public func generateStream(
    imageData: Data,
    image: MLXArray,
    parameters: LayeredGenerationParameters
  ) -> AsyncThrowingStream<QwenLayeredGenerationEvent, Error> {
    AsyncThrowingStream { continuation in
      let emitter = AsyncThrowingStreamEmitter<QwenLayeredGenerationEvent>(continuation)
      let task = Task {
        do {
          let imageHash = configuration.useFastImageHash
            ? ImageHashUtility.fastHash(from: imageData)
            : ImageHashUtility.sha256Hash(from: imageData)

          let promptText = parameters.prompt ?? ""

          let cacheKey = LayeredCaptionCacheKey(
            modelId: modelId,
            revision: revision,
            prompt: promptText,
            negativePrompt: parameters.negativePrompt,
            imageHash: imageHash
          )

          let (promptEncoding, negativeEncoding) = try await getOrComputeEncodings(
            cacheKey: cacheKey,
            prompt: promptText,
            negativePrompt: parameters.negativePrompt,
            dtype: image.dtype
          )

          if configuration.releaseTextEncoderAfterEncoding {
            await pipelineExecutor.releaseTextEncoder()
          }

          let layers = try await pipelineExecutor.generate(
            image: image,
            parameters: parameters,
            promptEncoding: promptEncoding,
            negativePromptEncoding: negativeEncoding,
            eventEmitter: emitter
          )
          emitter.yield(.output(layers))
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

  /// Generate layer images from an image file URL.
  ///
  /// Convenience method that loads image data and converts to MLXArray.
  ///
  /// - Parameters:
  ///   - imageURL: URL to the image file.
  ///   - parameters: Generation parameters.
  ///   - imageConverter: Function to convert image data to MLXArray.
  ///   - progress: Optional progress callback.
  /// - Returns: Array of decoded layer images.
  public func generate(
    imageURL: URL,
    parameters: LayeredGenerationParameters,
    imageConverter: (Data) throws -> MLXArray,
    progress: ((Int, Int, Float) -> Void)? = nil
  ) async throws -> [MLXArray] {
    guard let imageData = ImageHashUtility.loadImageData(from: imageURL) else {
      throw LayeredPipelineError.invalidImageShape
    }

    let imageArray = try imageConverter(imageData)

    return try await generate(
      imageData: imageData,
      image: imageArray,
      parameters: parameters,
      progress: progress
    )
  }

  public func generateStream(
    imageURL: URL,
    parameters: LayeredGenerationParameters,
    imageConverter: @escaping (Data) throws -> MLXArray
  ) -> AsyncThrowingStream<QwenLayeredGenerationEvent, Error> {
    AsyncThrowingStream { continuation in
      let emitter = AsyncThrowingStreamEmitter<QwenLayeredGenerationEvent>(continuation)
      let task = Task {
        do {
          guard let imageData = ImageHashUtility.loadImageData(from: imageURL) else {
            throw LayeredPipelineError.invalidImageShape
          }
          let imageArray = try imageConverter(imageData)
          for try await event in generateStream(imageData: imageData, image: imageArray, parameters: parameters) {
            emitter.yield(event)
          }
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

  // MARK: - Encoding

  /// Get or compute caption encodings with caching.
  private func getOrComputeEncodings(
    cacheKey: LayeredCaptionCacheKey,
    prompt: String,
    negativePrompt: String?,
    dtype: DType
  ) async throws -> (LayeredPromptEncoding, LayeredPromptEncoding?) {
    // Check cache first
    if let cached = await captionCache.get(key: cacheKey) {
      logger.debug("Cache hit for caption encoding")
      return cached
    }

    logger.debug("Cache miss, encoding caption")

    // Encode prompt
    let promptEncoding = try await pipelineExecutor.encodePromptToEncoding(prompt, dtype: dtype)

    // Encode negative prompt if provided
    var negativeEncoding: LayeredPromptEncoding? = nil
    if let negPrompt = negativePrompt {
      negativeEncoding = try await pipelineExecutor.encodePromptToEncoding(negPrompt, dtype: dtype)
    }

    // Cache the encodings
    await captionCache.set(
      key: cacheKey,
      encoding: promptEncoding,
      negativeEncoding: negativeEncoding
    )

    return (promptEncoding, negativeEncoding)
  }

  /// Encode a prompt without caching.
  /// Use this for one-off encodings or when you want to manage caching externally.
  public func encodePrompt(_ prompt: String, dtype: DType) async throws -> LayeredPromptEncoding {
    try await pipelineExecutor.encodePromptToEncoding(prompt, dtype: dtype)
  }

  // MARK: - Lifecycle Management

  /// Release the text encoder to free memory.
  public func releaseTextEncoder() async {
    await pipelineExecutor.releaseTextEncoder()
    logger.debug("Text encoder released explicitly")
  }

  /// Release the tokenizer to free memory.
  public func releaseTokenizer() async {
    await pipelineExecutor.releaseTokenizer()
  }

  /// Reload the text encoder (if weights directory is set).
  public func reloadTextEncoder() async throws {
    try await pipelineExecutor.reloadTextEncoder()
  }

  /// Reload the tokenizer (if weights directory is set).
  public func reloadTokenizer() async throws {
    try await pipelineExecutor.reloadTokenizer()
  }

  // MARK: - Cache Management

  /// Clear all cached caption embeddings.
  public func clearCache() async {
    await captionCache.invalidateAll()
    logger.debug("Caption cache cleared")
  }

  /// Invalidate cache entries for a specific image.
  /// - Parameter imageData: The image data to invalidate.
  public func invalidateCacheForImage(_ imageData: Data) async {
    let hash = configuration.useFastImageHash
      ? ImageHashUtility.fastHash(from: imageData)
      : ImageHashUtility.sha256Hash(from: imageData)
    await captionCache.invalidateForImage(hash)
  }

  /// Invalidate cache entries for the current model.
  public func invalidateModelCache() async {
    await captionCache.invalidateForModel(modelId)
  }

  /// The current number of cached captions.
  public var cacheCount: Int {
    get async {
      await captionCache.count
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

  // MARK: - LoRA

  /// Apply a LoRA adapter to the pipeline.
  /// - Parameters:
  ///   - url: URL to the LoRA safetensors file.
  ///   - scale: LoRA scale factor.
  public func applyLora(from url: URL, scale: Float = 1.0) async throws {
    if let appliedLora {
      guard appliedLora.url == url, appliedLora.scale == scale else {
        throw QwenImageRuntimeError.loraAlreadyApplied(
          appliedURL: appliedLora.url,
          appliedScale: appliedLora.scale,
          requestedURL: url,
          requestedScale: scale
        )
      }
      return
    }

    try await pipelineExecutor.applyLora(from: url, scale: scale)
    // Invalidate cache since model weights changed
    await captionCache.invalidateAll()
    appliedLora = (url: url, scale: scale)
    logger.info("Applied LoRA and invalidated cache")
  }
}

// MARK: - Convenience

extension LayeredPipelineSession {
  /// Check if an encoding is cached for the given parameters.
  nonisolated public func hasCachedEncoding(
    imageData: Data,
    prompt: String,
    negativePrompt: String?
  ) async -> Bool {
    let hash = configuration.useFastImageHash
      ? ImageHashUtility.fastHash(from: imageData)
      : ImageHashUtility.sha256Hash(from: imageData)

    let cacheKey = LayeredCaptionCacheKey(
      modelId: modelId,
      revision: revision,
      prompt: prompt,
      negativePrompt: negativePrompt,
      imageHash: hash
    )

    return await captionCache.contains(key: cacheKey)
  }
}
