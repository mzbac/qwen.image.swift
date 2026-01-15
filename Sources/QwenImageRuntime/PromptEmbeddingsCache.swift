import Foundation
import QwenImage
import MLX

// MARK: - Cache Key

/// Cache key for prompt embeddings - includes all factors that affect encoding.
///
/// This key captures everything that could affect the encoded embeddings:
/// - Model identity (ID, revision, quantization)
/// - Tokenizer configuration
/// - Prompt content
public struct PromptEmbeddingsCacheKey: Hashable, Sendable {
  /// Model identifier (repo ID or local path)
  public let modelId: String

  /// Model revision/commit hash
  public let revision: String

  /// Model architecture/config identity (prevents cross-variant cache collisions)
  public let modelDescriptorId: String

  /// Quantization configuration identifier (e.g., "q4_64_affine" or "none")
  public let quantizationId: String

  /// Data type used for encoding (e.g., "bfloat16", "float32")
  public let dtype: String

  /// Maximum sequence length used for encoding
  public let maxLength: Int

  /// The prompt text
  public let prompt: String

  /// The negative prompt text (if any)
  public let negativePrompt: String?

  public init(
    modelId: String,
    revision: String = "main",
    modelDescriptorId: String = "",
    quantizationId: String = "none",
    dtype: String = "bfloat16",
    maxLength: Int,
    prompt: String,
    negativePrompt: String? = nil
  ) {
    self.modelId = modelId
    self.revision = revision
    self.modelDescriptorId = modelDescriptorId
    self.quantizationId = quantizationId
    self.dtype = dtype
    self.maxLength = maxLength
    self.prompt = prompt
    self.negativePrompt = negativePrompt
  }
}

// MARK: - Cache Entry

/// A cached prompt encoding with metadata.
struct PromptEmbeddingsCacheEntry {
  let encoding: QwenGuidanceEncoding
  let createdAt: Date
  let accessCount: Int

  init(encoding: QwenGuidanceEncoding) {
    self.encoding = encoding
    self.createdAt = Date()
    self.accessCount = 1
  }

  func incrementingAccessCount() -> PromptEmbeddingsCacheEntry {
    PromptEmbeddingsCacheEntry(
      encoding: encoding,
      createdAt: createdAt,
      accessCount: accessCount + 1
    )
  }

  private init(encoding: QwenGuidanceEncoding, createdAt: Date, accessCount: Int) {
    self.encoding = encoding
    self.createdAt = createdAt
    self.accessCount = accessCount
  }
}

// MARK: - Cache

/// Thread-safe cache for prompt embeddings.
///
/// This cache stores precomputed prompt embeddings that can be reused across
/// multiple generation calls. The cache key includes all factors that affect
/// the encoded embeddings (model, quantization, prompt text, etc.).
///
/// Usage:
/// ```swift
/// let cache = PromptEmbeddingsCache(maxEntries: 10)
///
/// let key = PromptEmbeddingsCacheKey(
///   modelId: "Qwen/Qwen-Image",
///   revision: "main",
///   maxLength: 256,
///   prompt: "A beautiful landscape"
/// )
///
/// if let cached = await cache.get(key: key) {
///   // Use cached encoding
/// } else {
///   let encoding = try pipeline.encodeGuidancePrompts(...)
///   await cache.set(key: key, value: encoding)
/// }
/// ```
public actor PromptEmbeddingsCache {
  private var cache: [PromptEmbeddingsCacheKey: PromptEmbeddingsCacheEntry] = [:]
  private let maxEntries: Int

  /// Create a new cache with the specified maximum number of entries.
  /// - Parameter maxEntries: Maximum number of cached embeddings. When exceeded,
  ///   the least recently used entry is evicted. Default is 10.
  public init(maxEntries: Int = 10) {
    self.maxEntries = max(1, maxEntries)
  }

  /// Get a cached encoding for the given key.
  /// - Parameter key: The cache key.
  /// - Returns: The cached encoding, or nil if not found.
  public func get(key: PromptEmbeddingsCacheKey) -> QwenGuidanceEncoding? {
    guard let entry = cache[key] else { return nil }
    // Update access count for LRU tracking
    cache[key] = entry.incrementingAccessCount()
    return entry.encoding
  }

  /// Cache an encoding for the given key.
  /// - Parameters:
  ///   - key: The cache key.
  ///   - value: The encoding to cache.
  public func set(key: PromptEmbeddingsCacheKey, value: QwenGuidanceEncoding) {
    // Evict if at capacity
    if cache.count >= maxEntries && cache[key] == nil {
      evictLeastRecentlyUsed()
    }
    cache[key] = PromptEmbeddingsCacheEntry(encoding: value)
  }

  /// Remove a specific cached encoding.
  /// - Parameter key: The cache key to invalidate.
  public func invalidate(key: PromptEmbeddingsCacheKey) {
    cache.removeValue(forKey: key)
  }

  /// Remove all cached encodings.
  public func invalidateAll() {
    cache.removeAll()
  }

  /// Remove all cached encodings for a specific model.
  /// - Parameter modelId: The model ID to invalidate.
  public func invalidateForModel(_ modelId: String) {
    let keysToRemove = cache.keys.filter { $0.modelId == modelId }
    for key in keysToRemove {
      cache.removeValue(forKey: key)
    }
  }

  /// Remove all cached encodings for a specific model and revision.
  /// - Parameters:
  ///   - modelId: The model ID.
  ///   - revision: The model revision.
  public func invalidateForModel(_ modelId: String, revision: String) {
    let keysToRemove = cache.keys.filter {
      $0.modelId == modelId && $0.revision == revision
    }
    for key in keysToRemove {
      cache.removeValue(forKey: key)
    }
  }

  /// The current number of cached entries.
  public var count: Int {
    cache.count
  }

  /// Check if a key is in the cache.
  /// - Parameter key: The cache key.
  /// - Returns: True if the key is cached.
  public func contains(key: PromptEmbeddingsCacheKey) -> Bool {
    cache[key] != nil
  }

  // MARK: - Private

  private func evictLeastRecentlyUsed() {
    // Find the entry with lowest access count (ties broken by oldest creation date)
    guard let lruKey = cache.min(by: { a, b in
      if a.value.accessCount != b.value.accessCount {
        return a.value.accessCount < b.value.accessCount
      }
      return a.value.createdAt < b.value.createdAt
    })?.key else {
      return
    }
    cache.removeValue(forKey: lruKey)
  }
}

// MARK: - Convenience Extensions

extension PromptEmbeddingsCacheKey {
  /// Create a cache key from pipeline configuration.
  public static func from(
    modelPath: URL,
    revision: String = "main",
    quantization: QwenQuantizationSpec? = nil,
    dtype: DType = .bfloat16,
    maxLength: Int,
    prompt: String,
    negativePrompt: String? = nil
  ) -> PromptEmbeddingsCacheKey {
    let modelId = modelPath.lastPathComponent
    let modelDescriptorId = (try? QwenModelDescriptor.load(from: modelPath).identity) ?? ""
    let quantId: String
    if let q = quantization {
      quantId = "q\(q.bits)_\(q.groupSize)_\(q.mode)"
    } else {
      quantId = "none"
    }
    let dtypeStr = dtype.stableName

    return PromptEmbeddingsCacheKey(
      modelId: modelId,
      revision: revision,
      modelDescriptorId: modelDescriptorId,
      quantizationId: quantId,
      dtype: dtypeStr,
      maxLength: maxLength,
      prompt: prompt,
      negativePrompt: negativePrompt
    )
  }
}
