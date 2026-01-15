/// QwenImageRuntime - Runtime layer for QwenImage pipelines
///
/// This module provides policy management and caching for QwenImage pipelines.
/// It sits on top of the core QwenImage library and handles:
///
/// - **Prompt embedding caching** with proper cache keys that include model ID,
///   revision, quantization, and prompt content
/// - **Layered caption caching** with image hashing from original bytes
///   (no MLX.eval needed for cache key computation)
/// - **Pipeline sessions** that wrap core pipelines with configurable policies
///   for encoder release, caching, and GPU memory management
/// - **GPU cache policy** utilities for managing MLX memory
///
/// ## Architecture
///
/// The separation between QwenImage (core) and QwenImageRuntime (runtime) follows
/// these principles:
///
/// - **Core is policy-free**: The core library provides deterministic operations
///   and explicit lifecycle hooks. It doesn't make decisions about when to cache,
///   when to release memory, or what GPU limits to use.
///
/// - **Runtime owns policy**: This module provides sessions that wrap core pipelines
///   and implement caching, memory management, and resource lifecycle policies.
///
/// ## Quick Start
///
/// ### Text-to-Image Generation
///
/// ```swift
/// import QwenImage
/// import QwenImageRuntime
///
/// // Create and configure the core pipeline
/// let pipeline = QwenImagePipeline(config: .textToImage)
/// pipeline.setBaseDirectory(modelPath)
/// try pipeline.prepareTokenizer(from: modelPath)
/// try pipeline.prepareTextEncoder(from: modelPath)
///
/// // Wrap in a session for automatic caching and memory management
/// let session = ImagePipelineSession(
///   pipeline: pipeline,
///   modelId: "Qwen/Qwen-Image",
///   configuration: .default  // Releases encoders after encoding
/// )
///
/// // Generate - embeddings are cached automatically
/// let params = GenerationParameters(prompt: "A cat", width: 1024, height: 1024, steps: 30)
/// let pixels = try await session.generate(parameters: params, model: modelConfig)
/// ```
///
/// ### Streaming Progress
///
/// ```swift
/// for try await event in session.generateStream(parameters: params, model: modelConfig) {
///   switch event {
///   case .progress(let info):
///     print("Step \\(info.step)/\\(info.total)")
///   case .output(let pixels):
///     print("Done: \\(pixels.shape)")
///   }
/// }
/// ```
///
/// ### Layered Image Generation
///
/// ```swift
/// import QwenImage
/// import QwenImageRuntime
///
/// // Load the core pipeline
/// let pipeline = try await QwenLayeredPipeline.load(from: modelPath)
///
/// // Wrap in a session
/// let session = LayeredPipelineSession(
///   pipeline: pipeline,
///   modelId: "Qwen/Qwen-Image-Layered"
/// )
///
/// // Load image data for cache key computation (no MLX.eval needed!)
/// let imageData = try Data(contentsOf: imageURL)
/// let imageArray = ... // Convert to MLXArray
///
/// // Generate - captions are cached per-image
/// let params = LayeredGenerationParameters(layers: 4, resolution: 640)
/// let layers = try await session.generate(
///   imageData: imageData,
///   image: imageArray,
///   parameters: params
/// )
/// ```
///
/// ```swift
/// for try await event in session.generateStream(imageData: imageData, image: imageArray, parameters: params) {
///   switch event {
///   case .progress(let info):
///     print("Step \\(info.step)/\\(info.total)")
///   case .output(let layers):
///     print("Layers: \\(layers.count)")
///   }
/// }
/// ```
///
/// ### GPU Memory Management
///
/// ```swift
/// import QwenImageRuntime
///
/// // Apply a preset based on system memory
/// GPUCachePolicy.applyPreset(.recommendedPreset())
///
/// // Or set a specific limit
/// GPUCachePolicy.setCacheLimit(8 * 1024 * 1024 * 1024)  // 8GB
///
/// // Clear cache when needed
/// GPUCachePolicy.clearCache()
/// ```

// Re-export commonly used types from QwenImage
@_exported import QwenImage

// Export all public types from this module
// (The individual files export their own public types)
