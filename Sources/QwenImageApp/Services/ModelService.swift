import Foundation
import QwenImage
import QwenImageRuntime
import MLX

private enum LoadedPipeline {
  case layered
  case imageEditing
}

actor ModelService {
  static let shared = ModelService()

  // MARK: - Session Storage

  private var layeredSession: LayeredPipelineSession?
  private var imageSession: ImagePipelineSession?

  private var cachedPaths: [String: URL] = [:]
  private var loadedMode: LoadedPipeline?
  private var layeredSessionPath: URL?
  private var imageSessionPath: URL?
  private var imageSessionConfig: QwenImageConfig?

  private init() {
  }

  // MARK: - Path Resolution

  func cachedModelPath(repoId: String) -> URL? {
    if let cached = cachedPaths[repoId] {
      return cached
    }

    let env = ProcessInfo.processInfo.environment
    let hubPath: URL
    if let hfHubCache = env["HF_HUB_CACHE"], !hfHubCache.isEmpty {
      hubPath = URL(fileURLWithPath: hfHubCache)
    } else if let hfHome = env["HF_HOME"], !hfHome.isEmpty {
      hubPath = URL(fileURLWithPath: hfHome).appending(path: "hub")
    } else {
      hubPath = URL(fileURLWithPath: NSHomeDirectory()).appending(path: ".cache/huggingface/hub")
    }

    let modelDir = hubPath.appending(path: "models").appending(path: repoId)

    guard FileManager.default.fileExists(atPath: modelDir.path) else {
      return nil
    }

    do {
      let contents = try FileManager.default.contentsOfDirectory(atPath: modelDir.path)
      if contents.isEmpty {
        return nil
      }
      cachedPaths[repoId] = modelDir
      return modelDir
    } catch {
      return nil
    }
  }

  // MARK: - Download

  func downloadModel(
    repoId: String,
    revision: String = "main",
    progressHandler: @escaping @Sendable (HubSnapshotProgress) -> Void
  ) async throws -> URL {
    let options = try QwenModelRepository.snapshotOptions(
      repoId: repoId,
      revision: revision
    )

    let snapshot = try HubSnapshot(options: options)
    let path = try await snapshot.prepare(progressHandler: progressHandler)

    cachedPaths[repoId] = path
    return path
  }

  /// Download the Lightning LoRA from HuggingFace
  func downloadLightningLoRA(
    progressHandler: @escaping @Sendable (HubSnapshotProgress) -> Void
  ) async throws -> URL {
    let repoId = "lightx2v/Qwen-Image-Edit-2511-Lightning"
    let options = HubSnapshotOptions(
      repoId: repoId,
      revision: "main",
      patterns: ["Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"]
    )

    let snapshot = try HubSnapshot(options: options)
    let path = try await snapshot.prepare(progressHandler: progressHandler)

    cachedPaths[repoId] = path
    return path
  }

  /// Download the Text-to-Image Lightning LoRA (Qwen-Image-2512) from HuggingFace.
  func downloadTextToImageLightningLoRA(
    progressHandler: @escaping @Sendable (HubSnapshotProgress) -> Void
  ) async throws -> URL {
    let repoId = "lightx2v/Qwen-Image-2512-Lightning"
    let options = HubSnapshotOptions(
      repoId: repoId,
      revision: "main",
      patterns: ["Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors"]
    )

    let snapshot = try HubSnapshot(options: options)
    let path = try await snapshot.prepare(progressHandler: progressHandler)

    cachedPaths[repoId] = path
    return path
  }

  /// Get the HuggingFace hub path respecting HF_HUB_CACHE and HF_HOME environment variables
  private func hubPath() -> URL {
    let env = ProcessInfo.processInfo.environment
    if let hfHubCache = env["HF_HUB_CACHE"], !hfHubCache.isEmpty {
      return URL(fileURLWithPath: hfHubCache)
    } else if let hfHome = env["HF_HOME"], !hfHome.isEmpty {
      return URL(fileURLWithPath: hfHome).appending(path: "hub")
    } else {
      return URL(fileURLWithPath: NSHomeDirectory()).appending(path: ".cache/huggingface/hub")
    }
  }

  /// Check if Lightning LoRA is installed
  func isLightningLoRAInstalled() -> Bool {
    let path = hubPath()
      .appending(path: "models/lightx2v/Qwen-Image-Edit-2511-Lightning")
      .appending(path: "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
    return FileManager.default.fileExists(atPath: path.path)
  }

  /// Get the Lightning LoRA path if it exists
  func lightningLoRAPath() -> URL? {
    let path = hubPath()
      .appending(path: "models/lightx2v/Qwen-Image-Edit-2511-Lightning")
      .appending(path: "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
    return FileManager.default.fileExists(atPath: path.path) ? path : nil
  }

  /// Check if Text-to-Image Lightning LoRA is installed.
  func isTextToImageLightningLoRAInstalled() -> Bool {
    let path = hubPath()
      .appending(path: "models/lightx2v/Qwen-Image-2512-Lightning")
      .appending(path: "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors")
    return FileManager.default.fileExists(atPath: path.path)
  }

  /// Get the Text-to-Image Lightning LoRA path if it exists.
  func textToImageLightningLoRAPath() -> URL? {
    let path = hubPath()
      .appending(path: "models/lightx2v/Qwen-Image-2512-Lightning")
      .appending(path: "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors")
    return FileManager.default.fileExists(atPath: path.path) ? path : nil
  }

  // MARK: - Session-Based Loading

  func loadLayeredSession(
    from path: URL,
    modelId: String = "Qwen/Qwen-Image-Layered",
    revision: String = "main",
    configuration: LayeredPipelineSessionConfiguration = .default
  ) async throws -> LayeredPipelineSession {
    if loadedMode == .imageEditing {
      unloadImagePipeline()
    }

    if let session = layeredSession {
      if layeredSessionPath?.standardizedFileURL == path.standardizedFileURL {
        loadedMode = .layered
        return session
      }
      unloadLayeredPipeline()
    }

    let pipeline = try await QwenLayeredPipeline.load(from: path)

    let session = LayeredPipelineSession(
      pipeline: pipeline,
      modelId: modelId,
      revision: revision,
      configuration: configuration
    )
    layeredSession = session
    layeredSessionPath = path
    loadedMode = .layered

    return session
  }

  func loadImageSession(
    from path: URL,
    config: QwenImageConfig,
    modelId: String = "Qwen/Qwen-Image",
    revision: String = "main",
    configuration: ImagePipelineSessionConfiguration = .default
  ) async throws -> ImagePipelineSession {
    if loadedMode == .layered {
      unloadLayeredPipeline()
    }

    if let session = imageSession {
      let samePath = imageSessionPath?.standardizedFileURL == path.standardizedFileURL
      let existingConfig = imageSessionConfig
      let canReuse = samePath && (existingConfig == .imageEditing || existingConfig == config)
      if canReuse {
        loadedMode = .imageEditing
        return session
      }
      unloadImagePipeline()
    }

    let pipeline = QwenImagePipeline(config: config)
    pipeline.setBaseDirectory(path)
    try pipeline.prepareTokenizer(from: path, maxLength: nil)
    try pipeline.prepareVAE(from: path)

    let session = ImagePipelineSession(
      pipeline: pipeline,
      modelId: modelId,
      revision: revision,
      configuration: configuration
    )
    imageSession = session
    imageSessionPath = path
    imageSessionConfig = config
    loadedMode = .imageEditing

    return session
  }

  // MARK: - Unloading

  func unloadLayeredPipeline() {
    layeredSession = nil
    layeredSessionPath = nil
    if loadedMode == .layered {
      loadedMode = nil
    }
    GPUCachePolicy.clearCache()
  }

  func unloadImagePipeline() {
    imageSession = nil
    imageSessionPath = nil
    imageSessionConfig = nil
    if loadedMode == .imageEditing {
      loadedMode = nil
    }
    GPUCachePolicy.clearCache()
  }

  func unloadAll() {
    layeredSession = nil
    imageSession = nil
    layeredSessionPath = nil
    imageSessionPath = nil
    imageSessionConfig = nil
    loadedMode = nil
    GPUCachePolicy.clearCache()
  }

  // MARK: - LoRA

  func applyLoRAToLayeredSession(from url: URL, scale: Float) async throws {
    guard let session = layeredSession else {
      throw ModelServiceError.noSessionLoaded
    }
    try await session.applyLora(from: url, scale: scale)
  }

  func applyLoRAToImageSession(from url: URL, scale: Float) async throws {
    guard let session = imageSession else {
      throw ModelServiceError.noSessionLoaded
    }
    try await session.applyLora(from: url, scale: scale)
  }

  // MARK: - Cache Management

  func clearAllCaches() async {
    if let session = layeredSession {
      await session.clearCache()
    }
    if let session = imageSession {
      await session.clearCache()
    }
  }

  // MARK: - Status

  var isLayeredLoaded: Bool {
    layeredSession != nil
  }

  var isImageLoaded: Bool {
    imageSession != nil
  }

  var hasLayeredSession: Bool {
    layeredSession != nil
  }

  var hasImageSession: Bool {
    imageSession != nil
  }
}

// MARK: - Errors

enum ModelServiceError: Error {
  case noSessionLoaded
}

// MARK: - HubSnapshotProgress Extensions

extension HubSnapshotProgress {
  var formattedSpeed: String? {
    guard let speed = estimatedSpeedBytesPerSecond else { return nil }
    return ByteCountFormatter.string(fromByteCount: Int64(speed), countStyle: .file) + "/s"
  }

  var formattedCompleted: String {
    ByteCountFormatter.string(fromByteCount: completedUnitCount, countStyle: .file)
  }

  var formattedTotal: String {
    totalUnitCount > 0
      ? ByteCountFormatter.string(fromByteCount: totalUnitCount, countStyle: .file)
      : "Unknown"
  }

  var formattedTimeRemaining: String? {
    guard let speed = estimatedSpeedBytesPerSecond, speed > 0 else { return nil }
    let remainingBytes = totalUnitCount - completedUnitCount
    let secondsRemaining = Double(remainingBytes) / Double(speed)

    if secondsRemaining < 60 {
      return "~\(Int(secondsRemaining))s remaining"
    } else if secondsRemaining < 3600 {
      return "~\(Int(secondsRemaining / 60))m remaining"
    } else {
      let hours = Int(secondsRemaining / 3600)
      let minutes = Int((secondsRemaining.truncatingRemainder(dividingBy: 3600)) / 60)
      return "~\(hours)h \(minutes)m remaining"
    }
  }
}
