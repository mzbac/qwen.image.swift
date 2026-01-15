import Foundation
import Hub

public struct HubSnapshotOptions {
  public var repoId: String
  public var revision: String
  public var repoType: Hub.RepoType
  public var patterns: [String]
  public var cacheDirectory: URL?
  public var hfToken: String?
  public var offline: Bool
  public var useBackgroundSession: Bool

  public init(
    repoId: String,
    revision: String = "main",
    repoType: Hub.RepoType = .models,
    patterns: [String] = [],
    cacheDirectory: URL? = nil,
    hfToken: String? = nil,
    offline: Bool = false,
    useBackgroundSession: Bool = false
  ) {
    self.repoId = repoId
    self.revision = revision
    self.repoType = repoType
    self.patterns = patterns
    self.cacheDirectory = cacheDirectory
    self.hfToken = hfToken
    self.offline = offline
    self.useBackgroundSession = useBackgroundSession
  }
}

public struct HubSnapshotProgress: Sendable {
  public let fractionCompleted: Double
  public let completedUnitCount: Int64
  public let totalUnitCount: Int64
  public let estimatedSpeedBytesPerSecond: Double?

  init(progress: Progress, speed: Double?) {
    self.fractionCompleted = progress.totalUnitCount > 0
      ? Double(progress.completedUnitCount) / Double(progress.totalUnitCount)
      : 0
    self.completedUnitCount = progress.completedUnitCount
    self.totalUnitCount = progress.totalUnitCount
    self.estimatedSpeedBytesPerSecond = speed
  }
}

public actor HubSnapshot {
  public typealias ProgressHandler = @Sendable (HubSnapshotProgress) -> Void

  private let options: HubSnapshotOptions
  private let hubApi: HubApi
  private let cacheDirectory: URL
  private var cachedSnapshotURL: URL?

  public init(
    options: HubSnapshotOptions,
    hubApi: HubApi? = nil
  ) throws {
    self.options = options

    let cacheDirectory = try HubSnapshot.resolveCacheDirectory(
      requested: options.cacheDirectory,
      fileManager: FileManager.default
    )
    self.cacheDirectory = cacheDirectory

    let api = hubApi ?? HubApi(
      downloadBase: cacheDirectory,
      hfToken: options.hfToken,
      useBackgroundSession: options.useBackgroundSession,
      useOfflineMode: options.offline ? true : nil
    )

    self.hubApi = api
  }

  public func prepare(progressHandler: ProgressHandler? = nil) async throws -> URL {
    if let cachedSnapshotURL,
      FileManager.default.fileExists(atPath: cachedSnapshotURL.path) {
      return cachedSnapshotURL
    }

    if let snapshotURL = Self.resolveHuggingFaceCliSnapshotURL(
      cacheDirectory: cacheDirectory,
      repoId: options.repoId,
      repoType: options.repoType,
      revision: options.revision,
      fileManager: FileManager.default
    ) {
      cachedSnapshotURL = snapshotURL
      return snapshotURL
    }

    let repo = Hub.Repo(id: options.repoId, type: options.repoType)
    let patterns = options.patterns
    let snapshotURL = try await hubApi.snapshot(
      from: repo,
      revision: options.revision,
      matching: patterns,
      progressHandler: { progress, speed in
        progressHandler?(HubSnapshotProgress(progress: progress, speed: speed))
      }
    )
    cachedSnapshotURL = snapshotURL
    return snapshotURL
  }

  public func fileURL(
    for relativePath: String,
    progressHandler: ProgressHandler? = nil
  ) async throws -> URL {
    let snapshot = try await prepare(progressHandler: progressHandler)
    let url = snapshot.appending(path: relativePath)
    guard FileManager.default.fileExists(atPath: url.path) else {
      throw Hub.HubClientError.fileNotFound(relativePath)
    }
    return url
  }

  public func invalidateCache() {
    cachedSnapshotURL = nil
  }

  private static func resolveHuggingFaceCliSnapshotURL(
    cacheDirectory: URL,
    repoId: String,
    repoType: Hub.RepoType,
    revision: String,
    fileManager: FileManager
  ) -> URL? {
    // huggingface-cli (huggingface_hub) cache layout:
    //   <HF_HUB_CACHE>/
    //     models--<org>--<repo>/
    //       refs/<revision>          (contains commit hash)
    //       snapshots/<commitHash>/  (materialized snapshot)
    //
    // This differs from swift-transformers' HubApi layout:
    //   <downloadBase>/<repoType>/<repoId>/
    let repoDirectoryName = "\(repoType.rawValue)--\(repoId.replacingOccurrences(of: "/", with: "--"))"
    let repoDirectory = cacheDirectory.appending(path: repoDirectoryName)

    var isRepoDirectory: ObjCBool = false
    guard fileManager.fileExists(atPath: repoDirectory.path, isDirectory: &isRepoDirectory),
          isRepoDirectory.boolValue
    else {
      return nil
    }

    let commitHash: String?
    if isCommitHash(revision) {
      commitHash = revision.lowercased()
    } else {
      commitHash = resolveCommitHash(
        revision: revision,
        repoDirectory: repoDirectory,
        fileManager: fileManager
      )
    }

    guard let commitHash else { return nil }

    let snapshotDirectory = repoDirectory
      .appending(path: "snapshots")
      .appending(path: commitHash)

    var isSnapshotDirectory: ObjCBool = false
    guard fileManager.fileExists(atPath: snapshotDirectory.path, isDirectory: &isSnapshotDirectory),
          isSnapshotDirectory.boolValue
    else {
      return nil
    }

    return snapshotDirectory
  }

  private static func resolveCommitHash(
    revision: String,
    repoDirectory: URL,
    fileManager: FileManager
  ) -> String? {
    let refsDirectory = repoDirectory.appending(path: "refs")

    var candidates = [revision]
    if revision.hasPrefix("refs/") {
      let trimmed = String(revision.dropFirst("refs/".count))
      if !trimmed.isEmpty {
        candidates.append(trimmed)
      }
    }

    for candidate in candidates {
      let refFile = refsDirectory.appending(path: candidate)
      guard fileManager.fileExists(atPath: refFile.path) else { continue }
      guard let contents = try? String(contentsOf: refFile, encoding: .utf8) else { continue }
      let hash = contents.trimmingCharacters(in: .whitespacesAndNewlines)
      if isCommitHash(hash) {
        return hash.lowercased()
      }
    }

    return nil
  }

  private static func isCommitHash(_ value: String) -> Bool {
    let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
    guard trimmed.count == 40 else { return false }
    let hexSet = CharacterSet(charactersIn: "0123456789abcdefABCDEF")
    return trimmed.unicodeScalars.allSatisfy { hexSet.contains($0) }
  }

  private static func resolveCacheDirectory(
    requested: URL?,
    fileManager: FileManager
  ) throws -> URL {
    if let explicit = requested {
      try fileManager.createDirectory(at: explicit, withIntermediateDirectories: true, attributes: nil)
      return explicit
    }

    // Follow HuggingFace cache convention:
    // 1. HF_HUB_CACHE env var
    // 2. HF_HOME env var + "/hub"
    // 3. ~/.cache/huggingface/hub (default)
    let env = ProcessInfo.processInfo.environment

    if let hfHubCache = env["HF_HUB_CACHE"], !hfHubCache.isEmpty {
      let directory = URL(fileURLWithPath: hfHubCache)
      try fileManager.createDirectory(at: directory, withIntermediateDirectories: true, attributes: nil)
      return directory
    }

    if let hfHome = env["HF_HOME"], !hfHome.isEmpty {
      let directory = URL(fileURLWithPath: hfHome).appending(path: "hub")
      try fileManager.createDirectory(at: directory, withIntermediateDirectories: true, attributes: nil)
      return directory
    }

    // Default: ~/.cache/huggingface/hub (standard HuggingFace location)
    let home = fileManager.homeDirectoryForCurrentUser
    let directory = home.appending(path: ".cache/huggingface/hub")
    try fileManager.createDirectory(at: directory, withIntermediateDirectories: true, attributes: nil)
    return directory
  }
}
