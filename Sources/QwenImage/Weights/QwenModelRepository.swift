import Foundation
import Hub

/// Convenience utilities for resolving Qwen model snapshots from Hugging Face Hub.
///
/// Mirrors the pattern used by `flux.swift`, where a single repo identifier is enough
/// to materialize a local snapshot containing all components needed by the pipeline.
public enum QwenModelRepository {
  /// Default upstream repo identifier for the base Qwen-Image weights.
  public static let defaultRepoId = "Qwen/Qwen-Image"

  /// Default file patterns required for the inference-only pipeline.
  ///
  /// The glob list intentionally captures entire directories to keep the logic simple
  /// and tolerant to upstream layout tweaks (e.g. additional JSON configs per component).
  public static let defaultPatterns: [String] = [
    "model_index.json",
    "config.json",
    "preprocessor_config.json",
    "quantization.json",
    "README.md",
    "tokenizer/*",
    "processor/*",
    "text_encoder/*",
    "text_encoder_2/*",
    "transformer/*",
    "vae/*",
    "scheduler/*"
  ]

  /// Construct snapshot options suitable for use with ``HubSnapshot``.
  public static func snapshotOptions(
    repoId: String = defaultRepoId,
    revision: String = "main",
    cacheDirectory: URL? = nil,
    hfToken: String? = nil,
    offline: Bool = false,
    useBackgroundSession: Bool = false,
    additionalPatterns: [String] = []
  ) throws -> HubSnapshotOptions {
    var patterns = defaultPatterns
    if !additionalPatterns.isEmpty {
      patterns.append(contentsOf: additionalPatterns)
    }
    return HubSnapshotOptions(
      repoId: repoId,
      revision: revision,
      repoType: Hub.RepoType.models,
      patterns: patterns,
      cacheDirectory: cacheDirectory,
      hfToken: hfToken,
      offline: offline,
      useBackgroundSession: useBackgroundSession
    )
  }
}
