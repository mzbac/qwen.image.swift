import Foundation
import MLX

extension QwenImagePipeline {
  public func prepareTokenizer(
    using snapshotOptions: HubSnapshotOptions,
    progress: QwenProgressHandler? = nil
  ) async throws {
    let snapshot = try HubSnapshot(options: snapshotOptions)
    try await loadTokenizer(from: snapshot, progress: progress)
  }

  public func prepareTokenizer(from directory: URL, maxLength: Int? = nil) throws {
    let tokenizer = try QwenTokenizer.load(from: directory, maxLengthOverride: maxLength)
    self.tokenizer = tokenizer
  }

  public func prepareTextEncoder(
    using snapshotOptions: HubSnapshotOptions,
    progress: QwenProgressHandler? = nil
  ) async throws {
    let snapshot = try HubSnapshot(options: snapshotOptions)
    try await loadTextEncoder(from: snapshot, progress: progress)
  }

  public func prepareTextEncoder(from directory: URL) throws {
    let descriptor = try ensureModelDescriptor(from: directory)
    var configuration = descriptor.textEncoderConfiguration
    let dtype = dtypeOverride ?? configuration.outputDType
    configuration.outputDType = dtype
    let encoder = QwenTextEncoder(configuration: configuration)
    let textEncoderDir = directory.appending(path: "text_encoder")
    let configURL = textEncoderDir.appending(path: "config.json")
    let quantPlan = combinedQuantizationPlan(
      configURL: configURL,
      snapshotRoot: directory,
      componentName: "text_encoder"
    )
    textEncoderQuantization = quantPlan
    try weightsLoader.loadTextEncoder(
      fromDirectory: directory,
      into: encoder,
      dtype: dtype,
      quantization: quantPlan
    )
    textEncoder = encoder
    textEncoderConfiguration = configuration
    applyRuntimeQuantizationIfNeeded(to: encoder, flag: &textEncoderRuntimeQuantized)
  }

  public func prepareTransformer(
    using snapshotOptions: HubSnapshotOptions,
    progress: QwenProgressHandler? = nil
  ) async throws {
    let snapshot = try HubSnapshot(options: snapshotOptions)
    let directory = try await snapshot.prepare { hubProgress in
      progress?(ProgressInfo(step: Int(hubProgress.completedUnitCount), total: Int(hubProgress.totalUnitCount)))
    }
    try prepareTransformer(from: directory)
  }

  public func prepareTransformer(from directory: URL) throws {
    let descriptor = try ensureModelDescriptor(from: directory)
    let transformer = QwenTransformer(configuration: descriptor.transformerConfiguration)
    let dtype = dtypeOverride ?? descriptor.textEncoderConfiguration.outputDType
    let configURL = directory.appending(path: "transformer").appending(path: "config.json")
    let quantPlan = combinedQuantizationPlan(
      configURL: configURL,
      snapshotRoot: directory,
      componentName: "transformer"
    )
    transformerQuantization = quantPlan
    try weightsLoader.loadTransformer(
      fromDirectory: directory,
      into: transformer,
      dtype: dtype,
      quantization: quantPlan
    )
    transformerDirectory = directory
    self.transformer = transformer
    visionTower = nil
    applyRuntimeQuantizationIfNeeded(to: transformer, flag: &transformerRuntimeQuantized)
    transformer.setAttentionQuantization(attentionQuantizationSpec)
  }

  public func prepareUNet(
    using snapshotOptions: HubSnapshotOptions,
    progress: QwenProgressHandler? = nil
  ) async throws {
    let snapshot = try HubSnapshot(options: snapshotOptions)
    let directory = try await snapshot.prepare { hubProgress in
      progress?(ProgressInfo(step: Int(hubProgress.completedUnitCount), total: Int(hubProgress.totalUnitCount)))
    }
    try prepareUNet(from: directory)
  }

  public func prepareUNet(from directory: URL) throws {
    let descriptor = try ensureModelDescriptor(from: directory)
    let unet = QwenUNet(configuration: descriptor.transformerConfiguration)
    let dtype = dtypeOverride ?? descriptor.textEncoderConfiguration.outputDType
    let configURL = directory.appending(path: "transformer").appending(path: "config.json")
    let quantPlan = combinedQuantizationPlan(
      configURL: configURL,
      snapshotRoot: directory,
      componentName: "transformer"
    )
    transformerQuantization = quantPlan
    try weightsLoader.loadUNet(
      fromDirectory: directory,
      into: unet,
      dtype: dtype,
      quantization: quantPlan
    )
    self.unet = unet
    transformerDirectory = directory
    visionTower = nil
    applyRuntimeQuantizationIfNeeded(to: unet.transformer, flag: &transformerRuntimeQuantized)
    unet.transformer.setAttentionQuantization(attentionQuantizationSpec)
  }

  public func prepareVAE(
    using snapshotOptions: HubSnapshotOptions,
    progress: QwenProgressHandler? = nil
  ) async throws {
    let snapshot = try HubSnapshot(options: snapshotOptions)
    let directory = try await snapshot.prepare { hubProgress in
      progress?(ProgressInfo(step: Int(hubProgress.completedUnitCount), total: Int(hubProgress.totalUnitCount)))
    }
    try prepareVAE(from: directory)
  }

  public func prepareAll(
    using snapshotOptions: HubSnapshotOptions,
    tokenizerMaxLength: Int? = nil,
    progress: QwenProgressHandler? = nil
  ) async throws {
    let snapshot = try HubSnapshot(options: snapshotOptions)
    let root = try await snapshot.prepare { hubProgress in
      progress?(ProgressInfo(step: Int(hubProgress.completedUnitCount), total: Int(hubProgress.totalUnitCount)))
    }
    try prepareTokenizer(from: root, maxLength: tokenizerMaxLength)
    try prepareTextEncoder(from: root)
    try prepareUNet(from: root)
    try prepareVAE(from: root)
  }

  public func prepareVAE(from directory: URL) throws {
    _ = try ensureModelDescriptor(from: directory)
    let vae = QwenVAE()
    let dtype = preferredWeightDType()
    try weightsLoader.loadVAE(fromDirectory: directory, into: vae, dtype: dtype)
    self.vae = vae
  }

  private func loadTokenizer(
    from snapshot: HubSnapshot,
    progress: QwenProgressHandler?
  ) async throws {
    let directory = try await snapshot.prepare { hubProgress in
      progress?(ProgressInfo(step: Int(hubProgress.completedUnitCount), total: Int(hubProgress.totalUnitCount)))
    }
    let tokenizer = try QwenTokenizer.load(from: directory)
    self.tokenizer = tokenizer
  }

  private func loadTextEncoder(
    from snapshot: HubSnapshot,
    progress: QwenProgressHandler?
  ) async throws {
    let directory = try await snapshot.prepare { hubProgress in
      progress?(ProgressInfo(step: Int(hubProgress.completedUnitCount), total: Int(hubProgress.totalUnitCount)))
    }
    try prepareTextEncoder(from: directory)
  }
}
