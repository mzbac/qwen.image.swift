import Foundation
import XCTest
@testable import QwenImage

final class ModelConfigLoadingTests: XCTestCase {
  func testTransformerConfigurationLoadsFromConfigJSON() throws {
    let fileManager = FileManager.default
    let root = fileManager.temporaryDirectory.appending(path: UUID().uuidString)
    let transformerDir = root.appending(path: "transformer")
    try fileManager.createDirectory(at: transformerDir, withIntermediateDirectories: true)
    defer { try? fileManager.removeItem(at: root) }

    let configURL = transformerDir.appending(path: "config.json")
    let payload: [String: Any] = [
      "in_channels": 64,
      "out_channels": 16,
      "num_layers": 2,
      "attention_head_dim": 32,
      "num_attention_heads": 4,
      "joint_attention_dim": 128,
      "patch_size": 2
    ]
    let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: configURL)

    let configuration = try QwenTransformerConfiguration.load(from: transformerDir)
    XCTAssertEqual(configuration.inChannels, 64)
    XCTAssertEqual(configuration.outChannels, 16)
    XCTAssertEqual(configuration.numberOfLayers, 2)
    XCTAssertEqual(configuration.attentionHeadDim, 32)
    XCTAssertEqual(configuration.numberOfHeads, 4)
    XCTAssertEqual(configuration.jointAttentionDim, 128)
    XCTAssertEqual(configuration.patchSize, 2)

    let model = QwenTransformer(configuration: configuration)
    XCTAssertEqual(model.transformerBlocks.count, 2)
  }

  func testTextEncoderConfigurationParsesTorchDType() throws {
    let fileManager = FileManager.default
    let root = fileManager.temporaryDirectory.appending(path: UUID().uuidString)
    let textEncoderDir = root.appending(path: "text_encoder")
    try fileManager.createDirectory(at: textEncoderDir, withIntermediateDirectories: true)
    defer { try? fileManager.removeItem(at: root) }

    let configURL = textEncoderDir.appending(path: "config.json")
    let payload: [String: Any] = [
      "torch_dtype": "float16",
      "vocab_size": 1_000,
      "hidden_size": 64,
      "num_hidden_layers": 3,
      "num_attention_heads": 8,
      "num_key_value_heads": 2,
      "intermediate_size": 128,
      "prompt_drop_index": 1
    ]
    let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: configURL)

    let configuration = try QwenTextEncoderConfiguration.load(from: textEncoderDir)
    XCTAssertEqual(configuration.outputDType, .float16)
    XCTAssertEqual(configuration.vocabSize, 1_000)
    XCTAssertEqual(configuration.hiddenSize, 64)
    XCTAssertEqual(configuration.numHiddenLayers, 3)
    XCTAssertEqual(configuration.numAttentionHeads, 8)
    XCTAssertEqual(configuration.numKeyValueHeads, 2)
    XCTAssertEqual(configuration.intermediateSize, 128)
    XCTAssertEqual(configuration.promptDropIndex, 1)

    let encoder = QwenTextEncoder(configuration: configuration)
    XCTAssertEqual(encoder.encoder.layers.count, 3)
  }

  func testTextEncoderConfigurationRejectsUnknownTorchDType() throws {
    let fileManager = FileManager.default
    let root = fileManager.temporaryDirectory.appending(path: UUID().uuidString)
    let textEncoderDir = root.appending(path: "text_encoder")
    try fileManager.createDirectory(at: textEncoderDir, withIntermediateDirectories: true)
    defer { try? fileManager.removeItem(at: root) }

    let configURL = textEncoderDir.appending(path: "config.json")
    let payload: [String: Any] = [
      "torch_dtype": "float8"
    ]
    let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: configURL)

    XCTAssertThrowsError(try QwenTextEncoderConfiguration.load(from: textEncoderDir)) { error in
      guard case QwenConfigLoadingError.invalidValue = error else {
        XCTFail("Expected invalidValue error, got \(error)")
        return
      }
    }
  }

  func testModelDescriptorLoadsRequiredConfigs() throws {
    let fileManager = FileManager.default
    let root = fileManager.temporaryDirectory.appending(path: UUID().uuidString)
    let transformerDir = root.appending(path: "transformer")
    let textEncoderDir = root.appending(path: "text_encoder")
    let schedulerDir = root.appending(path: "scheduler")
    try fileManager.createDirectory(at: transformerDir, withIntermediateDirectories: true)
    try fileManager.createDirectory(at: textEncoderDir, withIntermediateDirectories: true)
    try fileManager.createDirectory(at: schedulerDir, withIntermediateDirectories: true)
    defer { try? fileManager.removeItem(at: root) }

    let transformerConfigURL = transformerDir.appending(path: "config.json")
    let transformerPayload: [String: Any] = [
      "num_layers": 1
    ]
    let transformerData = try JSONSerialization.data(withJSONObject: transformerPayload, options: [.prettyPrinted, .sortedKeys])
    try transformerData.write(to: transformerConfigURL)

    let textEncoderConfigURL = textEncoderDir.appending(path: "config.json")
    let textEncoderPayload: [String: Any] = [
      "torch_dtype": "bfloat16",
      "vision_config": [
        "tokens_per_second": 3
      ]
    ]
    let textEncoderData = try JSONSerialization.data(withJSONObject: textEncoderPayload, options: [.prettyPrinted, .sortedKeys])
    try textEncoderData.write(to: textEncoderConfigURL)

    let schedulerConfigURL = schedulerDir.appending(path: "scheduler_config.json")
    try Data("{}".utf8).write(to: schedulerConfigURL)

    let descriptor = try QwenModelDescriptor.load(from: root)
    XCTAssertEqual(descriptor.transformerConfiguration.numberOfLayers, 1)
    XCTAssertEqual(descriptor.textEncoderConfiguration.outputDType, .bfloat16)
    XCTAssertEqual(descriptor.visionTokensPerSecond, 3)
    XCTAssertFalse(descriptor.identity.isEmpty)
  }

  func testFlowMatchConfigHardFailsWhenMissing() throws {
    let fileManager = FileManager.default
    let root = fileManager.temporaryDirectory.appending(path: UUID().uuidString)
    let schedulerDir = root.appending(path: "scheduler")
    try fileManager.createDirectory(at: schedulerDir, withIntermediateDirectories: true)
    defer { try? fileManager.removeItem(at: root) }

    XCTAssertThrowsError(try QwenFlowMatchConfig.load(fromSchedulerDirectory: schedulerDir)) { error in
      guard case QwenConfigLoadingError.missingFile = error else {
        XCTFail("Expected missingFile error, got \(error)")
        return
      }
    }
  }

  func testSnapshotOptionsAllowsArbitraryRepoId() throws {
    XCTAssertNoThrow(try QwenModelRepository.snapshotOptions(repoId: "SomeOrg/SomeModel"))
  }

  func testModelDescriptorHardFailsWhenTransformerConfigMissing() throws {
    let fileManager = FileManager.default
    let root = fileManager.temporaryDirectory.appending(path: UUID().uuidString)
    let transformerDir = root.appending(path: "transformer")
    let textEncoderDir = root.appending(path: "text_encoder")
    let schedulerDir = root.appending(path: "scheduler")
    try fileManager.createDirectory(at: transformerDir, withIntermediateDirectories: true)
    try fileManager.createDirectory(at: textEncoderDir, withIntermediateDirectories: true)
    try fileManager.createDirectory(at: schedulerDir, withIntermediateDirectories: true)
    defer { try? fileManager.removeItem(at: root) }

    let textEncoderConfigURL = textEncoderDir.appending(path: "config.json")
    try Data("{}".utf8).write(to: textEncoderConfigURL)
    let schedulerConfigURL = schedulerDir.appending(path: "scheduler_config.json")
    try Data("{}".utf8).write(to: schedulerConfigURL)

    XCTAssertThrowsError(try QwenModelDescriptor.load(from: root)) { error in
      guard case QwenConfigLoadingError.missingFile(let url) = error else {
        XCTFail("Expected missingFile error, got \(error)")
        return
      }
      XCTAssertEqual(url.lastPathComponent, "config.json")
      XCTAssertTrue(url.path.contains("/transformer/"))
    }
  }

  func testModelDescriptorHardFailsWhenTextEncoderConfigMissing() throws {
    let fileManager = FileManager.default
    let root = fileManager.temporaryDirectory.appending(path: UUID().uuidString)
    let transformerDir = root.appending(path: "transformer")
    let textEncoderDir = root.appending(path: "text_encoder")
    let schedulerDir = root.appending(path: "scheduler")
    try fileManager.createDirectory(at: transformerDir, withIntermediateDirectories: true)
    try fileManager.createDirectory(at: textEncoderDir, withIntermediateDirectories: true)
    try fileManager.createDirectory(at: schedulerDir, withIntermediateDirectories: true)
    defer { try? fileManager.removeItem(at: root) }

    let transformerConfigURL = transformerDir.appending(path: "config.json")
    try Data("{}".utf8).write(to: transformerConfigURL)
    let schedulerConfigURL = schedulerDir.appending(path: "scheduler_config.json")
    try Data("{}".utf8).write(to: schedulerConfigURL)

    XCTAssertThrowsError(try QwenModelDescriptor.load(from: root)) { error in
      guard case QwenConfigLoadingError.missingFile(let url) = error else {
        XCTFail("Expected missingFile error, got \(error)")
        return
      }
      XCTAssertEqual(url.lastPathComponent, "config.json")
      XCTAssertTrue(url.path.contains("/text_encoder/"))
    }
  }
}
