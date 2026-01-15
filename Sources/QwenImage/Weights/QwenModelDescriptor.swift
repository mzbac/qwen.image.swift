import Foundation
import MLX

public struct QwenModelDescriptor {
  public let transformerConfiguration: QwenTransformerConfiguration
  public let textEncoderConfiguration: QwenTextEncoderConfiguration
  public let flowMatchConfiguration: QwenFlowMatchConfig
  public let visionTokensPerSecond: Int?

  public init(
    transformerConfiguration: QwenTransformerConfiguration,
    textEncoderConfiguration: QwenTextEncoderConfiguration,
    flowMatchConfiguration: QwenFlowMatchConfig,
    visionTokensPerSecond: Int?
  ) {
    self.transformerConfiguration = transformerConfiguration
    self.textEncoderConfiguration = textEncoderConfiguration
    self.flowMatchConfiguration = flowMatchConfiguration
    self.visionTokensPerSecond = visionTokensPerSecond
  }

  public var identity: String {
    let tx = transformerConfiguration
    let te = textEncoderConfiguration
    let parts: [String] = [
      "tx_l=\(tx.numberOfLayers)",
      "tx_h=\(tx.numberOfHeads)",
      "tx_d=\(tx.attentionHeadDim)",
      "tx_j=\(tx.jointAttentionDim)",
      "tx_ps=\(tx.patchSize)",
      "te_l=\(te.numHiddenLayers)",
      "te_h=\(te.numAttentionHeads)",
      "te_kv=\(te.numKeyValueHeads)",
      "te_d=\(te.hiddenSize)",
      "te_vocab=\(te.vocabSize)",
      "te_dtype=\(te.outputDType.stableName)"
    ]
    return parts.joined(separator: ";")
  }

  public static func load(from root: URL) throws -> QwenModelDescriptor {
    let transformerDirectory = root.appending(path: "transformer")
    let textEncoderDirectory = root.appending(path: "text_encoder")
    let schedulerDirectory = root.appending(path: "scheduler")

    let transformerConfiguration = try QwenTransformerConfiguration.load(from: transformerDirectory)
    let textEncoderConfiguration = try QwenTextEncoderConfiguration.load(from: textEncoderDirectory)
    let flowMatchConfiguration = try QwenFlowMatchConfig.load(fromSchedulerDirectory: schedulerDirectory)
    let visionTokensPerSecond = try loadVisionTokensPerSecond(from: textEncoderDirectory)

    return QwenModelDescriptor(
      transformerConfiguration: transformerConfiguration,
      textEncoderConfiguration: textEncoderConfiguration,
      flowMatchConfiguration: flowMatchConfiguration,
      visionTokensPerSecond: visionTokensPerSecond
    )
  }

  private struct VisionConfigJSON: Decodable {
    let tokensPerSecond: Int?
  }

  private struct TextEncoderMetadataJSON: Decodable {
    let visionConfig: VisionConfigJSON?
  }

  private static func loadVisionTokensPerSecond(from textEncoderDirectory: URL) throws -> Int? {
    let configURL = textEncoderDirectory.appending(path: "config.json")
    guard FileManager.default.fileExists(atPath: configURL.path) else {
      throw QwenConfigLoadingError.missingFile(configURL)
    }

    let data = try Data(contentsOf: configURL)
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase

    let decoded: TextEncoderMetadataJSON
    do {
      decoded = try decoder.decode(TextEncoderMetadataJSON.self, from: data)
    } catch {
      return nil
    }

    return decoded.visionConfig?.tokensPerSecond
  }
}

