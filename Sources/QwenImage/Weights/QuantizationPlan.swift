import Foundation
import MLX

// swiftlint:disable nesting

public struct QwenQuantizationSpec: Equatable, Decodable {
  public let groupSize: Int
  public let bits: Int
  public let mode: QuantizationMode

  public init(groupSize: Int, bits: Int, mode: QuantizationMode = .affine) {
    self.groupSize = groupSize
    self.bits = bits
    self.mode = mode
  }

  public init(from decoder: any Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let groupSize = try container.decode(Int.self, forKey: .groupSize)
    let bits = try container.decode(Int.self, forKey: .bits)
    let modeString = try container.decodeIfPresent(String.self, forKey: .mode)
    let mode = QuantizationMode.fromString(modeString)
    self.init(groupSize: groupSize, bits: bits, mode: mode)
  }

  enum CodingKeys: String, CodingKey {
    case groupSize = "group_size"
    case bits
    case mode
  }

  var tuple: (Int, Int, QuantizationMode) { (groupSize, bits, mode) }
}

public enum QwenQuantizationSelection {
  case skip
  case quantize(QwenQuantizationSpec)
}

public struct QwenPrepackedLayer {
  public let spec: QwenQuantizationSpec
  public let quantizedFile: String?
}

public struct QwenQuantizationPlan {
  public var defaultSpec: QwenQuantizationSpec?
  public var perLayer: [String: QwenQuantizationSelection]
  public var prepackedLayers: [String: QwenPrepackedLayer]

  public init(
    defaultSpec: QwenQuantizationSpec? = nil,
    perLayer: [String: QwenQuantizationSelection] = [:],
    prepackedLayers: [String: QwenPrepackedLayer] = [:]
  ) {
    self.defaultSpec = defaultSpec
    self.perLayer = perLayer
    self.prepackedLayers = prepackedLayers
  }

  public var isEnabled: Bool {
    defaultSpec != nil || !perLayer.isEmpty
  }

  public func quantization(for tensorName: String) -> QwenQuantizationSpec? {
    if let selection = perLayer[tensorName] {
      switch selection {
      case .skip:
        return nil
      case .quantize(let spec):
        return spec
      }
    }
    return defaultSpec
  }

  public func prepackedInfo(for tensorName: String) -> QwenPrepackedLayer? {
    prepackedLayers[tensorName]
  }

  public func merging(_ other: QwenQuantizationPlan?) -> QwenQuantizationPlan {
    guard let other else { return self }
    var merged = QwenQuantizationPlan(
      defaultSpec: other.defaultSpec ?? self.defaultSpec,
      perLayer: self.perLayer,
      prepackedLayers: self.prepackedLayers
    )
    for (key, value) in other.perLayer {
      merged.perLayer[key] = value
    }
    for (key, value) in other.prepackedLayers {
      merged.prepackedLayers[key] = value
    }
    return merged
  }

  public mutating func registerPrepackedLayers(
    from manifest: QwenQuantizedSnapshotManifest,
    component: String? = nil
  ) {
    for layer in manifest.layers {
      if let component, let layerComponent = layer.component, layerComponent != component {
        continue
      }
      let spec = layer.spec ?? manifest.defaultSpec
      perLayer[layer.name] = .quantize(spec)
      prepackedLayers[layer.name] = QwenPrepackedLayer(spec: spec, quantizedFile: layer.quantFile ?? layer.file)
    }
  }
}

extension QwenQuantizationPlan {
  public static func load(from configURL: URL) -> QwenQuantizationPlan? {
    guard let data = try? Data(contentsOf: configURL) else { return nil }
    let decoder = JSONDecoder()
    guard let wrapper = try? decoder.decode(QuantizationFile.self, from: data),
          let container = wrapper.quantization
    else {
      return nil
    }
    return QwenQuantizationPlan(
      defaultSpec: container.quantization,
      perLayer: container.perLayer
    )
  }
}

private struct QuantizationFile: Decodable {
  let quantization: QuantizationContainer?

  enum CodingKeys: String, CodingKey {
    case quantization
    case quantizationConfig = "quantization_config"
  }

  init(from decoder: any Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    if let quantization = try container.decodeIfPresent(QuantizationContainer.self, forKey: .quantization) {
      self.quantization = quantization
    } else if let quantization = try container.decodeIfPresent(
      QuantizationContainer.self,
      forKey: .quantizationConfig
    ) {
      self.quantization = quantization
    } else {
      self.quantization = nil
    }
  }
}

private struct QuantizationContainer: Decodable {
  let quantization: QwenQuantizationSpec
  let perLayer: [String: QwenQuantizationSelection]

  struct DynamicCodingKeys: CodingKey {
    var stringValue: String
    var intValue: Int?

    init?(stringValue: String) {
      self.stringValue = stringValue
      self.intValue = Int(stringValue)
    }

    init?(intValue: Int) {
      self.stringValue = "\(intValue)"
      self.intValue = intValue
    }
  }

  init(from decoder: any Decoder) throws {
    self.quantization = try QwenQuantizationSpec(from: decoder)
    var overrides: [String: QwenQuantizationSelection] = [:]
    let container = try decoder.container(keyedBy: DynamicCodingKeys.self)
    for key in container.allKeys {
      switch key.stringValue {
      case QwenQuantizationSpec.CodingKeys.groupSize.rawValue,
           QwenQuantizationSpec.CodingKeys.bits.rawValue,
           QwenQuantizationSpec.CodingKeys.mode.rawValue:
        continue
      case "quant_method", "linear_class", "quantization_mode":
        continue
      default:
        if let flag = try? container.decode(Bool.self, forKey: key) {
          if !flag {
            overrides[key.stringValue] = .skip
          }
        } else if let spec = try? container.decode(QwenQuantizationSpec.self, forKey: key) {
          overrides[key.stringValue] = .quantize(spec)
        }
      }
    }
    self.perLayer = overrides
  }
}

private extension QuantizationMode {
  static func fromString(_ raw: String?) -> QuantizationMode {
    guard let value = raw?.lowercased() else {
      return .affine
    }
    switch value {
    case "mxfp4":
      return .mxfp4
    default:
      return .affine
    }
  }
}

public struct QwenQuantizedSnapshotManifest: Decodable {
  public struct Layer: Decodable {
    public let component: String?
    public let name: String
    public let file: String?
    public let shape: [Int]?
    public let inDim: Int?
    public let outDim: Int?
    public let quantFile: String?
    public let groupSize: Int?
    public let bits: Int?
    public let mode: QuantizationMode?

    enum CodingKeys: String, CodingKey {
      case component
      case name
      case file
      case shape
      case inDim = "in_dim"
      case outDim = "out_dim"
      case quantFile = "quant_file"
      case groupSize = "group_size"
      case bits
      case mode
    }

    public init(from decoder: any Decoder) throws {
      let container = try decoder.container(keyedBy: CodingKeys.self)
      component = try container.decodeIfPresent(String.self, forKey: .component)
      name = try container.decode(String.self, forKey: .name)
      file = try container.decodeIfPresent(String.self, forKey: .file)
      shape = try container.decodeIfPresent([Int].self, forKey: .shape)
      inDim = try container.decodeIfPresent(Int.self, forKey: .inDim)
      outDim = try container.decodeIfPresent(Int.self, forKey: .outDim)
      quantFile = try container.decodeIfPresent(String.self, forKey: .quantFile)
      groupSize = try container.decodeIfPresent(Int.self, forKey: .groupSize)
      bits = try container.decodeIfPresent(Int.self, forKey: .bits)
      if let modeString = try container.decodeIfPresent(String.self, forKey: .mode) {
        mode = QuantizationMode.fromString(modeString)
      } else {
        mode = nil
      }
    }

    var spec: QwenQuantizationSpec? {
      guard let groupSize, let bits else { return nil }
      return QwenQuantizationSpec(groupSize: groupSize, bits: bits, mode: mode ?? .affine)
    }
  }

  public let version: Int
  public let snapshot: String?
  public let groupSize: Int
  public let bits: Int
  public let mode: QuantizationMode
  public let layers: [Layer]

  enum CodingKeys: String, CodingKey {
    case version
    case snapshot
    case groupSize = "group_size"
    case bits
    case mode
    case layers
  }

  public static func load(from directory: URL) -> QwenQuantizedSnapshotManifest? {
    let manifestURL = directory.appending(path: "quantization.json")
    guard FileManager.default.fileExists(atPath: manifestURL.path),
          let data = try? Data(contentsOf: manifestURL) else {
      return nil
    }
    let decoder = JSONDecoder()
    return try? decoder.decode(QwenQuantizedSnapshotManifest.self, from: data)
  }

  public var defaultSpec: QwenQuantizationSpec {
    QwenQuantizationSpec(groupSize: groupSize, bits: bits, mode: mode)
  }

  public init(from decoder: any Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    version = try container.decode(Int.self, forKey: .version)
    snapshot = try container.decodeIfPresent(String.self, forKey: .snapshot)
    groupSize = try container.decode(Int.self, forKey: .groupSize)
    bits = try container.decode(Int.self, forKey: .bits)
    let modeString = try container.decodeIfPresent(String.self, forKey: .mode)
    mode = QuantizationMode.fromString(modeString)
    layers = try container.decode([Layer].self, forKey: .layers)
  }
}
// swiftlint:enable nesting
