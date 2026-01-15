import Foundation

#if canImport(AppKit)
import AppKit
public typealias PipelineImage = NSImage
#elseif canImport(UIKit)
import UIKit
public typealias PipelineImage = UIImage
#else
public typealias PipelineImage = AnyObject
#endif

// LoadOptions removed; quantization will be added in a future revision.

public enum QwenImageConfig {
  case textToImage
  case imageEditing
}

public struct QwenFlowMatchConfig {
  public var useDynamicShifting: Bool
  public var shift: Float
  public var baseShift: Float
  public var maxShift: Float
  public var baseImageSequenceLength: Int
  public var maxImageSequenceLength: Int
  public var patchSize: Int

  public init(
    useDynamicShifting: Bool = false,
    shift: Float = 1.0,
    baseShift: Float = 0.5,
    maxShift: Float = 1.15,
    baseImageSequenceLength: Int = 256,
    maxImageSequenceLength: Int = 4096,
    patchSize: Int = 16
  ) {
    precondition(baseImageSequenceLength > 0, "baseImageSequenceLength must be positive")
    precondition(maxImageSequenceLength >= baseImageSequenceLength, "maxImageSequenceLength must be >= baseImageSequenceLength")
    precondition(patchSize > 0, "patchSize must be positive")
    precondition(maxShift >= baseShift, "maxShift must be >= baseShift")
    self.useDynamicShifting = useDynamicShifting
    self.shift = shift
    self.baseShift = baseShift
    self.maxShift = maxShift
    self.baseImageSequenceLength = baseImageSequenceLength
    self.maxImageSequenceLength = maxImageSequenceLength
    self.patchSize = patchSize
  }

  public static func load(fromSchedulerDirectory directory: URL) throws -> QwenFlowMatchConfig {
    let configURL = directory.appending(path: "scheduler_config.json")
    guard FileManager.default.fileExists(atPath: configURL.path) else {
      throw QwenConfigLoadingError.missingFile(configURL)
    }
    let data = try Data(contentsOf: configURL)
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let raw = try decoder.decode(FlowMatchSchedulerJSON.self, from: data)
    return QwenFlowMatchConfig(
      useDynamicShifting: raw.useDynamicShifting ?? false,
      shift: raw.shift ?? 1.0,
      baseShift: raw.baseShift ?? 0.5,
      maxShift: raw.maxShift ?? 1.15,
      baseImageSequenceLength: raw.baseImageSeqLen ?? 256,
      maxImageSequenceLength: raw.maxImageSeqLen ?? 4096,
      patchSize: raw.patchSize ?? 16
    )
  }

  private struct FlowMatchSchedulerJSON: Decodable {
    let useDynamicShifting: Bool?
    let shift: Float?
    let baseShift: Float?
    let maxShift: Float?
    let baseImageSeqLen: Int?
    let maxImageSeqLen: Int?
    let patchSize: Int?
  }
}

public struct QwenModelConfiguration {
  public var numTrainSteps: Int
  public var requiresSigmaShift: Bool
  public var maxSequenceLength: Int
  public var flowMatch: QwenFlowMatchConfig

  public init(
    numTrainSteps: Int = 1_000,
    requiresSigmaShift: Bool = false,
    maxSequenceLength: Int = 1_024,
    flowMatch: QwenFlowMatchConfig = .init()
  ) {
    self.numTrainSteps = numTrainSteps
    self.requiresSigmaShift = requiresSigmaShift
    self.maxSequenceLength = maxSequenceLength
    self.flowMatch = flowMatch
  }
}

public struct GenerationParameters {
  public var prompt: String
  public var negativePrompt: String?
  public var width: Int
  public var height: Int
  public var steps: Int
  public var guidanceScale: Float
  public var seed: UInt64?
  public var trueCFGScale: Float?
  /// Optional base resolution override for edit runs (e.g. 512 or 1024).
  /// When set, the scheduler/runtime will use this square base resolution
  /// instead of `width`/`height` for latents and UNet.
  public var editResolution: Int?

  public init(
    prompt: String,
    width: Int = 1024,
    height: Int = 1024,
    steps: Int = 30,
    guidanceScale: Float = 4.0,
    negativePrompt: String? = nil,
    seed: UInt64? = nil,
    trueCFGScale: Float? = nil,
    editResolution: Int? = nil
  ) {
    self.prompt = prompt
    self.width = width
    self.height = height
    self.steps = steps
    self.guidanceScale = guidanceScale
    self.negativePrompt = negativePrompt
    self.seed = seed
    self.trueCFGScale = trueCFGScale
    self.editResolution = editResolution
  }
}

public struct ProgressInfo {
  public let step: Int
  public let total: Int
  public let preview: PipelineImage?

  public init(step: Int, total: Int, preview: PipelineImage? = nil) {
    self.step = step
    self.total = total
    self.preview = preview
  }
}

public typealias QwenProgressHandler = (ProgressInfo) -> Void

// MARK: - Layered Generation Parameters

/// Parameters for layered image decomposition.
/// Used with `generateLayered()` to decompose an image into multiple layers.
public struct LayeredGenerationParameters {
  /// Number of layers to generate (excludes the original image layer)
  public var layers: Int

  /// Resolution bucket: 640 or 1024
  public var resolution: Int

  /// Number of inference steps
  public var numInferenceSteps: Int

  /// True CFG scale (only applied when negativePrompt is set)
  public var trueCFGScale: Float

  /// Whether to normalize CFG output
  public var cfgNormalize: Bool

  /// Optional text prompt for generation
  public var prompt: String?

  /// Optional negative prompt (enables True CFG)
  public var negativePrompt: String?

  /// Random seed for reproducibility
  public var seed: UInt64?

  public init(
    layers: Int = 4,
    resolution: Int = 640,
    numInferenceSteps: Int = 50,
    trueCFGScale: Float = 4.0,
    cfgNormalize: Bool = true,
    prompt: String? = nil,
    negativePrompt: String? = nil,
    seed: UInt64? = nil
  ) {
    precondition(layers > 0, "layers must be positive")
    precondition(resolution == 640 || resolution == 1024, "resolution must be 640 or 1024")
    precondition(numInferenceSteps > 0, "numInferenceSteps must be positive")
    precondition(trueCFGScale >= 1.0, "trueCFGScale must be >= 1.0")

    self.layers = layers
    self.resolution = resolution
    self.numInferenceSteps = numInferenceSteps
    self.trueCFGScale = trueCFGScale
    self.cfgNormalize = cfgNormalize
    self.prompt = prompt
    self.negativePrompt = negativePrompt
    self.seed = seed
  }
}
