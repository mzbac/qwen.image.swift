import Foundation
import MLX

public protocol QwenScheduler {
  var sigmas: MLXArray { get }
  func scaleModelInput(_ latents: MLXArray, timestep: Int) -> MLXArray
  func step(modelOutput: MLXArray, timestep: Int, sample: MLXArray) -> MLXArray
}

public struct QwenSchedulerState {
  public let sigmas: MLXArray

  public init(sigmas: MLXArray) {
    self.sigmas = sigmas
  }
}

public struct QwenRuntimeConfig {
  public var height: Int
  public var width: Int
  public var guidanceScale: Float
  public var numInferenceSteps: Int
  public var scheduler: QwenSchedulerState

  public init(
    height: Int,
    width: Int,
    guidanceScale: Float,
    numInferenceSteps: Int,
    scheduler: QwenSchedulerState
  ) {
    self.height = height
    self.width = width
    self.guidanceScale = guidanceScale
    self.numInferenceSteps = numInferenceSteps
    self.scheduler = scheduler
  }
}

public struct QwenFlowMatchScheduler: QwenScheduler {
  public let sigmas: MLXArray
  public let requiresSigmaShift: Bool
  public let width: Int
  public let height: Int
  public let numInferenceSteps: Int
  public let flowMatchConfig: QwenFlowMatchConfig
  public let mu: Float?  // Dynamic shifting parameter for layered generation

  public init(
    numInferenceSteps: Int,
    width: Int,
    height: Int,
    requiresSigmaShift: Bool = false,
    flowMatchConfig: QwenFlowMatchConfig = .init(),
    mu: Float? = nil  // Optional mu for dynamic shifting (overrides config if provided)
  ) {
    precondition(numInferenceSteps > 0, "numInferenceSteps must be positive")
    self.numInferenceSteps = numInferenceSteps
    self.width = width
    self.height = height
    self.requiresSigmaShift = requiresSigmaShift
    self.flowMatchConfig = flowMatchConfig
    self.mu = mu

    let sigmaValues = Self.computeSigmas(
      steps: numInferenceSteps,
      width: width,
      height: height,
      requiresSigmaShift: requiresSigmaShift,
      flowMatchConfig: flowMatchConfig,
      mu: mu
    )
    self.sigmas = MLXArray(sigmaValues, [sigmaValues.count])
  }

  public func scaleModelInput(_ latents: MLXArray, timestep: Int) -> MLXArray {
    latents
  }

  public func step(modelOutput: MLXArray, timestep: Int, sample: MLXArray) -> MLXArray {
    precondition(timestep < numInferenceSteps, "timestep must be within inference steps")
    let dt = (sigmas[timestep + 1] - sigmas[timestep]).asType(sample.dtype)
    return sample + modelOutput * dt
  }

  public func makeState() -> QwenSchedulerState {
    QwenSchedulerState(sigmas: sigmas)
  }
}

extension QwenFlowMatchScheduler {
  private static func computeSigmas(
    steps: Int,
    width: Int,
    height: Int,
    requiresSigmaShift: Bool,
    flowMatchConfig: QwenFlowMatchConfig,
    mu: Float? = nil
  ) -> [Float32] {
    var values: [Float32] = []
    values.reserveCapacity(steps + 1)

    // Match Python's linspace(1.0, 0.0, steps+1)[:-1]
    let sigmaMax: Float32 = 1.0
    let sigmaMin: Float32 = 0.0

    if steps == 1 {
      values.append(sigmaMax)
    } else {
      for index in 0..<steps {
        let t = Float32(index) / Float32(steps)
        let sigma = sigmaMax + t * (sigmaMin - sigmaMax)
        values.append(sigma)
      }
    }

    // Apply dynamic sigma shifting if mu is provided (for layered generation)
    if let mu = mu {
      values = applyLinearTimeShift(values, mu: mu)
    } else if requiresSigmaShift {
      if flowMatchConfig.useDynamicShifting {
        let computedMu = computeDynamicShiftMu(
          width: width,
          height: height,
          flowMatchConfig: flowMatchConfig
        )
        values = applyDynamicSigmaShift(values, mu: computedMu)
      } else {
        values = applyStaticSigmaShift(values, shift: flowMatchConfig.shift)
      }
    } else if flowMatchConfig.useDynamicShifting {
      let computedMu = computeDynamicShiftMu(
        width: width,
        height: height,
        flowMatchConfig: flowMatchConfig
      )
      values = applyDynamicSigmaShift(values, mu: computedMu)
    } else if abs(flowMatchConfig.shift - 1.0) > Float.ulpOfOne {
      values = applyStaticSigmaShift(values, shift: flowMatchConfig.shift)
    }

    values.append(0.0)
    return values
  }

  /// Linear time shift: mu * t / (1 + (mu - 1) * t)
  /// Used for layered generation (matches Python's _time_shift_linear function)
  private static func applyLinearTimeShift(
    _ sigmas: [Float32],
    mu: Float32
  ) -> [Float32] {
    guard !sigmas.isEmpty else { return sigmas }

    return sigmas.map { sigma in
      guard sigma > 0 else { return sigma }
      let numerator = mu * sigma
      let denominator = 1.0 + (mu - 1.0) * sigma
      guard denominator != 0 else { return sigma }
      return numerator / denominator
    }
  }

  /// Exponential dynamic sigma shift (original behavior)
  private static func applyDynamicSigmaShift(
    _ sigmas: [Float32],
    mu: Float32,
    exponent: Float32 = 1.0
  ) -> [Float32] {
    guard !sigmas.isEmpty else { return sigmas }

    let expMu = Float32(Foundation.exp(Double(mu)))

    return sigmas.map { sigma in
      let inverseBase = max(1 / sigma - 1, 0)
      let inverse = Float32(pow(Double(inverseBase), Double(exponent)))
      let denominator = expMu + inverse
      guard denominator != 0 else { return sigma }
      return expMu / denominator
    }
  }

  private static func applyStaticSigmaShift(
    _ sigmas: [Float32],
    shift: Float32
  ) -> [Float32] {
    guard abs(shift - 1.0) > Float.ulpOfOne, !sigmas.isEmpty else { return sigmas }
    return sigmas.map { sigma in
      let numerator = shift * sigma
      let denominator = 1 + (shift - 1) * sigma
      guard denominator != 0 else { return sigma }
      return numerator / denominator
    }
  }

  private static func computeDynamicShiftMu(
    width: Int,
    height: Int,
    flowMatchConfig: QwenFlowMatchConfig
  ) -> Float32 {
    let sequenceLength = computeImageSequenceLength(
      width: width,
      height: height,
      flowMatchConfig: flowMatchConfig
    )
    let baseLen = Float32(flowMatchConfig.baseImageSequenceLength)
    let maxLen = Float32(flowMatchConfig.maxImageSequenceLength)
    let slope: Float32
    if abs(maxLen - baseLen) <= .ulpOfOne {
      slope = 0
    } else {
      slope = (flowMatchConfig.maxShift - flowMatchConfig.baseShift) / (maxLen - baseLen)
    }
    let intercept = flowMatchConfig.baseShift - slope * baseLen
    let clamped = min(max(sequenceLength, baseLen), maxLen)
    return slope * clamped + intercept
  }

  private static func computeImageSequenceLength(
    width: Int,
    height: Int,
    flowMatchConfig: QwenFlowMatchConfig
  ) -> Float32 {
    let patchSize = max(1, flowMatchConfig.patchSize)
    let latentWidth = max(1, width / patchSize)
    let latentHeight = max(1, height / patchSize)
    return Float32(latentWidth * latentHeight)
  }
}

public enum QwenSchedulerFactory {
  public static func flowMatchSchedulerRuntime(
    model: QwenModelConfiguration,
    generation: GenerationParameters
  ) -> (scheduler: QwenFlowMatchScheduler, runtime: QwenRuntimeConfig) {
    // For edit runs, an optional editResolution can override the canvas
    // width/height used for latents and UNet. When provided, compute dimensions
    // that preserve the user's aspect ratio while matching the target resolution area.
    // When nil, fall back to the requested width/height (text-to-image and legacy behavior).
    let baseWidth: Int
    let baseHeight: Int
    if let editRes = generation.editResolution {
      // Preserve aspect ratio while targeting editResolution area
      let targetArea = editRes * editRes
      let (w, h) = EditSizing.computeDimensions(
        referenceWidth: generation.width,
        referenceHeight: generation.height,
        targetArea: targetArea,
        multiple: 16
      )
      baseWidth = w
      baseHeight = h
    } else {
      baseWidth = generation.width
      baseHeight = generation.height
    }

    let width = adjustedDimension(baseWidth)
    let height = adjustedDimension(baseHeight)

    let scheduler = QwenFlowMatchScheduler(
      numInferenceSteps: generation.steps,
      width: width,
      height: height,
      requiresSigmaShift: model.requiresSigmaShift,
      flowMatchConfig: model.flowMatch
    )
    let runtime = QwenRuntimeConfig(
      height: height,
      width: width,
      guidanceScale: generation.guidanceScale,
      numInferenceSteps: generation.steps,
      scheduler: scheduler.makeState()
    )
    return (scheduler, runtime)
  }

  public static func flowMatchSchedulerForLayered(
    numInferenceSteps: Int,
    width: Int,
    height: Int,
    flowMatchConfig: QwenFlowMatchConfig
  ) -> QwenFlowMatchScheduler {
    let patchSize = max(1, flowMatchConfig.patchSize)
    let tokenWidth = max(1, width / patchSize)
    let tokenHeight = max(1, height / patchSize)
    let imageSequenceLength = Float(tokenWidth * tokenHeight)
    let baseSequenceLength = max(1, flowMatchConfig.baseImageSequenceLength)
    let mu = sqrt(imageSequenceLength / Float(baseSequenceLength))

    return QwenFlowMatchScheduler(
      numInferenceSteps: numInferenceSteps,
      width: width,
      height: height,
      requiresSigmaShift: false,
      flowMatchConfig: flowMatchConfig,
      mu: mu
    )
  }

  private static func adjustedDimension(_ value: Int) -> Int {
    guard value > 0 else { return 16 }
    let multiple = max(1, value / 16)
    let adjusted = multiple * 16
    return max(16, adjusted)
  }
}
