import Foundation
import SwiftUI
import QwenImage
import MLX
import Logging

private let logger = Logger(label: "https://github.com/mzbac/qwen.image.swift/LayeredGeneration")

enum GenerationState: Equatable {
  case idle
  case loading
  case generating(step: Int, total: Int, progress: Float)
  case complete
  case error(String)

  var isGenerating: Bool {
    switch self {
    case .loading, .generating:
      return true
    default:
      return false
    }
  }
}

@Observable @MainActor
final class LayeredViewModel {
  var inputImage: NSImage?
  var inputImageURL: URL?

  var layers: Int = 2
  var resolution: Int = 640
  var steps: Int = 4
  var trueCFGScale: Float = 1.0
  var cfgNormalize: Bool = true
  var seed: UInt64? = nil
  var useRandomSeed: Bool = true

  var showAdvancedOptions: Bool = false
  var selectedLoRAPath: URL? = nil
  var loraScale: Float = 1.0

  var selectedPreset: GenerationPreset = .quick {
    didSet {
      if selectedPreset != .custom {
        applyPreset(selectedPreset)
      }
    }
  }

  var generatedLayers: [NSImage] = []
  var generationState: GenerationState = .idle {
    didSet {
      appState?.setGenerationState(generationState, for: .layered)
    }
  }

  private var generationTask: Task<Void, Never>?
  var appState: AppState?

  // MARK: - Time Estimation

  var estimatedTime: String {
    let preset = calculateCurrentPreset()
    return preset.timeDisplay
  }

  var estimatedTimeRemaining: String {
    guard case .generating(let step, let total, _) = generationState else {
      return estimatedTime
    }

    let stepsRemaining = total - step
    let secondsPerStep = estimateSecondsPerStep()
    let totalSeconds = Double(stepsRemaining) * secondsPerStep

    if totalSeconds < 60 {
      return "~\(Int(totalSeconds))s remaining"
    } else {
      return "~\(Int(totalSeconds / 60))m remaining"
    }
  }

  var currentStepDescription: String {
    guard case .generating(let step, let total, _) = generationState else {
      return ""
    }

    let progress = Double(step) / Double(total)

    if progress < 0.25 {
      return "Analyzing image structure..."
    } else if progress < 0.5 {
      return "Extracting main elements..."
    } else if progress < 0.75 {
      return "Refining layer details..."
    } else {
      return "Finalizing layers..."
    }
  }

  // MARK: - Preset Management

  func calculateCurrentPreset() -> GenerationPreset {
    for preset in GenerationPreset.allCases where preset != .custom {
      if preset.matches(layers: layers, steps: steps, resolution: resolution, cfgScale: trueCFGScale) {
        return preset
      }
    }
    return .custom
  }

  func applyPreset(_ preset: GenerationPreset) {
    preset.apply(to: self)
  }

  func isUsingPreset(_ preset: GenerationPreset) -> Bool {
    calculateCurrentPreset() == preset
  }

  // MARK: - Time Estimation Helpers

  private func estimateSecondsPerStep() -> Double {
    let baseSeconds = 3.0
    let resolutionMultiplier = resolution == 1024 ? 4.0 : 1.0
    return baseSeconds * resolutionMultiplier
  }

  // MARK: - Initialization

  init() {}

  func applyDefaults(from settings: AppSettings) {
    layers = settings.defaultLayers
    resolution = settings.defaultResolution
    steps = settings.defaultSteps
    trueCFGScale = settings.defaultCFGScale
    showAdvancedOptions = settings.showAdvancedOptions

    selectedLoRAPath = kDefaultLightningLoRAPath

    for preset in GenerationPreset.allCases where preset != .custom {
      if preset.matches(layers: layers, steps: steps, resolution: resolution, cfgScale: trueCFGScale) {
        selectedPreset = preset
        return
      }
    }
    selectedPreset = .custom
  }

  // MARK: - Generation

  func generate() {
    guard let inputImage else {
      generationState = .error("No input image selected")
      return
    }

    guard let appState else {
      generationState = .error("App state not available")
      return
    }

    guard let modelPath = appState.modelPath(for: .layered) else {
      generationState = .error("Layered model not downloaded. Please download it first.")
      return
    }

    let modelService = appState.modelService
    let loraPath = selectedLoRAPath
    let loraScaleValue = loraScale
    let layerCount = layers
    let resolutionValue = resolution
    let stepCount = steps
    let cfgScale = trueCFGScale
    let cfgNorm = cfgNormalize
    let randomSeed = useRandomSeed
    let seedValue = seed

    generationState = .loading
    generatedLayers = []

    generationTask = Task.detached { [weak self] in
      let taskId = UUID().uuidString.prefix(8)
      logger.info("Task \(taskId): Starting generation... layers=\(layerCount), resolution=\(resolutionValue), steps=\(stepCount)")

      do {
        logger.info("Task \(taskId): Loading pipeline from \(modelPath.path)")
        let pipeline = try await modelService.loadLayeredPipeline(from: modelPath)
        logger.info("Task \(taskId): Pipeline loaded successfully")

        if let loraPath {
          try await modelService.applyLoRA(to: pipeline, from: loraPath, scale: loraScaleValue)
        }

        let cgImage = try ImageIOService.cgImage(from: inputImage)
        let inputArray = ImageIOService.cgImageToMLXArray(cgImage)

        let actualSeed = randomSeed ? UInt64.random(in: 0...UInt64.max) : seedValue
        let params = LayeredGenerationParameters(
          layers: layerCount,
          resolution: resolutionValue,
          numInferenceSteps: stepCount,
          trueCFGScale: cfgScale,
          cfgNormalize: cfgNorm,
          prompt: nil,
          negativePrompt: nil,
          seed: actualSeed
        )

        let weakSelf = self
        logger.info("Task \(taskId): Starting pipeline.generate()...")
        let layerArrays = try pipeline.generate(
          image: inputArray,
          parameters: params
        ) { step, total, progress in
          Task { @MainActor in
            weakSelf?.generationState = .generating(step: step, total: total, progress: progress)
          }
        }
        logger.info("Task \(taskId): Generation complete, got \(layerArrays.count) layers")

        if Task.isCancelled {
          await MainActor.run { [weak self] in
            self?.generationState = .idle
          }
          return
        }

        let images: [NSImage] = try layerArrays.map { layerArray in
          try ImageIOService.mlxArrayToNSImage(layerArray)
        }

        await MainActor.run { [weak self, images] in
          guard let self else { return }
          self.generatedLayers = images
          self.generationState = .complete
          if randomSeed {
            self.seed = actualSeed
          }
          self.addToHistory(images: images)
        }

      } catch {
        logger.error("Task \(taskId): ERROR - \(error.localizedDescription)")
        logger.error("Task \(taskId): Error type: \(String(describing: type(of: error)))")
        await MainActor.run { [weak self] in
          guard let self else { return }
          if Task.isCancelled {
            self.generationState = .idle
          } else {
            self.generationState = .error(error.localizedDescription)
          }
        }
      }
      logger.info("Task \(taskId): Task completed")
    }
  }

  func cancelGeneration() {
    generationTask?.cancel()
    generationTask = nil
    generationState = .idle
  }

  // MARK: - Export

  func exportLayers(to directory: URL) async throws -> [URL] {
    guard !generatedLayers.isEmpty else {
      throw LayeredGenerationError.noLayersToExport
    }

    let baseName = inputImageURL?.deletingPathExtension().lastPathComponent ?? "layer"
    return try ImageIOService.exportLayers(
      generatedLayers,
      to: directory,
      baseName: baseName,
      format: .png
    )
  }

  func clear() {
    inputImage = nil
    inputImageURL = nil
    generatedLayers = []
    generationState = .idle
  }

  // MARK: - History

  private func addToHistory(images: [NSImage]) {
  }
}

enum LayeredGenerationError: LocalizedError {
  case invalidInput(String)
  case noLayersToExport

  var errorDescription: String? {
    switch self {
    case .invalidInput(let message):
      return "Invalid input: \(message)"
    case .noLayersToExport:
      return "No layers to export. Generate layers first."
    }
  }
}
