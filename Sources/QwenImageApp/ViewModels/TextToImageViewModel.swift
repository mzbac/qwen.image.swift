import Foundation
import SwiftUI
import QwenImage
import QwenImageRuntime
import MLX

@Observable @MainActor
final class TextToImageViewModel {
  private enum LightningDefaults {
    static let steps: Int = 4
    static let guidanceScale: Float = 1.0
    static let trueCFGScale: Float = 1.0
    static let loraFilenameSubstring: String = "Lightning-4steps"
  }

  var prompt: String = ""
  var negativePrompt: String = ""

  var width: Int = 512
  var height: Int = 512
  var steps: Int = 20
  var guidanceScale: Float = 1.0
  var trueCFGScale: Float = 1.0
  var seed: UInt64? = nil
  var useRandomSeed: Bool = true

  var showAdvancedOptions: Bool = false
  var selectedLoRAPath: URL? = nil
  var loraScale: Float = 1.0
  private var didAutoApplyLightningDefaults = false

  var generatedImage: NSImage?
  var generationState: GenerationState = .idle {
    didSet {
      appState?.setGenerationState(generationState, for: .textToImage)
    }
  }
  private var generationTask: Task<Void, Never>?

  init() {
    selectedLoRAPath = kDefaultTextToImageLightningLoRAPath
    applyLightningDefaultsIfNeeded()
  }
  var appState: AppState?

  func applyLightningDefaultsIfNeeded() {
    let isLightningLoRASelected =
      selectedLoRAPath?.lastPathComponent.contains(LightningDefaults.loraFilenameSubstring) == true

    if isLightningLoRASelected {
      guard !didAutoApplyLightningDefaults else { return }
      steps = LightningDefaults.steps
      guidanceScale = LightningDefaults.guidanceScale
      trueCFGScale = LightningDefaults.trueCFGScale
      didAutoApplyLightningDefaults = true
    } else {
      didAutoApplyLightningDefaults = false
    }
  }

  func generate() {
    guard !prompt.isEmpty else {
      generationState = .error("Please enter a prompt")
      return
    }

    guard let appState else {
      generationState = .error("App state not available")
      return
    }

    guard let modelPath = appState.modelPath(for: .textToImage) else {
      generationState = .error("Text-to-image model not downloaded. Please download it first.")
      return
    }

    let modelService = appState.modelService
    let promptText = prompt
    let negPromptText = negativePrompt.isEmpty ? nil : negativePrompt
    let widthValue = width
    let heightValue = height
    let stepCount = steps
    let guidance = guidanceScale
    let cfgScale = trueCFGScale
    let randomSeed = useRandomSeed
    let seedValue = seed
    let loraURL = selectedLoRAPath
    let loraScaleValue = loraScale
    let modelVariant = appState.selectedVariant(for: .textToImage)
    let modelRepoId = ModelDefinition.textToImage.repoId(for: modelVariant)

    generationState = .loading
    generatedImage = nil

    generationTask = Task.detached { [weak self] in
      do {
        let session = try await modelService.loadImageSession(
          from: modelPath,
          config: .textToImage,
          modelId: modelRepoId
        )
        if let url = loraURL {
          try await modelService.applyLoRAToImageSession(from: url, scale: loraScaleValue)
        }

        let actualSeed = randomSeed ? UInt64.random(in: 0...UInt64.max) : seedValue
        let params = GenerationParameters(
          prompt: promptText,
          width: widthValue,
          height: heightValue,
          steps: stepCount,
          guidanceScale: guidance,
          negativePrompt: negPromptText,
          seed: actualSeed,
          trueCFGScale: cfgScale
        )

        var modelConfig = QwenModelConfiguration()
        let descriptor = try await session.loadModelDescriptor()
        modelConfig.flowMatch = descriptor.flowMatchConfiguration
        modelConfig.requiresSigmaShift = false

        await MainActor.run { [weak self] in
          self?.generationState = .generating(step: 0, total: stepCount, progress: 0)
        }

        var pixels: MLXArray?
        for try await event in await session.generateStream(parameters: params, model: modelConfig, seed: actualSeed) {
          switch event {
          case .progress(let info):
            let fraction = Float(info.step) / Float(info.total)
            Task { @MainActor [weak self] in
              self?.generationState = .generating(step: info.step, total: info.total, progress: fraction)
            }
          case .output(let outputPixels):
            pixels = outputPixels
          }
        }

        guard let pixels else {
          throw TextToImageError.generationDidNotProduceOutput
        }

        if Task.isCancelled {
          await MainActor.run { [weak self] in
            self?.generationState = .idle
          }
          return
        }

        let image = try await session.makeImage(from: pixels)

        await MainActor.run { [weak self] in
          guard let self else { return }
          self.generatedImage = image
          self.generationState = .complete
          if randomSeed {
            self.seed = actualSeed
          }
        }

      } catch {
        await MainActor.run { [weak self] in
          guard let self else { return }
          if Task.isCancelled {
            self.generationState = .idle
          } else {
            self.generationState = .error(error.localizedDescription)
          }
        }
      }
    }
  }

  func cancelGeneration() {
    generationTask?.cancel()
    generationTask = nil
    generationState = .idle
  }

  func exportImage(to url: URL) throws {
    guard let image = generatedImage else {
      throw TextToImageError.noImageToExport
    }
    try ImageIOService.saveImage(image, to: url, format: .png)
  }

  func clear() {
    generatedImage = nil
    generationState = .idle
  }
}

enum TextToImageError: LocalizedError {
  case noImageToExport
  case generationDidNotProduceOutput

  var errorDescription: String? {
    switch self {
    case .noImageToExport:
      return "No image to export. Generate an image first."
    case .generationDidNotProduceOutput:
      return "Generation did not produce an output image."
    }
  }
}
