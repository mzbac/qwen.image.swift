import Foundation
import SwiftUI
import QwenImage
import QwenImageRuntime
import MLX
import Combine

@Observable @MainActor
final class TextToImageViewModel {
  var prompt: String = ""
  var negativePrompt: String = ""

  var width: Int = 1024
  var height: Int = 1024
  var steps: Int = 4
  var guidanceScale: Float = 1.0
  var trueCFGScale: Float = 1.0
  var seed: UInt64? = nil
  var useRandomSeed: Bool = true

  var showAdvancedOptions: Bool = false
  var selectedLoRAPath: URL? = nil
  var loraScale: Float = 1.0

  var generatedImage: NSImage?
  var generationState: GenerationState = .idle {
    didSet {
      appState?.setGenerationState(generationState, for: .textToImage)
    }
  }
  private var generationTask: Task<Void, Never>?

  init() {
    selectedLoRAPath = kDefaultLightningLoRAPath
  }

  private var progressCancellable: AnyCancellable?
  var appState: AppState?

  func generate() {
    guard !prompt.isEmpty else {
      generationState = .error("Please enter a prompt")
      return
    }

    guard let appState else {
      generationState = .error("App state not available")
      return
    }

    guard let modelPath = appState.modelPath(for: .edit) else {
      generationState = .error("Image model not downloaded. Please download it first.")
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

    generationState = .loading
    generatedImage = nil

    generationTask = Task.detached { [weak self] in
      do {
        let pipeline = try await modelService.loadImagePipeline(
          from: modelPath,
          config: .textToImage
        )
        if let url = loraURL {
          pipeline.setPendingLora(from: url, scale: 1.0)
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

        let modelConfig = QwenModelConfiguration()

        await MainActor.run { [weak self] in
          self?.generationState = .generating(step: 0, total: stepCount, progress: 0)
          self?.progressCancellable = pipeline.progress?.receive(on: DispatchQueue.main).sink { [weak self] info in
            guard let self else { return }
            let progress = Float(info.step) / Float(info.total)
            self.generationState = .generating(step: info.step, total: info.total, progress: progress)
          }
        }

        // Use pipeline directly for generation (required for LoRA support)
        let pixels = try pipeline.generatePixels(
          parameters: params,
          model: modelConfig,
          seed: actualSeed
        )

        if Task.isCancelled {
          await MainActor.run { [weak self] in
            self?.progressCancellable = nil
            self?.generationState = .idle
          }
          return
        }

        let image = try pipeline.makeImage(from: pixels)

        await MainActor.run { [weak self] in
          guard let self else { return }
          self.progressCancellable = nil
          self.generatedImage = image
          self.generationState = .complete
          if randomSeed {
            self.seed = actualSeed
          }
        }

      } catch {
        await MainActor.run { [weak self] in
          guard let self else { return }
          self.progressCancellable = nil
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
    progressCancellable = nil
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

  var errorDescription: String? {
    switch self {
    case .noImageToExport:
      return "No image to export. Generate an image first."
    }
  }
}
