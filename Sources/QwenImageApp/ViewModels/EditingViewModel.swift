import Foundation
import SwiftUI
import QwenImage
import QwenImageRuntime
import MLX

@Observable @MainActor
final class EditingViewModel {
  var referenceImages: [NSImage] = []
  var prompt: String = ""
  var negativePrompt: String = ""

  var width: Int = 1024
  var height: Int = 1024
  var useCustomSize: Bool = false
  var editResolution: Int = 1024
  var steps: Int = 4
  var guidanceScale: Float = 1.0
  var trueCFGScale: Float = 1.0
  var seed: UInt64? = nil
  var useRandomSeed: Bool = true

  var showAdvancedOptions: Bool = false
  var selectedLoRAPath: URL? = nil
  var loraScale: Float = 1.0

  var editedImage: NSImage?
  var generationState: GenerationState = .idle {
    didSet {
      appState?.setGenerationState(generationState, for: .editing)
    }
  }
  private var generationTask: Task<Void, Never>?
  var appState: AppState?

  init() {
    selectedLoRAPath = kDefaultLightningLoRAPath
  }

  var canAddMoreReferences: Bool {
    referenceImages.count < 2
  }

  func addReferenceImage(_ image: NSImage) {
    guard canAddMoreReferences else { return }
    referenceImages.append(image)
  }

  func removeReferenceImage(at index: Int) {
    guard index < referenceImages.count else { return }
    referenceImages.remove(at: index)
  }

  func clearReferenceImages() {
    referenceImages.removeAll()
  }

  func generate() {
    guard !referenceImages.isEmpty else {
      generationState = .error("Please add at least one reference image")
      return
    }

    guard !prompt.isEmpty else {
      generationState = .error("Please enter a prompt describing the edit")
      return
    }

    guard let appState else {
      generationState = .error("App state not available")
      return
    }

    guard let modelPath = appState.modelPath(for: .edit) else {
      generationState = .error("Edit model not downloaded. Please download it first.")
      return
    }

    let modelService = appState.modelService
    let refImages = referenceImages
    let promptText = prompt
    let negPromptText = negativePrompt.isEmpty ? nil : negativePrompt
    let widthValue = width
    let heightValue = height
    let editRes = editResolution
    let stepCount = steps
    let guidance = guidanceScale
    let cfgScale = trueCFGScale
    let randomSeed = useRandomSeed
    let seedValue = seed
    let loraURL = selectedLoRAPath
    let loraScaleValue = loraScale
    let modelVariant = appState.selectedVariant(for: .edit)
    let modelRepoId = ModelDefinition.edit.repoId(for: modelVariant)

    generationState = .loading
    editedImage = nil

    generationTask = Task.detached { [weak self] in
      do {
        let session = try await modelService.loadImageSession(
          from: modelPath,
          config: .imageEditing,
          modelId: modelRepoId
        )

        if let url = loraURL {
          try await modelService.applyLoRAToImageSession(from: url, scale: loraScaleValue)
        }

        var cgImages: [CGImage] = []
        for nsImage in refImages {
          let cgImage = try ImageIOService.cgImage(from: nsImage)
          cgImages.append(cgImage)
        }

        guard !cgImages.isEmpty else {
          throw EditingError.invalidReferenceImages
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
          trueCFGScale: cfgScale,
          editResolution: editRes
        )

        var modelConfig = QwenModelConfiguration()
        let descriptor = try await session.loadModelDescriptor()
        modelConfig.flowMatch = descriptor.flowMatchConfiguration
        modelConfig.requiresSigmaShift = false

        await MainActor.run { [weak self] in
          self?.generationState = .generating(step: 0, total: stepCount, progress: 0)
        }

        let stream: AsyncThrowingStream<QwenGenerationEvent, Error>
        if cgImages.count == 1 {
          stream = await session.generateEditedPixelsStream(
            parameters: params,
            model: modelConfig,
            referenceImage: cgImages[0],
            maxPromptLength: nil,
            seed: actualSeed
          )
        } else {
          stream = await session.generateEditedPixelsStream(
            parameters: params,
            model: modelConfig,
            referenceImages: cgImages,
            maxPromptLength: nil,
            seed: actualSeed
          )
        }

        var pixels: MLXArray?
        for try await event in stream {
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
          throw EditingError.generationDidNotProduceOutput
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
          self.editedImage = image
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
    guard let image = editedImage else {
      throw EditingError.noImageToExport
    }
    try ImageIOService.saveImage(image, to: url, format: .png)
  }

  func clear() {
    referenceImages.removeAll()
    editedImage = nil
    generationState = .idle
  }
}

enum EditingError: LocalizedError {
  case invalidReferenceImages
  case noImageToExport
  case generationDidNotProduceOutput

  var errorDescription: String? {
    switch self {
    case .invalidReferenceImages:
      return "Failed to process reference images"
    case .noImageToExport:
      return "No image to export. Generate an edit first."
    case .generationDidNotProduceOutput:
      return "Generation did not produce an output image."
    }
  }
}
