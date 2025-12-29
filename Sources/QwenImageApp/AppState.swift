import Foundation
import SwiftUI
import QwenImage

/// Default LoRA path for lightning-fast generation
let kDefaultLightningLoRAPath: URL? = {
    let env = ProcessInfo.processInfo.environment
    let hubPath: URL
    if let hfHubCache = env["HF_HUB_CACHE"], !hfHubCache.isEmpty {
        hubPath = URL(fileURLWithPath: hfHubCache)
    } else if let hfHome = env["HF_HOME"], !hfHome.isEmpty {
        hubPath = URL(fileURLWithPath: hfHome).appending(path: "hub")
    } else {
        hubPath = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
    }
    let path = hubPath
        .appending(path: "models/lightx2v/Qwen-Image-Edit-2511-Lightning")
        .appending(path: "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
    return FileManager.default.fileExists(atPath: path.path) ? path : nil
}()

enum SidebarItem: Hashable, Identifiable {
  case mode(GenerationMode)
  case modelManager

  var id: String {
    switch self {
    case .mode(let mode): return mode.rawValue
    case .modelManager: return "modelManager"
    }
  }
}

enum GenerationMode: String, CaseIterable, Identifiable {
  case layered = "Layered"
  case textToImage = "Text to Image"
  case editing = "Image Editing"

  var id: String { rawValue }

  var icon: String {
    switch self {
    case .layered: return "square.stack.3d.up"
    case .textToImage: return "text.below.photo"
    case .editing: return "photo.on.rectangle.angled"
    }
  }

  var description: String {
    switch self {
    case .layered: return "Decompose images into layers"
    case .textToImage: return "Generate images from text"
    case .editing: return "Edit images with prompts"
    }
  }
}

enum ModelStatus: Equatable {
  case notDownloaded
  case downloading(progress: Double, description: String)
  case downloaded(path: URL)
  case loading
  case ready
  case error(String)

  var isReady: Bool {
    if case .ready = self { return true }
    return false
  }

  var isDownloaded: Bool {
    switch self {
    case .downloaded, .loading, .ready:
      return true
    default:
      return false
    }
  }
}

enum ModelVariant: String, CaseIterable, Identifiable {
  case full = "Full (bf16)"
  case quantized8bit = "8-bit"
  case quantized6bit = "6-bit"

  var id: String { rawValue }

  var sizeMultiplier: Double {
    switch self {
    case .full: return 1.0
    case .quantized8bit: return 0.5
    case .quantized6bit: return 0.375
    }
  }

  var description: String {
    switch self {
    case .full: return "Best quality, original bf16 weights"
    case .quantized8bit: return "Good quality, 8-bit quantized weights"
    case .quantized6bit: return "Compact size, 6-bit quantized weights"
    }
  }
}

struct ModelDefinition: Identifiable, Equatable {
  let id: String
  let name: String
  let baseSize: String
  let description: String
  let modes: Set<GenerationMode>
  let variantRepoIds: [ModelVariant: String]

  var availableVariants: [ModelVariant] {
    Array(variantRepoIds.keys).sorted { $0.sizeMultiplier > $1.sizeMultiplier }
  }

  func repoId(for variant: ModelVariant) -> String {
    variantRepoIds[variant] ?? variantRepoIds[.full]!
  }

  func size(for variant: ModelVariant) -> String {
    guard let sizeGB = Double(baseSize.replacingOccurrences(of: "~", with: "").replacingOccurrences(of: "GB", with: "")) else {
      return baseSize
    }
    let adjustedSize = sizeGB * variant.sizeMultiplier
    return "~\(Int(adjustedSize))GB"
  }

  static let layered = ModelDefinition(
    id: "layered",
    name: "Qwen-Image-Layered",
    baseSize: "~54GB",
    description: "Layer decomposition model for splitting images into foreground/background",
    modes: [.layered],
    variantRepoIds: [
      .full: "Qwen/Qwen-Image-Layered",
      .quantized8bit: "mzbac/Qwen-Image-Layered-8bit",
      .quantized6bit: "mzbac/Qwen-Image-Layered-6bit"
    ]
  )

  static let edit = ModelDefinition(
    id: "edit",
    name: "Qwen-Image-Edit",
    baseSize: "~54GB",
    description: "Text-to-image generation and image editing model",
    modes: [.textToImage, .editing],
    variantRepoIds: [
      .full: "Qwen/Qwen-Image-Edit-2509",
      .quantized8bit: "mzbac/Qwen-Image-Edit-2509-8bit"
    ]
  )

  static let all: [ModelDefinition] = [.layered, .edit]
}

@Observable @MainActor
final class AppState {
  var selectedSidebarItem: SidebarItem? = .mode(.layered)

  var selectedMode: GenerationMode? {
    if case .mode(let mode) = selectedSidebarItem {
      return mode
    }
    return nil
  }

  var modelStatuses: [String: ModelStatus] = [
    ModelDefinition.layered.id: .notDownloaded,
    ModelDefinition.edit.id: .notDownloaded
  ]
  var selectedVariants: [String: ModelVariant] = [
    ModelDefinition.layered.id: .quantized8bit,
    ModelDefinition.edit.id: .quantized8bit
  ]
  var downloadedVariants: [String: Set<ModelVariant>] = [:]
  var downloadedVariantPaths: [String: [ModelVariant: URL]] = [:]
  let modelService = ModelService.shared
  var settings = AppSettings()
  var showingError = false
  var errorMessage = ""

  // Lightning LoRA status
  var lightningLoRAStatus: LightningLoRAStatus = .notDownloaded
  
  enum LightningLoRAStatus: Equatable {
    case notDownloaded
    case downloading(progress: Double, description: String)
    case downloaded
    case error(String)
  }

  // Onboarding state
  var hasCompletedOnboarding: Bool {
    get { settings.hasCompletedOnboarding }
    set { settings.hasCompletedOnboarding = newValue }
  }

  // Shared ViewModels - persisted across navigation
  // Use @ObservationIgnored since ViewModels are Observable themselves
  @ObservationIgnored private var _textToImageViewModel: TextToImageViewModel?
  @ObservationIgnored private var _editingViewModel: EditingViewModel?
  @ObservationIgnored private var _layeredViewModel: LayeredViewModel?

  var textToImageViewModel: TextToImageViewModel {
    if let vm = _textToImageViewModel { return vm }
    let vm = TextToImageViewModel()
    vm.appState = self
    _textToImageViewModel = vm
    return vm
  }

  var editingViewModel: EditingViewModel {
    if let vm = _editingViewModel { return vm }
    let vm = EditingViewModel()
    vm.appState = self
    _editingViewModel = vm
    return vm
  }

  var layeredViewModel: LayeredViewModel {
    if let vm = _layeredViewModel { return vm }
    let vm = LayeredViewModel()
    vm.appState = self
    _layeredViewModel = vm
    return vm
  }

  var activeGenerations: [GenerationMode: GenerationState] = [:]
  var lastActiveGenerationMode: GenerationMode?

  func setGenerationState(_ state: GenerationState, for mode: GenerationMode) {
    if state == .idle || state == .complete {
      activeGenerations.removeValue(forKey: mode)
      if activeGenerations.isEmpty {
        lastActiveGenerationMode = nil
      } else if lastActiveGenerationMode == mode {
        lastActiveGenerationMode = activeGenerations.keys.first
      }
    } else {
      activeGenerations[mode] = state
      lastActiveGenerationMode = mode
    }
  }

  func generationState(for mode: GenerationMode) -> GenerationState? {
    activeGenerations[mode]
  }

  func isGenerating(mode: GenerationMode) -> Bool {
    if let state = activeGenerations[mode] {
      return state.isGenerating
    }
    return false
  }

  var hasActiveGeneration: Bool {
    activeGenerations.values.contains { $0.isGenerating }
  }

  var primaryActiveGeneration: (mode: GenerationMode, state: GenerationState)? {
    if let mode = lastActiveGenerationMode, let state = activeGenerations[mode] {
      return (mode, state)
    }
    if let entry = activeGenerations.first {
      return (entry.key, entry.value)
    }
    return nil
  }

  func reconcileGenerationState(_ viewModelState: GenerationState, for mode: GenerationMode) -> GenerationState {
    if let active = activeGenerations[mode] {
      return active
    }
    if viewModelState.isGenerating {
      activeGenerations[mode] = viewModelState
      lastActiveGenerationMode = mode
    }
    return viewModelState
  }

  init() {
    // Load persisted settings
    settings = AppSettings.load()

    // Restore selected variants from persisted settings
    for (modelId, variantRawValue) in settings.selectedVariants {
      if let variant = ModelVariant(rawValue: variantRawValue) {
        selectedVariants[modelId] = variant
      }
    }

    Task {
      await checkModelStatuses()
    }
  }

  func status(for model: ModelDefinition) -> ModelStatus {
    modelStatuses[model.id] ?? .notDownloaded
  }

  func setStatus(_ status: ModelStatus, for model: ModelDefinition) {
    modelStatuses[model.id] = status
  }

  func modelPath(for model: ModelDefinition) -> URL? {
    let variant = selectedVariant(for: model)
    return pathForVariant(variant, model: model)
  }

  func selectedVariant(for model: ModelDefinition) -> ModelVariant {
    selectedVariants[model.id] ?? .quantized8bit
  }

  func setSelectedVariant(_ variant: ModelVariant, for model: ModelDefinition) {
    selectedVariants[model.id] = variant
    // Persist to settings
    settings.selectedVariants[model.id] = variant.rawValue
    settings.save()
  }

  func downloadedVariantsFor(_ model: ModelDefinition) -> Set<ModelVariant> {
    downloadedVariants[model.id] ?? []
  }

  func isVariantDownloaded(_ variant: ModelVariant, for model: ModelDefinition) -> Bool {
    downloadedVariants[model.id]?.contains(variant) ?? false
  }

  func addDownloadedVariant(_ variant: ModelVariant, path: URL, for model: ModelDefinition) {
    if downloadedVariants[model.id] == nil {
      downloadedVariants[model.id] = []
    }
    downloadedVariants[model.id]?.insert(variant)

    if downloadedVariantPaths[model.id] == nil {
      downloadedVariantPaths[model.id] = [:]
    }
    downloadedVariantPaths[model.id]?[variant] = path
  }

  func pathForVariant(_ variant: ModelVariant, model: ModelDefinition) -> URL? {
    downloadedVariantPaths[model.id]?[variant]
  }

  func downloadedVariant(for model: ModelDefinition) -> ModelVariant? {
    selectedVariants[model.id]
  }

  func setDownloadedVariant(_ variant: ModelVariant, for model: ModelDefinition) {
    if let path = pathForVariant(variant, model: model) {
      addDownloadedVariant(variant, path: path, for: model)
    }
    selectedVariants[model.id] = variant
  }

  func isModelReadyFor(mode: GenerationMode) -> Bool {
    let requiredModels = ModelDefinition.all.filter { $0.modes.contains(mode) }
    return requiredModels.allSatisfy { status(for: $0).isDownloaded }
  }

  func checkModelStatuses() async {
    for model in ModelDefinition.all {
      var firstDownloadedPath: URL? = nil
      var firstDownloadedVariant: ModelVariant? = nil

      for variant in model.availableVariants {
        let repoId = model.repoId(for: variant)
        if let path = await modelService.cachedModelPath(repoId: repoId) {
          addDownloadedVariant(variant, path: path, for: model)

          if firstDownloadedPath == nil {
            firstDownloadedPath = path
            firstDownloadedVariant = variant
          }
        }
      }

      if let path = firstDownloadedPath, let variant = firstDownloadedVariant {
        // Use persisted selection if that variant is downloaded, otherwise use first downloaded
        let persistedVariant = selectedVariants[model.id]
        if let persisted = persistedVariant, isVariantDownloaded(persisted, for: model),
           let persistedPath = pathForVariant(persisted, model: model) {
          modelStatuses[model.id] = .downloaded(path: persistedPath)
          // selectedVariants already has the persisted value, no need to change
        } else {
          modelStatuses[model.id] = .downloaded(path: path)
          selectedVariants[model.id] = variant
        }
      }
    }
  }

  func showError(_ message: String) {
    errorMessage = message
    showingError = true
  }

  /// Unload all models (useful for app backgrounding or memory pressure)
  func unloadAllModels() async {
    await modelService.unloadAll()
  }

  /// Get the appropriate model definition for the current mode
  func modelDefinitionForCurrentMode() -> ModelDefinition? {
    guard let mode = selectedMode else { return nil }
    switch mode {
    case .layered:
      return .layered
    case .textToImage, .editing:
      return .edit
    }
  }
}

struct AppSettings: Codable, Equatable {
  var defaultLayers: Int = 4
  var defaultResolution: Int = 640
  var defaultSteps: Int = 50
  var defaultCFGScale: Float = 4.0
  var quantizationEnabled: Bool = false
  var quantizationBits: Int = 8
  var showAdvancedOptions: Bool = false

  /// When true, text encoder is unloaded after prompt encoding to free ~7GB VRAM.
  /// Set to false to keep it in memory for faster generation with different prompts.
  var unloadTextEncoderAfterEncoding: Bool = true

  /// Selected model variant per model ID (persisted)
  var selectedVariants: [String: String] = [:]

  /// Whether the user has completed the onboarding tutorial
  var hasCompletedOnboarding: Bool = false

  private static let key = "AppSettings"

  static func load() -> AppSettings {
    guard let data = UserDefaults.standard.data(forKey: key),
          let settings = try? JSONDecoder().decode(AppSettings.self, from: data)
    else {
      return AppSettings()
    }
    return settings
  }

  func save() {
    guard let data = try? JSONEncoder().encode(self) else { return }
    UserDefaults.standard.set(data, forKey: Self.key)
  }
}
