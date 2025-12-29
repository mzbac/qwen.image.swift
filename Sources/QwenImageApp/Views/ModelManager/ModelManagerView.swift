import SwiftUI
import QwenImage
import Hub

@MainActor
struct ModelManagerView: View {
  @Environment(AppState.self) private var appState
  @State private var viewModel = ModelManagerViewModel()

  var body: some View {
    ScrollView {
      VStack(spacing: 24) {
        // Header
        VStack(spacing: 8) {
          Text("Model Manager")
            .font(.largeTitle.bold())
          Text("Download and manage AI models for image generation")
            .foregroundStyle(.secondary)
        }
        .padding(.top, 20)

        // Model Cards
        VStack(spacing: 16) {
          ForEach(ModelDefinition.all, id: \.id) { model in
            ModelCard(model: model, viewModel: viewModel)
          }
        }
        .padding(.horizontal, 40)

        // Lightning LoRA Card
        LightningLoRACard(viewModel: viewModel)
          .padding(.horizontal, 40)

        // Info Section
        VStack(alignment: .leading, spacing: 12) {
          Label("About Models", systemImage: "info.circle")
            .font(.headline)

          Text("Models are downloaded from HuggingFace and stored locally. Full models are approximately 54GB; quantized versions (8-bit/6-bit) are smaller.")
            .font(.callout)
            .foregroundStyle(.secondary)

          Text("Models are stored in: ~/.cache/huggingface/hub/")
            .font(.caption)
            .foregroundStyle(.tertiary)
            .fontDesign(.monospaced)
        }
        .padding()
        .frame(maxWidth: 600, alignment: .leading)
        .background(
          RoundedRectangle(cornerRadius: 12)
            .fill(Color.blue.opacity(0.05))
        )
        .padding(.horizontal, 40)
        .padding(.top, 20)

        Spacer(minLength: 40)
      }
    }
    .frame(maxWidth: .infinity, maxHeight: .infinity)
    .background(Color(nsColor: .windowBackgroundColor))
    .onAppear {
      viewModel.appState = appState
    }
  }
}

@MainActor
struct ModelCard: View {
  @Environment(AppState.self) private var appState
  let model: ModelDefinition
  @Bindable var viewModel: ModelManagerViewModel

  private var status: ModelStatus {
    appState.status(for: model)
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      // Header
      HStack {
        VStack(alignment: .leading, spacing: 4) {
          Text(model.name)
            .font(.title2.bold())
          Text(model.description)
            .font(.subheadline)
            .foregroundStyle(.secondary)
        }

        Spacer()

        StatusBadge(status: status)
      }

      // Modes
      HStack(spacing: 8) {
        Text("Supports:")
          .font(.caption)
          .foregroundStyle(.tertiary)

        ForEach(Array(model.modes), id: \.self) { mode in
          Text(mode.rawValue)
            .font(.caption)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(
              Capsule()
                .fill(Color.secondary.opacity(0.1))
            )
        }
      }

      // Progress or Actions
      switch status {
      case .notDownloaded:
        VStack(alignment: .leading, spacing: 12) {
          Text("Select Version")
            .font(.subheadline.bold())

          Picker("Variant", selection: Binding(
            get: { appState.selectedVariant(for: model) },
            set: { appState.setSelectedVariant($0, for: model) }
          )) {
            ForEach(model.availableVariants) { variant in
              VStack(alignment: .leading) {
                Text(variant.rawValue)
              }
              .tag(variant)
            }
          }
          .pickerStyle(.segmented)

          Text(appState.selectedVariant(for: model).description)
            .font(.caption)
            .foregroundStyle(.secondary)

          Button {
            Task {
              await viewModel.downloadModel(model, variant: appState.selectedVariant(for: model))
            }
          } label: {
            Label("Download", systemImage: "arrow.down.circle")
              .frame(maxWidth: .infinity)
          }
          .buttonStyle(.borderedProminent)
          .controlSize(.large)
        }

      case .downloading(let progress, let description):
        VStack(alignment: .leading, spacing: 8) {
          ProgressView(value: progress)
            .progressViewStyle(.linear)

          HStack {
            VStack(alignment: .leading, spacing: 2) {
              Text(description)
                .font(.caption)
                .foregroundStyle(.secondary)
              if let timeRemaining = viewModel.estimatedTimeRemaining(for: model) {
                Text(timeRemaining)
                  .font(.caption2)
                  .foregroundStyle(.tertiary)
              }
            }
            Spacer()
            Text(String(format: "%.1f%%", progress * 100))
              .font(.caption)
              .foregroundStyle(.secondary)
              .monospacedDigit()
          }

          Button("Cancel") {
            viewModel.cancelDownload(model)
          }
          .buttonStyle(.bordered)
        }

      case .downloaded:
        VStack(alignment: .leading, spacing: 12) {
          // Show downloaded variants
          let downloadedVariants = appState.downloadedVariantsFor(model)
          let selectedVariant = appState.selectedVariant(for: model)

          if !downloadedVariants.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
              Text("Available Versions")
                .font(.subheadline.bold())

              ForEach(model.availableVariants) { variant in
                let isDownloaded = downloadedVariants.contains(variant)
                let isSelected = variant == selectedVariant

                HStack {
                  if isDownloaded {
                    Image(systemName: isSelected ? "checkmark.circle.fill" : "circle.fill")
                      .foregroundStyle(isSelected ? .green : .secondary)
                  } else {
                    Image(systemName: "circle.dashed")
                      .foregroundStyle(.secondary.opacity(0.5))
                  }

                  VStack(alignment: .leading, spacing: 2) {
                    Text(variant.rawValue)
                      .foregroundStyle(isDownloaded ? .primary : .secondary)
                    Text(variant.description)
                      .font(.caption)
                      .foregroundStyle(.secondary)
                    if isDownloaded, let path = appState.pathForVariant(variant, model: model) {
                      Text(viewModel.formattedSize(for: path))
                        .font(.caption2)
                        .foregroundStyle(.green)
                    }
                  }

                  Spacer()

                  if isDownloaded {
                    if isSelected {
                      Text("Active")
                        .font(.caption)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(Capsule().fill(Color.green.opacity(0.15)))
                        .foregroundStyle(.green)
                    } else {
                      Button("Use") {
                        appState.setSelectedVariant(variant, for: model)
                        if let path = appState.pathForVariant(variant, model: model) {
                          appState.setStatus(.downloaded(path: path), for: model)
                        }
                      }
                      .buttonStyle(.bordered)
                      .controlSize(.small)
                    }
                  } else {
                    Button("Download") {
                      Task {
                        await viewModel.downloadModel(model, variant: variant)
                      }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                  }
                }
                .padding(.vertical, 4)
              }
            }

            Divider()

            // Show in Finder for selected variant
            if let path = appState.pathForVariant(selectedVariant, model: model) {
              HStack {
                Text("Model location:")
                  .font(.caption)
                  .foregroundStyle(.tertiary)
                Spacer()
                Button("Show in Finder") {
                  NSWorkspace.shared.activateFileViewerSelecting([path])
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
              }
            }
          }
        }

      case .loading:
        HStack {
          ProgressView()
            .scaleEffect(0.8)
          Text("Loading model...")
            .foregroundStyle(.secondary)
        }

      case .ready:
        HStack {
          Image(systemName: "checkmark.circle.fill")
            .foregroundStyle(.green)
          Text("Ready to use")
            .foregroundStyle(.green)
        }

      case .error(let message):
        VStack(alignment: .leading, spacing: 8) {
          HStack {
            Image(systemName: "exclamationmark.triangle.fill")
              .foregroundStyle(.red)
            Text(message)
              .font(.callout)
              .foregroundStyle(.red)
          }

          Button("Retry Download") {
            Task {
              await viewModel.downloadModel(model, variant: appState.selectedVariant(for: model))
            }
          }
          .buttonStyle(.borderedProminent)
        }
      }
    }
    .padding(20)
    .frame(maxWidth: 600)
    .background(
      RoundedRectangle(cornerRadius: 16)
        .fill(Color(nsColor: .controlBackgroundColor))
    )
    .overlay(
      RoundedRectangle(cornerRadius: 16)
        .strokeBorder(Color.secondary.opacity(0.15), lineWidth: 1)
    )
  }
}

// MARK: - Lightning LoRA Card

@MainActor
struct LightningLoRACard: View {
  @Environment(AppState.self) private var appState
  @Bindable var viewModel: ModelManagerViewModel
  
  private var isInstalled: Bool {
    kDefaultLightningLoRAPath != nil
  }
  
  private var loraSize: String? {
    guard let path = kDefaultLightningLoRAPath else { return nil }
    return viewModel.formattedSize(for: path)
  }
  
  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      // Header
      HStack {
        VStack(alignment: .leading, spacing: 4) {
          Text("Lightning LoRA")
            .font(.title2.bold())
          Text("Speed up generation to just 4 steps with reduced quality")
            .font(.subheadline)
            .foregroundStyle(.secondary)
        }
        
        Spacer()
        
        // Status Badge
        statusBadge
      }
      
      // Content based on status
      switch appState.lightningLoRAStatus {
      case .notDownloaded:
        if isInstalled {
          installedContent
        } else {
          notInstalledContent
        }
      case .downloading(let progress, let description):
        downloadingContent(progress: progress, description: description)
      case .downloaded:
        installedContent
      case .error(let message):
        errorContent(message: message)
      }
    }
    .padding(20)
    .frame(maxWidth: 600)
    .background(
      RoundedRectangle(cornerRadius: 16)
        .fill(Color(nsColor: .controlBackgroundColor))
    )
    .overlay(
      RoundedRectangle(cornerRadius: 16)
        .strokeBorder(borderColor.opacity(0.3), lineWidth: 1)
    )
    .onAppear {
      // Check if already installed on appear
      if isInstalled && appState.lightningLoRAStatus == .notDownloaded {
        appState.lightningLoRAStatus = .downloaded
      }
    }
  }
  
  private var borderColor: Color {
    switch appState.lightningLoRAStatus {
    case .notDownloaded:
      return isInstalled ? .green : .orange
    case .downloading:
      return .blue
    case .downloaded:
      return .green
    case .error:
      return .red
    }
  }
  
  @ViewBuilder
  private var statusBadge: some View {
    let (color, text): (Color, String) = {
      switch appState.lightningLoRAStatus {
      case .notDownloaded:
        return isInstalled ? (.green, "Installed") : (.secondary, "Not Installed")
      case .downloading:
        return (.blue, "Downloading")
      case .downloaded:
        return (.green, "Installed")
      case .error:
        return (.red, "Error")
      }
    }()
    
    HStack(spacing: 4) {
      if case .downloading = appState.lightningLoRAStatus {
        ProgressView()
          .scaleEffect(0.6)
      } else {
        Circle()
          .fill(color)
          .frame(width: 8, height: 8)
      }
      Text(text)
        .font(.caption)
    }
    .padding(.horizontal, 10)
    .padding(.vertical, 5)
    .background(
      Capsule()
        .fill(color.opacity(0.1))
    )
  }
  
  @ViewBuilder
  private var installedContent: some View {
    HStack {
      Image(systemName: "checkmark.circle.fill")
        .foregroundStyle(.green)
      VStack(alignment: .leading, spacing: 2) {
        Text("Ready to use")
          .foregroundStyle(.green)
        if let size = loraSize {
          Text(size)
            .font(.caption2)
            .foregroundStyle(.green)
        }
      }
      Spacer()
      Button("Show in Finder") {
        if let path = kDefaultLightningLoRAPath {
          NSWorkspace.shared.activateFileViewerSelecting([path])
        }
      }
      .buttonStyle(.bordered)
      .controlSize(.small)
    }
  }
  
  @ViewBuilder
  private var notInstalledContent: some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack {
        Image(systemName: "bolt.fill")
          .foregroundStyle(.orange)
        Text("Optional: Download for faster generation")
          .font(.callout)
      }
      
      Text("The Lightning LoRA enables 4-step generation (instead of 20+ steps) for faster results with slightly reduced quality.")
        .font(.caption)
        .foregroundStyle(.secondary)
      
      Button {
        Task {
          await viewModel.downloadLightningLoRA()
        }
      } label: {
        Label("Download Lightning LoRA", systemImage: "arrow.down.circle")
          .frame(maxWidth: .infinity)
      }
      .buttonStyle(.borderedProminent)
      .controlSize(.regular)
    }
  }
  
  @ViewBuilder
  private func downloadingContent(progress: Double, description: String) -> some View {
    VStack(alignment: .leading, spacing: 8) {
      ProgressView(value: progress)
        .progressViewStyle(.linear)
      
      HStack {
        Text(description)
          .font(.caption)
          .foregroundStyle(.secondary)
        Spacer()
        Text(String(format: "%.1f%%", progress * 100))
          .font(.caption)
          .foregroundStyle(.secondary)
          .monospacedDigit()
      }
      
      Button("Cancel") {
        viewModel.cancelLightningLoRADownload()
      }
      .buttonStyle(.bordered)
    }
  }
  
  @ViewBuilder
  private func errorContent(message: String) -> some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack {
        Image(systemName: "exclamationmark.triangle.fill")
          .foregroundStyle(.red)
        Text(message)
          .font(.callout)
          .foregroundStyle(.red)
      }
      
      Button("Retry Download") {
        Task {
          await viewModel.downloadLightningLoRA()
        }
      }
      .buttonStyle(.borderedProminent)
    }
  }
}

struct StatusBadge: View {
  let status: ModelStatus

  var body: some View {
    HStack(spacing: 4) {
      Circle()
        .fill(statusColor)
        .frame(width: 8, height: 8)
      Text(statusText)
        .font(.caption)
    }
    .padding(.horizontal, 10)
    .padding(.vertical, 5)
    .background(
      Capsule()
        .fill(statusColor.opacity(0.1))
    )
  }

  private var statusText: String {
    switch status {
    case .notDownloaded: return "Not Downloaded"
    case .downloading: return "Downloading"
    case .downloaded: return "Downloaded"
    case .loading: return "Loading"
    case .ready: return "Ready"
    case .error: return "Error"
    }
  }

  private var statusColor: Color {
    switch status {
    case .notDownloaded: return .secondary
    case .downloading, .loading: return .blue
    case .downloaded, .ready: return .green
    case .error: return .red
    }
  }
}

// MARK: - ViewModel

@Observable @MainActor
final class ModelManagerViewModel {
  var appState: AppState?
  private var downloadTasks: [String: Task<Void, Never>] = [:]
  private var downloadProgresses: [String: HubSnapshotProgress] = [:]

  /// Get the actual formatted size of a downloaded model (recursively calculates directory size)
  func formattedSize(for path: URL) -> String {
    let fm = FileManager.default
    var isDirectory: ObjCBool = false
    guard fm.fileExists(atPath: path.path, isDirectory: &isDirectory) else {
      return "Unknown"
    }
    
    let totalSize: Int64
    if isDirectory.boolValue {
      // Recursively calculate the total size of all files in the directory
      totalSize = calculateDirectorySize(at: path, fileManager: fm)
    } else {
      // Single file
      if let attrs = try? fm.attributesOfItem(atPath: path.path),
         let size = attrs[.size] as? Int64 {
        totalSize = size
      } else {
        return "Unknown"
      }
    }
    
    return ByteCountFormatter.string(fromByteCount: totalSize, countStyle: .file)
  }
  
  /// Recursively calculate the total size of a directory
  private func calculateDirectorySize(at url: URL, fileManager: FileManager) -> Int64 {
    var totalSize: Int64 = 0
    
    guard let enumerator = fileManager.enumerator(
      at: url,
      includingPropertiesForKeys: [.fileSizeKey, .isDirectoryKey],
      options: [.skipsHiddenFiles]
    ) else {
      return 0
    }
    
    for case let fileURL as URL in enumerator {
      do {
        let resourceValues = try fileURL.resourceValues(forKeys: [.fileSizeKey, .isDirectoryKey])
        if resourceValues.isDirectory == false {
          totalSize += Int64(resourceValues.fileSize ?? 0)
        }
      } catch {
        // Skip files we can't read
        continue
      }
    }
    
    return totalSize
  }

  /// Estimate the remaining download time for a model
  func estimatedTimeRemaining(for model: ModelDefinition) -> String? {
    guard let progress = downloadProgresses[model.id] else { return nil }
    return progress.formattedTimeRemaining
  }

  func downloadModel(_ model: ModelDefinition, variant: ModelVariant) async {
    guard let appState else { return }

    appState.setStatus(.downloading(progress: 0, description: "Starting..."), for: model)

    let localAppState = appState
    let repoId = model.repoId(for: variant)

    let task = Task {
      do {
        let path = try await localAppState.modelService.downloadModel(
          repoId: repoId,
          progressHandler: { progress in
            Task { @MainActor in
              // Store progress for time estimation
              self.downloadProgresses[model.id] = progress

              let description: String
              if let speed = progress.formattedSpeed {
                description = "\(progress.formattedCompleted) / \(progress.formattedTotal) @ \(speed)"
              } else {
                description = "\(progress.formattedCompleted) / \(progress.formattedTotal)"
              }

              localAppState.setStatus(
                .downloading(progress: progress.fractionCompleted, description: description),
                for: model
              )
            }
          }
        )

        await MainActor.run {
          localAppState.addDownloadedVariant(variant, path: path, for: model)
          localAppState.setSelectedVariant(variant, for: model)
          localAppState.setStatus(.downloaded(path: path), for: model)
          // Clear progress tracking
          self.downloadProgresses.removeValue(forKey: model.id)
        }
      } catch {
        await MainActor.run {
          if Task.isCancelled {
            localAppState.setStatus(.notDownloaded, for: model)
          } else {
            localAppState.setStatus(.error(error.localizedDescription), for: model)
          }
          self.downloadProgresses.removeValue(forKey: model.id)
        }
      }
    }

    downloadTasks[model.id] = task
    await task.value
    downloadTasks[model.id] = nil
  }

  func cancelDownload(_ model: ModelDefinition) {
    downloadTasks[model.id]?.cancel()
    downloadTasks[model.id] = nil
    downloadProgresses.removeValue(forKey: model.id)
    appState?.setStatus(.notDownloaded, for: model)
  }

  // MARK: - Lightning LoRA Download

  private var lightningLoRATask: Task<Void, Never>?

  func downloadLightningLoRA() async {
    guard let appState else { return }

    appState.lightningLoRAStatus = .downloading(progress: 0, description: "Starting...")

    let localAppState = appState

    let task = Task {
      do {
        _ = try await localAppState.modelService.downloadLightningLoRA { progress in
          Task { @MainActor in
            let description: String
            if let speed = progress.formattedSpeed {
              description = "\(progress.formattedCompleted) / \(progress.formattedTotal) @ \(speed)"
            } else {
              description = "\(progress.formattedCompleted) / \(progress.formattedTotal)"
            }

            localAppState.lightningLoRAStatus = .downloading(
              progress: progress.fractionCompleted,
              description: description
            )
          }
        }

        let loraPath = await localAppState.modelService.lightningLoRAPath()
        await MainActor.run {
          localAppState.lightningLoRAStatus = .downloaded
          if let loraPath {
            localAppState.textToImageViewModel.selectedLoRAPath = loraPath
            localAppState.editingViewModel.selectedLoRAPath = loraPath
            localAppState.layeredViewModel.selectedLoRAPath = loraPath
          }
        }
      } catch {
        await MainActor.run {
          if Task.isCancelled {
            localAppState.lightningLoRAStatus = .notDownloaded
          } else {
            localAppState.lightningLoRAStatus = .error(error.localizedDescription)
          }
        }
      }
    }

    lightningLoRATask = task
    await task.value
    lightningLoRATask = nil
  }

  func cancelLightningLoRADownload() {
    lightningLoRATask?.cancel()
    lightningLoRATask = nil
    appState?.lightningLoRAStatus = .notDownloaded
  }
}

#Preview {
  ModelManagerView()
    .environment(AppState())
    .frame(width: 800, height: 700)
}
