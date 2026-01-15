import SwiftUI

@MainActor
struct LayeredWorkspaceView: View {
  @Environment(AppState.self) private var appState
  @State private var showExportSheet = false
  @State private var showLoRAFilePicker = false

  var body: some View {
    @Bindable var viewModel = appState.layeredViewModel

    HSplitView {
      mainCanvas(viewModel: viewModel)
        .frame(minWidth: 500)

      parametersPanel(viewModel: viewModel)
        .frame(width: 300)
    }
    .navigationTitle("Extract Layers")
    .toolbar {
      ToolbarItemGroup(placement: .primaryAction) {
        if appState.activeGenerations[.layered]?.isGenerating == true {
          ProgressView()
            .scaleEffect(0.7)
        }
      }
    }
    .onAppear {
      viewModel.applyDefaults(from: appState.settings)
      let reconciled = appState.reconcileGenerationState(viewModel.generationState, for: .layered)
      if reconciled != viewModel.generationState {
        viewModel.generationState = reconciled
      }
    }
    .onReceive(NotificationCenter.default.publisher(for: .generateRequested)) { _ in
      if viewModel.inputImage != nil && !viewModel.generationState.isGenerating {
        viewModel.generate()
      }
    }
    .onReceive(NotificationCenter.default.publisher(for: .cancelRequested)) { _ in
      viewModel.cancelGeneration()
    }
    .fileExporter(
      isPresented: $showExportSheet,
      document: LayerExportDocument(layers: viewModel.generatedLayers),
      contentType: .folder,
      defaultFilename: "layers"
    ) { result in
      switch result {
      case .success(let url):
        print("Exported to: \(url)")
      case .failure(let error):
        appState.showError(error.localizedDescription)
      }
    }
    .fileImporter(
      isPresented: $showLoRAFilePicker,
      allowedContentTypes: [.data, .fileURL, .zip],
      allowsMultipleSelection: false
    ) { result in
      switch result {
      case .success(let urls):
        if let url = urls.first {
          guard url.startAccessingSecurityScopedResource() else { return }
          defer { url.stopAccessingSecurityScopedResource() }
          viewModel.selectedLoRAPath = url
        }
      case .failure(let error):
        appState.showError(error.localizedDescription)
      }
    }
    .onChange(of: viewModel.selectedLoRAPath) { _, _ in
      guard !viewModel.generationState.isGenerating else { return }
      Task {
        guard await appState.modelService.hasLayeredSession else { return }
        await appState.modelService.unloadLayeredPipeline()
      }
    }
  }

  // MARK: - Main Canvas

  @ViewBuilder
  private func mainCanvas(viewModel: LayeredViewModel) -> some View {
    @Bindable var vm = viewModel
    VStack(spacing: 0) {
      if viewModel.generatedLayers.isEmpty {
        emptyStateView(viewModel: viewModel)
      } else {
        LayerPreviewGrid(
          layers: viewModel.generatedLayers,
          inputImage: viewModel.inputImage
        )
      }

      actionBar(viewModel: viewModel)
    }
    .background(Color(nsColor: .windowBackgroundColor))
  }

  // MARK: - Rich Empty State

  @ViewBuilder
  private func emptyStateView(viewModel: LayeredViewModel) -> some View {
    @Bindable var vm = viewModel
    VStack(spacing: 24) {
      Spacer()

      VStack(spacing: 16) {
        Image(systemName: "square.layers")
          .font(.system(size: 64))
          .foregroundStyle(.secondary)

        Text("Extract Layers from Your Image")
          .font(.title2.bold())

        Text("Drop an image to automatically separate it into transparent layers")
          .font(.subheadline)
          .foregroundStyle(.secondary)
          .multilineTextAlignment(.center)
      }
      .padding(.horizontal)

      ImageDropZone(
        droppedImage: $vm.inputImage,
        placeholder: "Drop image here",
        onImageLoaded: { url in
          viewModel.inputImageURL = url
        }
      )
      .frame(maxWidth: 500, maxHeight: 350)
      .padding(.horizontal)

      VStack(spacing: 8) {
        Text("Tips for best results")
          .font(.subheadline.bold())

        HStack(spacing: 24) {
          TipCard(
            icon: "photo",
            title: "Clear Images",
            description: "Use high-contrast images"
          )
          TipCard(
            icon: "rectangle.split.2x1",
            title: "Simple Scenes",
            description: "Fewer elements = better results"
          )
          TipCard(
            icon: "square.3.layers.3d",
            title: "Multiple Layers",
            description: "Up to 8 layers available"
          )
        }
      }
      .padding()
      .background(
        RoundedRectangle(cornerRadius: 12)
          .fill(Color.secondary.opacity(0.05))
      )
      .padding(.horizontal, 60)

      Spacer()
    }
  }

  // MARK: - Action Bar

  @ViewBuilder
  private func actionBar(viewModel: LayeredViewModel) -> some View {
    VStack(spacing: 0) {
      if let state = appState.activeGenerations[.layered] {
        switch state {
        case .generating(let step, let total, let progress):
          generatingProgressView(step: step, total: total, progress: progress, viewModel: viewModel)
        case .loading:
          loadingView
        default:
          EmptyView()
        }
      }

      HStack {
        if !viewModel.generatedLayers.isEmpty {
          Button {
            viewModel.clear()
          } label: {
            Label("New", systemImage: "plus")
          }
          .keyboardShortcut("n", modifiers: .command)
        }

        Spacer()

        if case .error(let message) = viewModel.generationState {
          errorView(message: message)
        }

        Spacer()

        if !viewModel.generatedLayers.isEmpty {
          Button {
            showExportSheet = true
          } label: {
            Label("Export", systemImage: "square.and.arrow.up")
          }
          .keyboardShortcut("e", modifiers: .command)
        }

        if appState.activeGenerations[.layered]?.isGenerating == true {
          Button {
            viewModel.cancelGeneration()
          } label: {
            Label("Cancel", systemImage: "xmark")
          }
          .keyboardShortcut(.escape, modifiers: [])
        } else {
          Button {
            viewModel.generate()
          } label: {
            Label("Generate", systemImage: "sparkles")
          }
          .buttonStyle(.borderedProminent)
          .disabled(viewModel.inputImage == nil)
          .keyboardShortcut("g", modifiers: .command)
        }
      }
      .padding()
    }
    .background(.regularMaterial)
  }

  private func generatingProgressView(step: Int, total: Int, progress: Float, viewModel: LayeredViewModel) -> some View {
    VStack(spacing: 8) {
      HStack {
        Text("Extracting layers...")
          .font(.caption)
          .foregroundStyle(.secondary)
        Spacer()
        Text(viewModel.estimatedTimeRemaining)
          .font(.caption)
          .foregroundStyle(.secondary)
      }

      HStack(spacing: 4) {
        ProgressView(value: progress)
          .progressViewStyle(.linear)

        Text("\(step)/\(total)")
          .font(.caption.monospacedDigit())
          .foregroundStyle(.secondary)
      }

      Text("Step \(viewModel.currentStepDescription)")
        .font(.caption2)
        .foregroundStyle(.tertiary)
    }
    .padding(.horizontal)
    .padding(.top, 8)
  }

  private var loadingView: some View {
    HStack {
      ProgressView()
        .scaleEffect(0.7)
      Text("Loading model...")
        .font(.caption)
        .foregroundStyle(.secondary)
    }
    .padding(.horizontal)
    .padding(.top, 8)
  }

  private func errorView(message: String) -> some View {
    HStack {
      Image(systemName: "exclamationmark.triangle.fill")
        .foregroundStyle(.red)
      Text(message)
        .font(.caption)
        .foregroundStyle(.red)
        .lineLimit(1)
    }
    .padding(.horizontal)
  }

  // MARK: - Parameters Panel

  @ViewBuilder
  private func parametersPanel(viewModel: LayeredViewModel) -> some View {
    @Bindable var vm = viewModel
    ScrollView {
      VStack(alignment: .leading, spacing: 20) {
        PresetPickerView(
          selectedPreset: $vm.selectedPreset,
          timeEstimate: viewModel.estimatedTime
        )
        .padding(.bottom, 8)

        Divider()

        Section {
          LayersStepper(layers: $vm.layers)

          ResolutionPicker(resolution: $vm.resolution)

          HStack {
            Text("Processing Time")
              .font(.subheadline.bold())
            Spacer()
            Text("\(vm.steps) steps")
              .font(.subheadline.monospacedDigit())
              .foregroundStyle(.secondary)
          }

          ParameterIntSlider(
            title: "Steps",
            value: $vm.steps,
            range: 10...100,
            step: 5,
            description: ParameterHelp.steps
          )

          SeedInputView(
            seed: $vm.seed,
            useRandomSeed: $vm.useRandomSeed
          )
        } header: {
          HStack {
            Text("Settings")
              .font(.headline)
            Spacer()
            HelpTooltip(
              title: "Generation Settings",
              content: "Configure how the AI extracts layers from your image. Use a preset for quick setup, or customize each setting."
            )
          }
        }

        Divider()

        CollapsibleSection(
          title: "Advanced",
          isExpanded: $vm.showAdvancedOptions
        ) {
          VStack(spacing: 16) {
            HStack {
              ParameterSlider(
                title: "True CFG Scale",
                value: $vm.trueCFGScale,
                range: 1...10,
                step: 0.5,
                description: ParameterHelp.trueCFGScale
              )
              HelpTooltip(
                title: "True CFG Scale",
                content: ParameterHelp.trueCFGScale
              )
            }

            Toggle("Normalize Output", isOn: $vm.cfgNormalize)

            VStack(alignment: .leading, spacing: 8) {
              HStack {
                Text("LoRA Adapter")
                  .font(.subheadline.bold())
                HelpTooltip(
                  title: "LoRA Adapter",
                  content: "Apply a LoRA adapter to influence the style of extracted layers."
                )
              }

              if let loraPath = viewModel.selectedLoRAPath {
                HStack {
                  Text(loraPath.lastPathComponent)
                    .font(.caption)
                    .lineLimit(1)
                  Spacer()
                  Button("Remove") {
                    viewModel.selectedLoRAPath = nil
                  }
                  .buttonStyle(.borderless)
                  .font(.caption)
                }
              } else {
                Button("Select LoRA...") {
                  showLoRAFilePicker = true
                }
                .buttonStyle(.bordered)
              }
            }
          }
        }
      }
      .padding()
    }
    .background(Color(nsColor: .controlBackgroundColor))
  }

}

// MARK: - Supporting Views

struct TipCard: View {
  let icon: String
  let title: String
  let description: String

  var body: some View {
    VStack(spacing: 6) {
      Image(systemName: icon)
        .font(.title3)
        .foregroundStyle(.secondary)
      Text(title)
        .font(.caption.bold())
      Text(description)
        .font(.caption2)
        .foregroundStyle(.tertiary)
        .multilineTextAlignment(.center)
    }
    .frame(maxWidth: 100)
  }
}

// MARK: - Layer Preview Grid

struct LayerPreviewGrid: View {
  let layers: [NSImage]
  let inputImage: NSImage?

  @State private var selectedLayerIndex: Int? = nil
  @State private var showComparison = false

  var body: some View {
    VStack(spacing: 16) {
      if let index = selectedLayerIndex, index < layers.count {
        Image(nsImage: layers[index])
          .resizable()
          .aspectRatio(contentMode: .fit)
          .frame(maxHeight: 400)
          .clipShape(RoundedRectangle(cornerRadius: 8))
          .shadow(radius: 4)
      } else if let input = inputImage {
        Image(nsImage: input)
          .resizable()
          .aspectRatio(contentMode: .fit)
          .frame(maxHeight: 400)
          .clipShape(RoundedRectangle(cornerRadius: 8))
          .overlay(
            Text("Original")
              .font(.caption)
              .padding(4)
              .background(.regularMaterial, in: Capsule()),
            alignment: .bottom
          )
      }

      VStack(spacing: 8) {
        HStack {
          ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 12) {
              if let input = inputImage {
                ThumbnailView(
                  image: input,
                  label: "Original",
                  isSelected: selectedLayerIndex == nil
                ) {
                  selectedLayerIndex = nil
                }
              }

              ForEach(Array(layers.enumerated()), id: \.offset) { index, layer in
                ThumbnailView(
                  image: layer,
                  label: "Layer \(index + 1)",
                  isSelected: selectedLayerIndex == index
                ) {
                  selectedLayerIndex = index
                }
              }
            }
            .padding(.horizontal)
          }

          if !layers.isEmpty {
            Divider()
              .frame(height: 40)

            Button {
              showComparison = true
            } label: {
              VStack(spacing: 2) {
                Image(systemName: "arrow.left.and.right")
                  .font(.caption)
                Text("Compare")
                  .font(.caption2)
              }
            }
            .buttonStyle(.borderless)
            .padding(.trailing, 8)
          }
        }
        .frame(height: 80)
      }
    }
    .padding()
    .sheet(isPresented: $showComparison) {
      if let input = inputImage {
        LayerComparisonView(
          layers: layers,
          inputImage: input
        )
      }
    }
  }
}

struct ThumbnailView: View {
  let image: NSImage
  let label: String
  let isSelected: Bool
  let action: () -> Void

  var body: some View {
    Button(action: action) {
      VStack(spacing: 4) {
        Image(nsImage: image)
          .resizable()
          .aspectRatio(contentMode: .fill)
          .frame(width: 60, height: 45)
          .clipShape(RoundedRectangle(cornerRadius: 4))
          .overlay(
            RoundedRectangle(cornerRadius: 4)
              .strokeBorder(isSelected ? Color.accentColor : Color.clear, lineWidth: 2)
          )

        Text(label)
          .font(.caption2)
          .foregroundStyle(isSelected ? .primary : .secondary)
      }
    }
    .buttonStyle(.plain)
  }
}

// MARK: - Export Document

import UniformTypeIdentifiers

struct LayerExportDocument: FileDocument {
  static var readableContentTypes: [UTType] { [.folder] }

  let layers: [NSImage]

  init(layers: [NSImage]) {
    self.layers = layers
  }

  init(configuration: ReadConfiguration) throws {
    layers = []
  }

  func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
    let directoryWrapper = FileWrapper(directoryWithFileWrappers: [:])

    for (index, layer) in layers.enumerated() {
      if let tiffData = layer.tiffRepresentation,
         let bitmap = NSBitmapImageRep(data: tiffData),
         let pngData = bitmap.representation(using: .png, properties: [:]) {
        let filename = "\(UUID().uuidString)_layer\(index + 1).png"
        let fileWrapper = FileWrapper(regularFileWithContents: pngData)
        fileWrapper.preferredFilename = filename
        directoryWrapper.addFileWrapper(fileWrapper)
      }
    }

    return directoryWrapper
  }
}

#Preview {
  LayeredWorkspaceView()
    .environment(AppState())
    .frame(width: 1000, height: 700)
}
