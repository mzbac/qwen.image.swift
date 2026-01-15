import SwiftUI
import UniformTypeIdentifiers

@MainActor
struct TextToImageView: View {
  @Environment(AppState.self) private var appState
  @State private var showExportSheet = false
  @State private var showLoRAFilePicker = false

  var body: some View {
    @Bindable var viewModel = appState.textToImageViewModel

    HSplitView {
      // Main Canvas Area
      mainCanvasView(viewModel: viewModel)
        .frame(minWidth: 500)

      // Parameters Panel
      parametersPanelView(viewModel: viewModel)
        .frame(width: 280)
    }
    .navigationTitle("Text to Image")
    .toolbar {
      ToolbarItemGroup(placement: .primaryAction) {
        if appState.activeGenerations[.textToImage]?.isGenerating == true {
          ProgressView()
            .scaleEffect(0.7)
        }
      }
    }
    .onAppear {
      let reconciled = appState.reconcileGenerationState(viewModel.generationState, for: .textToImage)
      if reconciled != viewModel.generationState {
        viewModel.generationState = reconciled
      }
    }
    .onReceive(NotificationCenter.default.publisher(for: .generateRequested)) { _ in
      if !viewModel.prompt.isEmpty && !viewModel.generationState.isGenerating {
        viewModel.generate()
      }
    }
    .onReceive(NotificationCenter.default.publisher(for: .cancelRequested)) { _ in
      viewModel.cancelGeneration()
    }
    .fileExporter(
      isPresented: $showExportSheet,
      document: ImageExportDocument(image: viewModel.generatedImage),
      contentType: .png,
      defaultFilename: UUID().uuidString
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
          // Start accessing security-scoped resource
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
      viewModel.applyLightningDefaultsIfNeeded()
      Task {
        guard await appState.modelService.hasImageSession else { return }
        await appState.modelService.unloadImagePipeline()
      }
    }
  }

  // MARK: - Main Canvas

  @ViewBuilder
  private func mainCanvasView(viewModel: TextToImageViewModel) -> some View {
    VStack(spacing: 0) {
      // Result Area
      ZStack {
        if let image = viewModel.generatedImage {
          Image(nsImage: image)
            .resizable()
            .aspectRatio(contentMode: .fit)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding()
        } else {
          VStack(spacing: 16) {
            Image(systemName: "text.below.photo")
              .font(.system(size: 64))
              .foregroundStyle(.secondary)

            Text("Enter a prompt in the sidebar and click Generate")
              .font(.title3)
              .foregroundStyle(.secondary)

            Text("Your generated image will appear here")
              .font(.subheadline)
              .foregroundStyle(.tertiary)
          }
        }
      }
      .frame(maxWidth: .infinity, maxHeight: .infinity)

      // Action Bar
      actionBarView(viewModel: viewModel)
    }
    .background(Color(nsColor: .windowBackgroundColor))
  }

  // MARK: - Action Bar

  @ViewBuilder
  private func actionBarView(viewModel: TextToImageViewModel) -> some View {
    VStack(spacing: 0) {
      // Progress bar when generating
      if let state = appState.activeGenerations[.textToImage] {
        switch state {
        case .generating(let step, let total, let progress):
          VStack(spacing: 4) {
            HStack {
              Text("Generating...")
                .font(.caption)
                .foregroundStyle(.secondary)
              Spacer()
              Text("Step \(step)/\(total)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            }
            ProgressView(value: progress)
              .progressViewStyle(.linear)
          }
          .padding(.horizontal)
          .padding(.top, 8)
        case .loading:
          HStack {
            ProgressView()
              .scaleEffect(0.7)
            Text("Loading model...")
              .font(.caption)
              .foregroundStyle(.secondary)
          }
          .padding(.horizontal)
          .padding(.top, 8)
        default:
          EmptyView()
        }
      }

      HStack {
        if viewModel.generatedImage != nil {
          Button {
            viewModel.clear()
          } label: {
            Label("Clear", systemImage: "trash")
          }
        }

        Spacer()

        if case .error(let message) = viewModel.generationState {
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

        Spacer()

        if viewModel.generatedImage != nil {
          Button {
            showExportSheet = true
          } label: {
            Label("Export", systemImage: "square.and.arrow.up")
          }
          .keyboardShortcut("e", modifiers: .command)
        }

        if appState.activeGenerations[.textToImage]?.isGenerating == true {
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
          .disabled(viewModel.prompt.isEmpty)
          .keyboardShortcut("g", modifiers: .command)
        }
      }
      .padding()
    }
    .background(.regularMaterial)
  }

  // MARK: - Parameters Panel

  @ViewBuilder
  private func parametersPanelView(viewModel: TextToImageViewModel) -> some View {
    @Bindable var vm = viewModel
    ScrollView {
      VStack(alignment: .leading, spacing: 20) {
        // Prompt Section
        Section {
          VStack(alignment: .leading, spacing: 8) {
            Text("Prompt")
              .font(.subheadline.bold())

            TextEditor(text: $vm.prompt)
              .font(.body)
              .frame(height: 80)
              .scrollContentBackground(.hidden)
              .padding(8)
              .background(
                RoundedRectangle(cornerRadius: 8)
                  .fill(Color(nsColor: .textBackgroundColor))
              )
              .overlay(
                RoundedRectangle(cornerRadius: 8)
                  .strokeBorder(Color.secondary.opacity(0.2), lineWidth: 1)
              )
          }

          VStack(alignment: .leading, spacing: 8) {
            Text("Negative Prompt")
              .font(.subheadline.bold())

            TextEditor(text: $vm.negativePrompt)
              .font(.body)
              .frame(height: 50)
              .scrollContentBackground(.hidden)
              .padding(8)
              .background(
                RoundedRectangle(cornerRadius: 8)
                  .fill(Color(nsColor: .textBackgroundColor))
              )
              .overlay(
                RoundedRectangle(cornerRadius: 8)
                  .strokeBorder(Color.secondary.opacity(0.2), lineWidth: 1)
              )
          }
        } header: {
          Text("Prompt")
            .font(.headline)
        }

        Divider()

        // Size
        Section {
          VStack(alignment: .leading, spacing: 8) {
            Text("Image Size")
              .font(.subheadline.bold())

            Picker("Size", selection: $vm.width) {
              Text("512x512").tag(512)
              Text("768x768").tag(768)
              Text("1024x1024").tag(1024)
            }
            .pickerStyle(.segmented)
            .onChange(of: viewModel.width) { _, newValue in
              viewModel.height = newValue
            }
          }

          ParameterIntSlider(
            title: "Steps",
            value: $vm.steps,
            range: 1...50,
            step: 1,
            description: "More steps = higher quality"
          )

          ParameterSlider(
            title: "Guidance Scale",
            value: $vm.guidanceScale,
            range: 1...10,
            step: 0.5,
            description: "How closely to follow the prompt"
          )

          SeedInputView(
            seed: $vm.seed,
            useRandomSeed: $vm.useRandomSeed
          )
        } header: {
          Text("Generation")
            .font(.headline)
        }

        Divider()

        // Advanced Options
        CollapsibleSection(
          title: "Advanced Options",
          isExpanded: $vm.showAdvancedOptions
        ) {
          VStack(spacing: 16) {
            // True CFG Scale (always enabled)
            ParameterSlider(
              title: "True CFG Scale",
              value: $vm.trueCFGScale,
              range: 1...10,
              step: 0.5
            )

            // LoRA Selection
            VStack(alignment: .leading, spacing: 8) {
              Text("LoRA Adapter")
                .font(.subheadline.bold())

              if let loraPath = vm.selectedLoRAPath {
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

// MARK: - Image Export Document

struct ImageExportDocument: FileDocument {
  static var readableContentTypes: [UTType] { [.png, .jpeg] }

  let image: NSImage?

  init(image: NSImage?) {
    self.image = image
  }

  init(configuration: ReadConfiguration) throws {
    image = nil
  }

  func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
    guard let image = image,
          let tiffData = image.tiffRepresentation,
          let bitmap = NSBitmapImageRep(data: tiffData)
    else {
      throw CocoaError(.fileWriteUnknown)
    }

    let data: Data?
    switch configuration.contentType {
    case .jpeg:
      data = bitmap.representation(using: .jpeg, properties: [.compressionFactor: 0.9])
    default:
      data = bitmap.representation(using: .png, properties: [:])
    }

    guard let imageData = data else {
      throw CocoaError(.fileWriteUnknown)
    }

    return FileWrapper(regularFileWithContents: imageData)
  }
}

#Preview {
  TextToImageView()
    .environment(AppState())
    .frame(width: 1000, height: 700)
}
