import SwiftUI
import UniformTypeIdentifiers

@MainActor
struct EditingWorkspaceView: View {
  @Environment(AppState.self) private var appState
  @State private var showExportSheet = false
  @State private var showImagePicker = false
  @State private var showLoRAFilePicker = false

  var body: some View {
    @Bindable var viewModel = appState.editingViewModel

    HSplitView {
      mainCanvas(viewModel: viewModel)
        .frame(minWidth: 500)

      parametersPanel(viewModel: viewModel)
        .frame(width: 280)
    }
    .navigationTitle("Image Editing")
    .toolbar {
      ToolbarItemGroup(placement: .primaryAction) {
        if appState.activeGenerations[.editing]?.isGenerating == true {
          ProgressView()
            .scaleEffect(0.7)
        }
      }
    }
    .onAppear {
      let reconciled = appState.reconcileGenerationState(viewModel.generationState, for: .editing)
      if reconciled != viewModel.generationState {
        viewModel.generationState = reconciled
      }
    }
    .onReceive(NotificationCenter.default.publisher(for: .generateRequested)) { _ in
      if !viewModel.referenceImages.isEmpty && !viewModel.prompt.isEmpty && !viewModel.generationState.isGenerating {
        viewModel.generate()
      }
    }
    .onReceive(NotificationCenter.default.publisher(for: .cancelRequested)) { _ in
      viewModel.cancelGeneration()
    }
    .fileExporter(
      isPresented: $showExportSheet,
      document: ImageExportDocument(image: viewModel.editedImage),
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
        guard await appState.modelService.hasImageSession else { return }
        await appState.modelService.unloadImagePipeline()
      }
    }
  }

  private func handleImagePicker(_ result: Result<[URL], Error>, viewModel: EditingViewModel) {
    switch result {
    case .success(let urls):
      if let url = urls.first {
        do {
          let image = try ImageIOService.loadImage(from: url)
          viewModel.addReferenceImage(image)
        } catch {
          appState.showError(error.localizedDescription)
        }
      }
    case .failure(let error):
      appState.showError(error.localizedDescription)
    }
  }

  // MARK: - Main Canvas

  @ViewBuilder
  private func mainCanvas(viewModel: EditingViewModel) -> some View {
    VStack(spacing: 0) {
      HStack(spacing: 20) {
        VStack(spacing: 12) {
          Text("Reference Images")
            .font(.headline)
            .foregroundStyle(.secondary)

          ForEach(Array(viewModel.referenceImages.enumerated()), id: \.offset) { index, image in
            ReferenceImageCard(
              image: image,
              index: index,
              onRemove: { viewModel.removeReferenceImage(at: index) }
            )
          }

          if viewModel.canAddMoreReferences {
            AddReferenceButton {
              showImagePicker = true
            }
            .fileImporter(
              isPresented: $showImagePicker,
              allowedContentTypes: [.jpeg, .png, .gif, .tiff, .bmp, .webp],
              allowsMultipleSelection: false
            ) { result in
              handleImagePicker(result, viewModel: viewModel)
            }
          }

          if viewModel.referenceImages.isEmpty {
            Text("Add 1-2 reference images")
              .font(.caption)
              .foregroundStyle(.tertiary)
          }
        }
        .frame(width: 200)

        if !viewModel.referenceImages.isEmpty {
          Image(systemName: "arrow.right")
            .font(.title)
            .foregroundStyle(.tertiary)
        }

        VStack(spacing: 12) {
          Text("Result")
            .font(.headline)
            .foregroundStyle(.secondary)

          if let image = viewModel.editedImage {
            Image(nsImage: image)
              .resizable()
              .aspectRatio(contentMode: .fit)
              .frame(maxHeight: 400)
              .clipShape(RoundedRectangle(cornerRadius: 8))
              .shadow(radius: 4)
          } else {
            RoundedRectangle(cornerRadius: 8)
              .fill(Color.secondary.opacity(0.1))
              .frame(width: 300, height: 300)
              .overlay {
                VStack(spacing: 8) {
                  Image(systemName: "photo.on.rectangle.angled")
                    .font(.largeTitle)
                    .foregroundStyle(.tertiary)
                  Text("Edited image will appear here")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                }
              }
          }
        }
        .frame(maxWidth: .infinity)
      }
      .frame(maxWidth: .infinity, maxHeight: .infinity)
      .padding()

      actionBar(viewModel: viewModel)
    }
    .background(Color(nsColor: .windowBackgroundColor))
  }

  // MARK: - Action Bar

  @ViewBuilder
  private func actionBar(viewModel: EditingViewModel) -> some View {
    VStack(spacing: 0) {
      if let state = appState.activeGenerations[.editing] {
        switch state {
        case .generating(let step, let total, let progress):
          VStack(spacing: 4) {
            HStack {
              Text("Editing...")
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
        if !viewModel.referenceImages.isEmpty || viewModel.editedImage != nil {
          Button {
            viewModel.clear()
          } label: {
            Label("Clear All", systemImage: "trash")
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

        if viewModel.editedImage != nil {
          Button {
            showExportSheet = true
          } label: {
            Label("Export", systemImage: "square.and.arrow.up")
          }
          .keyboardShortcut("e", modifiers: .command)
        }

        if appState.activeGenerations[.editing]?.isGenerating == true {
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
            Label("Edit", systemImage: "sparkles")
          }
          .buttonStyle(.borderedProminent)
          .disabled(
            viewModel.referenceImages.isEmpty ||
            viewModel.prompt.isEmpty
          )
          .keyboardShortcut("g", modifiers: .command)
        }
      }
      .padding()
    }
    .background(.regularMaterial)
  }

  // MARK: - Parameters Panel

  @ViewBuilder
  private func parametersPanel(viewModel: EditingViewModel) -> some View {
    @Bindable var vm = viewModel
    ScrollView {
      VStack(alignment: .leading, spacing: 20) {
        Section {
          VStack(alignment: .leading, spacing: 8) {
            Text("Edit Prompt")
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

        Section {
          VStack(alignment: .leading, spacing: 8) {
            Text("Output Size")
              .font(.subheadline.bold())

            HStack(spacing: 8) {
              ForEach([512, 768, 1024], id: \.self) { size in
                Button {
                  vm.useCustomSize = false
                  vm.width = size
                  vm.height = size
                } label: {
                  Text("\(size)x\(size)")
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(!vm.useCustomSize && vm.width == size ? Color.accentColor : Color.secondary.opacity(0.2))
                    .foregroundColor(!vm.useCustomSize && vm.width == size ? .white : .primary)
                    .cornerRadius(6)
                }
                .buttonStyle(.plain)
              }

              Button {
                vm.useCustomSize = true
              } label: {
                Text("Custom")
                  .font(.caption)
                  .padding(.horizontal, 8)
                  .padding(.vertical, 4)
                  .background(vm.useCustomSize ? Color.accentColor : Color.secondary.opacity(0.2))
                  .foregroundColor(vm.useCustomSize ? .white : .primary)
                  .cornerRadius(6)
              }
              .buttonStyle(.plain)
            }

            if vm.useCustomSize {
              HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                  Text("Width")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                  TextField("Width", value: $vm.width, format: .number)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 80)
                }

                Text("×")
                  .foregroundStyle(.secondary)

                VStack(alignment: .leading, spacing: 4) {
                  Text("Height")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                  TextField("Height", value: $vm.height, format: .number)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 80)
                }
              }
              .padding(.top, 4)
            }
          }

          VStack(alignment: .leading, spacing: 8) {
            Text("Edit Resolution")
              .font(.subheadline.bold())

            Picker("Edit Resolution", selection: $vm.editResolution) {
              Text("512").tag(512)
              Text("1024").tag(1024)
            }
            .pickerStyle(.segmented)

            Text("Higher = more detail, slower")
              .font(.caption)
              .foregroundStyle(.tertiary)
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

        CollapsibleSection(
          title: "Advanced Options",
          isExpanded: $vm.showAdvancedOptions
        ) {
          VStack(spacing: 16) {
            ParameterSlider(
              title: "True CFG Scale",
              value: $vm.trueCFGScale,
              range: 1...10,
              step: 0.5
            )

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

// MARK: - Reference Image Card

struct ReferenceImageCard: View {
  let image: NSImage
  let index: Int
  let onRemove: () -> Void

  var body: some View {
    VStack(spacing: 4) {
      Image(nsImage: image)
        .resizable()
        .aspectRatio(contentMode: .fill)
        .frame(width: 150, height: 150)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(alignment: .topTrailing) {
          Button {
            onRemove()
          } label: {
            Image(systemName: "xmark.circle.fill")
              .foregroundStyle(.white)
              .shadow(radius: 2)
          }
          .buttonStyle(.plain)
          .padding(6)
        }

      Text("Reference \(index + 1)")
        .font(.caption)
        .foregroundStyle(.secondary)
    }
  }
}

struct AddReferenceButton: View {
  let action: () -> Void

  var body: some View {
    Button(action: action) {
      VStack(spacing: 8) {
        Image(systemName: "plus.circle")
          .font(.title)
        Text("Add Image")
          .font(.caption)
      }
      .foregroundStyle(.secondary)
      .frame(width: 150, height: 150)
      .background(
        RoundedRectangle(cornerRadius: 8)
          .strokeBorder(style: StrokeStyle(lineWidth: 2, dash: [8]))
          .foregroundStyle(.secondary.opacity(0.5))
      )
    }
    .buttonStyle(.plain)
  }
}

#Preview {
  EditingWorkspaceView()
    .environment(AppState())
    .frame(width: 1000, height: 700)
}
