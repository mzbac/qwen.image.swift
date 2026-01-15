import SwiftUI

struct WelcomeView: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    VStack(spacing: 32) {
      Spacer()

      // App Icon / Logo
      Image(systemName: "photo.stack")
        .font(.system(size: 80))
        .foregroundStyle(.blue.gradient)

      // Title
      VStack(spacing: 8) {
        Text("Qwen Image")
          .font(.largeTitle.bold())
        Text("Generate, Edit, and Decompose Images")
          .font(.title3)
          .foregroundStyle(.secondary)
      }

      // Mode Cards
      VStack(spacing: 16) {
        ForEach(GenerationMode.allCases) { mode in
          ModeCard(mode: mode)
        }
      }
      .padding(.horizontal, 40)

      // Lightning LoRA Suggestion (if not installed)
      if kDefaultLightningLoRAPath == nil && appState.lightningLoRAStatus != .downloaded {
        LightningLoRASuggestion()
      }

      Spacer()

      // Model Status Footer
      ModelStatusFooter()
        .padding(.bottom, 20)
    }
    .frame(maxWidth: .infinity, maxHeight: .infinity)
    .background(Color(nsColor: .windowBackgroundColor))
  }
}

struct ModeCard: View {
  @Environment(AppState.self) private var appState
  let mode: GenerationMode

  var body: some View {
    @Bindable var state = appState

    Button {
      if appState.isModelReadyFor(mode: mode) {
        appState.selectedSidebarItem = .mode(mode)
      }
    } label: {
      HStack(spacing: 16) {
        Image(systemName: mode.icon)
          .font(.system(size: 32))
          .foregroundStyle(modeColor)
          .frame(width: 50)

        VStack(alignment: .leading, spacing: 4) {
          Text(mode.rawValue)
            .font(.headline)
            .foregroundStyle(.primary)
          Text(mode.description)
            .font(.subheadline)
            .foregroundStyle(.secondary)
        }

        Spacer()

        if appState.isModelReadyFor(mode: mode) {
          Image(systemName: "chevron.right")
            .foregroundStyle(.tertiary)
        } else {
          Text("Setup Required")
            .font(.caption)
            .foregroundStyle(.orange)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(.orange.opacity(0.15), in: Capsule())
        }
      }
      .padding()
      .frame(maxWidth: 500)
      .background(
        RoundedRectangle(cornerRadius: 12)
          .fill(Color(nsColor: .controlBackgroundColor))
      )
      .overlay(
        RoundedRectangle(cornerRadius: 12)
          .strokeBorder(Color.secondary.opacity(0.2), lineWidth: 1)
      )
    }
    .buttonStyle(.plain)
    .disabled(!appState.isModelReadyFor(mode: mode))
  }

  private var modeColor: Color {
    switch mode {
    case .layered: return .blue
    case .textToImage: return .purple
    case .editing: return .orange
    }
  }
}

struct ModelStatusFooter: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    HStack(spacing: 20) {
      ForEach(ModelDefinition.all, id: \.id) { model in
        ModelStatusPill(model: model, status: appState.status(for: model))
      }
    }
    .padding(.horizontal)
  }
}

struct LightningLoRASuggestion: View {
  @Environment(AppState.self) private var appState
  @AppStorage("hideLightningLoRASuggestion") private var hideSuggestion = false
  
  var body: some View {
    if !hideSuggestion {
      HStack(spacing: 12) {
        Image(systemName: "bolt.fill")
          .foregroundStyle(.yellow)
        
        VStack(alignment: .leading, spacing: 2) {
          Text("Speed up generation with Lightning LoRA")
            .font(.caption.bold())
          Text("Generate in 4 steps instead of 20+")
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
        
        Spacer()
        
        Button {
          appState.selectedSidebarItem = .modelManager
        } label: {
          Text("Get it")
            .font(.caption.bold())
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.small)
        
        Button {
          hideSuggestion = true
        } label: {
          Image(systemName: "xmark")
            .font(.caption2)
        }
        .buttonStyle(.plain)
        .foregroundStyle(.secondary)
      }
      .padding(.horizontal, 16)
      .padding(.vertical, 10)
      .frame(maxWidth: 500)
      .background(
        RoundedRectangle(cornerRadius: 10)
          .fill(Color.yellow.opacity(0.1))
      )
      .overlay(
        RoundedRectangle(cornerRadius: 10)
          .strokeBorder(Color.yellow.opacity(0.3), lineWidth: 1)
      )
    }
  }
}

struct ModelStatusPill: View {
  let model: ModelDefinition
  let status: ModelStatus

  var body: some View {
    HStack(spacing: 6) {
      statusIcon
        .foregroundStyle(statusColor)

      Text(model.name)
        .font(.caption)
        .foregroundStyle(.secondary)
    }
    .padding(.horizontal, 10)
    .padding(.vertical, 6)
    .background(
      Capsule()
        .fill(statusColor.opacity(0.1))
    )
  }

  @ViewBuilder
  private var statusIcon: some View {
    switch status {
    case .notDownloaded:
      Image(systemName: "arrow.down.circle")
    case .downloading:
      ProgressView()
        .scaleEffect(0.6)
    case .downloaded, .ready:
      Image(systemName: "checkmark.circle.fill")
    case .loading:
      ProgressView()
        .scaleEffect(0.6)
    case .error:
      Image(systemName: "exclamationmark.circle.fill")
    }
  }

  private var statusColor: Color {
    switch status {
    case .notDownloaded:
      return .secondary
    case .downloading, .loading:
      return .blue
    case .downloaded, .ready:
      return .green
    case .error:
      return .red
    }
  }
}

#Preview {
  WelcomeView()
    .environment(AppState())
    .frame(width: 900, height: 700)
}
