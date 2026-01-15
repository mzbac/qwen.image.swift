import SwiftUI

@MainActor
struct ContentView: View {
  @Environment(AppState.self) private var appState
  @State private var showOnboarding: Bool

  init() {
    // Initialize from settings directly
    let settings = AppSettings.load()
    _showOnboarding = State(initialValue: !settings.hasCompletedOnboarding)
  }

  var body: some View {
    @Bindable var state = appState

    NavigationSplitView {
      SidebarView()
    } detail: {
      VStack(spacing: 0) {
        if let active = appState.primaryActiveGeneration, active.state.isGenerating {
          let isAlreadyShowing: Bool = {
            guard case .mode(let selectedMode) = state.selectedSidebarItem else { return false }
            return selectedMode == active.mode
          }()

          GenerationStatusBanner(
            mode: active.mode,
            state: active.state,
            onShow: isAlreadyShowing ? nil : { state.selectedSidebarItem = .mode(active.mode) }
          )
        }
        detailView(for: state.selectedSidebarItem)
      }
    }
    .navigationSplitViewStyle(.balanced)
    .sheet(isPresented: $showOnboarding) {
      OnboardingView(onComplete: {
        showOnboarding = false
      })
      .environment(appState)
    }
    .alert("Error", isPresented: $state.showingError) {
      Button("OK", role: .cancel) {}
    } message: {
      Text(appState.errorMessage)
    }
  }

  @ViewBuilder
  private func detailView(for item: SidebarItem?) -> some View {
    switch item {
    case .mode(let mode):
      switch mode {
      case .layered:
        LayeredWorkspaceView()
      case .textToImage:
        TextToImageView()
      case .editing:
        EditingWorkspaceView()
      }
    case .modelManager:
      ModelManagerView()
    case nil:
      WelcomeView()
    }
  }
}

@MainActor
struct SidebarView: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    @Bindable var state = appState

    List(selection: $state.selectedSidebarItem) {
      Section("Generation Modes") {
        ForEach(GenerationMode.allCases) { mode in
          sidebarRow(for: mode)
            .tag(SidebarItem.mode(mode))
        }
      }

      Section {
        Label("Model Manager", systemImage: "arrow.down.circle")
          .tag(SidebarItem.modelManager)
      }
    }
    .listStyle(.sidebar)
    .navigationTitle("Qwen Image")
    .frame(minWidth: 220)
  }

  @ViewBuilder
  private func sidebarRow(for mode: GenerationMode) -> some View {
    HStack {
      Label {
        VStack(alignment: .leading, spacing: 2) {
          Text(mode.rawValue)
            .font(.headline)
          Text(mode.description)
            .font(.caption)
            .foregroundStyle(.secondary)
        }
      } icon: {
        Image(systemName: mode.icon)
          .foregroundStyle(modeColor(for: mode))
      }

      Spacer()

      if appState.isGenerating(mode: mode) {
        GeneratingIndicator()
      }
    }
  }

  private func modeColor(for mode: GenerationMode) -> Color {
    switch mode {
    case .layered: return .blue
    case .textToImage: return .purple
    case .editing: return .orange
    }
  }
}

struct GeneratingIndicator: View {
  @State private var isAnimating = false

  var body: some View {
    HStack(spacing: 4) {
      ProgressView()
        .scaleEffect(0.5)
        .frame(width: 16, height: 16)
      Text("Generating")
        .font(.caption2)
        .foregroundStyle(.secondary)
    }
    .padding(.horizontal, 6)
    .padding(.vertical, 2)
    .background(
      Capsule()
        .fill(Color.accentColor.opacity(0.15))
    )
  }
}

#Preview {
  ContentView()
    .environment(AppState())
    .frame(width: 1200, height: 800)
}
