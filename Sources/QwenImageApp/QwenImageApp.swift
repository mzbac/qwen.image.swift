import SwiftUI
import MLX
import AppKit

@main
@MainActor
struct QwenImageApp: App {
  @State private var appState = AppState()

  init() {
    Self.configureGPULimits()
    Self.applyUserGPULimitOverride()
  }

  var body: some Scene {
    WindowGroup {
      ContentView()
        .environment(appState)
        .onAppear {
          DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            NSApp.setActivationPolicy(.regular)
            NSApp.activate(ignoringOtherApps: true)
            if let window = NSApp.windows.first {
              window.makeKeyAndOrderFront(nil)
              window.makeFirstResponder(window.contentView)
            }
          }
        }
    }
    .windowStyle(.automatic)
    .defaultSize(width: 1200, height: 800)
    .commands {
      CommandGroup(replacing: .newItem) {}
      CommandMenu("Generation") {
        Button("Generate") {
          NotificationCenter.default.post(name: .generateRequested, object: nil)
        }
        .keyboardShortcut("g", modifiers: .command)

        Button("Cancel") {
          NotificationCenter.default.post(name: .cancelRequested, object: nil)
        }
        .keyboardShortcut(.escape, modifiers: [])
      }
    }

    Settings {
      SettingsView()
        .environment(appState)
    }
  }

  private static nonisolated func configureGPULimits() {
    let systemMemory = ProcessInfo.processInfo.physicalMemory
    let oneGB: UInt64 = 1024 * 1024 * 1024

    if systemMemory < 16 * oneGB {
      MLX.GPU.set(cacheLimit: 1 * 1024 * 1024 * 1024)
    } else if systemMemory < 32 * oneGB {
      MLX.GPU.set(cacheLimit: 4 * 1024 * 1024 * 1024)
    } else if systemMemory < 64 * oneGB {
      MLX.GPU.set(cacheLimit: 8 * 1024 * 1024 * 1024)
    } else {
      MLX.GPU.set(cacheLimit: 16 * 1024 * 1024 * 1024)
    }
  }

  private static nonisolated func applyUserGPULimitOverride() {
    let settings = AppSettings.load()
    let limitGB = max(1, settings.gpuCacheLimitGB)
    MLX.GPU.set(cacheLimit: limitGB * 1024 * 1024 * 1024)
  }
}

extension Notification.Name {
  static let generateRequested = Notification.Name("generateRequested")
  static let cancelRequested = Notification.Name("cancelRequested")
}
