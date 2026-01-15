import SwiftUI
import QwenImageRuntime

struct SettingsView: View {
  @Environment(AppState.self) private var appState

  var body: some View {
    @Bindable var state = appState

    TabView {
      GeneralSettingsTab(settings: $state.settings)
        .tabItem {
          Label("General", systemImage: "gearshape")
        }

      AdvancedSettingsTab(settings: $state.settings)
        .tabItem {
          Label("Advanced", systemImage: "slider.horizontal.3")
        }
    }
    .padding(20)
    .frame(width: 500, height: 380)
  }
}

struct GeneralSettingsTab: View {
  @Binding var settings: AppSettings

  var body: some View {
    Form {
      Section("Default Generation Settings") {
        Picker("Image Quality", selection: $settings.defaultResolution) {
          Text("Standard (640px) - Faster").tag(640)
          Text("High (1024px) - More Detail").tag(1024)
        }

        Stepper(
          "Layers: \(settings.defaultLayers)",
          value: $settings.defaultLayers,
          in: 1...8
        )
        .help("Number of layers to extract by default")

        Stepper(
          "Processing Steps: \(settings.defaultSteps)",
          value: $settings.defaultSteps,
          in: 10...100,
          step: 5
        )
        .help("More steps = higher quality but takes longer")
      }

      Section("User Interface") {
        Toggle("Show Advanced Options by Default", isOn: $settings.showAdvancedOptions)
      }
    }
    .formStyle(.grouped)
    .onChange(of: settings) { _, newValue in
      newValue.save()
    }
  }
}

struct AdvancedSettingsTab: View {
  @Binding var settings: AppSettings

  var body: some View {
    Form {
      Section("Memory Optimization") {
        Picker("GPU Cache Limit", selection: $settings.gpuCacheLimitGB) {
          Text("2GB (Conservative)").tag(2)
          Text("4GB").tag(4)
          Text("8GB").tag(8)
          Text("16GB").tag(16)
        }
        .onChange(of: settings.gpuCacheLimitGB) { _, newValue in
          let clamped = max(1, newValue)
          GPUCachePolicy.setCacheLimit(clamped * 1024 * 1024 * 1024)
          GPUCachePolicy.clearCache()
        }

        Text("Lower values reduce peak memory but may slow generation. Changes apply immediately.")
          .font(.caption)
          .foregroundStyle(.secondary)

        Toggle("Enable Quantization", isOn: $settings.quantizationEnabled)

        if settings.quantizationEnabled {
          Picker("Quantization", selection: $settings.quantizationBits) {
            Text("4-bit (Uses less memory)").tag(4)
            Text("8-bit (Better quality)").tag(8)
          }

          Text("Quantization reduces memory usage but may affect output quality.")
            .font(.caption)
            .foregroundStyle(.secondary)
        }

        Toggle("Free memory after encoding", isOn: $settings.unloadTextEncoderAfterEncoding)

        Text("Frees ~7GB VRAM after encoding. Disable to keep encoder loaded for faster prompt changes.")
          .font(.caption)
          .foregroundStyle(.secondary)
      }

      Section("Default True CFG Scale") {
        HStack {
          Slider(value: $settings.defaultCFGScale, in: 1...10, step: 0.5)
          Text(String(format: "%.1f", settings.defaultCFGScale))
            .frame(width: 40)
            .monospacedDigit()
        }

        HStack {
          Text("1.0")
            .font(.caption)
            .foregroundStyle(.secondary)
          Spacer()
          Text("10.0")
            .font(.caption)
            .foregroundStyle(.secondary)
        }

        Text("Higher values follow prompts more closely but may reduce creativity.")
          .font(.caption)
          .foregroundStyle(.secondary)
      }
    }
    .formStyle(.grouped)
    .onChange(of: settings) { _, newValue in
      newValue.save()
    }
  }
}

#Preview {
  SettingsView()
    .environment(AppState())
}
