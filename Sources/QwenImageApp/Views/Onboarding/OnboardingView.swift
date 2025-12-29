import SwiftUI

/// First-launch tutorial for new users
struct OnboardingView: View {
  @Environment(AppState.self) private var appState
  @State private var currentStep = 0
  let onComplete: () -> Void

  private let steps: [OnboardingStep] = [
    .welcome,
    .whatItDoes,
    .modelDownload,
    .lightningLoRA,
    .workflow,
    .ready
  ]

  var body: some View {
    VStack(spacing: 0) {
      // Progress indicator
      progressIndicator

      // Content
      onboardingContent

      // Navigation
      navigationButtons
    }
    .frame(width: 600, height: 500)
    .background(Color(nsColor: .windowBackgroundColor))
  }

  @ViewBuilder
  private var onboardingContent: some View {
    Group {
      switch currentStep {
      case 0:
        welcomeStep
      case 1:
        whatItDoesStep
      case 2:
        modelDownloadStep
      case 3:
        lightningLoRAStep
      case 4:
        workflowStep
      case 5:
        readyStep
      default:
        welcomeStep
      }
    }
    .animation(.easeInOut, value: currentStep)
  }

  // MARK: - Progress Indicator

  private var progressIndicator: some View {
    HStack(spacing: 8) {
      ForEach(0..<steps.count, id: \.self) { index in
        Capsule()
          .fill(index <= currentStep ? Color.accentColor : Color.secondary.opacity(0.2))
          .frame(height: 4)
      }
    }
    .padding(.horizontal, 40)
    .padding(.top, 20)
  }

  // MARK: - Steps

  private var welcomeStep: some View {
    VStack(spacing: 24) {
      Spacer()

      Image(systemName: "square.layers")
        .font(.system(size: 80))
        .foregroundStyle(.blue.gradient)

      VStack(spacing: 12) {
        Text("Welcome to Qwen Image")
          .font(.largeTitle.bold())

        Text("Extract transparent layers from any image using AI")
          .font(.title3)
          .foregroundStyle(.secondary)
          .multilineTextAlignment(.center)
      }

      Spacer()
      Spacer()
    }
    .padding(.horizontal, 60)
  }

  private var whatItDoesStep: some View {
    VStack(spacing: 24) {
      Spacer()

      HStack(spacing: 40) {
        VStack(spacing: 16) {
          Image(systemName: "photo")
            .font(.system(size: 40))
            .foregroundStyle(.secondary)
          Text("Your Image")
            .font(.caption)
            .foregroundStyle(.secondary)
        }

        Image(systemName: "arrow.right")
          .font(.title2)
          .foregroundStyle(.tertiary)

        VStack(spacing: 16) {
          Image(systemName: "square.split.2x1")
            .font(.system(size: 40))
            .foregroundStyle(.blue)
          Text("Layers")
            .font(.caption)
            .foregroundStyle(.secondary)
        }
      }

      VStack(spacing: 12) {
        Text("What It Does")
          .font(.title2.bold())

        Text("Qwen Image analyzes your image and automatically separates it into multiple transparent layers. This is perfect for:\n\n• Removing backgrounds\n• Extracting subjects\n• Compositing images\n• Graphic design projects")
          .font(.callout)
          .foregroundStyle(.secondary)
          .multilineTextAlignment(.center)
      }

      Spacer()
      Spacer()
    }
    .padding(.horizontal, 60)
  }

  private var modelDownloadStep: some View {
    VStack(spacing: 24) {
      Spacer()

      Image(systemName: "arrow.down.circle.dashed")
        .font(.system(size: 60))
        .foregroundStyle(.orange)

      VStack(spacing: 12) {
        Text("Download the Model")
          .font(.title2.bold())

        Text("Before you can extract layers, the app needs to download the Qwen-Image-Layered AI model.\n\n• Size: ~54GB\n• Location: ~/.cache/huggingface/hub\n• One-time download")
          .font(.callout)
          .foregroundStyle(.secondary)
          .multilineTextAlignment(.center)
      }

      HStack(spacing: 20) {
        FeatureBadge(
          icon: "wifi",
          title: "Stable Connection",
          description: "~35GB for 8-bit"
        )
        FeatureBadge(
          icon: "clock",
          title: "~15 minutes",
          description: "On fast internet"
        )
        FeatureBadge(
          icon: "harddrive",
          title: "54GB Storage",
          description: "For full model"
        )
      }

      Spacer()
      Spacer()
    }
    .padding(.horizontal, 40)
  }

  private var lightningLoRAStep: some View {
    VStack(spacing: 24) {
      Spacer()

      Image(systemName: "bolt.fill")
        .font(.system(size: 60))
        .foregroundStyle(.yellow.gradient)

      VStack(spacing: 12) {
        Text("Speed Boost (Optional)")
          .font(.title2.bold())

        Text("The Lightning LoRA enables blazing-fast generation in just 4 steps instead of 20+.\n\nThis is optional but recommended for faster iteration.")
          .font(.callout)
          .foregroundStyle(.secondary)
          .multilineTextAlignment(.center)
      }

      // Status based on appState
      switch appState.lightningLoRAStatus {
      case .notDownloaded:
        if kDefaultLightningLoRAPath != nil {
          installedIndicator
        } else {
          downloadButton
        }
      case .downloading(let progress, let description):
        downloadingIndicator(progress: progress, description: description)
      case .downloaded:
        installedIndicator
      case .error(let message):
        errorIndicator(message: message)
      }

      Text("You can skip this and download later from Model Manager")
        .font(.caption)
        .foregroundStyle(.tertiary)

      Spacer()
      Spacer()
    }
    .padding(.horizontal, 40)
    .onAppear {
      // Check if already installed on appear
      if kDefaultLightningLoRAPath != nil && appState.lightningLoRAStatus == .notDownloaded {
        appState.lightningLoRAStatus = .downloaded
      }
    }
  }

  @ViewBuilder
  private var installedIndicator: some View {
    HStack {
      Image(systemName: "checkmark.circle.fill")
        .foregroundStyle(.green)
      Text("Already installed!")
        .foregroundStyle(.green)
    }
    .padding()
    .background(
      RoundedRectangle(cornerRadius: 12)
        .fill(Color.green.opacity(0.1))
    )
  }

  @ViewBuilder
  private var downloadButton: some View {
    VStack(spacing: 12) {
      Button {
        Task {
          await downloadLightningLoRA()
        }
      } label: {
        Label("Download Lightning LoRA", systemImage: "arrow.down.circle")
          .frame(maxWidth: 300)
      }
      .buttonStyle(.borderedProminent)
    }
  }

  @ViewBuilder
  private func downloadingIndicator(progress: Double, description: String) -> some View {
    VStack(spacing: 8) {
      ProgressView(value: progress)
        .progressViewStyle(.linear)
        .frame(maxWidth: 300)
      
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
      .frame(maxWidth: 300)
      
      Button("Cancel") {
        cancelLightningLoRADownload()
      }
      .buttonStyle(.bordered)
    }
  }

  @ViewBuilder
  private func errorIndicator(message: String) -> some View {
    VStack(spacing: 8) {
      HStack {
        Image(systemName: "exclamationmark.triangle.fill")
          .foregroundStyle(.red)
        Text(message)
          .font(.callout)
          .foregroundStyle(.red)
      }
      
      Button("Retry") {
        Task {
          await downloadLightningLoRA()
        }
      }
      .buttonStyle(.borderedProminent)
    }
  }

  // MARK: - Lightning LoRA Download

  @State private var lightningLoRATask: Task<Void, Never>?

  private func downloadLightningLoRA() async {
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

        await MainActor.run {
          localAppState.lightningLoRAStatus = .downloaded
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

  private func cancelLightningLoRADownload() {
    lightningLoRATask?.cancel()
    lightningLoRATask = nil
    appState.lightningLoRAStatus = .notDownloaded
  }

  private var workflowStep: some View {
    VStack(spacing: 24) {
      Spacer()

      VStack(spacing: 12) {
        Text("How to Use")
          .font(.title2.bold())
      }

      VStack(spacing: 16) {
        WorkflowStep(
          number: 1,
          title: "Drop an Image",
          description: "Drag and drop any image onto the workspace"
        )
        WorkflowStep(
          number: 2,
          title: "Choose Settings",
          description: "Pick a preset or customize layers and quality"
        )
        WorkflowStep(
          number: 3,
          title: "Generate",
          description: "Click Generate to extract layers"
        )
        WorkflowStep(
          number: 4,
          title: "Export",
          description: "Save layers as PNG files"
        )
      }

      Spacer()
      Spacer()
    }
    .padding(.horizontal, 40)
  }

  private var readyStep: some View {
    VStack(spacing: 32) {
      Spacer()

      Image(systemName: "checkmark.circle.fill")
        .font(.system(size: 80))
        .foregroundStyle(.green)

      VStack(spacing: 12) {
        Text("You're Ready!")
          .font(.largeTitle.bold())

        Text("The model manager will guide you through downloading the AI model. You can access it from the sidebar anytime.")
          .font(.callout)
          .foregroundStyle(.secondary)
          .multilineTextAlignment(.center)
      }

      Button {
        Task { @MainActor in
          appState.hasCompletedOnboarding = true
          onComplete()
        }
      } label: {
        Text("Get Started")
          .font(.headline)
          .frame(maxWidth: .infinity)
          .padding()
      }
      .buttonStyle(.borderedProminent)
      .controlSize(.large)

      Spacer()
    }
    .padding(.horizontal, 60)
  }

  // MARK: - Navigation

  private var navigationButtons: some View {
    HStack {
      if currentStep > 0 {
        Button("Back") {
          withAnimation {
            currentStep -= 1
          }
        }
        .keyboardShortcut(.leftArrow, modifiers: [])
      }

      Spacer()

      if currentStep < steps.count - 1 {
        Button("Next") {
          withAnimation {
            currentStep += 1
          }
        }
        .buttonStyle(.borderedProminent)
        .keyboardShortcut(.rightArrow, modifiers: [])
      }
    }
    .padding(.horizontal, 40)
    .padding(.bottom, 20)
  }
}

// MARK: - Supporting Types

struct OnboardingStep {
  let title: String
  let description: String
  let icon: String

  static let welcome = OnboardingStep(
    title: "Welcome",
    description: "Welcome to Qwen Image",
    icon: "square.layers"
  )

  static let whatItDoes = OnboardingStep(
    title: "What It Does",
    description: "Extract layers from images",
    icon: "photo.split"
  )

  static let modelDownload = OnboardingStep(
    title: "Model Download",
    description: "Download the AI model",
    icon: "arrow.down.circle"
  )

  static let workflow = OnboardingStep(
    title: "How to Use",
    description: "Simple 4-step workflow",
    icon: "arrow.right"
  )

  static let lightningLoRA = OnboardingStep(
    title: "Speed Boost",
    description: "Optional Lightning LoRA",
    icon: "bolt.fill"
  )

  static let ready = OnboardingStep(
    title: "Ready",
    description: "Start using the app",
    icon: "checkmark.circle"
  )
}

struct FeatureBadge: View {
  let icon: String
  let title: String
  let description: String

  var body: some View {
    VStack(spacing: 8) {
      Image(systemName: icon)
        .font(.title2)
        .foregroundStyle(.secondary)

      Text(title)
        .font(.caption.bold())

      Text(description)
        .font(.caption2)
        .foregroundStyle(.tertiary)
    }
    .frame(maxWidth: 100)
  }
}

struct WorkflowStep: View {
  let number: Int
  let title: String
  let description: String

  var body: some View {
    HStack(spacing: 16) {
      ZStack {
        Circle()
          .fill(Color.accentColor)
          .frame(width: 32, height: 32)
        Text("\(number)")
          .font(.headline)
          .foregroundStyle(.white)
      }

      VStack(alignment: .leading, spacing: 2) {
        Text(title)
          .font(.subheadline.bold())
        Text(description)
          .font(.caption)
          .foregroundStyle(.secondary)
      }

      Spacer()
    }
    .padding()
    .background(
      RoundedRectangle(cornerRadius: 12)
        .fill(Color.secondary.opacity(0.05))
    )
  }
}

#Preview {
  OnboardingView(onComplete: {})
    .environment(AppState())
}
