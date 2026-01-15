import SwiftUI

struct GenerationStatusBanner: View {
  let mode: GenerationMode
  let state: GenerationState
  let onShow: (() -> Void)?

  var body: some View {
    VStack(spacing: 6) {
      HStack(spacing: 12) {
        if case .loading = state {
          ProgressView()
            .scaleEffect(0.7)
        }

        VStack(alignment: .leading, spacing: 2) {
          Text(titleText)
            .font(.subheadline.bold())

          if let detail = detailText {
            Text(detail)
              .font(.caption)
              .foregroundStyle(.secondary)
          }
        }

        Spacer()

        if let onShow {
          Button("Show") {
            onShow()
          }
          .buttonStyle(.bordered)
        }
      }

      if case .generating(_, _, let progress) = state {
        ProgressView(value: progress)
          .progressViewStyle(.linear)
      }
    }
    .padding(.horizontal)
    .padding(.vertical, 10)
    .frame(maxWidth: .infinity)
    .background(.regularMaterial)
    .overlay(Divider(), alignment: .bottom)
  }

  private var titleText: String {
    switch state {
    case .loading:
      return "Loading \(mode.rawValue) model"
    case .generating:
      return "Generating \(mode.rawValue)"
    case .complete:
      return "\(mode.rawValue) complete"
    case .error:
      return "\(mode.rawValue) error"
    case .idle:
      return mode.rawValue
    }
  }

  private var detailText: String? {
    switch state {
    case .generating(let step, let total, _):
      return "Step \(step) of \(total)"
    case .loading:
      return "Preparing pipeline"
    default:
      return nil
    }
  }
}
