import Foundation

public enum QwenImageRuntimeError: Error, LocalizedError {
  case loraAlreadyApplied(appliedURL: URL, appliedScale: Float, requestedURL: URL, requestedScale: Float)

  public var errorDescription: String? {
    switch self {
    case .loraAlreadyApplied(let appliedURL, let appliedScale, let requestedURL, let requestedScale):
      if appliedURL == requestedURL {
        return "LoRA '\(appliedURL.lastPathComponent)' is already applied with scale \(appliedScale). Reload the session to apply scale \(requestedScale)."
      }
      return "LoRA '\(appliedURL.lastPathComponent)' is already applied. Reload the session to apply '\(requestedURL.lastPathComponent)'."
    }
  }
}

