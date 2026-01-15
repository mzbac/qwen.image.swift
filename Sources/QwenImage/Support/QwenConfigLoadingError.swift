import Foundation

public enum QwenConfigLoadingError: Error, LocalizedError {
  case missingFile(URL)
  case decodeFailed(URL, underlying: Error)
  case invalidValue(URL, message: String)

  public var errorDescription: String? {
    switch self {
    case .missingFile(let url):
      return "Missing required config file: \(url.path)"
    case .decodeFailed(let url, let underlying):
      return "Failed to decode config file at \(url.path): \(underlying)"
    case .invalidValue(let url, let message):
      return "Invalid config at \(url.path): \(message)"
    }
  }
}

