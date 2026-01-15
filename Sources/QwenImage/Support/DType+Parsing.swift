import Foundation
import MLX

extension DType {
  public static func parsingTorchDType(_ value: String) -> DType? {
    let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    switch normalized {
    case "bfloat16", "bf16", "torch.bfloat16":
      return .bfloat16
    case "float16", "fp16", "half", "torch.float16":
      return .float16
    case "float32", "fp32", "torch.float32":
      return .float32
    default:
      return nil
    }
  }

  public var stableName: String {
    switch self {
    case .bfloat16:
      return "bfloat16"
    case .float16:
      return "float16"
    case .float32:
      return "float32"
    default:
      return "\(self)"
    }
  }
}
