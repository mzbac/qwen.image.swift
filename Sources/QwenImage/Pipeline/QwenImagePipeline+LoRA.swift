import Foundation
import Logging
import MLXNN

extension QwenImagePipeline {
  public func setPendingLora(from url: URL, scale: Float = 1.0) {
    pendingLoraURL = url
    pendingLoraScale = scale
  }

  public func applyLora(from fileURL: URL, scale: Float = 1.0) throws {
    guard let unet else {
      throw PipelineError.componentNotLoaded("UNet")
    }
    var targets: [(label: String, module: Module)] = [("UNet transformer", unet.transformer)]
    if let transformer {
      targets.append(("standalone transformer", transformer))
    }

    _ = try QwenLoRAApplier.apply(
      from: fileURL,
      scale: scale,
      computeDType: preferredWeightDType(),
      targets: targets,
      logger: pipelineLogger,
      emptyError: PipelineError.invalidTensorShape(
        "LoRA file \(fileURL.lastPathComponent) contained no recognized transformer blocks."
      )
    )
  }
}
