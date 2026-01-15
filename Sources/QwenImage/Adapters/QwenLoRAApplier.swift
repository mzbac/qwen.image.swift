import Foundation
import Logging
import MLX
import MLXNN

enum QwenLoRAApplier {
  static func apply(
    from fileURL: URL,
    scale: Float,
    computeDType: DType,
    targets: [(label: String, module: Module)],
    logger: Logger,
    emptyError: Error? = nil
  ) throws -> Int {
    let layers = try QwenLoRA.loadLayers(from: fileURL)
    if layers.isEmpty {
      logger.warning("LoRA: no adapter layers found in \(fileURL.path)")
      if let emptyError {
        throw emptyError
      }
      return 0
    }

    logger.info("LoRA: loaded \(layers.count) adapter bases from \(fileURL.lastPathComponent)")

    var total = 0
    for target in targets {
      let applied = QwenLoRA.applyLayers(
        layers,
        to: target.module,
        globalScale: scale,
        computeDType: computeDType,
        logger: logger
      )
      total += applied
      logger.info("LoRA: applied to \(target.label) (\(applied) layers updated).")
    }
    return total
  }
}

