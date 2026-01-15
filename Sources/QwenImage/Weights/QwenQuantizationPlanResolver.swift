import Foundation
import Logging

public enum QwenQuantizationPlanResolver {
  public static func resolve(
    configURL: URL,
    snapshotRoot: URL,
    componentName: String? = nil,
    logger: Logger? = nil
  ) -> QwenQuantizationPlan? {
    var plan = QwenQuantizationPlan.load(from: configURL)
    if let manifest = QwenQuantizedSnapshotManifest.load(from: snapshotRoot) {
      var workingPlan = plan ?? QwenQuantizationPlan()
      workingPlan.registerPrepackedLayers(from: manifest, component: componentName)
      plan = workingPlan
      if let logger, let componentName {
        let layerCount = manifest.layers.filter { layer in
          guard let layerComponent = layer.component else { return true }
          return layerComponent == componentName
        }.count
        logger.info(
          "Detected pre-packed quantization manifest for \(componentName) (\(layerCount) layers, bits=\(manifest.bits), group=\(manifest.groupSize))."
        )
      }
    }
    return plan
  }

  public static func resolve(
    root: URL,
    configRelativePath: String,
    componentName: String? = nil,
    logger: Logger? = nil
  ) -> QwenQuantizationPlan? {
    resolve(
      configURL: root.appending(path: configRelativePath),
      snapshotRoot: root,
      componentName: componentName,
      logger: logger
    )
  }
}
