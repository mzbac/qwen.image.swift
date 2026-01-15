import Foundation
import Logging
import MLX
import MLXNN

extension QwenImagePipeline {
  func combinedQuantizationPlan(
    configURL: URL,
    snapshotRoot: URL,
    componentName: String
  ) -> QwenQuantizationPlan? {
    let plan = QwenQuantizationPlanResolver.resolve(
      configURL: configURL,
      snapshotRoot: snapshotRoot,
      componentName: componentName,
      logger: pipelineLogger
    )
    applyAttentionQuantization()
    return plan
  }

  func applyAttentionQuantization() {
    if let transformer {
      transformer.setAttentionQuantization(attentionQuantizationSpec)
    }
    if let unet {
      unet.transformer.setAttentionQuantization(attentionQuantizationSpec)
    }
  }

  func runtimeQuantizationAllowed(path: String) -> Bool {
    for suffix in runtimeQuantAllowedSuffixes where path.hasSuffix(suffix) {
      return true
    }
    return false
  }

  public func setRuntimeQuantization(_ spec: QwenQuantizationSpec?, minInputDim: Int = 0) {
    runtimeQuantizationSpec = spec
    runtimeQuantMinInputDim = minInputDim
    textEncoderRuntimeQuantized = false
    transformerRuntimeQuantized = false
    visionRuntimeQuantized = false
    if let spec {
      let layersDescription = runtimeQuantAllowedSuffixes.joined(separator: ", ")
      pipelineLogger.info(
        "Runtime quantization enabled (bits: \(spec.bits), group: \(spec.groupSize), mode: \(spec.mode), min-dim: \(minInputDim), layers: \(layersDescription))."
      )
      attentionQuantizationSpec = spec
    } else {
      pipelineLogger.info("Runtime quantization disabled.")
      attentionQuantizationSpec = nil
    }
    if let encoder = textEncoder {
      applyRuntimeQuantizationIfNeeded(to: encoder, flag: &textEncoderRuntimeQuantized)
    }
    if let transformer = transformer {
      applyRuntimeQuantizationIfNeeded(to: transformer, flag: &transformerRuntimeQuantized)
    } else if let unet = unet {
      applyRuntimeQuantizationIfNeeded(to: unet.transformer, flag: &transformerRuntimeQuantized)
    }
    if let tower = visionTower {
      applyRuntimeQuantizationIfNeeded(to: tower, flag: &visionRuntimeQuantized)
    }
    applyAttentionQuantization()
  }

  public func setAttentionQuantization(_ spec: QwenQuantizationSpec?) {
    attentionQuantizationSpec = spec
    applyAttentionQuantization()
  }

  func applyRuntimeQuantizationIfNeeded(
    to module: Module?,
    flag: inout Bool
  ) {
    guard let spec = runtimeQuantizationSpec, !flag, let module else { return }
    if quantizeModule(module, spec: spec) {
      flag = true
    }
  }

  private func quantizeModule(_ module: Module, spec: QwenQuantizationSpec) -> Bool {
    let supportedGroupSizes: Set<Int> = [32, 64, 128]
    guard supportedGroupSizes.contains(spec.groupSize) else {
      pipelineLogger.warning(
        "Runtime quantization skipped: group size \(spec.groupSize) is not supported (allowed: \(supportedGroupSizes.sorted()))."
      )
      return false
    }

    let shouldRestrictToAllowedSuffixes = module is QwenTransformer
    if shouldRestrictToAllowedSuffixes {
      if quantizeModule(module, spec: spec, restrictToAllowedSuffixes: true) {
        return true
      }
      pipelineLogger.warning(
        "Runtime quantization did not match any allowed layers in \(type(of: module)). Falling back to quantizing all Linear/Embedding layers."
      )
      return quantizeModule(module, spec: spec, restrictToAllowedSuffixes: false)
    }

    return quantizeModule(module, spec: spec, restrictToAllowedSuffixes: false)
  }

  private func quantizeModule(
    _ module: Module,
    spec: QwenQuantizationSpec,
    restrictToAllowedSuffixes: Bool
  ) -> Bool {
    var quantizedLayerCount = 0
    quantize(
      model: module,
      filter: { path, submodule in
        if restrictToAllowedSuffixes, !runtimeQuantizationAllowed(path: path) {
          return nil
        }
        guard let tuple = runtimeQuantizationTuple(for: submodule, spec: spec, path: path) else { return nil }
        quantizedLayerCount += 1
        return tuple
      }
    )

    guard quantizedLayerCount > 0 else {
      let scope = restrictToAllowedSuffixes ? "allowed suffix list" : "all Linear/Embedding layers"
      pipelineLogger.warning(
        "Runtime quantization skipped: no layers in \(type(of: module)) satisfied the group size requirement (scope: \(scope), min-dim: \(runtimeQuantMinInputDim))."
      )
      return false
    }

    let scope = restrictToAllowedSuffixes ? "allowed suffix list" : "all Linear/Embedding layers"
    pipelineLogger.info(
      "Runtime quantized \(quantizedLayerCount) layers in \(type(of: module)) (\(spec.bits)-bit, group size \(spec.groupSize), mode \(spec.mode), scope: \(scope))."
    )
    return true
  }

  private func runtimeQuantizationTuple(
    for module: Module,
    spec: QwenQuantizationSpec,
    path: String
  ) -> (groupSize: Int, bits: Int, mode: QuantizationMode)? {
    if let linear = module as? Linear {
      let inputDims = linear.weight.dim(1)
      guard inputDims % spec.groupSize == 0 else { return nil }
      guard inputDims >= runtimeQuantMinInputDim else { return nil }
      return spec.tuple
    }
    if let embedding = module as? Embedding {
      let dim = embedding.weight.dim(1)
      guard dim % spec.groupSize == 0 else { return nil }
      guard dim >= runtimeQuantMinInputDim else { return nil }
      return spec.tuple
    }
    return nil
  }
}
