import Foundation
import Logging
import MLX
import MLXNN

enum QwenLoRA {
  typealias LayerMap = [String: (down: MLXArray, up: MLXArray, alpha: Float)]

  static func loadLayers(from fileURL: URL) throws -> LayerMap {
    let reader = try SafeTensorsReader(fileURL: fileURL)
    var alphaTensors: [String: MLXArray] = [:]
    var downTensors: [String: MLXArray] = [:]
    var upTensors: [String: MLXArray] = [:]

    for name in reader.tensorNames {
      if name.hasSuffix(".alpha") {
        let base = String(name.dropLast(".alpha".count))
        alphaTensors[base] = try reader.tensor(named: name)
      } else if name.hasSuffix(".lora_down.weight") {
        let base = String(name.dropLast(".lora_down.weight".count))
        downTensors[base] = try reader.tensor(named: name)
      } else if name.hasSuffix(".lora_up.weight") {
        let base = String(name.dropLast(".lora_up.weight".count))
        upTensors[base] = try reader.tensor(named: name)
      }
    }

    var layers: LayerMap = [:]
    for (base, down) in downTensors {
      guard let up = upTensors[base] else { continue }
      let rank = down.dim(0)
      guard rank > 0 else { continue }

      let alphaValue: Float
      if let alphaTensor = alphaTensors[base] {
        let scalar = alphaTensor.asType(.float32)
        MLX.eval(scalar)
        alphaValue = scalar.item(Float.self)
      } else {
        alphaValue = Float(rank)
      }

      layers[base] = (down: down, up: up, alpha: alphaValue)
    }

    return layers
  }

  static func transformerBaseName(for path: String) -> String {
    var name = path
    name = name.replacingOccurrences(of: ".img_ff.mlp_in", with: ".img_mlp.net.0.proj")
    name = name.replacingOccurrences(of: ".img_ff.mlp_out", with: ".img_mlp.net.2")
    name = name.replacingOccurrences(of: ".txt_ff.mlp_in", with: ".txt_mlp.net.0.proj")
    name = name.replacingOccurrences(of: ".txt_ff.mlp_out", with: ".txt_mlp.net.2")
    name = name.replacingOccurrences(of: ".attn.attn_to_out", with: ".attn.to_out.0")
    name = name.replacingOccurrences(of: ".img_norm1.mod_linear", with: ".img_mod.1")
    name = name.replacingOccurrences(of: ".txt_norm1.mod_linear", with: ".txt_mod.1")
    name = name.replacingOccurrences(of: ".img_mlp.linear1", with: ".img_mlp.net.0.proj")
    name = name.replacingOccurrences(of: ".img_mlp.linear2", with: ".img_mlp.net.2")
    name = name.replacingOccurrences(of: ".txt_mlp.linear1", with: ".txt_mlp.net.0.proj")
    name = name.replacingOccurrences(of: ".txt_mlp.linear2", with: ".txt_mlp.net.2")
    name = name.replacingOccurrences(of: ".img_mod.lin", with: ".img_mod.1")
    name = name.replacingOccurrences(of: ".txt_mod.lin", with: ".txt_mod.1")
    return name
  }

  static func applyLayers(
    _ layers: LayerMap,
    to model: Module,
    globalScale: Float,
    computeDType: DType,
    logger: Logger? = nil
  ) -> Int {
    var layerUpdates: [String: MLXArray] = [:]
    var appliedCount = 0

    for (path, module) in model.namedModules() {
      let base = transformerBaseName(for: path)
      guard let layer = layers[base] else { continue }

      let rank = layer.down.dim(0)
      guard rank > 0 else { continue }

      let effectiveScale = globalScale * layer.alpha / Float(rank)
      let loraDown = layer.down.asType(computeDType)
      let loraUp = layer.up.asType(computeDType)
      var delta = matmul(loraUp, loraDown)
      let scaleArray = MLXArray(Float32(effectiveScale))
      delta = (delta * scaleArray).asType(computeDType)

      if let quantizedLinear = module as? QuantizedLinear {
        logger?.debug("LoRA: applying to quantized layer \(path)")
        let baseWeightFP32 = dequantized(
          quantizedLinear.weight,
          scales: quantizedLinear.scales,
          biases: quantizedLinear.biases,
          groupSize: quantizedLinear.groupSize,
          bits: quantizedLinear.bits
        )
        let baseWeight = baseWeightFP32.asType(computeDType)
        let fusedWeight = baseWeight + delta

        let fusedLinear = Linear(
          weight: fusedWeight,
          bias: quantizedLinear.bias
        )
        let requantized = QuantizedLinear(
          fusedLinear,
          groupSize: quantizedLinear.groupSize,
          bits: quantizedLinear.bits
        )

        layerUpdates["\(path).weight"] = requantized.weight
        layerUpdates["\(path).scales"] = requantized.scales
        if let biases = requantized.biases {
          layerUpdates["\(path).biases"] = biases
        }
        appliedCount += 1
      } else if let linear = module as? Linear {
        logger?.debug("LoRA: fusing into linear layer \(path)")
        let currentWeight = linear.weight.asType(.float32)
        let fusedWeight = currentWeight + delta
        let finalWeight = fusedWeight.asType(linear.weight.dtype)
        layerUpdates["\(path).weight"] = finalWeight
        appliedCount += 1
      }
    }

    if !layerUpdates.isEmpty {
      model.update(parameters: ModuleParameters.unflattened(layerUpdates))
    }
    return appliedCount
  }
}
