import Foundation
import XCTest
@testable import QwenImage

final class LayeredQuantizationTests: XCTestCase {
  func testQuantizationPlanUsesManifestForLayeredTransformer() throws {
    let fileManager = FileManager.default
    let root = fileManager.temporaryDirectory.appending(path: UUID().uuidString)
    let transformerDir = root.appending(path: "transformer")
    try fileManager.createDirectory(at: transformerDir, withIntermediateDirectories: true)

    defer { try? fileManager.removeItem(at: root) }

    // Minimal config with 8-bit, group-32 spec
    let configURL = transformerDir.appending(path: "config.json")
    let configPayload: [String: Any] = [
      "quantization_config": [
        "bits": 8,
        "group_size": 32,
        "mode": "affine"
      ]
    ]
    let configData = try JSONSerialization.data(withJSONObject: configPayload, options: [.prettyPrinted])
    try configData.write(to: configURL)

    // Manifest that pre-packs a single layer
    let manifestURL = root.appending(path: "quantization.json")
    let manifestPayload: [String: Any] = [
      "version": 1,
      "snapshot": root.path,
      "group_size": 32,
      "bits": 8,
      "mode": "affine",
      "layers": [
        [
          "component": "transformer",
          "name": "transformer_blocks.0.img_mod.1",
          "file": "diffusion_pytorch_model-00001-of-00001.safetensors",
          "shape": [64, 64],
          "in_dim": 64,
          "out_dim": 64,
          "quant_file": "diffusion_pytorch_model-00001-of-00001.safetensors",
          "group_size": 32,
          "bits": 8,
          "mode": "affine"
        ]
      ]
    ]
    let manifestData = try JSONSerialization.data(withJSONObject: manifestPayload, options: [.prettyPrinted, .sortedKeys])
    try manifestData.write(to: manifestURL)

    let plan = try XCTUnwrap(
      QwenQuantizationPlanResolver.resolve(
        root: root,
        configRelativePath: "transformer/config.json"
      )
    )
    XCTAssertEqual(plan.defaultSpec?.groupSize, 32)
    XCTAssertEqual(plan.defaultSpec?.bits, 8)

    let layerSpec = plan.quantization(for: "transformer_blocks.0.img_mod.1")
    XCTAssertEqual(layerSpec?.groupSize, 32)
    XCTAssertEqual(layerSpec?.bits, 8)

    let prepacked = plan.prepackedLayers["transformer_blocks.0.img_mod.1"]
    XCTAssertEqual(prepacked?.spec.groupSize, 32)
    XCTAssertEqual(prepacked?.spec.bits, 8)
    XCTAssertEqual(prepacked?.quantizedFile, "diffusion_pytorch_model-00001-of-00001.safetensors")
  }

  func testLayeredTensorNameTransform() {
    XCTAssertEqual(
      QwenWeightsLoader.layeredTransformerTensorName("transformer_blocks.3.img_mlp.linear1"),
      "transformer_blocks.3.img_mlp.net.0.proj"
    )
    XCTAssertEqual(
      QwenWeightsLoader.layeredTransformerTensorName("transformer_blocks.5.txt_mlp.linear2"),
      "transformer_blocks.5.txt_mlp.net.2"
    )
    XCTAssertEqual(
      QwenWeightsLoader.layeredTransformerTensorName("transformer_blocks.1.img_mod.lin"),
      "transformer_blocks.1.img_mod.1"
    )
  }
}
