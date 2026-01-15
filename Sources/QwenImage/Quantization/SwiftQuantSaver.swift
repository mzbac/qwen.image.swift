import Foundation
import MLX
import MLXNN

public enum SwiftQuantSaverError: Error {
  case missingComponentDirectory(String, URL)
  case noSafetensorsFound(URL)
}

public struct SwiftQuantSaver {
  private static func isAllowed(base: String, allowedLayerMap: [String: Set<String>]?, component: String) -> Bool {
    guard let allowedLayerMap else { return true }
    guard let allowed = allowedLayerMap[component] else { return false }
    return allowed.contains(base)
  }

  public static func quantizeAndSave(
    from sourceRoot: URL,
    to outputRoot: URL,
    components: [String],
    spec: QwenQuantizationSpec,
    allowedLayerMap: [String: Set<String>]? = nil,
    progress: ((String) -> Void)? = nil
  ) throws {
    try FileManager.default.createDirectory(at: outputRoot, withIntermediateDirectories: true)

    var layers: [(component: String, name: String, shape: [Int], inDim: Int, outDim: Int, file: String)] = []

    for component in components {
      switch component {
      case "transformer":
        let src = sourceRoot.appending(path: "transformer")
        let dst = outputRoot.appending(path: "transformer")
        try FileManager.default.createDirectory(at: dst, withIntermediateDirectories: true)
        let fileName = try quantizeAndSaveComponent(
          componentName: "transformer",
          inputDirectory: src,
          outputDirectory: dst,
          baseFileName: "diffusion_pytorch_model.safetensors",
          spec: spec,
          allowedLayerMap: allowedLayerMap,
          progress: progress,
          record: &layers
        )
        _ = fileName
        try copyComponentMetadata(from: src, to: dst)
      case "text_encoder":
        let src = sourceRoot.appending(path: "text_encoder")
        let dst = outputRoot.appending(path: "text_encoder")
        try FileManager.default.createDirectory(at: dst, withIntermediateDirectories: true)
        let fileName = try quantizeAndSaveComponent(
          componentName: "text_encoder",
          inputDirectory: src,
          outputDirectory: dst,
          baseFileName: "model.safetensors",
          spec: spec,
          allowedLayerMap: allowedLayerMap,
          progress: progress,
          record: &layers
        )
        _ = fileName
        try copyComponentMetadata(from: src, to: dst)
      default:
        continue
      }
    }

    try writeManifest(
      at: outputRoot,
      sourceRoot: sourceRoot,
      spec: spec,
      layers: layers
    )

    try copyAncillary(from: sourceRoot, to: outputRoot)
  }

  private static func collectSafetensors(in directory: URL) throws -> [URL] {
    let fm = FileManager.default
    guard fm.fileExists(atPath: directory.path) else {
      throw SwiftQuantSaverError.missingComponentDirectory(directory.lastPathComponent, directory)
    }
    let items = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
    let files = items.filter { $0.pathExtension == "safetensors" }
    guard !files.isEmpty else { throw SwiftQuantSaverError.noSafetensorsFound(directory) }
    return files.sorted { $0.path < $1.path }
  }

  /// Copy non-safetensor metadata files (e.g., config.json) from the source component directory.
  private static func copyComponentMetadata(from src: URL, to dst: URL) throws {
    let fm = FileManager.default
    guard let entries = try? fm.contentsOfDirectory(at: src, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) else {
      return
    }
    for entry in entries {
      guard entry.pathExtension.lowercased() != "safetensors" else { continue }
      let name = entry.lastPathComponent
      // Skip manifest from source snapshot
      if name == "quantization.json" { continue }
      let dstPath = dst.appending(path: name)
      try? fm.removeItem(at: dstPath)
      try fm.copyItem(at: entry, to: dstPath)
    }
  }

  @discardableResult
  private static func quantizeAndSaveComponent(
    componentName: String,
    inputDirectory: URL,
    outputDirectory: URL,
    baseFileName: String,
    spec: QwenQuantizationSpec,
    allowedLayerMap: [String: Set<String>]?,
    progress: ((String) -> Void)?,
    record layers: inout [(component: String, name: String, shape: [Int], inDim: Int, outDim: Int, file: String)]
  ) throws -> String {
    let shardURLs = try collectSafetensors(in: inputDirectory)

    let supportedGroupSizes: Set<Int> = [32, 64, 128]
    let group = spec.groupSize
    guard supportedGroupSizes.contains(group) else { return baseFileName }

    for shardURL in shardURLs {
      let reader = try SafeTensorsReader(fileURL: shardURL)
      var tensors = try reader.loadAllTensors(as: nil)
      let shardFileName = shardURL.lastPathComponent
      progress?("Quantizing \(componentName): \(shardFileName)")

      // Quantize weights in this shard only (keeps peak memory bounded to a single safetensors file).
      let keys = tensors.keys.sorted()
      for key in keys {
        guard let tensor = tensors[key] else { continue }
        guard key.hasSuffix(".weight") else { continue }
        guard tensor.ndim == 2 else { continue }
        let outDim = tensor.dim(0)
        let inDim = tensor.dim(1)
        let base = String(key.dropLast(".weight".count))
        guard isAllowed(base: base, allowedLayerMap: allowedLayerMap, component: componentName) else { continue }
        guard inDim % group == 0 else { continue }
        guard tensor.dtype == .float32 || tensor.dtype == .bfloat16 || tensor.dtype == .float16 else { continue }

        let f = tensor.asType(.float32)
        let (wq, scales, biases) = MLX.quantized(f, groupSize: group, bits: spec.bits, mode: spec.mode)
        tensors[key] = wq
        tensors["\(base).scales"] = scales
        if let b = biases { tensors["\(base).biases"] = b }
        layers.append((componentName, base, [outDim, inDim], inDim, outDim, shardFileName))
      }

      let dst = outputDirectory.appending(path: shardFileName)
      try? FileManager.default.removeItem(at: dst)
      try MLX.save(arrays: tensors, metadata: reader.fileMetadata, url: dst)

      // Avoid cache retention across shards when running on unified memory.
      GPU.clearCache()
      progress?("Saved \(componentName): \(shardFileName)")
    }
    return baseFileName
  }

  private static func writeManifest(
    at root: URL,
    sourceRoot: URL,
    spec: QwenQuantizationSpec,
    layers: [(component: String, name: String, shape: [Int], inDim: Int, outDim: Int, file: String)]
  ) throws {
    var payload: [String: Any] = [:]
    payload["version"] = 1
    payload["group_size"] = spec.groupSize
    payload["bits"] = spec.bits
    payload["mode"] = (spec.mode == .mxfp4 ? "mxfp4" : "affine")

    var layerArray: [[String: Any]] = []
    layerArray.reserveCapacity(layers.count)
    for entry in layers {
      var rec: [String: Any] = [:]
      rec["component"] = entry.component
      rec["name"] = entry.name
      rec["file"] = entry.file
      rec["shape"] = entry.shape
      rec["in_dim"] = entry.inDim
      rec["out_dim"] = entry.outDim
      rec["quant_file"] = entry.file
      rec["group_size"] = spec.groupSize
      rec["bits"] = spec.bits
      rec["mode"] = (spec.mode == .mxfp4 ? "mxfp4" : "affine")
      layerArray.append(rec)
    }
    payload["layers"] = layerArray

    let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
    let url = root.appending(path: "quantization.json")
    try data.write(to: url)
  }

  private static func copyAncillary(from src: URL, to dst: URL) throws {
    let fm = FileManager.default
    let entries = try fm.contentsOfDirectory(atPath: src.path)
    let skipDirs: Set<String> = ["transformer", "text_encoder"]
    let copyDirs: Set<String> = ["tokenizer", "processor", "scheduler", "vae", "text_encoder_2"]
    let copyFileExts: Set<String> = ["json", "md", "txt"]
    for name in entries {
      if skipDirs.contains(name) { continue }
      var srcPath = src.appending(path: name)
      let dstPath = dst.appending(path: name)
      var isDir: ObjCBool = false
      if fm.fileExists(atPath: srcPath.path, isDirectory: &isDir) {
        if let rv = try? srcPath.resourceValues(forKeys: [.isSymbolicLinkKey]), rv.isSymbolicLink == true {
          srcPath = srcPath.resolvingSymlinksInPath()
          _ = fm.fileExists(atPath: srcPath.path, isDirectory: &isDir)
        }
        if isDir.boolValue {
          guard copyDirs.contains(name) else { continue }
          try? fm.removeItem(at: dstPath)
          try fm.copyItem(at: srcPath, to: dstPath)
        } else {
          let ext = srcPath.pathExtension.lowercased()
          guard copyFileExts.contains(ext) else { continue }
          if name == "quantization.json" { continue }
          try? fm.removeItem(at: dstPath)
          try fm.copyItem(at: srcPath, to: dstPath)
        }
      }
    }
  }
}
