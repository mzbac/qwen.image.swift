import Darwin
import Dispatch
import Foundation
import Logging
import MLX
import MLXNN
import MLXRandom
import QwenImage
import QwenImageRuntime
import Metal

private struct CLIProfiler {
  private let startTime: DispatchTime
  private var lastTime: DispatchTime
  private var lastRSSBytes: UInt64?

  init() {
    let now = DispatchTime.now()
    startTime = now
    lastTime = now
    lastRSSBytes = Self.currentResidentBytes()
  }

  mutating func mark(_ label: String, logger: Logger) {
    let now = DispatchTime.now()
    let rssBytes = Self.currentResidentBytes()
    let elapsed = Self.elapsedSeconds(from: lastTime, to: now)
    let total = Self.elapsedSeconds(from: startTime, to: now)

    var message = "[profile] \(label) (+\(Self.formatSeconds(elapsed))s, total \(Self.formatSeconds(total))s"

    if let rssBytes {
      message += ", rss \(Self.formatBytes(rssBytes))"
    }

    if let rssBytes, let lastRSSBytes {
      let delta = Int64(rssBytes) - Int64(lastRSSBytes)
      if delta != 0 {
        let sign = delta > 0 ? "+" : ""
        message += ", Δrss \(sign)\(Self.formatBytes(UInt64(abs(delta))))"
      }
    }

    message += ")"
    logger.info("\(message)")

    lastTime = now
    lastRSSBytes = rssBytes
  }

  private static func elapsedSeconds(from start: DispatchTime, to end: DispatchTime) -> Double {
    let delta = end.uptimeNanoseconds &- start.uptimeNanoseconds
    return Double(delta) / 1_000_000_000.0
  }

  private static func formatSeconds(_ value: Double) -> String {
    String(format: "%.2f", value)
  }

  private static func formatBytes(_ bytes: UInt64) -> String {
    ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory)
  }

  private static func currentResidentBytes() -> UInt64? {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.stride / MemoryLayout<integer_t>.stride)
    let kerr = withUnsafeMutablePointer(to: &info) { pointer in
      pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
        task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), rebound, &count)
      }
    }
    guard kerr == KERN_SUCCESS else {
      return nil
    }
    return UInt64(info.resident_size)
  }
}

private struct FileLogHandler: LogHandler {
  let label: String
  var metadata: Logger.Metadata = [:]
  var logLevel: Logger.Level
  private let fileHandle: FileHandle?

  init(label: String, fileURL: URL, level: Logger.Level) {
    self.label = label
    self.logLevel = level
    let fm = FileManager.default
    var handle: FileHandle? = nil
    let directory = fileURL.deletingLastPathComponent()
    try? fm.createDirectory(at: directory, withIntermediateDirectories: true)
    if !fm.fileExists(atPath: fileURL.path) {
      fm.createFile(atPath: fileURL.path, contents: nil)
    }
    if let fh = try? FileHandle(forWritingTo: fileURL) {
      fh.seekToEndOfFile()
      handle = fh
    }
    self.fileHandle = handle
  }

  subscript(metadataKey key: String) -> Logger.Metadata.Value? {
    get { metadata[key] }
    set { metadata[key] = newValue }
  }

  func log(
    level: Logger.Level,
    message: Logger.Message,
    metadata: Logger.Metadata?,
    source: String,
    file: String,
    function: String,
    line: UInt
  ) {
    guard let fileHandle else { return }
    var combined = self.metadata
    if let metadata {
      combined.merge(metadata) { _, new in new }
    }
    let metaDescription = combined.isEmpty ? "" : " \(combined)"
    let lineString = "\(Date()) [\(level)] [\(label)] \(message)\(metaDescription)\n"
    if let data = lineString.data(using: .utf8) {
      fileHandle.write(data)
    }
  }
}

private enum QwenImageCLILogging {
  static func bootstrap() {
    LoggingSystem.bootstrap { label in
      let env = ProcessInfo.processInfo.environment
      let levelString = env["QWEN_IMAGE_LOG_LEVEL"]?.lowercased() ?? "info"
      let level = Logger.Level(rawValue: levelString) ?? .info

      var stderrHandler = StreamLogHandler.standardError(label: label)
      stderrHandler.logLevel = level

      if let logPath = env["QWEN_IMAGE_LOG_FILE"] {
        let fileURL = URL(fileURLWithPath: logPath)
        let fileHandler = FileLogHandler(label: label, fileURL: fileURL, level: level)
        return MultiplexLogHandler([stderrHandler, fileHandler])
      } else {
        return stderrHandler
      }
    }
  }
}

#if canImport(AppKit)
import AppKit
#elseif canImport(UIKit)
import UIKit
#endif

struct QwenImageCLIEntry {
  static var logger: Logger = {
    var logger = Logger(label: "qwen.image.cli")
    logger.logLevel = .info
    return logger
  }()
  static func run() async throws {
    if let dev = MTLCreateSystemDefaultDevice() {
      logger.info("Metal device: \(dev.name)")
    } else {
      logger.warning("No Metal device detected; MLX may run on CPU.")
    }
    Device.setDefault(device: .gpu)
    do {
      let current = Device.defaultDevice()
      logger.info("MLX default device: \(current)")
    }
    var prompt: String?
    var negativePrompt: String?
    var steps = 30
    var guidance: Float = 4.0
    var width = 1024
    var height = 1024
    var editResolution: Int?
    var seed: UInt64?
    var modelArg: String?
    var loraArg: String?
    var revision = "main"
    var outputPath = "qwen-image.png"
    var trueCFGScale: Float?
    var referenceImagePaths: [String] = []
    var quantBits: Int?
    var quantGroupSize = 64
    var quantMode: QuantizationMode = .affine
    var quantAttnBits: Int?
    var quantAttnGroupSize: Int?
    var quantAttnMode: QuantizationMode?
    var quantAttnSpecOverride: QwenQuantizationSpec?
    var quantMinDim: Int?
    var quantizeComponents: [String] = ["transformer", "text_encoder"]
    var quantizeSnapshotSwiftOut: String?
    var dtypeOverride: DType?
    var gpuCacheLimitBytes: Int?
    var gpuCacheLimitSpecified = false
    var profileEnabled = false
    var clearCacheBetweenStages = false

    var isLayeredMode = false
    var layeredImagePath: String?
    var layeredLayers = 4
    var layeredResolution = 640
    var layeredCFGNormalize = true

    let args = CommandLine.arguments.dropFirst()
    var iterator = args.makeIterator()
    while let argument = iterator.next() {
      switch argument {
      case "--prompt", "-p":
        prompt = nextValue(for: argument, iterator: &iterator)
      case "--negative-prompt", "--np":
        negativePrompt = nextValue(for: argument, iterator: &iterator)
      case "--reference-image":
        referenceImagePaths.append(nextValue(for: argument, iterator: &iterator))
      case "--steps", "-s":
        let value = nextValue(for: argument, iterator: &iterator)
        if let intValue = Int(value) {
          steps = max(1, intValue)
        } else {
          fail("Expected integer value after \(argument)")
        }
      case "--guidance", "-g":
        let value = nextValue(for: argument, iterator: &iterator)
        if let floatValue = Float(value) {
          guidance = max(0, floatValue)
        } else {
          fail("Expected float value after \(argument)")
        }
      case "--width", "-W":
        let value = nextValue(for: argument, iterator: &iterator)
        if let intValue = Int(value) {
          width = max(16, intValue)
        } else {
          fail("Expected integer value after \(argument)")
        }
      case "--height", "-H":
        let value = nextValue(for: argument, iterator: &iterator)
        if let intValue = Int(value) {
          height = max(16, intValue)
        } else {
          fail("Expected integer value after \(argument)")
        }
      case "--edit-resolution":
        let value = nextValue(for: argument, iterator: &iterator)
        guard let intValue = Int(value), intValue == 512 || intValue == 1024 else {
          fail("Expected 512 or 1024 after --edit-resolution")
        }
        editResolution = intValue
      case "--seed":
        let value = nextValue(for: argument, iterator: &iterator)
        if let uintValue = UInt64(value) {
          seed = uintValue
        } else {
          fail("Expected integer value after \(argument)")
        }
      case "--model":
        modelArg = nextValue(for: argument, iterator: &iterator)
      case "--lora":
        loraArg = nextValue(for: argument, iterator: &iterator)
      case "--revision":
        revision = nextValue(for: argument, iterator: &iterator)
      case "--output", "-o":
        outputPath = nextValue(for: argument, iterator: &iterator)
      case "--dtype":
        let value = nextValue(for: argument, iterator: &iterator)
        guard let dtype = DType.parsingTorchDType(value), dtype == .bfloat16 else {
          fail("Unsupported --dtype \(value). Only bfloat16 is supported.")
        }
        dtypeOverride = dtype
      case "--gpu-cache-limit":
        let value = nextValue(for: argument, iterator: &iterator)
        gpuCacheLimitSpecified = true
        if value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "unlimited" {
          gpuCacheLimitBytes = nil
        } else if let parsed = parseByteSize(value) {
          gpuCacheLimitBytes = parsed
        } else {
          fail("Unsupported --gpu-cache-limit \(value). Use e.g. 16gb, 4096mb, 17179869184, or unlimited.")
        }
      case "--profile":
        profileEnabled = true
      case "--clear-cache-between-stages":
        clearCacheBetweenStages = true
      case "--true-cfg-scale":
        let value = nextValue(for: argument, iterator: &iterator)
        guard let scale = Float(value) else {
          fail("Expected float value after \(argument)")
        }
        trueCFGScale = scale
      case "--quant-bits":
        let value = nextValue(for: argument, iterator: &iterator)
        guard let bits = Int(value), [2, 4, 6, 8].contains(bits) else {
          fail("Expected 2, 4, 6, or 8 after --quant-bits")
        }
        quantBits = bits
      case "--quant-group-size":
        let value = nextValue(for: argument, iterator: &iterator)
        guard let group = Int(value), group > 0 else {
          fail("Expected positive integer after --quant-group-size")
        }
        quantGroupSize = group
      case "--quant-mode":
        let value = nextValue(for: argument, iterator: &iterator).lowercased()
        switch value {
        case "affine":
          quantMode = .affine
        case "mxfp4":
          quantMode = .mxfp4
        default:
          fail("Unsupported quantization mode \(value). Use affine or mxfp4.")
        }
      case "--quant-min-dim":
        let value = nextValue(for: argument, iterator: &iterator)
        guard let dim = Int(value), dim >= 0 else {
          fail("Expected integer >= 0 after --quant-min-dim")
        }
        quantMinDim = dim
      case "--quant-attn":
        quantAttnBits = quantAttnBits ?? quantBits ?? 8
      case "--quant-attn-bits":
        let value = nextValue(for: argument, iterator: &iterator)
        guard let bits = Int(value), [2, 4, 6, 8].contains(bits) else {
          fail("Expected 2, 4, 6, or 8 after --quant-attn-bits")
        }
        quantAttnBits = bits
      case "--quant-attn-group-size":
        let value = nextValue(for: argument, iterator: &iterator)
        guard let group = Int(value), group > 0 else {
          fail("Expected positive integer after --quant-attn-group-size")
        }
        quantAttnGroupSize = group
      case "--quant-attn-mode":
        let value = nextValue(for: argument, iterator: &iterator).lowercased()
        switch value {
        case "affine":
          quantAttnMode = .affine
        case "mxfp4":
        quantAttnMode = .mxfp4
        default:
          fail("Unsupported quantization mode \(value). Use affine or mxfp4.")
        }
      case "--quantize-model":
        quantizeSnapshotSwiftOut = nextValue(for: argument, iterator: &iterator)
      case "--quantize-components":
        let value = nextValue(for: argument, iterator: &iterator)
        quantizeComponents = value.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
      case "--layered":
        isLayeredMode = true
      case "--layered-image":
        layeredImagePath = nextValue(for: argument, iterator: &iterator)
        isLayeredMode = true
      case "--layered-layers":
        let value = nextValue(for: argument, iterator: &iterator)
        guard let intValue = Int(value), intValue > 0 else {
          fail("Expected positive integer after --layered-layers")
        }
        layeredLayers = intValue
      case "--layered-resolution":
        let value = nextValue(for: argument, iterator: &iterator)
        guard let intValue = Int(value), intValue == 640 || intValue == 1024 else {
          fail("Expected 640 or 1024 after --layered-resolution")
        }
        layeredResolution = intValue
      case "--layered-cfg-normalize":
        let value = nextValue(for: argument, iterator: &iterator).lowercased()
        layeredCFGNormalize = value == "true" || value == "1" || value == "yes"
      case "--help", "-h":
        printUsage()
        return
      default:
        continue
      }
    }

    if ProcessInfo.processInfo.environment["QWEN_IMAGE_PROFILE"] != nil {
      profileEnabled = true
    }

    if gpuCacheLimitSpecified {
      if let bytes = gpuCacheLimitBytes {
        GPUCachePolicy.setCacheLimit(bytes)
        let formatted = ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory)
        logger.info("Applied GPU cache limit: \(formatted) (system memory: \(GPUCachePolicy.totalSystemMemoryFormatted))")
      } else {
        GPUCachePolicy.applyPreset(.unlimited)
        logger.info("Applied GPU cache preset: Unlimited (system memory: \(GPUCachePolicy.totalSystemMemoryFormatted))")
      }
    } else {
      let recommendedPreset = GPUCachePolicy.recommendedPreset()
      GPUCachePolicy.applyPreset(recommendedPreset)
      logger.info("Applied GPU cache preset: \(recommendedPreset) (system memory: \(GPUCachePolicy.totalSystemMemoryFormatted))")
    }

    if quantAttnBits != nil || quantAttnGroupSize != nil || quantAttnMode != nil {
      let bits = quantAttnBits ?? quantBits ?? 8
      let group = quantAttnGroupSize ?? quantGroupSize
      let mode = quantAttnMode ?? quantMode
      quantAttnSpecOverride = QwenQuantizationSpec(groupSize: group, bits: bits, mode: mode)
    }

    let runTask = Task {
      var profiler: CLIProfiler? = profileEnabled ? CLIProfiler() : nil
      profiler?.mark("Run started", logger: logger)
      if isLayeredMode {
        guard let imagePath = layeredImagePath else {
          fail("Layered mode requires --layered-image PATH")
        }
        let env = ProcessInfo.processInfo.environment
        let hfHomePath = env["HF_HOME"].map { NSString(string: $0).expandingTildeInPath }
        let cacheOverridePath: String? = hfHomePath.map { $0 + "/hub" }

        let snapshotRoot = try await resolveSnapshot(
          model: modelArg ?? "Qwen/Qwen-Image-Layered",
          revision: revision,
          cacheDirectory: cacheOverridePath,
          hfToken: nil,
          offlineMode: false,
          useBackgroundSession: false
        )
        profiler?.mark("Snapshot resolved", logger: logger)

        try await runLayeredGeneration(
          imagePath: imagePath,
          snapshotRoot: snapshotRoot,
          outputPath: outputPath,
          layers: layeredLayers,
          resolution: layeredResolution,
          steps: steps,
          prompt: prompt,
          negativePrompt: negativePrompt,
          trueCFGScale: trueCFGScale,
          cfgNormalize: layeredCFGNormalize,
          seed: seed,
          loraPath: loraArg,
          dtypeOverride: dtypeOverride
        )
        profiler?.mark("Layered run finished", logger: logger)
        return
      }

      if let outDir = quantizeSnapshotSwiftOut {
        guard let modelValue = modelArg else {
          fail("--quantize-model requires --model to point at a local snapshot directory or HF repo id.")
        }
        let env = ProcessInfo.processInfo.environment
        let hfHomePath = env["HF_HOME"].map { NSString(string: $0).expandingTildeInPath }
        let cacheOverridePath: String? = hfHomePath.map { $0 + "/hub" }
        let resolvedSnapshot = try await resolveSnapshot(
          model: modelValue,
          revision: revision,
          cacheDirectory: cacheOverridePath,
          hfToken: nil,
          offlineMode: false,
          useBackgroundSession: false
        )
        profiler?.mark("Snapshot resolved", logger: logger)
        try runQuantizeSnapshot(
          modelPath: resolvedSnapshot.path,
          outputPath: outDir,
          components: quantizeComponents,
          bits: quantBits ?? 8,
          groupSize: quantGroupSize,
          mode: quantMode,
          profileEnabled: profileEnabled
        )
        profiler?.mark("Quantize snapshot finished", logger: logger)
        return
      }

      guard let prompt else {
        fail("Missing required --prompt")
      }
      logger.debug("Using prompt: \(prompt)")

      let env = ProcessInfo.processInfo.environment
      let hfHomePath = env["HF_HOME"].map { NSString(string: $0).expandingTildeInPath }
      let cacheOverridePath: String? = hfHomePath.map { $0 + "/hub" }

      let snapshotRoot = try await resolveSnapshot(
        model: modelArg,
        revision: revision,
        cacheDirectory: cacheOverridePath,
        hfToken: nil,
        offlineMode: false,
        useBackgroundSession: false
      )
      profiler?.mark("Snapshot resolved", logger: logger)
      if profileEnabled {
        logSnapshotFootprint(snapshotRoot: snapshotRoot, logger: logger)
      }
      let outputURL = URL(fileURLWithPath: outputPath).standardizedFileURL
      try prepareOutputDirectory(for: outputURL)

      let generation = GenerationParameters(
        prompt: prompt,
        width: width,
        height: height,
        steps: steps,
        guidanceScale: guidance,
        negativePrompt: negativePrompt,
        seed: seed,
        trueCFGScale: trueCFGScale,
        editResolution: editResolution
      )

      let isEdit = !referenceImagePaths.isEmpty
      let pipeline = QwenImagePipeline(config: isEdit ? .imageEditing : .textToImage)
      pipeline.setDTypeOverride(dtypeOverride)
      pipeline.setBaseDirectory(snapshotRoot)

      var modelConfig = QwenModelConfiguration()
      let descriptor = try QwenModelDescriptor.load(from: snapshotRoot)
      modelConfig.flowMatch = descriptor.flowMatchConfiguration
      modelConfig.requiresSigmaShift = false
      logger.info("Loaded model descriptor (\(descriptor.identity))")
      profiler?.mark("Model descriptor loaded", logger: logger)

      if let bits = quantBits {
        let spec = QwenQuantizationSpec(groupSize: quantGroupSize, bits: bits, mode: quantMode)
        let resolvedMinDim = quantMinDim ?? defaultRuntimeQuantMinDim(for: descriptor)
        pipeline.setRuntimeQuantization(spec, minInputDim: resolvedMinDim)
        logger.info(
          "Enabled runtime quantization (bits: \(bits), group: \(quantGroupSize), mode: \(quantMode), min-dim: \(resolvedMinDim))"
        )
      }

      logger.info("Loading tokenizer and text encoder from \(snapshotRoot.path)")
      try pipeline.prepareTokenizer(from: snapshotRoot, maxLength: nil)
      profiler?.mark("Tokenizer loaded", logger: logger)
      try pipeline.prepareTextEncoder(from: snapshotRoot)
      profiler?.mark("Text encoder loaded", logger: logger)
      if isEdit {
        try pipeline.prepareVAE(from: snapshotRoot)
        profiler?.mark("VAE loaded", logger: logger)
      }

      if let lora = loraArg {
        let loraURL = try await resolveLoraSafetensors(
          lora: lora,
          cacheDirectory: cacheOverridePath,
          hfToken: nil,
          offlineMode: false,
          useBackgroundSession: false
        )
        logger.info("Queuing LoRA for lazy application from \(loraURL.path)")
        pipeline.setPendingLora(from: loraURL)
        profiler?.mark("LoRA resolved", logger: logger)
      }

      if let spec = quantAttnSpecOverride {
        pipeline.setAttentionQuantization(spec)
        logger.info("Enabled attention quantization override (bits: \(spec.bits), group: \(spec.groupSize), mode: \(spec.mode)).")
      }

      let pixels: MLXArray
      if !isEdit {
        if profileEnabled || clearCacheBetweenStages {
          let promptLength = modelConfig.maxSequenceLength
          let dtype = dtypeOverride ?? descriptor.textEncoderConfiguration.outputDType
          let guidanceEncoding = try pipeline.encodeGuidancePrompts(
            prompt: generation.prompt,
            negativePrompt: generation.negativePrompt,
            maxLength: promptLength
          )
          let materialized = materializeGuidanceEncoding(guidanceEncoding, dtype: dtype)
          profiler?.mark("Prompts encoded", logger: logger)

          pipeline.releaseTextEncoder()
          profiler?.mark("Text encoder released", logger: logger)

          if clearCacheBetweenStages {
            GPUCachePolicy.clearCache()
            profiler?.mark("GPU cache cleared", logger: logger)
          }

          try pipeline.prepareUNet(from: snapshotRoot)
          profiler?.mark("UNet loaded", logger: logger)

          try pipeline.prepareVAE(from: snapshotRoot)
          profiler?.mark("VAE loaded", logger: logger)

          logger.info("Generating image")
          pixels = try await pipeline.generatePixels(
            parameters: generation,
            model: modelConfig,
            guidanceEncoding: materialized,
            seed: seed
          )
          profiler?.mark("Denoising finished", logger: logger)
        } else {
          logger.info("Generating image")
          pixels = try await pipeline.generatePixels(
            parameters: generation,
            model: modelConfig,
            seed: seed
          )
          profiler?.mark("Generation finished", logger: logger)
        }
      } else {
#if canImport(CoreGraphics)
        guard !referenceImagePaths.isEmpty else {
          fail("Edit mode requires at least one --reference-image.")
        }
        let primaryReference = loadCGImage(at: referenceImagePaths[0])
        logger.info("Using edit canvas \(generation.width)x\(generation.height)")
        logger.info("Generating image edit")
        if referenceImagePaths.count == 1 {
          pixels = try await pipeline.generateEditedPixels(
            parameters: generation,
            model: modelConfig,
            referenceImage: primaryReference,
            maxPromptLength: nil,
            seed: seed
          )
        } else {
          var images: [CGImage] = [primaryReference]
          if referenceImagePaths.count > 2 {
            logger.warning("Received \(referenceImagePaths.count) reference images; only the first two will be used.")
          }
          let second = loadCGImage(at: referenceImagePaths[1])
          images.append(second)
          pixels = try await pipeline.generateEditedPixels(
            parameters: generation,
            model: modelConfig,
            referenceImages: images,
            maxPromptLength: nil,
            seed: seed
          )
        }
        profiler?.mark("Edit generation finished", logger: logger)
#else
        fail("Native edit mode requires CoreGraphics support on this platform.")
#endif
      }

      let image = try pipeline.makeImage(from: pixels)
      try save(image: image, to: outputURL)
      logger.info("Saved image to \(outputURL.path)")
      profiler?.mark("Output saved", logger: logger)
    }

    let signalSources = installCancellationSignals(for: runTask)
    defer {
      for source in signalSources {
        source.cancel()
      }
    }

    do {
      try await runTask.value
    } catch is CancellationError {
      logger.warning("Cancelled.")
      exit(130)
    }
  }

  private static func nextValue(
    for option: String,
    iterator: inout IndexingIterator<ArraySlice<String>>
  ) -> String {
    guard let value = iterator.next() else {
      fail("Expected a value after \(option)")
    }
    return value
  }

  private static func parseByteSize(_ raw: String) -> Int? {
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return nil }
    let normalized = trimmed.replacingOccurrences(of: "_", with: "").lowercased()

    var numberPart = ""
    var suffixPart = ""

    for character in normalized {
      if character.isNumber || character == "." {
        if !suffixPart.isEmpty { return nil }
        numberPart.append(character)
      } else {
        suffixPart.append(character)
      }
    }

    guard let value = Double(numberPart) else { return nil }

    let multiplier: Double
    switch suffixPart {
    case "", "b", "bytes":
      multiplier = 1
    case "k", "kb":
      multiplier = 1024
    case "m", "mb":
      multiplier = 1024 * 1024
    case "g", "gb":
      multiplier = 1024 * 1024 * 1024
    case "t", "tb":
      multiplier = 1024 * 1024 * 1024 * 1024
    default:
      return nil
    }

    let bytes = value * multiplier
    guard bytes.isFinite, bytes > 0 else { return nil }
    return Int(bytes.rounded())
  }

  private static func directorySizeBytes(_ url: URL) -> UInt64? {
    let fm = FileManager.default
    guard let enumerator = fm.enumerator(
      at: url,
      includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
      options: [.skipsHiddenFiles]
    ) else {
      return nil
    }

    var total: UInt64 = 0
    for case let fileURL as URL in enumerator {
      do {
        let values = try fileURL.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard values.isRegularFile == true else { continue }
        total &+= UInt64(values.fileSize ?? 0)
      } catch {
        continue
      }
    }
    return total
  }

  private static func logSnapshotFootprint(snapshotRoot: URL, logger: Logger) {
    let transformer = snapshotRoot.appending(path: "transformer")
    let textEncoder = snapshotRoot.appending(path: "text_encoder")
    let vae = snapshotRoot.appending(path: "vae")

    let transformerBytes = directorySizeBytes(transformer)
    let textEncoderBytes = directorySizeBytes(textEncoder)
    let vaeBytes = directorySizeBytes(vae)

    func format(_ bytes: UInt64?) -> String {
      guard let bytes else { return "unknown" }
      return ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }

    logger.info(
      "[profile] Snapshot footprint: transformer=\(format(transformerBytes)), text_encoder=\(format(textEncoderBytes)), vae=\(format(vaeBytes))"
    )
  }

  private static func materializeGuidanceEncoding(
    _ encoding: QwenGuidanceEncoding,
    dtype: DType
  ) -> QwenGuidanceEncoding {
    let unconditionalEmbeddings = encoding.unconditionalEmbeddings.asType(dtype)
    let conditionalEmbeddings = encoding.conditionalEmbeddings.asType(dtype)
    let unconditionalMask = encoding.unconditionalMask.asType(.int32)
    let conditionalMask = encoding.conditionalMask.asType(.int32)
    MLX.eval(unconditionalEmbeddings, unconditionalMask, conditionalEmbeddings, conditionalMask)

    return QwenGuidanceEncoding(
      unconditionalEmbeddings: unconditionalEmbeddings,
      conditionalEmbeddings: conditionalEmbeddings,
      unconditionalMask: unconditionalMask,
      conditionalMask: conditionalMask,
      tokenBatch: encoding.tokenBatch
    )
  }

  private static func defaultRuntimeQuantMinDim(for descriptor: QwenModelDescriptor) -> Int {
    let maxDefault = 4096
    let textDim = descriptor.textEncoderConfiguration.hiddenSize
    let transformer = descriptor.transformerConfiguration
    let innerDim = transformer.numberOfHeads * transformer.attentionHeadDim
    return max(0, min(maxDefault, innerDim, textDim))
  }

  private static func resolveSnapshot(
    model: String?,
    revision: String,
    cacheDirectory: String?,
    hfToken: String?,
    offlineMode: Bool,
    useBackgroundSession: Bool
	  ) async throws -> URL {
	    if let model {
	      let url = URL(fileURLWithPath: model).standardizedFileURL
	      var isDirectory: ObjCBool = false
      if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) {
        guard isDirectory.boolValue else {
          fail("Model path must be a directory: \(url.path)")
        }
        return url
      }
	      let cacheURL = cacheDirectory.map { URL(fileURLWithPath: $0).standardizedFileURL }
	      let options: HubSnapshotOptions
	      do {
	        options = try QwenModelRepository.snapshotOptions(
	          repoId: model,
	          revision: revision,
	          cacheDirectory: cacheURL,
	          hfToken: hfToken,
	          offline: offlineMode,
	          useBackgroundSession: useBackgroundSession
	        )
	      } catch {
	        fail(error.localizedDescription)
	      }
	      logger.info("Resolving snapshot for \(model) (revision: \(revision))")
	      let snapshotURL = try await downloadSnapshot(options: options)
	      logger.info("Snapshot ready at \(snapshotURL.path)")
	      return snapshotURL
	    }
	    fail("Model not provided. Pass --model with a local path or HF repo id.")
	  }

  private static func resolveLoraSafetensors(
    lora: String,
    cacheDirectory: String?,
    hfToken: String?,
    offlineMode: Bool,
    useBackgroundSession: Bool
  ) async throws -> URL {
    let fm = FileManager.default
    let localURL = URL(fileURLWithPath: lora).standardizedFileURL

    var isDir: ObjCBool = false
    let exists = fm.fileExists(atPath: localURL.path, isDirectory: &isDir)
    if exists {
      if isDir.boolValue {
        let contents = try fm.contentsOfDirectory(at: localURL, includingPropertiesForKeys: nil)
        if let safetensors = contents.first(where: { $0.pathExtension == "safetensors" }) {
          return safetensors
        }
        fail("No .safetensors file found in LoRA directory \(localURL.path)")
      } else if localURL.pathExtension == "safetensors" {
        return localURL
      } else {
        fail("LoRA path \(localURL.path) is not a .safetensors file")
      }
    }

    let cacheURL = cacheDirectory.map { URL(fileURLWithPath: $0).standardizedFileURL }
    let reference = LoraReferenceParser.parse(lora)
    let repoId = reference?.repoId ?? lora
    let revision = reference?.revision ?? "main"
    let patterns = LoraReferenceParser.patterns(for: reference?.filePath)
    let options = HubSnapshotOptions(
      repoId: repoId,
      revision: revision,
      patterns: patterns,
      cacheDirectory: cacheURL,
      hfToken: hfToken,
      offline: offlineMode,
      useBackgroundSession: useBackgroundSession
    )
    logger.info("Resolving LoRA snapshot for \(repoId) (revision: \(revision))")
    let snapshotRoot = try await downloadSnapshot(options: options)

    var snapshotIsDir: ObjCBool = false
    if fm.fileExists(atPath: snapshotRoot.path, isDirectory: &snapshotIsDir) {
      if !snapshotIsDir.boolValue, snapshotRoot.pathExtension == "safetensors" {
        return snapshotRoot
      }
    }

    if let filePath = reference?.filePath {
      let trimmedPath = filePath.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
      if !trimmedPath.isEmpty {
        let directURL = snapshotRoot.appending(path: trimmedPath)
        if fm.fileExists(atPath: directURL.path) {
          return directURL
        }
        let fileName = URL(fileURLWithPath: trimmedPath).lastPathComponent
        if let match = findSafetensors(named: fileName, in: snapshotRoot) {
          return match
        }
      }
      fail("LoRA file \(filePath) not found in snapshot at \(snapshotRoot.path)")
    }

    let resourceKeys: [URLResourceKey] = [.isDirectoryKey]
    if let enumerator = fm.enumerator(
      at: snapshotRoot,
      includingPropertiesForKeys: resourceKeys,
      options: [.skipsHiddenFiles]
    ) {
      for case let url as URL in enumerator where url.pathExtension == "safetensors" {
        return url
      }
    }
    fail("No .safetensors file found in LoRA snapshot at \(snapshotRoot.path)")
  }

  private static func findSafetensors(named fileName: String, in root: URL) -> URL? {
    let resourceKeys: [URLResourceKey] = [.isDirectoryKey]
    if let enumerator = FileManager.default.enumerator(
      at: root,
      includingPropertiesForKeys: resourceKeys,
      options: [.skipsHiddenFiles]
    ) {
      for case let url as URL in enumerator where url.lastPathComponent == fileName {
        return url
      }
    }
    return nil
  }

  private static func prepareOutputDirectory(for url: URL) throws {
    let directory = url.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
  }

  private static func downloadSnapshot(options: HubSnapshotOptions) async throws -> URL {
    let snapshot = try HubSnapshot(options: options)
    let progressFlag = MutableBox(false)
    let url = try await snapshot.prepare { progress in
      guard progress.totalUnitCount > 0 else { return }
      progressFlag.value = true
      Self.printDownloadProgress(progress)
    }
    if progressFlag.value {
      let data = Data("\n".utf8)
      try? FileHandle.standardError.write(contentsOf: data)
    }
    return url
  }

  private static func printDownloadProgress(_ progress: HubSnapshotProgress) {
    let percent = progress.totalUnitCount > 0
      ? min(100, max(0, progress.fractionCompleted * 100))
      : 0
    let completed = formatByteCount(progress.completedUnitCount)
    let total = progress.totalUnitCount > 0 ? formatByteCount(progress.totalUnitCount) : "?"
    let speedString: String
    if let speed = progress.estimatedSpeedBytesPerSecond {
      speedString = " @ \(formatSpeed(speed))"
    } else {
      speedString = ""
    }
    let message = String(
      format: "[qwen.image] Downloading snapshot: %@/%@ (%.1f%%)%@",
      completed,
      total,
      percent,
      speedString
    )
    let data = Data("\r\(message)".utf8)
    try? FileHandle.standardError.write(contentsOf: data)
  }

  private static func formatByteCount(_ count: Int64) -> String {
    ByteCountFormatter.string(fromByteCount: count, countStyle: .file)
  }

  private static func formatSpeed(_ speed: Double) -> String {
    let count = Int64(speed)
    return "\(formatByteCount(count))/s"
  }

  private final class MutableBox<Value>: @unchecked Sendable {
    var value: Value
    init(_ value: Value) {
      self.value = value
    }
  }

  private static func save(image: PipelineImage, to url: URL) throws {
#if canImport(AppKit)
    guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
      fail("Failed to extract CGImage from NSImage.")
    }
    let ext = url.pathExtension.lowercased()
    let bitmap = NSBitmapImageRep(cgImage: cgImage)
    let fileType: NSBitmapImageRep.FileType
    switch ext {
    case "png": fileType = .png
    case "jpg", "jpeg": fileType = .jpeg
    case "tif", "tiff": fileType = .tiff
    case "bmp": fileType = .bmp
    default: fileType = .png
    }
    let props: [NSBitmapImageRep.PropertyKey: Any] = [:]
    guard let data = bitmap.representation(using: fileType, properties: props) else {
      fail("Failed to encode image as \(fileType)")
    }
    try data.write(to: url)
#elseif canImport(UIKit)
    guard let cgImage = image.cgImage else {
      fail("Failed to extract CGImage from UIImage.")
    }
    let uiImage = UIImage(cgImage: cgImage)
    let ext = url.pathExtension.lowercased()
    let data: Data?
    switch ext {
    case "png": data = uiImage.pngData()
    case "jpg", "jpeg": data = uiImage.jpegData(compressionQuality: 0.95)
    default: data = uiImage.pngData()
    }
    guard let final = data else { fail("Failed to encode image data.") }
    try final.write(to: url)
#else
    fail("Image saving is not supported on this platform.")
#endif
  }

#if canImport(CoreGraphics)
  private static func loadCGImage(at path: String) -> CGImage {
#if canImport(AppKit)
    guard let nsImage = NSImage(contentsOfFile: path) else {
      fail("Failed to load reference image at \(path)")
    }
    var proposedRect = NSRect(origin: .zero, size: nsImage.size)
    guard let cgImage = nsImage.cgImage(forProposedRect: &proposedRect, context: nil, hints: nil) else {
      fail("Failed to extract CGImage from reference image at \(path)")
    }
    return cgImage
#elseif canImport(UIKit)
    guard let uiImage = UIImage(contentsOfFile: path), let cgImage = uiImage.cgImage else {
      fail("Failed to load reference image at \(path)")
    }
    return cgImage
#else
    fail("Loading reference images is not supported on this platform.")
#endif
  }
#endif

  private static func printUsage() {
    let usage = """
    qwen-image-cli
      --prompt, -p            Prompt text (required for text-to-image)
      --negative-prompt, --np Negative prompt (optional)
      --steps, -s             Number of diffusion steps (default: 30)
      --guidance, -g          Guidance scale (default: 4.0)
      --true-cfg-scale        True CFG scale (matches diffusers `true_cfg_scale`)
      --dtype                 Override weight/compute dtype (bfloat16)
      --gpu-cache-limit       Override MLX GPU cache limit (e.g. 16gb, 4096mb, unlimited)
      --profile               Emit profiling checkpoints (timing + RSS)
      --clear-cache-between-stages Clear MLX GPU cache between prompt encoding and denoising
      --width, -W             Image width (default: 1024)
      --height, -H            Image height (default: 1024)
      --seed                  Random seed (optional)
      --model                 HF repo ID (e.g. Qwen/Qwen-Image-Edit-2511) or local snapshot path
      --lora                  Optional LoRA safetensors path, HF repo ID, or HF file URL (e.g. Osrivers/Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors or repo:file.safetensors)
      --revision              Snapshot revision/tag/commit (used with HF repo IDs; default: main)
      --output, -o            Output PNG path (default: qwen-image.png)

      Output format is inferred from extension: .png, .bmp, .tiff, .jpg
      Use .bmp or .tiff for faster saves during benchmarking.

      Layered image decomposition mode:
      --layered                   Enable layered decomposition mode
      --layered-image PATH        Input image path for decomposition (required for layered mode)
      --layered-layers N          Number of layers to generate (default: 4)
      --layered-resolution N      Resolution bucket: 640 or 1024 (default: 640)
      --layered-cfg-normalize     Enable/disable CFG normalization (true/false; default: true)
      Use --model to specify a custom layered model (default: Qwen/Qwen-Image-Layered)

      Example:
      qwen-image-cli --layered-image input.png --layered-layers 2 \\
        --layered-resolution 640 --steps 20 --seed 42 -o ./layers/output.png

      Quantize + save snapshot:
      --quantize-model DIR               Write a pre-packed snapshot to DIR (Swift-side; quantizes all Linear layers that pass group-size checks)
      --quantize-components LIST         Comma-separated components to quantize (default: transformer,text_encoder)
      --quant-bits                       Bit-width for snapshot weights (2, 4, 6, or 8; default: 8)
      --quant-group-size                 Group size for snapshot weights (default: 64)
      --quant-mode                       Quantization mode for snapshot weights (affine or mxfp4; default: affine)

      Edit mode:
      Pass --reference-image to enable edit mode (repeat to pass two images)

      Environment:
      QWEN_IMAGE_LOG_LEVEL   Minimum log level (debug, info, warning, error; default: info)
      QWEN_IMAGE_LOG_FILE    Optional path to a log file; if unset, logs are written to stderr only

      --help, -h              Show this help
    """
    if let data = (usage + "\n").data(using: .utf8) {
      try? FileHandle.standardOutput.write(contentsOf: data)
    }
  }

  private static func fail(_ message: String) -> Never {
    logger.error("\(message)")
    exit(2)
  }

  private static func installCancellationSignals(for task: Task<Void, Error>) -> [DispatchSourceSignal] {
    let signals: [Int32] = [SIGINT, SIGTERM]
    let queue = DispatchQueue(label: "qwen.image.cli.signals")
    var sources: [DispatchSourceSignal] = []
    sources.reserveCapacity(signals.count)

    for signalNumber in signals {
      signal(signalNumber, SIG_IGN)
      let source = DispatchSource.makeSignalSource(signal: signalNumber, queue: queue)
      source.setEventHandler {
        logger.warning("Received signal \(signalNumber); cancelling task.")
        task.cancel()
      }
      source.resume()
      sources.append(source)
    }
    return sources
  }

  private static func runLayeredGeneration(
    imagePath: String,
    snapshotRoot: URL,
    outputPath: String,
    layers: Int,
    resolution: Int,
    steps: Int,
    prompt: String?,
    negativePrompt: String?,
    trueCFGScale: Float?,
    cfgNormalize: Bool,
    seed: UInt64?,
    loraPath: String?,
    dtypeOverride: DType?
  ) async throws {
#if canImport(CoreGraphics)
    logger.info("Loading layered pipeline from \(snapshotRoot.path)")
    let pipeline = try await QwenLayeredPipeline.load(from: snapshotRoot, dtypeOverride: dtypeOverride)

    if let loraPath = loraPath {
      let loraURL = try await resolveLoraSafetensors(
        lora: loraPath,
        cacheDirectory: nil,
        hfToken: nil,
        offlineMode: false,
        useBackgroundSession: false
      )
      logger.info("Applying LoRA from \(loraURL.path)")
      try pipeline.applyLora(from: loraURL, scale: 1.0)
    }

    logger.info("Loading input image from \(imagePath)")
    let cgImage = loadCGImage(at: imagePath)
    let inputArray = cgImageToMLXArray(cgImage)

    let parameters = LayeredGenerationParameters(
      layers: layers,
      resolution: resolution,
      numInferenceSteps: steps,
      trueCFGScale: trueCFGScale ?? 4.0,
      cfgNormalize: cfgNormalize,
      prompt: prompt,
      negativePrompt: negativePrompt,
      seed: seed
    )

    logger.info("Generating \(layers) layers at \(resolution)x\(resolution) with \(steps) steps")

    let layerImages = try await pipeline.generate(
      image: inputArray,
      parameters: parameters
    ) { _, _, _ in
    }

    let outputURL = URL(fileURLWithPath: outputPath).standardizedFileURL
    let isDirectory = outputPath.hasSuffix("/") || {
      var isDir: ObjCBool = false
      return FileManager.default.fileExists(atPath: outputPath, isDirectory: &isDir) && isDir.boolValue
    }()

    let outputDir: URL
    let baseName: String
    let ext: String

    if isDirectory {
      outputDir = outputURL
      baseName = "layer"
      ext = "png"
    } else {
      outputDir = outputURL.deletingLastPathComponent()
      baseName = outputURL.deletingPathExtension().lastPathComponent
      ext = outputURL.pathExtension.isEmpty ? "png" : outputURL.pathExtension
    }

    try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

    for (i, layerArray) in layerImages.enumerated() {
      let layerIndex = i + 1
      let layerFileName = isDirectory ? "\(baseName)_\(layerIndex).\(ext)" : "\(baseName)_layer_\(layerIndex).\(ext)"
      let layerURL = outputDir.appendingPathComponent(layerFileName)

      let image = try mlxArrayToImage(layerArray)
      try save(image: image, to: layerURL)
      logger.info("Saved layer \(layerIndex) to \(layerURL.path)")
    }

    logger.info("Layered generation complete. Generated \(layerImages.count) layers.")
#else
    fail("Layered generation requires CoreGraphics support.")
#endif
  }

#if canImport(CoreGraphics)
  private static func cgImageToMLXArray(_ cgImage: CGImage) -> MLXArray {
    let width = cgImage.width
    let height = cgImage.height
    let bytesPerPixel = 4
    let bytesPerRow = width * bytesPerPixel
    let totalBytes = height * bytesPerRow

    var pixelData = [UInt8](repeating: 0, count: totalBytes)
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

    guard let context = CGContext(
      data: &pixelData,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: bytesPerRow,
      space: colorSpace,
      bitmapInfo: bitmapInfo.rawValue
    ) else {
      fatalError("Failed to create CGContext for image conversion")
    }

    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

    var floatData = [Float](repeating: 0, count: width * height * 4)
    for y in 0..<height {
      for x in 0..<width {
        let srcIdx = y * bytesPerRow + x * bytesPerPixel
        let dstIdx = y * width + x
        floatData[dstIdx] = Float(pixelData[srcIdx]) / 127.5 - 1.0  // R
        floatData[width * height + dstIdx] = Float(pixelData[srcIdx + 1]) / 127.5 - 1.0  // G
        floatData[2 * width * height + dstIdx] = Float(pixelData[srcIdx + 2]) / 127.5 - 1.0  // B
        floatData[3 * width * height + dstIdx] = Float(pixelData[srcIdx + 3]) / 127.5 - 1.0  // A
      }
    }

    let array = MLXArray(floatData, [1, 4, height, width])
    return array.asType(.bfloat16)
  }

  private static func mlxArrayToImage(_ array: MLXArray) -> PipelineImage {
    var pixels = array
    if pixels.ndim == 4 {
      pixels = pixels.squeezed(axis: 0)
    }

    pixels = pixels * 0.5 + 0.5
    pixels = MLX.clip(pixels, min: 0, max: 1)
    pixels = (pixels * 255).asType(.uint8)

    let channels = pixels.dim(0)
    let height = pixels.dim(1)
    let width = pixels.dim(2)

    pixels = pixels.transposed(1, 2, 0)

    let flatArray = pixels.reshaped([-1])
    MLX.eval(flatArray)
    let count = Int(flatArray.size)
    var byteData = [UInt8](repeating: 0, count: count)
    flatArray.asData().withUnsafeBytes { ptr in
      _ = ptr.copyBytes(to: UnsafeMutableBufferPointer(start: &byteData, count: count))
    }

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo: CGBitmapInfo
    if channels == 4 {
      bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
    } else {
      bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
    }

    let bytesPerRow = width * channels
    guard let provider = CGDataProvider(data: Data(byteData) as CFData) else {
      fatalError("Failed to create CGDataProvider")
    }

    guard let cgImage = CGImage(
      width: width,
      height: height,
      bitsPerComponent: 8,
      bitsPerPixel: 8 * channels,
      bytesPerRow: bytesPerRow,
      space: colorSpace,
      bitmapInfo: bitmapInfo,
      provider: provider,
      decode: nil,
      shouldInterpolate: false,
      intent: .defaultIntent
    ) else {
      fatalError("Failed to create CGImage from pixel data")
    }

#if canImport(AppKit)
    return NSImage(cgImage: cgImage, size: NSSize(width: width, height: height))
#elseif canImport(UIKit)
    return UIImage(cgImage: cgImage)
#endif
  }
#endif

  private static func allowedLinearLayerBases(
    snapshotRoot: URL
  ) throws -> [String: Set<String>] {
    let descriptor = try QwenModelDescriptor.load(from: snapshotRoot)

    let transformerLayers = descriptor.transformerConfiguration.numberOfLayers
    var transformerBases = Set<String>()
    transformerBases.insert("img_in")
    transformerBases.insert("txt_in")
    transformerBases.insert("time_text_embed.timestep_embedder.linear_1")
    transformerBases.insert("time_text_embed.timestep_embedder.linear_2")
    transformerBases.insert("norm_out.linear")
    transformerBases.insert("proj_out")
    for index in 0..<transformerLayers {
      let prefix = "transformer_blocks.\(index)"
      transformerBases.insert("\(prefix).img_mod.1")
      transformerBases.insert("\(prefix).txt_mod.1")
      transformerBases.insert("\(prefix).attn.to_q")
      transformerBases.insert("\(prefix).attn.to_k")
      transformerBases.insert("\(prefix).attn.to_v")
      transformerBases.insert("\(prefix).attn.add_q_proj")
      transformerBases.insert("\(prefix).attn.add_k_proj")
      transformerBases.insert("\(prefix).attn.add_v_proj")
      transformerBases.insert("\(prefix).attn.to_out.0")
      transformerBases.insert("\(prefix).attn.to_add_out")
      transformerBases.insert("\(prefix).img_mlp.net.0.proj")
      transformerBases.insert("\(prefix).img_mlp.net.2")
      transformerBases.insert("\(prefix).txt_mlp.net.0.proj")
      transformerBases.insert("\(prefix).txt_mlp.net.2")
    }

    let textLayers = descriptor.textEncoderConfiguration.numHiddenLayers
    var textEncoderBases = Set<String>()
    textEncoderBases.insert("model.embed_tokens")
    for index in 0..<textLayers {
      let prefix = "model.layers.\(index)"
      textEncoderBases.insert("\(prefix).self_attn.q_proj")
      textEncoderBases.insert("\(prefix).self_attn.k_proj")
      textEncoderBases.insert("\(prefix).self_attn.v_proj")
      textEncoderBases.insert("\(prefix).self_attn.o_proj")
      textEncoderBases.insert("\(prefix).mlp.gate_proj")
      textEncoderBases.insert("\(prefix).mlp.up_proj")
      textEncoderBases.insert("\(prefix).mlp.down_proj")
    }

    return [
      "transformer": transformerBases,
      "text_encoder": textEncoderBases
    ]
  }

  private static func runQuantizeSnapshot(
    modelPath: String,
    outputPath: String,
    components: [String],
    bits: Int,
    groupSize: Int,
    mode: QuantizationMode,
    profileEnabled: Bool = false
  ) throws {
    let snapshotURL = URL(fileURLWithPath: modelPath)
    let outputURL = URL(fileURLWithPath: outputPath)
    try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)
    let spec = QwenQuantizationSpec(groupSize: groupSize, bits: bits, mode: mode)
    logger.info("Swift quantize+save (bits=\(bits), group=\(groupSize), mode=\(mode), components=\(components.joined(separator: ",")))")
    var shardProfiler: CLIProfiler? = profileEnabled ? CLIProfiler() : nil
    let allowedMap = try allowedLinearLayerBases(snapshotRoot: snapshotURL)
    try SwiftQuantSaver.quantizeAndSave(
      from: snapshotURL,
      to: outputURL,
      components: components,
      spec: spec,
      allowedLayerMap: allowedMap,
      progress: { label in
        if profileEnabled {
          shardProfiler?.mark(label, logger: logger)
        } else {
          logger.info("\(label)")
        }
      }
    )

    let excludedDirs: Set<String> = ["transformer", "text_encoder"]
    let excludedFiles: Set<String> = ["quantization.json"]
    let fm = FileManager.default
    if let enumerator = fm.enumerator(at: snapshotURL, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles]) {
      for case let fileURL as URL in enumerator {
        let relativePath = fileURL.path.replacingOccurrences(of: snapshotURL.path + "/", with: "")
        let firstComponent = relativePath.split(separator: "/").first.map(String.init) ?? relativePath
        if excludedDirs.contains(firstComponent) || excludedFiles.contains(relativePath) {
          continue
        }
        let destURL = outputURL.appendingPathComponent(relativePath)
        let isDir = (try? fileURL.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false
        if isDir {
          try? fm.createDirectory(at: destURL, withIntermediateDirectories: true)
        } else {
          try? fm.createDirectory(at: destURL.deletingLastPathComponent(), withIntermediateDirectories: true)
          if !fm.fileExists(atPath: destURL.path) {
            try? fm.copyItem(at: fileURL, to: destURL)
          }
        }
      }
    }
    logger.info("Swift quantized snapshot written to \(outputURL.path)")
  }
}

@main
enum QwenImageCLIEntryPoint {
  static func main() async {
    QwenImageCLILogging.bootstrap()
    do {
      try await QwenImageCLIEntry.run()
    } catch {
      QwenImageCLIEntry.logger.error("Unhandled error: \(error)")
      exit(1)
    }
  }
}
