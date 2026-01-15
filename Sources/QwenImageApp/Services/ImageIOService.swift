import AppKit
import CoreGraphics
import ImageIO
import MLX
import QwenImage

enum ImageIOService {
  static func loadImage(from url: URL) throws -> NSImage {
    guard url.startAccessingSecurityScopedResource() else {
      throw ImageIOError.accessDenied(url)
    }
    defer { url.stopAccessingSecurityScopedResource() }

    if let image = NSImage(contentsOf: url) {
      return image
    }

    if let source = CGImageSourceCreateWithURL(url as CFURL, nil),
       let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) {
      return NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
    }

    throw ImageIOError.loadFailed(url)
  }

  static func cgImage(from nsImage: NSImage) throws -> CGImage {
    var rect = NSRect(origin: .zero, size: nsImage.size)
    guard let cgImage = nsImage.cgImage(forProposedRect: &rect, context: nil, hints: nil) else {
      throw ImageIOError.conversionFailed("Failed to extract CGImage from NSImage")
    }
    return cgImage
  }

  static func cgImageToMLXArray(_ cgImage: CGImage) -> MLXArray {
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
        floatData[dstIdx] = Float(pixelData[srcIdx]) / 127.5 - 1.0
        floatData[width * height + dstIdx] = Float(pixelData[srcIdx + 1]) / 127.5 - 1.0
        floatData[2 * width * height + dstIdx] = Float(pixelData[srcIdx + 2]) / 127.5 - 1.0
        floatData[3 * width * height + dstIdx] = Float(pixelData[srcIdx + 3]) / 127.5 - 1.0
      }
    }

    let array = MLXArray(floatData, [1, 4, height, width])
    return array.asType(.bfloat16)
  }

  static func cgImageToMLXArrayRGB(_ cgImage: CGImage) -> MLXArray {
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

    var floatData = [Float](repeating: 0, count: width * height * 3)
    for y in 0..<height {
      for x in 0..<width {
        let srcIdx = y * bytesPerRow + x * bytesPerPixel
        let dstIdx = y * width + x
        floatData[dstIdx] = Float(pixelData[srcIdx]) / 127.5 - 1.0
        floatData[width * height + dstIdx] = Float(pixelData[srcIdx + 1]) / 127.5 - 1.0
        floatData[2 * width * height + dstIdx] = Float(pixelData[srcIdx + 2]) / 127.5 - 1.0
      }
    }

    let array = MLXArray(floatData, [1, 3, height, width])
    return array.asType(.bfloat16)
  }

  static func mlxArrayToNSImage(_ array: MLXArray) throws -> NSImage {
    var pixels = array
    if pixels.ndim == 4 {
      pixels = pixels.squeezed(axis: 0)
    }

    guard pixels.ndim == 3 else {
      throw ImageIOError.invalidShape("Expected 3D or 4D array, got \(pixels.ndim)D")
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
      throw ImageIOError.conversionFailed("Failed to create CGDataProvider")
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
      throw ImageIOError.conversionFailed("Failed to create CGImage from pixel data")
    }

    return NSImage(cgImage: cgImage, size: NSSize(width: width, height: height))
  }

  enum ImageFormat {
    case png
    case jpeg(quality: Float)
    case tiff

    var fileExtension: String {
      switch self {
      case .png: return "png"
      case .jpeg: return "jpg"
      case .tiff: return "tiff"
      }
    }
  }

  static func saveImage(_ image: NSImage, to url: URL, format: ImageFormat) throws {
    guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
      throw ImageIOError.conversionFailed("Failed to get CGImage for saving")
    }

    let bitmap = NSBitmapImageRep(cgImage: cgImage)

    let fileType: NSBitmapImageRep.FileType
    var properties: [NSBitmapImageRep.PropertyKey: Any] = [:]

    switch format {
    case .png:
      fileType = .png
    case .jpeg(let quality):
      fileType = .jpeg
      properties[.compressionFactor] = quality
    case .tiff:
      fileType = .tiff
    }

    guard let data = bitmap.representation(using: fileType, properties: properties) else {
      throw ImageIOError.saveFailed("Failed to encode image")
    }

    try data.write(to: url)
  }

  static func pngData(from cgImage: CGImage) throws -> Data {
    let bitmap = NSBitmapImageRep(cgImage: cgImage)
    guard let data = bitmap.representation(using: .png, properties: [:]) else {
      throw ImageIOError.conversionFailed("Failed to encode CGImage as PNG")
    }
    return data
  }

  static func exportLayers(
    _ images: [NSImage],
    to directory: URL,
    baseName: String = "layer",
    format: ImageFormat = .png
  ) throws -> [URL] {
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

    var urls: [URL] = []
    for (index, image) in images.enumerated() {
      let filename = "\(baseName)_\(index + 1).\(format.fileExtension)"
      let url = directory.appending(path: filename)
      try saveImage(image, to: url, format: format)
      urls.append(url)
    }
    return urls
  }
}

enum ImageIOError: LocalizedError {
  case accessDenied(URL)
  case loadFailed(URL)
  case conversionFailed(String)
  case saveFailed(String)
  case invalidShape(String)

  var errorDescription: String? {
    switch self {
    case .accessDenied(let url):
      return "Access denied to file: \(url.lastPathComponent)"
    case .loadFailed(let url):
      return "Failed to load image: \(url.lastPathComponent)"
    case .conversionFailed(let message):
      return "Image conversion failed: \(message)"
    case .saveFailed(let message):
      return "Failed to save image: \(message)"
    case .invalidShape(let message):
      return "Invalid image shape: \(message)"
    }
  }
}
