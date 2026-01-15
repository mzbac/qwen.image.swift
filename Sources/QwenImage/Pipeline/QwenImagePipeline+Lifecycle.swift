import Foundation
import Logging

extension QwenImagePipeline {
  public var isTokenizerLoaded: Bool {
    tokenizer != nil
  }

  public var isTextEncoderLoaded: Bool {
    textEncoder != nil
  }

  public var isVisionTowerLoaded: Bool {
    visionTower != nil
  }

  public var isUNetLoaded: Bool {
    unet != nil
  }

  public var isVAELoaded: Bool {
    vae != nil
  }

  public func releaseEncoders() {
    textEncoder = nil
    textEncoderQuantization = nil
    textEncoderRuntimeQuantized = false
    visionTower = nil
    visionRuntimeQuantized = false
    pipelineLogger.debug("Encoders released (text encoder + vision tower)")
  }

  public func releaseTextEncoder() {
    textEncoder = nil
    textEncoderQuantization = nil
    textEncoderRuntimeQuantized = false
    pipelineLogger.debug("Text encoder released")
  }

  public func releaseVisionTower() {
    visionTower = nil
    visionRuntimeQuantized = false
    pipelineLogger.debug("Vision tower released")
  }

  public func releaseTokenizer() {
    tokenizer = nil
    pipelineLogger.debug("Tokenizer released")
  }

  public func reloadTextEncoder() throws {
    try ensureTextEncoder()
    pipelineLogger.debug("Text encoder reloaded")
  }

  public func reloadTokenizer() throws {
    try ensureTokenizer()
    pipelineLogger.debug("Tokenizer reloaded")
  }

  func ensureTokenizer() throws {
    if tokenizer != nil { return }
    guard let directory = baseWeightsDirectory else {
      throw PipelineError.componentNotLoaded("TokenizerWeightsDirectory")
    }
    try prepareTokenizer(from: directory, maxLength: nil)
  }

  func ensureTextEncoder() throws {
    if textEncoder != nil { return }
    guard let directory = transformerDirectory ?? baseWeightsDirectory else {
      throw PipelineError.componentNotLoaded("TextEncoderWeightsDirectory")
    }
    try prepareTextEncoder(from: directory)
  }

  func ensureUNetAndVAE(model: QwenModelConfiguration) throws {
    guard let directory = transformerDirectory ?? baseWeightsDirectory else {
      throw PipelineError.componentNotLoaded("UNet/VAE weights directory")
    }
    if unet == nil {
      try prepareUNet(from: directory)
    }
    if vae == nil {
      try prepareVAE(from: directory)
    }
    if let url = pendingLoraURL {
      try applyLora(from: url, scale: pendingLoraScale)
      pendingLoraURL = nil
    }
  }
}
