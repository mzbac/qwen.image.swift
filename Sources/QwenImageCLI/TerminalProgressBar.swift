import Darwin
import Foundation

final class TerminalProgressBar {
  struct Options: Sendable {
    var minUpdateInterval: TimeInterval
    var minBarWidth: Int
    var maxBarWidth: Int

    init(
      minUpdateInterval: TimeInterval = 0.1,
      minBarWidth: Int = 10,
      maxBarWidth: Int = 30
    ) {
      self.minUpdateInterval = minUpdateInterval
      self.minBarWidth = minBarWidth
      self.maxBarWidth = maxBarWidth
    }
  }

  private let enabled: Bool
  private let label: String
  private let output: FileHandle
  private let outputFD: Int32
  private let options: Options

  private var startNs: UInt64?
  private var lastStepNs: UInt64?
  private var lastRenderNs: UInt64?
  private var averageStepSeconds: Double?
  private var lastRenderedLine: String = ""
  private var renderedOnce = false
  private var finished = false

  init(
    label: String,
    enabled: Bool,
    output: FileHandle = .standardError,
    outputFD: Int32 = STDERR_FILENO,
    options: Options = .init()
  ) {
    self.label = label
    self.enabled = enabled
    self.output = output
    self.outputFD = outputFD
    self.options = options
  }

  static func defaultEnabled(outputFD: Int32 = STDERR_FILENO) -> Bool {
    guard isatty(outputFD) != 0 else { return false }
    let term = (ProcessInfo.processInfo.environment["TERM"] ?? "").lowercased()
    guard term != "dumb" else { return false }
    return true
  }

  func start(total: Int, renderInitial: Bool) {
    guard enabled, !finished else { return }
    let now = DispatchTime.now().uptimeNanoseconds
    startNs = now
    lastStepNs = now
    if renderInitial {
      render(step: 0, total: total, nowNs: now, force: true)
    }
  }

  func update(step: Int, total: Int) {
    guard enabled, !finished else { return }
    let now = DispatchTime.now().uptimeNanoseconds

    if startNs == nil {
      startNs = now
    }
    if lastStepNs == nil {
      lastStepNs = now
    }

    let clampedTotal = max(1, total)
    let clampedStep = max(0, min(step, clampedTotal))

    if clampedStep > 0, let last = lastStepNs, last < now {
      let deltaSeconds = Double(now - last) / 1_000_000_000
      if let avg = averageStepSeconds {
        averageStepSeconds = (avg * 0.9) + (deltaSeconds * 0.1)
      } else {
        averageStepSeconds = deltaSeconds
      }
      lastStepNs = now
    }

    render(step: clampedStep, total: clampedTotal, nowNs: now, force: clampedStep >= clampedTotal)

    if clampedStep >= clampedTotal {
      finish()
    }
  }

  func finish() {
    guard enabled, !finished else { return }
    finished = true
    guard renderedOnce else { return }
    writeLine(lastRenderedLine, appendNewline: true)
  }

  private func render(step: Int, total: Int, nowNs: UInt64, force: Bool) {
    guard enabled, !finished else { return }
    if !force, let last = lastRenderNs {
      let minIntervalNs = UInt64(options.minUpdateInterval * 1_000_000_000)
      if nowNs - last < minIntervalNs {
        return
      }
    }
    lastRenderNs = nowNs

    let line = makeLine(step: step, total: total, nowNs: nowNs)
    lastRenderedLine = line
    renderedOnce = true
    writeLine(line, appendNewline: false)
  }

  private func writeLine(_ line: String, appendNewline: Bool) {
    guard enabled else { return }
    let prefix = "\u{001B}[2K\r"
    let final = appendNewline ? "\(prefix)\(line)\n" : "\(prefix)\(line)"
    guard let data = final.data(using: .utf8) else { return }
    output.write(data)
  }

  private func makeLine(step: Int, total: Int, nowNs: UInt64) -> String {
    let fraction = Double(step) / Double(max(1, total))
    let percent = Int((fraction * 100.0).rounded())

    let elapsedSeconds: Double
    if let startNs {
      elapsedSeconds = Double(nowNs - startNs) / 1_000_000_000
    } else {
      elapsedSeconds = 0
    }

    let etaText: String
    if step >= 2, let avg = averageStepSeconds, avg.isFinite, avg > 0 {
      let remainingSteps = max(0, total - step)
      etaText = Self.formatDuration(seconds: Double(remainingSteps) * avg)
    } else {
      etaText = "--"
    }

    let prefix = "\(label) \(step)/\(total) "
    let suffix = " \(percent)% ETA \(etaText)"

    let columns = terminalColumns() ?? 80
    let fixedWidth = prefix.count + suffix.count
    let availableForBar = columns - fixedWidth - 2  // "[]"

    if availableForBar < options.minBarWidth {
      return "\(prefix)\(percent)% ETA \(etaText) (elapsed \(Self.formatDuration(seconds: elapsedSeconds)))"
    }

    let barWidth = min(options.maxBarWidth, max(options.minBarWidth, availableForBar))
    let filled = min(barWidth, max(0, Int(Double(barWidth) * fraction)))
    let bar = String(repeating: ">", count: filled) + String(repeating: "-", count: barWidth - filled)
    return "\(prefix)[\(bar)]\(suffix)"
  }

  private func terminalColumns() -> Int? {
    var size = winsize()
    let result = ioctl(outputFD, TIOCGWINSZ, &size)
    guard result == 0 else { return nil }
    let columns = Int(size.ws_col)
    return columns > 0 ? columns : nil
  }

  private static func formatDuration(seconds: Double) -> String {
    guard seconds.isFinite, seconds >= 0 else { return "--" }
    let totalSeconds = Int(seconds.rounded())
    if totalSeconds < 60 {
      return "\(totalSeconds)s"
    }
    let minutes = totalSeconds / 60
    let remSeconds = totalSeconds % 60
    if minutes < 60 {
      return "\(minutes)m\(String(format: "%02d", remSeconds))s"
    }
    let hours = minutes / 60
    let remMinutes = minutes % 60
    return "\(hours)h\(String(format: "%02d", remMinutes))m"
  }
}
