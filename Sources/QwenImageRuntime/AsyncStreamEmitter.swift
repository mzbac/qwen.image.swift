import Foundation

final class AsyncThrowingStreamEmitter<Element>: @unchecked Sendable {
  private let lock = NSLock()
  private var continuation: AsyncThrowingStream<Element, Error>.Continuation?

  init(_ continuation: AsyncThrowingStream<Element, Error>.Continuation) {
    self.continuation = continuation
  }

  func yield(_ element: Element) {
    lock.lock()
    let continuation = continuation
    lock.unlock()
    continuation?.yield(element)
  }

  func finish() {
    lock.lock()
    let continuation = continuation
    self.continuation = nil
    lock.unlock()
    continuation?.finish()
  }

  func finish(throwing error: Error) {
    lock.lock()
    let continuation = continuation
    self.continuation = nil
    lock.unlock()
    continuation?.finish(throwing: error)
  }
}

