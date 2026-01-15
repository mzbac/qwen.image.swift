import MLX
import XCTest
@testable import QwenImage

final class MLXImageResizerTests: XCTestCase {
  func testResizeNCHWProducesExpectedShape() {
    let input = MLX.zeros([1, 4, 10, 20], dtype: .float32)
    let resized = QwenMLXImageResizer.resizeNCHW(input, targetHeight: 5, targetWidth: 10)
    XCTAssertEqual(resized.shape, [1, 4, 5, 10])
    XCTAssertEqual(resized.dtype, .float32)
  }

  func testResizeCHWProducesExpectedShape() {
    let input = MLX.zeros([3, 10, 20], dtype: .bfloat16)
    let resized = QwenMLXImageResizer.resizeCHW(input, targetHeight: 8, targetWidth: 12)
    XCTAssertEqual(resized.shape, [3, 8, 12])
    XCTAssertEqual(resized.dtype, .bfloat16)
  }
}

