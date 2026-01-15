import XCTest
@testable import QwenImage

final class SchedulerFactoryTests: XCTestCase {
  func testLayeredSchedulerFactoryComputesMuFor640() {
    let flowMatchConfig = QwenFlowMatchConfig()
    let scheduler = QwenSchedulerFactory.flowMatchSchedulerForLayered(
      numInferenceSteps: 5,
      width: 640,
      height: 640,
      flowMatchConfig: flowMatchConfig
    )
    XCTAssertNotNil(scheduler.mu)
    XCTAssertEqual(scheduler.mu!, 2.5, accuracy: 1e-6)
  }

  func testLayeredSchedulerFactoryComputesMuFor1024() {
    let flowMatchConfig = QwenFlowMatchConfig()
    let scheduler = QwenSchedulerFactory.flowMatchSchedulerForLayered(
      numInferenceSteps: 5,
      width: 1024,
      height: 1024,
      flowMatchConfig: flowMatchConfig
    )
    XCTAssertNotNil(scheduler.mu)
    XCTAssertEqual(scheduler.mu!, 4.0, accuracy: 1e-6)
  }
}

