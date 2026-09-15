import AppKit
import AVFoundation
import CoreImage
import ScreenCaptureKit

final class RecordingObserver: NSObject, SCRecordingOutputDelegate, SCStreamOutput, @unchecked Sendable {
  private let lock = NSLock()
  private var started = false
  private var finished = false
  private var failure: String?
  private var firstPTS: Double?
  private var latestBuffer: CVPixelBuffer?
  private var latestPTS: Double = 0
  private var latestHash: UInt64 = 0
  private var changedAt: Double = 0

  func state() -> (Bool, Bool, String?, Double?) {
    lock.lock(); defer { lock.unlock() }
    return (started, finished, failure, firstPTS)
  }

  func recordingOutputDidStartRecording(_: SCRecordingOutput) {
    lock.lock(); started = true; lock.unlock()
  }

  func recordingOutputDidFinishRecording(_: SCRecordingOutput) {
    lock.lock(); finished = true; lock.unlock()
  }

  func recordingOutput(_: SCRecordingOutput, didFailWithError error: Error) {
    lock.lock(); failure = String(describing: error); lock.unlock()
  }

  func stream(_: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
    guard type == .screen, sampleBuffer.isValid,
          let attachments = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]],
          attachments.first?[.status] as? Int == SCFrameStatus.complete.rawValue else { return }
    lock.lock()
    if firstPTS == nil { firstPTS = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds }
    latestBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)
    latestPTS = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
    if let buffer = latestBuffer {
      CVPixelBufferLockBaseAddress(buffer, .readOnly)
      if let base = CVPixelBufferGetBaseAddress(buffer) {
        let rows = CVPixelBufferGetHeight(buffer), stride = CVPixelBufferGetBytesPerRow(buffer)
        let bytes = base.assumingMemoryBound(to: UInt8.self)
        var hash: UInt64 = 1_469_598_103_934_665_603
        for y in Swift.stride(from: 0, to: rows, by: max(1, rows / 64)) {
          for x in Swift.stride(from: 0, to: stride, by: max(1, stride / 128)) {
            hash = (hash ^ UInt64(bytes[y * stride + x])) &* 1_099_511_628_211
          }
        }
        if hash != latestHash { changedAt = CMClockGetTime(CMClockGetHostTimeClock()).seconds; latestHash = hash }
      }
      CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
    }
    lock.unlock()
  }

  func isSettled() -> Bool {
    lock.lock(); defer { lock.unlock() }
    return latestBuffer != nil && CMClockGetTime(CMClockGetHostTimeClock()).seconds - changedAt >= 0.2
  }

  func screenshot(path: String) throws -> [String: Any] {
    lock.lock()
    let buffer = latestBuffer
    let time = latestPTS - (firstPTS ?? latestPTS)
    lock.unlock()
    guard let buffer else { throw DriverFailure("No video frame is available for a screenshot") }
    let image = CIImage(cvPixelBuffer: buffer)
    guard let cg = CIContext().createCGImage(image, from: image.extent),
          let png = NSBitmapImageRep(cgImage: cg).representation(using: .png, properties: [:]) else { throw DriverFailure("Video-frame PNG encoding failed") }
    try png.write(to: URL(fileURLWithPath: path), options: .atomic)
    return ["event": "screenshot", "width": cg.width, "height": cg.height, "bytes": png.count, "frameTime": time]
  }
}

func emitJSON(_ value: [String: Any]) throws {
  try FileHandle.standardOutput.write(JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]) + Data("\n".utf8))
}

func readRecorderLine() async -> String? {
  await withCheckedContinuation { continuation in
    DispatchQueue.global().async { continuation.resume(returning: readLine()) }
  }
}

extension Driver {
  @MainActor func recordVideo(path: String) async throws {
    guard CGPreflightScreenCaptureAccess() else { throw DriverFailure("Screen Recording permission is missing") }
    let content = try await SCShareableContent.excludingDesktopWindows(true, onScreenWindowsOnly: true)
    let windows = content.windows.filter { $0.owningApplication?.processID == app.processIdentifier && $0.windowLayer == 0 && $0.title == "Mentra" }
    guard windows.count == 1 else { throw DriverFailure("Cannot identify a unique Mentra recording window") }
    let target = windows[0]
    guard let display = content.displays.first(where: { $0.frame.contains(CGPoint(x: target.frame.midX, y: target.frame.midY)) }) else { throw DriverFailure("Target display not found") }
    // Display stream with an explicit window allowlist survives target relaunch.
    // Every other window, the desktop and Dock are excluded by this filter.
    let filter = SCContentFilter(display: display, including: [target])
    let configuration = SCStreamConfiguration()
    configuration.sourceRect = target.frame.offsetBy(dx: -display.frame.minX, dy: -display.frame.minY)
    configuration.width = Int(target.frame.width * 2) / 2 * 2
    configuration.height = Int(target.frame.height * 2) / 2 * 2
    configuration.minimumFrameInterval = CMTime(value: 1, timescale: 15)
    configuration.capturesAudio = false
    configuration.captureMicrophone = false
    configuration.showsCursor = false
    configuration.includeChildWindows = true
    let observer = RecordingObserver()
    let stream = SCStream(filter: filter, configuration: configuration, delegate: nil)
    try stream.addStreamOutput(observer, type: .screen, sampleHandlerQueue: DispatchQueue(label: "mentra.e2e.frames"))
    let recordingConfig = SCRecordingOutputConfiguration()
    recordingConfig.outputURL = URL(fileURLWithPath: path)
    recordingConfig.videoCodecType = .h264
    recordingConfig.outputFileType = .mp4
    let recording = SCRecordingOutput(configuration: recordingConfig, delegate: observer)
    try stream.addRecordingOutput(recording)
    try await stream.startCapture()
    let deadline = Date().addingTimeInterval(10)
    while !observer.state().0 || observer.state().3 == nil, Date() < deadline {
      if let error = observer.state().2 { throw DriverFailure(error) }
      try await Task.sleep(for: .milliseconds(20))
    }
    guard let origin = observer.state().3, observer.state().0 else {
      try? await stream.stopCapture()
      throw DriverFailure("Video did not produce its first frame")
    }
    func timestamp() -> Double {
      max(0, CMClockGetTime(CMClockGetHostTimeClock()).seconds - origin)
    }
    try emitJSON(["event": "ready", "time": timestamp(), "path": path, "width": configuration.width, "height": configuration.height])
    while let line = await readRecorderLine() {
      if let error = observer.state().2 { throw DriverFailure(error) }
      if line == "stop" { break }
      if line == "mark" { try emitJSON(["event": "mark", "time": timestamp()]); continue }
      if line.hasPrefix("{") {
        guard let request = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: String],
              request["op"] == "screenshot", let path = request["path"] else { throw DriverFailure("Invalid recorder request") }
        let settleDeadline = Date().addingTimeInterval(2)
        while !observer.isSettled(), Date() < settleDeadline {
          try await Task.sleep(for: .milliseconds(30))
        }
        var capture = try observer.screenshot(path: path)
        capture["settled"] = observer.isSettled()
        try emitJSON(capture)
        continue
      }
      if line == "reattach" {
        var restoredWindow: SCWindow?
        let restoreDeadline = Date().addingTimeInterval(8)
        while restoredWindow == nil, Date() < restoreDeadline {
          let fresh = try await SCShareableContent.excludingDesktopWindows(true, onScreenWindowsOnly: true)
          let matches = fresh.windows.filter { $0.owningApplication?.bundleIdentifier == bundleID && $0.title == "Mentra" && $0.windowLayer == 0 }
          if matches.count == 1 {
            let candidate = matches[0]
            if abs(candidate.frame.minX - target.frame.minX) < 2, abs(candidate.frame.minY - target.frame.minY) < 2,
               abs(candidate.frame.width - target.frame.width) < 2, abs(candidate.frame.height - target.frame.height) < 2 { restoredWindow = candidate }
          }
          if restoredWindow == nil { try await Task.sleep(for: .milliseconds(100)) }
        }
        guard let newWindow = restoredWindow else { throw DriverFailure("Window geometry changed during recording; restore the initial window bounds") }
        try await stream.updateContentFilter(SCContentFilter(display: display, including: [newWindow]))
        try emitJSON(["event": "reattached", "time": timestamp()])
      }
    }
    try await stream.stopCapture()
    let finishDeadline = Date().addingTimeInterval(10)
    while !observer.state().1, Date() < finishDeadline {
      if let error = observer.state().2 { throw DriverFailure(error) }
      try await Task.sleep(for: .milliseconds(20))
    }
    guard observer.state().1 else { throw DriverFailure("Video finalization timed out") }
    // SCRecordingOutput's progress duration can be rounded to whole seconds.
    // Use the finalized MP4 timeline so late chapter/screenshot times stay valid.
    let asset = AVURLAsset(url: URL(fileURLWithPath: path))
    let duration = try await asset.load(.duration).seconds
    try emitJSON(["event": "finished", "duration": duration, "bytes": recording.recordedFileSize])
  }
}
