import AppKit
import ApplicationServices
import ScreenCaptureKit

struct DriverFailure: Error, CustomStringConvertible {
  let description: String
  init(_ message: String) {
    description = message
  }
}

func attribute(_ element: AXUIElement, _ name: String) -> CFTypeRef? {
  var value: CFTypeRef?
  return AXUIElementCopyAttributeValue(element, name as CFString, &value) == .success ? value : nil
}

func stringAttribute(_ element: AXUIElement, _ name: String) -> String {
  let value = attribute(element, name)
  return value as? String ?? ""
}

func frameOf(_ element: AXUIElement) -> CGRect? {
  guard let position = attribute(element, kAXPositionAttribute), let size = attribute(element, kAXSizeAttribute),
        CFGetTypeID(position) == AXValueGetTypeID(), CFGetTypeID(size) == AXValueGetTypeID() else { return nil }
  var point = CGPoint.zero
  var dimensions = CGSize.zero
  guard AXValueGetValue(position as! AXValue, .cgPoint, &point),
        AXValueGetValue(size as! AXValue, .cgSize, &dimensions) else { return nil }
  return CGRect(origin: point, size: dimensions)
}

func frameJSON(_ frame: CGRect) -> [String: Double] {
  ["x": frame.minX, "y": frame.minY, "width": frame.width, "height": frame.height]
}

struct Element {
  let ax: AXUIElement
  let data: [String: Any]
  let ancestors: [[String: Any]]
}

func validateSelector(_ selector: [String: Any]) throws {
  let strings: Set = ["role", "subrole", "title", "description", "value", "placeholder", "identifier", "text", "contains"]
  let bools: Set = ["enabled", "focused", "visible"]
  for (key, value) in selector {
    if strings.contains(key), value is String { continue }
    if bools.contains(key), value is Bool { continue }
    if key == "ancestor", let ancestor = value as? [String: Any] {
      try validateSelector(ancestor)
      continue
    }
    throw DriverFailure("Unsupported selector field or type: \(key). Use an accessibility identifier or label; coordinates are unsupported")
  }
}

func matches(_ data: [String: Any], _ selector: [String: Any]) -> Bool {
  for key in ["role", "subrole", "title", "description", "value", "placeholder", "identifier"] {
    if let expected = selector[key] as? String, data[key] as? String != expected { return false }
  }
  let texts = ["title", "description", "value", "placeholder"].compactMap { data[$0] as? String }
  if let expected = selector["text"] as? String, !texts.contains(expected) { return false }
  if let expected = selector["contains"] as? String, !texts.contains(where: { $0.contains(expected) }) { return false }
  if let expected = selector["enabled"] as? Bool, data["enabled"] as? Bool != expected { return false }
  if let expected = selector["focused"] as? Bool, data["focused"] as? Bool != expected { return false }
  return true
}

@MainActor
final class Driver {
  let bundleID: String
  let app: NSRunningApplication
  let root: AXUIElement

  init(bundleID: String) throws {
    self.bundleID = bundleID
    let candidates = NSWorkspace.shared.runningApplications.filter { $0.bundleIdentifier == bundleID && !$0.isTerminated }
    guard candidates.count == 1 else { throw DriverFailure("Expected one running \(bundleID) process, found \(candidates.count)") }
    app = candidates[0]
    root = AXUIElementCreateApplication(app.processIdentifier)
    AXUIElementSetMessagingTimeout(root, 3)
  }

  func window() throws -> AXUIElement {
    guard AXIsProcessTrusted() else { throw DriverFailure("Accessibility permission is missing for this runner") }
    let windows = attribute(root, kAXWindowsAttribute) as? [AXUIElement] ?? []
    let standard = windows.filter { stringAttribute($0, kAXSubroleAttribute) == kAXStandardWindowSubrole }
    guard standard.count == 1 else { throw DriverFailure("Expected one standard target window, found \(standard.count)") }
    return standard[0]
  }

  func elements() throws -> [Element] {
    let window = try window()
    let windowFrame = frameOf(window) ?? .zero
    var result: [Element] = []
    func visit(_ ax: AXUIElement, path: String, ancestors: [[String: Any]], depth: Int) {
      guard depth < 60, result.count < 5000 else { return }
      let role = stringAttribute(ax, kAXRoleAttribute)
      let subrole = stringAttribute(ax, kAXSubroleAttribute)
      let secure = subrole == kAXSecureTextFieldSubrole
      var data: [String: Any] = ["path": path, "role": role, "subrole": subrole,
                                 "title": stringAttribute(ax, kAXTitleAttribute), "description": stringAttribute(ax, kAXDescriptionAttribute),
                                 "identifier": stringAttribute(ax, kAXIdentifierAttribute), "placeholder": stringAttribute(ax, kAXPlaceholderValueAttribute),
                                 "value": secure ? "[REDACTED]" : stringAttribute(ax, kAXValueAttribute),
                                 "enabled": attribute(ax, kAXEnabledAttribute) as? Bool ?? true,
                                 "focused": attribute(ax, kAXFocusedAttribute) as? Bool ?? false]
      var names: CFArray?
      AXUIElementCopyActionNames(ax, &names)
      data["actions"] = names as? [String] ?? []
      if let frame = frameOf(ax) {
        data["frame"] = frameJSON(frame)
        data["visible"] = frame.width > 0 && frame.height > 0 && frame.intersects(windowFrame)
      } else { data["visible"] = false }
      result.append(Element(ax: ax, data: data, ancestors: ancestors))
      let children = attribute(ax, kAXChildrenAttribute) as? [AXUIElement] ?? []
      for (index, child) in children.enumerated() {
        visit(child, path: "\(path).\(index)", ancestors: ancestors + [data], depth: depth + 1)
      }
    }
    visit(window, path: "0", ancestors: [], depth: 0)
    return result
  }

  func find(_ selector: [String: Any]) throws -> Element {
    try validateSelector(selector)
    let names = ["identifier", "title", "description", "placeholder", "text", "contains"]
    guard names.contains(where: { !(selector[$0] as? String ?? "").isEmpty }) else {
      throw DriverFailure("A named accessibility target is required; anonymous or position-based targets are unsupported")
    }
    let found = try elements().filter { element in
      matches(element.data, selector) && (selector["visible"] as? Bool == false || element.data["visible"] as? Bool == true)
        && ((selector["ancestor"] as? [String: Any]).map { ancestor in element.ancestors.contains { matches($0, ancestor) } } ?? true)
    }
    guard found.count == 1 else { throw DriverFailure("Selector matched \(found.count) elements; expected exactly one") }
    return found[0]
  }

  func capture(_ path: String) async throws -> [String: Any] {
    guard CGPreflightScreenCaptureAccess() else { throw DriverFailure("Screen Recording permission is missing for this runner") }
    let content = try await SCShareableContent.excludingDesktopWindows(true, onScreenWindowsOnly: true)
    let title = try stringAttribute(window(), kAXTitleAttribute)
    let windows = content.windows.filter { $0.owningApplication?.processID == app.processIdentifier && $0.title == title && $0.windowLayer == 0 }
    guard windows.count == 1 else { throw DriverFailure("Expected one capturable target window, found \(windows.count)") }
    let window = windows[0]
    let filter = SCContentFilter(desktopIndependentWindow: window)
    let configuration = SCStreamConfiguration()
    let scale = Double(filter.pointPixelScale)
    configuration.width = Int(filter.contentRect.width * scale)
    configuration.height = Int(filter.contentRect.height * scale)
    configuration.showsCursor = false
    configuration.ignoreShadowsSingleWindow = true
    let image = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: configuration)
    guard let png = NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:]) else { throw DriverFailure("PNG encoding failed") }
    try png.write(to: URL(fileURLWithPath: path), options: .atomic)
    return ["path": path, "width": image.width, "height": image.height, "bytes": png.count, "scale": scale]
  }

  func execute(_ command: [String: Any]) async throws -> [String: Any] {
    let allowedFields: Set = ["op", "bundleId", "selector", "method", "text", "action", "path"]
    guard Set(command.keys).isSubset(of: allowedFields) else {
      throw DriverFailure("Unsupported command fields; foreground and coordinate input are not supported")
    }
    if let method = command["method"] {
      guard method as? String == "ax-value", command["op"] as? String == "type" else {
        throw DriverFailure("Only accessibility actions are supported; no mouse, keyboard, or visual fallback")
      }
    }
    let operation = command["op"] as? String ?? "snapshot"
    let selector = command["selector"] as? [String: Any] ?? [:]
    switch operation {
    case "request-screen-capture":
      return ["screenCapture": CGRequestScreenCaptureAccess()]
    case "doctor":
      return ["accessibility": AXIsProcessTrusted(), "screenCapture": CGPreflightScreenCaptureAccess(), "postEvents": CGPreflightPostEventAccess(), "pid": app.processIdentifier,
              "frontmostBundleId": NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? "unknown",
              "bundleId": bundleID, "bundlePath": app.bundleURL?.path ?? "", "version": Bundle(url: app.bundleURL!)?.infoDictionary?["CFBundleShortVersionString"] ?? "unknown",
              "build": Bundle(url: app.bundleURL!)?.infoDictionary?["CFBundleVersion"] ?? "unknown"]
    case "snapshot":
      return try ["pid": app.processIdentifier, "window": frameJSON(frameOf(window()) ?? .zero), "elements": elements().map(\.data)]
    case "relaunch":
      guard let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleID) ?? app.bundleURL else { throw DriverFailure("Cannot resolve installed app URL") }
      let originalFrame = try frameOf(window())
      guard app.terminate() else { throw DriverFailure("Target refused normal termination") }
      let deadline = Date().addingTimeInterval(8)
      while !app.isTerminated, Date() < deadline {
        try await Task.sleep(for: .milliseconds(100))
      }
      guard app.isTerminated else { throw DriverFailure("Target did not terminate within eight seconds") }
      let config = NSWorkspace.OpenConfiguration()
      config.activates = false
      let reopened = try await NSWorkspace.shared.openApplication(at: url, configuration: config)
      guard reopened.bundleIdentifier == bundleID else { throw DriverFailure("Relaunch opened an unexpected app") }
      let fresh = try Driver(bundleID: bundleID)
      let windowDeadline = Date().addingTimeInterval(10)
      while (try? fresh.window()) == nil, Date() < windowDeadline {
        try await Task.sleep(for: .milliseconds(100))
      }
      if let originalFrame, let newWindow = try? fresh.window() {
        var point = originalFrame.origin, size = originalFrame.size
        if let position = AXValueCreate(.cgPoint, &point), let dimensions = AXValueCreate(.cgSize, &size) {
          AXUIElementSetAttributeValue(newWindow, kAXPositionAttribute as CFString, position)
          AXUIElementSetAttributeValue(newWindow, kAXSizeAttribute as CFString, dimensions)
        }
      }
      return ["pid": reopened.processIdentifier]
    case "press":
      let target = try find(selector)
      let actions = target.data["actions"] as? [String] ?? []
      guard actions.contains(kAXPressAction) else {
        throw DriverFailure("AXPress is not exposed. Fix this control's accessibility in the Mentra App and install the rebuilt app; coordinate fallback is unsupported")
      }
      let result = AXUIElementPerformAction(target.ax, kAXPressAction as CFString)
      guard result == .success else { throw DriverFailure("AXPress failed: \(result.rawValue)") }
      return ["method": "AXPress"]
    case "perform":
      let allowed = ["AXScrollDownByPage", "AXScrollUpByPage", "AXScrollToVisible"]
      guard let action = command["action"] as? String, allowed.contains(action) else {
        throw DriverFailure("Unsupported accessibility action; only semantic scrolling is available here")
      }
      let target = try find(selector)
      guard (target.data["actions"] as? [String] ?? []).contains(action) else {
        throw DriverFailure("Requested accessibility action is not exposed by the target")
      }
      let result = AXUIElementPerformAction(target.ax, action as CFString)
      guard result == .success else { throw DriverFailure("Accessibility action failed: \(result.rawValue)") }
      return ["method": action]
    case "type":
      let target = try find(selector)
      guard target.data["role"] as? String == kAXTextFieldRole || target.data["role"] as? String == kAXTextAreaRole else { throw DriverFailure("Text target is not an editable field") }
      guard let text = command["text"] as? String else { throw DriverFailure("Text entry requires a string") }
      let result = AXUIElementSetAttributeValue(target.ax, kAXValueAttribute as CFString, text as CFString)
      guard result == .success else { throw DriverFailure("AXValue text entry failed: \(result.rawValue). Fix the editable control in the Mentra App; keyboard injection is unsupported") }
      return ["method": "AXValue"]
    case "screenshot":
      guard let path = command["path"] as? String else { throw DriverFailure("Screenshot path is required") }
      return try await capture(path)
    default: throw DriverFailure("Unknown operation: \(operation)")
    }
  }
}

@main
struct Main {
  @MainActor static func main() async {
    do {
      _ = NSApplication.shared
      NSApp.setActivationPolicy(.prohibited)
      if CommandLine.arguments.count == 3, CommandLine.arguments[1] == "--record" {
        try await Driver(bundleID: "com.mentra.mentra").recordVideo(path: CommandLine.arguments[2])
        return
      }
      let input = FileHandle.standardInput.readDataToEndOfFile()
      guard let command = try JSONSerialization.jsonObject(with: input) as? [String: Any] else { throw DriverFailure("Expected a JSON command") }
      let driver = try Driver(bundleID: command["bundleId"] as? String ?? "com.mentra.mentra")
      let result = try await driver.execute(command)
      let output = try JSONSerialization.data(withJSONObject: ["ok": true, "result": result], options: [.sortedKeys])
      FileHandle.standardOutput.write(output + Data("\n".utf8))
    } catch {
      let output = try! JSONSerialization.data(withJSONObject: ["ok": false, "error": String(describing: error)], options: [.sortedKeys])
      FileHandle.standardOutput.write(output + Data("\n".utf8))
      exit(1)
    }
  }
}
