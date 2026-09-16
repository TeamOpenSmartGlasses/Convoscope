import AppKit
import ApplicationServices

/// Only known system dialogs naming this app are eligible. Other applications,
/// Keychain prompts and arbitrary "Allow" buttons are never targeted.
@MainActor
func handleMentraLaunchDialogs() throws {
    guard AXIsProcessTrusted() else { return }
    let owners = ["com.apple.UserNotificationCenter", "com.apple.CoreServicesUIAgent", "com.apple.SecurityAgent"]
    func read(_ element: AXUIElement, _ key: String) -> CFTypeRef? {
        var value: CFTypeRef?
        return AXUIElementCopyAttributeValue(element, key as CFString, &value) == .success ? value : nil
    }
    func normalized(_ text: String) -> String {
        text.lowercased().replacingOccurrences(of: "“", with: "").replacingOccurrences(of: "”", with: "")
            .replacingOccurrences(of: "\"", with: "").replacingOccurrences(of: "’", with: "'")
            .trimmingCharacters(in: .whitespacesAndNewlines.union(CharacterSet(charactersIn: ".")))
    }
    for app in NSWorkspace.shared.runningApplications where owners.contains(app.bundleIdentifier ?? "") {
        let root = AXUIElementCreateApplication(app.processIdentifier)
        AXUIElementSetMessagingTimeout(root, 0.5)
        for window in read(root, kAXWindowsAttribute) as? [AXUIElement] ?? [] {
            var nodes: [AXUIElement] = []
            func visit(_ element: AXUIElement, depth: Int) {
                guard depth < 20, nodes.count < 1000 else { return }
                nodes.append(element)
                for child in read(element, kAXChildrenAttribute) as? [AXUIElement] ?? [] {
                    visit(child, depth: depth + 1)
                }
            }
            visit(window, depth: 0)
            let texts = nodes.flatMap { node in
                [kAXTitleAttribute, kAXDescriptionAttribute, kAXValueAttribute].compactMap { read(node, $0) as? String }.map(normalized)
            }
            let bluetooth = texts.contains("mentra would like to use bluetooth")
            let invalid = texts.contains("you can't open the application mentra because this application is not supported on this mac")
                || (texts.contains("mentra no longer available") && texts.contains(where: { $0.contains("provisioning profile is invalid") }))
            guard bluetooth || invalid else { continue }
            let labels = bluetooth ? ["allow", "ok"] : ["ok"]
            let buttons = nodes.filter { node in
                (read(node, kAXRoleAttribute) as? String) == kAXButtonRole
                    && [kAXTitleAttribute, kAXDescriptionAttribute].contains { labels.contains(normalized(read(node, $0) as? String ?? "")) }
            }
            guard buttons.count == 1, AXUIElementPerformAction(buttons[0], kAXPressAction as CFString) == .success else {
                throw NSError(domain: "MentraLauncher", code: 5, userInfo: [NSLocalizedDescriptionKey: "Recognized Mentra launch dialog has no unique accessible response"])
            }
            if invalid {
                throw NSError(domain: "MentraLauncher", code: 6, userInfo: [NSLocalizedDescriptionKey: "Dismissed Mentra's invalid/unsupported build dialog; rebuild the signed Mac wrapper before retrying"])
            }
            print("Allowed Mentra's Bluetooth request through its named system dialog.")
        }
    }
}

@main
struct LaunchIOSOnMac {
    @MainActor
    static func main() async {
        do {
            try await launch()
        } catch {
            FileHandle.standardError.write(Data("Launch failed: \(error)\n".utf8))
            exit(1)
        }
    }

    @MainActor
    static func launch() async throws {
        guard CommandLine.arguments.count == 2 else {
            throw NSError(domain: "MentraLauncher", code: 1, userInfo: [NSLocalizedDescriptionKey: "Expected a built .app path"])
        }
        let url = URL(fileURLWithPath: CommandLine.arguments[1]).standardizedFileURL
        let innerURL = url.appendingPathComponent("WrappedBundle").resolvingSymlinksInPath()
        guard let bundle = Bundle(url: innerURL), let id = bundle.bundleIdentifier else {
            throw NSError(domain: "MentraLauncher", code: 2, userInfo: [NSLocalizedDescriptionKey: "Cannot read app identity"])
        }
        _ = NSApplication.shared
        NSApp.setActivationPolicy(.prohibited)
        let existing = NSRunningApplication.runningApplications(withBundleIdentifier: id)
        for app in existing {
            guard app.terminate() else {
                throw NSError(domain: "MentraLauncher", code: 3, userInfo: [NSLocalizedDescriptionKey: "App refused normal termination"])
            }
        }
        let deadline = Date().addingTimeInterval(10)
        while existing.contains(where: { !$0.isTerminated }), Date() < deadline {
            try await Task.sleep(for: .milliseconds(100))
        }
        guard existing.allSatisfy(\.isTerminated) else {
            throw NSError(domain: "MentraLauncher", code: 4, userInfo: [NSLocalizedDescriptionKey: "App did not terminate; no forced kill performed"])
        }
        let configuration = NSWorkspace.OpenConfiguration()
        configuration.activates = false
        let timeout = Task { @MainActor in
            try await Task.sleep(for: .seconds(30))
            FileHandle.standardError.write(Data("Launch did not finish within 30 seconds. Inspect macOS setup permissions; no unrelated dialog was accepted.\n".utf8))
            exit(1)
        }
        let dialogs = Task { @MainActor in
            while !Task.isCancelled {
                do {
                    try handleMentraLaunchDialogs()
                    try await Task.sleep(for: .milliseconds(250))
                } catch is CancellationError { return }
                catch {
                    FileHandle.standardError.write(Data("Launch dialog failed: \(error)\n".utf8))
                    exit(1)
                }
            }
        }
        defer { timeout.cancel(); dialogs.cancel() }
        let app = try await NSWorkspace.shared.openApplication(at: url, configuration: configuration)
        // Privacy prompts can arrive after Launch Services has returned the process.
        try await Task.sleep(for: .seconds(2))
        print("Launched \(id) pid=\(app.processIdentifier) without requesting foreground activation.")
    }
}
