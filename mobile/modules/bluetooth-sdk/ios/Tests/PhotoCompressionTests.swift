@testable import MentraBluetoothSDK
import XCTest

/// `compress` tiers: canonical none/low/medium/high, `heavy` legacy alias, wire spelling for old firmware.
final class PhotoCompressionTests: XCTestCase {
    private let baseParams: [String: Any] = [
        "size": "medium",
        "webhookUrl": "https://example.com/upload",
    ]

    func testEveryTierParsesFromBridgeParams() throws {
        let expectations: [(String, PhotoCompression)] = [
            ("none", .none), ("low", .low), ("medium", .medium), ("high", .high), ("heavy", .heavy),
        ]
        for (raw, expected) in expectations {
            var params = baseParams
            params["compress"] = raw
            let request = try PhotoRequest.from(params: params)
            XCTAssertEqual(request.compress, expected, "compress=\(raw)")
        }
    }

    func testOmittedAndUnknownCompressionFallBackToNone() throws {
        let omitted = try PhotoRequest.from(params: baseParams)
        XCTAssertEqual(omitted.compress?.wireValue ?? "none", "none")

        var params = baseParams
        params["compress"] = "ultra"
        let unknown = try PhotoRequest.from(params: params)
        XCTAssertNil(unknown.compress)
        XCTAssertEqual(unknown.compress?.wireValue ?? "none", "none")
    }

    func testHighIsSentAsLegacyHeavyAndOtherTiersVerbatim() {
        XCTAssertEqual(PhotoCompression.none.wireValue, "none")
        XCTAssertEqual(PhotoCompression.low.wireValue, "low")
        XCTAssertEqual(PhotoCompression.medium.wireValue, "medium")
        XCTAssertEqual(PhotoCompression.high.wireValue, "heavy")
        XCTAssertEqual(PhotoCompression.heavy.wireValue, "heavy")
    }

    func testHeavyIsTheLegacyAliasForHigh() {
        XCTAssertEqual(PhotoCompression.heavy.canonical, .high)
        XCTAssertEqual(PhotoCompression.high.canonical, .high)
        for tier in [PhotoCompression.none, .low, .medium] {
            XCTAssertEqual(tier.canonical, tier)
        }
    }

    func testCompressionSurvivesRequestIdRewrite() throws {
        var params = baseParams
        params["compress"] = "low"
        params["requestId"] = "photo-1"
        let request = try PhotoRequest.from(params: params)
        XCTAssertEqual(request.withRequestId("routed").compress, .low)
    }
}
