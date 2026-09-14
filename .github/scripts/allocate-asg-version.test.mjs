import assert from "node:assert/strict"
import test from "node:test"

import {allocateAsgVersion, asgVersionCodePrefix} from "./allocate-asg-version.mjs"

const fingerprint = "a".repeat(64)
const other = "b".repeat(64)

function asset(id, versionCode, selectedFingerprint, extension) {
  return {id, name: `mentra-live-asg-${versionCode}-${selectedFingerprint}.${extension}`}
}

test("derives the version code prefix from the family build-number formula", () => {
  assert.equal(asgVersionCodePrefix("3.1.0"), 301_000_000)
  assert.equal(asgVersionCodePrefix("3.1.1"), 301_010_000)
  assert.equal(asgVersionCodePrefix("3.2.4"), 302_040_000)
  assert.equal(asgVersionCodePrefix("20.99.99"), 2_099_990_000)
  assert.throws(() => asgVersionCodePrefix("3.1.0-beta.5"), /plain X\.Y\.Z/)
  assert.throws(() => asgVersionCodePrefix("1.9.0"), /major 1 must be between 2 and 20/)
  assert.throws(() => asgVersionCodePrefix("21.0.0"), /major 21 must be between 2 and 20/)
  assert.throws(() => asgVersionCodePrefix("3.100.0"), /minor 100 must be at most 99/)
  assert.throws(() => asgVersionCodePrefix("3.1.100"), /patch 100 must be at most 99/)
})

test("uses the run number as the sequence and ignores every legacy namespace", () => {
  const result = allocateAsgVersion({
    assets: [
      asset(1, 52_000_000, other, "apk"),
      asset(2, 52_000_000, other, "json"),
      asset(3, 100_000_173, "c".repeat(64), "apk"),
      asset(4, 100_000_173, "c".repeat(64), "json"),
      asset(5, 301_000_001, "d".repeat(64), "apk"),
      asset(6, 301_000_001, "d".repeat(64), "json"),
    ],
    fingerprint,
    baseVersion: "3.1.1",
    sequence: 230,
  })
  assert.equal(result.exists, false)
  assert.equal(result.versionCode, 301_010_230)
  assert.equal(result.apkAsset, `mentra-live-asg-301010230-${fingerprint}.apk`)
  assert.deepEqual(result.orphanAssetIds, [])
})

test("never lands on or below a code already used in the family window", () => {
  const assets = [
    asset(1, 301_010_240, other, "apk"),
    asset(2, 301_010_240, other, "json"),
    asset(3, 302_000_007, "d".repeat(64), "apk"),
    asset(4, 302_000_007, "d".repeat(64), "json"),
  ]
  assert.equal(allocateAsgVersion({assets, fingerprint, baseVersion: "3.1.1", sequence: 240}).versionCode, 301_010_241)
  assert.equal(allocateAsgVersion({assets, fingerprint, baseVersion: "3.1.1", sequence: 250}).versionCode, 301_010_250)
  assert.equal(allocateAsgVersion({assets, fingerprint, baseVersion: "3.2.0", sequence: 5}).versionCode, 302_000_008)
  assert.equal(
    allocateAsgVersion({assets: [], fingerprint, baseVersion: "3.2.0", sequence: 5}).versionCode,
    302_000_005,
  )
})

test("reuses the recorded code for an existing complete fingerprint in the family window", () => {
  const result = allocateAsgVersion({
    assets: [asset(1, 301_010_042, fingerprint, "apk"), asset(2, 301_010_042, fingerprint, "json")],
    fingerprint,
    baseVersion: "3.1.1",
    sequence: 300,
  })
  assert.equal(result.exists, true)
  assert.equal(result.versionCode, 301_010_042)
  // A pair of the same fingerprint under an older scheme is not this family's build.
  const legacy = allocateAsgVersion({
    assets: [asset(1, 100_000_173, fingerprint, "apk"), asset(2, 100_000_173, fingerprint, "json")],
    fingerprint,
    baseVersion: "3.1.0",
    sequence: 300,
  })
  assert.equal(legacy.exists, false)
  assert.equal(legacy.versionCode, 301_000_300)
  assert.deepEqual(legacy.orphanAssetIds, [])
})

test("marks an interrupted asset pair for removal before rebuilding", () => {
  const result = allocateAsgVersion({
    assets: [asset(7, 301_010_005, fingerprint, "apk")],
    fingerprint,
    baseVersion: "3.1.1",
    sequence: 5,
  })
  assert.equal(result.exists, false)
  assert.equal(result.versionCode, 301_010_006)
  assert.deepEqual(result.orphanAssetIds, [7])
})

test("rejects duplicate and mismatched complete pairs, bad sequences, and an exhausted base version", () => {
  assert.throws(
    () =>
      allocateAsgVersion({
        assets: [asset(1, 301_010_057, fingerprint, "apk"), asset(2, 301_010_057, fingerprint, "apk")],
        fingerprint,
        baseVersion: "3.1.1",
        sequence: 60,
      }),
    /Duplicate/,
  )
  assert.throws(
    () =>
      allocateAsgVersion({
        assets: [asset(1, 301_010_057, fingerprint, "apk"), asset(2, 301_010_058, fingerprint, "json")],
        fingerprint,
        baseVersion: "3.1.1",
        sequence: 60,
      }),
    /different version codes/,
  )
  assert.throws(() => allocateAsgVersion({assets: [], fingerprint, baseVersion: "3.1.1", sequence: 0}), /between 1 and/)
  assert.throws(
    () =>
      allocateAsgVersion({
        assets: [asset(1, 301_019_999, other, "apk"), asset(2, 301_019_999, other, "json")],
        fingerprint,
        baseVersion: "3.1.1",
        sequence: 10,
      }),
    /exhausted/,
  )
})
