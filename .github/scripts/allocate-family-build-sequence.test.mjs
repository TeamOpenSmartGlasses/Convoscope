import assert from "node:assert/strict"
import test from "node:test"

import {
  allocateFamilyBuildNumber,
  familyBuildNumberMarker,
  markerAssetName,
  recordedFamilyBuildNumbers,
} from "./allocate-family-build-sequence.mjs"
import {familyBuildNumber} from "./release-family.mjs"

const fp = "a".repeat(64)
const marker = (version, sequence) => ({name: markerAssetName(familyBuildNumber(version, sequence))})
const asg = (version, sequence, extension = "apk") => ({
  name: `mentra-live-asg-${familyBuildNumber(version, sequence)}-${fp}.${extension}`,
})

test("a new family starts at sequence 1 and every later run takes the next free number", () => {
  assert.deepEqual(allocateFamilyBuildNumber({assets: [], baseVersion: "3.1.1"}), {
    familyBaseVersion: "3.1.1",
    buildNumber: familyBuildNumber("3.1.1", 1),
    sequence: 1,
    markerAsset: "mentra-build-number-301010001.json",
  })
  const assets = [
    marker("3.1.1", 1),
    marker("3.1.1", 2),
    asg("3.1.1", 2),
    asg("3.1.1", 2, "json"),
    asg("3.1.1", 5),
    marker("3.2.0", 40),
    {name: "mentra-release-plan-3.1.1-beta.240.json"},
    {name: "mentra-live-asg-100000173-" + fp + ".apk"},
  ]
  assert.deepEqual(recordedFamilyBuildNumbers(assets, "3.1.1"), [
    familyBuildNumber("3.1.1", 1),
    familyBuildNumber("3.1.1", 2),
    familyBuildNumber("3.1.1", 5),
  ])
  assert.equal(allocateFamilyBuildNumber({assets, baseVersion: "3.1.1"}).sequence, 6)
  assert.equal(allocateFamilyBuildNumber({assets, baseVersion: "3.2.0"}).sequence, 41)
  assert.equal(allocateFamilyBuildNumber({assets, baseVersion: "3.3.0"}).sequence, 1)
})

test("the release band is bounded and the marker is deterministic", () => {
  assert.throws(() => allocateFamilyBuildNumber({assets: [marker("3.1.1", 2999)], baseVersion: "3.1.1"}), /exhausted/)
  assert.throws(() => allocateFamilyBuildNumber({assets: null, baseVersion: "3.1.1"}), /must be an array/)
  const built = familyBuildNumberMarker({baseVersion: "3.1.1", buildNumber: familyBuildNumber("3.1.1", 7)})
  assert.deepEqual(built, {
    schemaVersion: 1,
    kind: "mentra-family-build-number",
    familyBaseVersion: "3.1.1",
    buildNumber: familyBuildNumber("3.1.1", 7),
    sequence: 7,
  })
  assert.throws(
    () => familyBuildNumberMarker({baseVersion: "3.1.1", buildNumber: 900000002}),
    /does not belong to family/,
  )
})
