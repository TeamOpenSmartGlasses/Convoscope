import assert from "node:assert/strict"
import test from "node:test"

import {prepareProductionPromotion} from "./prepare-production-promotion.mjs"
import {createReleasePlan, familyBuildNumber, loadReleaseFamily, releaseRecordSha256} from "./release-family.mjs"

const family = loadReleaseFamily()
// Family-window numbers derived from the loaded family, so the fixtures follow
// the repository's family base version.
const n = (sequence) => familyBuildNumber(family.familyBaseVersion, sequence)

const betaPlan = createReleasePlan({
  family,
  channel: "beta",
  sequence: 57,
  sourceCommit: "a".repeat(40),
  nativeBuildNumber: n(57),
})
const betaManifest = {
  schemaVersion: 1,
  releaseSetId: betaPlan.releaseSetId,
  releaseIdentity: betaPlan.releaseIdentity,
  familyBaseVersion: betaPlan.familyBaseVersion,
  channel: "beta",
  sourceCommit: betaPlan.sourceCommit,
  native: betaPlan.native,
  releasePlanSha256: releaseRecordSha256(betaPlan),
  completedAt: "2026-08-28T10:00:00.000Z",
  otaManifest: {url: "https://example.com/ota.json", sha256: "d".repeat(64)},
}
const previousManifest = {
  releaseIdentity: "3.0.0",
  sourceCommit: "f".repeat(40),
  native: {marketingVersion: "3.0.0", buildNumber: 300000100},
  url: "https://github.com/Mentra-Community/MentraOS/releases/download/mentra-v3.0.0/manifest.json",
}

// The real inventory contract: App Store builds with their marketing version,
// Play tracks as {track: [versionCode, ...]}.
function inventory(bundleId, current, appleMax, googleMax, extra = {}) {
  const currentVersion = current?.marketingVersion ?? null
  const builds = [
    ...(current ? [{buildNumber: current.buildNumber, marketingVersion: currentVersion}] : []),
    {buildNumber: appleMax, marketingVersion: family.familyBaseVersion},
    ...(extra.appleBuilds || []),
  ]
  const internal = [googleMax, ...(extra.googleInternal || [])]
  return {
    apple: {
      bundleId,
      current,
      maxBuildNumber: Math.max(...builds.map((build) => build.buildNumber)),
      builds,
    },
    google: {
      packageName: bundleId,
      currentVersionCode: current?.buildNumber ?? null,
      maxVersionCode: Math.max(...internal),
      tracks: {internal, production: current ? [current.buildNumber] : []},
    },
  }
}

function prepare(overrides = {}) {
  return prepareProductionPromotion({
    family,
    betaPlan,
    betaManifest,
    betaManifestUrl: "https://github.com/Mentra-Community/MentraOS/releases/download/mentra-builds-v3.1.0/beta.json",
    betaManifestSha256: "b".repeat(64),
    previousManifest,
    mentraInventory: inventory("com.mentra.mentra", {marketingVersion: "3.0.0", buildNumber: 300000100}, n(60), n(59)),
    attempt: 1,
    actor: "release-owner",
    createdAt: "2026-08-28T20:00:00.000Z",
    provenanceUrl: "https://github.com/Mentra-Community/MentraOS/actions/runs/123",
    ...overrides,
  })
}

test("freezes selected source and allocates new store build numbers", () => {
  const {productionPlan, record} = prepare()
  assert.equal(productionPlan.channel, "production")
  // The lab rebuilds the current 3.0.0 app, so its number follows that family.
  assert.equal(record.coordinates.compatibilityLab.ios.buildNumber, 300000101)
  assert.equal(record.coordinates.compatibilityLab.android.buildNumber, 300000101)
  assert.equal(productionPlan.native.buildNumber, n(61))
  assert.equal(record.coordinates.candidates.mentraApp.ios.buildNumber, n(61))
  assert.deepEqual(Object.keys(record.coordinates.candidates), ["mentraApp"])
  assert.deepEqual(Object.keys(record.source), ["mentraosCommit"])
  assert.deepEqual(productionPlan.promotion.otaManifest, betaManifest.otaManifest)
  assert.equal(record.coordinates.currentMentraApp.sourceCommit, "f".repeat(40))
  assert.equal(record.coordinates.currentMentraApp.provenance, "coordinated")
  assert.equal(record.state, "selected")
})

test("first promotion freezes the store-observed public app and skips the compatibility lab", () => {
  const {productionPlan, record} = prepare({
    previousManifest: null,
    mentraInventory: inventory("com.mentra.mentra", {marketingVersion: "3.0", buildNumber: 51180073}, n(60), n(59)),
  })
  assert.equal(record.state, "staging-compatible")
  assert.deepEqual(record.coordinates.currentMentraApp, {
    provenance: "store-observed",
    sourceCommit: null,
    provenanceUrl: null,
    ios: {marketingVersion: "3.0", buildNumber: 51180073},
    android: {marketingVersion: "3.0", buildNumber: 51180073},
  })
  assert.equal(record.coordinates.compatibilityLab, null)
  assert.equal(productionPlan.native.buildNumber, n(61))
  assert.equal(record.coordinates.candidates.mentraApp.android.buildNumber, n(61))
})

test("allocates inside the family window from both stores and ignores strays outside it", () => {
  const current = {marketingVersion: "3.0", buildNumber: 50572313}
  // A stray 900000001 uploaded under another marketing version and a next-family
  // upload on Play sit outside the window and must not move the candidate.
  const {productionPlan, record} = prepare({
    previousManifest: null,
    mentraInventory: inventory("com.mentra.mentra", current, n(60), n(59), {
      appleBuilds: [{buildNumber: 900000001, marketingVersion: "3.0"}],
      googleInternal: [familyBuildNumber("9.9.9", 217)],
    }),
  })
  assert.equal(productionPlan.native.buildNumber, n(61))
  assert.equal(record.coordinates.candidates.mentraApp.ios.buildNumber, n(61))
  assert.equal(record.coordinates.candidates.mentraApp.android.buildNumber, n(61))

  // Play track codes inside the window count even when App Store Connect is lower.
  const crowded = prepare({
    previousManifest: null,
    mentraInventory: inventory("com.mentra.mentra", current, n(60), n(59), {googleInternal: [n(221)]}),
  })
  assert.equal(crowded.productionPlan.native.buildNumber, n(222))

  // A stray upload under the family's own version string above the window
  // cannot be outranked by anything the window offers: Apple would refuse it.
  assert.throws(
    () =>
      prepare({
        previousManifest: null,
        mentraInventory: inventory("com.mentra.mentra", current, n(60), n(59), {
          appleBuilds: [{buildNumber: 900000001, marketingVersion: family.familyBaseVersion}],
        }),
      }),
    /already holds build 900000001 for version .* above its family window/,
  )

  const strayBeta = {...betaPlan, native: {...betaPlan.native, buildNumber: 900000002}}
  assert.throws(
    () =>
      prepare({
        betaPlan: strayBeta,
        betaManifest: {...betaManifest, native: strayBeta.native, releasePlanSha256: releaseRecordSha256(strayBeta)},
      }),
    /outside the .* family window/,
  )
  assert.throws(
    () =>
      prepare({
        previousManifest: null,
        mentraInventory: inventory("com.mentra.mentra", current, n(9999), n(59)),
      }),
    /exhausted/,
  )
  assert.throws(
    () =>
      prepare({
        previousManifest: null,
        mentraInventory: {
          ...inventory("com.mentra.mentra", current, n(60), n(59)),
          google: {...inventory("com.mentra.mentra", null, n(60), n(59)).google, currentVersionCode: n(70)},
        },
      }),
    /not above the Google Play production version code/,
  )
  const noBuilds = inventory("com.mentra.mentra", {marketingVersion: "3.0.0", buildNumber: 300000100}, n(60), n(59))
  delete noBuilds.apple.builds
  assert.throws(() => prepare({mentraInventory: noBuilds}), /lists no builds/)
  const noTracks = inventory("com.mentra.mentra", {marketingVersion: "3.0.0", buildNumber: 300000100}, n(60), n(59))
  delete noTracks.google.tracks
  assert.throws(() => prepare({mentraInventory: noTracks}), /lists no tracks/)
})

test("first promotion still requires a public app in both stores", () => {
  assert.throws(
    () =>
      prepare({
        previousManifest: null,
        mentraInventory: inventory("com.mentra.mentra", null, n(60), n(59)),
      }),
    /has no current public store release/,
  )
})

test("rejects store state that does not match current production provenance", () => {
  assert.throws(
    () =>
      prepare({
        mentraInventory: inventory(
          "com.mentra.mentra",
          {marketingVersion: "3.0.0", buildNumber: 300000099},
          n(60),
          n(59),
        ),
      }),
    /do not match the previous production manifest/,
  )
})

test("rejects an incomplete selected beta", () => {
  assert.throws(() => prepare({betaManifest: {...betaManifest, completedAt: undefined}}), /not complete/)
})
