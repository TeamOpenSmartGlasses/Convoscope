import assert from "node:assert/strict"
import test from "node:test"

import {prepareProductionPromotion} from "./prepare-production-promotion.mjs"
import {createReleasePlan, loadReleaseFamily, releaseRecordSha256} from "./release-family.mjs"

const family = loadReleaseFamily()
const betaPlan = createReleasePlan({
  family,
  channel: "beta",
  sequence: 57,
  sourceCommit: "a".repeat(40),
  nativeBuildNumber: 310000057,
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

function inventory(bundleId, current, appleMax, googleMax, extra = {}) {
  const appleNumbers = [...(current ? [current.buildNumber] : []), appleMax, ...(extra.appleNumbers || [])]
  const googleNumbers = [googleMax, ...(extra.googleNumbers || [])]
  return {
    apple: {bundleId, current, maxBuildNumber: Math.max(...appleNumbers), buildNumbers: appleNumbers},
    google: {
      packageName: bundleId,
      currentVersionCode: current?.buildNumber ?? null,
      maxVersionCode: Math.max(...googleNumbers),
      tracks: {
        internal: [{name: "internal", status: "completed", versionCodes: googleNumbers}],
        production: current ? [{name: "production", status: "completed", versionCodes: [current.buildNumber]}] : [],
      },
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
    mentraInventory: inventory(
      "com.mentra.mentra",
      {marketingVersion: "3.0.0", buildNumber: 300000100},
      310000060,
      310000059,
    ),
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
  assert.equal(productionPlan.native.buildNumber, 310000061)
  assert.equal(record.coordinates.candidates.mentraApp.ios.buildNumber, 310000061)
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
    mentraInventory: inventory(
      "com.mentra.mentra",
      {marketingVersion: "3.0", buildNumber: 51180073},
      310000060,
      310000059,
    ),
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
  assert.equal(productionPlan.native.buildNumber, 310000061)
  assert.equal(record.coordinates.candidates.mentraApp.android.buildNumber, 310000061)
})

test("allocates inside the family window and ignores stray store builds outside it", () => {
  // A stray 900000001 in App Store Connect and a 3.2.0 dev upload on Play sit
  // outside the 3.1.0 window and must not move the candidate.
  const {productionPlan, record} = prepare({
    previousManifest: null,
    mentraInventory: inventory(
      "com.mentra.mentra",
      {marketingVersion: "3.0", buildNumber: 50572313},
      310000060,
      310000059,
      {appleNumbers: [900000001], googleNumbers: [320000217]},
    ),
  })
  assert.equal(productionPlan.native.buildNumber, 310000061)
  assert.equal(record.coordinates.candidates.mentraApp.ios.buildNumber, 310000061)
  assert.equal(record.coordinates.candidates.mentraApp.android.buildNumber, 310000061)

  // Numbers already used inside the window are still respected.
  const crowded = prepare({
    previousManifest: null,
    mentraInventory: inventory(
      "com.mentra.mentra",
      {marketingVersion: "3.0", buildNumber: 50572313},
      310000060,
      310000059,
      {googleNumbers: [310000221]},
    ),
  })
  assert.equal(crowded.productionPlan.native.buildNumber, 310000222)

  const strayBeta = {...betaPlan, native: {...betaPlan.native, buildNumber: 900000002}}
  assert.throws(
    () =>
      prepare({
        betaPlan: strayBeta,
        betaManifest: {...betaManifest, native: strayBeta.native, releasePlanSha256: releaseRecordSha256(strayBeta)},
      }),
    /outside the 3\.1\.0 family window/,
  )
  assert.throws(
    () =>
      prepare({
        previousManifest: null,
        mentraInventory: inventory(
          "com.mentra.mentra",
          {marketingVersion: "3.0", buildNumber: 50572313},
          310099999,
          310000059,
        ),
      }),
    /exhausted/,
  )
  assert.throws(
    () =>
      prepare({
        previousManifest: null,
        mentraInventory: {
          ...inventory("com.mentra.mentra", {marketingVersion: "3.0", buildNumber: 50572313}, 310000060, 310000059),
          google: {
            ...inventory("com.mentra.mentra", null, 310000060, 310000059).google,
            currentVersionCode: 310000070,
          },
        },
      }),
    /not above the Google Play production version code/,
  )
  assert.throws(
    () =>
      prepare({
        mentraInventory: {
          ...inventory("com.mentra.mentra", {marketingVersion: "3.0.0", buildNumber: 300000100}, 310000060, 310000059),
          apple: {
            bundleId: "com.mentra.mentra",
            current: {marketingVersion: "3.0.0", buildNumber: 300000100},
            maxBuildNumber: 310000060,
          },
        },
      }),
    /lists no build numbers/,
  )
})

test("first promotion still requires a public app in both stores", () => {
  assert.throws(
    () =>
      prepare({
        previousManifest: null,
        mentraInventory: inventory("com.mentra.mentra", null, 310000060, 310000059),
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
          310000060,
          310000059,
        ),
      }),
    /do not match the previous production manifest/,
  )
})

test("rejects an incomplete selected beta", () => {
  assert.throws(() => prepare({betaManifest: {...betaManifest, completedAt: undefined}}), /not complete/)
})
