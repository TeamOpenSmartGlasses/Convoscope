#!/usr/bin/env node
import {createHash} from "node:crypto"
import {mkdirSync, readFileSync, writeFileSync} from "node:fs"
import path from "node:path"
import {fileURLToPath} from "node:url"

import {createInitialPromotionRecord, promotionAssetName} from "./production-promotion-state.mjs"
import {
  createReleasePlan,
  familyBuildNumberWindow,
  loadReleaseFamily,
  releaseRecordSha256,
  serializeReleaseRecord,
} from "./release-family.mjs"

const COMMIT_PATTERN = /^[0-9a-f]{40}$/

function sha256File(file) {
  return createHash("sha256").update(readFileSync(file)).digest("hex")
}

function requireInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error(`${label} must be a non-negative safe integer`)
  return value
}

function validateInventory(inventory, {bundleId, allowNoCurrent}) {
  if (inventory.apple?.bundleId !== bundleId) throw new Error(`Apple inventory does not identify ${bundleId}`)
  if (inventory.google?.packageName !== bundleId) throw new Error(`Google inventory does not identify ${bundleId}`)
  requireInteger(inventory.apple.maxBuildNumber, `${bundleId} Apple maxBuildNumber`)
  requireInteger(inventory.google.maxVersionCode, `${bundleId} Google maxVersionCode`)
  if (!Array.isArray(inventory.apple.buildNumbers))
    throw new Error(`${bundleId} Apple inventory lists no build numbers`)
  if (!inventory.google.tracks || typeof inventory.google.tracks !== "object") {
    throw new Error(`${bundleId} Google inventory lists no tracks`)
  }
  if (!allowNoCurrent && (!inventory.apple.current || !Number.isSafeInteger(inventory.google.currentVersionCode))) {
    throw new Error(`${bundleId} has no current public store release`)
  }
  return inventory
}

// Build numbers the stores already hold inside one family's window. The global
// store maximum is never used for allocation: a single stray upload outside the
// window (a 900000001 once sat in App Store Connect) would otherwise drag every
// later candidate above it for good.
export function storeBuildNumbersWithin(inventory, window) {
  const inWindow = (value) => Number.isSafeInteger(value) && value >= window.first && value <= window.last
  const apple = inventory.apple.buildNumbers.filter(inWindow)
  const google = Object.values(inventory.google.tracks)
    .flat()
    .flatMap((release) => release?.versionCodes || [])
    .map(Number)
    .filter(inWindow)
  return [...apple, ...google]
}

function validateCurrentMentraApp(previousManifest, inventory) {
  if (previousManifest === null) {
    // First coordinated promotion: no mentra-vX.Y.Z release describes the public
    // app, so freeze exactly what both stores serve today. The app cannot be
    // rebuilt for the compatibility lab, but Phase 5 still verifies it against
    // production Cloud N+1 using these coordinates.
    const {marketingVersion, buildNumber} = inventory.apple.current
    return {
      provenance: "store-observed",
      sourceCommit: null,
      provenanceUrl: null,
      ios: {marketingVersion, buildNumber},
      android: {marketingVersion, buildNumber: inventory.google.currentVersionCode},
    }
  }
  const expected = previousManifest.native
  if (
    !expected ||
    inventory.apple.current.marketingVersion !== expected.marketingVersion ||
    inventory.apple.current.buildNumber !== expected.buildNumber ||
    inventory.google.currentVersionCode !== expected.buildNumber
  ) {
    throw new Error("Current App Store and Google Play builds do not match the previous production manifest")
  }
  if (!COMMIT_PATTERN.test(previousManifest.sourceCommit || "")) {
    throw new Error("Previous production manifest has no full source commit")
  }
  return {
    provenance: "coordinated",
    sourceCommit: previousManifest.sourceCommit,
    provenanceUrl: previousManifest.url,
    ios: {marketingVersion: expected.marketingVersion, buildNumber: expected.buildNumber},
    android: {marketingVersion: expected.marketingVersion, buildNumber: expected.buildNumber},
  }
}

// Shared by promotion preparation and stable package publication: the selected
// beta must be complete, internally consistent, pinned to an immutable OTA
// manifest, and belong to the checked-out release family.
export function validateSelectedBeta({family, betaPlan, betaManifest}) {
  if (
    betaPlan.channel !== "beta" ||
    betaManifest.channel !== "beta" ||
    betaPlan.releaseSetId !== betaManifest.releaseSetId ||
    betaPlan.releaseIdentity !== betaManifest.releaseIdentity ||
    betaPlan.sourceCommit !== betaManifest.sourceCommit ||
    betaManifest.releasePlanSha256 !== releaseRecordSha256(betaPlan)
  ) {
    throw new Error("Selected beta plan and completed manifest do not match")
  }
  if (betaPlan.familyBaseVersion !== family.familyBaseVersion) {
    throw new Error("Selected beta belongs to a different checked-out release family")
  }
  if (!betaManifest.completedAt) throw new Error("Selected beta is not complete")
  if (
    !/^https:\/\//.test(betaManifest.otaManifest?.url || "") ||
    !/^[0-9a-f]{64}$/.test(betaManifest.otaManifest?.sha256 || "")
  ) {
    throw new Error("Selected beta has no immutable OTA manifest pin")
  }
  return betaPlan
}

export function prepareProductionPromotion({
  family,
  betaPlan,
  betaManifest,
  betaManifestUrl,
  betaManifestSha256,
  previousManifest,
  mentraInventory,
  attempt,
  actor,
  createdAt,
  provenanceUrl,
}) {
  validateSelectedBeta({family, betaPlan, betaManifest})
  validateInventory(mentraInventory, {bundleId: "com.mentra.mentra", allowNoCurrent: false})
  const currentMentraApp = validateCurrentMentraApp(previousManifest, mentraInventory)
  const window = familyBuildNumberWindow(family.familyBaseVersion)
  if (betaPlan.native.buildNumber < window.first || betaPlan.native.buildNumber > window.last) {
    throw new Error(
      `Selected beta build number ${betaPlan.native.buildNumber} is outside the ${family.familyBaseVersion} family window`,
    )
  }
  const mentraBuildNumber =
    Math.max(...storeBuildNumbersWithin(mentraInventory, window), betaPlan.native.buildNumber) + 1
  if (mentraBuildNumber > window.last) {
    throw new Error(`Family ${family.familyBaseVersion} has exhausted its store build numbers`)
  }
  // The compatibility lab rebuilds the current public app, so its number lives
  // in that app's own family window, above what the stores hold there.
  const hasCompatibilityLab = currentMentraApp.provenance === "coordinated"
  let compatibilityLabBuildNumber = null
  if (hasCompatibilityLab) {
    const currentWindow = familyBuildNumberWindow(currentMentraApp.ios.marketingVersion)
    compatibilityLabBuildNumber =
      Math.max(
        ...storeBuildNumbersWithin(mentraInventory, currentWindow),
        currentMentraApp.ios.buildNumber,
        currentMentraApp.android.buildNumber,
      ) + 1
    if (compatibilityLabBuildNumber > currentWindow.last) {
      throw new Error(`Family ${currentMentraApp.ios.marketingVersion} has exhausted its store build numbers`)
    }
  }
  // Google Play only publishes a production release above the one it serves.
  if (mentraBuildNumber <= mentraInventory.google.currentVersionCode) {
    throw new Error(
      `Candidate build number ${mentraBuildNumber} is not above the Google Play production version code ${mentraInventory.google.currentVersionCode}`,
    )
  }
  const productionPlan = createReleasePlan({
    family,
    channel: "production",
    sourceCommit: betaPlan.sourceCommit,
    nativeBuildNumber: mentraBuildNumber,
    otaInputs: betaPlan.otaInputs,
  })
  productionPlan.promotion = {
    selectedBetaReleaseSetId: betaPlan.releaseSetId,
    selectedBetaIdentity: betaPlan.releaseIdentity,
    selectedBetaManifest: {url: betaManifestUrl, sha256: betaManifestSha256},
    otaManifest: betaManifest.otaManifest,
  }
  const record = createInitialPromotionRecord({
    releaseIdentity: productionPlan.releaseIdentity,
    attempt,
    selectedBeta: {
      identity: betaPlan.releaseIdentity,
      releaseSetId: betaPlan.releaseSetId,
      manifestUrl: betaManifestUrl,
      manifestSha256: betaManifestSha256,
    },
    source: {mentraosCommit: betaPlan.sourceCommit},
    coordinates: {
      currentMentraApp,
      compatibilityLab: hasCompatibilityLab
        ? {
            ios: {marketingVersion: currentMentraApp.ios.marketingVersion, buildNumber: compatibilityLabBuildNumber},
            android: {
              marketingVersion: currentMentraApp.android.marketingVersion,
              buildNumber: compatibilityLabBuildNumber,
            },
          }
        : null,
      candidates: {
        mentraApp: {
          ios: {marketingVersion: productionPlan.native.marketingVersion, buildNumber: mentraBuildNumber},
          android: {marketingVersion: productionPlan.native.marketingVersion, buildNumber: mentraBuildNumber},
        },
      },
    },
    actor,
    createdAt,
    provenanceUrl,
    evidence: [
      {
        kind: "selected-beta-manifest",
        url: betaManifestUrl,
        sha256: betaManifestSha256,
        assetName: path.basename(new URL(betaManifestUrl).pathname),
      },
    ],
  })
  return {productionPlan, record}
}

function parseArgs(args) {
  const values = {}
  for (let index = 0; index < args.length; index += 2) {
    const option = args[index]
    const value = args[index + 1]
    if (!option?.startsWith("--") || value === undefined) throw new Error("Expected --name value pairs")
    values[option.slice(2)] = value
  }
  return values
}

function readJson(file) {
  return JSON.parse(readFileSync(path.resolve(file), "utf8"))
}

function main() {
  const args = parseArgs(process.argv.slice(2))
  const betaManifestPath = path.resolve(args["beta-manifest"])
  let previousManifest = null
  if (args["previous-manifest"] || args["previous-manifest-url"]) {
    previousManifest = readJson(args["previous-manifest"])
    previousManifest.url = args["previous-manifest-url"]
  }
  const result = prepareProductionPromotion({
    family: loadReleaseFamily({rootDir: path.resolve(args.root || process.cwd()), requireVersionMirrors: true}),
    betaPlan: readJson(args["beta-plan"]),
    betaManifest: readJson(betaManifestPath),
    betaManifestUrl: args["beta-manifest-url"],
    betaManifestSha256: sha256File(betaManifestPath),
    previousManifest,
    mentraInventory: readJson(args["mentra-inventory"]),
    attempt: Number(args.attempt),
    actor: args.actor,
    createdAt: args["created-at"],
    provenanceUrl: args["provenance-url"],
  })
  writeFileSync(path.resolve(args["plan-output"]), serializeReleaseRecord(result.productionPlan))
  const recordDirectory = path.resolve(args["record-directory"])
  const recordFile = path.join(recordDirectory, promotionAssetName(result.record))
  mkdirSync(recordDirectory, {recursive: true})
  writeFileSync(recordFile, serializeReleaseRecord(result.record))
  console.log(recordFile)
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main()
