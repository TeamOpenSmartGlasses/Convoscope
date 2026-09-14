// Store-side build number facts used by every production allocator: which
// numbers the stores already hold inside one family's window, and whether App
// Store Connect leaves room for a new build of a marketing version.
//
// Inventory contract (produced by mobile/scripts/app-store-connect-build.mjs
// inventory and mobile/ci/fastlane-android/Fastfile google_play_inventory):
//   apple.builds:  [{buildNumber, marketingVersion}] for every build the app holds
//   google.tracks: {track: [versionCode, ...]} for every track's releases
import {familyBuildNumberWindow} from "./release-family.mjs"

export function appleBuilds(apple, label = "Apple inventory") {
  if (!Array.isArray(apple?.builds)) throw new Error(`${label} lists no builds`)
  for (const build of apple.builds) {
    if (!Number.isSafeInteger(build?.buildNumber))
      throw new Error(`${label} has a build without a numeric build number`)
  }
  return apple.builds
}

export function googleVersionCodes(google, label = "Google inventory") {
  if (!google?.tracks || typeof google.tracks !== "object" || Array.isArray(google.tracks)) {
    throw new Error(`${label} lists no tracks`)
  }
  const codes = Object.values(google.tracks).flat()
  for (const code of codes) {
    if (!Number.isSafeInteger(code)) throw new Error(`${label} has a non-numeric version code`)
  }
  return codes
}

export function storeBuildNumbersWithin({apple, google}, window) {
  const inWindow = (value) => value >= window.first && value <= window.last
  return [
    ...appleBuilds(apple).map((build) => build.buildNumber),
    ...(google === null ? [] : googleVersionCodes(google)),
  ].filter(inWindow)
}

// The highest build App Store Connect already holds under this marketing
// version; a new build of that version string must exceed it.
export function applePriorBuildOfVersion(apple, marketingVersion) {
  const numbers = appleBuilds(apple)
    .filter((build) => build.marketingVersion === marketingVersion)
    .map((build) => build.buildNumber)
  return numbers.length === 0 ? 0 : Math.max(...numbers)
}

// Allocate the next store build number for `marketingVersion` (a family base
// version) inside its family window: above everything both stores hold in the
// window, above `atLeast` (the promoted beta, the current app, ...), and above
// every App Store Connect build of that version string. When a stray upload
// under the same version string sits above the window, Apple would refuse any
// candidate the window can offer, and the family version itself has to move.
export function allocateStoreBuildNumber({
  marketingVersion,
  apple,
  google,
  atLeast = [],
  reserve = 1,
  label = marketingVersion,
}) {
  const window = familyBuildNumberWindow(marketingVersion)
  const prior = applePriorBuildOfVersion(apple, marketingVersion)
  const buildNumber =
    Math.max(...storeBuildNumbersWithin({apple, google}, window), ...atLeast, prior, window.prefix) + reserve
  if (prior > window.last || buildNumber > window.last) {
    throw new Error(
      prior > window.last
        ? `App Store Connect already holds build ${prior} for version ${marketingVersion}, above its family window; release under a new family version`
        : `Family ${label} has exhausted its store build numbers`,
    )
  }
  return buildNumber
}
