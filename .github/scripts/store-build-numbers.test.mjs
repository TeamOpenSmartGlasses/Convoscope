import assert from "node:assert/strict"
import test from "node:test"

import {familyBuildNumber} from "./release-family.mjs"
import {allocateStoreBuildNumber, applePriorBuildOfVersion, storeBuildNumbersWithin} from "./store-build-numbers.mjs"
import {familyBuildNumberWindow} from "./release-family.mjs"

const apple = (builds) => ({builds})
const google = (tracks) => ({tracks})

test("collects only the store numbers inside a family window, from both stores", () => {
  const window = familyBuildNumberWindow("3.1.1")
  const numbers = storeBuildNumbersWithin(
    {
      apple: apple([
        {buildNumber: 900000001, marketingVersion: "3.1.0"},
        {buildNumber: familyBuildNumber("3.1.1", 5), marketingVersion: "3.1.1"},
        {buildNumber: familyBuildNumber("3.2.0", 9), marketingVersion: "3.2.0"},
      ]),
      google: google({internal: [familyBuildNumber("3.1.1", 7)], production: [50572796]}),
    },
    window,
  )
  assert.deepEqual(numbers.sort(), [familyBuildNumber("3.1.1", 5), familyBuildNumber("3.1.1", 7)])
  assert.deepEqual(storeBuildNumbersWithin({apple: apple([]), google: null}, window), [])
  assert.throws(() => storeBuildNumbersWithin({apple: {}, google: null}, window), /lists no builds/)
  assert.throws(() => storeBuildNumbersWithin({apple: apple([]), google: {tracks: []}}, window), /lists no tracks/)
  assert.throws(
    () => storeBuildNumbersWithin({apple: apple([{buildNumber: "x"}]), google: null}, window),
    /without a numeric build number/,
  )
})

test("allocates above the window's store numbers, the floor, and the version string's own history", () => {
  const beta = familyBuildNumber("3.1.1", 230)
  const inventory = {
    apple: apple([
      {buildNumber: 900000002, marketingVersion: "3.1.0"},
      {buildNumber: familyBuildNumber("3.1.1", 231), marketingVersion: "3.1.1"},
    ]),
    google: google({internal: [900000002, familyBuildNumber("3.1.1", 240)], production: [50572796]}),
  }
  assert.equal(
    allocateStoreBuildNumber({marketingVersion: "3.1.1", ...inventory, atLeast: [beta]}),
    familyBuildNumber("3.1.1", 241),
  )
  assert.equal(
    allocateStoreBuildNumber({marketingVersion: "3.1.1", apple: apple([]), google: null, atLeast: [beta]}),
    familyBuildNumber("3.1.1", 231),
  )
  assert.equal(
    allocateStoreBuildNumber({marketingVersion: "3.2.0", apple: apple([]), google: null}),
    familyBuildNumber("3.2.0", 1),
  )
  assert.equal(applePriorBuildOfVersion(inventory.apple, "3.1.0"), 900000002)
  assert.equal(applePriorBuildOfVersion(inventory.apple, "4.0.0"), 0)
  // The poisoned 3.1.0 train: App Store Connect holds 900000002 under "3.1.0".
  assert.throws(
    () =>
      allocateStoreBuildNumber({marketingVersion: "3.1.0", ...inventory, atLeast: [familyBuildNumber("3.1.0", 212)]}),
    /already holds build 900000002 for version 3\.1\.0, above its family window/,
  )
  assert.throws(
    () =>
      allocateStoreBuildNumber({
        marketingVersion: "3.1.1",
        apple: apple([{buildNumber: familyBuildNumber("3.1.1", 99999), marketingVersion: "3.1.1"}]),
        google: null,
      }),
    /exhausted/,
  )
})
