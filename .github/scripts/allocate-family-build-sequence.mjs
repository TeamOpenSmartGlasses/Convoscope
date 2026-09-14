#!/usr/bin/env node
// Allocates the one build number a coordinated run uses for the Mentra App and,
// when it rebuilds, the ASG client: the next free sequence of the family above
// every number already recorded in the family's build container, whether by an
// earlier run's marker (mentra-build-number-<code>.json) or by an ASG client
// pair (mentra-live-asg-<code>-<fingerprint>.*). Sequences restart at 1 for
// every family; see notes/superpowers/specs/2026-09-14-family-build-numbers.md.
import {mkdirSync, readFileSync, writeFileSync} from "node:fs"
import path from "node:path"
import {fileURLToPath} from "node:url"

import {BUILD_NUMBER_RELEASE_SEQUENCE_LIMIT, familyBuildNumberWindow} from "./release-family.mjs"

const MARKER_PATTERN = /^mentra-build-number-(\d+)\.json$/
const ASG_PATTERN = /^mentra-live-asg-(\d+)-[0-9a-f]{64}\.(?:apk|json)$/

export function markerAssetName(buildNumber) {
  return `mentra-build-number-${buildNumber}.json`
}

export function recordedFamilyBuildNumbers(assets, baseVersion) {
  if (!Array.isArray(assets)) throw new Error("GitHub release assets must be an array")
  const window = familyBuildNumberWindow(baseVersion)
  const numbers = new Set()
  for (const asset of assets) {
    const match = MARKER_PATTERN.exec(asset?.name ?? "") || ASG_PATTERN.exec(asset?.name ?? "")
    if (!match) continue
    const code = Number(match[1])
    if (code >= window.first && code <= window.last) numbers.add(code)
  }
  return [...numbers].sort((left, right) => left - right)
}

export function allocateFamilyBuildNumber({assets, baseVersion}) {
  const window = familyBuildNumberWindow(baseVersion)
  const recorded = recordedFamilyBuildNumbers(assets, baseVersion)
  const buildNumber = recorded.length === 0 ? window.first : recorded.at(-1) + 1
  const sequence = buildNumber - window.prefix
  if (sequence > BUILD_NUMBER_RELEASE_SEQUENCE_LIMIT) {
    throw new Error(`Family ${baseVersion} has exhausted its release build numbers`)
  }
  return {familyBaseVersion: baseVersion, buildNumber, sequence, markerAsset: markerAssetName(buildNumber)}
}

// The marker recorded in the container: deterministic, so a rerun republishes
// identical bytes and the immutable publisher accepts it.
export function familyBuildNumberMarker({baseVersion, buildNumber}) {
  const window = familyBuildNumberWindow(baseVersion)
  if (!Number.isSafeInteger(buildNumber) || buildNumber < window.first || buildNumber > window.last) {
    throw new Error(`Build number ${buildNumber} does not belong to family ${baseVersion}`)
  }
  return {
    schemaVersion: 1,
    kind: "mentra-family-build-number",
    familyBaseVersion: baseVersion,
    buildNumber,
    sequence: buildNumber - window.prefix,
  }
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

function main() {
  const command = process.argv[2]
  const args = parseArgs(process.argv.slice(3))
  if (command === "allocate") {
    const result = allocateFamilyBuildNumber({
      assets: JSON.parse(readFileSync(path.resolve(args.assets), "utf8")),
      baseVersion: args["base-version"],
    })
    writeFileSync(path.resolve(args.output), `${JSON.stringify(result, null, 2)}\n`)
    console.log(
      `Allocated ${result.familyBaseVersion} build number ${result.buildNumber} (sequence ${result.sequence})`,
    )
    return
  }
  if (command === "marker") {
    const marker = familyBuildNumberMarker({
      baseVersion: args["base-version"],
      buildNumber: Number(args["build-number"]),
    })
    const directory = path.resolve(args["output-dir"])
    mkdirSync(directory, {recursive: true})
    const file = path.join(directory, markerAssetName(marker.buildNumber))
    writeFileSync(file, `${JSON.stringify(marker, null, 2)}\n`)
    console.log(file)
    return
  }
  throw new Error(`Unknown family build number command ${JSON.stringify(command)}`)
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main()
