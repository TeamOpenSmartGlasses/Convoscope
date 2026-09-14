#!/usr/bin/env node
import {readFileSync, writeFileSync} from "node:fs"
import path from "node:path"
import {fileURLToPath} from "node:url"

import {BUILD_NUMBER_MAX_SEQUENCE, familyBuildNumberPrefix, familyBuildNumberWindow} from "./release-family.mjs"

const ASSET_PATTERN = /^mentra-live-asg-(\d+)-([0-9a-f]{64})\.(apk|json)$/
// ASG version codes use the family build-number formula shared with the Mentra
// App (see release-family.mjs): the family prefix plus a sequence. The
// coordinated run number is the sequence, so an ASG rebuilt in a run carries
// the same number as the app built in that run; a fingerprint already built
// keeps its recorded code. Assets outside the family window belong to older
// schemes and are ignored.
const MAX_SEQUENCE = BUILD_NUMBER_MAX_SEQUENCE

export function asgVersionCodePrefix(baseVersion) {
  return familyBuildNumberPrefix(baseVersion)
}

export function allocateAsgVersion({assets, fingerprint, baseVersion, sequence}) {
  if (!Array.isArray(assets)) throw new Error("GitHub release assets must be an array")
  if (!/^[0-9a-f]{64}$/.test(fingerprint)) throw new Error("Invalid ASG fingerprint")
  if (!Number.isSafeInteger(sequence) || sequence < 1 || sequence > MAX_SEQUENCE) {
    throw new Error(`ASG build sequence ${JSON.stringify(sequence)} must be between 1 and ${MAX_SEQUENCE}`)
  }
  const window = familyBuildNumberWindow(baseVersion)
  const prefix = window.prefix
  const recognized = assets.flatMap((asset) => {
    const match = ASSET_PATTERN.exec(asset.name ?? "")
    if (!match) return []
    const versionCode = Number(match[1])
    if (versionCode < window.first || versionCode > window.last) return []
    return [{id: asset.id, name: asset.name, versionCode, fingerprint: match[2], type: match[3]}]
  })
  const matching = recognized.filter((asset) => asset.fingerprint === fingerprint)
  const apks = matching.filter((asset) => asset.type === "apk")
  const provenance = matching.filter((asset) => asset.type === "json")
  if (apks.length > 1 || provenance.length > 1) throw new Error("Duplicate immutable ASG release assets found")
  if (apks.length === 1 && provenance.length === 1) {
    if (apks[0].versionCode !== provenance[0].versionCode) {
      throw new Error("ASG artifact and provenance use different version codes")
    }
    return {
      exists: true,
      versionCode: apks[0].versionCode,
      apkAsset: apks[0].name,
      provenanceAsset: provenance[0].name,
      orphanAssetIds: [],
    }
  }
  // The run number is the sequence; when a rerun or an older run would land
  // on or below a code already used for another build, take the next free one
  // so codes stay unique and increasing within the family.
  const usedSequences = recognized.map((asset) => asset.versionCode - prefix)
  const highestUsed = usedSequences.length === 0 ? 0 : Math.max(...usedSequences)
  const allocated = Math.max(sequence, highestUsed + 1)
  if (allocated > MAX_SEQUENCE) throw new Error(`Base version ${baseVersion} has exhausted its ASG version codes`)
  const versionCode = prefix + allocated
  return {
    exists: false,
    versionCode,
    apkAsset: `mentra-live-asg-${versionCode}-${fingerprint}.apk`,
    provenanceAsset: `mentra-live-asg-${versionCode}-${fingerprint}.json`,
    orphanAssetIds: matching.map((asset) => asset.id),
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
  const args = parseArgs(process.argv.slice(2))
  const result = allocateAsgVersion({
    assets: JSON.parse(readFileSync(path.resolve(args.assets), "utf8")),
    fingerprint: args.fingerprint,
    baseVersion: args["base-version"],
    sequence: Number(args.sequence),
  })
  writeFileSync(path.resolve(args.output), `${JSON.stringify(result, null, 2)}\n`)
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main()
