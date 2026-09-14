/// <reference types="bun-types" />

import {describe, expect, test} from "bun:test"
import {readFileSync} from "node:fs"

import {SoftapCallError, SOFTAP_STEPS} from "../SoftapCallTransport"
import {awaitCleanupBarrier} from "../SoftapCleanupBarrier"

// Exercise the actual private host methods without loading the Expo app singleton and all
// its hardware services. Only the I/O boundaries are faked; reservation, retirement, and the
// resource-ownership checks come directly from LocalMiniappRuntime's implementation.
const source = readFileSync(new URL("../LocalMiniappRuntime.ts", import.meta.url), "utf8")
const methods = [
  "joinSoftapMeeting",
  "createSoftapAttempt",
  "checkpointSoftapAttempt",
  "runSoftapAttempt",
  "retireSoftapAttempt",
  "teardownSoftapAttempt",
  "settleSoftapTeardown",
  "forceSoftapCleanup",
  "emitSoftapProgress",
]
  .map((name) => {
    const start = source.search(new RegExp(`^  private (?:async )?${name}\\(`, "m"))
    if (start < 0) throw new Error(`Missing runtime method ${name}`)
    const rest = source.slice(start)
    const end = rest.search(/^  }$/m)
    if (end < 0) throw new Error(`Missing end of runtime method ${name}`)
    return rest.slice(0, end + 3)
  })
  .join("\n")
const compiled = new Bun.Transpiler({loader: "ts"}).transformSync(`class Host { ${methods} }`)

function deferred() {
  let resolve!: () => void
  let reject!: (error: Error) => void
  const promise = new Promise<void>((yes, no) => {
    resolve = yes
    reject = no
  })
  return {promise, resolve, reject}
}
const tick = () => new Promise<void>((resolve) => setTimeout(resolve, 0))

/** The module-private helper the extracted methods call; same shape, kept trivial on purpose. */
function withTimeout<T>(work: Promise<T>, ms: number, reason: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(reason)), ms)
    work.then(
      (value) => {
        clearTimeout(timer)
        resolve(value)
      },
      (error) => {
        clearTimeout(timer)
        reject(error)
      },
    )
  })
}

/**
 * Same extraction as the lifecycle methods: the settle/force gates are pulled from the source so
 * the teardown these tests drive is the one that ships, including the two waits (ingest close and
 * hotspot ack) that a fake host would otherwise silently skip.
 */
function fixture(
  gates: {ingestClosed?: boolean; hotspot?: "disabled" | "enabled" | "throw"} = {},
): ReturnType<typeof buildFixture> {
  return buildFixture(gates)
}

function buildFixture(gates: {ingestClosed?: boolean; hotspot?: "disabled" | "enabled" | "throw"}) {
  const cleanup = deferred()
  let nativeReleases = 0
  let preflights = 0
  let forcedIngestCloses = 0
  const hotspotCommands: boolean[] = []
  const native = {
    beginScopedTeardown() {},
    async leaveScopedNetwork() {
      nativeReleases++
      if (nativeReleases === 1) await cleanup.promise
    },
    async awaitValidatedDefaultNetwork() {
      return {usable: true, detail: "cellular"}
    },
    async leaveAndAwait() {
      return {completed: true}
    },
    async awaitIngestClosed() {
      // After a forced close the port really is gone, so the re-verify answers true even when the
      // first bounded wait did not.
      return gates.ingestClosed !== false || forcedIngestCloses > 0
    },
    async forceCloseIngest() {
      forcedIngestCloses++
    },
  }
  const bluetooth = {
    async setHotspotState(enabled: boolean) {
      hotspotCommands.push(enabled)
      if (gates.hotspot === "throw") throw new Error("glasses did not answer")
      return {state: gates.hotspot === "enabled" ? "enabled" : "disabled"}
    },
  }
  const permissions = {
    async check() {
      preflights++
      // End at the first I/O boundary: these tests verify that getting here is safely ordered.
      throw new Error("test preflight ended")
    },
  }
  const Host = new Function(
    "acquireGlassesHotspot",
    "softapTrace",
    "softapTraceFailure",
    "SOFTAP_CLEANUP_STALL_LOG_MS",
    "SOFTAP_CLEANUP_NARRATE_AFTER_MS",
    "SoftapCallError",
    "SOFTAP_STEPS",
    "awaitCleanupBarrier",
    "acsMeetingService",
    "permissions",
    "PermissionFeatures",
    "console",
    "BluetoothSdk",
    "withTimeout",
    "SOFTAP_INGEST_CLOSE_WAIT_MS",
    "SOFTAP_HOTSPOT_OFF_ACK_MS",
    "SOFTAP_FORCED_VERIFY_MS",
    `${compiled}; return Host`,
  )(
    () => () => {},
    () => {},
    () => {},
    10_000,
    0,
    SoftapCallError,
    SOFTAP_STEPS,
    awaitCleanupBarrier,
    native,
    permissions,
    {LOCAL_WIFI: "wifi"},
    {log() {}, warn() {}},
    bluetooth,
    withTimeout,
    50,
    50,
    50,
  )
  const host = new Host()
  host.softapAttemptSeq = 0
  host.softapCleanupError = null
  host.narrateSoftapPreflight = () => {}
  const old = host.createSoftapAttempt("com.mentra.call")
  old.ownsResources = true
  old.body = Promise.resolve()
  host.softapAttempt = old
  const join = () =>
    host.joinSoftapMeeting("com.mentra.call", {}).then(
      () => "unexpected success",
      (error: Error) => error.message,
    )
  return {
    host,
    old,
    cleanup,
    join,
    preflights: () => preflights,
    nativeReleases: () => nativeReleases,
    forcedIngestCloses: () => forcedIngestCloses,
    hotspotCommands: () => hotspotCommands,
  }
}

describe("SoftAP host attempt lifecycle", () => {
  test("a new join waits for an explicitly retiring call", async () => {
    const f = fixture()
    const leave = f.host.retireSoftapAttempt()
    const join = f.join()
    await tick()
    expect(f.preflights()).toBe(0)
    expect(f.nativeReleases()).toBe(1)
    f.cleanup.resolve()
    await leave
    expect(await join).toBe("test preflight ended")
    expect(f.preflights()).toBe(1)
  })

  test("finished teardown still waits for the previous native join body", async () => {
    const f = fixture()
    const body = deferred()
    f.old.body = body.promise
    const leave = f.host.retireSoftapAttempt()
    f.cleanup.resolve()
    await leave
    const join = f.join()
    await tick()
    expect(f.preflights()).toBe(0)
    body.resolve()
    expect(await join).toBe("test preflight ended")
  })

  test("multiple Starts and Cancel while cleanup is pending acquire no resources", async () => {
    const f = fixture()
    const leave = f.host.retireSoftapAttempt()
    const second = f.join()
    const third = f.join()
    await f.host.retireSoftapAttempt()
    await tick()
    expect(f.preflights()).toBe(0)
    expect(f.nativeReleases()).toBe(1)
    f.cleanup.resolve()
    await leave
    expect(await second).toContain("cancelled")
    expect(await third).toContain("cancelled")
    await tick()
    expect(f.preflights()).toBe(0)
    expect(f.nativeReleases()).toBe(1)
    expect(f.host.softapAttempt).toBeNull()
  })

  test("only the latest queued Start proceeds after the previous cleanup", async () => {
    const f = fixture()
    const leave = f.host.retireSoftapAttempt()
    const second = f.join()
    const third = f.join()
    await tick()
    expect(f.preflights()).toBe(0)
    expect(f.nativeReleases()).toBe(1)
    f.cleanup.resolve()
    await leave
    expect(await second).toContain("cancelled")
    expect(await third).toBe("test preflight ended")
    expect(f.preflights()).toBe(1)
    expect(f.nativeReleases()).toBe(2)
  })

  test("a cleanup failure blocks the waiting join before it acquires resources", async () => {
    const f = fixture()
    const leave = f.host.retireSoftapAttempt()
    const join = f.join()
    f.cleanup.reject(new Error("radio did not release"))
    await leave
    expect(await join).toContain("Previous call cleanup failed")
    expect(f.preflights()).toBe(0)
    expect(f.nativeReleases()).toBe(1)
  })
})

/**
 * The two gates that stop-then-start actually fails on: the WHIP listener still inside its
 * tombstone, and a glasses hotspot that is still up. Both are bounded waits, and the point of
 * these tests is that an expired bound is never read as "released".
 */
describe("SoftAP teardown barrier", () => {
  test("an ingest port still held is taken by force and re-verified, not assumed free", async () => {
    const f = fixture({ingestClosed: false})
    f.cleanup.resolve()

    await f.host.retireSoftapAttempt()

    expect(f.forcedIngestCloses()).toBe(1)
    // Recovered, so nothing is recorded against the next call.
    expect(f.host.softapCleanupError).toBeNull()
    expect(await f.join()).toBe("test preflight ended")
  })

  test("glasses that never acknowledge hotspot off refuse the next call by name", async () => {
    const f = fixture({hotspot: "throw"})
    f.cleanup.resolve()

    await f.host.retireSoftapAttempt()

    // Once in the settle gate, once more in forced cleanup — the second is the re-ask, not a retry
    // loop, and it is what makes the failure below a fact rather than a timeout.
    expect(f.hotspotCommands()).toEqual([false, false])
    expect(f.host.softapCleanupError).toContain("glasses hotspot off")
    expect(await f.join()).toContain("Previous call cleanup failed")
  })

  test("a hotspot that answers 'enabled' is a failure, not an acknowledgement", async () => {
    const f = fixture({hotspot: "enabled"})
    f.cleanup.resolve()

    await f.host.retireSoftapAttempt()

    expect(f.host.softapCleanupError).toContain("still enabled")
  })
})

/**
 * The outer fence. Native has its own generations inside a session; this one exists because a
 * callback can arrive after the attempt it belongs to stopped owning the call at all.
 */
describe("SoftAP attempt fencing", () => {
  test("a stage that completes after the attempt was superseded cannot continue", () => {
    const f = fixture()
    const retired = f.host.softapAttempt
    f.host.softapAttempt = f.host.createSoftapAttempt("com.mentra.call")

    expect(() => f.host.checkpointSoftapAttempt(retired, "acs_join")).toThrow(/cancelled/)
  })

  test("a progress snapshot from a retired attempt is dropped instead of drawn", () => {
    const f = fixture()
    const retired = f.host.softapAttempt
    const sent: unknown[] = []
    f.host.sendToMiniapp = (_pkg: string, message: unknown) => sent.push(message)
    retired.cancelled = true

    f.host.emitSoftapProgress(retired, {phase: "hotspot", traceId: "t-1", steps: []})

    expect(sent).toEqual([])
  })
})

/**
 * The OS-permission gate at join. Extracted the same way as the lifecycle methods above, because
 * what is being asserted is the difference between `check` and `request` — and that difference is
 * invisible to any test that stubs the whole method.
 */
function permissionGateFixture(granted: boolean) {
  const start = source.search(/^ {2}private async requireOsPermission\(/m)
  if (start < 0) throw new Error("Missing runtime method requireOsPermission")
  const rest = source.slice(start)
  const end = rest.search(/^ {2}}$/m)
  const body = new Bun.Transpiler({loader: "ts"}).transformSync(`class Host { ${rest.slice(0, end + 3)} }`)
  const requested: string[] = []
  const results: Array<{ok: boolean; error?: {code: string; message: string; permission?: string}}> = []
  const Host = new Function(
    "permissions",
    "softapTraceFailure",
    "MiniappErrorCode",
    "MiniappRequestType",
    `${body}; return Host`,
  )(
    {
      async check() {
        return granted
      },
      async request(feature: string) {
        requested.push(feature)
        return true
      },
    },
    () => {},
    {PERMISSION_DENIED: "PERMISSION_DENIED"},
    {MEETING_JOIN: "meeting:join"},
  )
  const host = new Host()
  host.sendResult = (
    _pkg: string,
    _id: string,
    ok: boolean,
    _data: unknown,
    error?: {code: string; message: string; permission?: string},
  ) => results.push({ok, error})
  return {host, requested, results}
}

describe("SoftAP join permission gate", () => {
  test("a denied camera refuses the join and never opens a prompt", async () => {
    const f = permissionGateFixture(false)

    const allowed = await f.host.requireOsPermission("com.mentra.call", "req-1", "camera", "camera")

    expect(allowed).toBe(false)
    // The prompt belongs at miniapp-open time. One here would sit on top of a join that is
    // already building a hotspot, and a wearer who denies it gets a half-existing call.
    expect(f.requested).toEqual([])
    expect(f.results[0]?.ok).toBe(false)
    expect(f.results[0]?.error?.code).toBe("PERMISSION_DENIED")
    expect(f.results[0]?.error?.permission).toBe("camera")
    expect(f.results[0]?.error?.message).toMatch(/Settings/)
  })

  test("a granted permission passes without sending a result", async () => {
    const f = permissionGateFixture(true)

    expect(await f.host.requireOsPermission("com.mentra.call", "req-1", "camera", "camera")).toBe(true)
    expect(f.results).toEqual([])
  })
})
