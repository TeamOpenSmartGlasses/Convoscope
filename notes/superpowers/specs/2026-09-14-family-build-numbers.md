---
status: active
owner: philippe
---

# Family build numbers

One formula for every store build number in a release family, for the Mentra
App (iOS build number and Android version code) and the ASG client (Android
version code) alike, so a number says which release it belongs to and every
channel of a family orders naturally on both stores.

## Formula

```
buildNumber = MAJOR × 100,000,000 + MINOR × 1,000,000 + PATCH × 10,000 + sequence
```

- `MAJOR` 2..20, `MINOR` 0..99, `PATCH` 0..99. The largest code, 20.99.99 with
  sequence 9,999, is 2,099,999,999, under Android's 2,100,000,000 limit. Major 20
  is decades away; a wider major would strand every device in the field below
  the 2.1 billion cap, and was rejected.
- The window of a family `X.Y.Z` is `prefix + 1 .. prefix + 9,999`.
- Release sequences (dev, beta, production) use 1..2,999; non-release builds use
  3,000..9,999 (see below).
- Two legacy namespaces sit below every window and are ignored by allocation:
  the timestamp scheme of the pre-coordinated releases (below 60 million) and the
  first coordinated ASG allocator's `100,000,000 + run number`. The Mentra App's
  3.1.0 betas and the first 3.2.0 dev builds used a flat `310,000,000 + run
  number`, above their families' windows; Android testers on those builds
  reinstall once. Glasses are never a floor: the phone can downgrade the ASG
  client through the detour (reinstall the firmware's system app, then upgrade to
  the target).

## Sequences restart per family

Every new family prefix starts again at 1. A coordinated run (dev or beta)
allocates **one sequence per run** from the family's build container
(`mentra-builds-vX.Y.Z`): the next free number above every number already
recorded there, whether recorded by the app (`mentra-build-number-<code>.json`
marker assets) or by the ASG client (`mentra-live-asg-<code>-<fingerprint>.*`
assets). The marker is published right after the container exists and before
any build starts, so a run that fails halfway never gives its number away, and
a rerun reuses the number from its restored plan. Two runs of the same family
allocating concurrently collide on the marker name and the second one fails
closed.

Release identities (`3.1.1-beta.235`) keep the coordinated run number: that is
only a name, not a build number.

### The ASG client shares the run's number

In a coordinated run the ASG client's fingerprint (its sources plus the family
base version) is looked up in the container:

- fingerprint already published: the run reuses that APK and its recorded code,
  no build, no new number;
- new fingerprint: the run builds and stamps the APK with the run's sequence,
  the same number the app gets in that run;
- rerun: same commit, same fingerprint, so the published pair is reused; an
  interrupted pair is deleted and rebuilt.

App and ASG codes are therefore equal whenever both were built in the same run,
and both live in the family window otherwise.

### Production candidates

Production is one more run of the family: the promotion's prepare step takes
the next family sequence from the same container and records its marker there
before anything is built, so the candidate is above the promoted beta and above
every earlier build of the family, and no later coordinated run can take the
number. The store inventories are still frozen for the record of the current
public app, but they play no part in allocation; the family container is the
single source of truth and nothing higher is expected on the stores. The
production Bluetooth example allocates the same way. A compatibility-lab
rebuild of the current public app takes the next sequence of that app's own
family, from that family's container. Google Play production must still be
exceeded, which the family window guarantees over the legacy timestamp codes.

## Non-release builds (local and PR CI) — after the 3.1.1 release

Local builds and PR CI builds of the app and the ASG client pin

```
sequence = 3,000 + (minutes since 2025-01-01 of the HEAD committer time) mod 7,000
```

- Derived from the commit, so the app and the ASG client built from the same
  commit share the code, in CI and on a laptop, without any shared counter.
- Always above every release of the family: a dev build installs over the store
  app without an uninstall, and a PR ASG build is an OTA update for glasses on a
  release build. Going back down costs one uninstall on Android (or
  `adb install -r -d` for debug builds) and the detour on glasses.
- Minutes, not seconds: with 7,000 slots the sequence wraps every 4.9 days,
  which covers iterating on a PR with glasses attached; seconds would wrap every
  two hours. Two commits in the same minute share a code, which only matters
  for a glasses OTA between them.
- Uniqueness beyond that is not a goal: Android accepts equal codes on install,
  and crash reports carry the commit hash.

The ASG PR workflow keeps reusing the coordinated ASG client when the client is
unchanged in the PR.

## Where it lives

- `.github/scripts/release-family.mjs`: prefix, window, `familyBuildNumber`,
  `buildNumberBelongsTo`, the release-sequence limit.
- `.github/scripts/allocate-family-build-sequence.mjs`: per-run allocation from
  the family container (coordinated plan job).
- `.github/scripts/allocate-asg-version.mjs`: ASG reuse or allocation at the
  run's number.
- `.github/scripts/store-build-numbers.mjs`: production and example allocation
  from the store inventories.
- `mobile/scripts/build-number.mjs`, `asg_client/app/build.gradle`: pinned by
  CI; the non-release band for local builds (follow-up).
