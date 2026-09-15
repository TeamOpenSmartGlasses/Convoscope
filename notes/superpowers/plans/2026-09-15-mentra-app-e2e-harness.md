---
status: active
owner: Philippe
---

# Mentra App E2E harness implementation

Design: [TestFlight-on-Mac harness](../specs/2026-09-15-mentra-app-e2e-harness.md).
English checklist: [ROUTINE.md](../../../tools/mentra-e2e/ROUTINE.md).

## Current state

- [x] Inspect the real TestFlight app, source, and native accessibility tree.
- [x] Isolate work on `codex/mentra-e2e-harness` from `origin/dev` at `95dfa6c4b2f341917b687798ff7b0065365d3115`.
- [x] Write the justified design and 40-step English routine.
- [x] Implement the Swift accessibility driver, Bun runner, snapshots and failure reporting.
- [x] Pass the original seven-step login/validation proof twice from a terminal.
- [x] Prove AXValue reaches real validation and authorized sign-in; redact secure values.
- [x] Add continuous target-window video, timestamped English chapters and per-step PNGs from the same stream.
- [x] Observe unpaired home, Gallery guard, Settings, Profile, account forms, logout cancellation and Feedback.
- [x] Confirm Settings/Profile interactions can run while another app retains desktop focus.
- [x] Remove coordinate/visual fallbacks, global input and foreground activation from the driver, per the user's direction.
- [x] Add labels, IDs and accessible activation to the missing app controls; add mobile tray activation tests.
- [x] Write Mac Mini provisioning, permission, credential, transfer and troubleshooting instructions.
- [x] Verify the old binary fails the new capsule accessibility preflight, exits nonzero and retains screenshot/video evidence.
- [ ] Verify app accessibility source changes in an installed build.
- [ ] Finish all required English walkthrough steps and encode the observed behavior.
- [ ] Qualify recording reattachment after normal app relaunch and fixture cleanup.
- [ ] Verify the HTML chapter viewer in a browser and final screenshot/video timing across a complete run. Browser automation rejected opening the local file under its URL policy; no workaround was attempted.
- [ ] Qualify three complete no-glasses replays with no model calls or manual corrections.
- [ ] Complete failure-path coverage for permission denial, target ambiguity and cleanup.

## Current dependency

The installed app is `com.mentra.mentra`, version `3.2.0`, build `320000235` (Settings: dev.235). It predates this branch's app accessibility changes. The read-only `accessibility-preflight` suite correctly reports missing `miniapp.minimize`. Install the updated binary before qualifying those controls; do not reintroduce coordinate, glyph or visual targeting to get around it.

The app is signed in, unpaired, on the empty Give Feedback form. Nothing was submitted. Real sign-in reached first-run onboarding, so the initial login suite's home assertion failed. Fixture setup reached home but its recording reattach failed. Preserve both failures as evidence; neither is a full suite pass.

## Next steps

1. Build/install the app with the source accessibility changes. Inspect each required identifier and invoke the real action as specified in [ACCESSIBILITY.md](../../../tools/mentra-e2e/ACCESSIBILITY.md).
2. Resume from the Feedback form via the named minimize control. Complete Speech, Privacy, Miniapp Developer Settings, all-apps search, switching, pairing cancellation, and no-glasses guard checks.
3. Encode only verified steps. Split conditional fixtures explicitly, and keep inactive feature coverage as declared not-applicable rather than silently skipped assertions.
4. Finish authentication validation, logout/sign-in restoration, app relaunch capture and cleanup verification.
5. Run three complete terminal replays and bounded failure checks. Finalize the routine and README with the exact build, harness revision and retained run folders.

## Validation so far

- Swift helper compiles; doctor confirms Accessibility and Screen Recording permissions.
- Seven native negative checks and one redaction test pass; two mobile tray accessibility activation tests pass.
- Harness TypeScript type-check passes. Mobile type-check reports only an existing `mobile/app.config.ts:107` nullable-name error when Expo's generated global types are included; the file is unchanged.
- Current preflight failure evidence: `.test-results/mentra-e2e/2026-09-15T21-43-42-280Z-accessibility-preflight-045bcd/`. H.264, 576×1090, 45 frames, 3.086667 seconds; the screenshot is readable and chapter timestamps lie inside the recording.
- All earlier failures and discoveries remain local. Historical discovery metadata saying zero model calls describes the runner only; those sessions were agent-directed and are not unattended qualification.

## Completion criterion

The initial harness is complete only when the full terminal routine has no model dependency, every executed step has readable evidence, behavior assertions fail correctly, fixture cleanup is verified, and all required coverage is implemented and passes. Current progress does not yet satisfy that criterion.
