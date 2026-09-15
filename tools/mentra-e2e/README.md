# Mentra App E2E harness

Start with the [40-step English routine](ROUTINE.md), [design and technology choices](../../notes/superpowers/specs/2026-09-15-mentra-app-e2e-harness.md), [app accessibility contract](ACCESSIBILITY.md), and [Mac Mini setup guide](SETUP.md).

This is an in-progress harness for the real TestFlight iOS app on an Apple Silicon Mac. The native Swift helper uses accessibility actions. Bun runs deterministic TypeScript steps without model calls. Each run saves a continuous video, searchable English chapter links, per-step PNG screenshots, assertions and accessibility snapshots.

**The full no-glasses routine is not qualified or fully encoded yet.** App source changes in this branch fix the inaccessible navigation controls discovered during the first walkthrough. Install a build containing them before continuing qualification; the installed dev.235 binary cannot acquire those changes from this checkout.

## Commands

From the repository root:

```sh
bun run tools/mentra-e2e/run.ts doctor
bun run tools/mentra-e2e/run.ts inspect
# With a miniapp open: read-only check of its accessible capsule controls.
bun run tools/mentra-e2e/run.ts run --suite accessibility-preflight
# From the signed-out welcome screen, on a build with navigation.back:
bun run tools/mentra-e2e/run.ts run --suite driver-proof
# From the signed-out welcome screen, with onboarding already completed:
bun run tools/mentra-e2e/run.ts run --suite login
```

Login prompts for credentials without echoing them. For automation, supply `MENTRA_E2E_EMAIL` and `MENTRA_E2E_PASSWORD` through an existing secret manager. Never put a password in command arguments or Git.

`onboarding` is an experimental fixture-preparation suite; its UI path was observed, but video reattachment on relaunch is not qualified. `discover` accepts one JSON `Step` per line and ends with the literal line `stop`; it is for interactive exploration, not an unattended pass. Do not send credentials through its input. `no-glasses` is not implemented and returns a nonzero error.

The app window must remain open and keep its recording geometry. You can use other applications while accessibility actions run: the driver contains no mouse/keyboard injection or foreground-activation command. Missing selectors or inaccessible controls stop the scenario; fix the app instead of adding a coordinate fallback.

## Run artifacts

Each command prints a unique directory under `.test-results/mentra-e2e/`. Open `index.html` to play `routine.mp4` and click an English step description to seek to it. `chapters.json` contains the same timestamps. `run.json`, `events.jsonl`, `checklist.md`, and `summary.md` contain the results, while `screenshots/` and `accessibility/` hold evidence per executed step. Runs stay local and are ignored by Git.

Browser automation verification of the local HTML viewer remains pending: the browser tool rejected the local file URL under its security policy. The report files and chapter timestamps were inspected directly.

Screenshots use frames from the video stream. Do not start independent screenshot capture while recording; that interrupted the ScreenCaptureKit connection during initial testing. Frame-settle status is recorded as diagnostic evidence; semantic assertions remain the behavior checks.

## Current evidence

All paths below are relative to `.test-results/mentra-e2e/` on the development Mac. The artifacts are not committed or automatically transferred with the branch.

| Evidence | Result |
| --- | --- |
| `2026-09-15T21-16-50-601Z-driver-proof-b2bf4e` and `2026-09-15T21-17-13-555Z-driver-proof-39aece` | Original seven-step login/validation proof passed twice. These precede the stricter accessibility identifiers and do not qualify current full replay. |
| `2026-09-15T21-18-22-086Z-login-68627e` | Real sign-in succeeded; expected home was wrong for first-run onboarding, so the run failed. |
| `2026-09-15T21-19-30-646Z-onboarding-47ca6f` | The UI reached unpaired home; video reattachment failed. Kept as failed. |
| `2026-09-15T21-26-34-336Z-discovery-e6eaa6` | Settings/Profile/account-form/logout-cancel/feedback observations using background accessibility actions. Includes failed exploratory expectations, not a qualified suite. |
| `2026-09-15T21-43-42-280Z-accessibility-preflight-045bcd` | Correctly failed on dev.235's missing `miniapp.minimize`; exited 1, retained PNG/AX evidence and a finalized 3.09-second H.264 video. No UI action was taken. |

## Development validation

Runtime replay only needs Bun, Swift and the permissions described in setup. Install these small development dependencies for TypeScript checks:

```sh
cd tools/mentra-e2e
bun install --frozen-lockfile
bun run typecheck
bun test runner
# On a provisioned Mac with Mentra open: reject unsupported input without acting.
MENTRA_E2E_NATIVE_CHECKS=1 bun test runner
```

Native negative checks passed (seven rejection cases), along with secret redaction and two mobile tray accessibility tests. The mobile check is:

```sh
cd mobile
bun run test --runInBand --runTestsByPath src/components/home/AppSwitcherButton.accessibility.test.tsx
```

The full mobile type-check currently reports an existing `app.config.ts:107` error (`name` can be null). That file is unchanged by this branch. No diagnostics were reported in the changed app files. Source checks cannot prove the new controls' native exposure; an installed-build walkthrough remains required.

Remaining work is tracked in the [implementation plan](../../notes/superpowers/plans/2026-09-15-mentra-app-e2e-harness.md): finish the walkthrough on the updated binary, encode the verified steps, qualify relaunch capture and cleanup, and run the entire routine three times without intervention.
