# Set up another Mac or Mac Mini

This guide provisions the standalone harness. It does not require an AI agent, a mobile source build, a simulator, or installing the monorepo's dependencies. The current development target is the TestFlight iOS app running on an Apple Silicon Mac.

## 1. Prepare the Mac

1. Use an Apple Silicon Mac with a logged-in graphical desktop. Keep a display connected for initial qualification; a headless Mac Mini has not been qualified.
2. Install macOS compatible with the TestFlight app. Match the qualified machine's macOS version when reproducing a failure. The native recorder requires macOS 15 or newer; the initial host is macOS 26.6.2. Older OS versions have not been qualified.
3. Install Apple's command line tools, or Xcode with its command line tools selected. The helper uses Swift 6 and the macOS SDK's AppKit, ApplicationServices, CoreImage and ScreenCaptureKit. It does not build the Mentra App.
4. Install Bun from its official distribution. Initial development uses Bun **1.4.0** and Swift **6.4**. Keep the actual versions in the run evidence; changing versions requires rerunning the driver proof.
5. Keep the Mac awake, unlocked, and available during runs. Other people may work in other apps. The driver uses accessibility actions without activating Mentra or moving the pointer. Do not move, resize, minimize or close the target window during a video run.

Verify the tools:

```sh
uname -m
sw_vers
xcode-select -p
swiftc --version
bun --version
```

If the Apple tools are missing, use Apple's installer:

```sh
xcode-select --install
```

Bun installation instructions: <https://bun.com/docs/installation>. Use a consistent version across test machines. `swiftformat` is needed only when editing Swift source, not when replaying tests. `ffprobe`/FFmpeg is useful for independent video verification, but recording uses native Apple frameworks.

## 2. Install the real app

1. Install TestFlight from Apple's App Store.
2. Sign in with the Apple account that has access to the Mentra TestFlight build and accept the applicable TestFlight invitation.
3. Install the intended Mentra iOS build on the Mac. This requires that the build is available for Apple Silicon Macs. If TestFlight does not offer it, resolve build/account availability before continuing; a simulator build is a different test target.
4. Launch Mentra and leave its window open. Initial reference: bundle ID `com.mentra.mentra`, version `3.2.0`, build `320000235`.
5. Install a build containing this branch’s app accessibility changes before qualifying the complete routine. The original build `320000235` lacks identifiers such as `miniapp.minimize` and `home.allApps.open`; the harness intentionally has no coordinate fallback. Source edits cannot change an already installed TestFlight binary. Follow [ACCESSIBILITY.md](ACCESSIBILITY.md) to validate native actions after installing the new build.
6. Use English for the qualified routine. Record any changes from the reference build. Do not install a newer app during a run.

There may be both `/Applications/Mentra.app` and a temporary iOS wrapper under `/var/folders/...`. Do not copy or hard-code the temporary wrapper path. The driver finds the running process by bundle ID and reads its actual bundle URL every invocation. Multiple running matches cause a setup failure rather than an arbitrary selection.

## 3. Get this harness revision

The working branch is `codex/mentra-e2e-harness`, based on `dev`. Once that branch is published, use a separate checkout/worktree on the Mac Mini:

```sh
git clone git@github.com:Mentra-Community/MentraOS.git MentraOS
cd MentraOS
git fetch origin codex/mentra-e2e-harness
git worktree add ../MentraOS-e2e -b local/mentra-e2e origin/codex/mentra-e2e-harness
cd ../MentraOS-e2e
```

The branch may still be local during development. Do not assume these remote-fetch commands work before it is published. To transfer committed work without publishing, create an incremental Git bundle on the source Mac and copy it to the Mac Mini:

```sh
# Source Mac, from the harness worktree, after committing the harness:
git bundle create /tmp/mentra-e2e.bundle origin/dev..codex/mentra-e2e-harness

# Mac Mini, in a clone that has fetched dev and contains the bundle prerequisites:
git bundle verify /path/to/mentra-e2e.bundle
git fetch /path/to/mentra-e2e.bundle codex/mentra-e2e-harness:refs/heads/local/mentra-e2e
git worktree add ../MentraOS-e2e local/mentra-e2e
```

Check the expected commit with `git rev-parse HEAD`. Do not transfer credentials, Keychain databases, device pairings, or old macOS privacy databases. Run artifacts are local and ignored; copy any historical evidence separately if wanted.

## 4. Build and check permissions

Run from the repository root, in the same terminal application you will use for replay:

```sh
bun run tools/mentra-e2e/run.ts doctor
```

The first invocation compiles the Swift helper into `.test-results/mentra-e2e/bin/mentra-driver`. Later invocations rebuild only when the native source or Swift toolchain changes.

`doctor` must report the intended app identity, `accessibility: true` and `screenCapture: true` (`postEvents` is diagnostic only; the driver no longer injects input events). It exits nonzero when required access is missing. No test should be counted as passing when this preflight fails.

macOS can attribute permissions to the **application launching the process**, rather than to the helper's filename:

- In Terminal, expect **Terminal**; in another terminal, expect that application.
- In this Codex desktop session, macOS displayed **ChatGPT** (`com.openai.codex`).
- **Codex Computer Use** is a separate permission entry. Enabling it does not authorize terminal replay.

Open **System Settings → Privacy & Security** and grant the responsible launcher:

1. **Accessibility**, for inspecting and controlling Mentra's interface.
2. **Screen & System Audio Recording**, for screenshots and video. The harness configures screen capture only; system audio and microphone capture are disabled.

If the launcher is absent from the recording list, trigger the standard request from that launcher:

```sh
printf '%s' '{"op":"request-screen-capture"}' | .test-results/mentra-e2e/bin/mentra-driver
```

If macOS presents a choice to select a window or allow direct capture, direct capture is needed for unattended replay to identify Mentra without asking you to pick the window each run. The OS permission is broader than the actual capture filter: the recorder allows only the Mentra window and excludes other applications, the desktop and Dock.

macOS may show **Quit & Reopen** / **Later** after a permission change. On the initial Mac, choosing **Later** and starting fresh helper processes was sufficient: `doctor` and real capture then worked. On the new Mac, verify instead of assuming. If either still fails, save any work, quit and reopen the responsible terminal application, and run `doctor` again. Do not reset all privacy permissions or modify the TCC database.

Changing the launcher, checkout location, helper signature or toolchain can require rechecking permissions. Do not grant Full Disk Access or disable SIP for this harness.

## 5. Verify the standalone driver before signing in

After installing the accessibility changes, with Mentra on its signed-out welcome screen:

```sh
bun run tools/mentra-e2e/run.ts run --suite driver-proof
bun run tools/mentra-e2e/run.ts run --suite driver-proof
```

The current proof requires the new `navigation.back` identifier. The original legacy proof passed twice on dev.235; those old passes do not qualify the stricter current revision. This opens Log In, enters a deliberately malformed email, verifies the real app's validation, dismisses it, clears the field, and returns to the welcome screen. It does not send bad-password requests or need account credentials.

Each run prints its unique artifact folder. Both runs must exit zero. Open `index.html`, play `routine.mp4`, and click the English step descriptions to check that seeking works. Inspect the input/validation screenshots. Merely compiling or seeing a successful `doctor` is insufficient.

The capture implementation takes screenshots from the same live stream as the video. Starting independent screenshot capture processes while recording caused recording-connection failures during development; use the runner's capture path rather than another recorder alongside it.

## 6. Provide the test account at runtime

For an account that has already completed onboarding, run the login suite from an interactive terminal:

```sh
bun run tools/mentra-e2e/run.ts run --suite login
```

A first-ever sign-in reaches onboarding instead of home and will fail the login suite’s home assertion. Preserve that result, finish fixture setup separately, and rerun once established. The runner prompts for the test account email and password without echoing them. Obtain the designated account credentials from the team's existing secure store or the person provisioning this Mac. They are deliberately absent from Git and this guide.

For an unattended runner, inject `MENTRA_E2E_EMAIL` and `MENTRA_E2E_PASSWORD` into its environment using an existing secret manager. Do not put the password in shell command arguments, shell history, checked-in `.env` files, screenshot filenames or test fixtures. The current harness does not provision a new Keychain item automatically.

The password stays in a secure UI field; do not use the visibility toggle during recording. Secure accessibility values are redacted in artifacts. Videos may still contain the test account's email and ordinary account content; keep them local unless deliberately shared.

## 7. Establish the fixture through the real UI

- **Unpaired** means no paired/selected device and no Phone Mode session. A physical device being powered off is not enough to establish this fixture.
- **Paired-disconnected** preserves an existing pairing. It is a separate fixture; do not unpair the user's glasses just to satisfy a test.
- **Phone Mode** connects a simulated device and has separate expectations.

On the initial fresh account, successful login reached first-run onboarding. The ordinary **Set up with glasses → model selector → Back → relaunch** path reached home without choosing a device. This works because the app records onboarding completion before model selection. The setup suite documents that transition; it does not scan for or connect to glasses. Verify the resulting **Pair glasses** home state rather than trusting the click sequence.

**Set up without glasses** can enter simulated pairing. Do not use it to prepare an unpaired run.

Do not clear app storage, reset Keychain, or overwrite deployment/server settings. Record app/backend identity, theme, locale, account alias, wearable profile and window geometry with the run. Treat unavailable fixture setup as a setup gap.

## 8. Operate and troubleshoot

- Run one harness at a time. A per-user lock prevents two checkouts from driving the same app concurrently.
- Keep the app window on the same display and at the same size/position for a video run. The recorder rejects an unexpected geometry change; it does not silently capture a different screen area.
- An assertion failure produces a nonzero exit and preserves the video, screenshots, accessibility snapshots, expected result, and timing. Inspect the failing step before rerunning.
- A recording failure makes the run incomplete even if some UI assertions passed.
- Runs go to `.test-results/mentra-e2e/<timestamp>-<suite>-<suffix>/`. Keep failed runs alongside successful runs while debugging. No automatic artifact upload occurs.
- App relaunch requires the recorder to reattach to the new process/window. Qualify that on the Mac Mini as part of the full routine; the initial setup path is not proof of final video reattachment.
- Keep the design's physical-device limits: this Mac lane does not qualify BLE, firmware updates, glasses capture, SoftAP transfers or iPhone screen-off operation.

Before accepting the Mac Mini as a test station, run the qualified no-glasses suite three times with zero manual correction, verify video chapter navigation, check deliberate selector/assertion failures, and record the installed app and harness revisions. Refer to the README and implementation plan for the currently qualified suites; a planned suite is not automatically implemented.
