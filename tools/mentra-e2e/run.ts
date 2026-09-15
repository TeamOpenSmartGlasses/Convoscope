import {parseArgs} from "node:util"
import {buildDriver, command, compact, snapshot, type Doctor} from "./runner/driver"
import {acquireLock, Report} from "./runner/report"
import {executeSteps} from "./runner/suite"
import {driverProof} from "./flows/driver-proof"
import {login} from "./flows/login"
import {credentials} from "./runner/credentials"
import {onboarding} from "./flows/onboarding"
import {discover} from "./runner/discover"
import {accessibilityPreflight} from "./flows/accessibility-preflight"

const {positionals, values} = parseArgs({
  args: process.argv.slice(2),
  allowPositionals: true,
  options: {
    suite: {type: "string", default: "driver-proof"},
    fixture: {type: "string", default: "unpaired"},
  },
})
const operation = positionals[0] ?? "doctor"
try {
  await buildDriver()
  if (operation === "doctor") {
    const doctor = await command<Doctor>({op: "doctor"})
    console.log(JSON.stringify(doctor, null, 2))
    if (!doctor.accessibility || !doctor.screenCapture) process.exitCode = 2
  } else if (operation === "inspect") {
    console.log(JSON.stringify(compact(await snapshot()), null, 2))
  } else if (operation === "discover") {
    await discover(values.fixture!)
  } else if (operation === "run") {
    const suites = {"driver-proof": driverProof, login, onboarding, "accessibility-preflight": accessibilityPreflight}
    const steps = suites[values.suite as keyof typeof suites]
    if (!steps) throw new Error(`Suite is not implemented: ${values.suite}`)
    const account = values.suite === "login" ? await credentials() : {email: "", password: ""}
    const release = await acquireLock()
    const report = new Report(values.suite!, [account.password])
    try {
      const doctor = await command<Doctor>({op: "doctor"})
      await report.start(doctor, values.fixture!)
      if (!doctor.accessibility || !doctor.screenCapture)
        throw new Error("macOS Accessibility and Screen Recording permissions are required; run doctor")
      await report.startVideo()
      const passed = await executeSteps(steps, {...account, fixture: values.fixture!}, report)
      const restored =
        values.suite === "driver-proof"
          ? "Authentication start restored"
          : values.suite === "accessibility-preflight"
          ? "No UI changes; inspected the current miniapp"
          : "Authenticated app reached"
      await report.finish(
        passed ? "passed" : "failed",
        passed ? restored : "No automatic recovery; inspect failure evidence",
      )
      if (!passed || report.metadata.status !== "passed") process.exitCode = 1
    } catch (error) {
      if (report.directory) await report.finish("incomplete", `Setup/run failure: ${String(error)}`)
      throw error
    } finally {
      await release()
    }
  } else throw new Error(`Unknown command: ${operation}`)
} catch (error) {
  console.error(String(error))
  process.exitCode = 1
}
