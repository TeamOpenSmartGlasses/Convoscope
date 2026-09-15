import * as WebBrowser from "expo-web-browser"

import {openAuthBrowser} from "./openAuthBrowser"

jest.mock("expo-web-browser", () => ({openAuthSessionAsync: jest.fn()}))

const callbackUrl = "com.mentra://auth/callback?code=handoff&state=state"

beforeEach(() => jest.clearAllMocks())

it("awaits callback processing from the native auth-session result", async () => {
  jest.mocked(WebBrowser.openAuthSessionAsync).mockResolvedValue({type: "success", url: callbackUrl})
  let finish!: () => void
  const processUrl = jest.fn(
    () =>
      new Promise<void>((resolve) => {
        finish = resolve
      }),
  )
  const finished = jest.fn()
  const result = openAuthBrowser("https://core.example/oauth/google/start", processUrl).then(finished)
  await Promise.resolve()

  expect(WebBrowser.openAuthSessionAsync).toHaveBeenCalledWith(
    "https://core.example/oauth/google/start",
    "com.mentra://auth/callback",
  )
  expect(processUrl).toHaveBeenCalledWith(callbackUrl)
  expect(finished).not.toHaveBeenCalled()
  finish()
  await result
  expect(finished).toHaveBeenCalledWith(true)
})

it.each(["cancel", "dismiss"] as const)("returns false for %s without exchanging credentials", async (type) => {
  jest.mocked(WebBrowser.openAuthSessionAsync).mockResolvedValue({type} as WebBrowser.WebBrowserResult)
  const processUrl = jest.fn()
  expect(await openAuthBrowser("https://core.example/oauth/google/start", processUrl)).toBe(false)
  expect(processUrl).not.toHaveBeenCalled()
})

it("propagates browser failures to the caller's error handling", async () => {
  jest.mocked(WebBrowser.openAuthSessionAsync).mockRejectedValue(new Error("browser unavailable"))
  await expect(openAuthBrowser("https://core.example/oauth/google/start", jest.fn())).rejects.toThrow(
    "browser unavailable",
  )
})
