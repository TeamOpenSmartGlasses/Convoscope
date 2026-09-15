import * as WebBrowser from "expo-web-browser"

/** Finish from the auth session's result even when iOS doesn't emit a Linking event. */
export async function openAuthBrowser(url: string, processUrl: (url: string) => Promise<void>): Promise<boolean> {
  const result = await WebBrowser.openAuthSessionAsync(url, "com.mentra://auth/callback")
  if (result.type !== "success") return false

  await processUrl(result.url)
  return true
}
