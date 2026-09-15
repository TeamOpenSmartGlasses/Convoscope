/**
 * @fileoverview DashboardAPI — noop surface in v1.
 *
 * Dashboard rendering is not implemented by the local miniapp runtime.
 * The API shape is retained, but calls only warn once and forward a request
 * that the host rejects as NOT_IMPLEMENTED.
 */

import {MiniappRequestType} from "../protocol"
import {MiniappSession} from "../session"

export type DashboardMode = "main" | "expanded" | "always_on"

export class DashboardAPI {
  private warned = false

  constructor(private readonly session: MiniappSession) {}

  setContent(mode: DashboardMode, content: string): void {
    if (!this.warned) {
      console.warn("[@mentra/miniapp] dashboard.setContent() is deferred in v1.")
      this.warned = true
    }
    // Still forward so the phone can log/ignore consistently.
    this.session.sendOneShot({
      type: MiniappRequestType.DASHBOARD_CONTENT_UPDATE,
      mode,
      content,
    })
  }
}
