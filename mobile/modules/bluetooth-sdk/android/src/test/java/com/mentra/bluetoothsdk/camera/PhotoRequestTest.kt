package com.mentra.bluetoothsdk.camera

import com.mentra.bluetoothsdk.PhotoCompression
import com.mentra.bluetoothsdk.PhotoMode
import com.mentra.bluetoothsdk.PhotoRequest
import com.mentra.bluetoothsdk.PhotoSize
import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.junit.Test

class PhotoRequestTest {
    @Test
    fun `compression tiers parse from the bridge map`() {
        val fields = mapOf("size" to "medium", "webhookUrl" to "https://example.com/upload")
        assertThat(PhotoRequest.fromMap(fields).compress).isEqualTo(PhotoCompression.NONE)
        assertThat(PhotoRequest.fromMap(fields + ("compress" to "none")).compress).isEqualTo(PhotoCompression.NONE)
        assertThat(PhotoRequest.fromMap(fields + ("compress" to "low")).compress).isEqualTo(PhotoCompression.LOW)
        assertThat(PhotoRequest.fromMap(fields + ("compress" to "medium")).compress).isEqualTo(PhotoCompression.MEDIUM)
        assertThat(PhotoRequest.fromMap(fields + ("compress" to "high")).compress).isEqualTo(PhotoCompression.HIGH)
        assertThat(PhotoRequest.fromMap(fields + ("compress" to "heavy")).compress).isEqualTo(PhotoCompression.HEAVY)
        assertThat(PhotoRequest.fromMap(fields + ("compress" to "ultra")).compress).isEqualTo(PhotoCompression.NONE)
    }

    @Test
    fun `high is sent as the legacy heavy wire value and every other tier is sent verbatim`() {
        assertThat(PhotoCompression.NONE.wireValue).isEqualTo("none")
        assertThat(PhotoCompression.LOW.wireValue).isEqualTo("low")
        assertThat(PhotoCompression.MEDIUM.wireValue).isEqualTo("medium")
        assertThat(PhotoCompression.HIGH.wireValue).isEqualTo("heavy")
        assertThat(PhotoCompression.HEAVY.wireValue).isEqualTo("heavy")
        assertThat(PhotoCompression.values().map { it.wireValue }).doesNotContain("high")
    }

    @Test
    fun `heavy is the legacy alias for high`() {
        assertThat(PhotoCompression.HEAVY.canonical).isEqualTo(PhotoCompression.HIGH)
        assertThat(PhotoCompression.HIGH.canonical).isEqualTo(PhotoCompression.HIGH)
        assertThat(PhotoCompression.fromValue("heavy").canonical).isEqualTo(PhotoCompression.fromValue("high").canonical)
        for (tier in listOf(PhotoCompression.NONE, PhotoCompression.LOW, PhotoCompression.MEDIUM)) {
            assertThat(tier.canonical).isEqualTo(tier)
        }
    }

    @Test
    fun `fromValue returns none for null and unknown values`() {
        assertThat(PhotoCompression.fromValue(null)).isEqualTo(PhotoCompression.NONE)
        assertThat(PhotoCompression.fromValue("")).isEqualTo(PhotoCompression.NONE)
        assertThat(PhotoCompression.fromValue("HIGH")).isEqualTo(PhotoCompression.NONE)
    }

    @Test
    fun `constructor defaults compression to none like iOS and the docs`() {
        val request = PhotoRequest(size = PhotoSize.MEDIUM, webhookUrl = "https://example.com/upload")
        assertThat(request.compress).isEqualTo(PhotoCompression.NONE)
        assertThat(request.compress.wireValue).isEqualTo("none")
    }

    @Test
    fun `thumbnail is opt in and survives request routing copies`() {
        val fields = mapOf("size" to "medium", "webhookUrl" to "https://example.com/upload")
        assertThat(PhotoRequest.fromMap(fields).presendThumbnail).isFalse()
        assertThat(PhotoRequest.fromMap(fields + ("presend_thumbnail" to false)).presendThumbnail).isFalse()
        val request = PhotoRequest.fromMap(fields + ("presend_thumbnail" to true))
        assertThat(request.copy(requestId = "routed").presendThumbnail).isTrue()
        assertThat(request.transferMethod).isEqualTo("auto")
    }
    @Test
    fun `constructor generates requestId when omitted`() {
        val request =
            PhotoRequest(
                size = PhotoSize.MEDIUM,
                webhookUrl = "https://example.com/upload",
                compress = PhotoCompression.NONE,
                sound = true,
            )

        assertThat(request.requestId).startsWith("photo-")
    }

    @Test
    fun `fromMap defaults exposureTimeNs null`() {
        val request =
            PhotoRequest.fromMap(
                mapOf(
                    "requestId" to "photo-1",
                    "size" to "medium",
                    "webhookUrl" to "https://example.com/upload",
                    "compress" to "none",
                    "sound" to true,
                )
            )

        assertThat(request.exposureTimeNs).isNull()
        assertThat(request.mode).isEqualTo(PhotoMode.PHOTO)
        assertThat(request.transferMethod).isEqualTo("auto")
    }

    @Test
    fun `fromMap generates requestId when omitted or blank`() {
        val withoutRequestId =
            PhotoRequest.fromMap(
                mapOf(
                    "size" to "medium",
                    "webhookUrl" to "https://example.com/upload",
                    "compress" to "none",
                    "sound" to true,
                )
            )
        val blankRequestId =
            PhotoRequest.fromMap(
                mapOf(
                    "requestId" to "  ",
                    "size" to "medium",
                    "webhookUrl" to "https://example.com/upload",
                    "compress" to "none",
                    "sound" to true,
                )
            )

        assertThat(withoutRequestId.requestId).startsWith("photo-")
        assertThat(blankRequestId.requestId).startsWith("photo-")
    }

    @Test
    fun `fromMap preserves explicit requestId`() {
        val request =
            PhotoRequest.fromMap(
                mapOf(
                    "requestId" to "photo-1",
                    "size" to "medium",
                    "webhookUrl" to "https://example.com/upload",
                    "compress" to "none",
                    "sound" to true,
                )
            )

        assertThat(request.requestId).isEqualTo("photo-1")
    }

    @Test
    fun `fromMap preserves text mode`() {
        val request =
            PhotoRequest.fromMap(
                mapOf(
                    "size" to "medium",
                    "mode" to "text",
                    "webhookUrl" to "https://example.com/upload",
                )
            )

        assertThat(request.mode).isEqualTo(PhotoMode.TEXT)
    }

    @Test
    fun `fromMap preserves forced BLE transfer`() {
        val request =
            PhotoRequest.fromMap(
                mapOf(
                    "size" to "medium",
                    "transferMethod" to "ble",
                    "webhookUrl" to "https://example.com/upload",
                )
            )

        assertThat(request.transferMethod).isEqualTo("ble")
    }

    @Test
    fun `fromMap preserves direct transfer without BLE fallback`() {
        val request =
            PhotoRequest.fromMap(
                mapOf(
                    "size" to "medium",
                    "transferMethod" to "direct",
                    "webhookUrl" to "https://example.com/upload",
                )
            )

        assertThat(request.transferMethod).isEqualTo("direct")
    }

    @Test
    fun `fromMap rejects an unknown transfer method`() {
        assertThatThrownBy {
            PhotoRequest.fromMap(
                mapOf(
                    "size" to "medium",
                    "transferMethod" to "wifi",
                    "webhookUrl" to "https://example.com/upload",
                )
            )
        }
            .isInstanceOf(IllegalArgumentException::class.java)
            .hasMessage("Invalid transferMethod \"wifi\". Expected auto, direct, or ble.")
    }
}
