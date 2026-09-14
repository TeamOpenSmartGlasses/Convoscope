package com.mentra.asg_client.camera.policy;

import android.util.Log;

import com.mentra.asg_client.AsgConstants;

/**
 * Canonical photo compression tiers for the {@code compress} request field.
 *
 * <p>Compression selects JPEG quality only. Pixel dimensions are governed independently by the
 * {@code size} tier ({@link PhotoSizeTier}), so two photos taken at the same size but different
 * compression levels have identical dimensions and differ only in encoder quality.
 *
 * <p>{@code heavy} is the legacy spelling of {@code high}: phone SDKs still send it on the wire so
 * older glasses firmware keeps its previous behavior. Unknown values fall back to {@link #NONE}
 * rather than silently degrading the image.
 */
public enum PhotoCompressionLevel {
    NONE("none", AsgConstants.PHOTO_JPEG_QUALITY_NONE),
    LOW("low", AsgConstants.PHOTO_JPEG_QUALITY_LOW),
    MEDIUM("medium", AsgConstants.PHOTO_JPEG_QUALITY_MEDIUM),
    HIGH("high", AsgConstants.PHOTO_JPEG_QUALITY_HIGH);

    private static final String TAG = "PhotoCompressionLevel";

    /** Legacy wire spelling accepted for {@link #HIGH}. */
    public static final String LEGACY_HEAVY = "heavy";

    private final String wireValue;
    private final int jpegQuality;

    PhotoCompressionLevel(String wireValue, int jpegQuality) {
        this.wireValue = wireValue;
        this.jpegQuality = jpegQuality;
    }

    /** Canonical wire spelling ({@code none | low | medium | high}). */
    public String wireValue() {
        return wireValue;
    }

    /** JPEG encoder quality (0-100) for this tier. */
    public int jpegQuality() {
        return jpegQuality;
    }

    /**
     * Whether a Wi-Fi/webhook upload must re-encode the captured file. {@link #NONE} uploads the
     * capture as written by the camera pipeline.
     */
    public boolean reencodesUpload() {
        return this != NONE;
    }

    /**
     * Parses a {@code compress} wire value. {@code null}, blank, and {@code none} map to
     * {@link #NONE}; {@code heavy} maps to {@link #HIGH}; unknown values log and map to
     * {@link #NONE}.
     */
    public static PhotoCompressionLevel fromWire(String value) {
        if (value == null) {
            return NONE;
        }
        String normalized = value.trim().toLowerCase(java.util.Locale.US);
        if (normalized.isEmpty()) {
            return NONE;
        }
        if (LEGACY_HEAVY.equals(normalized)) {
            return HIGH;
        }
        for (PhotoCompressionLevel level : values()) {
            if (level.wireValue.equals(normalized)) {
                return level;
            }
        }
        Log.w(TAG, "Unknown photo compression '" + value + "' — using none");
        return NONE;
    }

    /** Normalizes any accepted spelling to the canonical wire value. */
    public static String normalize(String value) {
        return fromWire(value).wireValue;
    }
}
