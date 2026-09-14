package com.mentra.asg_client.io.media.core;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import com.mentra.asg_client.camera.lifecycle.PhotoExifMetadataWriter;
import com.mentra.asg_client.camera.policy.PhotoCompressionLevel;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

/**
 * Re-encodes a captured JPEG for Wi-Fi/webhook upload at the requested {@link
 * PhotoCompressionLevel}. Pixel dimensions are never changed: the {@code size} tier already
 * bounded them at capture time, and {@code compress} only selects JPEG quality.
 */
final class PhotoUploadCompressor {
    private PhotoUploadCompressor() {}

    /** Path used for the re-encoded copy; the original capture is left in place. */
    static File outputFileFor(File original, PhotoCompressionLevel level) {
        String name = original.getName();
        int dot = name.lastIndexOf('.');
        String stem = dot > 0 ? name.substring(0, dot) : name;
        return new File(original.getParentFile(), stem + "_compressed_" + level.wireValue() + ".jpg");
    }

    /**
     * Returns the file to upload for {@code level}. {@link PhotoCompressionLevel#NONE} returns
     * {@code original} untouched; other tiers write a same-dimension JPEG copy carrying the
     * original's capture EXIF and return it.
     *
     * @throws IOException when the source cannot be decoded or the copy cannot be written
     */
    static File compress(File original, PhotoCompressionLevel level) throws IOException {
        if (!level.reencodesUpload()) {
            return original;
        }
        Bitmap bitmap = BitmapFactory.decodeFile(original.getAbsolutePath());
        if (bitmap == null) {
            throw new IOException("Could not decode " + original + " for upload compression");
        }
        File output = outputFileFor(original, level);
        try (FileOutputStream out = new FileOutputStream(output)) {
            if (!bitmap.compress(Bitmap.CompressFormat.JPEG, level.jpegQuality(), out)) {
                throw new IOException("JPEG encoding failed for " + output);
            }
        } catch (IOException e) {
            //noinspection ResultOfMethodCallIgnored
            output.delete();
            throw e;
        } finally {
            bitmap.recycle();
        }
        PhotoExifMetadataWriter.copyImuMetadata(original.getAbsolutePath(), output.getAbsolutePath());
        return output;
    }
}
