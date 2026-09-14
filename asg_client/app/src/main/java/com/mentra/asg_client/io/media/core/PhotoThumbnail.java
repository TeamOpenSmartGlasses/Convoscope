package com.mentra.asg_client.io.media.core;

import android.graphics.Bitmap;
import com.mentra.asg_client.AsgConstants;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

/** Encodes an opt-in preview without modifying or recycling the full-photo source. */
final class PhotoThumbnail {
    static byte[] encode(Bitmap source) throws IOException {
        float scale =
                Math.min(
                        1f,
                        (float) AsgConstants.PHOTO_THUMBNAIL_LONG_EDGE
                                / Math.max(source.getWidth(), source.getHeight()));
        Bitmap thumbnail =
                Bitmap.createScaledBitmap(
                        source,
                        Math.max(1, Math.round(source.getWidth() * scale)),
                        Math.max(1, Math.round(source.getHeight() * scale)),
                        true);
        try {
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            if (!thumbnail.compress(
                    Bitmap.CompressFormat.JPEG,
                    AsgConstants.PHOTO_THUMBNAIL_JPEG_QUALITY,
                    output)) {
                throw new IOException("Thumbnail JPEG encoding failed");
            }
            return output.toByteArray();
        } finally {
            if (thumbnail != source) thumbnail.recycle();
        }
    }
}
