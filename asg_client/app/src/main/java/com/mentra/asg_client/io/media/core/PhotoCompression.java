package com.mentra.asg_client.io.media.core;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import com.mentra.asg_client.AsgConstants;
import com.mentra.asg_client.camera.lifecycle.PhotoExifMetadataWriter;
import java.io.FileOutputStream;
import java.io.IOException;

/** Transport-independent JPEG quality. Pixel limits are owned by size/crop policy. */
enum PhotoCompression {
  NONE(AsgConstants.PHOTO_JPEG_QUALITY_NONE),
  LOW(AsgConstants.PHOTO_JPEG_QUALITY_LOW),
  MEDIUM(AsgConstants.PHOTO_JPEG_QUALITY_MEDIUM),
  HIGH(AsgConstants.PHOTO_JPEG_QUALITY_HIGH);

  final int jpegQuality;

  PhotoCompression(int jpegQuality) {
    this.jpegQuality = jpegQuality;
  }

  static PhotoCompression fromValue(String value) {
    if ("low".equals(value)) return LOW;
    if ("medium".equals(value)) return MEDIUM;
    if ("high".equals(value) || "heavy".equals(value)) return HIGH;
    return NONE;
  }

  /** Re-encode the capture at the selected quality without resizing or modifying the source.
   * BLE fallback must reuse the original capture, never this already-compressed upload copy.
   */
  void encodeUpload(String originalPath, String uploadPath) throws IOException {
    Bitmap source = BitmapFactory.decodeFile(originalPath);
    if (source == null) throw new IOException("Could not decode photo for upload");
    try (FileOutputStream output = new FileOutputStream(uploadPath)) {
      if (!source.compress(Bitmap.CompressFormat.JPEG, jpegQuality, output)) {
        throw new IOException("Photo JPEG encoding failed");
      }
    } finally {
      source.recycle();
    }
    PhotoExifMetadataWriter.copyImuMetadata(originalPath, uploadPath);
  }
}
