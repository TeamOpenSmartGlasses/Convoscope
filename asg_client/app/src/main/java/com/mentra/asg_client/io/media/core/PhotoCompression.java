package com.mentra.asg_client.io.media.core;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import androidx.exifinterface.media.ExifInterface;
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

  /**
   * Returns the file to upload. {@code NONE} uploads the untouched capture, so every EXIF tag the
   * camera wrote (orientation included) survives. Other levels re-encode at {@link #jpegQuality}
   * without resizing and carry the orientation and IMU metadata over, because a decoded bitmap
   * loses the EXIF block. BLE fallback must reuse the original capture, never the upload copy.
   */
  String prepareUpload(String originalPath, String uploadPath) throws IOException {
    if (this == NONE) return originalPath;
    Bitmap source = BitmapFactory.decodeFile(originalPath);
    if (source == null) throw new IOException("Could not decode photo for upload");
    try (FileOutputStream output = new FileOutputStream(uploadPath)) {
      if (!source.compress(Bitmap.CompressFormat.JPEG, jpegQuality, output)) {
        throw new IOException("Photo JPEG encoding failed");
      }
    } finally {
      source.recycle();
    }
    copyOrientation(originalPath, uploadPath);
    PhotoExifMetadataWriter.copyImuMetadata(originalPath, uploadPath);
    return uploadPath;
  }

  private static void copyOrientation(String sourcePath, String destPath) throws IOException {
    String orientation = new ExifInterface(sourcePath).getAttribute(ExifInterface.TAG_ORIENTATION);
    if (orientation == null) return;
    ExifInterface dest = new ExifInterface(destPath);
    dest.setAttribute(ExifInterface.TAG_ORIENTATION, orientation);
    dest.saveAttributes();
  }
}
