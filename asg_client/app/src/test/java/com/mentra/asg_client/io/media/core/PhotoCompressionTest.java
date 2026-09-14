package com.mentra.asg_client.io.media.core;

import static org.junit.Assert.*;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import androidx.exifinterface.media.ExifInterface;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.nio.file.Files;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.annotation.Config;
import org.robolectric.annotation.GraphicsMode;

@RunWith(RobolectricTestRunner.class)
@Config(manifest = Config.NONE, sdk = 28)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
public class PhotoCompressionTest {
  @Rule public TemporaryFolder files = new TemporaryFolder();

  @Test
  public void everySpellingUsesTheSameQualityForUploadBleAndWifiFallback() throws Exception {
    String[] spellings = {"none", "low", "medium", "high", "heavy"};
    int[] qualities = {95, 88, 78, 60, 60};
    Bitmap sensor = Bitmap.createBitmap(120, 80, Bitmap.Config.ARGB_8888);
    for (int y = 0; y < 80; y++) {
      for (int x = 0; x < 120; x++) sensor.setPixel(x, y, 0xff000000 | x * 131071 + y * 8191);
    }
    File original = files.newFile("original.jpg");
    ByteArrayOutputStream capture = new ByteArrayOutputStream();
    assertTrue(sensor.compress(Bitmap.CompressFormat.JPEG, 100, capture));
    sensor.recycle();
    byte[] capturedBytes = capture.toByteArray();
    Files.write(original.toPath(), capturedBytes);

    for (int i = 0; i < spellings.length; i++) {
      PhotoCompression policy = PhotoCompression.fromValue(spellings[i]);
      assertEquals(qualities[i], policy.jpegQuality);
      File upload = files.newFile(spellings[i] + ".jpg");
      String uploaded = policy.prepareUpload(original.getPath(), upload.getPath());
      if (policy == PhotoCompression.NONE) {
        // none uploads the capture itself; only the BLE resize forces a Q95 re-encode.
        assertEquals(original.getPath(), uploaded);
        assertEquals(0, upload.length());
        continue;
      }
      assertEquals(upload.getPath(), uploaded);
      byte[] direct = Files.readAllBytes(upload.toPath());
      Bitmap ramCapture = BitmapFactory.decodeByteArray(capturedBytes, 0, capturedBytes.length);
      Bitmap fallbackCapture = BitmapFactory.decodeFile(original.getPath());
      try {
        // RAM-first BLE and file-backed Wi-Fi fallback use the same encoder and policy.
        byte[] ble = BlePhotoEncoders.encode(ramCapture, BleCodec.JPEG_FAST,
            policy.jpegQuality, null, null, null).data;
        byte[] fallback = BlePhotoEncoders.encode(fallbackCapture, BleCodec.JPEG_FAST,
            policy.jpegQuality, original.getPath(), null, null).data;
        assertArrayEquals(quantizationTables(direct), quantizationTables(ble));
        assertArrayEquals(quantizationTables(direct), quantizationTables(fallback));
        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        BitmapFactory.decodeByteArray(direct, 0, direct.length, bounds);
        assertEquals(120, bounds.outWidth);
        assertEquals(80, bounds.outHeight);
        assertArrayEquals(capturedBytes, Files.readAllBytes(original.toPath()));
      } finally {
        ramCapture.recycle();
        fallbackCapture.recycle();
      }
    }
  }

  @Test
  public void reencodedUploadKeepsExifOrientation() throws Exception {
    Bitmap sensor = Bitmap.createBitmap(40, 20, Bitmap.Config.ARGB_8888);
    File original = files.newFile("rotated.jpg");
    try (java.io.FileOutputStream out = new java.io.FileOutputStream(original)) {
      assertTrue(sensor.compress(Bitmap.CompressFormat.JPEG, 100, out));
    }
    sensor.recycle();
    ExifInterface exif = new ExifInterface(original.getPath());
    exif.setAttribute(ExifInterface.TAG_ORIENTATION,
        String.valueOf(ExifInterface.ORIENTATION_ROTATE_90));
    exif.saveAttributes();

    File upload = files.newFile("rotated-low.jpg");
    PhotoCompression.LOW.prepareUpload(original.getPath(), upload.getPath());

    assertEquals(ExifInterface.ORIENTATION_ROTATE_90,
        new ExifInterface(upload.getPath()).getAttributeInt(
            ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_UNDEFINED));
  }

  @Test
  public void legacyAliasAndOmittedCompressionNormalizeOnce() {
    assertSame(PhotoCompression.HIGH, PhotoCompression.fromValue("heavy"));
    assertSame(PhotoCompression.NONE, PhotoCompression.fromValue(null));
    assertSame(PhotoCompression.NONE, PhotoCompression.fromValue(""));
    assertSame(PhotoCompression.NONE, PhotoCompression.fromValue("unknown"));
  }

  // Compare JPEG quality independently of transport-specific EXIF metadata.
  private static byte[] quantizationTables(byte[] jpeg) {
    ByteArrayOutputStream tables = new ByteArrayOutputStream();
    for (int offset = 2; offset + 4 <= jpeg.length; ) {
      int marker = jpeg[offset + 1] & 0xff;
      if (marker == 0xda || marker == 0xd9) break;
      int length = ((jpeg[offset + 2] & 0xff) << 8) | (jpeg[offset + 3] & 0xff);
      if (marker == 0xdb) tables.write(jpeg, offset + 4, length - 2);
      offset += length + 2;
    }
    assertTrue("JPEG must contain quantization tables", tables.size() > 0);
    return tables.toByteArray();
  }
}
