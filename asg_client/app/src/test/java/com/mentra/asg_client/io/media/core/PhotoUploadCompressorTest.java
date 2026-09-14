package com.mentra.asg_client.io.media.core;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import com.mentra.asg_client.camera.lifecycle.PhotoExifMetadataWriter;
import com.mentra.asg_client.camera.policy.PhotoCompressionLevel;

import org.json.JSONObject;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.annotation.Config;
import org.robolectric.annotation.GraphicsMode;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Random;

/** Wi-Fi upload re-encode: {@code compress} changes JPEG quality only, never dimensions. */
@RunWith(RobolectricTestRunner.class)
@Config(manifest = Config.NONE, sdk = 28)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
public class PhotoUploadCompressorTest {
    private static final int WIDTH = 640;
    private static final int HEIGHT = 480;

    @Rule public TemporaryFolder folder = new TemporaryFolder();

    private File original;

    @Before
    public void writeNoisyCapture() throws Exception {
        // Random pixels defeat JPEG's entropy coding so quality changes are visible in file size.
        Bitmap source = Bitmap.createBitmap(WIDTH, HEIGHT, Bitmap.Config.ARGB_8888);
        Random random = new Random(1234);
        int[] pixels = new int[WIDTH * HEIGHT];
        for (int i = 0; i < pixels.length; i++) {
            pixels[i] = 0xFF000000 | random.nextInt(0x00FFFFFF);
        }
        source.setPixels(pixels, 0, WIDTH, 0, 0, WIDTH, HEIGHT);
        original = new File(folder.newFolder("IMG_test"), "base.jpg");
        try (FileOutputStream out = new FileOutputStream(original)) {
            assertTrue(source.compress(Bitmap.CompressFormat.JPEG, 100, out));
        }
        source.recycle();
    }

    @Test
    public void noneUploadsTheOriginalFileUntouched() throws Exception {
        long before = original.length();
        File result = PhotoUploadCompressor.compress(original, PhotoCompressionLevel.NONE);
        assertSame(original, result);
        assertEquals(before, original.length());
        assertEquals(1, original.getParentFile().listFiles().length);
    }

    @Test
    public void reencodedTiersKeepDimensionsAndShrinkWithHigherCompression() throws Exception {
        File low = PhotoUploadCompressor.compress(original, PhotoCompressionLevel.LOW);
        File medium = PhotoUploadCompressor.compress(original, PhotoCompressionLevel.MEDIUM);
        File high = PhotoUploadCompressor.compress(original, PhotoCompressionLevel.HIGH);

        for (File file : new File[] {low, medium, high}) {
            assertNotEquals(original.getAbsolutePath(), file.getAbsolutePath());
            assertTrue(file.exists());
            BitmapFactory.Options bounds = new BitmapFactory.Options();
            bounds.inJustDecodeBounds = true;
            BitmapFactory.decodeFile(file.getAbsolutePath(), bounds);
            assertEquals("image/jpeg", bounds.outMimeType);
            assertEquals(WIDTH, bounds.outWidth);
            assertEquals(HEIGHT, bounds.outHeight);
        }

        assertTrue("low should be smaller than the Q100 source", low.length() < original.length());
        assertTrue("medium should be smaller than low", medium.length() < low.length());
        assertTrue("high should be smaller than medium", high.length() < medium.length());
        assertTrue("original must survive re-encoding", original.exists());
    }

    @Test
    public void outputFileIsNamedByCanonicalTierBesideTheOriginal() {
        File out = PhotoUploadCompressor.outputFileFor(original, PhotoCompressionLevel.HIGH);
        assertEquals(original.getParentFile(), out.getParentFile());
        assertEquals("base_compressed_high.jpg", out.getName());
        assertEquals(
                "base_compressed_low.jpg",
                PhotoUploadCompressor.outputFileFor(original, PhotoCompressionLevel.LOW).getName());
        assertEquals(
                "noext_compressed_medium.jpg",
                PhotoUploadCompressor.outputFileFor(
                                new File(original.getParentFile(), "noext"),
                                PhotoCompressionLevel.MEDIUM)
                        .getName());
    }

    @Test
    public void legacyHeavyAndHighProduceTheSameOutput() throws Exception {
        File high = PhotoUploadCompressor.compress(original, PhotoCompressionLevel.HIGH);
        File heavy =
                PhotoUploadCompressor.compress(original, PhotoCompressionLevel.fromWire("heavy"));
        assertEquals(high.getAbsolutePath(), heavy.getAbsolutePath());
        assertEquals(high.length(), heavy.length());
    }

    @Test
    public void captureExifSurvivesReencode() throws Exception {
        JSONObject imu = new JSONObject().put("yaw", 1.5).put("pitch", -0.25);
        PhotoExifMetadataWriter.writeImuPayload(original.getAbsolutePath(), imu);
        assertTrue(PhotoExifMetadataWriter.hasImuMetadata(original.getAbsolutePath()));

        File medium = PhotoUploadCompressor.compress(original, PhotoCompressionLevel.MEDIUM);
        assertTrue(PhotoExifMetadataWriter.hasImuMetadata(medium.getAbsolutePath()));
        assertEquals(
                PhotoExifMetadataWriter.readImuJsonFromJpeg(original.getAbsolutePath()),
                PhotoExifMetadataWriter.readImuJsonFromJpeg(medium.getAbsolutePath()));
    }

    @Test
    public void undecodableSourceFailsInsteadOfUploadingGarbage() throws Exception {
        File broken = new File(original.getParentFile(), "broken.jpg");
        try (FileOutputStream out = new FileOutputStream(broken)) {
            out.write(new byte[] {1, 2, 3, 4});
        }
        assertThrows(
                IOException.class,
                () -> PhotoUploadCompressor.compress(broken, PhotoCompressionLevel.MEDIUM));
        assertFalse(
                PhotoUploadCompressor.outputFileFor(broken, PhotoCompressionLevel.MEDIUM).exists());
    }
}
