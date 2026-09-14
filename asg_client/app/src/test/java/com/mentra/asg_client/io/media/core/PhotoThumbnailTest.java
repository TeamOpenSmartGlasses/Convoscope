package com.mentra.asg_client.io.media.core;

import static org.junit.Assert.*;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import com.mentra.asg_client.AsgConstants;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.annotation.Config;
import org.robolectric.annotation.GraphicsMode;

@RunWith(RobolectricTestRunner.class)
@Config(manifest = Config.NONE, sdk = 28)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
public class PhotoThumbnailTest {
    @Test
    public void previewPreservesAspectRatioAndNeverUpscales() throws Exception {
        checkSize(1600, 1200, 500, 375);
        checkSize(1200, 1600, 375, 500);
        checkSize(500, 500, 500, 500);
        checkSize(200, 100, 200, 100);
        assertEquals(50, AsgConstants.PHOTO_THUMBNAIL_JPEG_QUALITY);
    }

    private void checkSize(int width, int height, int expectedWidth, int expectedHeight)
            throws Exception {
        Bitmap source = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        byte[] bytes = PhotoThumbnail.encode(source);
        assertFalse(source.isRecycled());
        assertEquals(width, source.getWidth());
        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        BitmapFactory.decodeByteArray(bytes, 0, bytes.length, bounds);
        assertEquals(expectedWidth, bounds.outWidth);
        assertEquals(expectedHeight, bounds.outHeight);
        assertEquals("image/jpeg", bounds.outMimeType);
        source.recycle();
    }
}
