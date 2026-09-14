package com.mentra.asg_client.camera.policy;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.mentra.asg_client.AsgConstants;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

@RunWith(RobolectricTestRunner.class)
public class PhotoCompressionLevelTest {

    @Test
    public void canonicalTiersParseToThemselves() {
        assertEquals(PhotoCompressionLevel.NONE, PhotoCompressionLevel.fromWire("none"));
        assertEquals(PhotoCompressionLevel.LOW, PhotoCompressionLevel.fromWire("low"));
        assertEquals(PhotoCompressionLevel.MEDIUM, PhotoCompressionLevel.fromWire("medium"));
        assertEquals(PhotoCompressionLevel.HIGH, PhotoCompressionLevel.fromWire("high"));
    }

    @Test
    public void legacyHeavyIsHigh() {
        assertEquals(PhotoCompressionLevel.HIGH, PhotoCompressionLevel.fromWire("heavy"));
        assertEquals("high", PhotoCompressionLevel.normalize("heavy"));
    }

    @Test
    public void omittedBlankAndUnknownFallBackToNone() {
        assertEquals(PhotoCompressionLevel.NONE, PhotoCompressionLevel.fromWire(null));
        assertEquals(PhotoCompressionLevel.NONE, PhotoCompressionLevel.fromWire(""));
        assertEquals(PhotoCompressionLevel.NONE, PhotoCompressionLevel.fromWire("   "));
        assertEquals(PhotoCompressionLevel.NONE, PhotoCompressionLevel.fromWire("ultra"));
        assertEquals("none", PhotoCompressionLevel.normalize(null));
        assertEquals("none", PhotoCompressionLevel.normalize("bogus"));
    }

    @Test
    public void parsingIsCaseAndWhitespaceTolerant() {
        assertEquals(PhotoCompressionLevel.MEDIUM, PhotoCompressionLevel.fromWire(" Medium "));
        assertEquals(PhotoCompressionLevel.HIGH, PhotoCompressionLevel.fromWire("HEAVY"));
    }

    @Test
    public void qualityLadderIsStrictlyDecreasingFromNoneToHigh() {
        assertEquals(AsgConstants.PHOTO_JPEG_QUALITY_NONE, PhotoCompressionLevel.NONE.jpegQuality());
        assertEquals(AsgConstants.PHOTO_JPEG_QUALITY_LOW, PhotoCompressionLevel.LOW.jpegQuality());
        assertEquals(
                AsgConstants.PHOTO_JPEG_QUALITY_MEDIUM, PhotoCompressionLevel.MEDIUM.jpegQuality());
        assertEquals(AsgConstants.PHOTO_JPEG_QUALITY_HIGH, PhotoCompressionLevel.HIGH.jpegQuality());

        assertEquals(95, PhotoCompressionLevel.NONE.jpegQuality());
        assertEquals(88, PhotoCompressionLevel.LOW.jpegQuality());
        assertEquals(78, PhotoCompressionLevel.MEDIUM.jpegQuality());
        assertEquals(60, PhotoCompressionLevel.HIGH.jpegQuality());

        PhotoCompressionLevel[] ladder = PhotoCompressionLevel.values();
        for (int i = 1; i < ladder.length; i++) {
            assertTrue(
                    ladder[i - 1] + " must be higher quality than " + ladder[i],
                    ladder[i - 1].jpegQuality() > ladder[i].jpegQuality());
        }
    }

    @Test
    public void onlyNoneSkipsUploadReencode() {
        assertFalse(PhotoCompressionLevel.NONE.reencodesUpload());
        assertTrue(PhotoCompressionLevel.LOW.reencodesUpload());
        assertTrue(PhotoCompressionLevel.MEDIUM.reencodesUpload());
        assertTrue(PhotoCompressionLevel.HIGH.reencodesUpload());
    }

    @Test
    public void wireValuesRoundTrip() {
        for (PhotoCompressionLevel level : PhotoCompressionLevel.values()) {
            assertEquals(level, PhotoCompressionLevel.fromWire(level.wireValue()));
            assertEquals(level.wireValue(), PhotoCompressionLevel.normalize(level.wireValue()));
        }
    }
}
