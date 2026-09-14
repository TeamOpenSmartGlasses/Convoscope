package com.mentra.asg_client.service.core.handlers;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.mentra.asg_client.service.communication.interfaces.ICommunicationManager;
import com.mentra.asg_client.service.communication.interfaces.IResponseBuilder;
import com.mentra.asg_client.service.legacy.managers.AsgClientServiceManager;
import com.mentra.asg_client.settings.AsgSettings;

import org.json.JSONObject;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.annotation.Config;

import java.util.ArrayList;
import java.util.List;

/** The stored button-photo compression preset uses the canonical none/low/medium/high tiers. */
@RunWith(RobolectricTestRunner.class)
@Config(sdk = 33)
public class SettingsCommandHandlerButtonPhotoCompressTest {

    private AsgSettings settings;
    private SettingsCommandHandler handler;
    private final List<JSONObject> responses = new ArrayList<>();

    @Before
    public void setUp() {
        AsgClientServiceManager serviceManager = mock(AsgClientServiceManager.class);
        ICommunicationManager communicationManager = mock(ICommunicationManager.class);
        settings = mock(AsgSettings.class);
        when(serviceManager.getAsgSettings()).thenReturn(settings);
        when(settings.getButtonPhotoSize()).thenReturn("medium");
        when(communicationManager.sendBluetoothResponse(org.mockito.ArgumentMatchers.any()))
                .thenAnswer(
                        invocation -> {
                            responses.add(invocation.getArgument(0));
                            return true;
                        });
        handler =
                new SettingsCommandHandler(
                        serviceManager, communicationManager, mock(IResponseBuilder.class));
    }

    @Test
    public void canonicalTiersArePersistedAndEchoedVerbatim() throws Exception {
        for (String tier : new String[] {"none", "low", "medium", "high"}) {
            responses.clear();
            assertThat(handler.handleCommand("button_photo_setting", request(tier))).isTrue();
            verify(settings).setButtonPhotoCompress(tier);
            assertThat(lastAck().getString("compress")).isEqualTo(tier);
        }
    }

    @Test
    public void legacyHeavyIsStoredAndAckedAsHigh() throws Exception {
        assertThat(handler.handleCommand("button_photo_setting", request("heavy"))).isTrue();
        verify(settings).setButtonPhotoCompress("high");
        verify(settings, never()).setButtonPhotoCompress("heavy");
        assertThat(lastAck().getString("compress")).isEqualTo("high");
    }

    @Test
    public void unknownCompressionFallsBackToNone() throws Exception {
        assertThat(handler.handleCommand("button_photo_setting", request("ultra"))).isTrue();
        verify(settings).setButtonPhotoCompress("none");
        assertThat(lastAck().getString("compress")).isEqualTo("none");
    }

    @Test
    public void omittedCompressionLeavesStoredPresetAlone() throws Exception {
        JSONObject data = new JSONObject().put("request_id", "req-1").put("size", "high");
        assertThat(handler.handleCommand("button_photo_setting", data)).isTrue();
        verify(settings, never()).setButtonPhotoCompress(org.mockito.ArgumentMatchers.anyString());
        assertThat(lastAck().has("compress")).isFalse();
    }

    private JSONObject lastAck() {
        assertThat(responses).isNotEmpty();
        JSONObject ack = responses.get(responses.size() - 1);
        assertThat(ack.optString("setting")).isEqualTo("button_photo");
        assertThat(ack.optString("status")).isEqualTo("applied");
        return ack;
    }

    private static JSONObject request(String compress) throws Exception {
        return new JSONObject().put("request_id", "req-" + compress).put("compress", compress);
    }
}
