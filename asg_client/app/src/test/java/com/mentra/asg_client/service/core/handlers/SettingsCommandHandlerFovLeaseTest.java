package com.mentra.asg_client.service.core.handlers;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.never;

import android.content.Context;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraCharacteristics;
import android.os.Looper;
import com.mentra.asg_client.AsgConstants;
import com.mentra.asg_client.service.core.CameraRestartCooldown;
import java.lang.reflect.Method;
import java.time.Duration;
import org.robolectric.Shadows;
import org.robolectric.annotation.LooperMode;
import com.dev.api.DevApi;
import com.mentra.asg_client.service.communication.interfaces.ICommunicationManager;
import com.mentra.asg_client.service.communication.interfaces.IResponseBuilder;
import com.mentra.asg_client.service.legacy.managers.AsgClientServiceManager;
import com.mentra.asg_client.service.system.core.SystemControllerFactory;
import com.mentra.asg_client.service.system.interfaces.ISystemController;
import com.mentra.asg_client.settings.AsgSettings;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import org.json.JSONObject;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.MockedStatic;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.annotation.Config;

/** Regression coverage for camera FOV lease teardown when ASG services are unavailable. */
@RunWith(RobolectricTestRunner.class)
@Config(sdk = 33)
@LooperMode(LooperMode.Mode.PAUSED)
public class SettingsCommandHandlerFovLeaseTest {

    private AsgClientServiceManager serviceManager;
    private ICommunicationManager communicationManager;
    private SettingsCommandHandler handler;
    private final List<JSONObject> responses = new ArrayList<>();

    @Before
    public void setUp() {
        serviceManager = mock(AsgClientServiceManager.class);
        communicationManager = mock(ICommunicationManager.class);
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
    public void reconnectSyncIsIdempotentAndBusyChangesPreservePreferencesAndLease() throws Exception {
        Context context = mock(Context.class);
        ISystemController systemController = mock(ISystemController.class);
        AsgSettings settings = mock(AsgSettings.class);
        when(serviceManager.getContext()).thenReturn(context);
        registerCamera(context);
        when(serviceManager.getAsgSettings()).thenReturn(settings);
        when(settings.getCameraFov()).thenReturn(118);
        when(settings.getCameraRoiPosition()).thenReturn(0);
        AtomicBoolean busy = new AtomicBoolean(false);
        handler = new SettingsCommandHandler(serviceManager, communicationManager,
                mock(IResponseBuilder.class), busy::get);
        JSONObject sync = new JSONObject().put("request_id", "sync")
                .put("params", new JSONObject().put("fov", 118).put("roi_position", 0));
        try (MockedStatic<DevApi> hardware = mockStatic(DevApi.class);
                MockedStatic<SystemControllerFactory> controllers = mockStatic(SystemControllerFactory.class)) {
            controllers.when(() -> SystemControllerFactory.get(context)).thenReturn(systemController);
            assertThat(handler.handleCommand("camera_fov_setting", sync)).isTrue();
            finishFovUpdate();
            busy.set(true);
            assertThat(handler.handleCommand("camera_fov_setting", sync)).isTrue();
            finishFovUpdate();
            verify(systemController, times(1)).restartCameraHal();
            hardware.verify(() -> DevApi.setCameraFov(118, 0), times(1));

            JSONObject changed = new JSONObject().put("request_id", "changed")
                    .put("params", new JSONObject().put("fov", 90).put("roi_position", 1));
            assertThat(handler.handleCommand("camera_fov_setting", changed)).isTrue();
            verify(settings, never()).setCameraFov(90, 1);
            assertThat(responses.get(responses.size() - 1).getString("error_code")).isEqualTo("camera_busy");
            assertThat(handler.handleCommand("camera_fov_override", overrideRequest("lease-1"))).isTrue();
            assertThat(responses.get(responses.size() - 1).getString("error_code")).isEqualTo("camera_busy");

            busy.set(false);
            assertThat(handler.handleCommand("camera_fov_override", overrideRequest("lease-1"))).isTrue();
            finishFovUpdate();
            busy.set(true);
            assertThat(handler.handleCommand("camera_fov_override_release", releaseRequest("lease-1"))).isTrue();
            assertThat(responses.get(responses.size() - 1).getString("error_code")).isEqualTo("camera_busy");
            verify(systemController, times(2)).restartCameraHal();
            busy.set(false);
            assertThat(handler.handleCommand("camera_fov_override_release", releaseRequest("lease-1"))).isTrue();
            verify(systemController, times(3)).restartCameraHal();
        }
    }

    @Test
    public void releaseRetainsLeaseWhenContextIsUnavailable() throws Exception {
        Context context = mock(Context.class);
        ISystemController systemController = mock(ISystemController.class);
        AsgSettings settings = mock(AsgSettings.class);
        when(serviceManager.getContext()).thenReturn(context);
        registerCamera(context);
        when(serviceManager.getAsgSettings()).thenReturn(settings);
        when(settings.getCameraFov()).thenReturn(102);
        when(settings.getCameraRoiPosition()).thenReturn(1);

        try (MockedStatic<DevApi> ignored = mockStatic(DevApi.class);
                MockedStatic<SystemControllerFactory> systemControllers =
                        mockStatic(SystemControllerFactory.class)) {
            systemControllers
                    .when(() -> SystemControllerFactory.get(context))
                    .thenReturn(systemController);

            assertThat(handler.handleCommand("camera_fov_override", overrideRequest("lease-1")))
                    .isTrue();

            finishFovUpdate();
            when(serviceManager.getContext()).thenReturn(null);
            assertThat(
                            handler.handleCommand(
                                    "camera_fov_override_release", releaseRequest("lease-1")))
                    .isTrue();
            assertThat(
                            handler.handleCommand(
                                    "camera_fov_override_release", releaseRequest("lease-1")))
                    .isTrue();
        }

        JSONObject lastResponse = responses.get(responses.size() - 1);
        assertThat(lastResponse.getString("status")).isEqualTo("error");
        assertThat(lastResponse.getString("error_code")).isEqualTo("camera_unavailable");
        assertThat(lastResponse.optBoolean("stale", false)).isFalse();
    }

    @Test
    public void readyWaitsForReadableCameraAfterCooldown() throws Exception {
        CameraManager camera = cameraManager();
        when(camera.getCameraIdList()).thenReturn(new String[] {"0"});
        when(camera.getCameraCharacteristics("0"))
                .thenThrow(new IllegalArgumentException("Unknown camera ID 0"));
        CameraRestartCooldown.setCooldownMs(5000);
        requestReadyAck();
        Shadows.shadowOf(Looper.getMainLooper()).idleFor(Duration.ofSeconds(11));
        assertThat(responses).isEmpty();
        org.mockito.Mockito.doReturn(mock(CameraCharacteristics.class))
                .when(camera).getCameraCharacteristics("0");
        Shadows.shadowOf(Looper.getMainLooper()).idleFor(Duration.ofMillis(250));
        assertThat(responses).hasSize(1);
        assertThat(responses.get(0).getString("status")).isEqualTo("ready");
        Shadows.shadowOf(Looper.getMainLooper()).idleFor(Duration.ofSeconds(25));
        assertThat(responses).hasSize(1);
    }

    @Test
    public void missingCameraReturnsErrorAndNeverReady() throws Exception {
        CameraManager camera = cameraManager();
        when(camera.getCameraIdList()).thenReturn(new String[0]);
        CameraRestartCooldown.setCooldownMs(0);
        requestReadyAck();
        Shadows.shadowOf(Looper.getMainLooper()).idleFor(
                Duration.ofMillis(AsgConstants.CAMERA_FOV_READY_TIMEOUT_MS));
        assertThat(responses).hasSize(1);
        assertThat(responses.get(0).getString("error_code")).isEqualTo("camera_unavailable");
        assertThat(responses.get(0).getBoolean("ready")).isFalse();
    }

    @Test
    public void registeredCameraStillWaitsForRestartCooldown() throws Exception {
        CameraManager camera = cameraManager();
        when(camera.getCameraIdList()).thenReturn(new String[] {"0"});
        when(camera.getCameraCharacteristics("0")).thenReturn(mock(CameraCharacteristics.class));
        CameraRestartCooldown.setCooldownMs(5000);
        requestReadyAck();
        Shadows.shadowOf(Looper.getMainLooper()).idleFor(Duration.ofSeconds(4));
        assertThat(responses).isEmpty();
        Shadows.shadowOf(Looper.getMainLooper()).idleFor(Duration.ofSeconds(1));
        assertThat(responses).hasSize(1);
        assertThat(responses.get(0).getString("status")).isEqualTo("ready");
    }

    private void registerCamera(Context context) throws Exception {
        CameraManager camera = mock(CameraManager.class);
        when(context.getSystemService(Context.CAMERA_SERVICE)).thenReturn(camera);
        when(camera.getCameraIdList()).thenReturn(new String[] {"0"});
        when(camera.getCameraCharacteristics("0")).thenReturn(mock(CameraCharacteristics.class));
    }

    private void finishFovUpdate() {
        Shadows.shadowOf(Looper.getMainLooper()).idleFor(Duration.ofSeconds(11));
    }

    private CameraManager cameraManager() {
        Context context = mock(Context.class);
        CameraManager camera = mock(CameraManager.class);
        when(serviceManager.getContext()).thenReturn(context);
        when(context.getSystemService(Context.CAMERA_SERVICE)).thenReturn(camera);
        return camera;
    }

    private void requestReadyAck() throws Exception {
        Method method = SettingsCommandHandler.class.getDeclaredMethod(
                "sendCameraFovReadyAck", String.class, int.class, int.class);
        method.setAccessible(true);
        method.invoke(handler, "ready-test", 62, 0);
    }

    private static JSONObject overrideRequest(String leaseId) throws Exception {
        return new JSONObject()
                .put("request_id", "set-1")
                .put(
                        "params",
                        new JSONObject()
                                .put("lease_id", leaseId)
                                .put("fov", 82)
                                .put("roi_position", 1)
                                .put("ttl_ms", 300_000));
    }

    private static JSONObject releaseRequest(String leaseId) throws Exception {
        return new JSONObject()
                .put("request_id", "release-1")
                .put("params", new JSONObject().put("lease_id", leaseId));
    }
}
