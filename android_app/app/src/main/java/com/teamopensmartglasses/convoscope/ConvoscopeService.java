package com.teamopensmartglasses.convoscope;

import static com.teamopensmartglasses.convoscope.Constants.BUTTON_EVENT_ENDPOINT;
import static com.teamopensmartglasses.convoscope.Constants.DIARIZE_QUERY_ENDPOINT;
import static com.teamopensmartglasses.convoscope.Constants.LLM_QUERY_ENDPOINT;
import static com.teamopensmartglasses.convoscope.Constants.UI_POLL_ENDPOINT;
import static com.teamopensmartglasses.convoscope.Constants.GEOLOCATION_STREAM_ENDPOINT;
import static com.teamopensmartglasses.convoscope.Constants.SET_USER_SETTINGS_ENDPOINT;
import static com.teamopensmartglasses.convoscope.Constants.GET_USER_SETTINGS_ENDPOINT;
import static com.teamopensmartglasses.convoscope.Constants.adhdStmbAgentKey;
import static com.teamopensmartglasses.convoscope.Constants.entityDefinitionsKey;
import static com.teamopensmartglasses.convoscope.Constants.explicitAgentQueriesKey;
import static com.teamopensmartglasses.convoscope.Constants.explicitAgentResultsKey;
import static com.teamopensmartglasses.convoscope.Constants.glassesCardTitle;
import static com.teamopensmartglasses.convoscope.Constants.languageLearningKey;
import static com.teamopensmartglasses.convoscope.Constants.llContextConvoKey;
import static com.teamopensmartglasses.convoscope.Constants.llWordSuggestUpgradeKey;
import static com.teamopensmartglasses.convoscope.Constants.proactiveAgentResultsKey;
import static com.teamopensmartglasses.convoscope.Constants.shouldUpdateSettingsKey;
import static com.teamopensmartglasses.convoscope.Constants.displayRequestsKey;
import static com.teamopensmartglasses.convoscope.Constants.wakeWordTimeKey;

import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.hardware.display.VirtualDisplay;
import android.media.projection.MediaProjection;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.service.notification.NotificationListenerService;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.preference.PreferenceManager;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;
import com.google.firebase.auth.GetTokenResult;
import com.teamopensmartglasses.convoscope.events.GoogleAuthFailedEvent;
import com.teamopensmartglasses.convoscope.events.GoogleAuthSucceedEvent;
import com.teamopensmartglasses.convoscope.convoscopebackend.BackendServerComms;
import com.teamopensmartglasses.convoscope.convoscopebackend.VolleyJsonCallback;
import com.teamopensmartglasses.convoscope.events.NewScreenImageEvent;
import com.teamopensmartglasses.convoscope.events.NewScreenTextEvent;
import com.teamopensmartglasses.convoscope.events.SignOutEvent;
import com.teamopensmartglasses.convoscope.ui.ConvoscopeUi;
import com.teamopensmartglasses.smartglassesmanager.eventbusmessages.DiarizationOutputEvent;
import com.teamopensmartglasses.smartglassesmanager.eventbusmessages.GlassesTapOutputEvent;
import com.teamopensmartglasses.smartglassesmanager.eventbusmessages.SmartRingButtonOutputEvent;
import com.teamopensmartglasses.smartglassesmanager.eventbusmessages.SpeechRecOutputEvent;

import org.greenrobot.eventbus.EventBus;
import org.greenrobot.eventbus.Subscribe;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Objects;
import java.util.LinkedList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import net.sourceforge.pinyin4j.PinyinHelper;
import net.sourceforge.pinyin4j.format.HanyuPinyinOutputFormat;
import net.sourceforge.pinyin4j.format.HanyuPinyinCaseType;
import net.sourceforge.pinyin4j.format.HanyuPinyinToneType;
import net.sourceforge.pinyin4j.format.HanyuPinyinVCharType;
import net.sourceforge.pinyin4j.format.exception.BadHanyuPinyinOutputFormatCombination;
import com.huaban.analysis.jieba.JiebaSegmenter;
import com.huaban.analysis.jieba.SegToken;

import com.teamopensmartglasses.smartglassesmanager.SmartGlassesAndroidService;
import com.teamopensmartglasses.smartglassesmanager.smartglassescommunicators.SmartGlassesFontSize;
import com.teamopensmartglasses.smartglassesmanager.speechrecognition.ASR_FRAMEWORKS;
import com.teamopensmartglasses.smartglassesmanager.supportedglasses.SmartGlassesDevice;
import com.teamopensmartglasses.smartglassesmanager.supportedglasses.SmartGlassesOperatingSystem;

public class ConvoscopeService extends SmartGlassesAndroidService {
    public final String TAG = "Convoscope_ConvoscopeService";

    private final IBinder binder = new LocalBinder();

    //our instance of the SGM library
    //public SGMLib sgmLib;

    private FirebaseAuth firebaseAuth;
    private FirebaseAuth.AuthStateListener authStateListener;
    private FirebaseAuth.IdTokenListener idTokenListener;

    //Convoscope stuff
    String authToken = "";
    private BackendServerComms backendServerComms;
    ArrayList<String> responsesBuffer;
    ArrayList<String> transcriptsBuffer;
    ArrayList<String> responsesToShare;
    private final Handler csePollLoopHandler = new Handler(Looper.getMainLooper());
    private Runnable cseRunnableCode;
    private final Handler displayPollLoopHandler = new Handler(Looper.getMainLooper());
    private final Handler locationSendingLoopHandler = new Handler(Looper.getMainLooper());
    private MediaProjection mediaProjection;
    private VirtualDisplay virtualDisplay;
    private Handler screenCaptureHandler = new Handler();
    private Runnable screenCaptureRunnable;
    private Runnable uiPollRunnableCode;
    private Runnable displayRunnableCode;
    private Runnable locationSendingRunnableCode;
    private boolean shouldPoll = true;
    private long lastDataSentTime = 0;
    private final long POLL_INTERVAL_ACTIVE = 200; // 200ms when actively sending data
    private final long POLL_INTERVAL_INACTIVE = 5000; // 5000ms (5s) when inactive
    private final long DATA_SENT_THRESHOLD = 90000; // 90 seconds
    private LocationSystem locationSystem;
    static final String deviceId = "android";
    public String proactiveAgents = "proactive_agent_insights";
    public String explicitAgent = "explicit_agent_insights";
    public String definerAgent = "intelligent_entity_definitions";
    public String languageLearningAgent = "language_learning";
    public String llWordSuggestUpgradeAgent = "ll_word_suggest_upgrade";
    public String llContextConvoAgent = "ll_context_convo";
    public String adhdStmbAgent = "adhd_stmb_agent_summaries";
    public double previousLat = 0;
    public double previousLng = 0;

    //language learning buffer stuff
    private LinkedList<DefinedWord> definedWords = new LinkedList<>();
    private LinkedList<STMBSummary> adhdStmbSummaries = new LinkedList<>();
    private LinkedList<LLUpgradeResponse> llUpgradeResponses = new LinkedList<>();
    private LinkedList<LLCombineResponse> llCombineResponses = new LinkedList<>();
    private LinkedList<ContextConvoResponse> contextConvoResponses = new LinkedList<>();
    private final long llDefinedWordsShowTime = 40 * 1000; // define in milliseconds
    private final long llContextConvoResponsesShowTime = 3 * 60 * 1000; // define in milliseconds
    private final long locationSendTime = 1000 * 10; // define in milliseconds
    private final long adhdSummaryShowTime = 10 * 60 * 1000; // define in milliseconds
    private final long llUpgradeShowTime = 5 * 60 * 1000; // define in milliseconds
    private final long llCombineShowTime = 5 * 60 * 1000; // define in milliseconds
    private final int maxDefinedWordsShow = 4;
    private final int maxLLCombineShow = 5;
    private final int maxAdhdStmbShowNum = 3;
    private final int maxContextConvoResponsesShow = 2;
    private final int maxLLUpgradeResponsesShow = 2;


    //    private SMSComms smsComms;
    static String phoneNumName = "Alex";
    static String phoneNum = "8477367492"; // Alex's phone number. Fun default.
    long previousWakeWordTime = -1; // Initialize this at -1
    int numConsecutiveAuthFailures = 0;
    private long currTime = 0;
    private long lastPressed = 0;
    private long lastTapped = 0;

    //clear screen to start
    public boolean clearedScreenYet = false;

    String oldLiveCaption = "";
    String oldLiveCaptionFinal = "";
    String currentLiveCaption = "";
    String llCurrentString = "";

    // Double clicking constants
    private final long doublePressTimeConst = 420;
    private final long doubleTapTimeConst = 600;

    private DisplayQueue displayQueue;

    public ConvoscopeService() {
        super(ConvoscopeUi.class,
                "convoscope_app",
                3588,
                "Convoscope",
                "Wearable intelligence upgrades. By TeamOpenSmartGlasses.", R.drawable.ic_launcher_foreground);
    }

    @Override
    public void onCreate() {
        super.onCreate();

        //setup event bus subscribers
        this.setupEventBusSubscribers();

        displayQueue = new DisplayQueue();

        //make responses holder
        responsesBuffer = new ArrayList<>();
        responsesToShare = new ArrayList<>();
        responsesBuffer.add("Welcome to Convoscope.");

        //make responses holder
        transcriptsBuffer = new ArrayList<>();

        //setup backend comms
        backendServerComms = new BackendServerComms(this);

        Log.d(TAG, "Convoscope service started");

        String asrApiKey = getResources().getString(R.string.google_api_key);
//        Log.d(TAG, "ASR KEY: " + asrApiKey);
        saveApiKey(this, asrApiKey);

        startNotificationService();

        completeInitialization();
    }

    public void completeInitialization(){
        Log.d(TAG, "COMPLETE CONVOSCOPE INITIALIZATION");
        setUpUiPolling();
        setUpLocationSending();

        //setup ASR version
        ConvoscopeService.saveChosenAsrFramework(this, ASR_FRAMEWORKS.AZURE_ASR_FRAMEWORK);
//        ConvoscopeService.saveChosenAsrFramework(this, ASR_FRAMEWORKS.DEEPGRAM_ASR_FRAMEWORK);
//        ConvoscopeService.saveChosenAsrFramework(this, ASR_FRAMEWORKS.GOOGLE_ASR_FRAMEWORK);

        //setup mode if not set yet
        getCurrentMode(this);

        this.aioConnectSmartGlasses();

        //update settings on backend on launch
        updateTargetLanguageOnBackend(this);
        updateSourceLanguageOnBackend(this);
        updateVocabularyUpgradeOnBackend(this);
        saveCurrentMode(this, getCurrentMode(this));
    }

    @Override
    protected void onGlassesConnected(SmartGlassesDevice device) {
        Log.d(TAG, "Glasses connected successfully: " + device.deviceModelName);
        setFontSize(SmartGlassesFontSize.LARGE);
        displayQueue.startQueue();
    }

    public void handleSignOut(){
        EventBus.getDefault().post(new SignOutEvent());
    }

    public void sendSettings(JSONObject settingsObj){
        try{
//            Log.d(TAG, "AUTH from Settings: " + authToken);
            settingsObj.put("timestamp", System.currentTimeMillis() / 1000);
            backendServerComms.restRequest(SET_USER_SETTINGS_ENDPOINT, settingsObj, new VolleyJsonCallback(){
                @Override
                public void onSuccess(JSONObject result){
                    try {
                        Log.d(TAG, "GOT Settings update result: " + result.toString());
                        String query_answer = result.getString("message");
                    } catch (JSONException e) {
                        throw new RuntimeException(e);
                    }
                }
                @Override
                public void onFailure(int code){
                    Log.d(TAG, "SOME FAILURE HAPPENED (sendSettings)");

                }

            });
        } catch (JSONException e){
            e.printStackTrace();
        }
    }

    public void getSettings(){
        try{
            Log.d(TAG, "Runnign get settings");
            Context mContext = this.getApplicationContext();
            JSONObject getSettingsObj = new JSONObject();
            backendServerComms.restRequest(GET_USER_SETTINGS_ENDPOINT, getSettingsObj, new VolleyJsonCallback(){
                @Override
                public void onSuccess(JSONObject result){
                    try {
                        Log.d(TAG, "GOT GET Settings update result: " + result.toString());
                        JSONObject settings = result.getJSONObject("settings");
                        Boolean useDynamicTranscribeLanguage = settings.getBoolean("use_dynamic_transcribe_language");
                        String dynamicTranscribeLanguage = settings.getString("dynamic_transcribe_language");
                        Log.d(TAG, "Should use dynamic? " + useDynamicTranscribeLanguage);
                        if (useDynamicTranscribeLanguage){
                            Log.d(TAG, "Switching running transcribe language to: " + dynamicTranscribeLanguage);
                            switchRunningTranscribeLanguage(dynamicTranscribeLanguage);
                        } else {
                            switchRunningTranscribeLanguage(getChosenTranscribeLanguage(mContext));
                        }
                    } catch (JSONException e) {
                        throw new RuntimeException(e);
                    }
                }
                @Override
                public void onFailure(int code){
                    Log.d(TAG, "SOME FAILURE HAPPENED (getSettings)");
                }
            });
        } catch (Exception e){
            e.printStackTrace();
            Log.d(TAG, "SOME FAILURE HAPPENED (getSettings)");
        }
    }

    public void setUpUiPolling(){
        uiPollRunnableCode = new Runnable() {
            @Override
            public void run() {
                if (shouldPoll) {
                    requestUiPoll();
                }
                long currentTime = System.currentTimeMillis();
                long interval = (currentTime - lastDataSentTime < DATA_SENT_THRESHOLD) ? POLL_INTERVAL_ACTIVE : POLL_INTERVAL_INACTIVE;
                csePollLoopHandler.postDelayed(this, interval);
            }
        };
        csePollLoopHandler.post(uiPollRunnableCode);
    }

    public void setUpLocationSending(){
        locationSystem = new LocationSystem(getApplicationContext());

        if (locationSendingLoopHandler != null){
            locationSendingLoopHandler.removeCallbacksAndMessages(this);
        }

        locationSendingRunnableCode = new Runnable() {
            @Override
            public void run() {
                requestLocation();
                locationSendingLoopHandler.postDelayed(this, locationSendTime);
            }
        };
        locationSendingLoopHandler.post(locationSendingRunnableCode);
    }

    @Override
    public void onDestroy(){
        csePollLoopHandler.removeCallbacks(uiPollRunnableCode);
        displayPollLoopHandler.removeCallbacks(displayRunnableCode);
        locationSystem.stopLocationUpdates();
        locationSendingLoopHandler.removeCallbacks(locationSendingRunnableCode);
        locationSendingLoopHandler.removeCallbacksAndMessages(null);
        screenCaptureHandler.removeCallbacks(screenCaptureRunnable);
        if (virtualDisplay != null) virtualDisplay.release();
        if (mediaProjection != null) mediaProjection.stop();
        EventBus.getDefault().unregister(this);

        if (authStateListener != null) {
            firebaseAuth.removeAuthStateListener(authStateListener);
        }
        if (idTokenListener != null) {
            firebaseAuth.removeIdTokenListener(idTokenListener);
        }

        stopNotificationService();

        if (displayQueue != null) displayQueue.stopQueue();

        super.onDestroy();
    }

    public void convoscopeStartCommandCallback(String args, long commandTriggeredTime){
        Log.d("TAG","Convoscope start callback called");
        //runConvoscope();
    }

    @Subscribe
    public void onGlassesTapSideEvent(GlassesTapOutputEvent event) {
        int numTaps = event.numTaps;
        boolean sideOfGlasses = event.sideOfGlasses;
        long time = event.timestamp;

        Log.d(TAG, "GLASSES TAPPED X TIMES: " + numTaps + " SIDEOFGLASSES: " + sideOfGlasses);
        if (numTaps == 3) {
//            sendLatestCSEResultViaSms();
            Log.d(TAG, "GOT A TRIPLE TAP");
        }
    }

    @Subscribe
    public void onSmartRingButtonEvent(SmartRingButtonOutputEvent event) {
        int buttonId = event.buttonId;
        long time = event.timestamp;
        boolean isDown = event.isDown;

        if(!isDown || buttonId != 1) return;
        Log.d(TAG,"DETECTED BUTTON PRESS W BUTTON ID: " + buttonId);
        currTime = System.currentTimeMillis();

        //Detect double presses
        if(isDown && currTime - lastPressed < doublePressTimeConst) {
            Log.d(TAG, "Double tap - CurrTime-lastPressed: "+ (currTime-lastPressed));
//            sendLatestCSEResultViaSms();
        }

        if(isDown) {
            lastPressed = System.currentTimeMillis();
        }
    }

//    public void sendLatestCSEResultViaSms(){
//        if (phoneNum == "") return;
//
//        if (responses.size() > 1) {
//            //Send latest CSE result via sms;
//            String messageToSend = responsesToShare.get(responsesToShare.size() - 1);
//
//            smsComms.sendSms(phoneNum, messageToSend);
//
//            sendReferenceCard("Convoscope", "Sending result(s) via SMS to " + phoneNumName);
//        }
//    }

    private Handler debounceHandler = new Handler(Looper.getMainLooper());
    private Runnable debounceRunnable;

    @Subscribe
    public void onDiarizeData(DiarizationOutputEvent event) {
        Log.d(TAG, "SENDING DIARIZATION STUFF");
        try{
            JSONObject jsonQuery = new JSONObject();
            jsonQuery.put("transcript_meta_data", event.diarizationData);
            jsonQuery.put("timestamp", System.currentTimeMillis() / 1000);
            backendServerComms.restRequest(DIARIZE_QUERY_ENDPOINT, jsonQuery, new VolleyJsonCallback(){
                @Override
                public void onSuccess(JSONObject result){
                    try {
                        parseSendTranscriptResult(result);
                    } catch (JSONException e) {
                        throw new RuntimeException(e);
                    }
                }
                @Override
                public void onFailure(int code){
                    Log.d(TAG, "SOME FAILURE HAPPENED (send Diarize Data)");
                }

            });
        } catch (JSONException e){
            e.printStackTrace();
        }
    }

    @Subscribe
    public void onTranscript(SpeechRecOutputEvent event) {
        String text = event.text;
        long time = event.timestamp;
        boolean isFinal = event.isFinal;
//        Log.d(TAG, "PROCESS TRANSCRIPTION CALLBACK. IS IT FINAL? " + isFinal + " " + text);

        if (isFinal) {
            transcriptsBuffer.add(text);
            sendFinalTranscriptToActivity(text);
        }

        //debounce and then send to backend
        debounceAndSendTranscript(text, isFinal);
//        getSettings();
        // Send transcript to user if live captions are enabled
        if (Objects.equals(getCurrentMode(this), "Language Learning") && getIsLiveCaptionsChecked(this)) {
//            showTranscriptsToUser(text, isFinal);
            debounceAndShowTranscriptOnGlasses(text, isFinal);
        }
    }

    private Handler glassesTranscriptDebounceHandler = new Handler(Looper.getMainLooper());
    private Runnable glassesTranscriptDebounceRunnable;
    private long glassesTranscriptLastSentTime = 0;
    private final long GLASSES_TRANSCRIPTS_DEBOUNCE_DELAY = 400; // in milliseconds
    private void debounceAndShowTranscriptOnGlasses(String transcript, boolean isFinal) {
        glassesTranscriptDebounceHandler.removeCallbacks(glassesTranscriptDebounceRunnable);
        long currentTime = System.currentTimeMillis();
        if (isFinal) {
            showTranscriptsToUser(transcript, isFinal);
        } else { //if intermediate
            if (currentTime - glassesTranscriptLastSentTime >= GLASSES_TRANSCRIPTS_DEBOUNCE_DELAY) {
                showTranscriptsToUser(transcript, isFinal);
                glassesTranscriptLastSentTime = currentTime;
            } else {
                glassesTranscriptDebounceRunnable = () -> {
                    showTranscriptsToUser(transcript, isFinal);
                    lastSentTime = System.currentTimeMillis();
                };
                glassesTranscriptDebounceHandler.postDelayed(glassesTranscriptDebounceRunnable, GLASSES_TRANSCRIPTS_DEBOUNCE_DELAY);
            }
        }
    }

    private void showTranscriptsToUser(final String transcript, final boolean isFinal) {
        String final_transcript;

        if (getChosenTranscribeLanguage(this).equals("Chinese (Pinyin)")) {
            final_transcript = convertToPinyin(transcript);
        } else {
            final_transcript = transcript;
        }

        sendTextWallLiveCaptionLL(final_transcript, isFinal, "");
    }

    private static String convertToPinyin(final String chineseText) {
        final JiebaSegmenter segmenter = new JiebaSegmenter();
        final List<SegToken> tokens = segmenter.process(chineseText, JiebaSegmenter.SegMode.SEARCH);

        final HanyuPinyinOutputFormat format = new HanyuPinyinOutputFormat();
        format.setCaseType(HanyuPinyinCaseType.LOWERCASE);
        format.setToneType(HanyuPinyinToneType.WITH_TONE_MARK);
        format.setVCharType(HanyuPinyinVCharType.WITH_U_UNICODE);

        StringBuilder pinyinText = new StringBuilder();

        for (SegToken token : tokens) {
            StringBuilder tokenPinyin = new StringBuilder();
            for (char character : token.word.toCharArray()) {
                try {
                    String[] pinyinArray = PinyinHelper.toHanyuPinyinStringArray(character, format);
                    if (pinyinArray != null) {
                        // Use the first Pinyin representation if there are multiple
                        tokenPinyin.append(pinyinArray[0]);
                    } else {
                        // If character is not a Chinese character, append it as is
                        tokenPinyin.append(character);
                    }
                } catch (BadHanyuPinyinOutputFormatCombination e) {
                    e.printStackTrace();
                }
            }
            // Ensure the token is concatenated with a space only if it's not empty
            if (tokenPinyin.length() > 0) {
                pinyinText.append(tokenPinyin.toString()).append(" ");
            }
        }

        // Output debug information
        System.out.println("Input: " + chineseText);
        System.out.println("Output Pinyin: " + pinyinText.toString().trim());

        return pinyinText.toString().trim();
    }

    private long lastSentTime = 0;
    private final long DEBOUNCE_DELAY = 250; // in milliseconds
    private void debounceAndSendTranscript(String transcript, boolean isFinal) {
        debounceHandler.removeCallbacks(debounceRunnable);
        long currentTime = System.currentTimeMillis();
        if (isFinal) {
            sendTranscriptRequest(transcript, isFinal);
        } else { //if intermediate
            if (currentTime - lastSentTime >= DEBOUNCE_DELAY) {
                sendTranscriptRequest(transcript, isFinal);
                lastSentTime = currentTime;
            } else {
                debounceRunnable = () -> {
                    sendTranscriptRequest(transcript, isFinal);
                    lastSentTime = System.currentTimeMillis();
                };
                debounceHandler.postDelayed(debounceRunnable, DEBOUNCE_DELAY);
            }
        }
    }

    public void sendTranscriptRequest(String query, boolean isFinal){
        updateLastDataSentTime();
        try{
            JSONObject jsonQuery = new JSONObject();
            jsonQuery.put("text", query);
            jsonQuery.put("transcribe_language", getChosenTranscribeLanguage(this));
            jsonQuery.put("isFinal", isFinal);
            jsonQuery.put("timestamp", System.currentTimeMillis() / 1000);
            backendServerComms.restRequest(LLM_QUERY_ENDPOINT, jsonQuery, new VolleyJsonCallback(){
                @Override
                public void onSuccess(JSONObject result){
                    try {
                        parseSendTranscriptResult(result);
                    } catch (JSONException e) {
                        throw new RuntimeException(e);
                    }
                }
                @Override
                public void onFailure(int code){
                    Log.d(TAG, "SOME FAILURE HAPPENED (sendChatRequest)");
                }

            });
        } catch (JSONException e){
            e.printStackTrace();
        }
    }

    public void requestUiPoll(){
        try{
            JSONObject jsonQuery = new JSONObject();
            jsonQuery.put("deviceId", deviceId);
            backendServerComms.restRequest(UI_POLL_ENDPOINT, jsonQuery, new VolleyJsonCallback(){
                @Override
                public void onSuccess(JSONObject result){
                    try {
                        parseConvoscopeResults(result);
                    } catch (JSONException e) {
                        throw new RuntimeException(e);
                    }
                }
                @Override
                public void onFailure(int code){
                    Log.d(TAG, "SOME FAILURE HAPPENED (requestUiPoll)");
                    if (code == 401){
                        EventBus.getDefault().post(new GoogleAuthFailedEvent("401 AUTH ERROR (requestUiPoll)"));
                    }
                }
            });
        } catch (JSONException e){
            e.printStackTrace();
        }
    }

    public void requestLocation(){
//        Log.d(TAG, "running request locatoin");
        try{
            // Get location data as JSONObject
            double latitude = locationSystem.lat;
            double longitude = locationSystem.lng;

            // TODO: Filter here... is it meaningfully different?
            if(latitude == 0 && longitude == 0) return;

//            Log.d(TAG, "Got a GOOD location!");

            JSONObject jsonQuery = new JSONObject();
            jsonQuery.put("deviceId", deviceId);
            jsonQuery.put("lat", latitude);
            jsonQuery.put("lng", longitude);

            backendServerComms.restRequest(GEOLOCATION_STREAM_ENDPOINT, jsonQuery, new VolleyJsonCallback(){
                @Override
                public void onSuccess(JSONObject result){
                    Log.d(TAG, "Request sent Successfully: " + result.toString());
                }
                @Override
                public void onFailure(int code){
                    Log.d(TAG, "SOME FAILURE HAPPENED (requestLocation)");
                    if (code == 401){
                        EventBus.getDefault().post(new GoogleAuthFailedEvent("401 AUTH ERROR (requestLocation)"));
                    }
                }
            });
        } catch (JSONException e){
            e.printStackTrace();
        }
    }

    public void parseSendTranscriptResult(JSONObject response) throws JSONException {
//        Log.d(TAG, "Got result from server: " + response.toString());
        String message = response.getString("message");
        //DEV
        return;
//        if (!message.equals("")) {
//            responses.add(message);
//            sendUiUpdateSingle(message);
//            speakTTS(message);
//        }
    }

//    public String[] calculateLLStringFormatted(JSONArray jsonArray){
//        //clear canvas if needed
//        if (!clearedScreenYet){
//            sendHomeScreen();
//            clearedScreenYet = true;
//        }
//
//        // Assuming jsonArray is your existing JSONArray object
//        int max_rows_allowed = 4;
//
//        String[] inWords = new String[jsonArray.length()];
//        String[] inWordsTranslations = new String[jsonArray.length()];
//        String[] llResults = new String[max_rows_allowed];
//        String enSpace = "\u2002"; // Using en space for padding
//
//        int minSpaces = 2;
//        for (int i = 0; i < jsonArray.length() && i < max_rows_allowed; i++) {
//            try {
//                JSONObject obj = jsonArray.getJSONObject(i);
//                inWords[i] = obj.getString("in_word");
//                inWordsTranslations[i] = obj.getString("in_word_translation");
//                int max_len = Math.max(inWords[i].length(), inWordsTranslations[i].length());
////                llResults[i] = inWords[i] + enSpace.repeat(Math.max(0, max_len - inWords[i].length()) + minSpaces) + "⟶" + enSpace.repeat(Math.max(0, max_len - inWordsTranslations[i].length()) + minSpaces) + inWordsTranslations[i];
//                llResults[i] = inWords[i] + enSpace.repeat(minSpaces) + "⟶" + enSpace.repeat(minSpaces) + inWordsTranslations[i];
//            } catch (JSONException e){
//                e.printStackTrace();
//            }
//        }
//
//        return llResults;
//
////        String enSpace = "\u2002"; // Using en space for padding
////        String llResult = "";
////        for (int i = 0; i < inWords.length; i++) {
////            String inWord = inWords[i];
////            String translation = inWordsTranslations[i];
////            llResult += inWord + enSpace.repeat(3) + "->"+ enSpace.repeat(3) + translation + "\n\n";
////        }
//
////        StringBuilder topLine = new StringBuilder();
////        StringBuilder bottomLine = new StringBuilder();
////
////        // Calculate initial padding for the first word based on the bottom line's first word
////        int initialPaddingLength = (inWordsTranslations[0].length() - inWords[0].length()) / 2;
////        if (initialPaddingLength > 0) {
////            topLine.append(String.valueOf(enSpace).repeat(initialPaddingLength));
////        } else {
////            initialPaddingLength = 0; // Ensure it's not negative for subsequent calculations
////        }
////
////        for (int i = 0; i < inWords.length; i++) {
////            String inWord = inWords[i];
////            String translation = inWordsTranslations[i];
////
////            topLine.append(inWord);
////            bottomLine.append(translation);
////
////            if (i < inWords.length - 1) {
////                // Calculate the minimum necessary space to add based on the length of the next words in both lines
////                int nextTopWordLength = inWords[i + 1].length();
////                int nextBottomWordLength = inWordsTranslations[i + 1].length();
////                int currentTopWordLength = inWord.length();
////                int currentBottomWordLength = translation.length();
////
////                // Calculate additional space needed for alignment
////                int additionalSpaceTop = nextTopWordLength - currentTopWordLength;
////                int additionalSpaceBottom = nextBottomWordLength - currentBottomWordLength;
////
////                // Ensure there's a minimum spacing for readability, reduce this as needed
////                int minSpace = 2; // Reduced minimum space for closer alignment
////                int spacesToAddTop = Math.max(additionalSpaceTop, minSpace);
////                int spacesToAddBottom = Math.max(additionalSpaceBottom, minSpace);
////
////                // Append the calculated spaces to each line
////                topLine.append(String.valueOf(enSpace).repeat(spacesToAddTop));
////                bottomLine.append(String.valueOf(enSpace).repeat(spacesToAddBottom));
////            }
////        }
////
////
////        // Adjust for the initial padding by ensuring the bottom line starts directly under the top line's first word
////        if (initialPaddingLength > 0) {
////            String initialPaddingForBottom = String.valueOf(enSpace).repeat(initialPaddingLength);
////            bottomLine = new StringBuilder(initialPaddingForBottom).append(bottomLine.toString());
////        }
//
////        String llResult = topLine.toString() + "\n" + bottomLine.toString();
//    }

//    public String[] calculateLLStringFormatted(LinkedList<DefinedWord> definedWords) {
//        int max_rows_allowed = 4;
//        String[] llResults = new String[Math.min(max_rows_allowed, definedWords.size())];
//        String enSpace = "\u2002"; // Using en space for padding
//
//        int minSpaces = 2;
//        int index = 0;
//        for (DefinedWord word : definedWords) {
//            if (index >= max_rows_allowed) break;
//            llResults[index] = word.inWord + enSpace.repeat(minSpaces) + "⟶" + enSpace.repeat(minSpaces) + word.inWordTranslation;
//            index++;
//        }
//
//        return llResults;
//    }

    public String[] calculateLLCombineResponseFormatted(LinkedList<LLCombineResponse> llCombineResponses) {
        int max_rows_allowed = 4;
        String[] llCombineResults = new String[Math.min(max_rows_allowed, llCombineResponses.size())];

        int minSpaces = 2;
        int index = 0;
        String enSpace = "\u2002";

        for (LLCombineResponse llCombineResponse : llCombineResponses) {
            if (index >= max_rows_allowed) break;
//            Log.d(TAG, llCombineResponse.toString());
            if(llCombineResponse.inWord!=null && llCombineResponse.inWordTranslation!=null){
                llCombineResults[index] = llCombineResponse.inWord + enSpace.repeat(minSpaces) + "⟶" + enSpace.repeat(minSpaces) + llCombineResponse.inWordTranslation;
            }else if (llCombineResponse.inUpgrade != null && llCombineResponse.inUpgradeMeaning!= null) {
                llCombineResults[index] = "⬆ " + llCombineResponse.inUpgrade + enSpace.repeat(minSpaces) + "-" + enSpace.repeat(minSpaces) + llCombineResponse.inUpgradeMeaning;
            }
            index++;
        }

        return llCombineResults;
    }

    public String[] calculateAdhdStmbStringFormatted(LinkedList<STMBSummary> summaries) {
        int max_rows_allowed = 4;
        String[] stmbResults = new String[Math.min(max_rows_allowed, summaries.size())];

        int minSpaces = 2;
        int index = 0;
        for (STMBSummary summary : summaries) {
            if (index >= max_rows_allowed) break;
            stmbResults[index] = summary.summary;
            index++;
        }

        return stmbResults;
    }

//    public String[] calculateLLUpgradeResponseFormatted(LinkedList<LLUpgradeResponse> llUpgradeResponses) {
//        int max_rows_allowed = 1;
//        String[] llUpgradeResults = new String[Math.min(max_rows_allowed, llUpgradeResponses.size())];
//
//        int minSpaces = 0;
//        int index = 0;
//        for (LLUpgradeResponse llUpgradeResponse : llUpgradeResponses) {
//            if (index >= max_rows_allowed) break;
//            llUpgradeResults[index] = "Upgrade: " + llUpgradeResponse.inUpgrade + " ( " + llUpgradeResponse.inUpgradeMeaning + " ) ";
//            index++;
//        }
//
//        return llUpgradeResults;
//    }

    public String[] calculateLLContextConvoResponseFormatted(LinkedList<ContextConvoResponse> contextConvoResponses) {
        int max_rows_allowed;
        if (getIsLiveCaptionsChecked(this)) max_rows_allowed = 3; // Only show 2 rows if live captions are enabled
        else max_rows_allowed = 4;

        if (!clearedScreenYet) {
            sendHomeScreen();
            clearedScreenYet = true;
        }


        String[] llResults = new String[Math.min(max_rows_allowed, contextConvoResponses.size())];
//        String enSpace = "\u2002"; // Using en space for padding

//        int minSpaces = 2;
        int index = 0;
        for (ContextConvoResponse contextConvoResponse: contextConvoResponses) {
            if (index >= max_rows_allowed) break;
            llResults[index] = contextConvoResponse.response;
            index++;
        }

        return llResults;
    }

    public void sendTextWallLiveCaptionLL(final String newLiveCaption, final boolean isLiveCaptionFinal, final String llString) {
        final String separatorLine = "";

        if (!newLiveCaption.isEmpty()) {
            if (!oldLiveCaption.isEmpty()) {
                oldLiveCaptionFinal = oldLiveCaption;
                oldLiveCaption = "";
            }
            if (isLiveCaptionFinal) oldLiveCaption = currentLiveCaption; // Only update old caption if the new one is final
            currentLiveCaption = newLiveCaption;
        }
        if (!llString.isEmpty()) llCurrentString = llString;

        // Truncate to the last 70 characters
        oldLiveCaption = oldLiveCaption.length() > 70 ? oldLiveCaption.substring(oldLiveCaption.length() - 70) : oldLiveCaption;
        currentLiveCaption = currentLiveCaption.length() > 70 ? currentLiveCaption.substring(currentLiveCaption.length() - 70) : currentLiveCaption;

        final String topSeparatorLine = !llCurrentString.isEmpty() ? separatorLine : "";
        final String bottomSeparatorLine = !oldLiveCaptionFinal.isEmpty() ? separatorLine + "\n" : "";

        String textBubble = "\uD83D\uDDE8";
        String preOldCaptionTextBubble;
        if (oldLiveCaptionFinal.equals("")){
            preOldCaptionTextBubble = "";
        } else {
            preOldCaptionTextBubble = textBubble;
        }
//        if (!clearedScreenYet) {
//            sendHomeScreen();
//            clearedScreenYet = true;
//        }

        displayQueue.addTask(new DisplayQueue.Task(() -> sendDoubleTextWall(llCurrentString + topSeparatorLine, preOldCaptionTextBubble + oldLiveCaptionFinal + bottomSeparatorLine + textBubble + currentLiveCaption), false, true));
    }

    public void parseConvoscopeResults(JSONObject response) throws JSONException {
//        Log.d(TAG, "GOT CSE RESULT: " + response.toString());
        String imgKey = "image_url";
        String mapImgKey = "map_image_path";

        boolean isLiveCaptionsChecked = getIsLiveCaptionsChecked(this);

        //explicit queries
        JSONArray explicitAgentQueries = response.has(explicitAgentQueriesKey) ? response.getJSONArray(explicitAgentQueriesKey) : new JSONArray();

        JSONArray explicitAgentResults = response.has(explicitAgentResultsKey) ? response.getJSONArray(explicitAgentResultsKey) : new JSONArray();

        //displayResults
        JSONArray displayRequests = response.has(displayRequestsKey) ? response.getJSONArray(displayRequestsKey) : new JSONArray();

        //proactive agents
        JSONArray proactiveAgentResults = response.has(proactiveAgentResultsKey) ? response.getJSONArray(proactiveAgentResultsKey) : new JSONArray();
        JSONArray entityDefinitions = response.has(entityDefinitionsKey) ? response.getJSONArray(entityDefinitionsKey) : new JSONArray();

        //adhd STMB results
        JSONArray adhdStmbResults = response.has(adhdStmbAgentKey) ? response.getJSONArray(adhdStmbAgentKey) : new JSONArray();
        if (adhdStmbResults.length() != 0) {
            Log.d(TAG, "ADHD RESULTS: ");
            Log.d(TAG, adhdStmbResults.toString());

            if (!clearedScreenYet) {
                sendHomeScreen();
                clearedScreenYet = true;
            }

            updateAdhdSummaries(adhdStmbResults);
            String dynamicSummary = adhdStmbResults.getJSONObject(0).getString("summary");
            String [] adhdResults = calculateAdhdStmbStringFormatted(getAdhdStmbSummaries());
            displayQueue.addTask(new DisplayQueue.Task(() -> this.sendRowsCard(adhdResults), false, true));
//            sendTextToSpeech("欢迎使用安卓文本到语音转换功能", "chinese");
//            Log.d(TAG, "GOT THAT ONEEEEEEEE:");
//            Log.d(TAG, String.join("\n", llResults));
//            sendUiUpdateSingle(String.join("\n", Arrays.copyOfRange(llResults, llResults.length, 0)));
//            List<String> list = Arrays.stream(Arrays.copyOfRange(llResults, 0, languageLearningResults.length())).filter(Objects::nonNull).collect(Collectors.toList());
//            Collections.reverse(list);
            //sendUiUpdateSingle(String.join("\n", list));
            sendUiUpdateSingle(dynamicSummary);
            responsesBuffer.add(dynamicSummary);
        }

//        JSONArray languageLearningResults = response.has(languageLearningKey) ? response.getJSONArray(languageLearningKey) : new JSONArray();
//        updateDefinedWords(languageLearningResults); //sliding buffer, time managed language learning card
//        String[] llResults;
//        if (languageLearningResults.length() != 0) {
//            if (!clearedScreenYet) {
//                sendHomeScreen();
//                clearedScreenYet = true;
//            }
//
//            llResults = calculateLLStringFormatted(getDefinedWords());
//            if (getConnectedDeviceModelOs() != SmartGlassesOperatingSystem.AUDIO_WEARABLE_GLASSES) {
////                sendRowsCard(llResults);
//                //pack it into a string since we're using text wall now
//                String textWallString = Arrays.stream(llResults)
//                        .reduce((a, b) -> b + "\n\n" + a)
//                        .orElse("");
//                sendTextWall(textWallString);
//            }

//            sendTextToSpeech("欢迎使用安卓文本到语音转换功能", "Chinese");
//            Log.d(TAG, "GOT THAT ONEEEEEEEE:");
//            Log.d(TAG, String.join("\n", llResults));
//            llResults = calculateLLStringFormatted(getDefinedWords());
//            String newLineSeparator = isLiveCaptionsChecked ? "\n" : "\n\n";
//
////            pack it into a string since we're using text wall now
//            String textWallString = Arrays.stream(llResults)
//                .reduce((a, b) -> b + newLineSeparator + a)
//                .orElse("");
//
//            if (getConnectedDeviceModelOs() != SmartGlassesOperatingSystem.AUDIO_WEARABLE_GLASSES) {
//                if (isLiveCaptionsChecked) sendTextWallLiveCaptionLL("", false, textWallString);
//                else sendTextWall(textWallString);
//            }

//            Log.d(TAG, textWallString);
//            sendUiUpdateSingle(String.join("\n", Arrays.copyOfRange(llResults, llResults.length, 0)));
//            List<String> list = Arrays.stream(Arrays.copyOfRange(llResults, 0, languageLearningResults.length())).filter(Objects::nonNull).collect(Collectors.toList());
//            Collections.reverse(list);
//            sendUiUpdateSingle(String.join("\n", llResults));
//            responsesBuffer.add(String.join("\n", llResults));
//        }


        JSONArray languageLearningResults = response.has(languageLearningKey) ? response.getJSONArray(languageLearningKey) : new JSONArray();
        JSONArray llWordSuggestUpgradeResults = response.has(llWordSuggestUpgradeKey) ? response.getJSONArray(llWordSuggestUpgradeKey) : new JSONArray();
        updateCombineResponse(languageLearningResults, llWordSuggestUpgradeResults);
//        Log.d(TAG, "ll results"+languageLearningResults.toString()+"\n"+"upgrade result:"+llWordSuggestUpgradeResults);
        if (languageLearningResults.length() != 0 || llWordSuggestUpgradeResults.length() != 0) {
//            if (!clearedScreenYet) {
//                sendHomeScreen();
//                clearedScreenYet = true;
//            }
            String [] llCombineResults = calculateLLCombineResponseFormatted(getLLCombineResponse());
            String newLineSeparator = isLiveCaptionsChecked ? "\n" : "\n\n";
            if (getConnectedDeviceModelOs() != SmartGlassesOperatingSystem.AUDIO_WEARABLE_GLASSES) {
                String textWallString = Arrays.stream(llCombineResults)
                        .reduce((a, b) -> b + newLineSeparator + a)
                        .orElse("");
                if (isLiveCaptionsChecked) sendTextWallLiveCaptionLL("", false, textWallString);
                else {
                    displayQueue.addTask(new DisplayQueue.Task(() -> this.sendTextWall(textWallString), false, true));
                }
            }
//            Log.d(TAG, "ll combine results"+ llCombineResults.toString());
            sendUiUpdateSingle(String.join("\n", llCombineResults));
            responsesBuffer.add(String.join("\n", llCombineResults));
        }


        JSONArray llContextConvoResults = response.has(llContextConvoKey) ? response.getJSONArray(llContextConvoKey) : new JSONArray();

        updateContextConvoResponses(llContextConvoResults); //sliding buffer, time managed context convo card
        String[] llContextConvoResponses;

        if (llContextConvoResults.length() != 0) {
            llContextConvoResponses = calculateLLContextConvoResponseFormatted(getContextConvoResponses());
            if (getConnectedDeviceModelOs() != SmartGlassesOperatingSystem.AUDIO_WEARABLE_GLASSES) {
                String textWallString = Arrays.stream(llContextConvoResponses)
                        .reduce((a, b) -> b + "\n\n" + a)
                        .orElse("");
                //sendRowsCard(llContextConvoResponses);

                if (isLiveCaptionsChecked) sendTextWallLiveCaptionLL("", false, textWallString);
                else {
                    displayQueue.addTask(new DisplayQueue.Task(() -> this.sendTextWall(textWallString), false, true));
                }
            }
            List<String> list = Arrays.stream(Arrays.copyOfRange(llContextConvoResponses, 0, llContextConvoResults.length())).filter(Objects::nonNull).collect(Collectors.toList());
            Collections.reverse(list);
            sendUiUpdateSingle(String.join("\n", list));
            responsesBuffer.add(String.join("\n", list));

            try {
                JSONObject llContextConvoResult = llContextConvoResults.getJSONObject(0);
//                Log.d(TAG, llContextConvoResult.toString());
                JSONObject toTTS = llContextConvoResult.getJSONObject("to_tts");
                String text = toTTS.getString("text");
                String language = toTTS.getString("language");
//                Log.d(TAG, "Text: " + text + ", Language: " + language);
                //sendTextToSpeech(text, language);
                displayQueue.addTask(new DisplayQueue.Task(() -> this.sendTextToSpeech(text, language), false, false));
            } catch (JSONException e) {
                e.printStackTrace();
            }

        }

        // displayResults
        for (int i = 0; i < displayRequests.length(); i++) {
            try {
                JSONObject obj = displayRequests.getJSONObject(i);
                JSONObject req = obj.getJSONObject("data");
                JSONObject content = req.getJSONObject("content");
                String layout = req.getString("layout");
                String title;
                String body;
                switch (layout){
                    case "REFERENCE_CARD":
                        title = content.getString("title");
                        body = content.getString("body");
                        queueOutput(title + ": " + body);
                        displayQueue.addTask(new DisplayQueue.Task(() -> this.sendReferenceCard(title, body), false, false));
                        break;
                    case "TEXT_WALL":
                    case "TEXT_LINE":
                        body = content.getString("body");
                        queueOutput(body);
                        displayQueue.addTask(new DisplayQueue.Task(() -> this.sendTextWall(body), false, false));
                        break;
                    case "DOUBLE_TEXT_WALL":
                        String bodyTop = content.getString("bodyTop");
                        String bodyBottom = content.getString("bodyBottom");
                        queueOutput(bodyTop + "\n\n" + bodyBottom);
                        displayQueue.addTask(new DisplayQueue.Task(() -> this.sendDoubleTextWall(bodyTop, bodyBottom), false, false));
                        break;
                    case "ROWS_CARD":
                        JSONArray rowsArray = content.getJSONArray("rows");
                        String[] stringsArray = new String[rowsArray.length()];
                        for (int k = 0; k < rowsArray.length(); k++)
                            stringsArray[k] = rowsArray.getString(k);
                        queueOutput(String.join("\n", stringsArray));
                        displayQueue.addTask(new DisplayQueue.Task(() -> this.sendRowsCard(stringsArray), false, false));
                        break;
                    default:
                        Log.d(TAG, "SOME ISSUE");
                }
            }
            catch (JSONException e){
                e.printStackTrace();
            }
        }

        // entityDefinitions
        for (int i = 0; i < entityDefinitions.length(); i++) {
            try {
                JSONObject obj = entityDefinitions.getJSONObject(i);
                String name = obj.getString("name");
                String body = obj.getString("summary");
                displayQueue.addTask(new DisplayQueue.Task(() -> this.sendReferenceCard("" + name + "", body), false, false));
                queueOutput(name + ": " + body);
            } catch (JSONException e){
                e.printStackTrace();
            }
        }

        long wakeWordTime = response.has(wakeWordTimeKey) ? response.getLong(wakeWordTimeKey) : -1;

        // Wake word indicator
        if (wakeWordTime != -1 && wakeWordTime != previousWakeWordTime){
            previousWakeWordTime = wakeWordTime;
            String body = "Listening... ";
            displayQueue.addTask(new DisplayQueue.Task(() -> this.sendReferenceCard(glassesCardTitle, body), true, true));
            queueOutput(body);
        }

        //go through explicit agent queries and add to resultsToDisplayList
        // "Processing query: " indicator
        for (int i = 0; i < explicitAgentQueries.length(); i++){
            try {
                JSONObject obj = explicitAgentQueries.getJSONObject(i);
                String title = "Processing Query";
                String body = "\"" + obj.getString("query") + "\"";
                displayQueue.addTask(new DisplayQueue.Task(() -> this.sendReferenceCard(title, body), true, true));
                queueOutput(body);
            } catch (JSONException e){
                e.printStackTrace();
            }
        }

        //go through explicit agent results and add to resultsToDisplayList
        // Show Wake Word Query
        for (int i = 0; i < explicitAgentResults.length(); i++){
            try {
                JSONObject obj = explicitAgentResults.getJSONObject(i);
                //String body = "Response: " + obj.getString("insight");
                String body = obj.getString("insight");
                displayQueue.addTask(new DisplayQueue.Task(() -> this.sendReferenceCard(glassesCardTitle, body), true, false));
                queueOutput(body);
            } catch (JSONException e){
                e.printStackTrace();
            }
        }

        //go through proactive agent results and add to resultsToDisplayList
        for (int i = 0; i < proactiveAgentResults.length(); i++){
            try {
                JSONObject obj = proactiveAgentResults.getJSONObject(i);
                String name = obj.getString("agent_name") + " says";
                String body = obj.getString("agent_insight");
                displayQueue.addTask(new DisplayQueue.Task(() -> this.sendReferenceCard(name, body), false, false));
                queueOutput(name + ": " + body);
            } catch (JSONException e){
                e.printStackTrace();
            }
        }

        //see if we should update user settings
        boolean shouldUpdateSettingsResult = response.has(shouldUpdateSettingsKey) && response.getBoolean(shouldUpdateSettingsKey);
        if (shouldUpdateSettingsResult){
            Log.d(TAG, "Runnign get settings because shouldUpdateSettings true");
            getSettings();
        }
    }

    public void parseLocationResults(JSONObject response) throws JSONException {
        Log.d(TAG, "GOT LOCATION RESULT: " + response.toString());
        // ll context convo
    }

    // Display things to the phone screen
    public void queueOutput(String item){
        responsesBuffer.add(item);
        sendUiUpdateSingle(item);
    }

    public void speakTTS(String toSpeak){
        sendTextLine(toSpeak);
    }

    public void sendUiUpdateFull(){
        Intent intent = new Intent();
        intent.setAction(ConvoscopeUi.UI_UPDATE_FULL);
        intent.putStringArrayListExtra(ConvoscopeUi.CONVOSCOPE_MESSAGE_STRING, responsesBuffer);
        intent.putStringArrayListExtra(ConvoscopeUi.TRANSCRIPTS_MESSAGE_STRING, transcriptsBuffer);
        sendBroadcast(intent);
    }

    public void sendUiUpdateSingle(String message) {
        Intent intent = new Intent();
        intent.setAction(ConvoscopeUi.UI_UPDATE_SINGLE);
        intent.putExtra(ConvoscopeUi.CONVOSCOPE_MESSAGE_STRING, message);
        sendBroadcast(intent);
    }

    public void sendFinalTranscriptToActivity(String transcript){
        Intent intent = new Intent();
        intent.setAction(ConvoscopeUi.UI_UPDATE_FINAL_TRANSCRIPT);
        intent.putExtra(ConvoscopeUi.FINAL_TRANSCRIPT, transcript);
        sendBroadcast(intent);
    }

    public void buttonDownEvent(int buttonNumber, boolean downUp){ //downUp if true if down, false if up
        if (!downUp){
            return;
        }

        try{
            JSONObject jsonQuery = new JSONObject();
            jsonQuery.put("button_num", buttonNumber);
            jsonQuery.put("button_activity", downUp);
            jsonQuery.put("timestamp", System.currentTimeMillis() / 1000);
            backendServerComms.restRequest(BUTTON_EVENT_ENDPOINT, jsonQuery, new VolleyJsonCallback(){
                @Override
                public void onSuccess(JSONObject result){
                    try {
                        Log.d(TAG, "GOT BUTTON RESULT: " + result.toString());
                        String query_answer = result.getString("message");
                        sendUiUpdateSingle(query_answer);
                        speakTTS(query_answer);
                    } catch (JSONException e) {
                        throw new RuntimeException(e);
                    }
                }
                @Override
                public void onFailure(int code){
                    Log.d(TAG, "SOME FAILURE HAPPENED (buttonDownEvent)");
                }

            });
        } catch (JSONException e){
            e.printStackTrace();
        }
    }

    public void setupAuthTokenMonitor(){
        idTokenListener = new FirebaseAuth.IdTokenListener() {
            @Override
            public void onIdTokenChanged(@NonNull FirebaseAuth firebaseAuth) {
                FirebaseUser user = firebaseAuth.getCurrentUser();
                if (user != null) {
                    user.getIdToken(true).addOnCompleteListener(new OnCompleteListener<GetTokenResult>() {
                        @Override
                        public void onComplete(@NonNull Task<GetTokenResult> task) {
                            if (task.isSuccessful()) {
                                String idToken = task.getResult().getToken();
                                Log.d(TAG, "GOT ONIDTOKENCHANGED Auth Token: " + idToken);
                                authToken = idToken;
                                PreferenceManager.getDefaultSharedPreferences(getApplicationContext())
                                        .edit()
                                        .putString("auth_token", idToken)
                                        .apply();
                                EventBus.getDefault().post(new GoogleAuthSucceedEvent());
                            } else {
                                Log.d(TAG, "Task failure in setAuthToken");
                                EventBus.getDefault().post(new GoogleAuthFailedEvent("#1 ERROR IN (setupAuthTokenMonitor)"));
                            }
                        }
                    });
                }
            }
        };
    }

    public void manualSetAuthToken() {
        Log.d(TAG, "GETTING AUTH TOKEN");
        FirebaseUser user = FirebaseAuth.getInstance().getCurrentUser();
        if (user != null) {
            user.getIdToken(true)
                    .addOnCompleteListener(new OnCompleteListener<GetTokenResult>() {
                        public void onComplete(@NonNull Task<GetTokenResult> task) {
                            if (task.isSuccessful()) {
                                String idToken = task.getResult().getToken();
                                Log.d(TAG, "GOT dat MANUAL Auth Token: " + idToken);
                                authToken = idToken;
                                PreferenceManager.getDefaultSharedPreferences(getApplicationContext())
                                        .edit()
                                        .putString("auth_token", idToken)
                                        .apply();
                                EventBus.getDefault().post(new GoogleAuthSucceedEvent());
                            } else {
                                Log.d(TAG, "Task failure in setAuthToken");
                                EventBus.getDefault().post(new GoogleAuthFailedEvent("#1 ERROR IN (SETAUTHTOKEN)"));
                            }
                        }
                    });
        } else {
            // not logged in, must log in
            Log.d(TAG, "User is null in setAuthToken");
            EventBus.getDefault().post(new GoogleAuthFailedEvent("#2 ERROR IN (SETAUTHTOKEN) (USER IS NULL)"));
        }
    }

    public static void saveChosenTargetLanguage(Context context, String targetLanguageString) {
        Log.d("CONVOSCOPE", "SAVING TARGET LANGUAGE: " + targetLanguageString);
        PreferenceManager.getDefaultSharedPreferences(context)
                .edit()
                .putString(context.getResources().getString(R.string.SHARED_PREF_TARGET_LANGUAGE), targetLanguageString)
                .apply();
    }

//    public Boolean isVocabularyUpgradeEnabled(Context context) {
//        return PreferenceManager.getDefaultSharedPreferences(context)
//                .getBoolean(context.getResources().getString(R.string.SHARED_PREF_VOCABULARY_UPGRADE), false);
//    }

    public Boolean isVocabularyUpgradeEnabled(Context context) {
        return PreferenceManager.getDefaultSharedPreferences(context)
                .getBoolean(context.getResources().getString(R.string.SHARED_PREF_VOCABULARY_UPGRADE), true);
    }

    public void setVocabularyUpgradeEnabled(Context context, boolean isEnabled) {
        PreferenceManager.getDefaultSharedPreferences(context)
                .edit()
                .putBoolean(context.getResources().getString(R.string.SHARED_PREF_VOCABULARY_UPGRADE), isEnabled)
                .apply();
    }

    public static void saveChosenSourceLanguage(Context context, String sourceLanguageString) {
        PreferenceManager.getDefaultSharedPreferences(context)
                .edit()
                .putString(context.getResources().getString(R.string.SHARED_PREF_SOURCE_LANGUAGE), sourceLanguageString)
                .apply();
    }

    public static void saveIsLiveCaptionsChecked(Context context, boolean isLiveCaptionsChecked) {
        PreferenceManager.getDefaultSharedPreferences(context)
                .edit()
                .putBoolean(context.getResources().getString(R.string.SHARED_PREF_LIVE_CAPTIONS), isLiveCaptionsChecked)
                .apply();
    }


    public static String getChosenTargetLanguage(Context context) {
        String targetLanguageString = PreferenceManager.getDefaultSharedPreferences(context).getString(context.getResources().getString(R.string.SHARED_PREF_TARGET_LANGUAGE), "");
        if (targetLanguageString.equals("")){
            saveChosenTargetLanguage(context, "Russian");
            targetLanguageString = "Russian";
        }
        return targetLanguageString;
    }

    public static String getChosenSourceLanguage(Context context) {
        String sourceLanguageString = PreferenceManager.getDefaultSharedPreferences(context).getString(context.getResources().getString(R.string.SHARED_PREF_SOURCE_LANGUAGE), "");
        if (sourceLanguageString.equals("")){
            saveChosenSourceLanguage(context, "English");
            sourceLanguageString = "English";
        }
        return sourceLanguageString;
    }

    public static boolean getIsLiveCaptionsChecked(Context context) {
        return PreferenceManager.getDefaultSharedPreferences(context).getBoolean(context.getResources().getString(R.string.SHARED_PREF_LIVE_CAPTIONS), false);
    }

//    public void changeMode(String currentModeString){
//        if (currentModeString.equals("Proactive Agents")){
//            features = new String[]{explicitAgent, proactiveAgents, definerAgent};
//        } else if (currentModeString.equals("Language Learning")){
//            features = new String[]{explicitAgent, languageLearningAgent, llContextConvoAgent};
//        } else if (currentModeString.equals("ADHD Glasses")){
//            Log.d(TAG, "Settings features for ADHD Glasses");
//            features = new String[]{explicitAgent, adhdStmbAgent};
//        }
//    }

    public void saveCurrentMode(Context context, String currentModeString) {
        sendHomeScreen();


        //save the new mode
        PreferenceManager.getDefaultSharedPreferences(context)
                .edit()
                .putString(context.getResources().getString(R.string.SHARED_PREF_CURRENT_MODE), currentModeString)
                .apply();

        try{
            JSONObject settingsObj = new JSONObject();
            settingsObj.put("current_mode", currentModeString);
            sendSettings(settingsObj);
        } catch (JSONException e){
            e.printStackTrace();
        }
    }

    public String getCurrentMode(Context context) {
        String currentModeString = PreferenceManager.getDefaultSharedPreferences(context).getString(context.getResources().getString(R.string.SHARED_PREF_CURRENT_MODE), "");
        if (currentModeString.equals("")){
            currentModeString = "Language Learning";
            saveCurrentMode(context, currentModeString);
        }
        return currentModeString;
    }

    public void updateVocabularyUpgradeOnBackend(Context context){
        Boolean upgradeEnabled = isVocabularyUpgradeEnabled(context);
        try{
            JSONObject settingsObj = new JSONObject();
            settingsObj.put("vocabulary_upgrade_enabled", upgradeEnabled);
            sendSettings(settingsObj);
        } catch (JSONException e){
            e.printStackTrace();
        }
    }
    public void updateTargetLanguageOnBackend(Context context){
        String targetLanguage = getChosenTargetLanguage(context);
        try{
            JSONObject settingsObj = new JSONObject();
            settingsObj.put("target_language", targetLanguage);
            sendSettings(settingsObj);
        } catch (JSONException e){
            e.printStackTrace();
        }
    }


    public void updateSourceLanguageOnBackend(Context context){
        String sourceLanguage = getChosenSourceLanguage(context);
        try{
            JSONObject settingsObj = new JSONObject();
            settingsObj.put("source_language", sourceLanguage);
            sendSettings(settingsObj);
        } catch (JSONException e){
            e.printStackTrace();
        }
    }

    @Subscribe
    public void onGoogleAuthSucceed(GoogleAuthSucceedEvent event){
        Log.d(TAG, "Running google auth succeed event response");
        //give the server our latest settings
        //updateTargetLanguageOnBackend(this);
        //updateSourceLanguageOnBackend(this);
        //saveCurrentMode(this, getCurrentMode(this));
    }


    //language learning
    public void updateDefinedWords(JSONArray newData) {
        long currentTime = System.currentTimeMillis();

        // Add new data to the list
        for (int i = 0; i < newData.length(); i++) {
            try {
                JSONObject wordData = newData.getJSONObject(i);
                definedWords.addFirst(new DefinedWord(
                        wordData.getString("in_word"),
                        wordData.getString("in_word_translation"),
                        wordData.getLong("timestamp"),
                        wordData.getString("uuid")
                ));
            } catch (JSONException e){
                e.printStackTrace();
            }
        }

        // Remove old words based on time constraint
        definedWords.removeIf(word -> (currentTime - (word.timestamp * 1000)) > llDefinedWordsShowTime);

        // Ensure list does not exceed max size
        while (definedWords.size() > maxDefinedWordsShow) {
            definedWords.removeLast();
        }
    }

    public void updateCombineResponse(JSONArray llData, JSONArray ugData) {
        long currentTime = System.currentTimeMillis();
        // Add new data to the list
        for (int i = 0; i < llData.length(); i++) {
            try {
                JSONObject wordData = llData.getJSONObject(i);
                llCombineResponses.addFirst(new LLCombineResponse(
                        null,
                        null,
                        wordData.getString("in_word"),
                        wordData.getString("in_word_translation"),
                        wordData.getLong("timestamp"),
                        wordData.getString("uuid")
                ));
            } catch (JSONException e){
                e.printStackTrace();
            }
        }

        for (int i = 0; i < ugData.length(); i++) {
            try {
                JSONObject resData = ugData.getJSONObject(i);
                llCombineResponses.addFirst(new LLCombineResponse(
                        resData.getString("in_upgrade"),
                        resData.getString("in_upgrade_meaning"),
                        null,
                        null,
                        resData.getLong("timestamp"),
                        resData.getString("uuid")
                ));
            } catch (JSONException e){
                e.printStackTrace();
            }
        }

        // Remove old words based on time constraint
        llCombineResponses.removeIf(word -> (currentTime - (word.timestamp * 1000)) > llCombineShowTime);

        // Ensure list does not exceed max size
        while (llCombineResponses.size() > maxLLCombineShow) {
            llCombineResponses.removeLast();
        }
    }

    public void updateAdhdSummaries(JSONArray newData) {
        long currentTime = System.currentTimeMillis();
        boolean foundNewFalseShift = false;
        STMBSummary newFalseShiftSummary = null;

        // First, identify if there's a new summary with true_shift = false and prepare it
        for (int i = 0; i < newData.length(); i++) {
            try {
                JSONObject summaryData = newData.getJSONObject(i);
                if (!summaryData.getBoolean("true_shift")) {
                    foundNewFalseShift = true;
                    newFalseShiftSummary = new STMBSummary(
                            summaryData.getString("summary"),
                            summaryData.getLong("timestamp"),
                            false,
                            summaryData.getString("uuid"));
                    break; // Stop after finding the first false shift as only one should exist
                }
            } catch (JSONException e) {
                e.printStackTrace();
            }
        }

        // If a new false shift summary exists, remove the old false shift summary
        if (foundNewFalseShift) {
            adhdStmbSummaries.removeIf(summary -> !summary.true_shift);
        }

        // Now, add new data while excluding the newly identified false shift summary
        for (int i = 0; i < newData.length(); i++) {
            try {
                JSONObject summaryData = newData.getJSONObject(i);
                if (summaryData.getBoolean("true_shift") || !foundNewFalseShift || !summaryData.getString("uuid").equals(newFalseShiftSummary.uuid)) {
                    adhdStmbSummaries.addFirst(new STMBSummary(
                            summaryData.getString("summary"),
                            summaryData.getLong("timestamp"),
                            summaryData.getBoolean("true_shift"),
                            summaryData.getString("uuid")
                    ));
                }
            } catch (JSONException e) {
                e.printStackTrace();
            }
        }

        // Add the new false shift summary at the beginning if it exists
        if (newFalseShiftSummary != null) {
            adhdStmbSummaries.addFirst(newFalseShiftSummary);
        }

        // Remove old words based on time constraint
        adhdStmbSummaries.removeIf(summary -> (currentTime - (summary.timestamp * 1000)) > adhdSummaryShowTime);

        // Ensure list does not exceed max size
        while (adhdStmbSummaries.size() > maxAdhdStmbShowNum) {
            adhdStmbSummaries.removeLast();
        }
    }

    // Getter for the list, if needed
    public LinkedList<DefinedWord> getDefinedWords() {
        return definedWords;
    }

    public LinkedList<STMBSummary> getAdhdStmbSummaries() {
        return adhdStmbSummaries;
    }

    public LinkedList<LLUpgradeResponse> getLLUpgradeResponse() {
        return llUpgradeResponses;
    }

    public LinkedList<LLCombineResponse> getLLCombineResponse() {
        return llCombineResponses;
    }

    // A simple representation of your word data
    private static class DefinedWord {
        String inWord;
        String inWordTranslation;
        long timestamp;
        String uuid;

        DefinedWord(String inWord, String inWordTranslation, long timestamp, String uuid) {
            this.inWord = inWord;
            this.inWordTranslation = inWordTranslation;
            this.timestamp = timestamp;
            this.uuid = uuid;
        }
    }

    // A simple representation of upgrade word data
    // private static class UpgradeWord {
    //     String inUpgrade;
    //     String inUpgradeMeaning;
    //     long timestamp;
    //     String uuid;

    //     UpgradeWord(String inUpgrade, String inUpgradeMeaning, long timestamp, String uuid) {
    //         this.inUpgrade = inUpgrade;
    //         this.inUpgradeMeaning = inUpgradeMeaning;
    //         this.timestamp = timestamp;
    //         this.uuid = uuid;
    //     }
    // }

    // A simple representation of ADHD STMB data
    private static class STMBSummary {
        String summary;
        long timestamp;
        boolean true_shift;
        String uuid;

        STMBSummary(String summary, long timestamp, boolean true_shift, String uuid) {
            this.summary = summary;
            this.timestamp = timestamp;
            this.true_shift = true_shift;
            this.uuid = uuid;
        }
    }


    //context convo
    public void updateContextConvoResponses(JSONArray newData) {
        long currentTime = System.currentTimeMillis();
//        Log.d(TAG, "GOT NEW DATA: ");
//        Log.d(TAG, newData.toString());

        // Add new data to the list
        for (int i = 0; i < newData.length(); i++) {
            try {
                JSONObject wordData = newData.getJSONObject(i);
                contextConvoResponses.addFirst(new ContextConvoResponse(
                        wordData.getString("ll_context_convo_response"),
                        wordData.getLong("timestamp"),
                        wordData.getString("uuid")
                ));
            } catch (JSONException e){
                e.printStackTrace();
            }
        }

        contextConvoResponses.removeIf(contextConvoResponse -> (currentTime - (contextConvoResponse.timestamp * 1000)) > llContextConvoResponsesShowTime);

        // Ensure list does not exceed max size
        while (contextConvoResponses.size() > maxContextConvoResponsesShow) {
            contextConvoResponses.removeLast();
        }
    }

    public void updateLLUpgradeResponse(JSONArray newData) {
        long currentTime = System.currentTimeMillis();
        // Add new data to the list
        for (int i = 0; i < newData.length(); i++) {
            try {
                JSONObject resData = newData.getJSONObject(i);
                llUpgradeResponses.addFirst(new LLUpgradeResponse(
                        resData.getString("in_upgrade"),
                        resData.getString("in_upgrade_meaning"),
                        resData.getLong("timestamp"),
                        resData.getString("uuid")
                ));
            } catch (JSONException e){
                e.printStackTrace();
            }
        }

        llUpgradeResponses.removeIf(llupgradeResponse -> (currentTime - (llupgradeResponse.timestamp * 1000)) > llUpgradeShowTime);

        // Ensure list does not exceed max size
        while (llUpgradeResponses.size() > maxLLUpgradeResponsesShow) {
            llUpgradeResponses.removeLast();
        }
    }



    // Getter for the list, if needed
    public LinkedList<ContextConvoResponse> getContextConvoResponses() {
        return contextConvoResponses;
    }

    // A simple representation of your word data
    private static class ContextConvoResponse {
        String response;
        long timestamp;
        String uuid;

        ContextConvoResponse(String response, long timestamp, String uuid) {
            this.response = response;
            this.timestamp = timestamp;
            this.uuid = uuid;
        }
    }

    private static class LLUpgradeResponse {
        String inUpgrade;
        String inUpgradeMeaning;
        long timestamp;
        String uuid;

        LLUpgradeResponse(String inUpgrade, String inUpgradeMeaning, long timestamp, String uuid) {
            this.inUpgrade = inUpgrade;
            this.inUpgradeMeaning = inUpgradeMeaning;
            this.timestamp = timestamp;
            this.uuid = uuid;
        }
    }

    // A simple representation of combination of ll rare and ll upgrade
    private static class LLCombineResponse {
        String inUpgrade;
        String inUpgradeMeaning;
        String inWord;
        String inWordTranslation;
        long timestamp;
        String uuid;

        LLCombineResponse(String inUpgrade, String inUpgradeMeaning,String inWord, String inWordTranslation, long timestamp, String uuid) {
            this.inUpgrade = inUpgrade;
            this.inUpgradeMeaning = inUpgradeMeaning;
            this.inWord = inWord;
            this.inWordTranslation = inWordTranslation;
            this.timestamp = timestamp;
            this.uuid = uuid;
        }
    }

    //retry auth right away if it failed, but don't do it too much as we have a max # refreshes/day
    private int max_google_retries = 3;
    private int googleAuthRetryCount = 0;
    private long lastGoogleAuthRetryTime = 0;

    @Subscribe
    public void onGoogleAuthFailedEvent(GoogleAuthFailedEvent event) {
        Log.d(TAG, "onGoogleAuthFailedEvent triggered");
        numConsecutiveAuthFailures += 1;
        if(numConsecutiveAuthFailures > 10) {
            Log.d("TAG", "ATTEMPT SIGN OUT");
            handleSignOut();
        }
    }

    // Used for notifications and for screen mirror
    @Subscribe
    public void onNewScreenTextEvent(NewScreenTextEvent event) {
        // Notification
        if (event.title != null && event.body != null) {
            displayQueue.addTask(new DisplayQueue.Task(() -> this.sendReferenceCard(event.title, event.body), false, false));
        }
        else if (event.body != null){ //Screen mirror text
            displayQueue.addTask(new DisplayQueue.Task(() -> this.sendTextWall(event.body), false, true));
        }
    }

    @Subscribe
    public void onNewScreenImageEvent(NewScreenImageEvent event) {
        displayQueue.addTask(new DisplayQueue.Task(() -> this.sendBitmap(event.bmp), false, true));
    }

    private void updateLastDataSentTime() {
        lastDataSentTime = System.currentTimeMillis();
    }

    private void startNotificationService() {
        Intent notificationServiceIntent = new Intent(this, MyNotificationListeners.class);
        startService(notificationServiceIntent);

        NotificationListenerService.requestRebind(
                new ComponentName(this, MyNotificationListeners.class));
    }

    private void stopNotificationService() {
        Intent notificationServiceIntent = new Intent(this, MyNotificationListeners.class);
        stopService(notificationServiceIntent);
    }

}
