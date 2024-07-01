package com.teamopensmartglasses.convoscope.ui;

import android.content.ComponentName;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.os.Bundle;
import android.text.Html;
import android.text.InputType;
import android.text.method.LinkMovementMethod;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.Switch;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.Fragment;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.preference.PreferenceManager;

import com.teamopensmartglasses.convoscope.ConvoscopeService;
import com.teamopensmartglasses.convoscope.MainActivity;
import com.teamopensmartglasses.convoscope.R;
import com.teamopensmartglasses.smartglassesmanager.speechrecognition.ASR_FRAMEWORKS;
import com.teamopensmartglasses.smartglassesmanager.supportedglasses.AudioWearable;

public class SettingsUi extends Fragment {
    private  final String TAG = "WearableAi_SettingsUIFragment";

    private final String fragmentLabel = "Settings";

    private NavController navController;

    //test card raw data
    public String testCardImg = "https://ichef.bbci.co.uk/news/976/cpsprodpb/7727/production/_103330503_musk3.jpg";
    public String testCardTitle = "Test Card";
    public String testCardContent = "This is an example of a reference card. This appears on your glasses after pressing the 'Send Test Card' button.";

    public SettingsUi() {
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.settings_fragment, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        UiUtils.setupTitle(getActivity(), fragmentLabel);

        navController = Navigation.findNavController(getActivity(), R.id.nav_host_fragment);
        Context mContext = this.getContext();

        final Button killServiceButton = view.findViewById(R.id.kill_wearableai_service);
        killServiceButton.setOnClickListener(new View.OnClickListener() {
                public void onClick(View v) {
                // Code here executes on main thread after user presses button
                ((MainActivity)getActivity()).stopConvoscopeService();
            }
        });

        final Button connectSmartGlassesButton = view.findViewById(R.id.connect_smart_glasses);
            connectSmartGlassesButton.setOnClickListener(new View.OnClickListener() {
                public void onClick(View v) {
                // Code here executes on main thread after user presses button
                //check to first make sure that user isn't trying to enable google without providing API key
                if (ConvoscopeService.getChosenAsrFramework(mContext) == ASR_FRAMEWORKS.GOOGLE_ASR_FRAMEWORK) {
                    String apiKey = ConvoscopeService.getApiKey(mContext);
                    if (apiKey == null || apiKey.equals("")) {
                        showNoGoogleAsrDialog();
                        return;
                    }
                }
                ((MainActivity)getActivity()).stopConvoscopeService();
                ((MainActivity)getActivity()).startConvoscopeService();

                //TODO: Temporarily disable this
                //navController.navigate(R.id.nav_select_smart_glasses);
            }
        });

        //final Button setGoogleApiKeyButton = view.findViewById(R.id.google_api_change);
        //final Switch switchGoogleAsr = view.findViewById(R.id.google_asr_switch);

        //find out the current ASR state, remember it
//        ConvoscopeService.saveChosenAsrFramework(mContext, ASR_FRAMEWORKS.GOOGLE_ASR_FRAMEWORK);
//        ConvoscopeService.saveChosenAsrFramework(mContext, ASR_FRAMEWORKS.DEEPGRAM_ASR_FRAMEWORK);
        ConvoscopeService.saveChosenAsrFramework(mContext, ASR_FRAMEWORKS.AZURE_ASR_FRAMEWORK);
        ASR_FRAMEWORKS asrFramework = ConvoscopeService.getChosenAsrFramework(mContext);
//        switchGoogleAsr.setChecked(asrFramework == ASR_FRAMEWORKS.GOOGLE_ASR_FRAMEWORK);
//
//        switchGoogleAsr.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
//            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
//                ((MainActivity)getActivity()).stopConvoscopeService();
//
//                //setGoogleApiKeyButton.setEnabled(isChecked);
//                //save explicitly as well as force change in case the service is down, we want this to be saved either way
//                if (isChecked) {
//                    ConvoscopeService.saveChosenAsrFramework(mContext, ASR_FRAMEWORKS.GOOGLE_ASR_FRAMEWORK);
//                } else {
//                    ConvoscopeService.saveChosenAsrFramework(mContext, ASR_FRAMEWORKS.VOSK_ASR_FRAMEWORK);
//                }
//            }
//        });

        final Switch glassesAudioToggle = view.findViewById(R.id.glasses_audio_toggle);
        glassesAudioToggle.setChecked(ConvoscopeService.getPreferredWearable(getContext()).equals(new AudioWearable().deviceModelName)); // off by default
        glassesAudioToggle.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean isChecked) {
                if (isChecked){
                    ConvoscopeService.savePreferredWearable(getContext(), new AudioWearable().deviceModelName);
                    ((MainActivity)getActivity()).restartConvoscopeService();
                }
                else {
                    ConvoscopeService.savePreferredWearable(getContext(), "");
                    ((MainActivity)getActivity()).restartConvoscopeService();
                }
            }
        });

        final Button logOutButton = view.findViewById(com.teamopensmartglasses.convoscope.R.id.log_out_button);
        logOutButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                ((MainActivity)getActivity()).signOut();
            }
        });

        //setup transcript language spinner
        Spinner transcribeLanguageSpinner = view.findViewById(R.id.transcribeLanguageSpinner);
        ArrayAdapter<CharSequence> transcribeAdapter = ArrayAdapter.createFromResource(mContext,
                R.array.language_options, android.R.layout.simple_spinner_item);
        transcribeAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        transcribeLanguageSpinner.setAdapter(transcribeAdapter);

        // Retrieve the saved transcribe language
        String savedTranscribeLanguage = ((MainActivity)getActivity()).mService.getChosenTranscribeLanguage(mContext);

        // Find the position of the saved language in the adapter
        int languageSpinnerPosition = transcribeAdapter.getPosition(savedTranscribeLanguage);

        // Set the Spinner to show the saved language
        transcribeLanguageSpinner.setSelection(languageSpinnerPosition);

        transcribeLanguageSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            boolean initTranscribeLanguageSetup = true;

            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                if (initTranscribeLanguageSetup){
                    initTranscribeLanguageSetup = false;
                    return;
                }
                String selectedLanguage = parent.getItemAtPosition(position).toString();
                // Save the selected language as the new transcribe language default
                Log.d(TAG, "TRANSCRIBE LANGUAGE SPINNER CHANGED");
                ((MainActivity)getActivity()).mService.saveChosenTranscribeLanguage(mContext, selectedLanguage);
                ((MainActivity)getActivity()).restartConvoscopeService();
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                // Another interface callback
            }
        });

        //setup target targetLanguage spinner
        Spinner targetLanguageSpinner = view.findViewById(R.id.targetLanguageSpinner);
        ArrayAdapter<CharSequence> tlAdapter = ArrayAdapter.createFromResource(mContext,
                R.array.language_options, android.R.layout.simple_spinner_item);
        tlAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        targetLanguageSpinner.setAdapter(tlAdapter);

        // Retrieve the saved target targetLanguage
        String savedTargetLanguage = ((MainActivity)getActivity()).mService.getChosenTargetLanguage(mContext);

        Log.d(TAG, "TARGET LANGUAGE IS: " + savedTargetLanguage);

        // Find the position of the saved targetLanguage in the adapter
        int targetLanguageSpinnerPosition = tlAdapter.getPosition(savedTargetLanguage);

        // Set the Spinner to show the saved targetLanguage
        targetLanguageSpinner.setSelection(targetLanguageSpinnerPosition);

        targetLanguageSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            boolean initTargetLanguageSetup = true;

            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                if (initTargetLanguageSetup){
                    initTargetLanguageSetup = false;
                    return;
                }
                String selectedLanguage = parent.getItemAtPosition(position).toString();
                // Save the selected targetLanguage as the new default
                Log.d(TAG, "TARGET LANGUAGE SPINNER CHANGED");
                ((MainActivity)getActivity()).mService.saveChosenTargetLanguage(mContext, selectedLanguage);
                if (((MainActivity)getActivity()).mService != null) {
                    ((MainActivity) getActivity()).mService.updateTargetLanguageOnBackend(mContext);
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                // Another interface callback
            }
        });

        //setup source sourceLanguage spinner
        Spinner sourceLanguageSpinner = view.findViewById(R.id.sourceLanguageSpinner);
        ArrayAdapter<CharSequence> slAdapter = ArrayAdapter.createFromResource(mContext,
                R.array.language_options, android.R.layout.simple_spinner_item);
        slAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        sourceLanguageSpinner.setAdapter(slAdapter);

        // Retrieve the saved source sourceLanguage
        String savedSourceLanguage = ((MainActivity)getActivity()).mService.getChosenSourceLanguage(mContext);

        // Find the position of the saved sourceLanguage in the adapter
        int sourceLanguageSpinnerPosition = slAdapter.getPosition(savedSourceLanguage);

        // Set the Spinner to show the saved sourceLanguage
        sourceLanguageSpinner.setSelection(sourceLanguageSpinnerPosition);

        sourceLanguageSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            boolean initSourceLanguageSetup = true;

            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                if (initSourceLanguageSetup){
                    initSourceLanguageSetup = false;
                    return;
                }
                String selectedLanguage = parent.getItemAtPosition(position).toString();
                // Save the selected targetLanguage as the new default
                Log.d(TAG, "SOURCE LANGUAGE SPINNER CHANGED");
                ((MainActivity)getActivity()).mService.saveChosenSourceLanguage(mContext, selectedLanguage);
                if (((MainActivity)getActivity()).mService != null) {
                    ((MainActivity)getActivity()).mService.updateSourceLanguageOnBackend(mContext);
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                // Another interface callback
            }
        });
    }

    public void sendTestCard(){
        Log.d(TAG, "SENDING TEST CARD");
        ((MainActivity)getActivity()).mService.sendTestCard(testCardTitle, testCardContent, testCardImg);
    }

    //open hotspot settings
    private void launchHotspotSettings(){
        final Intent intent = new Intent(Intent.ACTION_MAIN, null);
        intent.addCategory(Intent.CATEGORY_LAUNCHER);
        final ComponentName cn = new ComponentName("com.android.settings", "com.android.settings.TetherSettings");
        intent.setComponent(cn);
        intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        startActivity( intent);
    }

    public void showNoGoogleAsrDialog(){
        new android.app.AlertDialog.Builder(this.getContext()) .setIcon(android.R.drawable.ic_dialog_alert)
                .setTitle("No Google API Key Provided")
                .setMessage("You have Google ASR enabled without an API key. Please turn off Google ASR or enter a valid API key.")
                .setPositiveButton("Ok", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                    }
                }).show();
    }

    @Override
    public void onResume() {
        super.onResume();
        UiUtils.setupTitle(getActivity(), fragmentLabel);
        // Show the back button when the fragment is visible
        ((AppCompatActivity) getActivity()).getSupportActionBar().setDisplayHomeAsUpEnabled(true);
    }

    @Override
    public void onPause() {
        super.onPause();
        // Hide the back button when the fragment is no longer visible
        ((AppCompatActivity) getActivity()).getSupportActionBar().setDisplayHomeAsUpEnabled(false);
    }
}
