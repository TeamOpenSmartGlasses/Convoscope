package com.teamopensmartglasses.convoscope.ui;

import android.app.AlertDialog;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Switch;
import android.widget.Toast;

import androidx.fragment.app.Fragment;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.preference.PreferenceManager;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.teamopensmartglasses.convoscope.MainActivity;
import com.teamopensmartglasses.convoscope.R;
import com.teamopensmartglasses.convoscope.ResponseTextUiAdapter;
import com.teamopensmartglasses.convoscope.TranscriptTextUiAdapter;
import com.teamopensmartglasses.convoscope.events.GoogleAuthFailedEvent;
import com.teamopensmartglasses.convoscope.events.UserIdChangedEvent;

import org.greenrobot.eventbus.EventBus;
import org.greenrobot.eventbus.Subscribe;

import java.util.ArrayList;

public class ConvoscopeUi extends Fragment {
    public String TAG = "WearableAi_ConvoscopeUi";

     //UI
  private ResponseTextUiAdapter responseTextUiAdapter;
  private RecyclerView responseRecyclerView;
  private TranscriptTextUiAdapter transcriptTextUiAdapter;
  private RecyclerView transcriptRecyclerView;
  public static final String UI_UPDATE_FULL = "UI_UPDATE_FULL";
  public static final String UI_UPDATE_SINGLE = "UI_UPDATE_SINGLE";
  public static final String UI_UPDATE_FINAL_TRANSCRIPT = "UI_UPDATE_FINAL_TRANSCRIPT";
  public static final String CONVOSCOPE_MESSAGE_STRING = "CONVOSCOPE_MESSAGE_STRING";
  public static final String TRANSCRIPTS_MESSAGE_STRING = "TRANSCRIPTS_MESSAGE_STRING";
  public static final String FINAL_TRANSCRIPT = "FINAL_TRANSCRIPT";
  private static final String fragmentLabel = "Convoscope";

    private NavController navController;


    public ConvoscopeUi() {
        // Required empty public constructor
    }

  private final BroadcastReceiver mMainServiceReceiver = new BroadcastReceiver() {
    @Override
    public void onReceive(Context context, Intent intent) {
      final String action = intent.getAction();
//      Log.d(TAG, "ConvoscopeUi got something");
      if (UI_UPDATE_SINGLE.equals(action)) {
        String message = intent.getStringExtra(CONVOSCOPE_MESSAGE_STRING);
        if (!message.equals("") && !message.equals(null)) {
//          Log.d(TAG, "Got message: " + message);
          addResponseTextBox(message);
        }
      } else if (UI_UPDATE_FULL.equals(action)){
        responseTextUiAdapter.clearTexts();
        transcriptTextUiAdapter.clearTexts();
        ArrayList<String> responsesBuffer = intent.getStringArrayListExtra(CONVOSCOPE_MESSAGE_STRING);
        ArrayList<String> transcrriptsBuffer = intent.getStringArrayListExtra(TRANSCRIPTS_MESSAGE_STRING);
        for (String message : responsesBuffer) {
          addResponseTextBox(message, false);
        }
        responseRecyclerView.scrollToPosition(responseTextUiAdapter.getItemCount() - 1);
        for (String transcript : transcrriptsBuffer) {
          addTranscriptTextBox(transcript);
        }
        transcriptRecyclerView.scrollToPosition(transcriptTextUiAdapter.getItemCount() - 1);
      } else if (UI_UPDATE_FINAL_TRANSCRIPT.equals(action)){
        String transcript = intent.getStringExtra(FINAL_TRANSCRIPT);
        addTranscriptTextBox(transcript);
      }
    }
  };


  @Override
  public void onResume() {
    super.onResume();

    //unregister receiver
    getActivity().registerReceiver(mMainServiceReceiver, makeMainServiceReceiverIntentFilter());

    //trigger UI update
    if (((MainActivity)getActivity()).mService != null){
       ((MainActivity)getActivity()).mService.sendUiUpdateFull();
    }
  }

  @Override
  public void onPause() {
    super.onPause();

    //unregister receiver
    getActivity().unregisterReceiver(mMainServiceReceiver);
  }

  //UI
  private static IntentFilter makeMainServiceReceiverIntentFilter() {
    final IntentFilter intentFilter = new IntentFilter();
    intentFilter.addAction(UI_UPDATE_FULL);
    intentFilter.addAction(UI_UPDATE_SINGLE);
    intentFilter.addAction(UI_UPDATE_FINAL_TRANSCRIPT);

    return intentFilter;
  }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // EventBus.getDefault().register(this);
    }

    @Override
    public void onDestroy() {
      super.onDestroy();
      // EventBus.getDefault().unregister(this);
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {

        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.convoscope_fragment, container, false);
    }

    @Override
    public void onViewCreated(View view, Bundle savedInstanceState){
        //setup titlebar
        UiUtils.setupTitle(getActivity(), fragmentLabel);

        //get the nav controller
        navController = Navigation.findNavController(getActivity(), R.id.nav_host_fragment);

        //setup UI
        responseRecyclerView = view.findViewById(R.id.recyclerView);
        responseRecyclerView.setLayoutManager(new LinearLayoutManager(this.getContext()));
        responseTextUiAdapter = new ResponseTextUiAdapter(new ArrayList<>());
        responseRecyclerView.setAdapter(responseTextUiAdapter);
        transcriptRecyclerView = view.findViewById(R.id.rawTranscriptsRecyclerView);
        transcriptRecyclerView.setLayoutManager(new LinearLayoutManager(this.getContext()));
        transcriptTextUiAdapter = new TranscriptTextUiAdapter(new ArrayList<>());
        transcriptRecyclerView.setAdapter(transcriptTextUiAdapter);

        final Button settingsButton = view.findViewById(R.id.settings_button);
        settingsButton.setOnClickListener(v -> openSettingsFragment());

//        Button pickContactButton = view.findViewById(R.id.pick_contact_button);
//        pickContactButton.setOnClickListener(new View.OnClickListener() {
//          @Override
//          public void onClick(View v) {
//            if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.READ_CONTACTS)
//                    != PackageManager.PERMISSION_GRANTED) {
//              ActivityCompat.requestPermissions(MainActivity.this,
//                      new String[]{Manifest.permission.READ_CONTACTS},
//                      READ_CONTACTS_PERMISSIONS_REQUEST);
//            } else {
//              pickContact();
//            }
//          }
//        });

        //scroll to bottom of scrolling UIs
        responseRecyclerView.scrollToPosition(responseTextUiAdapter.getItemCount() - 1);
        transcriptRecyclerView.scrollToPosition(transcriptTextUiAdapter.getItemCount() - 1);

//        //setup list of smart glasses
//        glassesList = view.findViewById(R.id.smart_glasses_list);
//
//        // on the below line we are initializing the adapter for our list view.
//        ArrayList<SmartGlassesDevice> glassesArrayList = new ArrayList<>();
//
//        //ArrayAdapter<String> glassesListAdapter = new ArrayAdapter<>(view.getContext(), android.R.layout.simple_list_item_activated_1, glassesArrayList);
//        SmartGlassesListAdapter glassesListAdapter = new SmartGlassesListAdapter(glassesArrayList, getContext());
//
//        //listen for list presses
//        glassesList.setOnItemClickListener(new AdapterView.OnItemClickListener()
//        {
//            @Override
//            public void onItemClick(AdapterView<?> adapter, View v, int position,
//                                    long arg3)
//            {
//                glassesListAdapter.setSelectedPosition(position);
//            }
//        });
//
//        // on below line we are setting adapter for our list view.
//        glassesList.setAdapter(glassesListAdapter);
//        for (SmartGlassesDevice device : smartGlassesDevices){
//            Log.d(TAG, device.getDeviceModelName());
//            glassesListAdapter.add(device);
//        }
//
//        //setup buttons
//        final Button cancelButton = view.findViewById(R.id.cancel_select_smart_glasses_button);
//        cancelButton.setOnClickListener(new View.OnClickListener() {
//            public void onClick(View v) {
//                // Code here executes on main thread after user presses button
//                navController.navigate(R.id.nav_settings);
//                //((MainActivity)getActivity()).onBackPressed();
//            }
//        });
//
//        final Button selectButton = view.findViewById(R.id.select_smart_glasses_button);
//        selectButton.setOnClickListener(new View.OnClickListener() {
//            public void onClick(View v) {
//                // Code here executes on main thread after user presses button
//                SmartGlassesDevice selectedDevice = glassesListAdapter.getSelectedDevice();
//                if (selectedDevice == null) {
//                    Log.d(TAG, "Please choose a smart glasses device to continue.");
//                    showNoGlassSelectedDialog();
//                } else if (!selectedDevice.getAnySupport()){
//                    Log.d(TAG, "Glasses not yet supported, we're working on it.");
//                    showUnsupportedGlassSelected();
//                } else {
//                    Log.d(TAG, "Connecting to " + selectedDevice.getDeviceModelName() + "...");
//                    ((MainActivity)getActivity()).connectSmartGlasses(selectedDevice);
//
//                    navController.navigate(R.id.nav_connecting_to_smart_glasses);
//                    //((MainActivity)getActivity()).onBackPressed();
//                }
//            }
//        });

      //setup mode switcher
      RadioGroup convoscopeModeSelector = view.findViewById(R.id.radioGroupOptions);

      String currentModeString = ((MainActivity)getActivity()).getCurrentMode(this.getContext());

      RadioButton radioButtonProactiveAgents = view.findViewById(R.id.radioButtonProactiveAgents);
      RadioButton radioButtonLanguageLearning = view.findViewById(R.id.radioButtonLanguageLearning);
      RadioButton radioButtonADHDGlasses = view.findViewById(R.id.radioButtonADHDGlasses);
      RadioButton radioButtonWalkNGrok = view.findViewById(R.id.radioButtonWalkNGrok);
      RadioButton radioButtonScreenMirror = view.findViewById(R.id.radioButtonScreenMirror);

      // Set the radio button as active based on the saved string
      if (currentModeString.equals(radioButtonProactiveAgents.getText().toString())) {
        convoscopeModeSelector.check(R.id.radioButtonProactiveAgents);
      } else if (currentModeString.equals(radioButtonLanguageLearning.getText().toString())) {
        convoscopeModeSelector.check(R.id.radioButtonLanguageLearning);
      } else if (currentModeString.equals(radioButtonWalkNGrok.getText().toString())) {
        convoscopeModeSelector.check(R.id.radioButtonWalkNGrok);
      } else if (currentModeString.equals(radioButtonADHDGlasses.getText().toString())) {
        convoscopeModeSelector.check(R.id.radioButtonADHDGlasses);
      } else if (currentModeString.equals(radioButtonScreenMirror.getText().toString())) {
        convoscopeModeSelector.check(R.id.radioButtonScreenMirror);
      }

      Context mContext = this.getContext();


      final Switch screenMirrorImageToggle = view.findViewById(R.id.screen_mirror_image_toggle);
      screenMirrorImageToggle.setChecked(PreferenceManager.getDefaultSharedPreferences(getContext()).getBoolean("screen_mirror_image", false));
      screenMirrorImageToggle.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
        @Override
        public void onCheckedChanged(CompoundButton compoundButton, boolean isChecked) {
          ((MainActivity)getActivity()).stopScreenCapture();
          PreferenceManager.getDefaultSharedPreferences(getContext())
                  .edit()
                  .putBoolean("screen_mirror_image", isChecked)
                  .apply();
        }
      });


      convoscopeModeSelector.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
        @Override
        public void onCheckedChanged(RadioGroup group, int checkedId) {
          ((MainActivity)getActivity()).stopScreenCapture();
          screenMirrorImageToggle.setEnabled(true);

          switch (checkedId) {
            case R.id.radioButtonProactiveAgents:
              // Implement action for Proactive Agents
              Log.d(TAG, "PROACTIVE AGENTS MODE SELECTED");
              ((MainActivity)getActivity()).mService.saveCurrentMode(mContext, "Proactive Agents");
              break;
            case R.id.radioButtonLanguageLearning:
              // Implement action for Language Learning
              Log.d(TAG, "LLSG MODE SELECTED");
              ((MainActivity)getActivity()).mService.saveCurrentMode(mContext, "Language Learning");
              break;
            case R.id.radioButtonWalkNGrok:
              // Note: This case is inactive but structured for completeness
              Log.d(TAG, "WALK_GROK MODE SELECTED");
              ((MainActivity)getActivity()).mService.saveCurrentMode(mContext, "Walk'n'Grok");
              break;
            case R.id.radioButtonADHDGlasses:
              // Note: This case is inactive but structured for completeness
              Log.d(TAG, "ADHD MODE SELECTED");
              ((MainActivity)getActivity()).mService.saveCurrentMode(mContext, "ADHD Glasses");
              break;
            case R.id.radioButtonScreenMirror:
              Log.d(TAG, "SCREEN MIRROR SELECTED");
              screenMirrorImageToggle.setEnabled(false);
              ((MainActivity)getActivity()).mService.saveCurrentMode(mContext, "");
              ((MainActivity)getActivity()).requestScreenCapturePermission();
          }
        }
      });


      ((MainActivity)getActivity()).startConvoscopeService();
    }

    private void pickContact() {
//        Intent pickContactIntent = new Intent(Intent.ACTION_PICK, ContactsContract.Contacts.CONTENT_URI);
//        startActivityForResult(pickContactIntent, PICK_CONTACT_REQUEST);
    }

  // Call this method to add a new text box to the list
  public void addResponseTextBox(String text) {
    addResponseTextBox(text, true);
  }

  public void addResponseTextBox(String text, boolean scroll) {
    responseTextUiAdapter.addText(text);
    if (scroll){
      responseRecyclerView.smoothScrollToPosition(responseTextUiAdapter.getItemCount() - 1);
    }
//      responseRecyclerView.scrollToPosition(responseTextUiAdapter.getItemCount() - 1);
  }

  // Call this method to add a new text box to the list
  public void addTranscriptTextBox(String text) {
   addTranscriptTextBox(text, true);
  }

  public void addTranscriptTextBox(String text, boolean scroll) {
    transcriptTextUiAdapter.addText(text);
    if (scroll) {
      transcriptRecyclerView.smoothScrollToPosition(transcriptTextUiAdapter.getItemCount() - 1);
    }
  }

  // Change UserID button
  public void showTextInputDialog(Context context) {
    AlertDialog.Builder builder = new AlertDialog.Builder(context);
    builder.setTitle("Enter new UserID");

    // Set up the input
    final EditText input = new EditText(context);
    LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.MATCH_PARENT);
    input.setLayoutParams(lp);
    builder.setView(input);

    // Set up the buttons
    builder.setPositiveButton("OK", new DialogInterface.OnClickListener() {
      @Override
      public void onClick(DialogInterface dialog, int which) {
        String result = input.getText().toString();
        // Use the text result as needed
        handleTextInputResult(result);
      }
    });
    builder.setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
      @Override
      public void onClick(DialogInterface dialog, int which) {
        dialog.cancel();
      }
    });

    builder.show();
  }

  private void handleTextInputResult(String result) {
    // Handle the text input result here
    Toast.makeText(this.getContext(), "Set UserID to: " + result, Toast.LENGTH_SHORT).show();
    EventBus.getDefault().post(new UserIdChangedEvent(result));
  }

  private void openSettingsFragment() {

    navController.navigate(R.id.nav_settings);
//    SettingsUi settingsFragment = new SettingsUi();
//    FragmentTransaction transaction = getSupportFragmentManager().beginTransaction();
//    transaction.replace(R.id.main_container, settingsFragment);
//    transaction.addToBackStack(null); // Add to back stack for back button support
//    transaction.commit();
  }

  //@Subscribe
  //public void onGoogleAuthFailedEvent(GoogleAuthFailedEvent event){
    //((MainActivity)getActivity()).signOut();
  //}

}

