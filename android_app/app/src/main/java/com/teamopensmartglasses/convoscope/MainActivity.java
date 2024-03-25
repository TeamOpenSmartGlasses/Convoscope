/*
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.teamopensmartglasses.convoscope;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.ActivityManager;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatDelegate;
import android.os.Bundle;
import android.os.Handler;
import android.os.IBinder;
import android.util.Log;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.FragmentTransaction;
import androidx.navigation.NavController;
import androidx.navigation.NavOptions;
import androidx.navigation.fragment.NavHostFragment;
import androidx.preference.PreferenceManager;

import com.firebase.ui.auth.AuthUI;
import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;
import com.teamopensmartglasses.convoscope.ui.LandingUi;
import com.teamopensmartglasses.convoscope.ui.SelectSmartGlassesUi;
import com.teamopensmartglasses.convoscope.ui.UiUtils;
import com.teamopensmartglasses.smartglassesmanager.supportedglasses.SmartGlassesDevice;
import com.teamopensmartglasses.smartglassesmanager.utils.PermissionsUtils;

public class MainActivity extends AppCompatActivity {
  public final String TAG = "Convoscope_MainActivity";
  public ConvoscopeService mService;
  boolean mBound;
  private NavController navController;
  PermissionsUtils permissionsUtils;

  //Permissions
  private static final int PERMISSIONS_REQUEST_RECORD_AUDIO = 1;
  private static final int PICK_CONTACT_REQUEST = 1;
  private static final int READ_CONTACTS_PERMISSIONS_REQUEST = 2;

  public SmartGlassesDevice selectedDevice;

  @SuppressLint("ClickableViewAccessibility")
  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    mBound = false;

    //setup the nav bar
    NavHostFragment navHostFragment = (NavHostFragment) getSupportFragmentManager().findFragmentById(R.id.nav_host_fragment);
    Log.d(TAG, getSupportFragmentManager().getFragments().toString());
    navController = navHostFragment.getNavController();
  }

  @Override
  public void onStart() {
    super.onStart();
    UiUtils.setupTitle(this, "Convoscope");

    permissionsUtils = new PermissionsUtils(this, TAG);
    if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
        != PackageManager.PERMISSION_GRANTED) {
      permissionsUtils.getSomePermissions();
      //ActivityCompat.requestPermissions(:w
      // this, new String[] {Manifest.permission.RECORD_AUDIO}, PERMISSIONS_REQUEST_RECORD_AUDIO);
      }
  }

  @Override
  public void onStop() {
    super.onStop();
  }

  @Override
  public void onDestroy() {
    super.onDestroy();
  }

  @Override
  public void onRequestPermissionsResult(
      int requestCode, String[] permissions, int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    switch (requestCode) {
      case PERMISSIONS_REQUEST_RECORD_AUDIO:
        // If request is cancelled, the result arrays are empty.
        if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
//          showAPIKeyDialog();
        } else {
          // This should nag user again if they launch without the permissions.
          Toast.makeText(
                  this,
                  "This app does not work without the Microphone permission.",
                  Toast.LENGTH_SHORT)
              .show();
          finish();
        }
        return;
      case READ_CONTACTS_PERMISSIONS_REQUEST:
        if (grantResults.length > 0 &&
                grantResults[0] == PackageManager.PERMISSION_GRANTED) {
          //pickContact();
        } else {
          Toast.makeText(this, "Permission to read contacts denied", Toast.LENGTH_SHORT).show();
        }
      default: // Should not happen. Something we did not request.
    }
  }

  @Override
  public void onConfigurationChanged(@NonNull Configuration newConfig) {
    super.onConfigurationChanged(newConfig);
    // Check for night mode status and apply theme accordingly
    int nightModeFlags = newConfig.uiMode & Configuration.UI_MODE_NIGHT_MASK;
    switch (nightModeFlags) {
      case Configuration.UI_MODE_NIGHT_YES:
        AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_YES);
        break;
      case Configuration.UI_MODE_NIGHT_NO:
        AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_NO);
        break;
      case Configuration.UI_MODE_NIGHT_UNDEFINED:
        break;
    }
  }

  @Override
  protected void onResume() {
    super.onResume();
//    UiUtils.setupTitle(this, defaultFragmentLabel);
    //register receiver that gets data from the service


    if (isMyServiceRunning(ConvoscopeService.class)) {
      //bind to WearableAi service
      bindConvoscopeService();

      //ask the service to send us all the Convoscope responses
      if (mService != null) {
        mService.sendUiUpdateFull();
      }
    }
  }

  @Override
  protected void onPause() {
    super.onPause();

    //unbind wearableAi service
    unbindConvoscopeService();
  }

  public void restartConvoscopeService() {
    stopConvoscopeService();
    Handler starter = new Handler();
    starter.postDelayed(new Runnable() {
      @Override
      public void run() {
        startConvoscopeService();
      }
    }, 300);
  }

  public void stopConvoscopeService() {
    unbindConvoscopeService();
    if (!isMyServiceRunning(ConvoscopeService.class)) return;
    Intent stopIntent = new Intent(this, ConvoscopeService.class);
    stopIntent.setAction(ConvoscopeService.ACTION_STOP_FOREGROUND_SERVICE);
    startService(stopIntent);
  }

  public void sendConvoscopeServiceMessage(String message) {
    if (!isMyServiceRunning(ConvoscopeService.class)) return;
    Intent messageIntent = new Intent(this, ConvoscopeService.class);
    messageIntent.setAction(message);
    startService(messageIntent);
  }

  public void startConvoscopeService() {
    if (isMyServiceRunning(ConvoscopeService.class)){
      Log.d(TAG, "Not starting Convoscope service because it's already started.");
      return;
    }

    Log.d(TAG, "Starting Convoscope service.");
    Intent startIntent = new Intent(this, ConvoscopeService.class);
    startIntent.setAction(ConvoscopeService.ACTION_START_FOREGROUND_SERVICE);
    startService(startIntent);
    bindConvoscopeService();
  }

  //check if service is running
  private boolean isMyServiceRunning(Class<?> serviceClass) {
    ActivityManager manager = (ActivityManager) getSystemService(Context.ACTIVITY_SERVICE);
    for (ActivityManager.RunningServiceInfo service : manager.getRunningServices(Integer.MAX_VALUE)) {
      if (serviceClass.getName().equals(service.service.getClassName())) {
        return true;
      }
    }
    return false;
  }

  public void bindConvoscopeService(){
    if (!mBound){
      Intent intent = new Intent(this, ConvoscopeService.class);
      bindService(intent, convoscopeAppServiceConnection, Context.BIND_AUTO_CREATE);
    }
  }

  public void unbindConvoscopeService() {
    if (mBound){
      unbindService(convoscopeAppServiceConnection);
      mBound = false;
    }
  }

  /** Defines callbacks for service binding, passed to bindService() */
  private ServiceConnection convoscopeAppServiceConnection = new ServiceConnection() {
    @Override
    public void onServiceConnected(ComponentName className,
                                   IBinder service) {
      // We've bound to LocalService, cast the IBinder and get LocalService instance
      ConvoscopeService.LocalBinder sgmLibServiceBinder = (ConvoscopeService.LocalBinder) service;
      mService = (ConvoscopeService) sgmLibServiceBinder.getService();
      mBound = true;

      //get update for UI
      mService.sendUiUpdateFull();
    }
    @Override
    public void onServiceDisconnected(ComponentName arg0) {
      mBound = false;
    }
  };



  @Override
  protected void onActivityResult(int requestCode, int resultCode, Intent data) {
    super.onActivityResult(requestCode, resultCode, data);
//    if (requestCode == PICK_CONTACT_REQUEST && resultCode == RESULT_OK) {
//      Uri contactUri = data.getData();
//
//      // Perform another query to get the contact's details
//      Cursor cursor = getContentResolver().query(contactUri, null,
//              null, null, null);
//
//      if (cursor != null && cursor.moveToFirst()) {
//        String id = cursor.getString(cursor.getColumnIndexOrThrow(ContactsContract.Contacts._ID));
//
//        Cursor phones = getContentResolver().query(
//                ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
//                null,
//                ContactsContract.CommonDataKinds.Phone.CONTACT_ID + " = " + id,
//                null, null);
//
//        while (phones != null && phones.moveToNext()) {
//          @SuppressLint("Range") String phoneNumber = phones.getString(phones.getColumnIndex(ContactsContract.CommonDataKinds.Phone.NUMBER));
//          @SuppressLint("Range") String name = phones.getString(phones.getColumnIndex(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME));
//          Toast.makeText(this, "Phone number: " + phoneNumber, Toast.LENGTH_LONG).show();
//          EventBus.getDefault().post(new SharingContactChangedEvent(name, phoneNumber));
//        }
//
//        cursor.close();
//
//        if (phones != null) {
//          phones.close();
//        }
//      }
//    }
  }

  @Override
  public boolean onSupportNavigateUp() {
    onBackPressed();
    return true;
  }

  public void setSavedAuthToken(Context context, String newAuthToken){
    PreferenceManager.getDefaultSharedPreferences(context)
            .edit()
            .putString("auth_token", newAuthToken)
            .apply();
  }

  public void signOut(){
    setSavedAuthToken(getApplicationContext(), "");
    AuthUI.getInstance()
            .signOut(this)
            .addOnCompleteListener(new OnCompleteListener<Void>() {
              public void onComplete(@NonNull Task<Void> task) {
                Log.d(TAG, "LOGGED OUT (SUCCESSFULLY)");
                stopService(new Intent(MainActivity.this, ConvoscopeService.class));
                NavOptions navOptions = new NavOptions.Builder()
                        .setPopUpTo(R.id.nav_landing, true) // Replace 'nav_graph_start_destination' with the ID of your start destination in the nav graph
                        .build();
                navController.navigate(R.id.nav_landing, null, navOptions);
              }
            })
            .addOnFailureListener(new OnFailureListener() {
              @Override
              public void onFailure(@NonNull Exception e) {
                Log.d(TAG, "LOGGED OUT (WITH ERROR)");
                stopService(new Intent(MainActivity.this, ConvoscopeService.class));
                NavOptions navOptions = new NavOptions.Builder()
                        .setPopUpTo(R.id.nav_landing, true) // Replace 'nav_graph_start_destination' with the ID of your start destination in the nav graph
                        .build();
                navController.navigate(R.id.nav_landing, null, navOptions);
              }
            });
  }

  public void connectSmartGlasses(SmartGlassesDevice device){
    this.selectedDevice = device;

    //check if the service is running. If not, we should start it first, so it doesn't die when we unbind
    if (!isMyServiceRunning(ConvoscopeService.class)){
      Log.e(TAG, "Something went wrong, service should be started and bound.");
    } else {
      mService.connectToSmartGlasses(device);
    }
  }

  public void openGlassesSelector() {
    SelectSmartGlassesUi selectorFragment = new SelectSmartGlassesUi();
    FragmentTransaction transaction = getSupportFragmentManager().beginTransaction();
    transaction.replace(com.teamopensmartglasses.convoscope.R.id.main_container, selectorFragment);
    transaction.addToBackStack(null); // Add to back stack for back button support
    transaction.commit();
  }

      public void saveCurrentMode(Context context, String currentModeString) {
        //update the features for the new mode
        if (mService != null) {
          mService.saveCurrentMode(this, currentModeString);
        } else {
          //save the new mode
          PreferenceManager.getDefaultSharedPreferences(context)
                  .edit()
                  .putString(context.getResources().getString(R.string.SHARED_PREF_CURRENT_MODE), currentModeString)
                  .apply();
        }
    }

    public String getCurrentMode(Context context) {
        String currentModeString = PreferenceManager.getDefaultSharedPreferences(context).getString(context.getResources().getString(R.string.SHARED_PREF_CURRENT_MODE), "");
        if (currentModeString.equals("")){
            saveCurrentMode(context, "Language Learning");
            currentModeString = "Language Learning";
        }
        return currentModeString;
    }

}
