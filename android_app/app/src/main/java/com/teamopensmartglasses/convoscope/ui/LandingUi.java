package com.teamopensmartglasses.convoscope.ui;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.Fragment;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.preference.PreferenceManager;

import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;
import com.teamopensmartglasses.convoscope.MainActivity;
import com.teamopensmartglasses.convoscope.R;
import com.teamopensmartglasses.smartglassesmanager.speechrecognition.ASR_FRAMEWORKS;

public class LandingUi extends Fragment {
  public final String TAG = "Convoscope_LandingActivity";
  public final String fragmentLabel = "Landing page";
  private NavController navController;
  private FirebaseAuth mAuth;

  public LandingUi(){

  }

  @Override
  public View onCreateView(LayoutInflater inflater, ViewGroup container,
                           Bundle savedInstanceState) {
    // Inflate the layout for this fragment
    return inflater.inflate(R.layout.landing_fragment, container, false);
  }

  @Override
  public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    navController = Navigation.findNavController(getActivity(), R.id.nav_host_fragment);

    // Check if we're already logged in... if so, skip to MainActivity
    mAuth = FirebaseAuth.getInstance();
    FirebaseUser currentUser = mAuth.getCurrentUser();
    if (currentUser != null) {
      navController.navigate(R.id.nav_convoscope);
    }

    final Button landingButton = view.findViewById(R.id.landing_button);
    landingButton.setOnClickListener(new View.OnClickListener() {
      public void onClick(View v) {
        navController.navigate(R.id.nav_login);
      }
    });
  }

  public String getSavedAuthToken(){
      return PreferenceManager.getDefaultSharedPreferences(getContext()).getString("auth_token", "");
  }
}