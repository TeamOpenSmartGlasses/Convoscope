from typing import Optional

import requests
import json
import warnings

google_maps_api_key = "AIzaSyDrLB_P3dXj-CktMkNrHzlLS7p5cAZS2_Q"

def get_user_location() -> dict:
    """
    Get the user's location using the Google Geolocation API. This function should be used only for testing purposes. The user's location should be provided by the user's device.
    """

    warnings.warn("This function should be used only for testing purposes. The user's location should be provided by the user's device.")

    geolocation_url = "https://www.googleapis.com/geolocation/v1/geolocate?key=" + google_maps_api_key
    data = {
        "considerIp": "true",
    }

    try:
        response = requests.post(geolocation_url, data=json.dumps(data))
        response.raise_for_status()
        location = response.json()
        return location.get('location', {})
    except requests.RequestException as e:
        print(f"Error fetching user location: {e}")
        return {}


def get_nearby_places(location: dict, radius: int = 100, type: Optional[str] = None) -> list:
    """
    Get nearby places based on the user's location, using the Google Places API.
    """

    if not location:
        warnings.warn("Location was not provided.")
        return []

    nearby_url = (
        "https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=" + 
        str(location['lat']) + "," + 
        str(location['lng']) + 
        "&radius=" + str(radius) + 
        ("&type=" + type if type else '') + 
        "&key=" + google_maps_api_key
    )

    try:
        response = requests.get(nearby_url)
        response.raise_for_status()
        places = response.json()
        
        results = list()

        for place in places.get('results', []):
            result = f"name: {place.get('name', '')}, types: {place.get('types', [])}"
            results.append(result)

        return results
    except requests.RequestException as e:
        print(f"Error fetching nearby places: {e}")
        return []
