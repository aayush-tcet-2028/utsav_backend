# uttsav_backend/app/services/google_maps_router.py

import requests
import urllib.parse

print("--- UTTSAV GOOGLE MAPS ROUTING ENGINE ---")

# TODO: Replace with your actual Google Cloud API Key
GOOGLE_MAPS_API_KEY = "your_api_key"

def fetch_route_options(origin: str, destination: str) -> dict:
    """
    Fetches walking/driving routes using Google Maps Directions API.
    Returns multiple alternatives with distance and duration metrics.
    """
    # URL encode the addresses (e.g., "Shivaji Park, Mumbai" -> "Shivaji+Park%2C+Mumbai")
    origin_enc = urllib.parse.quote(origin)
    dest_enc = urllib.parse.quote(destination)
    
    # We use mode=walking for processions, and alternatives=true to give the user choices
    url = f"https://maps.googleapis.com/maps/api/directions/json?origin={origin_enc}&destination={dest_enc}&mode=walking&alternatives=true&key={GOOGLE_MAPS_API_KEY}"
    
    response = requests.get(url)
    if response.status_code != 200:
        return {"status": "error", "message": "API connection failed."}
        
    data = response.json()
    
    if data['status'] != 'OK':
        return {"status": "error", "message": data['status']}

    routes_summary = []
    
    # Parse the different route alternatives Google provides
    for idx, route in enumerate(data['routes'], 1):
        leg = route['legs'][0] # For simple point A to B, there is one 'leg'
        
        # We extract the steps to calculate exact spatio-temporal locations later
        timeline_steps = []
        current_time_offset = 0 # seconds from start
        
        for step in leg['steps']:
            step_duration = step['duration']['value']
            current_time_offset += step_duration
            
            timeline_steps.append({
                "end_location": step['end_location'], # lat/lng
                "time_offset_seconds": current_time_offset,
                "instructions": step['html_instructions']
            })
            
        routes_summary.append({
            "route_id": f"R{idx}",
            "summary": route['summary'],
            "total_distance": leg['distance']['text'],
            "total_duration": leg['duration']['text'],
            "total_duration_seconds": leg['duration']['value'],
            "encoded_polyline": route['overview_polyline']['points'],
            "timeline": timeline_steps # This is crucial for our Collision Engine
        })
        
    return {"status": "success", "routes": routes_summary}

# =====================================================================
# SYSTEM TEST
# =====================================================================
if __name__ == "__main__":
    test_origin = "Dadar TT Circle, Mumbai"
    test_destination = "Shivaji Park, Mumbai"
    
    print(f"\nScenario: Fetching moving procession routes from {test_origin} to {test_destination}.")
    
    result = fetch_route_options(test_origin, test_destination)
    
    if result['status'] == 'success':
        print(f"\nFound {len(result['routes'])} alternative routes:")
        for r in result['routes']:
            print(f"\n--- {r['route_id']}: Via {r['summary']} ---")
            print(f"Distance: {r['total_distance']}")
            print(f"Estimated Time: {r['total_duration']}")
            print(f"Waypoints Extracted: {len(r['timeline'])} points tracked for collision detection.")
    else:
        print(f"\nError: {result['message']}")
        print("(Make sure you pasted your actual Google Maps API Key!)") 
