# uttsav_backend/app/services/spatio_temporal_engine.py

import math
from datetime import datetime, timedelta

print("--- UTTSAV SPATIO-TEMPORAL COLLISION ENGINE ---")

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates distance between two points in kilometers."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2)**2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def detect_route_clash(event_a: dict, event_b: dict, spatial_threshold_km: float = 0.1, temporal_threshold_minutes: int = 30) -> dict:
    """
    Compares two event timelines to see if they intersect geographically AND chronologically.
    - spatial_threshold_km: How close is considered a "clash" (0.1 km = 100 meters).
    - temporal_threshold_minutes: How much time buffer is needed between crowds.
    """
    
    # We parse the planned start times
    start_time_a = datetime.fromisoformat(event_a['planned_start_time'])
    start_time_b = datetime.fromisoformat(event_b['planned_start_time'])
    
    clashes_found = []

    # O(N*M) Comparison: Compare every waypoint of Event A against every waypoint of Event B
    for step_a in event_a['timeline']:
        # Calculate EXACT absolute time Crowd A is at this waypoint
        absolute_time_a = start_time_a + timedelta(seconds=step_a['time_offset_seconds'])
        lat_a, lng_a = step_a['end_location']['lat'], step_a['end_location']['lng']
        
        for step_b in event_b['timeline']:
            absolute_time_b = start_time_b + timedelta(seconds=step_b['time_offset_seconds'])
            lat_b, lng_b = step_b['end_location']['lat'], step_b['end_location']['lng']
            
            # 1. TEMPORAL CHECK: Are they at these points at roughly the same time?
            time_difference = abs((absolute_time_a - absolute_time_b).total_seconds()) / 60.0
            
            if time_difference <= temporal_threshold_minutes:
                # 2. SPATIAL CHECK: Since they are moving at the same time, are they physically close?
                distance = haversine_distance(lat_a, lng_a, lat_b, lng_b)
                
                if distance <= spatial_threshold_km:
                    clashes_found.append({
                        "distance_apart_km": round(distance, 3),
                        "time_difference_minutes": round(time_difference, 1),
                        "clash_time": absolute_time_a.strftime("%I:%M %p"),
                        "event_a_instruction": step_a['instructions'],
                        "event_b_instruction": step_b['instructions']
                    })
                    
    if clashes_found:
        # Sort by the most severe (closest distance)
        clashes_found.sort(key=lambda x: x['distance_apart_km'])
        return {"status": "CLASH_DETECTED", "critical_points": clashes_found}
    
    return {"status": "SAFE", "message": "No spatio-temporal overlaps detected."}

# =====================================================================
# SYSTEM TEST (Using Mock Google Maps Data)
# =====================================================================
if __name__ == "__main__":
    # Mocking what we would normally pull from Supabase after Google Maps API routing
    
    # Event A: A morning rally moving down the main road. Starts at 10:00 AM.
    event_a_mock = {
        "event_name": "Political Rally A",
        "planned_start_time": "2026-04-15T10:00:00", 
        "timeline": [
            {"time_offset_seconds": 600, "end_location": {"lat": 19.0178, "lng": 72.8478}, "instructions": "Walk north on Main St"},
            {"time_offset_seconds": 1200, "end_location": {"lat": 19.0200, "lng": 72.8480}, "instructions": "Turn left at the junction"}, # THE CLASH POINT
            {"time_offset_seconds": 1800, "end_location": {"lat": 19.0250, "lng": 72.8500}, "instructions": "Arrive at Park"}
        ]
    }

    # Event B: A religious procession. Starts at 10:15 AM (15 mins later)
    event_b_mock = {
        "event_name": "Religious Procession B",
        "planned_start_time": "2026-04-15T10:15:00", 
        "timeline": [
            {"time_offset_seconds": 300, "end_location": {"lat": 19.0201, "lng": 72.8481}, "instructions": "Cross the junction"}, # THE CLASH POINT
            {"time_offset_seconds": 900, "end_location": {"lat": 19.0300, "lng": 72.8600}, "instructions": "Head east to temple"}
        ]
    }

    print("Analyzing Trajectories for Spatio-Temporal Intersections...\n")
    
    result = detect_route_clash(event_a_mock, event_b_mock)
    
    if result['status'] == 'CLASH_DETECTED':
        print("🚨 SEVERE WARNING: ROUTE COLLISION DETECTED 🚨")
        print(f"The crowds will intersect at approximately {result['critical_points'][0]['clash_time']}.")
        print(f"Distance between crowds: {result['critical_points'][0]['distance_apart_km'] * 1000} meters.")
        print(f"Event A will be doing: '{result['critical_points'][0]['event_a_instruction']}'")
        print(f"Event B will be doing: '{result['critical_points'][0]['event_b_instruction']}'")
        print("\nACTION: Recommend alternative route to Event B via Google Maps Alternatives.")
    else:
        print("✅ Routes are mathematically safe. No overlap detected.")