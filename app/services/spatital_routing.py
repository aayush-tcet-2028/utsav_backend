# uttsav_backend/app/services/spatial_routing.py

import math

print("--- UTTSAV SPATIAL ROUTING ENGINE ---")

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates the great-circle distance between two points 
    on the Earth in kilometers.
    """
    R = 6371.0 # Radius of Earth in kilometers
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat / 2)**2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2)
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# =====================================================================
# MOCK VENUE DATABASE (To be replaced by Supabase later)
# =====================================================================
master_venues = [
    {"id": "V001", "name": "Shivaji Park", "lat": 19.0268, "lon": 72.8386, "max_capacity": 100000},
    {"id": "V002", "name": "MMRDA Grounds, BKC", "lat": 19.0645, "lon": 72.8640, "max_capacity": 150000},
    {"id": "V003", "name": "Azad Maidan", "lat": 18.9388, "lon": 72.8313, "max_capacity": 50000},
    {"id": "V004", "name": "Jio World Garden", "lat": 19.0656, "lon": 72.8656, "max_capacity": 15000},
    {"id": "V005", "name": "Wankhede Stadium", "lat": 18.9388, "lon": 72.8258, "max_capacity": 33000}
]

def find_alternative_venues(target_lat: float, target_lon: float, required_capacity: int, search_radius_km: float = 15.0) -> list:
    """
    Finds venues within a specific radius that can safely hold the expected crowd.
    """
    valid_alternatives = []
    
    for venue in master_venues:
        # 1. Strict Safety Check: Can this venue hold the crowd?
        if venue["max_capacity"] >= required_capacity:
            
            # 2. Spatial Check: How far is it from the original requested location?
            dist = haversine_distance(target_lat, target_lon, venue["lat"], venue["lon"])
            
            if dist <= search_radius_km:
                venue_data = venue.copy()
                venue_data["distance_km"] = round(dist, 2)
                valid_alternatives.append(venue_data)
                
    # 3. Sort the valid venues by closest distance
    valid_alternatives.sort(key=lambda x: x["distance_km"])
    
    return valid_alternatives

# =====================================================================
# SYSTEM TEST
# =====================================================================
if __name__ == "__main__":
    # Scenario: User wants a venue near Dadar (Lat: 19.0178, Lon: 72.8478)
    # The event expects 45,000 people. 
    # But let's assume their first choice is fully booked.
    
    requested_lat = 19.0178
    requested_lon = 72.8478
    expected_crowd = 45000
    
    print(f"\nScenario: Searching for alternatives near Dadar for {expected_crowd} people.")
    print("Executing Spatial & Capacity Filtering...\n")
    
    alternatives = find_alternative_venues(
        target_lat=requested_lat, 
        target_lon=requested_lon, 
        required_capacity=expected_crowd,
        search_radius_km=20.0
    )
    
    if not alternatives:
        print("No suitable venues found in this radius.")
    else:
        print("RECOMMENDED ALTERNATIVES:")
        for idx, alt in enumerate(alternatives, 1):
            print(f"{idx}. {alt['name']} | Distance: {alt['distance_km']} km | Max Capacity: {alt['max_capacity']}")