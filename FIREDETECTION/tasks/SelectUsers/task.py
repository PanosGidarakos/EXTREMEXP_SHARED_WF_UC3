[sys.path.append(os.path.join(os.getcwd(), folder)) for folder in variables.get("dependent_modules_folders").split(",")]
import proactive_helper as ph
import os
import requests
import math
import json
import time
from typing import Tuple, List, Optional


class TaskSelectUsers:
    def __init__(
        self,
        osrm_url: str,
        selection_diameter_km: Optional[float] = None,
        user_profile_selection: str = "driving",
        filter_only_available: bool = True,
        sort_by: str = "distance",
        osrm_timeout_s: int = 5,
        euclidian_filter_km: Optional[float] = None,
    ):
        """
        Initialize the user selection process.
        :param osrm_url: URL of the OSRM service for calculating distances
        :param selection_diameter_km: Maximum diameter (km) to include users
        :param user_profile_selection: Routing profile (driving, walking, cycling)
        :param filter_only_available: Only include users marked available
        :param sort_by: Sort by 'distance' or 'travel_time'
        :param osrm_timeout_s: Timeout for OSRM requests in seconds
        :param euclidian_filter_km: Pre-filter by straight-line distance (km)
        """
        self.osrm_url = osrm_url
        self.selection_diameter_km = selection_diameter_km
        self.user_profile_selection = user_profile_selection
        self.filter_only_available = filter_only_available
        self.sort_by = sort_by
        self.osrm_timeout_s = osrm_timeout_s
        self.euclidian_filter_km = euclidian_filter_km

    def get_all_users(self) -> List[dict]:
        try:
            usersInfo = variables.get("UsersInfo")
            usersInfoFile = usersInfo + "/mock_users.json"
            with open(usersInfoFile, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading mock users: {e}")
            return []

    def haversine_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        R = 6371.0  # Earth radius in km
        lat1, lon1 = map(math.radians, loc1)
        lat2, lon2 = map(math.radians, loc2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def calculate_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> Tuple[float, float, dict]:
        """
        Return driving distance (km), travel time (s) and full OSRM response between two coords using OSRM.
        """
        print(f"Calculating distance from {loc1} to {loc2} using OSRM at {self.osrm_url}")
        try:
            url = (
                f"{self.osrm_url}/route/v1/{self.user_profile_selection}/"
                f"{loc1[1]},{loc1[0]};{loc2[1]},{loc2[0]}?overview=full&geometries=geojson"
            )
            resp = requests.get(url, timeout=self.osrm_timeout_s)
            resp.raise_for_status()
            full_response = resp.json()
            data = full_response['routes'][0]
            print(f"OSRM response: {data}")

            distance_km = data['distance'] / 1000.0
            duration_s = data['duration']
            return distance_km, duration_s, full_response
        except Exception as e:
            print(f"Error in OSRM request: {e}")
            return float('inf'), float('inf'), {}

    def select_nearest_available_users(
        self,
        alert_location: Tuple[float, float],
        num_users: int = 2
    ) -> List[dict]:
        """
        Select users by combined filters and sorting.
        """
        # Start timing the total computation
        total_start_time = time.time()
        
        users = self.get_all_users()
        print(f"Total users loaded: {len(users)}")
        
        candidates: List[Tuple[dict, float, float, dict]] = []
        osrm_results_map = {}
        osrm_response_values = []
        user_computation_times = {}  # Store computation time for each user
        
        # Counters for metrics
        status_filtered = 0
        haversine_filtered = 0
        diameter_filtered = 0
        osrm_errors = 0
        invalid_coordinates = 0

        for idx, user in enumerate(users):
            # Start timing for this user
            user_start_time = time.time()
            lat = user.get('latitude')
            lon = user.get('longitude')
            if lat is None or lon is None:
                invalid_coordinates += 1
                continue
            loc = (lat, lon)
            
            # Filter by user status (driving, walking, cycling)
            user_status = user.get('status')
            if user_status != self.user_profile_selection:
                status_filtered += 1
                continue
            
            # Pre-filter by straight-line distance
            if self.euclidian_filter_km is not None:
                haversine_dist = self.haversine_distance(alert_location, loc)
                if haversine_dist > self.euclidian_filter_km:
                    haversine_filtered += 1
                    continue
            # Calculate route metrics
            distance_km, duration_s, full_response = self.calculate_distance(alert_location, loc)
            
            # Check for OSRM errors
            if distance_km == float('inf') or duration_s == float('inf'):
                osrm_errors += 1
            
            # End timing for this user
            user_end_time = time.time()
            user_computation_time = user_end_time - user_start_time
            
            # Store OSRM result for this user (by index or user id if available)
            user_key = user.get('id', idx)
            osrm_results_map[user_key] = {'distance_km': distance_km, 'duration_s': duration_s}
            user_computation_times[user_key] = user_computation_time
            
            # Store full OSRM response for OSRM_RESPONSE_VALUES
            if full_response:
                osrm_response_values.append({
                    "user_id": str(user_key),
                    "osrm_response": full_response
                })
            
            # Filter by diameter
            if self.selection_diameter_km is not None:
                if distance_km > (self.selection_diameter_km / 2):
                    diameter_filtered += 1
                    continue
            candidates.append((user, distance_km, duration_s, full_response))

        # Sort candidates
        if self.sort_by == 'travel_time':
            sorted_list = sorted(candidates, key=lambda x: x[2])
        else:
            sorted_list = sorted(candidates, key=lambda x: x[1])

        selected: List[dict] = []
        for user, dist, dur, full_resp in sorted_list:
            if self.filter_only_available and not user.get('available', True):
                continue
            selected.append(user)
            if len(selected) >= num_users:
                break

        # End timing the total computation
        total_end_time = time.time()
        total_computation_time = total_end_time - total_start_time
        
        # Log the selection results
        print(f"Profile filter: {self.user_profile_selection}")
        print(f"Selected {len(selected)} users out of {len(users)} total users")
        print(f"Total computation time: {total_computation_time:.4f} seconds")
        if not selected:
            print("Warning: No users selected based on current criteria")

        # Create filtering metrics
        filtering_metrics = {
            'total_users': len(users),
            'invalid_coordinates': invalid_coordinates,
            'status_filtered': status_filtered,
            'haversine_filtered': haversine_filtered,
            'osrm_errors': osrm_errors,
            'diameter_filtered': diameter_filtered,
            'candidates': len(candidates),
            'selected': len(selected)
        }
        
        return selected, osrm_results_map, osrm_response_values, total_computation_time, user_computation_times, filtering_metrics

if __name__ == '__main__':
    OSRM_URL =''
    SELECTION_DIAMETER_KM = float(variables.get("selection_diameter_km"))
    USER_PROFILE_SELECTION = variables.get("user_profile_selection")
    FILTER_ONLY_AVAILABLE = bool(variables.get("filter_only_available"))
    SORT_BY = variables.get("sort_by")
    OSRM_TIMEOUT_S = int(variables.get("osrm_timeout_s"))
    EUCLIDIAN_FILTER_KM = float(variables.get("euclidian_filter_km"))
    NUM_USERS_SELECTION = int(variables.get("num_users_selection"))

    selector = TaskSelectUsers(
        osrm_url=OSRM_URL,
        selection_diameter_km=SELECTION_DIAMETER_KM,
        user_profile_selection=USER_PROFILE_SELECTION,
        filter_only_available=FILTER_ONLY_AVAILABLE,
        sort_by=SORT_BY,
        osrm_timeout_s=OSRM_TIMEOUT_S,
        euclidian_filter_km=EUCLIDIAN_FILTER_KM,
    )

    alert_location = (48.79842194845064, 1.9707823090763419)
    selected_users, osrm_results_map, osrm_response_values, total_computation_time, user_computation_times, filtering_metrics = selector.select_nearest_available_users(
        alert_location,
        num_users=NUM_USERS_SELECTION
    )

    print(f"Selected {len(selected_users)} users: {selected_users}")
    print(f"Total computation time: {total_computation_time:.4f} seconds")
    print(f"Per-user computation times: {user_computation_times}")
    
    resultMap.put("SELECTED_USERS", json.dumps(selected_users))
    resultMap.put("OSRM_RESULTS", json.dumps(osrm_results_map))
    resultMap.put("OSRM_RESPONSE_VALUES", json.dumps(osrm_response_values))
    resultMap.put("TOTAL_COMPUTATION_TIME", total_computation_time)
    resultMap.put("USER_COMPUTATION_TIMES", json.dumps(user_computation_times))
    resultMap.put("FILTERING_METRICS", json.dumps(filtering_metrics))
