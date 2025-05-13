import requests
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

class OpenSkyConnector:
    """
    Connector for OpenSky Network REST API to fetch flight tracking data.
    
    Implements direct HTTP requests to the OpenSky Network REST API endpoints
    as described in https://openskynetwork.github.io/opensky-api/rest.html
    """
    
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize the OpenSky Network API connector.
        
        Args:
            username: Optional OpenSky Network username
            password: Optional OpenSky Network password
        """
        self.base_url = "https://opensky-network.org/api"
        self.auth = (username, password) if username and password else None
        self.logger = logging.getLogger(__name__)
        
        # If credentials are provided, log that we're using authenticated access
        if username and password:
            self.logger.info("Using authenticated access to OpenSky API")
        else:
            self.logger.info("Using anonymous access to OpenSky API (rate limited)")
    
    def get_states(self, 
                  time: Optional[int] = None, 
                  icao24: Optional[List[str]] = None,
                  bbox: Optional[Tuple[float, float, float, float]] = None) -> Dict[str, Any]:
        """
        Retrieve current state vectors for aircraft using the /states/all endpoint.
        
        Args:
            time: Unix timestamp (seconds since epoch)
            icao24: One or more ICAO24 transponder addresses (hex string)
            bbox: Bounding box (min_latitude, min_longitude, max_latitude, max_longitude)
            
        Returns:
            State vectors for aircraft
        """
        endpoint = f"{self.base_url}/states/all"
        params = {}
        
        # Add optional parameters
        if time is not None:
            params['time'] = time
        
        if icao24 is not None:
            params['icao24'] = ','.join(icao24) if isinstance(icao24, list) else icao24
        
        if bbox is not None:
            min_lat, min_lon, max_lat, max_lon = bbox
            params['lamin'] = min_lat
            params['lomin'] = min_lon
            params['lamax'] = max_lat
            params['lomax'] = max_lon
        
        try:
            self.logger.debug(f"Making request to {endpoint} with params {params}")
            response = requests.get(endpoint, params=params, auth=self.auth, timeout=30)
            
            # Check for rate limiting or other errors
            if response.status_code == 429:
                self.logger.error("Rate limit exceeded for OpenSky API")
                return {"error": "Rate limit exceeded"}
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching states: {str(e)}")
            return {"error": str(e)}
    
    def get_own_states(self, 
                      time: Optional[int] = None,
                      icao24: Optional[List[str]] = None,
                      serials: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Retrieve state vectors from your receivers (requires authentication).
        
        This endpoint is only available for OpenSky contributors who feed data
        to the network.
        
        Args:
            time: Unix timestamp (seconds since epoch)
            icao24: One or more ICAO24 transponder addresses (hex string)
            serials: Serial numbers of your receivers
            
        Returns:
            State vectors for aircraft
        """
        if not self.auth:
            self.logger.error("Authentication required for get_own_states")
            return {"error": "Authentication required"}
            
        endpoint = f"{self.base_url}/states/own"
        params = {}
        
        # Add optional parameters
        if time is not None:
            params['time'] = time
        
        if icao24 is not None:
            params['icao24'] = ','.join(icao24) if isinstance(icao24, list) else icao24
            
        if serials is not None:
            params['serials'] = ','.join(map(str, serials))
        
        try:
            self.logger.debug(f"Making request to {endpoint} with params {params}")
            response = requests.get(endpoint, params=params, auth=self.auth, timeout=30)
            
            if response.status_code == 429:
                self.logger.error("Rate limit exceeded for OpenSky API")
                return {"error": "Rate limit exceeded"}
                
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching own states: {str(e)}")
            return {"error": str(e)}
    
    def get_flights_by_interval(self,
                               begin_time: datetime,
                               end_time: datetime) -> List[Dict[str, Any]]:
        """
        Retrieve all flights within a time interval using the /flights/all endpoint.
        
        The time interval must not exceed two hours.
        
        Args:
            begin_time: Start time for the query
            end_time: End time for the query
            
        Returns:
            List of flights within the time interval
        """
        endpoint = f"{self.base_url}/flights/all"
        
        # Convert datetime to UNIX timestamp
        begin = int(begin_time.timestamp())
        end = int(end_time.timestamp())
        
        # Check if time interval exceeds 2 hours (7200 seconds)
        if end - begin > 7200:
            self.logger.warning("Time interval exceeds 2 hours, truncating to 2 hours")
            end = begin + 7200
        
        params = {
            'begin': begin,
            'end': end
        }
        
        try:
            self.logger.debug(f"Making request to {endpoint} with params {params}")
            response = requests.get(endpoint, params=params, auth=self.auth, timeout=30)
            
            if response.status_code == 429:
                self.logger.error("Rate limit exceeded for OpenSky API")
                return []
                
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching flights for interval: {str(e)}")
            return []
    
    def get_flights_by_aircraft(self, 
                               icao24: str, 
                               begin_time: datetime, 
                               end_time: datetime) -> List[Dict[str, Any]]:
        """
        Retrieve flights for a specific aircraft within a time range.
        
        Args:
            icao24: ICAO24 transponder address (hex string)
            begin_time: Start time for the query
            end_time: End time for the query
            
        Returns:
            List of flights for the aircraft
        """
        endpoint = f"{self.base_url}/flights/aircraft"
        
        # Convert datetime to UNIX timestamp
        begin = int(begin_time.timestamp())
        end = int(end_time.timestamp())
        
        params = {
            'icao24': icao24,
            'begin': begin,
            'end': end
        }
        
        try:
            self.logger.debug(f"Making request to {endpoint} with params {params}")
            response = requests.get(endpoint, params=params, auth=self.auth, timeout=30)
            
            if response.status_code == 429:
                self.logger.error("Rate limit exceeded for OpenSky API")
                return []
                
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching flights for {icao24}: {str(e)}")
            return []
    
    def get_arrivals(self, airport: str, begin_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        Retrieve flights arriving at a specific airport within a time interval.
        
        Args:
            airport: ICAO code of the airport
            begin_time: Start time for the query
            end_time: End time for the query
            
        Returns:
            List of arriving flights
        """
        endpoint = f"{self.base_url}/flights/arrival"
        
        # Convert datetime to UNIX timestamp
        begin = int(begin_time.timestamp())
        end = int(end_time.timestamp())
        
        params = {
            'airport': airport,
            'begin': begin,
            'end': end
        }
        
        try:
            self.logger.debug(f"Making request to {endpoint} with params {params}")
            response = requests.get(endpoint, params=params, auth=self.auth, timeout=30)
            
            if response.status_code == 429:
                self.logger.error("Rate limit exceeded for OpenSky API")
                return []
                
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching arrivals for {airport}: {str(e)}")
            return []
    
    def get_departures(self, airport: str, begin_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        Retrieve flights departing from a specific airport within a time interval.
        
        Args:
            airport: ICAO code of the airport
            begin_time: Start time for the query
            end_time: End time for the query
            
        Returns:
            List of departing flights
        """
        endpoint = f"{self.base_url}/flights/departure"
        
        # Convert datetime to UNIX timestamp
        begin = int(begin_time.timestamp())
        end = int(end_time.timestamp())
        
        params = {
            'airport': airport,
            'begin': begin,
            'end': end
        }
        
        try:
            self.logger.debug(f"Making request to {endpoint} with params {params}")
            response = requests.get(endpoint, params=params, auth=self.auth, timeout=30)
            
            if response.status_code == 429:
                self.logger.error("Rate limit exceeded for OpenSky API")
                return []
                
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching departures for {airport}: {str(e)}")
            return []
    
    def get_track(self, icao24: str, time: Optional[int] = None) -> Dict[str, Any]:
        """
        Retrieve the trajectory for a certain aircraft.
        
        Note: This endpoint is experimental and not officially documented.
        
        Args:
            icao24: ICAO24 transponder address (hex string)
            time: Unix timestamp (seconds since epoch)
            
        Returns:
            Trajectory data for the aircraft
        """
        endpoint = f"{self.base_url}/tracks/all"
        
        params = {
            'icao24': icao24
        }
        
        if time is not None:
            params['time'] = time
        
        try:
            self.logger.debug(f"Making request to {endpoint} with params {params}")
            response = requests.get(endpoint, params=params, auth=self.auth, timeout=30)
            
            if response.status_code == 429:
                self.logger.error("Rate limit exceeded for OpenSky API")
                return {"error": "Rate limit exceeded"}
                
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching track for {icao24}: {str(e)}")
            return {"error": str(e)}
    
    def format_state_vectors(self, states_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format raw state vectors data into a more usable dictionary format.
        
        Args:
            states_data: Raw state vectors data from the API
            
        Returns:
            List of formatted state vectors
        """
        if "error" in states_data:
            return []
            
        if "states" not in states_data or not states_data["states"]:
            return []
        
        formatted_states = []
        for state in states_data["states"]:
            if len(state) >= 17:  # Ensure we have all fields
                formatted_state = {
                    "icao24": state[0],
                    "callsign": state[1].strip() if state[1] else None,
                    "origin_country": state[2],
                    "time_position": state[3],
                    "last_contact": state[4],
                    "longitude": state[5],
                    "latitude": state[6],
                    "baro_altitude": state[7],  # in meters, can be None
                    "on_ground": state[8],
                    "velocity": state[9],  # in m/s, can be None
                    "true_track": state[10],  # in degrees, can be None
                    "vertical_rate": state[11],  # in m/s, can be None
                    "sensors": state[12],
                    "geo_altitude": state[13],  # in meters, can be None
                    "squawk": state[14],
                    "spi": state[15],
                    "position_source": state[16]
                }
                formatted_states.append(formatted_state)
        
        return formatted_states

    def get_aircraft_position(self, icao24: str) -> Dict[str, Any]:
        """
        Convenient method to get the current position of a specific aircraft.
        
        Args:
            icao24: ICAO24 transponder address (hex string)
            
        Returns:
            Position and flight information for the aircraft
        """
        states = self.get_states(icao24=[icao24])
        formatted_states = self.format_state_vectors(states)
        
        if formatted_states:
            return formatted_states[0]
        else:
            return {"error": f"No position data found for aircraft {icao24}"}
    
    def get_traffic_around_point(self, 
                               lat: float, 
                               lon: float, 
                               radius_km: float = 50.0) -> List[Dict[str, Any]]:
        """
        Get all aircraft within a certain radius of a point.
        
        Args:
            lat: Latitude of the center point
            lon: Longitude of the center point
            radius_km: Radius in kilometers
            
        Returns:
            List of aircraft within the radius
        """
        # TODO 
        # Convert radius to approximate bounding box
        # This is a simple approximation - 1 degree is roughly 111km at the equator
        # For more accurate results, a proper geospatial calculation would be needed
        deg_radius = radius_km / 111.0
        
        bbox = (
            lat - deg_radius,  # min_lat
            lon - deg_radius,  # min_lon
            lat + deg_radius,  # max_lat
            lon + deg_radius   # max_lon
        )
        
        states = self.get_states(bbox=bbox)
        formatted_states = self.format_state_vectors(states)
        
        # Additional filtering could be done here to filter by exact distance
        # from the center point if needed
        
        return formatted_states