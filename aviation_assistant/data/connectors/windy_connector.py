import requests
import logging
import math
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta

class WindyConnector:
    """
    Connector for Windy API.
    
    Provides access to detailed weather forecasts, wind patterns,
    and aviation-specific meteorological data.
    """
    
    # Weather models available in Windy API
    AVAILABLE_MODELS = {
        "gfs": "Global Forecast System (GFS)",
        "ecmwf": "European Centre for Medium-Range Weather Forecasts (ECMWF)",
        "iconEu": "ICON European Model",
        "namConus": "North American Mesoscale for CONUS",
        "arome": "Météo-France AROME",
        "iconGlobal": "ICON Global Model",
        "gfsWave": "GFS Wave Model",
        "wavewatch": "Wave Watch III Model",
        "gem": "Global Environmental Multiscale Model"
    }
    
    # Available parameters in Windy API
    AVAILABLE_PARAMETERS = {
        # Basic parameters
        "wind": "Wind components (u,v)",
        "wind_u": "Wind U component (west<0, east>0)",
        "wind_v": "Wind V component (south<0, north>0)",
        "temp": "Temperature",
        "dewpoint": "Dew point temperature",
        "rh": "Relative humidity",
        "pressure": "Pressure",
        "precip": "Precipitation",
        "convPrecip": "Convective precipitation",
        
        # Cloud parameters
        "clouds": "Total cloud cover",
        "lclouds": "Low cloud cover",
        "mclouds": "Medium cloud cover",
        "hclouds": "High cloud cover",
        "cloudbase": "Height of cloud base",
        "cloudtop": "Height of cloud top",
        "visibility": "Visibility",
        "fog": "Fog",
        
        # Instability parameters 
        "cape": "Convective Available Potential Energy",
        "cin": "Convective Inhibition",
        "k_index": "K-index (thunderstorm potential)",
        
        # Wind-related parameters
        "gust": "Wind gust",
        "gustAccu": "Wind gust accumulation",
        "windThermalMin": "Minimum thermal wind",
        "windThermalMax": "Maximum thermal wind",
        
        # Wave parameters (for marine models)
        "waves": "Wave height",
        "wavePeriod": "Wave period",
        "waveDirection": "Wave direction",
        "swell1Height": "Swell 1 height",
        "swell1Period": "Swell 1 period",
        "swell1Direction": "Swell 1 direction",
        "swell2Height": "Swell 2 height",
        "swell2Period": "Swell 2 period",
        "swell2Direction": "Swell 2 direction",
        
        # Aviation-specific parameters
        "freezingLevel": "Freezing level altitude",
        "snowDepth": "Snow depth",
        "snowAccu": "Snow accumulation",
        "rainAccu": "Rain accumulation",
        "deg0": "Height of the 0°C isotherm"
    }
    
    # Available altitude levels in Windy API
    AVAILABLE_LEVELS = {
        "surface": "Surface level (2m for temperature, 10m for wind)",
        "1000h": "1000 hPa pressure level (near surface)",
        "950h": "950 hPa pressure level (~500m)",
        "925h": "925 hPa pressure level (~750m)",
        "900h": "900 hPa pressure level (~1km)",
        "850h": "850 hPa pressure level (~1.5km)",
        "800h": "800 hPa pressure level (~2km)",
        "700h": "700 hPa pressure level (~3km)",
        "600h": "600 hPa pressure level (~4km)",
        "500h": "500 hPa pressure level (~5.5km)",
        "400h": "400 hPa pressure level (~7km)",
        "300h": "300 hPa pressure level (~9km)",
        "250h": "250 hPa pressure level (~10km)",
        "200h": "200 hPa pressure level (~12km)",
        "150h": "150 hPa pressure level (~13.5km)",
        "100h": "100 hPa pressure level (~16km)",
        "70h": "70 hPa pressure level (~18km)",
        "50h": "50 hPa pressure level (~20km)"
    }
    
    # Standard pressure levels with approximate altitudes
    PRESSURE_LEVEL_ALTITUDES = {
        "1000h": 100,     # ~100m
        "950h": 500,      # ~500m
        "925h": 750,      # ~750m 
        "900h": 1000,     # ~1km
        "850h": 1500,     # ~1.5km
        "800h": 2000,     # ~2km
        "700h": 3000,     # ~3km
        "600h": 4200,     # ~4.2km
        "500h": 5500,     # ~5.5km
        "400h": 7000,     # ~7km
        "300h": 9000,     # ~9km
        "250h": 10500,    # ~10.5km
        "200h": 12000,    # ~12km
        "150h": 13500,    # ~13.5km
        "100h": 16000,    # ~16km
        "70h": 18000,     # ~18km
        "50h": 20000      # ~20km
    }
    
    # Flight level to altitude mapping (in meters)
    # FL multiplied by 30.48 to convert from hundreds of feet to meters
    FL_TO_ALTITUDE = {
        f"FL{fl}": fl * 30.48 for fl in range(10, 601, 10)  # FL10 to FL600
    }
    
    def __init__(self, api_key: str):
        """
        Initialize the Windy API connector.
        
        Args:
            api_key: Your Windy API key
        """
        self.api_key = api_key
        self.base_url = "https://api.windy.com/api/point-forecast/v2"
        self.logger = logging.getLogger(__name__)
        
        # Set up session for connection pooling
        self.session = requests.Session()
    
    def __del__(self):
        """Clean up session on object destruction."""
        if hasattr(self, 'session'):
            self.session.close()
    
    def get_forecast(self, 
                    lat: float, 
                    lon: float, 
                    model: str = "gfs", 
                    parameters: Optional[List[str]] = None,
                    levels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Retrieve detailed weather forecast for a specific location.
        
        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            model: Weather model to use (default: "gfs")
                  Available models: gfs, ecmwf, iconEu, namConus, arome, etc.
            parameters: List of weather parameters to include
                       Available parameters: wind, temp, rh, pressure, etc.
            levels: List of altitude/pressure levels
                   Available levels: surface, 1000h, 850h, 700h, etc.
            
        Returns:
            Detailed weather forecast data
        """
        # Validate inputs
        if not self._validate_coordinates(lat, lon):
            self.logger.error(f"Invalid coordinates: lat={lat}, lon={lon}")
            return {"error": "Invalid coordinates"}
        
        # Validate and set default parameters
        if parameters is None:
            parameters = ["wind", "temp", "pressure", "rh", "clouds"]
        else:
            parameters = self._validate_parameters(parameters)
        
        # Validate and set default levels
        if levels is None:
            levels = ["surface"]
        else:
            levels = self._validate_levels(levels)
            
        # Validate model
        if model not in self.AVAILABLE_MODELS:
            self.logger.warning(f"Unknown model: {model}, using 'gfs' instead")
            model = "gfs"
        
        # Construct payload
        payload = {
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "model": model,
            "parameters": parameters,
            "levels": levels,
            "key": self.api_key
        }
        
        try:
            response = self.session.post(self.base_url, json=payload, timeout=30)
            response.raise_for_status()
            forecast_data = response.json()
            
            # Check for API error messages
            if "error" in forecast_data:
                self.logger.error(f"API Error: {forecast_data['error']}")
                return {"error": forecast_data['error']}
                
            # Check for warning messages
            if "warning" in forecast_data:
                self.logger.warning(f"API Warning: {forecast_data['warning']}")
            
            # Process and format the response
            processed_data = self._process_forecast_data(forecast_data)
            return processed_data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching Windy forecast for {lat},{lon}: {str(e)}")
            return {"error": str(e)}
        except ValueError as e:
            self.logger.error(f"Error parsing response: {str(e)}")
            return {"error": f"Error parsing response: {str(e)}"}
    
    def get_route_forecast(self, 
                          route_coords: List[Tuple[float, float]], 
                          model: str = "gfs",
                          parameters: Optional[List[str]] = None,
                          levels: Optional[List[str]] = None,
                          route_segments: Optional[int] = None) -> Dict[str, Any]:
        """
        Retrieve weather forecast along a flight route.
        
        Args:
            route_coords: List of (lat, lon) coordinates defining the route
            model: Weather model to use (default: "gfs")
            parameters: List of weather parameters to include
            levels: List of altitude levels
            route_segments: Optional number of segments to divide the route into
                           If provided, will create interpolated points along the route
            
        Returns:
            Weather forecast data for each point along the route
        """
        if not route_coords or len(route_coords) < 2:
            return {"error": "At least two coordinates are required for a route"}
            
        if parameters is None:
            parameters = ["wind", "temp", "pressure", "rh", "clouds"]
            
        if levels is None:
            levels = ["surface"]
        
        # Interpolate route if route_segments is provided
        if route_segments and route_segments > len(route_coords):
            route_coords = self._interpolate_route(route_coords, route_segments)
        
        # Create a forecast for each point along the route
        forecasts = {}
        for i, (lat, lon) in enumerate(route_coords):
            point_name = f"point_{i}"
            forecast = self.get_forecast(lat, lon, model, parameters, levels)
            forecasts[point_name] = forecast
            
            # Add coordinate information to each point
            if "error" not in forecast:
                forecasts[point_name]["coordinates"] = {
                    "lat": lat,
                    "lon": lon
                }
        
        # Add route metadata
        route_metadata = {
            "route_length_km": self._calculate_route_length(route_coords),
            "point_count": len(route_coords),
            "model": model,
            "parameters": parameters,
            "levels": levels
        }
        
        return {
            "metadata": route_metadata,
            "forecasts": forecasts
        }
    
    def get_aviation_weather(self,
                           lat: float,
                           lon: float,
                           flight_levels: Union[List[str], str],
                           model: str = "gfs") -> Dict[str, Any]:
        """
        Retrieve aviation-specific weather forecast for a location.
        
        Includes data relevant for pilots like wind at various altitudes,
        turbulence, icing conditions, and visibility information.
        
        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            flight_levels: Flight level(s) (e.g., "FL100", "FL180", "FL350") or list of flight levels
            model: Weather model to use (default: "gfs")
            
        Returns:
            Aviation weather forecast
        """
        # Handle single flight level or list
        if isinstance(flight_levels, str):
            flight_levels = [flight_levels]
        
        # Convert flight levels to pressure levels
        levels = ["surface"]
        flight_level_mappings = {}
        
        for fl in flight_levels:
            if not fl.startswith("FL"):
                self.logger.warning(f"Invalid flight level format: {fl}. Expected format: FL followed by a number (e.g., FL350)")
                continue
                
            # Exact conversion from FL to pressure level
            pressure_level = self._flight_level_to_pressure_level(fl)
            levels.append(pressure_level)
            flight_level_mappings[pressure_level] = fl
        
        # Aviation-specific parameters
        parameters = [
            "wind",         # Wind (u,v components)
            "temp",         # Temperature
            "rh",           # Relative humidity
            "pressure",     # Pressure
            "clouds",       # Cloud cover
            "lclouds",      # Low cloud cover
            "mclouds",      # Medium cloud cover
            "hclouds",      # High cloud cover
            "cape",         # Convective Available Potential Energy (thunderstorm potential)
            "dewpoint",     # Dewpoint
            "gust",         # Wind gust
            "visibility",   # Visibility
            "freezingLevel" # Freezing level altitude
        ]
        
        # Get the forecast
        forecast = self.get_forecast(lat, lon, model, parameters, levels)
        
        if "error" in forecast:
            return forecast
        
        # Post-process for aviation-specific information
        aviation_data = self._process_aviation_data(forecast, flight_level_mappings)
        
        # Add derived aviation metrics
        aviation_data["derived"] = {
            "turbulence": self._calculate_turbulence(forecast),
            "icing_risk": self._calculate_icing_risk(forecast),
            "wind_shear": self._calculate_wind_shear(forecast),
            "crosswind_component": self._calculate_crosswind(forecast, 0),  # Default runway heading of 0
            "thermal_updrafts": self._estimate_thermal_updrafts(forecast)
        }
        
        # Add location information
        aviation_data["location"] = {
            "lat": lat,
            "lon": lon
        }
        
        return aviation_data
    
    def get_airport_weather(self,
                           icao_code: str,
                           model: str = "gfs") -> Dict[str, Any]:
        """
        Retrieve weather forecast for an airport using its ICAO code.
        
        This method requires airport coordinates to be known or retrievable.
        Since this connector doesn't have a built-in airport database,
        it uses a hardcoded mapping for demonstration purposes.
        
        Args:
            icao_code: ICAO airport code (e.g., "KJFK", "EGLL")
            model: Weather model to use (default: "gfs")
            
        Returns:
            Weather forecast for the airport
        """
        # This is a minimal subset of major airports for demonstration
        # In a real implementation, you would use a proper airport database
        airport_coords = {
            "KJFK": (40.6413, -73.7781),  # New York JFK
            "KLAX": (33.9416, -118.4085), # Los Angeles
            "EGLL": (51.4700, -0.4543),   # London Heathrow
            "LFPG": (49.0097, 2.5479),    # Paris Charles de Gaulle
            "EDDF": (50.0379, 8.5622),    # Frankfurt
            "RJAA": (35.7647, 140.3864),  # Tokyo Narita
            "VHHH": (22.3080, 113.9185),  # Hong Kong
            "YSSY": (-33.9399, 151.1753)  # Sydney
        }
        
        icao_code = icao_code.upper()
        if icao_code not in airport_coords:
            return {"error": f"Airport coordinates not found for ICAO code: {icao_code}"}
        
        lat, lon = airport_coords[icao_code]
        
        # Get surface weather for the airport
        surface_params = [
            "wind", "temp", "pressure", "rh", "clouds", 
            "visibility", "gust", "dewpoint", "precip"
        ]
        
        surface_forecast = self.get_forecast(lat, lon, model, surface_params, ["surface"])
        
        # Get weather at different flight levels for approach and departure
        flight_levels = ["FL050", "FL100", "FL180", "FL240"]
        aviation_forecast = self.get_aviation_weather(lat, lon, flight_levels, model)
        
        # Combine the data
        if "error" in surface_forecast or "error" in aviation_forecast:
            return {"error": "Error retrieving airport weather"}
        
        airport_weather = {
            "icao": icao_code,
            "location": {
                "lat": lat,
                "lon": lon
            },
            "surface_conditions": self._extract_surface_conditions(surface_forecast),
            "flight_level_conditions": aviation_forecast.get("flight_levels", {}),
            "derived": aviation_forecast.get("derived", {}),
            "forecast_time": surface_forecast.get("forecast_time", []),
            "raw_data": {
                "surface": surface_forecast,
                "aviation": aviation_forecast
            }
        }
        
        return airport_weather
    
    def get_significant_weather(self,
                               lat: float,
                               lon: float,
                               radius_km: float = 100.0,
                               model: str = "gfs") -> Dict[str, Any]:
        """
        Analyze forecast to identify significant weather phenomena.
        
        Identifies thunderstorms, strong winds, heavy precipitation,
        low visibility, and other significant weather conditions.
        
        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            radius_km: Radius to analyze in kilometers
            model: Weather model to use (default: "gfs")
            
        Returns:
            Dictionary of significant weather phenomena
        """
        # Generate points in a grid around the center point
        grid_points = self._generate_grid_points(lat, lon, radius_km)
        
        # Parameters to check for significant weather
        parameters = [
            "wind", "gust", "cape", "precip", "visibility",
            "clouds", "temp", "rh", "pressure"
        ]
        
        # Get forecasts for center point with all parameters
        center_forecast = self.get_forecast(lat, lon, model, parameters, ["surface"])
        
        if "error" in center_forecast:
            return center_forecast
        
        # Sample a few points around to identify spatially significant weather
        sample_forecasts = []
        for i, (sample_lat, sample_lon) in enumerate(grid_points[:4]):  # Sample only 4 points
            forecast = self.get_forecast(sample_lat, sample_lon, model, parameters, ["surface"])
            if "error" not in forecast:
                sample_forecasts.append(forecast)
        
        # Analyze for significant weather
        significant_weather = self._identify_significant_weather(center_forecast, sample_forecasts)
        
        return {
            "center_point": {"lat": lat, "lon": lon},
            "radius_km": radius_km,
            "significant_weather": significant_weather,
            "forecast_times": center_forecast.get("forecast_time", []),
            "data_source": f"Windy API ({model} model)"
        }
    
    def get_turbulence_forecast(self,
                               lat: float,
                               lon: float,
                               flight_level: str,
                               model: str = "gfs") -> Dict[str, Any]:
        """
        Retrieve turbulence forecast for a specific location and flight level.
        
        Uses multiple parameters to estimate turbulence potential more accurately.
        
        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            flight_level: Flight level (e.g., "FL350")
            model: Weather model to use (default: "gfs")
            
        Returns:
            Turbulence forecast data with confidence levels
        """
        # Convert flight level to pressure level
        pressure_level = self._flight_level_to_pressure_level(flight_level)
        
        # Get adjacent pressure levels for wind shear calculation
        levels = [pressure_level]
        adjacent_level = self._get_adjacent_pressure_level(pressure_level, higher=True)
        if adjacent_level:
            levels.append(adjacent_level)
        
        # Parameters needed for turbulence calculation
        parameters = [
            "wind",      # Wind components - essential for wind shear
            "temp",      # Temperature - for lapse rates and stability
            "rh",        # Relative humidity - affects cloud formation
            "cape",      # CAPE - convective potential
            "cin",       # CIN - convective inhibition
            "pressure"   # Pressure - for pressure gradients
        ]
        
        # Get the forecast
        forecast = self.get_forecast(lat, lon, model, parameters, levels)
        
        if "error" in forecast:
            return forecast
        
        # Perform comprehensive turbulence calculation
        turbulence_data = self._calculate_turbulence_comprehensive(forecast, flight_level, pressure_level)
        
        # Add location and flight level information
        turbulence_data["location"] = {
            "lat": lat,
            "lon": lon
        }
        turbulence_data["flight_level"] = flight_level
        turbulence_data["pressure_level"] = pressure_level
        
        return turbulence_data
    
    def get_icing_forecast(self,
                          lat: float,
                          lon: float,
                          flight_level: str,
                          model: str = "gfs") -> Dict[str, Any]:
        """
        Retrieve aircraft icing forecast for a specific location and flight level.
        
        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            flight_level: Flight level (e.g., "FL180")
            model: Weather model to use (default: "gfs")
            
        Returns:
            Icing forecast data with risk levels
        """
        # Convert flight level to pressure level
        pressure_level = self._flight_level_to_pressure_level(flight_level)
        
        # Parameters needed for icing calculation
        parameters = [
            "temp",      # Temperature - essential for icing
            "rh",        # Relative humidity - moisture content
            "clouds",    # Cloud coverage - liquid water content proxy
            "dewpoint",  # Dewpoint - saturation indicator
            "pressure"   # Pressure
        ]
        
        # Get the forecast
        forecast = self.get_forecast(lat, lon, model, parameters, [pressure_level])
        
        # Also get freezing level information
        freezing_params = ["temp", "freezingLevel"]
        freezing_forecast = self.get_forecast(lat, lon, model, freezing_params, ["surface"])
        
        if "error" in forecast or "error" in freezing_forecast:
            return {"error": "Error retrieving icing forecast data"}
        
        # Calculate icing potential
        icing_data = self._calculate_icing_potential(forecast, freezing_forecast, flight_level, pressure_level)
        
        # Add location and flight level information
        icing_data["location"] = {
            "lat": lat,
            "lon": lon
        }
        icing_data["flight_level"] = flight_level
        icing_data["pressure_level"] = pressure_level
        
        return icing_data
    
    def get_wind_data(self,
                     lat: float,
                     lon: float,
                     flight_levels: List[str],
                     model: str = "gfs") -> Dict[str, Any]:
        """
        Retrieve detailed wind data for multiple flight levels.
        
        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            flight_levels: List of flight levels (e.g., ["FL100", "FL200", "FL300"])
            model: Weather model to use (default: "gfs")
            
        Returns:
            Detailed wind data for each requested flight level
        """
        # Convert flight levels to pressure levels
        pressure_levels = []
        level_mapping = {}
        
        for fl in flight_levels:
            if not fl.startswith("FL"):
                continue
                
            level = self._flight_level_to_pressure_level(fl)
            pressure_levels.append(level)
            level_mapping[level] = fl
        
        # Always include surface level
        if "surface" not in pressure_levels:
            pressure_levels.append("surface")
            level_mapping["surface"] = "Surface"
        
        # Get wind data
        parameters = ["wind", "wind_u", "wind_v", "gust"]
        forecast = self.get_forecast(lat, lon, model, parameters, pressure_levels)
        
        if "error" in forecast:
            return forecast
        
        # Process and format wind data
        wind_data = {
            "location": {"lat": lat, "lon": lon},
            "forecast_time": forecast.get("forecast_time", []),
            "levels": {}
        }
        
        # Extract wind data for each level
        for level in pressure_levels:
            fl_name = level_mapping.get(level, level)
            
            # Find wind components for this level
            u_key = next((k for k in forecast.keys() if k.startswith("wind_u-") and k.endswith(level)), None)
            v_key = next((k for k in forecast.keys() if k.startswith("wind_v-") and k.endswith(level)), None)
            
            if u_key and v_key and u_key in forecast and v_key in forecast:
                u_values = forecast[u_key]
                v_values = forecast[v_key]
                
                # Calculate wind speed and direction
                wind_speeds = []
                wind_directions = []
                
                for u, v in zip(u_values, v_values):
                    if u is not None and v is not None:
                        wind_speed = math.sqrt(u**2 + v**2)
                        # Calculate meteorological wind direction (from where the wind is blowing)
                        wind_dir = (270 - math.degrees(math.atan2(v, u))) % 360
                        wind_speeds.append(round(wind_speed, 2))
                        wind_directions.append(round(wind_dir, 1))
                    else:
                        wind_speeds.append(None)
                        wind_directions.append(None)
                
                wind_data["levels"][fl_name] = {
                    "wind_speed": wind_speeds,                 # in m/s
                    "wind_direction": wind_directions,         # in degrees
                    "u_component": u_values,                   # in m/s
                    "v_component": v_values,                   # in m/s
                    "speed_knots": [round(ws * 1.94384, 1) if ws is not None else None for ws in wind_speeds]  # Convert to knots
                }
                
                # Add gust data for surface level
                if level == "surface":
                    gust_key = next((k for k in forecast.keys() if k.startswith("gust-")), None)
                    if gust_key and gust_key in forecast:
                        wind_data["levels"][fl_name]["gust"] = forecast[gust_key]  # in m/s
                        wind_data["levels"][fl_name]["gust_knots"] = [round(g * 1.94384, 1) if g is not None else None for g in forecast[gust_key]]  # Convert to knots
        
        return wind_data
    
    def get_route_winds(self,
                       route_coords: List[Tuple[float, float]],
                       flight_level: str,
                       model: str = "gfs",
                       route_segments: int = 10) -> Dict[str, Any]:
        """
        Calculate wind components along a route at a specific flight level.
        
        Args:
            route_coords: List of (lat, lon) coordinates defining the route
            flight_level: Flight level (e.g., "FL350")
            model: Weather model to use (default: "gfs")
            route_segments: Number of segments to divide the route into
            
        Returns:
            Wind components and headings along the route
        """
        if not route_coords or len(route_coords) < 2:
            return {"error": "At least two coordinates are required for a route"}
            
        # Convert flight level to pressure level
        pressure_level = self._flight_level_to_pressure_level(flight_level)
        
        # Interpolate route
        if route_segments > 0:
            route_points = self._interpolate_route(route_coords, route_segments)
        else:
            route_points = route_coords
        
        # Calculate route bearings (true course) between consecutive points
        bearings = []
        for i in range(len(route_points) - 1):
            lat1, lon1 = route_points[i]
            lat2, lon2 = route_points[i + 1]
            bearing = self._calculate_bearing(lat1, lon1, lat2, lon2)
            bearings.append(bearing)
        # Add the last bearing again for consistency with points count
        if bearings:
            bearings.append(bearings[-1])
        
        # Get wind data for each point
        parameters = ["wind_u", "wind_v"]
        wind_data = []
        
        for i, (lat, lon) in enumerate(route_points):
            forecast = self.get_forecast(lat, lon, model, parameters, [pressure_level])
            
            if "error" in forecast:
                wind_data.append({"error": forecast["error"]})
                continue
                
            # Get wind components
            u_key = next((k for k in forecast.keys() if k.startswith("wind_u-")), None)
            v_key = next((k for k in forecast.keys() if k.startswith("wind_v-")), None)
            
            if not u_key or not v_key or u_key not in forecast or v_key not in forecast:
                wind_data.append({"error": "Wind data not available"})
                continue
                
            u_values = forecast[u_key]
            v_values = forecast[v_key]
            
            # Calculate wind speed, direction, and crosswind component
            point_wind_data = {
                "position": i,
                "coordinates": {"lat": lat, "lon": lon},
                "u_component": u_values,
                "v_component": v_values,
                "route_bearing": bearings[i] if i < len(bearings) else None,
                "wind_speed": [],
                "wind_direction": [],
                "headwind_component": [],
                "crosswind_component": [],
                "forecast_time": forecast.get("forecast_time", [])
            }
            
            bearing = bearings[i] if i < len(bearings) else 0
            
            # Calculate derived wind data for each timestamp
            for u, v in zip(u_values, v_values):
                if u is not None and v is not None:
                    wind_speed = math.sqrt(u**2 + v**2)
                    wind_dir = (270 - math.degrees(math.atan2(v, u))) % 360
                    
                    # Calculate headwind and crosswind components
                    angle = abs(((bearing - wind_dir + 180) % 360) - 180)
                    headwind = wind_speed * math.cos(math.radians(angle))
                    crosswind = wind_speed * math.sin(math.radians(angle))
                    
                    point_wind_data["wind_speed"].append(round(wind_speed, 2))
                    point_wind_data["wind_direction"].append(round(wind_dir, 1))
                    point_wind_data["headwind_component"].append(round(headwind, 2))
                    point_wind_data["crosswind_component"].append(round(crosswind, 2))
                else:
                    point_wind_data["wind_speed"].append(None)
                    point_wind_data["wind_direction"].append(None)
                    point_wind_data["headwind_component"].append(None)
                    point_wind_data["crosswind_component"].append(None)
            
            wind_data.append(point_wind_data)
        
        # Calculate route metadata
        route_length = self._calculate_route_length(route_points)
        
        return {
            "flight_level": flight_level,
            "pressure_level": pressure_level,
            "route_length_km": route_length,
            "point_count": len(route_points),
            "points": wind_data
        }
    
    def get_visualization_url(self,
                            lat: float,
                            lon: float,
                            zoom: int = 7,
                            overlay: str = "wind",
                            level: str = "surface") -> str:
        """
        Generate a URL for Windy.com visualization.
        
        Creates a link to Windy.com's web interface with specified parameters
        for visualization of weather data.
        
        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            zoom: Zoom level (1-18)
            overlay: Visualization overlay (wind, temp, clouds, etc.)
            level: Altitude level (surface, 850h, 500h, etc.)
            
        Returns:
            URL to Windy.com with specified parameters
        """
        # Validate parameters
        zoom = max(1, min(18, zoom))  # Ensure zoom is between 1 and 18
        
        # Map certain parameters to their Windy web equivalent
        overlay_map = {
            "wind": "wind",
            "temp": "temperature",
            "rh": "rh",
            "pressure": "pressure",
            "clouds": "clouds",
            "precip": "rain",
            "cape": "capes",
            "gust": "gust",
            "visibility": "visibility"
        }
        
        overlay_param = overlay_map.get(overlay, overlay)
        
        # Map pressure levels to Windy web equivalent
        level_map = {
            "surface": "surface",
            "1000h": "1000",
            "925h": "925",
            "850h": "850",
            "700h": "700",
            "500h": "500",
            "300h": "300",
            "250h": "250",
            "200h": "200",
            "150h": "150"
        }
        
        level_param = level_map.get(level, level.replace("h", ""))
        
        # Construct the URL
        url = (
            f"https://www.windy.com/?"
            f"{overlay_param},{lat},{lon},{zoom}"
            f",i:{level_param}"
        )
        
        return url
    
    def _validate_coordinates(self, lat: float, lon: float) -> bool:
        """Validate latitude and longitude coordinates."""
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            return False
            
        if lat < -90 or lat > 90:
            return False
            
        if lon < -180 or lon > 180:
            return False
            
        return True
    
    def _validate_parameters(self, parameters: List[str]) -> List[str]:
        """Validate and filter parameter list."""
        valid_params = []
        
        for param in parameters:
            if param in self.AVAILABLE_PARAMETERS:
                valid_params.append(param)
            else:
                self.logger.warning(f"Unknown parameter: {param}")
        
        if not valid_params:
            self.logger.warning("No valid parameters provided, using defaults")
            return ["wind", "temp", "pressure", "rh", "clouds"]
            
        return valid_params
    
    def _validate_levels(self, levels: List[str]) -> List[str]:
        """Validate and filter level list."""
        valid_levels = []
        
        for level in levels:
            if level in self.AVAILABLE_LEVELS or (level.endswith("h") and level[:-1].isdigit()):
                valid_levels.append(level)
            else:
                self.logger.warning(f"Unknown level: {level}")
        
        if not valid_levels:
            self.logger.warning("No valid levels provided, using surface level")
            return ["surface"]
            
        return valid_levels
    
    def _process_forecast_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and format the raw forecast data from the API."""
        # Extract timestamp array
        timestamps = raw_data.get("ts", [])
        
        # Convert timestamps from milliseconds to datetime objects
        forecast_time = []
        for ts in timestamps:
            try:
                dt = datetime.fromtimestamp(ts / 1000)
                forecast_time.append(dt)
            except (ValueError, TypeError):
                forecast_time.append(None)
        
        # Create processed data dictionary
        processed_data = {
            "forecast_time": forecast_time,
            "units": raw_data.get("units", {})
        }
        
        # Copy all parameter data
        for key, value in raw_data.items():
            if key not in ["ts", "units", "warning"]:
                processed_data[key] = value
        
        return processed_data
    
    def _flight_level_to_pressure_level(self, flight_level: str) -> str:
        """
        Convert flight level to corresponding pressure level.
        
        Uses standard atmosphere approximation.
        
        Args:
            flight_level: Flight level string (e.g., "FL350")
            
        Returns:
            Corresponding pressure level (e.g., "250h")
        """
        if not flight_level.startswith("FL"):
            return "surface"
            
        try:
            fl_num = int(flight_level[2:])
        except ValueError:
            return "surface"
        
        altitude_meters = fl_num * 30.48  # Convert FL to meters
        
        # Find the closest pressure level based on altitude
        closest_level = "surface"
        min_diff = float('inf')
        
        for level, alt in self.PRESSURE_LEVEL_ALTITUDES.items():
            diff = abs(altitude_meters - alt)
            if diff < min_diff:
                min_diff = diff
                closest_level = level
        
        return closest_level
    
    def _get_adjacent_pressure_level(self, pressure_level: str, higher: bool = True) -> Optional[str]:
        """
        Get the adjacent pressure level (higher or lower).
        
        Args:
            pressure_level: Current pressure level (e.g., "500h")
            higher: If True, get higher altitude (lower pressure); if False, get lower altitude (higher pressure)
            
        Returns:
            Adjacent pressure level or None if not found
        """
        if pressure_level == "surface":
            if higher:
                return "925h"
            else:
                return None
        
        # Standard pressure levels in order from lowest to highest altitude
        standard_levels = [
            "surface", "1000h", "950h", "925h", "900h", "850h", "800h", 
            "700h", "600h", "500h", "400h", "300h", "250h", "200h", 
            "150h", "100h", "70h", "50h"
        ]
        
        try:
            idx = standard_levels.index(pressure_level)
            if higher and idx < len(standard_levels) - 1:
                return standard_levels[idx + 1]
            elif not higher and idx > 0:
                return standard_levels[idx - 1]
        except ValueError:
            pass
            
        return None
    
    def _interpolate_route(self, route_coords: List[Tuple[float, float]], num_points: int) -> List[Tuple[float, float]]:
        """
        Interpolate points along a route.
        
        Args:
            route_coords: List of (lat, lon) coordinate tuples
            num_points: Total number of points to generate
            
        Returns:
            List of interpolated coordinate tuples
        """
        if len(route_coords) < 2:
            return route_coords
            
        # If we already have enough points, return the original
        if len(route_coords) >= num_points:
            return route_coords
            
        # Calculate the total route distance
        total_distance = self._calculate_route_length(route_coords)
        
        # Distances between each pair of consecutive points
        segment_distances = []
        for i in range(len(route_coords) - 1):
            lat1, lon1 = route_coords[i]
            lat2, lon2 = route_coords[i + 1]
            distance = self._haversine_distance(lat1, lon1, lat2, lon2)
            segment_distances.append(distance)
        
        # Calculate cumulative distances
        cum_distances = [0]
        for distance in segment_distances:
            cum_distances.append(cum_distances[-1] + distance)
        
        # Generate evenly spaced points
        interpolated_coords = []
        for i in range(num_points):
            # Calculate the target distance
            target_distance = i * total_distance / (num_points - 1) if num_points > 1 else 0
            
            # Find which segment this point belongs to
            segment_idx = 0
            while segment_idx < len(cum_distances) - 1 and cum_distances[segment_idx + 1] < target_distance:
                segment_idx += 1
                
            if segment_idx >= len(route_coords) - 1:
                # Add the last point
                interpolated_coords.append(route_coords[-1])
                continue
                
            # Calculate interpolation ratio within the segment
            segment_start_dist = cum_distances[segment_idx]
            segment_end_dist = cum_distances[segment_idx + 1]
            segment_length = segment_end_dist - segment_start_dist
            
            if segment_length > 0:
                ratio = (target_distance - segment_start_dist) / segment_length
            else:
                ratio = 0
                
            # Interpolate coordinates
            lat1, lon1 = route_coords[segment_idx]
            lat2, lon2 = route_coords[segment_idx + 1]
            
            lat = lat1 + ratio * (lat2 - lat1)
            lon = lon1 + ratio * (lon2 - lon1)
            
            interpolated_coords.append((lat, lon))
        
        return interpolated_coords
    
    def _calculate_route_length(self, route_coords: List[Tuple[float, float]]) -> float:
        """
        Calculate the total length of a route in kilometers.
        
        Args:
            route_coords: List of (lat, lon) coordinate tuples
            
        Returns:
            Route length in kilometers
        """
        if len(route_coords) < 2:
            return 0
            
        total_distance = 0
        for i in range(len(route_coords) - 1):
            lat1, lon1 = route_coords[i]
            lat2, lon2 = route_coords[i + 1]
            distance = self._haversine_distance(lat1, lon1, lat2, lon2)
            total_distance += distance
            
        return total_distance
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two points using Haversine formula.
        
        Args:
            lat1, lon1: Coordinates of first point in decimal degrees
            lat2, lon2: Coordinates of second point in decimal degrees
            
        Returns:
            Distance in kilometers
        """
        # Convert decimal degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        # Earth radius in kilometers
        earth_radius = 6371.0
        
        return earth_radius * c
    
    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate initial bearing between two points.
        
        Args:
            lat1, lon1: Coordinates of first point in decimal degrees
            lat2, lon2: Coordinates of second point in decimal degrees
            
        Returns:
            Initial bearing in degrees (0-360)
        """
        # Convert decimal degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Calculate bearing
        y = math.sin(lon2_rad - lon1_rad) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(lon2_rad - lon1_rad)
        bearing_rad = math.atan2(y, x)
        
        # Convert to degrees
        bearing_deg = math.degrees(bearing_rad)
        
        # Normalize to 0-360
        return (bearing_deg + 360) % 360
    
    def _process_aviation_data(self, forecast: Dict[str, Any], flight_level_mappings: Dict[str, str]) -> Dict[str, Any]:
        """
        Process forecast data for aviation use.
        
        Args:
            forecast: Raw forecast data
            flight_level_mappings: Mapping from pressure levels to flight levels
            
        Returns:
            Processed aviation data
        """
        aviation_data = {
            "forecast_time": forecast["forecast_time"],
            "surface_conditions": self._extract_surface_conditions(forecast),
            "flight_levels": {}
        }
        
        # Process each flight level
        for pressure_level, flight_level in flight_level_mappings.items():
            level_data = {}
            
            # Extract wind data
            u_key = next((k for k in forecast.keys() if k.startswith("wind_u-") and k.endswith(pressure_level)), None)
            v_key = next((k for k in forecast.keys() if k.startswith("wind_v-") and k.endswith(pressure_level)), None)
            
            if u_key and v_key and u_key in forecast and v_key in forecast:
                u_values = forecast[u_key]
                v_values = forecast[v_key]
                
                # Calculate wind speed and direction
                wind_speeds = []
                wind_directions = []
                
                for u, v in zip(u_values, v_values):
                    if u is not None and v is not None:
                        speed = math.sqrt(u**2 + v**2)
                        direction = (270 - math.degrees(math.atan2(v, u))) % 360
                        wind_speeds.append(round(speed, 2))
                        wind_directions.append(round(direction, 1))
                    else:
                        wind_speeds.append(None)
                        wind_directions.append(None)
                
                level_data["wind_speed"] = wind_speeds
                level_data["wind_direction"] = wind_directions
                level_data["wind_speed_knots"] = [round(s * 1.94384, 1) if s is not None else None for s in wind_speeds]
            
            # Extract temperature data
            temp_key = next((k for k in forecast.keys() if k.startswith("temp-") and k.endswith(pressure_level)), None)
            if temp_key and temp_key in forecast:
                # Convert from Kelvin to Celsius
                level_data["temperature_c"] = [round(t - 273.15, 1) if t is not None else None for t in forecast[temp_key]]
            
            # Extract relative humidity data
            rh_key = next((k for k in forecast.keys() if k.startswith("rh-") and k.endswith(pressure_level)), None)
            if rh_key and rh_key in forecast:
                level_data["relative_humidity"] = forecast[rh_key]
            
            # Add the level data to the result
            aviation_data["flight_levels"][flight_level] = level_data
        
        return aviation_data
    
    def _extract_surface_conditions(self, forecast: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract surface weather conditions from forecast data.
        
        Args:
            forecast: Forecast data
            
        Returns:
            Dictionary of surface weather conditions
        """
        surface_data = {}
        
        # Surface parameters to extract
        params = {
            "wind": ("wind_speed", "wind_direction"),
            "temp": "temperature",
            "pressure": "pressure",
            "rh": "relative_humidity",
            "clouds": "cloud_cover",
            "visibility": "visibility",
            "gust": "wind_gust",
            "dewpoint": "dewpoint",
            "precip": "precipitation"
        }
        
        # Extract each parameter
        for api_param, output_name in params.items():
            # Find the key for this parameter at surface level
            if isinstance(output_name, tuple):
                # Handle special case for wind
                u_key = next((k for k in forecast.keys() if k.startswith("wind_u-surface")), None)
                v_key = next((k for k in forecast.keys() if k.startswith("wind_v-surface")), None)
                
                if u_key and v_key and u_key in forecast and v_key in forecast:
                    u_values = forecast[u_key]
                    v_values = forecast[v_key]
                    
                    # Calculate wind speed and direction
                    wind_speeds = []
                    wind_directions = []
                    
                    for u, v in zip(u_values, v_values):
                        if u is not None and v is not None:
                            speed = math.sqrt(u**2 + v**2)
                            direction = (270 - math.degrees(math.atan2(v, u))) % 360
                            wind_speeds.append(round(speed, 2))
                            wind_directions.append(round(direction, 1))
                        else:
                            wind_speeds.append(None)
                            wind_directions.append(None)
                    
                    surface_data["wind_speed"] = wind_speeds
                    surface_data["wind_direction"] = wind_directions
                    surface_data["wind_speed_knots"] = [round(s * 1.94384, 1) if s is not None else None for s in wind_speeds]
            else:
                # Handle regular parameters
                param_key = next((k for k in forecast.keys() if k.startswith(f"{api_param}-surface")), None)
                
                if param_key and param_key in forecast:
                    values = forecast[param_key]
                    
                    # Process specific parameters
                    if output_name == "temperature" and values:
                        # Convert from Kelvin to Celsius
                        surface_data["temperature_c"] = [round(t - 273.15, 1) if t is not None else None for t in values]
                        # Also add Fahrenheit
                        surface_data["temperature_f"] = [round((t - 273.15) * 9/5 + 32, 1) if t is not None else None for t in values]
                    elif output_name == "dewpoint" and values:
                        # Convert from Kelvin to Celsius
                        surface_data["dewpoint_c"] = [round(t - 273.15, 1) if t is not None else None for t in values]
                    else:
                        surface_data[output_name] = values
        
        # Add visibility in miles if available
        if "visibility" in surface_data:
            # Convert meters to miles
            surface_data["visibility_miles"] = [round(v * 0.000621371, 1) if v is not None else None for v in surface_data["visibility"]]
        
        return surface_data
    
    def _calculate_turbulence(self, forecast: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate turbulence based on forecast data.
        
        This is a basic implementation using CAPE as a primary indicator.
        A more comprehensive implementation would use multiple parameters
        and vertical profiles.
        
        Args:
            forecast: Forecast data
            
        Returns:
            Dictionary with turbulence levels and confidence
        """
        turbulence_data = {
            "level": [],
            "confidence": [],
            "cape_value": []
        }
        
        # Get CAPE values if available
        cape_key = next((k for k in forecast.keys() if k.startswith("cape-")), None)
        
        if not cape_key or cape_key not in forecast:
            return turbulence_data
            
        cape_values = forecast[cape_key]
        
        # Evaluate turbulence based on CAPE
        for cape in cape_values:
            if cape is None:
                turbulence_data["level"].append(None)
                turbulence_data["confidence"].append(None)
                turbulence_data["cape_value"].append(None)
            else:
                # Store CAPE value
                turbulence_data["cape_value"].append(cape)
                
                # Determine turbulence level based on CAPE
                if cape < 300:
                    turbulence_data["level"].append("light")
                    turbulence_data["confidence"].append(0.6)
                elif cape < 1000:
                    turbulence_data["level"].append("moderate")
                    turbulence_data["confidence"].append(0.7)
                elif cape < 2000:
                    turbulence_data["level"].append("moderate-severe")
                    turbulence_data["confidence"].append(0.8)
                else:
                    turbulence_data["level"].append("severe")
                    turbulence_data["confidence"].append(0.9)
        
        return turbulence_data
    
    def _calculate_turbulence_comprehensive(self, forecast: Dict[str, Any], flight_level: str, pressure_level: str) -> Dict[str, Any]:
        """
        Calculate turbulence using a comprehensive approach with multiple indicators.
        
        Args:
            forecast: Forecast data
            flight_level: Flight level (e.g., "FL350")
            pressure_level: Pressure level (e.g., "250h")
            
        Returns:
            Detailed turbulence analysis
        """
        turbulence_data = {
            "level": [],
            "confidence": [],
            "contributors": {
                "cape": [],
                "wind_shear": [],
                "vertical_velocity": [],
                "mountain_wave": []
            },
            "forecast_time": forecast.get("forecast_time", [])
        }
        
        # Get adjacent pressure level for vertical calculations
        adjacent_level = self._get_adjacent_pressure_level(pressure_level, higher=True)
        
        # Get important parameters for turbulence calculation
        cape_key = next((k for k in forecast.keys() if k.startswith("cape-")), None)
        u_key = next((k for k in forecast.keys() if k.startswith("wind_u-") and k.endswith(pressure_level)), None)
        v_key = next((k for k in forecast.keys() if k.startswith("wind_v-") and k.endswith(pressure_level)), None)
        
        # Adjacent level keys for wind shear calculation
        adj_u_key = None
        adj_v_key = None
        if adjacent_level:
            adj_u_key = next((k for k in forecast.keys() if k.startswith("wind_u-") and k.endswith(adjacent_level)), None)
            adj_v_key = next((k for k in forecast.keys() if k.startswith("wind_v-") and k.endswith(adjacent_level)), None)
        
        # Process each timestamp
        num_timestamps = len(forecast.get("forecast_time", []))
        
        for i in range(num_timestamps):
            # Initialize contributors for this timestamp
            contributors = {
                "cape": 0,
                "wind_shear": 0,
                "vertical_velocity": 0,
                "mountain_wave": 0
            }
            
            # 1. Evaluate CAPE contribution
            if cape_key and cape_key in forecast and i < len(forecast[cape_key]):
                cape_value = forecast[cape_key][i]
                if cape_value is not None:
                    if cape_value < 300:
                        contributors["cape"] = 0.1
                    elif cape_value < 1000:
                        contributors["cape"] = 0.3
                    elif cape_value < 2000:
                        contributors["cape"] = 0.5
                    else:
                        contributors["cape"] = 0.8
            
            # 2. Evaluate wind shear contribution
            if (u_key and v_key and adj_u_key and adj_v_key and 
                u_key in forecast and v_key in forecast and 
                adj_u_key in forecast and adj_v_key in forecast and 
                i < len(forecast[u_key]) and i < len(forecast[v_key]) and 
                i < len(forecast[adj_u_key]) and i < len(forecast[adj_v_key])):
                
                u1 = forecast[u_key][i]
                v1 = forecast[v_key][i]
                u2 = forecast[adj_u_key][i]
                v2 = forecast[adj_v_key][i]
                
                if u1 is not None and v1 is not None and u2 is not None and v2 is not None:
                    # Calculate wind vector change
                    du = u2 - u1
                    dv = v2 - v1
                    shear_magnitude = math.sqrt(du**2 + dv**2)
                    
                    # Map shear magnitude to contribution
                    if shear_magnitude < 5:
                        contributors["wind_shear"] = 0.1
                    elif shear_magnitude < 10:
                        contributors["wind_shear"] = 0.3
                    elif shear_magnitude < 15:
                        contributors["wind_shear"] = 0.5
                    else:
                        contributors["wind_shear"] = 0.8
            
            # 3. Simulate vertical velocity contribution (real implementation would use actual vertical velocity)
            # This is a placeholder for an actual vertical velocity calculation
            if cape_key and cape_key in forecast and i < len(forecast[cape_key]):
                cape_value = forecast[cape_key][i]
                if cape_value is not None:
                    # Rough estimate based on CAPE
                    contributors["vertical_velocity"] = min(0.7, cape_value / 3000)
            
            # 4. Add placeholder for mountain wave detection
            # Real implementation would use terrain data and wind direction/speed
            contributors["mountain_wave"] = 0.0
            
            # Store the contributor values for this timestamp
            for key in contributors:
                turbulence_data["contributors"][key].append(contributors[key])
            
            # Calculate overall turbulence level
            # Weighted combination of contributors
            weights = {
                "cape": 0.2,
                "wind_shear": 0.5,
                "vertical_velocity": 0.2,
                "mountain_wave": 0.1
            }
            
            weighted_sum = sum(contributors[k] * weights[k] for k in contributors)
            
            # Map weighted sum to turbulence level
            if weighted_sum < 0.2:
                turbulence_data["level"].append("none-light")
                turbulence_data["confidence"].append(0.7)
            elif weighted_sum < 0.4:
                turbulence_data["level"].append("light")
                turbulence_data["confidence"].append(0.75)
            elif weighted_sum < 0.6:
                turbulence_data["level"].append("moderate")
                turbulence_data["confidence"].append(0.8)
            elif weighted_sum < 0.75:
                turbulence_data["level"].append("moderate-severe")
                turbulence_data["confidence"].append(0.85)
            else:
                turbulence_data["level"].append("severe")
                turbulence_data["confidence"].append(0.9)
        
        return turbulence_data
    
    def _calculate_icing_risk(self, forecast: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate icing risk based on forecast data.
        
        Basic implementation using temperature and humidity.
        
        Args:
            forecast: Forecast data
            
        Returns:
            Dictionary with icing risk levels
        """
        icing_data = {
            "risk_level": [],
            "confidence": []
        }
        
        # Get temperature and humidity values
        temp_key = next((k for k in forecast.keys() if k.startswith("temp-")), None)
        rh_key = next((k for k in forecast.keys() if k.startswith("rh-")), None)
        
        if not temp_key or not rh_key or temp_key not in forecast or rh_key not in forecast:
            return icing_data
            
        temp_values = forecast[temp_key]
        rh_values = forecast[rh_key]
        
        # Check each timestamp
        for i in range(min(len(temp_values), len(rh_values))):
            temp = temp_values[i]
            rh = rh_values[i]
            
            if temp is None or rh is None:
                icing_data["risk_level"].append(None)
                icing_data["confidence"].append(None)
                continue
            
            # Convert Kelvin to Celsius
            temp_c = temp - 273.15
            
            # Determine icing risk based on temperature and humidity
            if temp_c < -20 or temp_c > 2:
                # Too cold or too warm for significant icing
                icing_data["risk_level"].append("none")
                icing_data["confidence"].append(0.8)
            elif -10 <= temp_c <= 0 and rh >= 60:
                # Ideal conditions for moderate to severe icing
                if rh >= 85:
                    icing_data["risk_level"].append("severe")
                    icing_data["confidence"].append(0.8)
                else:
                    icing_data["risk_level"].append("moderate")
                    icing_data["confidence"].append(0.75)
            elif (-15 <= temp_c < -10 or 0 < temp_c <= 2) and rh >= 70:
                # Conditions for light to moderate icing
                icing_data["risk_level"].append("light-moderate")
                icing_data["confidence"].append(0.7)
            elif rh >= 80:
                # High humidity can lead to light icing in other temperature ranges
                icing_data["risk_level"].append("light")
                icing_data["confidence"].append(0.6)
            else:
                # Low risk
                icing_data["risk_level"].append("none-light")
                icing_data["confidence"].append(0.7)
        
        return icing_data
    
    def _calculate_icing_potential(self, forecast: Dict[str, Any], freezing_forecast: Dict[str, Any], flight_level: str, pressure_level: str) -> Dict[str, Any]:
        """
        Calculate detailed aircraft icing potential.
        
        Args:
            forecast: Forecast data at the specific level
            freezing_forecast: Forecast data with freezing level information
            flight_level: Flight level (e.g., "FL180")
            pressure_level: Pressure level (e.g., "500h")
            
        Returns:
            Detailed icing analysis
        """
        icing_data = {
            "risk_level": [],
            "confidence": [],
            "severity": [],
            "icing_type": [],
            "forecast_time": forecast.get("forecast_time", [])
        }
        
        # Extract parameters
        temp_key = next((k for k in forecast.keys() if k.startswith("temp-")), None)
        rh_key = next((k for k in forecast.keys() if k.startswith("rh-")), None)
        cloud_key = next((k for k in forecast.keys() if k.startswith("clouds-")), None)
        
        # Extract freezing level if available
        freezing_level_key = next((k for k in freezing_forecast.keys() if k.startswith("freezingLevel-")), None)
        
        if not temp_key or not rh_key or temp_key not in forecast or rh_key not in forecast:
            return icing_data
        
        temp_values = forecast[temp_key]
        rh_values = forecast[rh_key]
        cloud_values = forecast[cloud_key] if cloud_key and cloud_key in forecast else None
        
        # Get freezing level values
        freezing_level_values = None
        if freezing_level_key and freezing_level_key in freezing_forecast:
            freezing_level_values = freezing_forecast[freezing_level_key]
        
        # Get flight level altitude in meters
        if flight_level.startswith("FL"):
            try:
                fl_num = int(flight_level[2:])
                flight_altitude = fl_num * 30.48  # Convert to meters
            except ValueError:
                flight_altitude = 0
        else:
            flight_altitude = 0
        
        # Process each timestamp
        for i in range(min(len(temp_values), len(rh_values))):
            temp = temp_values[i]
            rh = rh_values[i]
            cloud = cloud_values[i] if cloud_values and i < len(cloud_values) else None
            freezing_level = freezing_level_values[i] if freezing_level_values and i < len(freezing_level_values) else None
            
            if temp is None or rh is None:
                icing_data["risk_level"].append(None)
                icing_data["confidence"].append(None)
                icing_data["severity"].append(None)
                icing_data["icing_type"].append(None)
                continue
            
            # Convert Kelvin to Celsius
            temp_c = temp - 273.15
            
            # Initialize variables
            risk_level = "none"
            confidence = 0.5
            severity = 0
            icing_type = "none"
            
            # Check freezing level proximity (if available)
            freezing_level_proximity = False
            if freezing_level is not None:
                # If within 1000m of freezing level, note proximity
                if abs(flight_altitude - freezing_level) < 1000:
                    freezing_level_proximity = True
            
            # Analyze temperature and humidity conditions
            if -18 <= temp_c <= 0:
                # Temperature range where icing is possible
                
                # Determine icing type based on temperature
                if -18 <= temp_c <= -12:
                    icing_type = "rime"
                elif -12 < temp_c <= -4:
                    icing_type = "mixed"
                else:  # -4 < temp_c <= 0
                    icing_type = "clear"
                
                # Determine severity based on temperature, humidity, and cloud coverage
                if rh >= 85 and (cloud is None or cloud >= 50):
                    # High humidity and significant cloud coverage
                    if -12 <= temp_c <= -4:
                        severity = 3  # Severe
                        risk_level = "high"
                        confidence = 0.8
                    else:
                        severity = 2  # Moderate
                        risk_level = "moderate"
                        confidence = 0.75
                elif rh >= 70:
                    # Moderate humidity
                    severity = 1  # Light
                    risk_level = "light"
                    confidence = 0.7
                else:
                    # Low humidity
                    severity = 0  # Trace or none
                    risk_level = "low"
                    confidence = 0.65
                
                # Adjust for freezing level proximity
                if freezing_level_proximity and severity < 3:
                    severity += 1
                    if risk_level == "low":
                        risk_level = "light"
                    elif risk_level == "light":
                        risk_level = "moderate"
                    elif risk_level == "moderate":
                        risk_level = "high"
                    confidence += 0.05
            else:
                # Temperature outside icing range
                risk_level = "none"
                confidence = 0.8
                severity = 0
                icing_type = "none"
            
            # Map severity to descriptive text
            severity_text = {
                0: "none",
                1: "light",
                2: "moderate",
                3: "severe"
            }.get(severity, "unknown")
            
            # Store results
            icing_data["risk_level"].append(risk_level)
            icing_data["confidence"].append(min(1.0, confidence))
            icing_data["severity"].append(severity_text)
            icing_data["icing_type"].append(icing_type)
        
        return icing_data
    
    def _calculate_wind_shear(self, forecast: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate wind shear from forecast data.
        
        Args:
            forecast: Forecast data
            
        Returns:
            Dictionary with wind shear values
        """
        wind_shear_data = {
            "vertical_shear": [],
            "severity": []
        }
        
        # Get wind data from different levels
        surface_u_key = next((k for k in forecast.keys() if k.startswith("wind_u-surface")), None)
        surface_v_key = next((k for k in forecast.keys() if k.startswith("wind_v-surface")), None)
        
        # Check for a higher level (e.g., 850h)
        upper_u_key = next((k for k in forecast.keys() if k.startswith("wind_u-") and not k.endswith("surface")), None)
        upper_v_key = next((k for k in forecast.keys() if k.startswith("wind_v-") and not k.endswith("surface")), None)
        
        if not surface_u_key or not surface_v_key or not upper_u_key or not upper_v_key:
            return wind_shear_data
            
        if (surface_u_key not in forecast or surface_v_key not in forecast or 
            upper_u_key not in forecast or upper_v_key not in forecast):
            return wind_shear_data
        
        surface_u = forecast[surface_u_key]
        surface_v = forecast[surface_v_key]
        upper_u = forecast[upper_u_key]
        upper_v = forecast[upper_v_key]
        
        # Extract altitude difference from key names (approximate)
        upper_level = upper_u_key.split('-')[1]
        altitude_diff = 0
        
        if upper_level in self.PRESSURE_LEVEL_ALTITUDES:
            altitude_diff = self.PRESSURE_LEVEL_ALTITUDES[upper_level]
        
        if altitude_diff == 0:
            altitude_diff = 1500  # Default to 1500m if we can't determine
        
        # Calculate wind shear for each timestamp
        for i in range(min(len(surface_u), len(surface_v), len(upper_u), len(upper_v))):
            if (surface_u[i] is None or surface_v[i] is None or 
                upper_u[i] is None or upper_v[i] is None):
                wind_shear_data["vertical_shear"].append(None)
                wind_shear_data["severity"].append(None)
                continue
            
            # Calculate wind vector difference
            du = upper_u[i] - surface_u[i]
            dv = upper_v[i] - surface_v[i]
            
            # Magnitude of the shear vector
            shear_magnitude = math.sqrt(du**2 + dv**2)
            
            # Normalize by altitude difference (in km)
            shear_per_km = shear_magnitude / (altitude_diff / 1000)
            
            wind_shear_data["vertical_shear"].append(round(shear_per_km, 2))
            
            # Determine severity
            if shear_per_km < 5:
                wind_shear_data["severity"].append("light")
            elif shear_per_km < 10:
                wind_shear_data["severity"].append("moderate")
            else:
                wind_shear_data["severity"].append("severe")
        
        return wind_shear_data
    
    def _calculate_crosswind(self, forecast: Dict[str, Any], runway_heading: float) -> Dict[str, Any]:
        """
        Calculate crosswind component for a given runway heading.
        
        Args:
            forecast: Forecast data
            runway_heading: Runway heading in degrees
            
        Returns:
            Dictionary with crosswind components
        """
        crosswind_data = {
            "crosswind": [],
            "headwind": [],
            "severity": []
        }
        
        # Get surface wind data
        u_key = next((k for k in forecast.keys() if k.startswith("wind_u-surface")), None)
        v_key = next((k for k in forecast.keys() if k.startswith("wind_v-surface")), None)
        
        if not u_key or not v_key or u_key not in forecast or v_key not in forecast:
            return crosswind_data
            
        u_values = forecast[u_key]
        v_values = forecast[v_key]
        
        # Convert runway heading to meteorological convention
        runway_dir = (90 - runway_heading) % 360
        runway_rad = math.radians(runway_dir)
        
        # Calculate crosswind and headwind components
        for i in range(min(len(u_values), len(v_values))):
            u = u_values[i]
            v = v_values[i]
            
            if u is None or v is None:
                crosswind_data["crosswind"].append(None)
                crosswind_data["headwind"].append(None)
                crosswind_data["severity"].append(None)
                continue
            
            # Calculate wind speed and direction
            wind_speed = math.sqrt(u**2 + v**2)
            wind_dir_rad = math.atan2(v, u)
            
            # Calculate crosswind component
            crosswind = wind_speed * math.cos(wind_dir_rad - runway_rad)
            headwind = wind_speed * math.sin(wind_dir_rad - runway_rad)
            
            crosswind_data["crosswind"].append(round(crosswind, 2))
            crosswind_data["headwind"].append(round(headwind, 2))
            
            # Determine severity
            if abs(crosswind) < 10:
                crosswind_data["severity"].append("light")
            elif abs(crosswind) < 20:
                crosswind_data["severity"].append("moderate")
            else:
                crosswind_data["severity"].append("severe")
        
        return crosswind_data
    
    def _estimate_thermal_updrafts(self, forecast: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate thermal updraft strength.
        
        Args:
            forecast: Forecast data
            
        Returns:
            Dictionary with thermal updraft estimates
        """
        thermal_data = {
            "strength": [],
            "confidence": []
        }
        
        # Get relevant parameters
        temp_key = next((k for k in forecast.keys() if k.startswith("temp-surface")), None)
        cape_key = next((k for k in forecast.keys() if k.startswith("cape-")), None)
        
        if not temp_key or not cape_key or temp_key not in forecast or cape_key not in forecast:
            return thermal_data
            
        temp_values = forecast[temp_key]
        cape_values = forecast[cape_key]
        
        # Calculate thermal strength for each timestamp
        for i in range(min(len(temp_values), len(cape_values))):
            temp = temp_values[i]
            cape = cape_values[i]
            
            if temp is None or cape is None:
                thermal_data["strength"].append(None)
                thermal_data["confidence"].append(None)
                continue
            
            # Convert Kelvin to Celsius
            temp_c = temp - 273.15
            
            # Simple thermal strength estimation based on temperature and CAPE
            # In reality, this depends on many factors including ground heating, lapse rate, etc.
            
            # Base strength on temperature (higher temperatures generally produce stronger thermals)
            if temp_c < 10:
                strength = 0.5  # Weak
            elif temp_c < 20:
                strength = 1.0  # Moderate
            elif temp_c < 30:
                strength = 1.5  # Strong
            else:
                strength = 2.0  # Very strong
            
            # Modify based on CAPE (indicates atmospheric instability)
            if cape < 100:
                strength *= 0.5
            elif cape < 500:
                strength *= 0.8
            elif cape < 1000:
                strength *= 1.0
            elif cape < 2000:
                strength *= 1.2
            else:
                strength *= 1.5
            
            # Cap the maximum strength
            strength = min(3.0, max(0, strength))
            
            # Map strength to descriptive level
            if strength < 0.5:
                level = "none"
            elif strength < 1.0:
                level = "weak"
            elif strength < 1.5:
                level = "moderate"
            elif strength < 2.0:
                level = "strong"
            else:
                level = "very strong"
            
            thermal_data["strength"].append(level)
            thermal_data["confidence"].append(0.7)  # Fixed confidence level for simplicity
        
        return thermal_data
    
    def _generate_grid_points(self, center_lat: float, center_lon: float, radius_km: float, num_points: int = 8) -> List[Tuple[float, float]]:
        """
        Generate a grid of points around a center within a given radius.
        
        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            radius_km: Radius in kilometers
            num_points: Number of points to generate
            
        Returns:
            List of (lat, lon) coordinate tuples
        """
        # Angular distance in radians for the radius
        angular_dist = radius_km / 6371.0
        
        points = []
        
        # Generate points in a circle
        for i in range(num_points):
            angle = math.radians(i * 360 / num_points)
            
            lat = center_lat + math.sin(angle) * angular_dist * (180 / math.pi)
            lon = center_lon + math.cos(angle) * angular_dist * (180 / math.pi) / math.cos(math.radians(center_lat))
            
            # Ensure coordinates are valid
            lat = max(-90, min(90, lat))
            lon = (lon + 180) % 360 - 180
            
            points.append((lat, lon))
        
        # Add center point
        points.append((center_lat, center_lon))
        
        return points
    
    def _identify_significant_weather(self, center_forecast: Dict[str, Any], sample_forecasts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze forecasts to identify significant weather phenomena.
        
        Args:
            center_forecast: Forecast for the center point
            sample_forecasts: Forecasts for sample points around the center
            
        Returns:
            Dictionary of significant weather events
        """
        significant_weather = {
            "thunderstorms": [],
            "strong_winds": [],
            "low_visibility": [],
            "heavy_precipitation": [],
            "extreme_temperatures": []
        }
        
        # Number of timestamps to analyze
        num_timestamps = len(center_forecast.get("forecast_time", []))
        
        for i in range(num_timestamps):
            # Check for thunderstorms (using CAPE)
            cape_key = next((k for k in center_forecast.keys() if k.startswith("cape-")), None)
            if cape_key and cape_key in center_forecast and i < len(center_forecast[cape_key]):
                cape = center_forecast[cape_key][i]
                if cape is not None and cape > 1000:
                    significant_weather["thunderstorms"].append({
                        "time_index": i,
                        "timestamp": center_forecast["forecast_time"][i] if i < len(center_forecast["forecast_time"]) else None,
                        "cape": cape,
                        "severity": "moderate" if cape < 2000 else "severe",
                        "confidence": 0.7 if cape < 2000 else 0.8
                    })
            
            # Check for strong winds
            u_key = next((k for k in center_forecast.keys() if k.startswith("wind_u-surface")), None)
            v_key = next((k for k in center_forecast.keys() if k.startswith("wind_v-surface")), None)
            
            if (u_key and v_key and u_key in center_forecast and v_key in center_forecast and 
                i < len(center_forecast[u_key]) and i < len(center_forecast[v_key])):
                u = center_forecast[u_key][i]
                v = center_forecast[v_key][i]
                
                if u is not None and v is not None:
                    wind_speed = math.sqrt(u**2 + v**2)
                    
                    if wind_speed > 10:  # 10 m/s threshold (approx. 22 mph)
                        significant_weather["strong_winds"].append({
                            "time_index": i,
                            "timestamp": center_forecast["forecast_time"][i] if i < len(center_forecast["forecast_time"]) else None,
                            "wind_speed": round(wind_speed, 1),
                            "severity": "moderate" if wind_speed < 20 else "severe",
                            "confidence": 0.8
                        })
            
            # Check for low visibility
            vis_key = next((k for k in center_forecast.keys() if k.startswith("visibility-")), None)
            if vis_key and vis_key in center_forecast and i < len(center_forecast[vis_key]):
                visibility = center_forecast[vis_key][i]
                if visibility is not None and visibility < 5000:  # 5000m threshold
                    significant_weather["low_visibility"].append({
                        "time_index": i,
                        "timestamp": center_forecast["forecast_time"][i] if i < len(center_forecast["forecast_time"]) else None,
                        "visibility": visibility,
                        "severity": "moderate" if visibility > 1000 else "severe",
                        "confidence": 0.75
                    })
            
            # Check for heavy precipitation
            precip_key = next((k for k in center_forecast.keys() if k.startswith("precip-")), None)
            if precip_key and precip_key in center_forecast and i < len(center_forecast[precip_key]):
                precip = center_forecast[precip_key][i]
                if precip is not None and precip > 2:  # 2mm/hr threshold
                    significant_weather["heavy_precipitation"].append({
                        "time_index": i,
                        "timestamp": center_forecast["forecast_time"][i] if i < len(center_forecast["forecast_time"]) else None,
                        "precipitation": precip,
                        "severity": "moderate" if precip < 10 else "severe",
                        "confidence": 0.7
                    })
            
            # Check for extreme temperatures
            temp_key = next((k for k in center_forecast.keys() if k.startswith("temp-surface")), None)
            if temp_key and temp_key in center_forecast and i < len(center_forecast[temp_key]):
                temp = center_forecast[temp_key][i]
                if temp is not None:
                    temp_c = temp - 273.15
                    if temp_c > 35 or temp_c < -10:
                        significant_weather["extreme_temperatures"].append({
                            "time_index": i,
                            "timestamp": center_forecast["forecast_time"][i] if i < len(center_forecast["forecast_time"]) else None,
                            "temperature": round(temp_c, 1),
                            "type": "hot" if temp_c > 35 else "cold",
                            "severity": "moderate" if 35 < temp_c < 40 or -15 < temp_c < -10 else "severe",
                            "confidence": 0.8
                        })
        
        return significant_weather