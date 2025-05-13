# aviation_assistant/data/services/integration_service.py

import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta

class IntegrationService:
    """
    Service for integrating data from multiple aviation data sources.
    
    Provides utility functions to combine weather, flight, and traffic data
    in useful ways for pre-flight briefing and planning.
    """
    
    def __init__(self, avwx_connector, windy_connector, opensky_connector=None):
        """Initialize the integration service."""
        self.avwx = avwx_connector
        self.windy = windy_connector
        self.opensky = opensky_connector
        self.logger = logging.getLogger(__name__)
    
    def get_airport_weather(self, icao_code: str) -> Dict[str, Any]:
        """
        Get comprehensive weather information for an airport.
        
        Combines METAR, TAF, and visualization links.
        
        Args:
            icao_code: ICAO airport code
            
        Returns:
            Dictionary with weather data and visualization links
        """
        # Get METAR, TAF and station data
        metar_data = self.avwx.get_metar(icao_code, {"options": "info,translate"})
        taf_data = self.avwx.get_taf(icao_code, {"options": "info,translate"})
        station_data = self.avwx.get_station(icao_code)
        
        # Create basic result
        result = {
            "icao": icao_code,
            "timestamp": datetime.now().isoformat(),
            "metar": self._extract_metar_info(metar_data),
            "taf": self._extract_taf_info(taf_data),
            "visualization_urls": {}
        }
        
        # Add station info if available
        if isinstance(station_data, dict) and "error" not in station_data:
            result["station"] = {
                "name": station_data.get("name", "Unknown"),
                "city": station_data.get("city", ""),
                "country": station_data.get("country", ""),
                "latitude": station_data.get("latitude"),
                "longitude": station_data.get("longitude"),
                "elevation_ft": station_data.get("elevation_ft")
            }
            
            # Create visualization URLs if coordinates are available
            lat = station_data.get("latitude")
            lon = station_data.get("longitude")
            
            if lat is not None and lon is not None:
                result["visualization_urls"] = self._generate_visualization_urls(lat, lon)
        
        # Create text summaries
        result["metar_summary"] = self._generate_metar_summary(result["metar"]) if result["metar"] else "No METAR data available"
        result["taf_summary"] = self._generate_taf_summary(result["taf"]) if result["taf"] else "No TAF data available"
        
        # Add a hazardous weather assessment
        result["hazardous_conditions"] = self._assess_hazardous_conditions(result)
        
        return result
    
    def get_airport_traffic(self, icao_code: str, hours: int = 2) -> Dict[str, Any]:
        """
        Get traffic information for an airport.
        
        Args:
            icao_code: ICAO airport code
            hours: Number of hours to look back (default: 2)
            
        Returns:
            Dictionary with traffic data
        """
        if not self.opensky:
            return {"error": "OpenSky connector not available"}
        
        # Calculate time range
        end_time = datetime.now()
        begin_time = end_time - timedelta(hours=hours)
        
        # Get arrivals and departures
        arrivals = self.opensky.get_arrivals(icao_code, begin_time, end_time)
        departures = self.opensky.get_departures(icao_code, begin_time, end_time)
        
        # Get station data for airport information
        station_data = self.avwx.get_station(icao_code)
        
        # Create result
        result = {
            "icao": icao_code,
            "timestamp": datetime.now().isoformat(),
            "time_range": {
                "begin": begin_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": hours
            },
            "arrivals": arrivals if isinstance(arrivals, list) else [],
            "departures": departures if isinstance(departures, list) else []
        }
        
        # Add airport information if available
        if isinstance(station_data, dict) and "error" not in station_data:
            result["airport"] = {
                "name": station_data.get("name", "Unknown"),
                "city": station_data.get("city", ""),
                "country": station_data.get("country", ""),
                "latitude": station_data.get("latitude"),
                "longitude": station_data.get("longitude")
            }
        
        # Generate text summary
        result["summary"] = self._generate_traffic_summary(result)
        
        return result
    
    def get_route_weather(self, departure: str, destination: str) -> Dict[str, Any]:
        """
        Get comprehensive weather information for a flight route.
        
        Combines data from departure and destination airports.
        
        Args:
            departure: Departure airport ICAO code
            destination: Destination airport ICAO code
            
        Returns:
            Dictionary with route weather data
        """
        # Get weather for both airports
        dep_weather = self.get_airport_weather(departure)
        dest_weather = self.get_airport_weather(destination)
        
        # Create route result
        result = {
            "departure": departure,
            "destination": destination,
            "timestamp": datetime.now().isoformat(),
            "departure_weather": dep_weather,
            "destination_weather": dest_weather,
            "route_visualization": {}
        }
        
        # Generate route visualizations if coordinates are available
        if (isinstance(dep_weather.get("station", {}), dict) and 
            isinstance(dest_weather.get("station", {}), dict)):
            
            dep_lat = dep_weather["station"].get("latitude")
            dep_lon = dep_weather["station"].get("longitude")
            dest_lat = dest_weather["station"].get("latitude")
            dest_lon = dest_weather["station"].get("longitude")
            
            if all(coord is not None for coord in [dep_lat, dep_lon, dest_lat, dest_lon]):
                # Calculate midpoint for visualization
                mid_lat = (dep_lat + dest_lat) / 2
                mid_lon = (dep_lon + dest_lon) / 2
                
                # Calculate appropriate zoom level
                lat_diff = abs(dep_lat - dest_lat)
                lon_diff = abs(dep_lon - dest_lon)
                max_diff = max(lat_diff, lon_diff)
                
                zoom = 5
                if max_diff < 3:
                    zoom = 7
                elif max_diff > 10:
                    zoom = 4
                
                # Generate route visualization URLs
                result["route_visualization"] = {
                    "wind": f"https://www.windy.com/?wind,{mid_lat},{mid_lon},{zoom}",
                    "turbulence": f"https://www.windy.com/?turbulence,{mid_lat},{mid_lon},{zoom},i:850h",
                    "clouds": f"https://www.windy.com/?clouds,{mid_lat},{mid_lon},{zoom}",
                    "cloudtop": f"https://www.windy.com/?cloudtop,{mid_lat},{mid_lon},{zoom}",
                    "precipitation": f"https://www.windy.com/?rain,{mid_lat},{mid_lon},{zoom}",
                    "thunder": f"https://www.windy.com/?thunder,{mid_lat},{mid_lon},{zoom}"
                }
        
        # Generate text summary
        result["summary"] = self._generate_route_summary(result)
        
        return result
    
    def get_nearby_traffic(self, lat: float, lon: float, radius_km: float = 100) -> Dict[str, Any]:
        """
        Get traffic near a specific location.
        
        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Radius in kilometers
            
        Returns:
            Dictionary with nearby traffic data
        """
        if not self.opensky:
            return {"error": "OpenSky connector not available"}
        
        # Get traffic from OpenSky
        traffic = self.opensky.get_traffic_around_point(lat, lon, radius_km)
        
        # Create result
        result = {
            "location": {
                "latitude": lat,
                "longitude": lon
            },
            "radius_km": radius_km,
            "timestamp": datetime.now().isoformat(),
            "aircraft": traffic if isinstance(traffic, list) else []
        }
        
        # Add aircraft count
        result["aircraft_count"] = len(result["aircraft"])
        
        # Add weather at location if Windy connector is available
        if self.windy:
            try:
                weather = self.windy.get_forecast(
                    lat=lat,
                    lon=lon,
                    model="gfs",
                    parameters=["wind", "temp", "clouds", "visibility"],
                    levels=["surface"]
                )
                
                if isinstance(weather, dict) and "error" not in weather:
                    result["weather"] = self._extract_windy_summary(weather)
            except Exception as e:
                self.logger.error(f"Error getting weather for traffic location: {str(e)}")
        
        # Add visualization links
        result["visualization_urls"] = {
            "weather": f"https://www.windy.com/?wind,{lat},{lon},7",
            "traffic": f"https://globe.adsbexchange.com/?lat={lat}&lon={lon}&zoom=9"
        }
        
        # Generate text summary
        result["summary"] = self._generate_nearby_traffic_summary(result)
        
        return result
    
    def get_aircraft_info(self, icao24: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific aircraft.
        
        Args:
            icao24: ICAO24 transponder address (hex string)
            
        Returns:
            Dictionary with aircraft information
        """
        if not self.opensky:
            return {"error": "OpenSky connector not available"}
        
        # Get current position
        position = self.opensky.get_aircraft_position(icao24)
        
        if "error" in position:
            return {"error": f"Aircraft not found: {icao24}"}
        
        # Create result
        result = {
            "icao24": icao24,
            "timestamp": datetime.now().isoformat(),
            "position": position
        }
        
        # Add flight history if available
        try:
            end_time = datetime.now()
            begin_time = end_time - timedelta(hours=6)
            
            flights = self.opensky.get_flights_by_aircraft(icao24, begin_time, end_time)
            result["recent_flights"] = flights if isinstance(flights, list) else []
        except Exception as e:
            self.logger.error(f"Error getting flight history: {str(e)}")
            result["recent_flights"] = []
        
        # Add weather at aircraft location if coordinates are available
        if self.windy:
            lat = position.get("latitude")
            lon = position.get("longitude")
            
            if lat is not None and lon is not None:
                try:
                    weather = self.windy.get_forecast(
                        lat=lat,
                        lon=lon,
                        model="gfs",
                        parameters=["wind", "temp", "clouds"],
                        levels=["surface"]
                    )
                    
                    if isinstance(weather, dict) and "error" not in weather:
                        result["weather_at_location"] = self._extract_windy_summary(weather)
                except Exception as e:
                    self.logger.error(f"Error getting weather for aircraft location: {str(e)}")
        
        # Generate text summary
        result["summary"] = self._generate_aircraft_summary(result)
        
        return result
    
    def get_flight_level_weather(self, lat: float, lon: float, flight_level: str) -> Dict[str, Any]:
        """
        Get weather at a specific flight level.
        
        Args:
            lat: Latitude
            lon: Longitude
            flight_level: Flight level (e.g., "FL350")
            
        Returns:
            Dictionary with flight level weather data
        """
        if not self.windy:
            return {"error": "Windy connector not available"}
        
        # Map flight level to pressure level
        pressure_level = self._flight_level_to_pressure_level(flight_level)
        
        # Get weather at flight level
        try:
            weather = self.windy.get_forecast(
                lat=lat,
                lon=lon,
                model="gfs",
                parameters=["wind", "temp", "rh"],
                levels=[pressure_level]
            )
            
            if "error" in weather:
                return {"error": f"Error getting flight level weather: {weather['error']}"}
            
            # Create result
            result = {
                "location": {
                    "latitude": lat,
                    "longitude": lon
                },
                "flight_level": flight_level,
                "pressure_level": pressure_level,
                "timestamp": datetime.now().isoformat(),
                "weather": self._extract_flight_level_data(weather, pressure_level)
            }
            
            # Add visualization URL
            result["visualization_url"] = f"https://www.windy.com/?wind,{lat},{lon},7,i:{pressure_level.replace('h', '')}"
            
            # Add turbulence information if available
            try:
                turbulence = self.windy.get_turbulence_forecast(lat, lon, flight_level)
                if "error" not in turbulence:
                    result["turbulence"] = {
                        "level": turbulence.get("level", []),
                        "severity": turbulence.get("severity", [])
                    }
            except Exception as e:
                self.logger.error(f"Error getting turbulence data: {str(e)}")
            
            # Generate text summary
            result["summary"] = self._generate_flight_level_summary(result)
            
            return result
        except Exception as e:
            self.logger.error(f"Error getting flight level weather: {str(e)}")
            return {"error": str(e)}
    
    def _extract_metar_info(self, metar_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant information from METAR data."""
        if not isinstance(metar_data, dict) or "error" in metar_data:
            return None
        
        # Extract basic METAR info
        result = {
            "raw": metar_data.get("raw", ""),
            "time": metar_data.get("time", {}),
            "flight_rules": metar_data.get("flight_rules", ""),
            "visibility": None,
            "wind": None,
            "temperature": None,
            "dewpoint": None,
            "cloud_layers": []
        }
        
        # Extract visibility
        if "visibility" in metar_data and isinstance(metar_data["visibility"], dict):
            result["visibility"] = {
                "value": metar_data["visibility"].get("value"),
                "units": metar_data["visibility"].get("units", "SM")
            }
        
        # Extract wind
        if "wind" in metar_data and isinstance(metar_data["wind"], dict):
            result["wind"] = {
                "direction": metar_data["wind"].get("direction"),
                "speed": metar_data["wind"].get("speed"),
                "gust": metar_data["wind"].get("gust"),
                "units": metar_data["wind"].get("units", "KT")
            }
        
        # Extract temperature
        if "temperature" in metar_data and isinstance(metar_data["temperature"], dict):
            result["temperature"] = {
                "value": metar_data["temperature"].get("value"),
                "units": metar_data["temperature"].get("units", "C")
            }
        
        # Extract dewpoint
        if "dewpoint" in metar_data and isinstance(metar_data["dewpoint"], dict):
            result["dewpoint"] = {
                "value": metar_data["dewpoint"].get("value"),
                "units": metar_data["dewpoint"].get("units", "C")
            }
        
        # Extract cloud layers
        if "clouds" in metar_data and isinstance(metar_data["clouds"], list):
            for cloud in metar_data["clouds"]:
                if isinstance(cloud, dict):
                    layer = {
                        "type": cloud.get("type"),
                        "altitude": None,
                        "modifier": cloud.get("modifier")
                    }
                    
                    # Extract altitude
                    if "altitude" in cloud:
                        if isinstance(cloud["altitude"], dict):
                            layer["altitude"] = cloud["altitude"].get("value")
                        else:
                            layer["altitude"] = cloud["altitude"]
                    
                    result["cloud_layers"].append(layer)
        
        # Extract weather phenomena
        if "weather" in metar_data and isinstance(metar_data["weather"], list):
            result["weather_phenomena"] = []
            for weather in metar_data["weather"]:
                if isinstance(weather, dict):
                    phenomena = {
                        "intensity": weather.get("intensity"),
                        "descriptor": weather.get("descriptor"),
                        "precipitation": weather.get("precipitation"),
                        "obscuration": weather.get("obscuration"),
                        "other": weather.get("other")
                    }
                    result["weather_phenomena"].append(phenomena)
        
        return result
    
    def _extract_taf_info(self, taf_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant information from TAF data."""
        if not isinstance(taf_data, dict) or "error" in taf_data:
            return None
        
        # Extract basic TAF info
        result = {
            "raw": taf_data.get("raw", ""),
            "start_time": taf_data.get("start_time", {}),
            "end_time": taf_data.get("end_time", {}),
            "forecast_periods": []
        }
        
        # Extract forecast periods
        if "forecast" in taf_data and isinstance(taf_data["forecast"], list):
            for period in taf_data["forecast"]:
                if isinstance(period, dict):
                    forecast_period = {
                        "start_time": period.get("start_time", {}),
                        "end_time": period.get("end_time", {}),
                        "flight_rules": period.get("flight_rules", ""),
                        "probability": period.get("probability"),
                        "visibility": None,
                        "wind": None,
                        "cloud_layers": []
                    }
                    
                    # Extract visibility
                    if "visibility" in period and isinstance(period["visibility"], dict):
                        forecast_period["visibility"] = {
                            "value": period["visibility"].get("value"),
                            "units": period["visibility"].get("units", "SM")
                        }
                    
                    # Extract wind
                    if "wind" in period and isinstance(period["wind"], dict):
                        forecast_period["wind"] = {
                            "direction": period["wind"].get("direction"),
                            "speed": period["wind"].get("speed"),
                            "gust": period["wind"].get("gust"),
                            "units": period["wind"].get("units", "KT")
                        }
                    
                    # Extract cloud layers
                    if "clouds" in period and isinstance(period["clouds"], list):
                        for cloud in period["clouds"]:
                            if isinstance(cloud, dict):
                                layer = {
                                    "type": cloud.get("type"),
                                    "altitude": None,
                                    "modifier": cloud.get("modifier")
                                }
                                
                                # Extract altitude
                                if "altitude" in cloud:
                                    if isinstance(cloud["altitude"], dict):
                                        layer["altitude"] = cloud["altitude"].get("value")
                                    else:
                                        layer["altitude"] = cloud["altitude"]
                                
                                forecast_period["cloud_layers"].append(layer)
                    
                    # Extract weather phenomena
                    if "weather" in period and isinstance(period["weather"], list):
                        forecast_period["weather_phenomena"] = []
                        for weather in period["weather"]:
                            if isinstance(weather, dict):
                                phenomena = {
                                    "intensity": weather.get("intensity"),
                                    "descriptor": weather.get("descriptor"),
                                    "precipitation": weather.get("precipitation"),
                                    "obscuration": weather.get("obscuration"),
                                    "other": weather.get("other")
                                }
                                forecast_period["weather_phenomena"].append(phenomena)
                    
                    result["forecast_periods"].append(forecast_period)
        
        return result
    
    def _generate_visualization_urls(self, lat: float, lon: float, zoom: int = 9) -> Dict[str, Any]:
        """Generate URLs for various weather visualizations."""
        urls = {
            "wind": f"https://www.windy.com/?wind,{lat},{lon},{zoom}",
            "temperature": f"https://www.windy.com/?temp,{lat},{lon},{zoom}",
            "clouds": f"https://www.windy.com/?clouds,{lat},{lon},{zoom}",
            "cloudtop": f"https://www.windy.com/?cloudtop,{lat},{lon},{zoom}",
            "precipitation": f"https://www.windy.com/?rain,{lat},{lon},{zoom}",
            "turbulence": f"https://www.windy.com/?turbulence,{lat},{lon},{zoom},i:850h",
            "visibility": f"https://www.windy.com/?visibility,{lat},{lon},{zoom}",
            "flight_levels": {
                "FL050": f"https://www.windy.com/?wind,{lat},{lon},{zoom},i:850h",
                "FL100": f"https://www.windy.com/?wind,{lat},{lon},{zoom},i:700h",
                "FL180": f"https://www.windy.com/?wind,{lat},{lon},{zoom},i:500h",
                "FL240": f"https://www.windy.com/?wind,{lat},{lon},{zoom},i:400h",
                "FL340": f"https://www.windy.com/?wind,{lat},{lon},{zoom},i:250h"
            }
        }
        return urls
    
    def _extract_windy_summary(self, weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract summary metrics from Windy forecast data."""
        summary = {
            "surface": {}
        }
        
        # Check for forecast times
        if "forecast_time" in weather_data:
            summary["forecast_times"] = weather_data["forecast_time"]
        
        # Process surface data
        for key, value in weather_data.items():
            if "-surface" in key:
                param = key.replace("-surface", "")
                
                # Only process value if it's a list of numbers
                if isinstance(value, list) and value:
                    valid_values = [v for v in value if v is not None and isinstance(v, (int, float))]
                    if valid_values:
                        summary["surface"][param] = {
                            "min": min(valid_values),
                            "max": max(valid_values),
                            "avg": sum(valid_values) / len(valid_values),
                            "current": valid_values[0] if valid_values else None
                        }
        
        return summary
    
    def _extract_flight_level_data(self, weather_data: Dict[str, Any], pressure_level: str) -> Dict[str, Any]:
        """Extract data for a specific flight level from Windy forecast data."""
        result = {}
        
        # Check for forecast times
        if "forecast_time" in weather_data:
            result["forecast_times"] = weather_data["forecast_time"]
        
        # Process parameters for the specified pressure level
        for key, value in weather_data.items():
            if f"-{pressure_level}" in key:
                param = key.replace(f"-{pressure_level}", "")
                
                # Only process value if it's a list of numbers
                if isinstance(value, list) and value:
                    valid_values = [v for v in value if v is not None and isinstance(v, (int, float))]
                    if valid_values:
                        result[param] = {
                            "min": min(valid_values),
                            "max": max(valid_values),
                            "avg": sum(valid_values) / len(valid_values),
                            "current": valid_values[0] if valid_values else None
                        }
        
        # Convert temperature from Kelvin to Celsius if present
        if "temp" in result and "current" in result["temp"]:
            if result["temp"]["current"] is not None:
                temp_k = result["temp"]["current"]
                result["temp_celsius"] = temp_k - 273.15
        
        # Process wind components if present
        if "wind_u" in result and "wind_v" in result:
            u_current = result["wind_u"].get("current")
            v_current = result["wind_v"].get("current")
            
            if u_current is not None and v_current is not None:
                import math
                
                # Calculate wind speed and direction
                wind_speed = math.sqrt(u_current**2 + v_current**2)
                wind_dir = (270 - math.degrees(math.atan2(v_current, u_current))) % 360
                
                result["wind_speed"] = wind_speed
                result["wind_direction"] = wind_dir
                result["wind_speed_knots"] = wind_speed * 1.94384  # Convert m/s to knots
        
        return result
    
    def _flight_level_to_pressure_level(self, flight_level: str) -> str:
        """Convert flight level to pressure level for Windy API."""
        # Simple mapping of common flight levels to pressure levels
        mapping = {
            "FL050": "850h",
            "FL100": "700h",
            "FL180": "500h",
            "FL240": "400h",
            "FL300": "300h",
            "FL340": "250h",
            "FL390": "200h",
            "FL450": "150h"
        }
        
        return mapping.get(flight_level, "500h")  # Default to 500h if not found
    
    def _assess_hazardous_conditions(self, weather_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify hazardous weather conditions from METAR and TAF data."""
        hazards = []
        
        # Check METAR for hazardous conditions
        metar = weather_data.get("metar", {})
        if metar:
            # Check visibility
            visibility = metar.get("visibility", {})
            if visibility and "value" in visibility:
                vis_value = visibility.get("value")
                if vis_value is not None:
                    if vis_value < 1:
                        hazards.append({
                            "type": "low_visibility",
                            "description": f"Low visibility: {vis_value} {visibility.get('units', 'SM')}",
                            "severity": "high",
                            "source": "METAR"
                        })
                    elif vis_value < 3:
                        hazards.append({
                            "type": "reduced_visibility",
                            "description": f"Reduced visibility: {vis_value} {visibility.get('units', 'SM')}",
                            "severity": "moderate",
                            "source": "METAR"
                        })
            
            # Check wind
            wind = metar.get("wind", {})
            if wind and "speed" in wind:
                wind_speed = wind.get("speed")
                wind_gust = wind.get("gust")
                
                if wind_speed is not None:
                    if wind_speed > 25:
                        hazards.append({
                            "type": "high_winds",
                            "description": f"High winds: {wind_speed} {wind.get('units', 'KT')}",
                            "severity": "moderate",
                            "source": "METAR"
                        })
                    
                    if wind_gust is not None and wind_gust - wind_speed > 10:
                        hazards.append({
                            "type": "wind_gusts",
                            "description": f"Significant wind gusts: {wind_gust} {wind.get('units', 'KT')}",
                            "severity": "moderate",
                            "source": "METAR"
                        })
            
            # Check cloud layers for low ceilings
            cloud_layers = metar.get("cloud_layers", [])
            for layer in cloud_layers:
                if isinstance(layer, dict):
                    cloud_type = layer.get("type")
                    altitude = layer.get("altitude")
                    
                    if cloud_type in ["BKN", "OVC"] and altitude is not None:
                        ceiling_ft = altitude * 100  # Convert to feet
                        
                        if ceiling_ft < 500:
                            hazards.append({
                                "type": "low_ceiling",
                                "description": f"Very low ceiling: {ceiling_ft} feet",
                                "severity": "high",
                                "source": "METAR"
                            })
                        elif ceiling_ft < 1000:
                            hazards.append({
                                "type": "low_ceiling",
                                "description": f"Low ceiling: {ceiling_ft} feet",
                                "severity": "moderate",
                                "source": "METAR"
                            })
                    
                    # Check for thunderstorms (CB clouds)
                    if layer.get("modifier") == "CB":
                        hazards.append({
                            "type": "thunderstorm",
                            "description": "Cumulonimbus clouds (thunderstorms)",
                            "severity": "high",
                            "source": "METAR"
                        })
            
            # Check weather phenomena
            weather_phenomena = metar.get("weather_phenomena", [])
            for phenomena in weather_phenomena:
                if isinstance(phenomena, dict):
                    # Check for thunderstorms
                    if phenomena.get("descriptor") == "TS":
                        hazards.append({
                            "type": "thunderstorm",
                            "description": "Thunderstorm reported",
                            "severity": "high",
                            "source": "METAR"
                        })
                    
                    # Check for freezing precipitation
                    if phenomena.get("descriptor") == "FZ":
                        hazards.append({
                            "type": "freezing_precipitation",
                            "description": "Freezing precipitation (risk of icing)",
                            "severity": "high",
                            "source": "METAR"
                        })
                    
                    # Check for heavy precipitation
                    if phenomena.get("intensity") == "+":
                        precip_type = phenomena.get("precipitation", "")
                        hazards.append({
                            "type": "heavy_precipitation",
                            "description": f"Heavy precipitation ({precip_type})",
                            "severity": "moderate",
                            "source": "METAR"
                        })
        
        # Check TAF for future hazardous conditions
        taf = weather_data.get("taf", {})
        if taf:
            forecast_periods = taf.get("forecast_periods", [])
            
            for period_idx, period in enumerate(forecast_periods):
                if isinstance(period, dict):
                    # Check visibility
                    visibility = period.get("visibility", {})
                    if visibility and "value" in visibility:
                        vis_value = visibility.get("value")
                        if vis_value is not None:
                            if vis_value < 1:
                                hazards.append({
                                    "type": "low_visibility_forecast",
                                    "description": f"Forecast low visibility: {vis_value} {visibility.get('units', 'SM')}",
                                    "severity": "high",
                                    "source": "TAF",
                                    "period": period_idx
                                })
                            elif vis_value < 3:
                                hazards.append({
                                    "type": "reduced_visibility_forecast",
                                    "description": f"Forecast reduced visibility: {vis_value} {visibility.get('units', 'SM')}",
                                    "severity": "moderate",
                                    "source": "TAF",
                                    "period": period_idx
                                })
                    
                    # Check wind
                    wind = period.get("wind", {})
                    if wind and "speed" in wind:
                        wind_speed = wind.get("speed")
                        wind_gust = wind.get("gust")
                        
                        if wind_speed is not None and wind_speed > 25:
                            hazards.append({
                                "type": "high_winds_forecast",
                                "description": f"Forecast high winds: {wind_speed} {wind.get('units', 'KT')}",
                                "severity": "moderate",
                                "source": "TAF",
                                "period": period_idx
                            })
                        
                        if wind_gust is not None and wind_gust - wind_speed > 10:
                            hazards.append({
                                "type": "wind_gusts_forecast",
                                "description": f"Forecast significant wind gusts: {wind_gust} {wind.get('units', 'KT')}",
                                "severity": "moderate",
                                "source": "TAF",
                                "period": period_idx
                            })
                    
                    # Check cloud layers
                    cloud_layers = period.get("cloud_layers", [])
                    for layer in cloud_layers:
                        if isinstance(layer, dict):
                            cloud_type = layer.get("type")
                            altitude = layer.get("altitude")
                            
                            if cloud_type in ["BKN", "OVC"] and altitude is not None:
                                ceiling_ft = altitude * 100  # Convert to feet
                                
                                if ceiling_ft < 500:
                                    hazards.append({
                                        "type": "low_ceiling_forecast",
                                        "description": f"Forecast very low ceiling: {ceiling_ft} feet",
                                        "severity": "high",
                                        "source": "TAF",
                                        "period": period_idx
                                    })
                                elif ceiling_ft < 1000:
                                    hazards.append({
                                        "type": "low_ceiling_forecast",
                                        "description": f"Forecast low ceiling: {ceiling_ft} feet",
                                        "severity": "moderate",
                                        "source": "TAF",
                                        "period": period_idx
                                    })
                            
                            # Check for thunderstorms (CB clouds)
                            if layer.get("modifier") == "CB":
                                hazards.append({
                                    "type": "thunderstorm_forecast",
                                    "description": "Forecast cumulonimbus clouds (thunderstorms)",
                                    "severity": "high",
                                    "source": "TAF",
                                    "period": period_idx
                                })
                    
                    # Check weather phenomena
                    weather_phenomena = period.get("weather_phenomena", [])
                    for phenomena in weather_phenomena:
                        if isinstance(phenomena, dict):
                            # Check for thunderstorms
                            if phenomena.get("descriptor") == "TS":
                                hazards.append({
                                    "type": "thunderstorm_forecast",
                                    "description": "Forecast thunderstorm",
                                    "severity": "high",
                                    "source": "TAF",
                                    "period": period_idx
                                })
                            
                            # Check for freezing precipitation
                            if phenomena.get("descriptor") == "FZ":
                                hazards.append({
                                    "type": "freezing_precipitation_forecast",
                                    "description": "Forecast freezing precipitation (risk of icing)",
                                    "severity": "high",
                                    "source": "TAF",
                                    "period": period_idx
                                })
        
        return hazards
    
    def _generate_metar_summary(self, metar: Dict[str, Any]) -> str:
        """Generate a human-readable summary of METAR data."""
        if not metar:
            return "No METAR data available"
            
        parts = []
        
        # Add raw METAR
        raw = metar.get("raw", "")
        parts.append(f"Raw METAR: {raw}")
        
        # Add flight rules
        flight_rules = metar.get("flight_rules", "")
        if flight_rules:
            parts.append(f"Flight category: {flight_rules}")
        
        # Add wind
        wind = metar.get("wind", {})
        if isinstance(wind, dict):
            direction = wind.get("direction")
            speed = wind.get("speed")
            gust = wind.get("gust")
            units = wind.get("units", "KT")
            
            if direction is not None and speed is not None:
                if direction == 0:
                    wind_text = f"Wind: Variable at {speed} {units}"
                else:
                    wind_text = f"Wind: {direction}° at {speed} {units}"
                
                if gust is not None:
                    wind_text += f", gusting to {gust} {units}"
                
                parts.append(wind_text)
        
        # Add visibility
        visibility = metar.get("visibility", {})
        if isinstance(visibility, dict):
            value = visibility.get("value")
            units = visibility.get("units", "SM")
            
            if value is not None:
                parts.append(f"Visibility: {value} {units}")
        
        # Add weather phenomena
        phenomena = metar.get("weather_phenomena", [])
        if phenomena:
            wx_texts = []
            for wx in phenomena:
                if isinstance(wx, dict):
                    intensity = wx.get("intensity", "")
                    descriptor = wx.get("descriptor", "")
                    precipitation = wx.get("precipitation", "")
                    obscuration = wx.get("obscuration", "")
                    
                    # Construct a text description
                    wx_text = ""
                    if intensity == "+":
                        wx_text += "Heavy "
                    elif intensity == "-":
                        wx_text += "Light "
                    
                    if descriptor == "TS":
                        wx_text += "Thunderstorm "
                    elif descriptor == "FZ":
                        wx_text += "Freezing "
                    
                    if precipitation == "RA":
                        wx_text += "Rain"
                    elif precipitation == "SN":
                        wx_text += "Snow"
                    elif precipitation == "DZ":
                        wx_text += "Drizzle"
                    
                    if obscuration == "FG":
                        wx_text += "Fog"
                    elif obscuration == "BR":
                        wx_text += "Mist"
                    
                    if wx_text:
                        wx_texts.append(wx_text.strip())
            
            if wx_texts:
                parts.append("Weather: " + ", ".join(wx_texts))
        
        # Add clouds
        layers = metar.get("cloud_layers", [])
        if layers:
            cloud_texts = []
            for layer in layers:
                if isinstance(layer, dict):
                    layer_type = layer.get("type", "")
                    altitude = layer.get("altitude")
                    modifier = layer.get("modifier", "")
                    
                    if layer_type and altitude is not None:
                        cloud_text = f"{layer_type} at {altitude}00 ft"
                        if modifier:
                            cloud_text += f" {modifier}"
                        cloud_texts.append(cloud_text)
            
            if cloud_texts:
                parts.append("Clouds: " + ", ".join(cloud_texts))
        
        # Add temperature and dewpoint
        temp = metar.get("temperature", {})
        dewpoint = metar.get("dewpoint", {})
        
        if isinstance(temp, dict) and isinstance(dewpoint, dict):
            temp_val = temp.get("value")
            dewpoint_val = dewpoint.get("value")
            
            if temp_val is not None and dewpoint_val is not None:
                parts.append(f"Temperature: {temp_val}°C, Dewpoint: {dewpoint_val}°C")
            elif temp_val is not None:
                parts.append(f"Temperature: {temp_val}°C")
        
        return "\n".join(parts)
    
    def _generate_taf_summary(self, taf: Dict[str, Any]) -> str:
        """Generate a human-readable summary of TAF data."""
        if not taf:
            return "No TAF data available"
            
        parts = []
        
        # Add raw TAF
        raw = taf.get("raw", "")
        parts.append(f"Raw TAF: {raw}")
        
        # Add validity period
        start_time = None
        end_time = None
        
        if "start_time" in taf and isinstance(taf["start_time"], dict):
            start_dt = taf["start_time"].get("dt")
            if start_dt:
                start_time = start_dt
        
        if "end_time" in taf and isinstance(taf["end_time"], dict):
            end_dt = taf["end_time"].get("dt")
            if end_dt:
                end_time = end_dt
        
        if start_time and end_time:
            parts.append(f"Valid: {start_time} to {end_time}")
        
        # Add forecast periods
        periods = taf.get("forecast_periods", [])
        if periods:
            parts.append("Forecast periods:")
            
            for i, period in enumerate(periods):
                # Get period time range
                period_start = None
                period_end = None
                
                if "start_time" in period and isinstance(period["start_time"], dict):
                    start_dt = period["start_time"].get("dt")
                    if start_dt:
                        period_start = start_dt
                
                if "end_time" in period and isinstance(period["end_time"], dict):
                    end_dt = period["end_time"].get("dt")
                    if end_dt:
                        period_end = end_dt
                
                time_str = ""
                if period_start and period_end:
                    time_str = f" ({period_start} to {period_end})"
                
                # Add probability if available
                prob_str = ""
                if period.get("probability") is not None:
                    prob_str = f" - {period['probability']}% probability"
                
                parts.append(f"  Period {i+1}{time_str}{prob_str}:")
                
                # Add flight rules
                flight_rules = period.get("flight_rules", "")
                if flight_rules:
                    parts.append(f"    Flight category: {flight_rules}")
                
                # Add wind
                wind = period.get("wind", {})
                if isinstance(wind, dict):
                    direction = wind.get("direction")
                    speed = wind.get("speed")
                    gust = wind.get("gust")
                    units = wind.get("units", "KT")
                    
                    if direction is not None and speed is not None:
                        if direction == 0:
                            wind_text = f"Wind: Variable at {speed} {units}"
                        else:
                            wind_text = f"Wind: {direction}° at {speed} {units}"
                        
                        if gust is not None:
                            wind_text += f", gusting to {gust} {units}"
                        
                        parts.append(f"    {wind_text}")
                
                # Add visibility
                visibility = period.get("visibility", {})
                if isinstance(visibility, dict):
                    value = visibility.get("value")
                    units = visibility.get("units", "SM")
                    
                    if value is not None:
                        parts.append(f"    Visibility: {value} {units}")
                
                # Add weather phenomena
                phenomena = period.get("weather_phenomena", [])
                if phenomena:
                    wx_texts = []
                    for wx in phenomena:
                        if isinstance(wx, dict):
                            intensity = wx.get("intensity", "")
                            descriptor = wx.get("descriptor", "")
                            precipitation = wx.get("precipitation", "")
                            obscuration = wx.get("obscuration", "")
                            
                            # Construct a text description
                            wx_text = ""
                            if intensity == "+":
                                wx_text += "Heavy "
                            elif intensity == "-":
                                wx_text += "Light "
                            
                            if descriptor == "TS":
                                wx_text += "Thunderstorm "
                            elif descriptor == "FZ":
                                wx_text += "Freezing "
                            
                            if precipitation == "RA":
                                wx_text += "Rain"
                            elif precipitation == "SN":
                                wx_text += "Snow"
                            elif precipitation == "DZ":
                                wx_text += "Drizzle"
                            
                            if obscuration == "FG":
                                wx_text += "Fog"
                            elif obscuration == "BR":
                                wx_text += "Mist"
                            
                            if wx_text:
                                wx_texts.append(wx_text.strip())
                    
                    if wx_texts:
                        parts.append("    Weather: " + ", ".join(wx_texts))
                
                # Add clouds
                layers = period.get("cloud_layers", [])
                if layers:
                    cloud_texts = []
                    for layer in layers:
                        if isinstance(layer, dict):
                            layer_type = layer.get("type", "")
                            altitude = layer.get("altitude")
                            
                            if layer_type and altitude is not None:
                                cloud_texts.append(f"{layer_type} at {altitude}00 ft")
                    
                    if cloud_texts:
                        parts.append("    Clouds: " + ", ".join(cloud_texts))
        
        return "\n".join(parts)
    
    def _generate_traffic_summary(self, traffic_data: Dict[str, Any]) -> str:
        """Generate a human-readable summary of airport traffic data."""
        icao = traffic_data.get("icao", "")
        arrivals = traffic_data.get("arrivals", [])
        departures = traffic_data.get("departures", [])
        
        parts = [f"Traffic summary for {icao}"]
        
        # Add airport name if available
        airport = traffic_data.get("airport", {})
        if airport and "name" in airport:
            parts.append(f"{airport['name']}")
        
        # Add time range
        time_range = traffic_data.get("time_range", {})
        if time_range:
            hours = time_range.get("hours", 2)
            parts.append(f"Traffic for the past {hours} hours:")
        
        # Add arrivals
        parts.append(f"Arrivals: {len(arrivals)}")
        
        for i, flight in enumerate(arrivals[:5]):  # Show first 5 arrivals
            callsign = flight.get("callsign", "Unknown")
            origin = flight.get("estDepartureAirport", "Unknown")
            
            # Get arrival time
            time = flight.get("lastSeen")
            time_str = "Unknown time"
            if time:
                try:
                    dt = datetime.fromtimestamp(time)
                    time_str = dt.strftime("%H:%M UTC")
                except:
                    pass
            
            parts.append(f"  {i+1}. {callsign} from {origin}, arrived at {time_str}")
        
        # Add departures
        parts.append(f"Departures: {len(departures)}")
        
        for i, flight in enumerate(departures[:5]):  # Show first 5 departures
            callsign = flight.get("callsign", "Unknown")
            destination = flight.get("estArrivalAirport", "Unknown")
            
            # Get departure time
            time = flight.get("firstSeen")
            time_str = "Unknown time"
            if time:
                try:
                    dt = datetime.fromtimestamp(time)
                    time_str = dt.strftime("%H:%M UTC")
                except:
                    pass
            
            parts.append(f"  {i+1}. {callsign} to {destination}, departed at {time_str}")
        
        return "\n".join(parts)
    
    def _generate_route_summary(self, route_data: Dict[str, Any]) -> str:
        """Generate a human-readable summary of route data."""
        dep = route_data.get("departure", "")
        dest = route_data.get("destination", "")
        
        parts = [f"Flight route: {dep} to {dest}"]
        
        # Add departure weather summary
        dep_weather = route_data.get("departure_weather", {})
        if dep_weather:
            dep_metar = dep_weather.get("metar", {})
            dep_station = dep_weather.get("station", {})
            
            if dep_station and "name" in dep_station:
                parts.append(f"\nDeparture airport: {dep_station['name']} ({dep})")
            else:
                parts.append(f"\nDeparture airport: {dep}")
            
            if dep_metar:
                flight_rules = dep_metar.get("flight_rules", "")
                if flight_rules:
                    parts.append(f"  Current conditions: {flight_rules}")
                
                visibility = dep_metar.get("visibility", {})
                if visibility and "value" in visibility:
                    parts.append(f"  Visibility: {visibility['value']} {visibility.get('units', 'SM')}")
                
                wind = dep_metar.get("wind", {})
                if wind and "speed" in wind:
                    direction = wind.get("direction")
                    if direction == 0:
                        direction_text = "Variable"
                    else:
                        direction_text = f"{direction}°"
                    
                    parts.append(f"  Wind: {direction_text} at {wind['speed']} {wind.get('units', 'KT')}")
            
            # Add hazards at departure
            hazards = dep_weather.get("hazardous_conditions", [])
            if hazards:
                hazard_texts = []
                for hazard in hazards[:3]:  # Show top 3 hazards
                    if "description" in hazard:
                        hazard_texts.append(hazard["description"])
                
                if hazard_texts:
                    parts.append("  Hazards: " + "; ".join(hazard_texts))
        
        # Add destination weather summary
        dest_weather = route_data.get("destination_weather", {})
        if dest_weather:
            dest_metar = dest_weather.get("metar", {})
            dest_station = dest_weather.get("station", {})
            
            if dest_station and "name" in dest_station:
                parts.append(f"\nDestination airport: {dest_station['name']} ({dest})")
            else:
                parts.append(f"\nDestination airport: {dest}")
            
            if dest_metar:
                flight_rules = dest_metar.get("flight_rules", "")
                if flight_rules:
                    parts.append(f"  Current conditions: {flight_rules}")
                
                visibility = dest_metar.get("visibility", {})
                if visibility and "value" in visibility:
                    parts.append(f"  Visibility: {visibility['value']} {visibility.get('units', 'SM')}")
                
                wind = dest_metar.get("wind", {})
                if wind and "speed" in wind:
                    direction = wind.get("direction")
                    if direction == 0:
                        direction_text = "Variable"
                    else:
                        direction_text = f"{direction}°"
                    
                    parts.append(f"  Wind: {direction_text} at {wind['speed']} {wind.get('units', 'KT')}")
            
            # Add hazards at destination
            hazards = dest_weather.get("hazardous_conditions", [])
            if hazards:
                hazard_texts = []
                for hazard in hazards[:3]:  # Show top 3 hazards
                    if "description" in hazard:
                        hazard_texts.append(hazard["description"])
                
                if hazard_texts:
                    parts.append("  Hazards: " + "; ".join(hazard_texts))
        
        # Add visualization links
        viz = route_data.get("route_visualization", {})
        if viz:
            parts.append("\nVisualization links:")
            for name, url in viz.items():
                parts.append(f"  {name.capitalize()}: {url}")
        
        return "\n".join(parts)
    
    def _generate_nearby_traffic_summary(self, traffic_data: Dict[str, Any]) -> str:
        """Generate a human-readable summary of nearby traffic data."""
        location = traffic_data.get("location", {})
        lat = location.get("latitude", 0)
        lon = location.get("longitude", 0)
        radius = traffic_data.get("radius_km", 0)
        aircraft = traffic_data.get("aircraft", [])
        
        parts = [f"Traffic within {radius} km of {lat:.4f}, {lon:.4f}"]
        parts.append(f"Total aircraft: {len(aircraft)}")
        
        # Add information about each aircraft
        for i, ac in enumerate(aircraft[:10]):  # Show first 10 aircraft
            callsign = ac.get("callsign", "").strip() or "Unknown"
            alt = ac.get("geo_altitude")
            alt_str = f"{alt:.0f}m ({int(alt * 3.28084)}ft)" if alt is not None else "Unknown"
            
            speed = ac.get("velocity")
            speed_str = f"{speed:.0f}m/s ({int(speed * 1.94384)}kts)" if speed is not None else "Unknown"
            
            heading = ac.get("true_track")
            heading_str = f"{heading:.0f}°" if heading is not None else "Unknown"
            
            parts.append(f"  {i+1}. {callsign} at {alt_str}, {speed_str}, heading {heading_str}")
        
        # Add weather at location if available
        weather = traffic_data.get("weather", {})
        if weather and "surface" in weather:
            surface = weather["surface"]
            parts.append("\nWeather at location:")
            
            # Add temperature if available
            if "temp" in surface:
                temp = surface["temp"]
                if "current" in temp and temp["current"] is not None:
                    temp_c = temp["current"] - 273.15  # Convert from Kelvin
                    parts.append(f"  Temperature: {temp_c:.1f}°C")
            
            # Add wind if available
            if "wind_speed" in surface and "wind_direction" in surface:
                ws = surface["wind_speed"]
                wd = surface["wind_direction"]
                if "current" in ws and "current" in wd:
                    speed = ws["current"]
                    direction = wd["current"]
                    if speed is not None and direction is not None:
                        parts.append(f"  Wind: {direction:.0f}° at {speed:.1f} m/s ({speed * 1.94384:.0f} kts)")
            
            # Add visibility if available
            if "visibility" in surface:
                vis = surface["visibility"]
                if "current" in vis and vis["current"] is not None:
                    vis_km = vis["current"] / 1000
                    parts.append(f"  Visibility: {vis_km:.1f} km")
        
        # Add visualization links
        viz = traffic_data.get("visualization_urls", {})
        if viz:
            parts.append("\nVisualization links:")
            for name, url in viz.items():
                parts.append(f"  {name.capitalize()}: {url}")
        
        return "\n".join(parts)
    
    def _generate_aircraft_summary(self, aircraft_data: Dict[str, Any]) -> str:
        """Generate a human-readable summary of aircraft data."""
        icao24 = aircraft_data.get("icao24", "")
        position = aircraft_data.get("position", {})
        
        parts = [f"Aircraft information for {icao24}"]
        
        # Add position information
        if position:
            callsign = position.get("callsign", "").strip() or "Unknown"
            parts.append(f"Callsign: {callsign}")
            
            lat = position.get("latitude")
            lon = position.get("longitude")
            if lat is not None and lon is not None:
                parts.append(f"Position: {lat:.4f}, {lon:.4f}")
            
            altitude = position.get("geo_altitude")
            if altitude is not None:
                alt_ft = altitude * 3.28084  # Convert to feet
                parts.append(f"Altitude: {alt_ft:.0f} feet")
            
            speed = position.get("velocity")
            if speed is not None:
                speed_kts = speed * 1.94384  # Convert to knots
                parts.append(f"Speed: {speed_kts:.0f} knots")
            
            heading = position.get("true_track")
            if heading is not None:
                parts.append(f"Heading: {heading:.0f}°")
            
            origin = position.get("origin_country", "Unknown")
            parts.append(f"Origin country: {origin}")
            
            on_ground = position.get("on_ground")
            if on_ground is not None:
                if on_ground:
                    parts.append("Status: On ground")
                else:
                    parts.append("Status: Airborne")
        
        # Add recent flights if available
        flights = aircraft_data.get("recent_flights", [])
        if flights:
            parts.append(f"\nRecent flights ({len(flights)}):")
            
            for i, flight in enumerate(flights[:3]):  # Show first 3 flights
                dep = flight.get("estDepartureAirport", "Unknown")
                arr = flight.get("estArrivalAirport", "Unknown")
                
                dep_time = flight.get("firstSeen")
                arr_time = flight.get("lastSeen")
                
                dep_time_str = "Unknown"
                if dep_time:
                    try:
                        dt = datetime.fromtimestamp(dep_time)
                        dep_time_str = dt.strftime("%Y-%m-%d %H:%M UTC")
                    except:
                        pass
                
                arr_time_str = "Unknown"
                if arr_time:
                    try:
                        dt = datetime.fromtimestamp(arr_time)
                        arr_time_str = dt.strftime("%Y-%m-%d %H:%M UTC")
                    except:
                        pass
                
                parts.append(f"  {i+1}. {dep} to {arr}, departed {dep_time_str}, arrived {arr_time_str}")
        
        # Add weather at aircraft location if available
        weather = aircraft_data.get("weather_at_location", {})
        if weather and "surface" in weather:
            surface = weather["surface"]
            parts.append("\nWeather at aircraft location:")
            
            # Add temperature if available
            if "temp" in surface:
                temp = surface["temp"]
                if "current" in temp and temp["current"] is not None:
                    temp_c = temp["current"] - 273.15  # Convert from Kelvin
                    parts.append(f"  Temperature: {temp_c:.1f}°C")
            
            # Add wind if available
            if "wind_speed" in surface and "wind_direction" in surface:
                ws = surface["wind_speed"]
                wd = surface["wind_direction"]
                if "current" in ws and "current" in wd:
                    speed = ws["current"]
                    direction = wd["current"]
                    if speed is not None and direction is not None:
                        parts.append(f"  Wind: {direction:.0f}° at {speed:.1f} m/s ({speed * 1.94384:.0f} kts)")
        
        return "\n".join(parts)
    
    def _generate_flight_level_summary(self, flight_level_data: Dict[str, Any]) -> str:
        """Generate a human-readable summary of flight level weather data."""
        flight_level = flight_level_data.get("flight_level", "")
        pressure_level = flight_level_data.get("pressure_level", "")
        location = flight_level_data.get("location", {})
        weather = flight_level_data.get("weather", {})
        
        lat = location.get("latitude", 0)
        lon = location.get("longitude", 0)
        
        parts = [f"Weather at {flight_level} ({lat:.4f}, {lon:.4f})"]
        
        # Add temperature
        if "temp_celsius" in weather:
            temp_c = weather["temp_celsius"]
            parts.append(f"Temperature: {temp_c:.1f}°C")
        
        # Add wind
        if "wind_speed" in weather and "wind_direction" in weather:
            speed = weather["wind_speed"]
            direction = weather["wind_direction"]
            speed_kts = weather.get("wind_speed_knots", speed * 1.94384)
            
            parts.append(f"Wind: {direction:.0f}° at {speed_kts:.0f} knots")
        
        # Add relative humidity if available
        if "rh" in weather:
            rh = weather["rh"]
            if "current" in rh and rh["current"] is not None:
                parts.append(f"Relative humidity: {rh['current']:.0f}%")
        
        # Add turbulence information if available
        turbulence = flight_level_data.get("turbulence", {})
        if turbulence:
            level = turbulence.get("level", [])
            if level and len(level) > 0:
                first_level = level[0]
                parts.append(f"Turbulence: {first_level}")
        
        # Add visualization URL
        url = flight_level_data.get("visualization_url")
        if url:
            parts.append(f"\nVisualization: {url}")
        
        return "\n".join(parts)