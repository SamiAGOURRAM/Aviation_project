from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import math

class WeatherProcessor:
    """
    Process and analyze aviation weather data.
    
    Standardizes and combines weather data from multiple sources,
    extracts relevant information, and identifies hazards.
    """
    
    # Weather phenomenon that are significant for aviation
    SIGNIFICANT_WEATHER = {
        # Visibility
        "visibility_below_3sm": {"description": "Low visibility", "severity": "moderate"},
        "visibility_below_1sm": {"description": "Very low visibility", "severity": "high"},
        "visibility_below_half_sm": {"description": "Extremely low visibility", "severity": "extreme"},
        
        # Wind
        "wind_above_15kt": {"description": "Moderate winds", "severity": "low"},
        "wind_above_25kt": {"description": "Strong winds", "severity": "moderate"},
        "wind_above_35kt": {"description": "Very strong winds", "severity": "high"},
        
        # Gusts
        "gust_factor_above_10kt": {"description": "Moderate wind gusts", "severity": "moderate"},
        "gust_factor_above_15kt": {"description": "Strong wind gusts", "severity": "high"},
        
        # Crosswind
        "crosswind_above_10kt": {"description": "Moderate crosswind", "severity": "moderate"},
        "crosswind_above_20kt": {"description": "Strong crosswind", "severity": "high"},
        
        # Ceiling
        "ceiling_below_3000ft": {"description": "Low ceiling", "severity": "low"},
        "ceiling_below_1000ft": {"description": "Very low ceiling", "severity": "moderate"},
        "ceiling_below_500ft": {"description": "Extremely low ceiling", "severity": "high"},
        
        # Precipitation
        "heavy_rain": {"description": "Heavy rain", "severity": "moderate"},
        "heavy_snow": {"description": "Heavy snow", "severity": "high"},
        "freezing_rain": {"description": "Freezing rain", "severity": "extreme"},
        "freezing_drizzle": {"description": "Freezing drizzle", "severity": "high"},
        "ice_pellets": {"description": "Ice pellets/Sleet", "severity": "high"},
        
        # Convective activity
        "thunderstorm": {"description": "Thunderstorm", "severity": "high"},
        "thunderstorm_with_hail": {"description": "Thunderstorm with hail", "severity": "extreme"},
        
        # Obscuration
        "fog": {"description": "Fog", "severity": "moderate"},
        "mist": {"description": "Mist", "severity": "low"},
        "haze": {"description": "Haze", "severity": "low"},
        "smoke": {"description": "Smoke", "severity": "moderate"},
        "dust_storm": {"description": "Dust storm", "severity": "high"},
        "sand_storm": {"description": "Sand storm", "severity": "high"},
        
        # Icing conditions
        "icing_conditions": {"description": "Potential aircraft icing conditions", "severity": "high"},
        "severe_icing_conditions": {"description": "Severe aircraft icing conditions", "severity": "extreme"},
        
        # Turbulence indicators
        "potential_turbulence": {"description": "Potential turbulence", "severity": "moderate"},
        "severe_turbulence": {"description": "Severe turbulence", "severity": "high"},
        "extreme_turbulence": {"description": "Extreme turbulence", "severity": "extreme"},
        
        # Wind shear
        "low_level_wind_shear": {"description": "Low-level wind shear", "severity": "high"},
        "microburst": {"description": "Microburst", "severity": "extreme"},
        
        # Mountain conditions
        "mountain_obscuration": {"description": "Mountain peaks obscured", "severity": "moderate"},
        "mountain_wave": {"description": "Mountain wave activity", "severity": "high"}
    }
    
    # Cloud types of interest for aviation
    CLOUD_TYPES = {
        "CB": {"name": "Cumulonimbus", "description": "Thunderstorm clouds", "hazard_level": "high"},
        "TCU": {"name": "Towering Cumulus", "description": "Strong vertical development", "hazard_level": "moderate"},
        "OVC": {"name": "Overcast", "description": "Complete cloud cover", "hazard_level": "low"},
        "BKN": {"name": "Broken", "description": "5/8 to 7/8 cloud cover", "hazard_level": "low"},
        "SCT": {"name": "Scattered", "description": "3/8 to 4/8 cloud cover", "hazard_level": "very_low"},
        "FEW": {"name": "Few", "description": "1/8 to 2/8 cloud cover", "hazard_level": "very_low"}
    }
    
    # Weather condition codes in METARs and their interpretation
    WEATHER_CODES = {
        # Intensity prefixes
        "-": "Light",
        "+": "Heavy",
        "VC": "Vicinity",
        
        # Descriptors
        "MI": "Shallow",
        "PR": "Partial",
        "BC": "Patches",
        "DR": "Low Drifting",
        "BL": "Blowing",
        "SH": "Shower",
        "TS": "Thunderstorm",
        "FZ": "Freezing",
        
        # Precipitation
        "RA": "Rain",
        "DZ": "Drizzle",
        "SN": "Snow",
        "SG": "Snow Grains",
        "IC": "Ice Crystals",
        "PL": "Ice Pellets",
        "GR": "Hail",
        "GS": "Small Hail",
        "UP": "Unknown Precipitation",
        
        # Obscuration
        "FG": "Fog",
        "BR": "Mist",
        "HZ": "Haze",
        "VA": "Volcanic Ash",
        "DU": "Widespread Dust",
        "SA": "Sand",
        "PY": "Spray",
        
        # Other
        "SQ": "Squall",
        "PO": "Dust/Sand Whirls",
        "DS": "Dust Storm",
        "SS": "Sand Storm",
        "FC": "Funnel Cloud/Tornado",
    }
    
    def __init__(self):
        """Initialize the weather processor."""
        self.logger = logging.getLogger(__name__)
    
    def process_metar(self, metar_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and standardize METAR data.
        
        Args:
            metar_data: Raw METAR data from connector
            
        Returns:
            Processed and standardized METAR information
        """
        if "error" in metar_data:
            self.logger.error(f"Error in METAR data: {metar_data['error']}")
            return {"error": metar_data["error"]}
        
        processed_data = {
            "station_id": metar_data.get("station", ""),
            "raw_text": metar_data.get("raw", ""),
            "time": self._parse_metar_time(metar_data),
            "wind": self._extract_wind_info(metar_data),
            "visibility": self._extract_visibility_info(metar_data),
            "clouds": self._extract_cloud_info(metar_data),
            "weather_conditions": self._extract_weather_conditions(metar_data),
            "temperature": self._extract_temperature_info(metar_data),
            "pressure": self._extract_pressure_info(metar_data),
            "flight_rules": metar_data.get("flight_rules", ""),
            "remarks": self._process_metar_remarks(metar_data.get("remarks", "")),
        }
        
        # Add derived information
        processed_data["hazards"] = self._identify_metar_hazards(processed_data)
        processed_data["summary"] = self._generate_metar_summary(processed_data)
        
        return processed_data
    
    def process_taf(self, taf_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and standardize TAF data.
        
        Args:
            taf_data: Raw TAF data from connector
            
        Returns:
            Processed and standardized TAF information
        """
        if "error" in taf_data:
            self.logger.error(f"Error in TAF data: {taf_data['error']}")
            return {"error": taf_data["error"]}
        
        processed_data = {
            "station_id": taf_data.get("station", ""),
            "raw_text": taf_data.get("raw", ""),
            "issue_time": self._parse_taf_time(taf_data, "time"),
            "valid_from": self._parse_taf_time(taf_data, "start_time"),
            "valid_until": self._parse_taf_time(taf_data, "end_time"),
            "forecast_periods": self._process_taf_periods(taf_data.get("forecast", [])),
            "remarks": taf_data.get("remarks", "")
        }
        
        # Add hazards across all forecast periods
        processed_data["hazards"] = self._identify_taf_hazards(processed_data)
        processed_data["summary"] = self._generate_taf_summary(processed_data)
        
        return processed_data
    
    def process_windy_forecast(self, windy_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and standardize Windy forecast data.
        
        Args:
            windy_data: Raw forecast data from Windy connector
            
        Returns:
            Processed and standardized weather forecast
        """
        if "error" in windy_data:
            self.logger.error(f"Error in Windy data: {windy_data['error']}")
            return {"error": windy_data["error"]}
        
        # Extract the forecast times
        forecast_times = windy_data.get("forecast_time", [])
        
        processed_data = {
            "location": windy_data.get("location", {}),
            "forecast_times": forecast_times,
            "surface_conditions": self._process_surface_conditions(windy_data),
            "upper_air": self._process_upper_air_conditions(windy_data),
            "hazards": self._identify_windy_hazards(windy_data),
            "summary": ""  # Will be generated after processing
        }
        
        # Generate summary based on processed data
        processed_data["summary"] = self._generate_windy_summary(processed_data)
        
        return processed_data
    
    def combine_weather_data(self, 
                          metar_data: Optional[Dict[str, Any]] = None,
                          taf_data: Optional[Dict[str, Any]] = None,
                          windy_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Combine weather data from multiple sources.
        
        Args:
            metar_data: Processed METAR data
            taf_data: Processed TAF data
            windy_data: Processed Windy forecast data
            
        Returns:
            Combined and integrated weather data
        """
        combined_data = {
            "current_conditions": metar_data if metar_data and "error" not in metar_data else None,
            "forecast": taf_data if taf_data and "error" not in taf_data else None,
            "detailed_forecast": windy_data if windy_data and "error" not in windy_data else None,
            "integrated_hazards": [],
            "summary": ""
        }
        
        # Merge hazards from all sources
        all_hazards = []
        
        if metar_data and "hazards" in metar_data and "error" not in metar_data:
            for hazard in metar_data["hazards"]:
                hazard["source"] = "METAR"
                hazard["time_relevance"] = "current"
                all_hazards.append(hazard)
        
        if taf_data and "hazards" in taf_data and "error" not in taf_data:
            for hazard in taf_data["hazards"]:
                hazard["source"] = "TAF"
                if "period" not in hazard:
                    hazard["time_relevance"] = "forecast"
                all_hazards.append(hazard)
        
        if windy_data and "hazards" in windy_data and "error" not in windy_data:
            for hazard in windy_data["hazards"]:
                hazard["source"] = "Windy"
                hazard["time_relevance"] = "forecast"
                all_hazards.append(hazard)
        
        # Sort by severity and time
        combined_data["integrated_hazards"] = sorted(
            all_hazards, 
            key=lambda x: (
                self._severity_to_value(x.get("severity", "low")),
                0 if x.get("time_relevance") == "current" else 1
            ),
            reverse=True
        )
        
        # Generate a comprehensive summary
        combined_data["summary"] = self._generate_combined_summary(combined_data)
        
        return combined_data
    
    def _parse_metar_time(self, metar_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format time information from METAR data."""
        time_info = {}
        
        if "time" in metar_data and metar_data["time"]:
            time_dict = metar_data["time"]
            
            # Extract components
            time_info["datetime"] = self._construct_datetime(
                time_dict.get("year"),
                time_dict.get("month"),
                time_dict.get("day"),
                time_dict.get("hour"),
                time_dict.get("minute")
            )
            
            # Format for display
            if time_info["datetime"]:
                time_info["local"] = time_info["datetime"].strftime("%Y-%m-%d %H:%M %Z")
                time_info["utc"] = time_info["datetime"].strftime("%Y-%m-%d %H:%M UTC")
                time_info["age_hours"] = self._calculate_age_hours(time_info["datetime"])
        
        return time_info
    
    def _parse_taf_time(self, taf_data: Dict[str, Any], time_key: str) -> Dict[str, Any]:
        """Extract and format time information from TAF data."""
        time_info = {}
        
        if time_key in taf_data and taf_data[time_key]:
            time_dict = taf_data[time_key]
            
            # Extract components
            time_info["datetime"] = self._construct_datetime(
                time_dict.get("year"),
                time_dict.get("month"),
                time_dict.get("day"),
                time_dict.get("hour"),
                time_dict.get("minute")
            )
            
            # Format for display
            if time_info["datetime"]:
                time_info["local"] = time_info["datetime"].strftime("%Y-%m-%d %H:%M %Z")
                time_info["utc"] = time_info["datetime"].strftime("%Y-%m-%d %H:%M UTC")
                
                # Calculate forecast age for issue time
                if time_key == "time":
                    time_info["age_hours"] = self._calculate_age_hours(time_info["datetime"])
        
        return time_info
    
    def _construct_datetime(self, year, month, day, hour, minute) -> Optional[datetime]:
        """Construct a datetime object from components."""
        try:
            if all(x is not None for x in [year, month, day, hour]):
                minute = minute or 0
                return datetime(year, month, day, hour, minute)
            return None
        except (ValueError, TypeError):
            return None
    
    def _calculate_age_hours(self, dt: datetime) -> float:
        """Calculate age in hours from a datetime to now."""
        now = datetime.utcnow()
        delta = now - dt
        return delta.total_seconds() / 3600
    
    def _extract_wind_info(self, metar_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract wind information from METAR data."""
        wind_info = {
            "direction": None,
            "speed": None,
            "gust": None,
            "variable_direction": False,
            "direction_range": None,
            "units": "KT"
        }
        
        if "wind_direction" in metar_data:
            wind_info["direction"] = metar_data["wind_direction"].get("value")
            
            # Check for variable direction
            if wind_info["direction"] is not None and wind_info["direction"] == 0:
                wind_info["variable_direction"] = True
                wind_info["direction"] = "VRB"
        
        if "wind_speed" in metar_data:
            wind_info["speed"] = metar_data["wind_speed"].get("value")
            wind_info["units"] = metar_data["wind_speed"].get("units", "KT")
        
        if "wind_gust" in metar_data:
            wind_info["gust"] = metar_data["wind_gust"].get("value")
        
        # Extract variable wind direction range if available
        if "wind_variable_direction" in metar_data and metar_data["wind_variable_direction"]:
            min_dir = metar_data["wind_variable_direction"].get("min", {}).get("value")
            max_dir = metar_data["wind_variable_direction"].get("max", {}).get("value")
            
            if min_dir is not None and max_dir is not None:
                wind_info["direction_range"] = (min_dir, max_dir)
        
        return wind_info
    
    def _extract_visibility_info(self, metar_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract visibility information from METAR data."""
        vis_info = {
            "distance": None,
            "units": "SM",
            "minimum": None,
            "minimum_direction": None
        }
        
        if "visibility" in metar_data:
            vis_data = metar_data["visibility"]
            
            # Handle main visibility
            if isinstance(vis_data, dict):
                vis_info["distance"] = vis_data.get("value")
                vis_info["units"] = vis_data.get("units", "SM")
            
            # Handle visibility as a complex object with minimum visibility
            if isinstance(vis_data, dict) and "minimum" in vis_data:
                min_vis = vis_data["minimum"]
                vis_info["minimum"] = min_vis.get("value")
                vis_info["minimum_direction"] = min_vis.get("direction")
        
        return vis_info
    
    def _extract_cloud_info(self, metar_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract cloud information from METAR data."""
        cloud_info = []
        
        if "clouds" in metar_data and isinstance(metar_data["clouds"], list):
            for cloud in metar_data["clouds"]:
                cloud_entry = {
                    "cover": cloud.get("type"),
                    "altitude": None,
                    "type": None,
                    "description": None,
                    "altitude_ft": None
                }
                
                # Extract altitude if available
                if "altitude" in cloud:
                    cloud_entry["altitude"] = cloud["altitude"].get("value")
                    cloud_entry["altitude_ft"] = cloud_entry["altitude"] * 100 if cloud_entry["altitude"] is not None else None
                
                # Extract cloud type if available
                if "type" in cloud:
                    cloud_entry["type"] = cloud["type"]
                
                # Add description based on cover type
                cover_type = cloud.get("type", "")
                if cover_type in self.CLOUD_TYPES:
                    cloud_entry["description"] = self.CLOUD_TYPES[cover_type]["name"]
                
                cloud_info.append(cloud_entry)
        
        return cloud_info
    
    def _extract_weather_conditions(self, metar_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract weather conditions from METAR data."""
        weather_conditions = []
        
        if "weather" in metar_data and isinstance(metar_data["weather"], list):
            for wx in metar_data["weather"]:
                condition = {
                    "intensity": None,
                    "descriptor": None,
                    "precipitation": None,
                    "obscuration": None,
                    "other": None,
                    "text": ""
                }
                
                # Extract individual components
                if "intensity" in wx:
                    condition["intensity"] = wx["intensity"]
                
                if "descriptor" in wx:
                    condition["descriptor"] = wx["descriptor"]
                
                if "precipitation" in wx:
                    condition["precipitation"] = wx["precipitation"]
                
                if "obscuration" in wx:
                    condition["obscuration"] = wx["obscuration"]
                
                if "other" in wx:
                    condition["other"] = wx["other"]
                
                # Generate human-readable text
                text_parts = []
                
                # Add intensity
                if condition["intensity"] and condition["intensity"] in self.WEATHER_CODES:
                    text_parts.append(self.WEATHER_CODES[condition["intensity"]])
                
                # Add descriptor
                if condition["descriptor"] and condition["descriptor"] in self.WEATHER_CODES:
                    text_parts.append(self.WEATHER_CODES[condition["descriptor"]])
                
                # Add precipitation
                if condition["precipitation"] and condition["precipitation"] in self.WEATHER_CODES:
                    text_parts.append(self.WEATHER_CODES[condition["precipitation"]])
                
                # Add obscuration
                if condition["obscuration"] and condition["obscuration"] in self.WEATHER_CODES:
                    text_parts.append(self.WEATHER_CODES[condition["obscuration"]])
                
                # Add other
                if condition["other"] and condition["other"] in self.WEATHER_CODES:
                    text_parts.append(self.WEATHER_CODES[condition["other"]])
                
                condition["text"] = " ".join(text_parts)
                weather_conditions.append(condition)
        
        return weather_conditions
    
    def _extract_temperature_info(self, metar_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temperature and dewpoint information from METAR data."""
        temp_info = {
            "temperature_c": None,
            "dewpoint_c": None,
            "temperature_f": None,
            "dewpoint_f": None,
            "relative_humidity": None
        }
        
        # Extract temperature
        if "temperature" in metar_data and metar_data["temperature"]:
            temp_c = metar_data["temperature"].get("value")
            if temp_c is not None:
                temp_info["temperature_c"] = temp_c
                temp_info["temperature_f"] = self._celsius_to_fahrenheit(temp_c)
        
        # Extract dewpoint
        if "dewpoint" in metar_data and metar_data["dewpoint"]:
            dewpoint_c = metar_data["dewpoint"].get("value")
            if dewpoint_c is not None:
                temp_info["dewpoint_c"] = dewpoint_c
                temp_info["dewpoint_f"] = self._celsius_to_fahrenheit(dewpoint_c)
        
        # Calculate relative humidity if both temperature and dewpoint are available
        if temp_info["temperature_c"] is not None and temp_info["dewpoint_c"] is not None:
            temp_info["relative_humidity"] = self._calculate_relative_humidity(
                temp_info["temperature_c"], 
                temp_info["dewpoint_c"]
            )
        
        return temp_info
    
    def _extract_pressure_info(self, metar_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract pressure information from METAR data."""
        pressure_info = {
            "altimeter_hg": None,
            "altimeter_mb": None
        }
        
        if "altimeter" in metar_data and metar_data["altimeter"]:
            alt_data = metar_data["altimeter"]
            value = alt_data.get("value")
            units = alt_data.get("units", "hPa")
            
            if value is not None:
                if units == "inHg":
                    pressure_info["altimeter_hg"] = value
                    pressure_info["altimeter_mb"] = self._inches_to_mb(value)
                else:  # Assume hPa or mb
                    pressure_info["altimeter_mb"] = value
                    pressure_info["altimeter_hg"] = self._mb_to_inches(value)
        
        return pressure_info
    
    def _process_metar_remarks(self, remarks: str) -> List[Dict[str, Any]]:
        """Process METAR remarks section."""
        if not remarks:
            return []
        
        # For now, just return the raw remarks
        # This could be enhanced to parse the remarks more intelligently
        return [{"text": remarks, "type": "raw"}]
    
    def _process_taf_periods(self, forecast_periods: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process TAF forecast periods."""
        processed_periods = []
        
        for period in forecast_periods:
            processed_period = {
                "start_time": self._parse_taf_time({"time": period.get("start_time", {})}, "time"),
                "end_time": self._parse_taf_time({"time": period.get("end_time", {})}, "time"),
                "wind": self._extract_wind_info(period),
                "visibility": self._extract_visibility_info(period),
                "clouds": self._extract_cloud_info(period),
                "weather_conditions": self._extract_weather_conditions(period),
                "probability": period.get("probability"),
                "raw_text": period.get("raw_line", ""),
                "type": period.get("type", "FM")  # FM, BECMG, TEMPO, etc.
            }
            
            # Identify hazards for this period
            processed_period["hazards"] = self._identify_metar_hazards(processed_period)
            
            processed_periods.append(processed_period)
        
        return processed_periods
    
    def _process_surface_conditions(self, windy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Windy surface weather conditions."""
        surface_data = {}
        
        # Copy surface conditions from Windy data
        if "surface_conditions" in windy_data:
            surface_data = windy_data["surface_conditions"].copy()
        
        # Otherwise, extract relevant fields from the raw data
        else:
            for key, value in windy_data.items():
                if key.endswith("-surface") or key.startswith("surface_"):
                    clean_key = key.replace("-surface", "").replace("surface_", "")
                    surface_data[clean_key] = value
            
            # Extract wind information if available
            wind_speed = surface_data.get("wind_speed", [])
            wind_direction = surface_data.get("wind_direction", [])
            
            if wind_speed and wind_direction and len(wind_speed) == len(wind_direction):
                surface_data["wind"] = [
                    {"speed": spd, "direction": dir} 
                    for spd, dir in zip(wind_speed, wind_direction)
                ]
        
        return surface_data
    
    def _process_upper_air_conditions(self, windy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Windy upper air conditions at various levels."""
        upper_air = {}
        
        # Extract flight levels if available
        if "flight_levels" in windy_data:
            upper_air["flight_levels"] = windy_data["flight_levels"]
            return upper_air
        
        # Otherwise, try to extract data from pressure levels
        pressure_levels = {}
        
        for key, value in windy_data.items():
            # Check for pressure level keys (e.g., "temp-850h")
            if "-" in key and any(level in key for level in ("850h", "700h", "500h", "300h", "250h", "200h")):
                param, level = key.split("-", 1)
                
                if level not in pressure_levels:
                    pressure_levels[level] = {}
                
                pressure_levels[level][param] = value
        
        # Convert pressure levels to flight levels
        for level, data in pressure_levels.items():
            # Map pressure levels to approximate flight levels
            flight_level = self._pressure_level_to_flight_level(level)
            
            if flight_level:
                if "flight_levels" not in upper_air:
                    upper_air["flight_levels"] = {}
                
                upper_air["flight_levels"][flight_level] = data
        
        return upper_air
    
    def _pressure_level_to_flight_level(self, pressure_level: str) -> Optional[str]:
        """Convert pressure level to approximate flight level."""
        # Simple mapping of common pressure levels to flight levels
        mapping = {
            "850h": "FL050",
            "700h": "FL100",
            "500h": "FL180",
            "300h": "FL300",
            "250h": "FL340",
            "200h": "FL380"
        }
        
        return mapping.get(pressure_level)
    
    def _identify_metar_hazards(self, metar_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify hazards from METAR data."""
        hazards = []
        
        # Check visibility
        if "visibility" in metar_data and metar_data["visibility"]:
            vis = metar_data["visibility"].get("distance")
            
            if vis is not None:
                if vis < 3:
                    hazards.append({
                        "type": "visibility_below_3sm",
                        "description": f"Visibility {vis} SM",
                        "severity": "moderate"
                    })
                if vis < 1:
                    hazards.append({
                        "type": "visibility_below_1sm",
                        "description": f"Visibility {vis} SM",
                        "severity": "high"
                    })
                if vis < 0.5:
                    hazards.append({
                        "type": "visibility_below_half_sm",
                        "description": f"Visibility {vis} SM",
                        "severity": "extreme"
                    })
        
        # Check wind
        if "wind" in metar_data and metar_data["wind"]:
            wind = metar_data["wind"]
            speed = wind.get("speed")
            gust = wind.get("gust")
            
            if speed is not None:
                if speed > 15:
                    hazards.append({
                        "type": "wind_above_15kt",
                        "description": f"Wind {speed} KT",
                        "severity": "low"
                    })
                if speed > 25:
                    hazards.append({
                        "type": "wind_above_25kt",
                        "description": f"Wind {speed} KT",
                        "severity": "moderate"
                    })
                if speed > 35:
                    hazards.append({
                        "type": "wind_above_35kt",
                        "description": f"Wind {speed} KT",
                        "severity": "high"
                    })
            
            # Check gust factor
            if gust is not None and speed is not None:
                gust_factor = gust - speed
                
                if gust_factor > 10:
                    hazards.append({
                        "type": "gust_factor_above_10kt",
                        "description": f"Gust factor {gust_factor} KT",
                        "severity": "moderate"
                    })
                if gust_factor > 15:
                    hazards.append({
                        "type": "gust_factor_above_15kt",
                        "description": f"Gust factor {gust_factor} KT",
                        "severity": "high"
                    })
        
        # Check ceiling
        lowest_ceiling = None
        
        if "clouds" in metar_data and metar_data["clouds"]:
            for cloud in metar_data["clouds"]:
                # Check if it's a ceiling (BKN or OVC)
                cover = cloud.get("cover")
                altitude_ft = cloud.get("altitude_ft")
                
                if cover in ["BKN", "OVC"] and altitude_ft is not None:
                    if lowest_ceiling is None or altitude_ft < lowest_ceiling:
                        lowest_ceiling = altitude_ft
                
                # Check for convective clouds (CB, TCU)
                cloud_type = cloud.get("type")
                
                if cloud_type == "CB":
                    hazards.append({
                        "type": "thunderstorm",
                        "description": "Cumulonimbus clouds",
                        "severity": "high"
                    })
                elif cloud_type == "TCU":
                    hazards.append({
                        "type": "potential_turbulence",
                        "description": "Towering cumulus clouds",
                        "severity": "moderate"
                    })
            
            # Add ceiling hazards
            if lowest_ceiling is not None:
                if lowest_ceiling < 3000:
                    hazards.append({
                        "type": "ceiling_below_3000ft",
                        "description": f"Ceiling at {lowest_ceiling} ft",
                        "severity": "low"
                    })
                if lowest_ceiling < 1000:
                    hazards.append({
                        "type": "ceiling_below_1000ft",
                        "description": f"Ceiling at {lowest_ceiling} ft",
                        "severity": "moderate"
                    })
                if lowest_ceiling < 500:
                    hazards.append({
                        "type": "ceiling_below_500ft",
                        "description": f"Ceiling at {lowest_ceiling} ft",
                        "severity": "high"
                    })
        
        # Check weather conditions
        if "weather_conditions" in metar_data and metar_data["weather_conditions"]:
            for condition in metar_data["weather_conditions"]:
                intensity = condition.get("intensity")
                descriptor = condition.get("descriptor")
                precipitation = condition.get("precipitation")
                obscuration = condition.get("obscuration")
                
                # Check for thunderstorms
                if descriptor == "TS":
                    hazard = {
                        "type": "thunderstorm",
                        "description": condition.get("text", "Thunderstorm"),
                        "severity": "high"
                    }
                    
                    # Add hail information if present
                    if precipitation in ["GR", "GS"]:
                        hazard["type"] = "thunderstorm_with_hail"
                        hazard["severity"] = "extreme"
                    
                    hazards.append(hazard)
                
                # Check for heavy precipitation
                if intensity == "+" and precipitation:
                    precip_type = self.WEATHER_CODES.get(precipitation, precipitation)
                    
                    hazard = {
                        "type": f"heavy_{precipitation.lower()}",
                        "description": f"Heavy {precip_type}",
                        "severity": "moderate"
                    }
                    
                    # Adjust severity for snow and freezing precipitation
                    if precipitation == "SN":
                        hazard["severity"] = "high"
                    
                    hazards.append(hazard)
                
                # Check for freezing precipitation
                if descriptor == "FZ" and precipitation:
                    precip_type = self.WEATHER_CODES.get(precipitation, precipitation)
                    
                    hazard = {
                        "type": f"freezing_{precipitation.lower()}",
                        "description": f"Freezing {precip_type}",
                        "severity": "high"
                    }
                    
                    # Freezing rain is particularly dangerous
                    if precipitation == "RA":
                        hazard["severity"] = "extreme"
                    
                    hazards.append(hazard)
                
                # Check for obscuration
                if obscuration:
                    hazard_type = obscuration.lower()
                    
                    if hazard_type in self.SIGNIFICANT_WEATHER:
                        hazards.append({
                            "type": hazard_type,
                            "description": condition.get("text", self.SIGNIFICANT_WEATHER[hazard_type]["description"]),
                            "severity": self.SIGNIFICANT_WEATHER[hazard_type]["severity"]
                        })
        
        # Check for potential icing conditions
        if "temperature" in metar_data and "clouds" in metar_data:
            temp_info = metar_data["temperature"]
            temp_c = temp_info.get("temperature_c")
            
            if temp_c is not None and 0 >= temp_c >= -15 and metar_data["clouds"]:
                # Look for clouds in the icing temperature range
                for cloud in metar_data["clouds"]:
                    if cloud.get("cover") in ["BKN", "OVC"]:
                        hazards.append({
                            "type": "icing_conditions",
                            "description": f"Potential icing with temperature {temp_c}°C and cloud cover",
                            "severity": "high"
                        })
                        break
        
        return hazards
    
    def _identify_taf_hazards(self, taf_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify hazards from TAF data."""
        all_hazards = []
        
        if "forecast_periods" in taf_data and taf_data["forecast_periods"]:
            for period_idx, period in enumerate(taf_data["forecast_periods"]):
                if "hazards" in period and period["hazards"]:
                    for hazard in period["hazards"]:
                        # Copy the hazard and add period information
                        hazard_copy = hazard.copy()
                        
                        # Add period information
                        start_time = period.get("start_time", {}).get("datetime")
                        end_time = period.get("end_time", {}).get("datetime")
                        
                        # Format times for display
                        start_str = start_time.strftime("%Y-%m-%d %H:%M UTC") if start_time else "Unknown"
                        end_str = end_time.strftime("%Y-%m-%d %H:%M UTC") if end_time else "Unknown"
                        
                        hazard_copy["period"] = {
                            "index": period_idx,
                            "type": period.get("type", "FM"),
                            "probability": period.get("probability"),
                            "start_time": start_str,
                            "end_time": end_str
                        }
                        
                        all_hazards.append(hazard_copy)
        
        return all_hazards
    
    def _identify_windy_hazards(self, windy_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify hazards from Windy forecast data."""
        hazards = []
        
        # Check surface conditions
        surface = self._process_surface_conditions(windy_data)
        
        # Check winds
        if "wind_speed" in surface and surface["wind_speed"]:
            forecast_times = windy_data.get("forecast_time", [])
            
            for i, speed in enumerate(surface["wind_speed"]):
                if speed is None:
                    continue
                    
                time_str = forecast_times[i].strftime("%Y-%m-%d %H:%M UTC") if i < len(forecast_times) else "Unknown"
                
                if speed > 25:
                    hazards.append({
                        "type": "wind_above_25kt",
                        "description": f"Wind {speed} m/s at {time_str}",
                        "severity": "moderate",
                        "time_index": i,
                        "forecast_time": time_str
                    })
                if speed > 35:
                    hazards.append({
                        "type": "wind_above_35kt",
                        "description": f"Wind {speed} m/s at {time_str}",
                        "severity": "high",
                        "time_index": i,
                        "forecast_time": time_str
                    })
        
        # Check for thunderstorms using CAPE
        if "cape_value" in windy_data.get("derived", {}).get("turbulence", {}):
            cape_values = windy_data["derived"]["turbulence"]["cape_value"]
            forecast_times = windy_data.get("forecast_time", [])
            
            for i, cape in enumerate(cape_values):
                if cape is None:
                    continue
                    
                time_str = forecast_times[i].strftime("%Y-%m-%d %H:%M UTC") if i < len(forecast_times) else "Unknown"
                
                if cape > 1000:
                    hazards.append({
                        "type": "potential_turbulence",
                        "description": f"High CAPE ({cape}) at {time_str}",
                        "severity": "moderate",
                        "time_index": i,
                        "forecast_time": time_str
                    })
                if cape > 2000:
                    hazards.append({
                        "type": "thunderstorm",
                        "description": f"Very high CAPE ({cape}) at {time_str}",
                        "severity": "high",
                        "time_index": i,
                        "forecast_time": time_str
                    })
        
        # Check turbulence
        if "level" in windy_data.get("derived", {}).get("turbulence", {}):
            turbulence_levels = windy_data["derived"]["turbulence"]["level"]
            forecast_times = windy_data.get("forecast_time", [])
            
            for i, level in enumerate(turbulence_levels):
                if level is None:
                    continue
                    
                time_str = forecast_times[i].strftime("%Y-%m-%d %H:%M UTC") if i < len(forecast_times) else "Unknown"
                
                if level == "moderate":
                    hazards.append({
                        "type": "potential_turbulence",
                        "description": f"Moderate turbulence at {time_str}",
                        "severity": "moderate",
                        "time_index": i,
                        "forecast_time": time_str
                    })
                elif level in ["moderate-severe", "severe"]:
                    hazards.append({
                        "type": "severe_turbulence",
                        "description": f"Severe turbulence at {time_str}",
                        "severity": "high",
                        "time_index": i,
                        "forecast_time": time_str
                    })
        
        # Check for icing risk
        if "risk_level" in windy_data.get("derived", {}).get("icing_risk", {}):
            icing_risks = windy_data["derived"]["icing_risk"]["risk_level"]
            forecast_times = windy_data.get("forecast_time", [])
            
            for i, risk in enumerate(icing_risks):
                if risk is None:
                    continue
                    
                time_str = forecast_times[i].strftime("%Y-%m-%d %H:%M UTC") if i < len(forecast_times) else "Unknown"
                
                if risk in ["moderate", "high"]:
                    hazards.append({
                        "type": "icing_conditions",
                        "description": f"{risk.capitalize()} icing risk at {time_str}",
                        "severity": "high",
                        "time_index": i,
                        "forecast_time": time_str
                    })
        
        return hazards
    
    def _generate_metar_summary(self, metar_data: Dict[str, Any]) -> str:
        """Generate a human-readable summary from METAR data."""
        parts = []
        
        # Add station ID
        station_id = metar_data.get("station_id", "")
        if station_id:
            parts.append(f"Weather at {station_id}")
        
        # Add time information
        time_info = metar_data.get("time", {})
        if "utc" in time_info:
            parts.append(f"observed at {time_info['utc']}")
        elif "age_hours" in time_info:
            age = time_info["age_hours"]
            age_text = f"{age:.1f} hours ago" if age < 24 else f"{age/24:.1f} days ago"
            parts.append(f"observed {age_text}")
        
        # Add flight rules
        flight_rules = metar_data.get("flight_rules", "")
        if flight_rules:
            parts.append(f"with {flight_rules} conditions")
        
        # Add wind information
        wind = metar_data.get("wind", {})
        if "speed" in wind and wind["speed"] is not None:
            direction = wind.get("direction")
            speed = wind["speed"]
            gust = wind.get("gust")
            
            if direction == "VRB":
                wind_text = f"Variable winds at {speed} KT"
            elif direction is not None:
                wind_text = f"Wind from {direction}° at {speed} KT"
            else:
                wind_text = f"Wind at {speed} KT"
            
            if gust is not None:
                wind_text += f", gusting to {gust} KT"
            
            parts.append(wind_text)
        
        # Add visibility
        vis = metar_data.get("visibility", {})
        if "distance" in vis and vis["distance"] is not None:
            vis_text = f"Visibility {vis['distance']} {vis.get('units', 'SM')}"
            parts.append(vis_text)
        
        # Add weather phenomena
        weather = metar_data.get("weather_conditions", [])
        if weather:
            wx_texts = [w.get("text", "") for w in weather if w.get("text")]
            if wx_texts:
                parts.append(", ".join(wx_texts))
        
        # Add cloud information
        clouds = metar_data.get("clouds", [])
        if clouds:
            cloud_texts = []
            
            for cloud in clouds:
                cover = cloud.get("cover")
                altitude = cloud.get("altitude_ft")
                cloud_type = cloud.get("type")
                
                if cover and altitude:
                    cloud_text = f"{cover} at {altitude} ft"
                    if cloud_type:
                        cloud_text += f" ({cloud_type})"
                    
                    cloud_texts.append(cloud_text)
            
            if cloud_texts:
                parts.append("; ".join(cloud_texts))
        
        # Add temperature and dewpoint
        temp = metar_data.get("temperature", {})
        if "temperature_c" in temp and temp["temperature_c"] is not None:
            temp_text = f"Temperature {temp['temperature_c']}°C"
            
            if "dewpoint_c" in temp and temp["dewpoint_c"] is not None:
                temp_text += f", dewpoint {temp['dewpoint_c']}°C"
            
            parts.append(temp_text)
        
        # Add pressure
        pressure = metar_data.get("pressure", {})
        if "altimeter_hg" in pressure and pressure["altimeter_hg"] is not None:
            pressure_text = f"Altimeter {pressure['altimeter_hg']:.2f} inHg"
            parts.append(pressure_text)
        
        # Add hazard summary if there are any
        hazards = metar_data.get("hazards", [])
        if hazards:
            # Sort by severity
            sorted_hazards = sorted(
                hazards, 
                key=lambda x: self._severity_to_value(x.get("severity", "low")), 
                reverse=True
            )
            
            # Take top 3 most severe hazards
            top_hazards = sorted_hazards[:3]
            hazard_texts = [h.get("description", "") for h in top_hazards]
            
            if hazard_texts:
                parts.append("WARNING: " + "; ".join(hazard_texts))
        
        return ". ".join(parts)
    
    def _generate_taf_summary(self, taf_data: Dict[str, Any]) -> str:
        """Generate a human-readable summary from TAF data."""
        parts = []
        
        # Add station ID
        station_id = taf_data.get("station_id", "")
        if station_id:
            parts.append(f"Forecast for {station_id}")
        
        # Add validity period
        valid_from = taf_data.get("valid_from", {}).get("utc")
        valid_until = taf_data.get("valid_until", {}).get("utc")
        
        if valid_from and valid_until:
            parts.append(f"valid from {valid_from} until {valid_until}")
        
        # Add issue time
        issue_time = taf_data.get("issue_time", {}).get("utc")
        if issue_time:
            parts.append(f"issued at {issue_time}")
        
        # Add forecast periods
        periods = taf_data.get("forecast_periods", [])
        
        if periods:
            period_summaries = []
            
            for period in periods:
                period_type = period.get("type", "FM")
                start_time = period.get("start_time", {}).get("datetime")
                start_str = start_time.strftime("%d/%H:%MZ") if start_time else "unknown time"
                
                probability = period.get("probability")
                prob_text = f"{probability}% probability of " if probability else ""
                
                # Summarize this period's weather
                period_parts = []
                
                # Add wind
                wind = period.get("wind", {})
                if "speed" in wind and wind["speed"] is not None:
                    direction = wind.get("direction")
                    speed = wind["speed"]
                    gust = wind.get("gust")
                    
                    if direction == "VRB":
                        wind_text = f"variable winds at {speed} KT"
                    elif direction is not None:
                        wind_text = f"wind from {direction}° at {speed} KT"
                    else:
                        wind_text = f"wind at {speed} KT"
                    
                    if gust is not None:
                        wind_text += f", gusting to {gust} KT"
                    
                    period_parts.append(wind_text)
                
                # Add visibility
                vis = period.get("visibility", {})
                if "distance" in vis and vis["distance"] is not None:
                    vis_text = f"visibility {vis['distance']} {vis.get('units', 'SM')}"
                    period_parts.append(vis_text)
                
                # Add weather phenomena
                weather = period.get("weather_conditions", [])
                if weather:
                    wx_texts = [w.get("text", "") for w in weather if w.get("text")]
                    if wx_texts:
                        period_parts.append(", ".join(wx_texts))
                
                # Add cloud information
                clouds = period.get("clouds", [])
                if clouds:
                    cloud_texts = []
                    
                    for cloud in clouds:
                        cover = cloud.get("cover")
                        altitude = cloud.get("altitude_ft")
                        cloud_type = cloud.get("type")
                        
                        if cover and altitude:
                            cloud_text = f"{cover} at {altitude} ft"
                            if cloud_type:
                                cloud_text += f" ({cloud_type})"
                            
                            cloud_texts.append(cloud_text)
                    
                    if cloud_texts:
                        period_parts.append("; ".join(cloud_texts))
                
                # Create the period summary
                period_text = f"From {start_str}: {prob_text}" + "; ".join(period_parts)
                
                # Add any hazards
                hazards = period.get("hazards", [])
                if hazards:
                    # Sort by severity
                    sorted_hazards = sorted(
                        hazards, 
                        key=lambda x: self._severity_to_value(x.get("severity", "low")), 
                        reverse=True
                    )
                    
                    # Take top 2 most severe hazards
                    top_hazards = sorted_hazards[:2]
                    hazard_texts = [h.get("description", "") for h in top_hazards]
                    
                    if hazard_texts:
                        period_text += " (WARNING: " + "; ".join(hazard_texts) + ")"
                
                period_summaries.append(period_text)
            
            # Add period summaries
            parts.extend(period_summaries)
        
        # Add overall hazard summary if there are any
        hazards = taf_data.get("hazards", [])
        if hazards:
            # Sort by severity
            sorted_hazards = sorted(
                hazards, 
                key=lambda x: self._severity_to_value(x.get("severity", "low")), 
                reverse=True
            )
            
            # Take top 3 most severe hazards
            top_hazards = sorted_hazards[:3]
            
            hazard_texts = []
            for h in top_hazards:
                desc = h.get("description", "")
                period_info = h.get("period", {})
                
                if period_info:
                    period_type = period_info.get("type", "")
                    start_time = period_info.get("start_time", "")
                    
                    if period_type and start_time:
                        desc += f" ({period_type} {start_time})"
                
                hazard_texts.append(desc)
            
            if hazard_texts:
                parts.append("Overall hazards: " + "; ".join(hazard_texts))
        
        return ". ".join(parts)
    
    def _generate_windy_summary(self, windy_data: Dict[str, Any]) -> str:
        """Generate a human-readable summary from Windy forecast data."""
        parts = []
        
        # Get location information
        location = windy_data.get("location", {})
        lat = location.get("lat")
        lon = location.get("lon")
        
        if lat is not None and lon is not None:
            parts.append(f"Weather forecast for location {lat:.4f}, {lon:.4f}")
        
        # Get forecast time information
        forecast_times = windy_data.get("forecast_times", [])
        
        if forecast_times:
            first_time = forecast_times[0]
            last_time = forecast_times[-1]
            
            first_str = first_time.strftime("%Y-%m-%d %H:%M UTC") if isinstance(first_time, datetime) else str(first_time)
            last_str = last_time.strftime("%Y-%m-%d %H:%M UTC") if isinstance(last_time, datetime) else str(last_time)
            
            parts.append(f"Forecast period from {first_str} to {last_str}")
        
        # Summarize surface conditions
        surface = windy_data.get("surface_conditions", {})
        
        if surface:
            surface_parts = []
            
            # Wind
            if "wind_speed" in surface and surface["wind_speed"]:
                speeds = surface["wind_speed"]
                directions = surface.get("wind_direction", [])
                
                # Calculate average wind
                valid_speeds = [s for s in speeds if s is not None]
                
                if valid_speeds:
                    avg_speed = sum(valid_speeds) / len(valid_speeds)
                    max_speed = max(valid_speeds)
                    
                    wind_text = f"average wind {avg_speed:.1f} m/s"
                    
                    if max_speed > avg_speed * 1.5:
                        wind_text += f", with gusts up to {max_speed:.1f} m/s"
                    
                    surface_parts.append(wind_text)
            
            # Temperature
            if "temperature_c" in surface and surface["temperature_c"]:
                temps = surface["temperature_c"]
                valid_temps = [t for t in temps if t is not None]
                
                if valid_temps:
                    min_temp = min(valid_temps)
                    max_temp = max(valid_temps)
                    
                    if max_temp - min_temp > 5:
                        temp_text = f"temperatures ranging from {min_temp:.1f}°C to {max_temp:.1f}°C"
                    else:
                        avg_temp = sum(valid_temps) / len(valid_temps)
                        temp_text = f"temperature around {avg_temp:.1f}°C"
                    
                    surface_parts.append(temp_text)
            
            # Precipitation
            if "precipitation" in surface and surface["precipitation"]:
                precip = surface["precipitation"]
                valid_precip = [p for p in precip if p is not None]
                
                if valid_precip:
                    max_precip = max(valid_precip)
                    
                    if max_precip > 0.5:
                        precip_text = f"precipitation expected, with maximum of {max_precip:.1f} mm/h"
                        surface_parts.append(precip_text)
            
            # Cloud cover
            cloud_keys = ["cloud_cover", "clouds"]
            for key in cloud_keys:
                if key in surface and surface[key]:
                    clouds = surface[key]
                    valid_clouds = [c for c in clouds if c is not None]
                    
                    if valid_clouds:
                        avg_cloud = sum(valid_clouds) / len(valid_clouds)
                        
                        if avg_cloud > 80:
                            cloud_text = "overcast conditions"
                        elif avg_cloud > 50:
                            cloud_text = "mostly cloudy conditions"
                        elif avg_cloud > 30:
                            cloud_text = "partly cloudy conditions"
                        else:
                            cloud_text = "mostly clear conditions"
                        
                        surface_parts.append(cloud_text)
                    
                    break
            
            if surface_parts:
                parts.append("Surface conditions: " + "; ".join(surface_parts))
        
        # Summarize upper air conditions
        upper_air = windy_data.get("upper_air", {})
        
        if "flight_levels" in upper_air and upper_air["flight_levels"]:
            flight_levels = upper_air["flight_levels"]
            
            fl_parts = []
            
            # Pick one or two representative flight levels
            selected_levels = list(flight_levels.keys())
            
            if len(selected_levels) > 2:
                selected_levels = [selected_levels[0], selected_levels[-1]]
            
            for fl in selected_levels:
                fl_data = flight_levels[fl]
                
                fl_text_parts = []
                
                # Wind
                if "wind_speed" in fl_data and fl_data["wind_speed"]:
                    speeds = fl_data["wind_speed"]
                    valid_speeds = [s for s in speeds if s is not None]
                    
                    if valid_speeds:
                        avg_speed = sum(valid_speeds) / len(valid_speeds)
                        wind_text = f"wind averaging {avg_speed:.1f} m/s"
                        fl_text_parts.append(wind_text)
                
                # Temperature
                if "temperature_c" in fl_data and fl_data["temperature_c"]:
                    temps = fl_data["temperature_c"]
                    valid_temps = [t for t in temps if t is not None]
                    
                    if valid_temps:
                        avg_temp = sum(valid_temps) / len(valid_temps)
                        temp_text = f"temperature around {avg_temp:.1f}°C"
                        fl_text_parts.append(temp_text)
                
                if fl_text_parts:
                    fl_parts.append(f"{fl}: " + "; ".join(fl_text_parts))
            
            if fl_parts:
                parts.append("Upper air conditions: " + ". ".join(fl_parts))
        
        # Add hazard summary if there are any
        hazards = windy_data.get("hazards", [])
        if hazards:
            # Sort by severity
            sorted_hazards = sorted(
                hazards, 
                key=lambda x: self._severity_to_value(x.get("severity", "low")), 
                reverse=True
            )
            
            # Take top 3 most severe hazards
            top_hazards = sorted_hazards[:3]
            hazard_texts = [h.get("description", "") for h in top_hazards]
            
            if hazard_texts:
                parts.append("Potential hazards: " + "; ".join(hazard_texts))
        
        return ". ".join(parts)
    
    def _generate_combined_summary(self, combined_data: Dict[str, Any]) -> str:
        """Generate a comprehensive summary from combined weather data."""
        parts = []
        
        # Get current conditions from METAR
        metar = combined_data.get("current_conditions")
        if metar and "summary" in metar:
            parts.append("Current conditions: " + metar["summary"])
        
        # Get forecast from TAF
        taf = combined_data.get("forecast")
        if taf and "summary" in taf:
            # Extract just the first part of the TAF summary (not all periods)
            taf_summary = taf["summary"].split(". From ")[0]
            parts.append("Forecast: " + taf_summary)
        
        # Summarize hazards
        hazards = combined_data.get("integrated_hazards", [])
        
        if hazards:
            current_hazards = [h for h in hazards if h.get("time_relevance") == "current"]
            forecast_hazards = [h for h in hazards if h.get("time_relevance") == "forecast"]
            
            # Summarize current hazards
            if current_hazards:
                texts = []
                for h in current_hazards[:3]:  # Top 3 current hazards
                    texts.append(h.get("description", ""))
                
                if texts:
                    parts.append("Current hazards: " + "; ".join(texts))
            
            # Summarize forecast hazards
            if forecast_hazards:
                texts = []
                for h in forecast_hazards[:3]:  # Top 3 forecast hazards
                    period_info = h.get("period", {})
                    desc = h.get("description", "")
                    
                    if period_info:
                        start_time = period_info.get("start_time", "")
                        if start_time:
                            desc += f" ({start_time})"
                    
                    texts.append(desc)
                
                if texts:
                    parts.append("Forecast hazards: " + "; ".join(texts))
        
        return ". ".join(parts)
    
    def _severity_to_value(self, severity: str) -> int:
        """Convert severity string to numeric value for sorting."""
        severity_map = {
            "extreme": 4,
            "high": 3,
            "moderate": 2,
            "low": 1,
            "very_low": 0
        }
        
        return severity_map.get(severity.lower(), 0)
    
    def _celsius_to_fahrenheit(self, celsius: float) -> float:
        """Convert Celsius to Fahrenheit."""
        return celsius * 9/5 + 32
    
    def _fahrenheit_to_celsius(self, fahrenheit: float) -> float:
        """Convert Fahrenheit to Celsius."""
        return (fahrenheit - 32) * 5/9
    
    def _mb_to_inches(self, mb: float) -> float:
        """Convert millibars (hPa) to inches of mercury."""
        return mb * 0.02953
    
    def _inches_to_mb(self, inches: float) -> float:
        """Convert inches of mercury to millibars (hPa)."""
        return inches / 0.02953
    
    def _calculate_relative_humidity(self, temp_c: float, dewpoint_c: float) -> float:
        """Calculate relative humidity from temperature and dewpoint."""
        # Using the Magnus approximation formula
        # Constants for water vapor in air
        b = 17.27
        c = 237.7  # °C
        
        # Calculate the actual vapor pressure
        gamma_d = (b * dewpoint_c) / (c + dewpoint_c)
        e_d = 6.112 * math.exp(gamma_d)
        
        # Calculate the saturation vapor pressure
        gamma_t = (b * temp_c) / (c + temp_c)
        e_s = 6.112 * math.exp(gamma_t)
        
        # Calculate the relative humidity
        rh = 100.0 * (e_d / e_s)
        
        return round(rh, 1)

