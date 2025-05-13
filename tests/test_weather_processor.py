#!/usr/bin/env python3
"""
Test script for the Weather Processor.

This script demonstrates the usage of the WeatherProcessor to process
METAR, TAF, and other weather data from various sources.

The output is saved to a file for easier review.
"""

import os
import sys
import logging
import json
import contextlib
from pprint import pprint
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aviation_assistant.data.processors.weather_processor import WeatherProcessor
from aviation_assistant.data.connectors.avwx_connector import AVWXConnector
from aviation_assistant.data.connectors.windy_connector import WindyConnector
from aviation_assistant.config.config_loader import ConfigLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File for output
OUTPUT_FILE = "weather_processor_test_output.txt"

@contextlib.contextmanager
def redirect_stdout_to_file(filename):
    """
    Context manager to redirect stdout to a file.
    
    Args:
        filename: Path to output file
    """
    original_stdout = sys.stdout
    try:
        with open(filename, 'w') as file:
            sys.stdout = file
            yield
    finally:
        sys.stdout = original_stdout

def print_section(title, data=None, width=80):
    """Print a formatted section header and optional data."""
    print("\n" + "=" * width)
    print(f" {title} ".center(width, "="))
    print("=" * width)
    
    if data is not None:
        if isinstance(data, dict) or isinstance(data, list):
            pprint(data)
        else:
            print(data)

def test_process_metar(avwx, processor, icao):
    """Test METAR processing for a given airport."""
    print_section(f"Testing METAR Processing for {icao}")
    
    # Get raw METAR data
    raw_metar = avwx.get_metar(icao)
    print_section("Raw METAR Data Sample", raw_metar)
    
    # Process METAR data
    processed_metar = processor.process_metar(raw_metar)
    
    # Print summary and category
    print_section("METAR Summary", processed_metar.get("summary"))
    print_section("Weather Category", processed_metar.get("weather_category"))
    
    # Print hazards if any
    hazards = processed_metar.get("hazards", [])
    if hazards:
        print_section("Weather Hazards", hazards)
    
    # Print key aspects of processed data
    print_section("Key Processed METAR Data")
    print(f"Station: {processed_metar.get('station')}")
    print(f"Time: {processed_metar.get('time', {}).get('dt')}")
    print(f"Flight Rules: {processed_metar.get('flight_rules')}")
    
    # Print parsed wind
    wind = processed_metar.get("parsed", {}).get("wind", {})
    if wind:
        print(f"Wind: {wind.get('direction')}° at {wind.get('speed')} knots")
        if wind.get("gust"):
            print(f"   Gusting to: {wind.get('gust')} knots")
    
    # Print parsed visibility
    visibility = processed_metar.get("parsed", {}).get("visibility", {})
    if visibility:
        print(f"Visibility: {visibility.get('miles')} miles")
    
    # Print parsed clouds and ceiling
    clouds = processed_metar.get("parsed", {}).get("clouds", [])
    print("Clouds:")
    for cloud in clouds:
        print(f"   {cloud.get('type')} at {cloud.get('height_ft')} feet")
    
    ceiling = processed_metar.get("parsed", {}).get("ceiling", {})
    if ceiling:
        print(f"Ceiling: {ceiling.get('height')} feet ({ceiling.get('type')})")
    
    # Print parsed temperature
    temp = processed_metar.get("parsed", {}).get("temperature", {})
    if temp:
        print(f"Temperature: {temp.get('celsius')}°C / {temp.get('fahrenheit')}°F")
        print(f"Dewpoint: {temp.get('dewpoint_celsius')}°C / {temp.get('dewpoint_fahrenheit')}°F")
        if temp.get("humidity"):
            print(f"Relative Humidity: {temp.get('humidity')}%")
    
    # Print parsed altimeter
    altimeter = processed_metar.get("parsed", {}).get("altimeter", {})
    if altimeter:
        print(f"Altimeter: {altimeter.get('value')} {altimeter.get('units')}")
        print(f"   In inHg: {altimeter.get('inHg')}")
        print(f"   In hPa: {altimeter.get('hPa')}")
    
    # Print parsed weather
    weather = processed_metar.get("parsed", {}).get("weather", [])
    if weather:
        print("Weather Phenomena:")
        for wx in weather:
            print(f"   {wx.get('raw')} (Significant: {wx.get('is_significant')})")
    
    return processed_metar

def test_process_taf(avwx, processor, icao):
    """Test TAF processing for a given airport."""
    print_section(f"Testing TAF Processing for {icao}")
    
    # Get raw TAF data
    raw_taf = avwx.get_taf(icao)
    print_section("Raw TAF Data Sample", raw_taf)
    
    # Process TAF data
    processed_taf = processor.process_taf(raw_taf)
    
    # Print summary and categories
    print_section("TAF Summary", processed_taf.get("summary"))
    print_section("Forecast Categories", processed_taf.get("forecast_categories"))
    
    # Print hazards if any
    hazards = processed_taf.get("hazards", [])
    if hazards:
        print_section("Forecast Hazards", hazards)
    
    # Print key aspects of processed data
    print_section("Key Processed TAF Data")
    print(f"Station: {processed_taf.get('station')}")
    print(f"Forecast Period: {processed_taf.get('forecast_period')}")
    
    # Print forecast periods
    forecasts = processed_taf.get("forecasts", [])
    print_section(f"Forecast Periods ({len(forecasts)})")
    
    for i, forecast in enumerate(forecasts[:3]):  # Print first 3 periods
        time = forecast.get("time", {})
        
        print(f"\nPeriod {i+1}:")
        print(f"   Type: {forecast.get('change_type')}")
        print(f"   From: {time.get('from')}")
        print(f"   To: {time.get('to')}")
        print(f"   Flight Category: {forecast.get('flight_category')}")
        
        # Print wind
        wind = forecast.get("wind", {})
        if wind:
            print(f"   Wind: {wind.get('direction')}° at {wind.get('speed')} knots")
            if wind.get("gust"):
                print(f"      Gusting to: {wind.get('gust')} knots")
        
        # Print visibility
        visibility = forecast.get("visibility", {})
        if visibility:
            print(f"   Visibility: {visibility.get('miles')} miles")
        
        # Print clouds and ceiling
        clouds = forecast.get("clouds", [])
        ceiling = forecast.get("ceiling", {})
        
        cloud_text = []
        for cloud in clouds:
            cloud_text.append(f"{cloud.get('type')} at {cloud.get('height_ft')} feet")
        
        print(f"   Clouds: {', '.join(cloud_text) if cloud_text else 'Clear'}")
        
        if ceiling:
            print(f"   Ceiling: {ceiling.get('height')} feet ({ceiling.get('type')})")
        
        # Print weather
        weather = forecast.get("weather", [])
        wx_text = []
        for wx in weather:
            wx_text.append(wx.get("raw", ""))
        
        print(f"   Weather: {', '.join(wx_text) if wx_text else 'No significant weather'}")
    
    return processed_taf

def test_process_station(avwx, processor, icao):
    """Test station information processing for a given airport."""
    print_section(f"Testing Station Information Processing for {icao}")
    
    # Get raw station data
    raw_station = avwx.get_station(icao)
    print_section("Raw Station Data Sample", raw_station)
    
    # Process station data
    processed_station = processor.process_station_info(raw_station)
    
    # Print key aspects of processed data
    print_section("Key Processed Station Data")
    print(f"ICAO: {processed_station.get('icao')}")
    print(f"IATA: {processed_station.get('iata')}")
    print(f"Name: {processed_station.get('name')}")
    print(f"Location: {processed_station.get('city')}, {processed_station.get('state')}, {processed_station.get('country')}")
    print(f"Elevation: {processed_station.get('elevation_ft')} ft / {processed_station.get('elevation_m')} m")
    
    coordinates = processed_station.get("coordinates", {})
    print(f"Coordinates: {coordinates.get('latitude')}, {coordinates.get('longitude')}")
    
    # Print runways
    runways = processed_station.get("runways", [])
    print(f"\nRunways ({len(runways)}):")
    
    for runway in runways:
        print(f"   {runway.get('ident')}: {runway.get('length_ft')} x {runway.get('width_ft')} ft")
        print(f"      Surface: {runway.get('surface')}")
        print(f"      Heading: {runway.get('heading')}°")
        print(f"      Lights: {runway.get('lights')}")
    
    return processed_station

def test_process_windy(windy, processor, lat, lon):
    """Test Windy forecast data processing for a given location."""
    print_section(f"Testing Windy Forecast Processing for {lat}, {lon}")
    
    # Get raw Windy forecast data
    parameters = ["wind", "temp", "rh", "pressure", "clouds", "precip", "gust"]
    levels = ["surface", "850h", "700h", "500h"]
    
    raw_windy = windy.get_forecast(lat, lon, "gfs", parameters, levels)
    print_section("Raw Windy Data Sample (truncated)", {k: v for k, v in raw_windy.items() if k not in ["forecast_time"]})
    
    # Process Windy data
    processed_windy = processor.process_windy_data(raw_windy)
    
    # Print summary
    print_section("Windy Forecast Summary", processed_windy.get("summary"))
    
    # Print key aspects of processed data
    print_section("Key Processed Windy Data")
    
    # Print surface conditions for the first time period
    surface = processed_windy.get("surface_conditions", {})
    print("\nSurface Conditions (first time period):")
    
    try:
        # Get the first time
        first_time = processed_windy.get("forecast_times", [])[0]
        print(f"Time: {first_time}")
        
        # Get temperature
        temp_c = surface.get("temperature_c", [])[0] if "temperature_c" in surface and surface["temperature_c"] else None
        temp_f = surface.get("temperature_f", [])[0] if "temperature_f" in surface and surface["temperature_f"] else None
        print(f"Temperature: {temp_c}°C / {temp_f}°F")
        
        # Get wind
        wind_speed = surface.get("wind_speed", [])[0] if "wind_speed" in surface and surface["wind_speed"] else None
        wind_speed_kt = surface.get("wind_speed_knots", [])[0] if "wind_speed_knots" in surface and surface["wind_speed_knots"] else None
        wind_dir = surface.get("wind_direction", [])[0] if "wind_direction" in surface and surface["wind_direction"] else None
        print(f"Wind: {wind_dir}° at {wind_speed} m/s ({wind_speed_kt} knots)")
        
        # Get pressure
        pressure = surface.get("pressure", [])[0] if "pressure" in surface and surface["pressure"] else None
        pressure_inhg = surface.get("pressure_inhg", [])[0] if "pressure_inhg" in surface and surface["pressure_inhg"] else None
        print(f"Pressure: {pressure} hPa ({pressure_inhg} inHg)")
        
        # Get humidity
        humidity = surface.get("relative_humidity", [])[0] if "relative_humidity" in surface and surface["relative_humidity"] else None
        print(f"Relative Humidity: {humidity}%")
        
        # Get precipitation
        precip = surface.get("precipitation", [])[0] if "precipitation" in surface and surface["precipitation"] else None
        precip_in = surface.get("precipitation_inches", [])[0] if "precipitation_inches" in surface and surface["precipitation_inches"] else None
        print(f"Precipitation: {precip} mm ({precip_in} inches)")
    except Exception as e:
        print(f"Error printing surface conditions: {str(e)}")
    
    # Print altitude levels
    altitude_levels = processed_windy.get("altitude_levels", {})
    print("\nAltitude Levels:")
    
    for level, data in altitude_levels.items():
        print(f"\n   {level}:")
        
        try:
            # Get wind
            wind_speed = data.get("wind_speed", [])[0] if "wind_speed" in data and data["wind_speed"] else None
            wind_speed_kt = data.get("wind_speed_knots", [])[0] if "wind_speed_knots" in data and data["wind_speed_knots"] else None
            wind_dir = data.get("wind_direction", [])[0] if "wind_direction" in data and data["wind_direction"] else None
            print(f"      Wind: {wind_dir}° at {wind_speed} m/s ({wind_speed_kt} knots)")
            
            # Get temperature
            temp_c = data.get("temperature_c", [])[0] if "temperature_c" in data and data["temperature_c"] else None
            print(f"      Temperature: {temp_c}°C")
            
            # Get humidity
            humidity = data.get("relative_humidity", [])[0] if "relative_humidity" in data and data["relative_humidity"] else None
            print(f"      Relative Humidity: {humidity}%")
        except Exception as e:
            print(f"      Error printing level data: {str(e)}")
    
    return processed_windy

def test_combine_sources(processor, metar, taf, windy, station):
    """Test combining weather data from multiple sources."""
    print_section("Testing Combined Weather Sources")
    
    # Combine data sources
    combined_data = processor.combine_weather_sources(
        avwx_metar=metar,
        avwx_taf=taf,
        windy_data=windy,
        station_info=station
    )
    
    # Print combined summary
    print_section("Combined Weather Summary", combined_data.get("flight_summary"))
    
    # Generate pilot brief
    weather_brief = processor.get_flight_weather_brief(combined_data)
    
    # Print brief sections
    print_section("Flight Weather Brief")
    
    # Print overview
    overview = weather_brief.get("overview", {})
    print("\nOverview:")
    print(f"   Weather Category: {overview.get('flight_category')} ({overview.get('color_code')})")
    print(f"   Summary: {overview.get('summary')}")
    print(f"   Update Time: {overview.get('update_time')}")
    
    if overview.get("significant_hazards"):
        print(f"   Significant Hazards: {', '.join(overview.get('significant_hazards'))}")
    
    # Print current conditions
    current = weather_brief.get("current_conditions", {})
    print("\nCurrent Conditions:")
    print(f"   Wind: {current.get('wind')}")
    print(f"   Visibility: {current.get('visibility')}")
    print(f"   Clouds: {current.get('clouds')}")
    print(f"   Ceiling: {current.get('ceiling')}")
    print(f"   Temperature: {current.get('temperature')}")
    print(f"   Dewpoint: {current.get('dewpoint')}")
    print(f"   Altimeter: {current.get('altimeter')}")
    
    # Print forecast
    forecast = weather_brief.get("forecast", {})
    
    if "taf" in forecast:
        taf = forecast["taf"]
        print("\nTAF Forecast:")
        print(f"   Valid Period: {taf.get('valid_period')}")
        
        for i, period in enumerate(taf.get("periods", [])[:3]):  # Print first 3 periods
            print(f"\n   Period {i+1}:")
            print(f"      Time: {period.get('time')}")
            print(f"      Change Type: {period.get('change_type')}")
            print(f"      Flight Category: {period.get('flight_category')}")
            print(f"      Wind: {period.get('wind')}")
            print(f"      Visibility: {period.get('visibility')}")
            print(f"      Clouds: {period.get('clouds')}")
            print(f"      Weather: {period.get('weather')}")
    
    if "model" in forecast:
        model = forecast["model"]
        print("\nModel Forecast:")
        print(f"   Source: {model.get('source')}")
        
        for i, period in enumerate(model.get("periods", [])[:3]):  # Print first 3 periods
            print(f"\n   Period {i+1}:")
            print(f"      Time: {period.get('time')}")
            print(f"      Temperature: {period.get('temperature')}")
            print(f"      Wind: {period.get('wind')}")
            print(f"      Humidity: {period.get('humidity')}")
            print(f"      Clouds: {period.get('clouds')}")
    
    # Print warnings and recommendations
    warnings = weather_brief.get("warnings", [])
    recommendations = weather_brief.get("recommendations", [])
    
    print_section("Warnings", warnings)
    print_section("Recommendations", recommendations)
    
    return weather_brief

def main():
    """Run weather processor tests."""
    # Load configuration
    config_path = os.getenv('CONFIG_PATH', 'config/config.yaml')
    config = ConfigLoader(config_path).get_config()
    
    # Create processor
    processor = WeatherProcessor()
    
    # Create connectors
    avwx = AVWXConnector(api_key="jLX2iQWBGYttAwYBxiFCUc_4ljrbNOqCNvHe5hjtS8o")
    windy = True
    
    # If windy API key is available, create the connector
    windy = WindyConnector(api_key="ymAvnkZvsbZpxgMVlHabWuDIG6ZgHsJd")
    
    # Test airports
    test_icao = "KJFK"  # JFK International Airport
    
    # Redirect stdout to file
    with redirect_stdout_to_file(OUTPUT_FILE):
        # Test METAR processing
        processed_metar = test_process_metar(avwx, processor, test_icao)
        
        # Test TAF processing
        processed_taf = test_process_taf(avwx, processor, test_icao)
        
        # Test station processing
        processed_station = test_process_station(avwx, processor, test_icao)
        
        # Test Windy processing (if available)
        processed_windy = None
        if windy:
            # Get coordinates from station info
            coords = processed_station.get("coordinates", {})
            lat = coords.get("latitude")
            lon = coords.get("longitude")
            
            if lat and lon:
                processed_windy = test_process_windy(windy, processor, lat, lon)
        
        # Test combined sources
        test_combine_sources(processor, processed_metar, processed_taf, processed_windy, processed_station)
    
    logger.info(f"All weather processor tests completed successfully. Output saved to {OUTPUT_FILE}")
    print(f"Test output saved to {OUTPUT_FILE}")
    
    # Also log to terminal some key information
    print("\nKey weather information:")
    if processed_metar:
        print(f"- METAR Summary: {processed_metar.get('summary')}")
        print(f"- Weather Category: {processed_metar.get('weather_category')}")
    
    if processed_taf:
        print(f"- TAF Summary: {processed_taf.get('summary')}")
        
    if processed_windy:
        print(f"- Windy Forecast Summary: {processed_windy.get('summary')}")

if __name__ == "__main__":
    main()