import os
import sys
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aviation_assistant.data.connectors.windy_connector import WindyConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def pretty_print(data: Dict[str, Any]) -> None:
    """Print dictionary in a readable format."""
    print(json.dumps(data, indent=4, default=str))

def plot_wind_data(wind_data: Dict[str, Any], title: str = "Wind Data") -> None:
    """
    Create a simple plot of wind data over time.
    
    Args:
        wind_data: Wind data dictionary from WindyConnector
        title: Plot title
    """
    if "levels" not in wind_data or not wind_data["levels"]:
        logger.error("No wind level data to plot")
        return
        
    # Get timestamps
    timestamps = wind_data.get("forecast_time", [])
    if not timestamps:
        logger.error("No timestamp data found")
        return
        
    # Format x-axis labels
    x_labels = [t.strftime("%m-%d %H:%M") if t else "" for t in timestamps]
    x = range(len(x_labels))
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(title)
    
    # Plot wind speed for each level
    for level_name, level_data in wind_data["levels"].items():
        if "wind_speed" in level_data:
            ax1.plot(x, level_data["wind_speed"], label=f"{level_name}")
    
    ax1.set_ylabel("Wind Speed (m/s)")
    ax1.set_title("Wind Speed by Flight Level")
    ax1.grid(True)
    ax1.legend()
    
    # Plot wind direction for first level only (to avoid clutter)
    first_level = next(iter(wind_data["levels"]))
    if "wind_direction" in wind_data["levels"][first_level]:
        directions = wind_data["levels"][first_level]["wind_direction"]
        ax2.plot(x, directions, 'o-', label=f"{first_level} Direction")
        
    ax2.set_ylabel("Wind Direction (degrees)")
    ax2.set_title("Wind Direction")
    ax2.set_ylim(0, 360)
    ax2.set_yticks(np.arange(0, 361, 45))
    ax2.set_yticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'])
    ax2.grid(True)
    
    # Set x-axis labels
    plt.xticks(x[::3], x_labels[::3], rotation=45)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("wind_data_plot.png")
    logger.info("Wind data plot saved to wind_data_plot.png")
    plt.close()

def plot_route_winds(route_winds: Dict[str, Any], title: str = "Route Winds") -> None:
    """
    Create a plot of wind conditions along a route.
    
    Args:
        route_winds: Route winds data from WindyConnector
        title: Plot title
    """
    if "points" not in route_winds or not route_winds["points"]:
        logger.error("No route point data to plot")
        return
        
    # Extract data for plotting
    positions = []
    headwinds = []
    crosswinds = []
    wind_speeds = []
    
    for point in route_winds["points"]:
        if "position" in point:
            positions.append(point["position"])
            
            # Get the first timestamp's data for each point
            if "headwind_component" in point and point["headwind_component"]:
                headwinds.append(point["headwind_component"][0])
            else:
                headwinds.append(None)
                
            if "crosswind_component" in point and point["crosswind_component"]:
                crosswinds.append(point["crosswind_component"][0])
            else:
                crosswinds.append(None)
                
            if "wind_speed" in point and point["wind_speed"]:
                wind_speeds.append(point["wind_speed"][0])
            else:
                wind_speeds.append(None)
    
    if not positions:
        logger.error("No position data found")
        return
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"{title} - {route_winds.get('flight_level', '')}")
    
    # Plot wind components
    ax1.plot(positions, headwinds, 'b-', label="Headwind (+) / Tailwind (-)")
    ax1.plot(positions, crosswinds, 'r-', label="Crosswind")
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_ylabel("Wind Component (m/s)")
    ax1.set_title("Wind Components Along Route")
    ax1.grid(True)
    ax1.legend()
    
    # Plot total wind speed
    ax2.plot(positions, wind_speeds, 'g-', label="Wind Speed")
    ax2.set_xlabel("Route Position")
    ax2.set_ylabel("Wind Speed (m/s)")
    ax2.set_title("Total Wind Speed Along Route")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("route_winds_plot.png")
    logger.info("Route winds plot saved to route_winds_plot.png")
    plt.close()

def test_windy_connector():
    """Test the Windy connector."""
    logger.info("============ TESTING WINDY CONNECTOR ============")
    
    # Get API key from environment variable
    api_key = os.environ.get("WINDY_API_KEY")
    if not api_key:
        logger.error("WINDY_API_KEY environment variable not set")
        return
    
    # Initialize the connector
    windy = WindyConnector(api_key=api_key)
    
    # TEST 1: Get basic forecast for a location (New York City)
    logger.info("\n=== TEST 1: Basic Weather Forecast ===")
    try:
        coords = (40.7128, -74.0060)  # New York City
        parameters = ["wind", "temp", "pressure", "rh", "clouds", "precip"]
        levels = ["surface", "850h", "500h"]
        
        forecast = windy.get_forecast(
            lat=coords[0],
            lon=coords[1],
            model="gfs",
            parameters=parameters,
            levels=levels
        )
        
        if "error" in forecast:
            logger.error(f"Error getting forecast: {forecast['error']}")
        else:
            logger.info(f"Successfully retrieved forecast for New York City")
            logger.info(f"Forecast times: {len(forecast.get('forecast_time', []))} timestamps")
            logger.info(f"Available parameters: {[k for k in forecast.keys() if k not in ['forecast_time', 'units']]}")
            
            # Sample temperature data
            temp_key = next((k for k in forecast.keys() if k.startswith("temp-surface")), None)
            if temp_key and temp_key in forecast:
                temps_k = forecast[temp_key][:3]  # First 3 values
                temps_c = [round(t - 273.15, 1) if t is not None else None for t in temps_k]
                logger.info(f"Sample surface temperatures (°C): {temps_c}")
    except Exception as e:
        logger.error(f"Error in TEST 1: {str(e)}")
    
    # TEST 2: Get aviation weather at different flight levels
    logger.info("\n=== TEST 2: Aviation Weather Data ===")
    try:
        coords = (40.7128, -74.0060)  # New York City
        flight_levels = ["FL050", "FL100", "FL200", "FL350"]
        
        aviation_weather = windy.get_aviation_weather(
            lat=coords[0],
            lon=coords[1],
            flight_levels=flight_levels,
            model="gfs"
        )
        
        if "error" in aviation_weather:
            logger.error(f"Error getting aviation weather: {aviation_weather['error']}")
        else:
            logger.info(f"Successfully retrieved aviation weather data")
            logger.info(f"Available flight levels: {list(aviation_weather.get('flight_levels', {}).keys())}")
            
            # Sample wind data for FL200
            if "flight_levels" in aviation_weather and "FL200" in aviation_weather["flight_levels"]:
                fl_data = aviation_weather["flight_levels"]["FL200"]
                if "wind_speed" in fl_data and fl_data["wind_speed"]:
                    logger.info(f"Sample FL200 wind speeds (m/s): {fl_data['wind_speed'][:3]}")
                if "wind_direction" in fl_data and fl_data["wind_direction"]:
                    logger.info(f"Sample FL200 wind directions: {fl_data['wind_direction'][:3]}")
            
            # Show turbulence data
            if "derived" in aviation_weather and "turbulence" in aviation_weather["derived"]:
                turbulence = aviation_weather["derived"]["turbulence"]
                if "level" in turbulence and turbulence["level"]:
                    logger.info(f"Sample turbulence levels: {turbulence['level'][:3]}")
    except Exception as e:
        logger.error(f"Error in TEST 2: {str(e)}")
    
    # TEST 3: Get weather data for a flight route
    logger.info("\n=== TEST 3: Route Weather Data ===")
    try:
        # Sample route from NY to Chicago
        route_coords = [
            (40.7128, -74.0060),  # New York
            (41.8781, -87.6298)   # Chicago
        ]
        
        parameters = ["wind", "temp", "clouds", "precip"]
        levels = ["surface", "700h"]
        
        route_forecast = windy.get_route_forecast(
            route_coords=route_coords,
            model="gfs",
            parameters=parameters,
            levels=levels,
            route_segments=5  # Generate 5 points along the route
        )
        
        if "error" in route_forecast:
            logger.error(f"Error getting route forecast: {route_forecast['error']}")
        elif "forecasts" not in route_forecast or not route_forecast["forecasts"]:
            logger.error("No route forecast data found")
        else:
            logger.info(f"Successfully retrieved route forecast data")
            logger.info(f"Route metadata: {route_forecast['metadata']}")
            logger.info(f"Number of route points: {len(route_forecast['forecasts'])}")
            
            # Sample data from first point
            first_point = next(iter(route_forecast["forecasts"]))
            point_data = route_forecast["forecasts"][first_point]
            logger.info(f"First point coordinates: {point_data.get('coordinates', {})}")
    except Exception as e:
        logger.error(f"Error in TEST 3: {str(e)}")
    
    # TEST 4: Get turbulence forecast
    logger.info("\n=== TEST 4: Turbulence Forecast ===")
    try:
        coords = (40.7128, -74.0060)  # New York City
        flight_level = "FL350"
        
        turbulence = windy.get_turbulence_forecast(
            lat=coords[0],
            lon=coords[1],
            flight_level=flight_level,
            model="gfs"
        )
        
        if "error" in turbulence:
            logger.error(f"Error getting turbulence forecast: {turbulence['error']}")
        else:
            logger.info(f"Successfully retrieved turbulence forecast")
            logger.info(f"Flight level: {turbulence.get('flight_level')}")
            
            if "level" in turbulence and turbulence["level"]:
                logger.info(f"Turbulence levels: {turbulence['level'][:3]}")
            
            if "contributors" in turbulence:
                logger.info(f"Turbulence contributors: {list(turbulence['contributors'].keys())}")
    except Exception as e:
        logger.error(f"Error in TEST 4: {str(e)}")
    
    # TEST 5: Get icing forecast
    logger.info("\n=== TEST 5: Icing Forecast ===")
    try:
        coords = (40.7128, -74.0060)  # New York City
        flight_level = "FL180"
        
        icing = windy.get_icing_forecast(
            lat=coords[0],
            lon=coords[1],
            flight_level=flight_level,
            model="gfs"
        )
        
        if "error" in icing:
            logger.error(f"Error getting icing forecast: {icing['error']}")
        else:
            logger.info(f"Successfully retrieved icing forecast")
            logger.info(f"Flight level: {icing.get('flight_level')}")
            
            if "risk_level" in icing and icing["risk_level"]:
                logger.info(f"Icing risk levels: {icing['risk_level'][:3]}")
            
            if "icing_type" in icing and icing["icing_type"]:
                logger.info(f"Icing types: {icing['icing_type'][:3]}")
    except Exception as e:
        logger.error(f"Error in TEST 5: {str(e)}")
    
    # TEST 6: Get detailed wind data
    logger.info("\n=== TEST 6: Detailed Wind Data ===")
    try:
        coords = (40.7128, -74.0060)  # New York City
        flight_levels = ["FL100", "FL200", "FL300"]
        
        wind_data = windy.get_wind_data(
            lat=coords[0],
            lon=coords[1],
            flight_levels=flight_levels,
            model="gfs"
        )
        
        if "error" in wind_data:
            logger.error(f"Error getting wind data: {wind_data['error']}")
        else:
            logger.info(f"Successfully retrieved wind data")
            logger.info(f"Available levels: {list(wind_data.get('levels', {}).keys())}")
            
            # Create a visualization
            try:
                plot_wind_data(wind_data, "New York City Wind Forecast")
            except Exception as e:
                logger.error(f"Error creating wind visualization: {str(e)}")
    except Exception as e:
        logger.error(f"Error in TEST 6: {str(e)}")
    
    # TEST 7: Get route winds
    logger.info("\n=== TEST 7: Route Winds ===")
    try:
        # Route from London to Paris
        route_coords = [
            (51.5074, -0.1278),  # London
            (48.8566, 2.3522)     # Paris
        ]
        
        flight_level = "FL300"
        
        route_winds = windy.get_route_winds(
            route_coords=route_coords,
            flight_level=flight_level,
            model="gfs",
            route_segments=8
        )
        
        if "error" in route_winds:
            logger.error(f"Error getting route winds: {route_winds['error']}")
        else:
            logger.info(f"Successfully retrieved route winds")
            logger.info(f"Flight level: {route_winds.get('flight_level')}")
            logger.info(f"Route length: {route_winds.get('route_length_km')} km")
            logger.info(f"Number of points: {route_winds.get('point_count')}")
            
            # Create a visualization
            try:
                plot_route_winds(route_winds, "London to Paris Route")
            except Exception as e:
                logger.error(f"Error creating route winds visualization: {str(e)}")
    except Exception as e:
        logger.error(f"Error in TEST 7: {str(e)}")
    
    # TEST 8: Get airport weather
    logger.info("\n=== TEST 8: Airport Weather ===")
    try:
        icao_code = "KJFK"  # JFK Airport
        
        airport_weather = windy.get_airport_weather(
            icao_code=icao_code,
            model="gfs"
        )
        
        if "error" in airport_weather:
            logger.error(f"Error getting airport weather: {airport_weather['error']}")
        else:
            logger.info(f"Successfully retrieved weather for {icao_code}")
            logger.info(f"Airport location: {airport_weather.get('location', {})}")
            
            # Sample surface conditions
            if "surface_conditions" in airport_weather:
                surface = airport_weather["surface_conditions"]
                if "temperature_c" in surface and surface["temperature_c"]:
                    logger.info(f"Surface temperature (°C): {surface['temperature_c'][0]}")
                if "wind_speed" in surface and surface["wind_speed"]:
                    logger.info(f"Surface wind speed (m/s): {surface['wind_speed'][0]}")
                if "wind_direction" in surface and surface["wind_direction"]:
                    logger.info(f"Surface wind direction: {surface['wind_direction'][0]}")
    except Exception as e:
        logger.error(f"Error in TEST 8: {str(e)}")
    
    # TEST 9: Get significant weather
    logger.info("\n=== TEST 9: Significant Weather ===")
    try:
        coords = (40.7128, -74.0060)  # New York City
        radius_km = 100.0
        
        significant_weather = windy.get_significant_weather(
            lat=coords[0],
            lon=coords[1],
            radius_km=radius_km,
            model="gfs"
        )
        
        if "error" in significant_weather:
            logger.error(f"Error getting significant weather: {significant_weather['error']}")
        else:
            logger.info(f"Successfully retrieved significant weather")
            logger.info(f"Center point: {significant_weather.get('center_point', {})}")
            logger.info(f"Analyzed radius: {significant_weather.get('radius_km')} km")
            
            # Report significant weather events
            events = significant_weather.get("significant_weather", {})
            for event_type, event_list in events.items():
                if event_list:
                    logger.info(f"Found {len(event_list)} {event_type} events")
    except Exception as e:
        logger.error(f"Error in TEST 9: {str(e)}")
    
    # TEST 10: Generate visualization URL
    logger.info("\n=== TEST 10: Visualization URL ===")
    try:
        coords = (40.7128, -74.0060)  # New York City
        
        url = windy.get_visualization_url(
            lat=coords[0],
            lon=coords[1],
            zoom=7,
            overlay="wind",
            level="500h"
        )
        
        logger.info(f"Windy visualization URL: {url}")
    except Exception as e:
        logger.error(f"Error in TEST 10: {str(e)}")
    
    logger.info("\n============ WINDY CONNECTOR TEST COMPLETE ============")

if __name__ == "__main__":
    test_windy_connector()