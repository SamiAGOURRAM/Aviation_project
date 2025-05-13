import os
import sys
import logging
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aviation_assistant.data.connectors.windy_connector import WindyConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def pretty_print(data):
    """Print dictionary in a readable format."""
    if isinstance(data, dict):
        print(json.dumps(data, indent=4, default=str))
    else:
        print(data)

def test_basic_forecast():
    """Test 1: Basic forecast with minimal parameters."""
    logger.info("\n=== TEST 1: Basic Forecast with Minimal Parameters ===")
    
    # api_key = os.environ.get("WINDY_API_KEY")
    api_key = "ymAvnkZvsbZpxgMVlHabWuDIG6ZgHsJd"
    windy = WindyConnector(api_key=api_key)
    
    # Start with minimal parameters - just temperature
    coords = (40.7128, -74.0060)  # New York City
    forecast = windy.get_forecast(
        lat=coords[0],
        lon=coords[1],
        model="gfs",
        parameters=["temp"],
        levels=["surface"]
    )
    
    if "error" in forecast:
        logger.error(f"Error: {forecast['error']}")
        return False
        
    logger.info("✅ Basic forecast successful")
    logger.info(f"Number of timestamps: {len(forecast.get('forecast_time', []))}")
    
    # Sample the first temperature value
    temp_key = next((k for k in forecast.keys() if k.startswith("temp-surface")), None)
    if temp_key and temp_key in forecast and forecast[temp_key]:
        temp_k = forecast[temp_key][0]
        temp_c = round(temp_k - 273.15, 1) if temp_k else None
        logger.info(f"First temperature reading: {temp_c}°C")
    
    return True

def test_multiple_parameters():
    """Test 2: Multiple weather parameters."""
    logger.info("\n=== TEST 2: Multiple Weather Parameters ===")
    
    api_key = "ymAvnkZvsbZpxgMVlHabWuDIG6ZgHsJd"
    windy = WindyConnector(api_key=api_key)
    
    # Use multiple supported parameters
    coords = (40.7128, -74.0060)  # New York City
    parameters = ["temp", "wind", "rh", "pressure", "dewpoint"]
    
    forecast = windy.get_forecast(
        lat=coords[0],
        lon=coords[1],
        model="gfs",
        parameters=parameters,
        levels=["surface"]
    )
    
    if "error" in forecast:
        logger.error(f"Error: {forecast['error']}")
        return False
        
    logger.info("✅ Multiple parameters forecast successful")
    logger.info(f"Available parameters: {[k.split('-')[0] for k in forecast.keys() if '-' in k]}")
    
    return True

def test_multiple_levels():
    """Test 3: Multiple altitude levels."""
    logger.info("\n=== TEST 3: Multiple Altitude Levels ===")
    
    api_key = "ymAvnkZvsbZpxgMVlHabWuDIG6ZgHsJd"
    windy = WindyConnector(api_key=api_key)
    
    # Use parameters that support multiple levels
    coords = (40.7128, -74.0060)  # New York City
    parameters = ["wind", "temp", "rh"]  # These support multiple levels
    levels = ["surface", "850h", "500h"]
    
    forecast = windy.get_forecast(
        lat=coords[0],
        lon=coords[1],
        model="gfs",
        parameters=parameters,
        levels=levels
    )
    
    if "error" in forecast:
        logger.error(f"Error: {forecast['error']}")
        return False
        
    logger.info("✅ Multiple levels forecast successful")
    
    # Check that we have data for different levels
    level_keys = []
    for level in levels:
        for param in parameters:
            if param == "wind":
                u_key = f"wind_u-{level}"
                v_key = f"wind_v-{level}"
                if u_key in forecast and v_key in forecast:
                    level_keys.extend([u_key, v_key])
            else:
                key = f"{param}-{level}"
                if key in forecast:
                    level_keys.append(key)
    
    logger.info(f"Level-specific data keys: {level_keys}")
    
    return True

def test_different_models():
    """Test 4: Different weather models."""
    logger.info("\n=== TEST 4: Different Weather Models ===")
    
    api_key = "ymAvnkZvsbZpxgMVlHabWuDIG6ZgHsJd"
    windy = WindyConnector(api_key=api_key)
    
    # Test multiple models
    coords = (40.7128, -74.0060)  # New York City
    models = ["gfs", "iconEu"]  # Test a couple of models
    
    success = True
    for model in models:
        logger.info(f"Testing model: {model}")
        
        # Get model-specific supported parameters
        params = windy._get_model_supported_params(model)[:3]  # Just use first 3 parameters
        
        forecast = windy.get_forecast(
            lat=coords[0],
            lon=coords[1],
            model=model,
            parameters=params,
            levels=["surface"]
        )
        
        if "error" in forecast:
            logger.error(f"Error with model {model}: {forecast['error']}")
            success = False
            continue
            
        logger.info(f"✅ Model {model} forecast successful with parameters {params}")
        
        # Add a delay to avoid rate limiting
        time.sleep(1)
    
    return success

def test_aviation_weather():
    """Test 5: Aviation weather data."""
    logger.info("\n=== TEST 5: Aviation Weather Data ===")
    
    api_key = "ymAvnkZvsbZpxgMVlHabWuDIG6ZgHsJd"
    windy = WindyConnector(api_key=api_key)
    
    # Test aviation weather at different flight levels
    coords = (40.7128, -74.0060)  # New York City
    flight_levels = ["FL100", "FL300"]
    
    aviation_weather = windy.get_aviation_weather(
        lat=coords[0],
        lon=coords[1],
        flight_levels=flight_levels,
        model="gfs"
    )
    
    if "error" in aviation_weather:
        logger.error(f"Error: {aviation_weather['error']}")
        return False
        
    logger.info("✅ Aviation weather data successful")
    logger.info(f"Retrieved data for flight levels: {list(aviation_weather.get('flight_levels', {}).keys())}")
    
    # Check for derived metrics
    if "derived" in aviation_weather:
        logger.info(f"Derived metrics: {list(aviation_weather['derived'].keys())}")
    
    return True

def test_route_forecast():
    """Test 6: Route forecast."""
    logger.info("\n=== TEST 6: Route Forecast ===")
    
    api_key = "ymAvnkZvsbZpxgMVlHabWuDIG6ZgHsJd"
    windy = WindyConnector(api_key=api_key)
    
    # Test a simple route with two points
    route = [
        (51.5074, -0.1278),  # London
        (48.8566, 2.3522)     # Paris
    ]
    
    # Use minimal parameters
    parameters = ["temp", "wind"]
    
    route_forecast = windy.get_route_forecast(
        route_coords=route,
        model="gfs",
        parameters=parameters,
        levels=["surface"],
        route_segments=4  # Generate 4 points total
    )
    
    if "error" in route_forecast:
        logger.error(f"Error: {route_forecast['error']}")
        return False
        
    logger.info("✅ Route forecast successful")
    logger.info(f"Route metadata: {route_forecast['metadata']}")
    logger.info(f"Number of route points: {len(route_forecast['forecasts'])}")
    
    # Verify each point has coordinates
    points_with_coords = 0
    for point_name, forecast in route_forecast['forecasts'].items():
        if "coordinates" in forecast:
            points_with_coords += 1
    
    logger.info(f"Points with valid coordinates: {points_with_coords} out of {len(route_forecast['forecasts'])}")
    
    return points_with_coords > 0

def test_turbulence_forecast():
    """Test 7: Turbulence forecast."""
    logger.info("\n=== TEST 7: Turbulence Forecast ===")
    
    api_key = "ymAvnkZvsbZpxgMVlHabWuDIG6ZgHsJd"
    windy = WindyConnector(api_key=api_key)
    
    # Test turbulence forecast at a flight level
    coords = (40.7128, -74.0060)  # New York City
    flight_level = "FL300"
    
    turbulence = windy.get_turbulence_forecast(
        lat=coords[0],
        lon=coords[1],
        flight_level=flight_level,
        model="gfs"
    )
    
    if "error" in turbulence:
        logger.error(f"Error: {turbulence['error']}")
        return False
        
    logger.info("✅ Turbulence forecast successful")
    logger.info(f"Flight level: {turbulence.get('flight_level')}")
    
    # Check turbulence levels
    if "level" in turbulence and turbulence["level"]:
        logger.info(f"First turbulence level: {turbulence['level'][0]}")
    
    return True

def test_visualization_url():
    """Test 8: Visualization URL generation."""
    logger.info("\n=== TEST 8: Visualization URL Generation ===")
    
    api_key = "ymAvnkZvsbZpxgMVlHabWuDIG6ZgHsJd"
    windy = WindyConnector(api_key=api_key)
    
    # Generate visualization URL
    coords = (40.7128, -74.0060)  # New York City
    
    url = windy.get_visualization_url(
        lat=coords[0],
        lon=coords[1],
        zoom=7,
        overlay="temp",
        level="500h"
    )
    
    logger.info(f"Visualization URL: {url}")
    logger.info("✅ Visualization URL generation successful")
    
    return True

def test_wind_data():
    """Test 9: Detailed wind data."""
    logger.info("\n=== TEST 9: Detailed Wind Data ===")
    
    api_key = "ymAvnkZvsbZpxgMVlHabWuDIG6ZgHsJd"
    windy = WindyConnector(api_key=api_key)
    
    # Get wind data at different flight levels
    coords = (40.7128, -74.0060)  # New York City
    flight_levels = ["FL100", "FL200"]
    
    wind_data = windy.get_wind_data(
        lat=coords[0],
        lon=coords[1],
        flight_levels=flight_levels,
        model="gfs"
    )
    
    if "error" in wind_data:
        logger.error(f"Error: {wind_data['error']}")
        return False
        
    logger.info("✅ Wind data retrieval successful")
    logger.info(f"Available levels: {list(wind_data.get('levels', {}).keys())}")
    
    # Check wind data for one level
    if flight_levels[0] in wind_data.get('levels', {}):
        level_data = wind_data['levels'][flight_levels[0]]
        if "wind_speed" in level_data and level_data["wind_speed"]:
            logger.info(f"First wind speed value at {flight_levels[0]}: {level_data['wind_speed'][0]} m/s")
        if "wind_direction" in level_data and level_data["wind_direction"]:
            logger.info(f"First wind direction at {flight_levels[0]}: {level_data['wind_direction'][0]}°")
    
    return True

def run_all_tests():
    """Run all tests with proper rate limiting."""
    tests = [
        test_basic_forecast,
        test_multiple_parameters,
        test_multiple_levels,
        test_different_models,
        test_aviation_weather,
        test_route_forecast,
        test_turbulence_forecast,
        test_visualization_url,
        test_wind_data
    ]
    
    results = []
    
    for i, test_func in enumerate(tests):
        logger.info(f"\nRunning test {i+1}/{len(tests)}: {test_func.__name__}")
        
        try:
            success = test_func()
            results.append((test_func.__name__, success))
        except Exception as e:
            logger.error(f"Error in {test_func.__name__}: {str(e)}")
            results.append((test_func.__name__, False))
        
        # Add delay between tests to avoid rate limiting
        if i < len(tests) - 1:
            logger.info("Waiting 2 seconds before next test...")
            time.sleep(2)
    
    # Summary of results
    logger.info("\n=== TEST RESULTS SUMMARY ===")
    successes = 0
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status}: {name}")
        if success:
            successes += 1
    
    logger.info(f"\nTotal: {successes}/{len(tests)} tests passed")

if __name__ == "__main__":
    # Check if API key is set
    # if not os.environ.get("WINDY_API_KEY"):
    #     logger.error("WINDY_API_KEY environment variable not set")
    #     sys.exit(1)
    
    # Run all tests
    run_all_tests()