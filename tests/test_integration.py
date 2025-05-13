# test_integration_service.py


import os
import sys
import logging
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import connectors and integration service
from aviation_assistant.data.connectors.avwx_connector import AVWXConnector
from aviation_assistant.data.connectors.windy_connector import WindyConnector
from aviation_assistant.data.connectors.opensky_connector import OpenSkyConnector
from aviation_assistant.data.services.integration_service import IntegrationService

def test_integration_service():
    """Test the enhanced integration service."""
    # Load configuration
    logger.info("Loading configuration...")
    
    # Load from environment variables
    config = {
        "avwx_api_key": os.getenv("AVWX_API_KEY"),
        "windy_api_key": os.getenv("WINDY_API_KEY"),
        "opensky_username": os.getenv("OPENSKY_USERNAME"),
        "opensky_password": os.getenv("OPENSKY_PASSWORD")
    }
    
    # Initialize connectors
    logger.info("Initializing connectors...")
    avwx = AVWXConnector(api_key="jLX2iQWBGYttAwYBxiFCUc_4ljrbNOqCNvHe5hjtS8o")
    windy = WindyConnector(api_key="ymAvnkZvsbZpxgMVlHabWuDIG6ZgHsJd")
    opensky = OpenSkyConnector(username="sami___ag", password="716879Sami.")
    
    # Initialize integration service
    integration = IntegrationService(avwx, windy, opensky)
    
    # Test airports
    airports = ["KJFK", "KLAX", "EGLL"]
    
    # Test routes
    routes = [("KJFK", "KLAX"), ("EGLL", "KJFK")]
    
    # Test results file
    with open("enhanced_integration_results.txt", "w") as outfile:
        outfile.write("ENHANCED INTEGRATION TEST RESULTS\n")
        outfile.write("=================================\n\n")
        outfile.write(f"Test conducted on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
        
        # 1. Test airport weather (including hazard assessment)
        outfile.write("1. AIRPORT WEATHER & HAZARDS\n")
        outfile.write("---------------------------\n\n")
        
        for airport in airports:
            try:
                logger.info(f"Testing airport weather and hazards for {airport}...")
                weather = integration.get_airport_weather(airport)
                
                outfile.write(f"Weather for {airport}:\n")
                outfile.write("-" * 30 + "\n")
                
                # Airport name
                station = weather.get("station", {})
                if station and "name" in station:
                    outfile.write(f"Airport: {station['name']} ({airport})\n\n")
                
                # METAR summary
                outfile.write("METAR Summary:\n")
                outfile.write(weather.get("metar_summary", "No METAR data available") + "\n\n")
                
                # TAF summary
                outfile.write("TAF Summary:\n")
                outfile.write(weather.get("taf_summary", "No TAF data available") + "\n\n")
                
                # Hazardous conditions
                hazards = weather.get("hazardous_conditions", [])
                if hazards:
                    outfile.write("Hazardous Conditions:\n")
                    for i, hazard in enumerate(hazards, 1):
                        description = hazard.get("description", "Unknown")
                        severity = hazard.get("severity", "unknown")
                        source = hazard.get("source", "unknown")
                        outfile.write(f"  {i}. {description} (Severity: {severity}, Source: {source})\n")
                    outfile.write("\n")
                
                # Visualization URLs
                outfile.write("Visualization Links:\n")
                for name, url in weather.get("visualization_urls", {}).items():
                    if name != "flight_levels":
                        outfile.write(f"  {name.capitalize()}: {url}\n")
                
                outfile.write("\n" + "=" * 50 + "\n\n")
            except Exception as e:
                logger.error(f"Error testing airport weather for {airport}: {str(e)}")
                outfile.write(f"Error testing airport weather for {airport}: {str(e)}\n\n")
        
        # 2. Test route weather
        outfile.write("2. ROUTE WEATHER\n")
        outfile.write("--------------\n\n")
        
        for dep, dest in routes:
            try:
                logger.info(f"Testing route weather for {dep}-{dest}...")
                route_weather = integration.get_route_weather(dep, dest)
                
                outfile.write(f"Route weather for {dep}-{dest}:\n")
                outfile.write("-" * 30 + "\n")
                
                # Route summary
                outfile.write(route_weather.get("summary", "No route summary available") + "\n\n")
                
                # Route visualization URLs
                outfile.write("Route Visualization Links:\n")
                for name, url in route_weather.get("route_visualization", {}).items():
                    outfile.write(f"  {name.capitalize()}: {url}\n")
                
                outfile.write("\n" + "=" * 50 + "\n\n")
            except Exception as e:
                logger.error(f"Error testing route weather for {dep}-{dest}: {str(e)}")
                outfile.write(f"Error testing route weather for {dep}-{dest}: {str(e)}\n\n")
        
        # 3. Test airport traffic if OpenSky is available
        outfile.write("3. AIRPORT TRAFFIC\n")
        outfile.write("----------------\n\n")
        
        for airport in airports[:1]:  # Test one airport to save time
            try:
                logger.info(f"Testing airport traffic for {airport}...")
                traffic = integration.get_airport_traffic(airport)
                
                outfile.write(f"Traffic for {airport}:\n")
                outfile.write("-" * 30 + "\n")
                
                if "error" in traffic:
                    outfile.write(f"Error: {traffic['error']}\n\n")
                else:
                    # Traffic summary
                    outfile.write(traffic.get("summary", "No traffic summary available") + "\n\n")
                
                outfile.write("\n" + "=" * 50 + "\n\n")
            except Exception as e:
                logger.error(f"Error testing airport traffic for {airport}: {str(e)}")
                outfile.write(f"Error testing airport traffic for {airport}: {str(e)}\n\n")
        
        # 4. Test nearby traffic if OpenSky is available
        outfile.write("4. NEARBY TRAFFIC\n")
        outfile.write("--------------\n\n")
        
        try:
            # Use JFK coordinates as an example
            lat = 40.639447
            lon = -73.779317
            
            logger.info(f"Testing nearby traffic at ({lat}, {lon})...")
            traffic = integration.get_nearby_traffic(lat, lon)
            
            outfile.write(f"Traffic near ({lat}, {lon}):\n")
            outfile.write("-" * 30 + "\n")
            
            if "error" in traffic:
                outfile.write(f"Error: {traffic['error']}\n\n")
            else:
                # Traffic summary
                outfile.write(traffic.get("summary", "No traffic summary available") + "\n\n")
            
            outfile.write("\n" + "=" * 50 + "\n\n")
        except Exception as e:
            logger.error(f"Error testing nearby traffic: {str(e)}")
            outfile.write(f"Error testing nearby traffic: {str(e)}\n\n")
        
        # 5. Test flight level weather
        outfile.write("5. FLIGHT LEVEL WEATHER\n")
        outfile.write("---------------------\n\n")
        
        try:
            # Use JFK coordinates as an example
            lat = 40.639447
            lon = -73.779317
            flight_level = "FL350"
            
            logger.info(f"Testing flight level weather at {flight_level}...")
            fl_weather = integration.get_flight_level_weather(lat, lon, flight_level)
            
            outfile.write(f"Weather at {flight_level} ({lat}, {lon}):\n")
            outfile.write("-" * 30 + "\n")
            
            if "error" in fl_weather:
                outfile.write(f"Error: {fl_weather['error']}\n\n")
            else:
                # Flight level summary
                outfile.write(fl_weather.get("summary", "No flight level summary available") + "\n\n")
                
                # Visualization URL
                url = fl_weather.get("visualization_url")
                if url:
                    outfile.write(f"Visualization: {url}\n\n")
            
            outfile.write("\n" + "=" * 50 + "\n\n")
        except Exception as e:
            logger.error(f"Error testing flight level weather: {str(e)}")
            outfile.write(f"Error testing flight level weather: {str(e)}\n\n")
        
        # Test complete
        outfile.write("\nTest completed successfully.\n")
    
    logger.info(f"Test results written to enhanced_integration_results.txt")

if __name__ == "__main__":
    test_integration_service()