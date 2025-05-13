import os
import sys
import json
import logging
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the connectors and processors
from aviation_assistant.data.connectors.avwx_connector import AVWXConnector
from aviation_assistant.data.connectors.windy_connector import WindyConnector
from aviation_assistant.data.processors.weather_processor import WeatherProcessor
from aviation_assistant.config.config_loader import ConfigLoader

def test_weather_processor():
    """Test the WeatherProcessor with real data from connectors."""
    # Load configuration
    logger.info("Loading configuration...")
    
    # First try to load from environment variables
    config = {
        "avwx_api_key": "jLX2iQWBGYttAwYBxiFCUc_4ljrbNOqCNvHe5hjtS8o",
        "windy_api_key": "ymAvnkZvsbZpxgMVlHabWuDIG6ZgHsJd"
    }
    
    # If environment variables aren't set, try loading from a config file
    if not config["avwx_api_key"] or not config["windy_api_key"]:
        try:
            config_loader = ConfigLoader("config.yaml")
            config = config_loader.get_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            logger.info("Please make sure your API keys are set in environment variables or config.yaml file")
            return
    
    # Initialize connectors
    logger.info("Initializing connectors...")
    try:
        avwx_connector = AVWXConnector(api_key=config["avwx_api_key"])
        windy_connector = WindyConnector(api_key=config["windy_api_key"])
    except Exception as e:
        logger.error(f"Error initializing connectors: {str(e)}")
        return
    
    # Initialize the weather processor
    weather_processor = WeatherProcessor()
    
    # List of airports to test
    airports = ["KJFK", "KLAX", "EGLL", "KORD", "KSFO"]
    
    # Open output file
    with open("weather_processor_test_results.txt", "w") as outfile:
        outfile.write("WEATHER PROCESSOR TEST RESULTS\n")
        outfile.write("=============================\n\n")
        outfile.write(f"Test conducted on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
        
        # Test each airport
        for airport in airports:
            try:
                outfile.write(f"\n\n{'=' * 40}\n")
                outfile.write(f"TESTING AIRPORT: {airport}\n")
                outfile.write(f"{'=' * 40}\n\n")
                
                # STEP 1: Fetch METAR data
                outfile.write("STEP 1: METAR Data Processing\n")
                outfile.write("--------------------------\n\n")
                
                try:
                    logger.info(f"Fetching METAR data for {airport}...")
                    metar_data = avwx_connector.get_metar(airport, {"options": "info,translate"})
                    
                    # Write raw METAR data
                    outfile.write("Raw METAR data:\n")
                    outfile.write(json.dumps(metar_data, indent=2, default=str)[:1000] + "...\n\n")
                    
                    # Process METAR data
                    logger.info(f"Processing METAR data for {airport}...")
                    processed_metar = weather_processor.process_metar(metar_data)
                    
                    # Write processed METAR data
                    outfile.write("Processed METAR data:\n")
                    outfile.write(json.dumps(processed_metar, indent=2, default=str) + "\n\n")
                    
                    # Write METAR summary
                    outfile.write("METAR Summary:\n")
                    outfile.write(processed_metar.get("summary", "No summary generated") + "\n\n")
                    
                except Exception as e:
                    logger.error(f"Error processing METAR data for {airport}: {str(e)}")
                    outfile.write(f"ERROR: Failed to process METAR data: {str(e)}\n\n")
                
                # STEP 2: Fetch TAF data
                outfile.write("STEP 2: TAF Data Processing\n")
                outfile.write("-------------------------\n\n")
                
                try:
                    logger.info(f"Fetching TAF data for {airport}...")
                    taf_data = avwx_connector.get_taf(airport, {"options": "info,translate"})
                    
                    # Write raw TAF data
                    outfile.write("Raw TAF data:\n")
                    outfile.write(json.dumps(taf_data, indent=2, default=str)[:1000] + "...\n\n")
                    
                    # Process TAF data
                    logger.info(f"Processing TAF data for {airport}...")
                    processed_taf = weather_processor.process_taf(taf_data)
                    
                    # Write processed TAF data
                    outfile.write("Processed TAF data:\n")
                    outfile.write(json.dumps(processed_taf, indent=2, default=str) + "\n\n")
                    
                    # Write TAF summary
                    outfile.write("TAF Summary:\n")
                    outfile.write(processed_taf.get("summary", "No summary generated") + "\n\n")
                    
                except Exception as e:
                    logger.error(f"Error processing TAF data for {airport}: {str(e)}")
                    outfile.write(f"ERROR: Failed to process TAF data: {str(e)}\n\n")
                
                # STEP 3: Fetch Windy forecast data
                outfile.write("STEP 3: Windy Forecast Processing\n")
                outfile.write("-------------------------------\n\n")
                
                try:
                    # Get airport coordinates from station info
                    logger.info(f"Fetching airport coordinates for {airport}...")
                    station_info = avwx_connector.get_station(airport)
                    
                    if "error" in station_info:
                        raise Exception(f"Error fetching station info: {station_info['error']}")
                    
                    lat = station_info.get("latitude")
                    lon = station_info.get("longitude")
                    
                    if lat is None or lon is None:
                        raise Exception("Could not find airport coordinates")
                    
                    logger.info(f"Fetching Windy forecast data for {airport} at coordinates {lat},{lon}...")
                    windy_data = windy_connector.get_forecast(
                        lat=lat,
                        lon=lon,
                        model="gfs",
                        parameters=["wind", "temp", "dewpoint", "rh", "pressure", "precip", "clouds", "visibility"],
                        levels=["surface", "850h", "700h", "500h"]
                    )
                    
                    # Write raw Windy data (truncated as it can be large)
                    outfile.write("Raw Windy forecast data (truncated):\n")
                    outfile.write(json.dumps(windy_data, indent=2, default=str)[:1000] + "...\n\n")
                    
                    # Process Windy data
                    logger.info(f"Processing Windy forecast data for {airport}...")
                    processed_windy = weather_processor.process_windy_forecast(windy_data)
                    
                    # Write processed Windy data
                    outfile.write("Processed Windy forecast data:\n")
                    outfile.write(json.dumps(processed_windy, indent=2, default=str) + "\n\n")
                    
                    # Write Windy summary
                    outfile.write("Windy Forecast Summary:\n")
                    outfile.write(processed_windy.get("summary", "No summary generated") + "\n\n")
                    
                except Exception as e:
                    logger.error(f"Error processing Windy data for {airport}: {str(e)}")
                    outfile.write(f"ERROR: Failed to process Windy forecast data: {str(e)}\n\n")
                
                # STEP 4: Combine all weather data
                outfile.write("STEP 4: Combined Weather Data\n")
                outfile.write("--------------------------\n\n")
                
                try:
                    logger.info(f"Combining all weather data for {airport}...")
                    
                    # Get local variables for processed data, handling potential failures
                    local_processed_metar = locals().get('processed_metar', None)
                    local_processed_taf = locals().get('processed_taf', None)
                    local_processed_windy = locals().get('processed_windy', None)
                    
                    combined_data = weather_processor.combine_weather_data(
                        metar_data=local_processed_metar,
                        taf_data=local_processed_taf,
                        windy_data=local_processed_windy
                    )
                    
                    # Write combined data
                    outfile.write("Combined weather data:\n")
                    outfile.write(json.dumps(combined_data, indent=2, default=str) + "\n\n")
                    
                    # Write combined summary
                    outfile.write("Combined Weather Summary:\n")
                    outfile.write(combined_data.get("summary", "No summary generated") + "\n\n")
                    
                    # Write hazards if any
                    hazards = combined_data.get("integrated_hazards", [])
                    if hazards:
                        outfile.write("Detected Hazards:\n")
                        for idx, hazard in enumerate(hazards, 1):
                            outfile.write(f"{idx}. {hazard.get('description', 'Unknown')} - " + 
                                          f"Severity: {hazard.get('severity', 'Unknown')}, " +
                                          f"Source: {hazard.get('source', 'Unknown')}\n")
                        outfile.write("\n")
                
                except Exception as e:
                    logger.error(f"Error combining weather data for {airport}: {str(e)}")
                    outfile.write(f"ERROR: Failed to combine weather data: {str(e)}\n\n")
                
            except Exception as e:
                logger.error(f"Error processing {airport}: {str(e)}")
                outfile.write(f"ERROR: Failed to process {airport}: {str(e)}\n\n")
        
        # Test complete
        outfile.write("\n\nTest completed successfully.\n")
    
    logger.info(f"Test results written to weather_processor_test_results.txt")

if __name__ == "__main__":
    test_weather_processor()