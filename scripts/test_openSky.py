import os
import sys
import logging
from datetime import datetime, timedelta
import json
import time
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the OpenSkyConnector - adjust the import path as needed
from aviation_assistant.data.connectors.opensky_connector import OpenSkyConnector

def pretty_print(data: Dict[str, Any]) -> None:
    """Print dictionary in a readable format."""
    print(json.dumps(data, indent=4, default=str))

def test_opensky_connector():
    """Test the OpenSky connector using direct REST API calls."""
    logger.info("============ TESTING OPENSKY CONNECTOR ============")
    
    # Get credentials from environment variables
    username = os.environ.get("OPENSKY_USERNAME")
    password = os.environ.get("OPENSKY_PASSWORD")
    
    # Initialize the connector
    opensky = OpenSkyConnector(username=username, password=password)
    
    # TEST 1: Get states in a specific region (Europe)
    logger.info("\n=== TEST 1: Get states in Europe ===")
    try:
        # Bounding box over Western Europe
        bbox = (45.0, 0.0, 55.0, 15.0)  # min_lat, min_lon, max_lat, max_lon
        states = opensky.get_states(bbox=bbox)
        
        if "states" in states and states["states"]:
            formatted_states = opensky.format_state_vectors(states)
            logger.info(f"Success! Found {len(formatted_states)} aircraft in European airspace")
            
            # Show the first 3 aircraft details
            for i, state in enumerate(formatted_states[:3]):
                logger.info(f"\nAircraft {i+1}:")
                pretty_print(state)
        else:
            logger.error(f"Failed to retrieve Europe area states: {states}")
    except Exception as e:
        logger.error(f"Error in TEST 1: {str(e)}")
    
    # Add a delay to avoid rate limits
    time.sleep(2)
    
    # TEST 2: Get traffic around a specific point (Frankfurt Airport)
    logger.info("\n=== TEST 2: Get traffic around Frankfurt Airport ===")
    try:
        # Coordinates for Frankfurt Airport (EDDF)
        lat, lon = 50.033, 8.570
        radius_km = 100
        
        traffic = opensky.get_traffic_around_point(lat, lon, radius_km)
        logger.info(f"Success! Found {len(traffic)} aircraft around Frankfurt Airport")
        
        # Show the first 3 aircraft
        for i, aircraft in enumerate(traffic[:3]):
            logger.info(f"\nAircraft {i+1}:")
            pretty_print(aircraft)
    except Exception as e:
        logger.error(f"Error in TEST 2: {str(e)}")
    
    # Add a delay to avoid rate limits
    time.sleep(2)
    
    # TEST 3: Get flights in a time interval
    logger.info("\n=== TEST 3: Get flights in time interval ===")
    try:
        # Use a 2-hour window (maximum allowed)
        end_time = datetime.now() - timedelta(hours=1)  # Use data from an hour ago for better results
        begin_time = end_time - timedelta(hours=2)
        
        logger.info(f"Fetching flights between {begin_time} and {end_time}")
        flights = opensky.get_flights_by_interval(begin_time, end_time)
        
        if isinstance(flights, list):
            logger.info(f"Success! Found {len(flights)} flights in the time interval")
            
            # Show the first 3 flights
            for i, flight in enumerate(flights[:3]):
                logger.info(f"\nFlight {i+1}:")
                pretty_print(flight)
        else:
            logger.error(f"Failed to retrieve flights: {flights}")
    except Exception as e:
        logger.error(f"Error in TEST 3: {str(e)}")
    
    # Add a delay to avoid rate limits
    time.sleep(2)
    
    # TEST 4: Get arrivals at a major airport
    logger.info("\n=== TEST 4: Get arrivals at Frankfurt Airport ===")
    try:
        airport = "EDDF"  # Frankfurt Airport
        end_time = datetime.now() - timedelta(hours=1)  # Use data from an hour ago
        begin_time = end_time - timedelta(hours=2)  # 2-hour window (maximum allowed)
        
        logger.info(f"Fetching arrivals at {airport} between {begin_time} and {end_time}")
        arrivals = opensky.get_arrivals(airport, begin_time, end_time)
        
        if isinstance(arrivals, list):
            logger.info(f"Success! Found {len(arrivals)} arrivals at {airport}")
            
            # Show the first 3 arrivals
            for i, arrival in enumerate(arrivals[:3]):
                logger.info(f"\nArrival {i+1}:")
                pretty_print(arrival)
        else:
            logger.error(f"Failed to retrieve arrivals: {arrivals}")
    except Exception as e:
        logger.error(f"Error in TEST 4: {str(e)}")
    
    # Add a delay to avoid rate limits
    time.sleep(2)
    
    # TEST 5: Get departures from a major airport
    logger.info("\n=== TEST 5: Get departures from London Heathrow ===")
    try:
        airport = "EGLL"  # London Heathrow
        end_time = datetime.now() - timedelta(hours=1)  # Use data from an hour ago
        begin_time = end_time - timedelta(hours=2)  # 2-hour window (maximum allowed)
        
        logger.info(f"Fetching departures from {airport} between {begin_time} and {end_time}")
        departures = opensky.get_departures(airport, begin_time, end_time)
        
        if isinstance(departures, list):
            logger.info(f"Success! Found {len(departures)} departures from {airport}")
            
            # Show the first 3 departures
            for i, departure in enumerate(departures[:3]):
                logger.info(f"\nDeparture {i+1}:")
                pretty_print(departure)
        else:
            logger.error(f"Failed to retrieve departures: {departures}")
    except Exception as e:
        logger.error(f"Error in TEST 5: {str(e)}")
    
    # Add a delay to avoid rate limits
    time.sleep(2)
    
    # TEST 6: Try to get information for a specific aircraft by ICAO24
    logger.info("\n=== TEST 6: Get information for a specific aircraft ===")
    try:
        # Get some known active aircraft from previous tests
        active_icao24 = None
        if "states" in states and states["states"] and len(states["states"]) > 0:
            active_icao24 = states["states"][0][0]  # Get ICAO24 from first aircraft
            logger.info(f"Using active aircraft with ICAO24: {active_icao24}")
            
            # Get position and track for this aircraft
            position = opensky.get_aircraft_position(active_icao24)
            logger.info(f"Aircraft position:")
            pretty_print(position)
            
            # Add a delay to avoid rate limits
            time.sleep(2)
            
            # Get flight history
            end_time = datetime.now()
            begin_time = end_time - timedelta(hours=6)  # Look back 6 hours
            
            logger.info(f"Fetching flight history for {active_icao24}")
            flights = opensky.get_flights_by_aircraft(active_icao24, begin_time, end_time)
            
            if flights:
                logger.info(f"Found {len(flights)} flights for {active_icao24}")
                for i, flight in enumerate(flights[:2]):  # Show up to 2 flights
                    logger.info(f"\nFlight {i+1}:")
                    pretty_print(flight)
            else:
                logger.info(f"No flight history found for {active_icao24}")
        else:
            logger.info("No active aircraft found to get specific information")
    except Exception as e:
        logger.error(f"Error in TEST 6: {str(e)}")
    
    logger.info("\n============ OPENSKY CONNECTOR TEST COMPLETE ============")

if __name__ == "__main__":
    # Run the OpenSky connector test
    test_opensky_connector()