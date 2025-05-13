import os
import sys
import logging
from dotenv import load_dotenv

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aviation_assistant.data.connectors.avwx_connector import AVWXConnector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API keys from environment variables
load_dotenv()

def test_avwx_connector():
    """Test the AVWX connector."""
    api_key = os.getenv("AVWX_API_KEY")
    if not api_key:
        logger.error("AVWX_API_KEY not found in environment variables")
        return
    
    logger.info("Testing AVWX connector...")
    connector = AVWXConnector(api_key=api_key)
    
    # Test get_station
    station_id = "KJFK"
    logger.info(f"Getting information for station {station_id}...")
    station_info = connector.get_station(station_id)
    logger.info(f"Station info: {station_info}")
    
    # Test get_metar
    logger.info(f"Getting METAR for station {station_id}...")
    metar_info = connector.get_metar(station_id)
    logger.info(f"METAR info: {metar_info}")
    
    # Test get_taf
    logger.info(f"Getting TAF for station {station_id}...")
    taf_info = connector.get_taf(station_id)
    logger.info(f"TAF info: {taf_info}")

if __name__ == "__main__":
    test_avwx_connector()