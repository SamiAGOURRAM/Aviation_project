import requests
import logging
from typing import Dict, Any, Optional, List

class AVWXConnector:
    """
    Connector for AVWX REST API to fetch aviation weather data.
    
    Provides access to METAR, TAF, and station information with
    translation capabilities to convert codes to plain English.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the AVWX API connector.
        
        Args:
            api_key: Your AVWX API key
        """
        self.api_key = api_key
        self.base_url = "https://avwx.rest/api"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.logger = logging.getLogger(__name__)
    
    def get_metar(self, station_id: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Retrieve current METAR for a station."""
        endpoint = f"{self.base_url}/metar/{station_id}"
        params = options or {}
        
        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching METAR for {station_id}: {str(e)}")
            return {"error": str(e)}
        
    def get_taf(self, station_id: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Retrieve current TAF for a station."""
        endpoint = f"{self.base_url}/taf/{station_id}"
        params = options or {}
        
        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching TAF for {station_id}: {str(e)}")
            return {"error": str(e)}
        
    def get_station(self, station_id: str) -> Dict[str, Any]:
        """Retrieve information about a station."""
        endpoint = f"{self.base_url}/station/{station_id}"
        
        try:
            response = requests.get(endpoint, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching station info for {station_id}: {str(e)}")
            return {"error": str(e)}
