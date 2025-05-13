import streamlit as st
import os
from dotenv import load_dotenv
import logging

from aviation_assistant.data.services.data_collection_service import DataCollectionService
from aviation_assistant.storage.document_store import AviationDocumentStore
from aviation_assistant.rag.embeddings.embedder import AviationEmbedder
from aviation_assistant.rag.pipelines.rag_pipeline import AviationRAGPipeline
from aviation_assistant.optimization.route_optimizer import RouteOptimizer
from aviation_assistant.visualization.weather_visualizer import WeatherVisualizer
from aviation_assistant.visualization.notam_visualizer import NOTAMVisualizer
from aviation_assistant.visualization.route_visualizer import RouteVisualizer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the application
@st.cache_resource
def initialize_app():
    """Initialize and cache app resources."""
    # Load configuration
    config = {
        "avwx_api_key": os.getenv("AVWX_API_KEY"),
        "opensky_username": os.getenv("OPENSKY_USERNAME"),
        "opensky_password": os.getenv("OPENSKY_PASSWORD"),
        "faa_notam_api_key": os.getenv("FAA_NOTAM_API_KEY"),
        "openrouter_api_key": os.getenv("OPENROUTER_API_KEY"),
        "vector_db_path": os.getenv("VECTOR_DB_PATH", "./data/vector_db")
    }
    
    # Initialize services
    data_service = DataCollectionService(config)
    document_store = AviationDocumentStore(vector_db_path=config["vector_db_path"])
    embedder = AviationEmbedder()
    rag_pipeline = AviationRAGPipeline(
        document_store=document_store,
        embedder=embedder,
        openrouter_api_key=config["openrouter_api_key"]
    )
    route_optimizer = RouteOptimizer()
    
    # Initialize visualizers
    weather_viz = WeatherVisualizer()
    notam_viz = NOTAMVisualizer()
    route_viz = RouteVisualizer()
    
    return {
        "data_service": data_service,
        "rag_pipeline": rag_pipeline,
        "route_optimizer": route_optimizer,
        "weather_viz": weather_viz,
        "notam_viz": notam_viz,
        "route_viz": route_viz
    }

# Main function
def main():
    """Main application function."""
    st.title("Aviation Pre-Flight Briefing Assistant")
    
    # Initialize app resources
    app_resources = initialize_app()
    
    # Sidebar for navigation
    option = st.sidebar.selectbox(
        "Select Mode",
        ["Airport Briefing", "Route Planning", "Ask a Question"]
    )
    
    if option == "Airport Briefing":
        airport_briefing()
    elif option == "Route Planning":
        route_planning()
    else:
        ask_question()

def airport_briefing():
    """Airport briefing mode."""
    st.header("Airport Briefing")
    
    # Airport input
    airport_code = st.text_input("Enter Airport ICAO Code", "KJFK").upper()
    
    if st.button("Generate Briefing"):
        with st.spinner("Collecting airport data..."):
            # TODO: Implement airport briefing logic
            st.write("Airport briefing functionality will be implemented here.")

def route_planning():
    """Route planning mode."""
    st.header("Route Planning")
    
    # Route inputs
    col1, col2 = st.columns(2)
    with col1:
        departure = st.text_input("Departure (ICAO)", "KJFK").upper()
    with col2:
        destination = st.text_input("Destination (ICAO)", "KLAX").upper()
    
    aircraft_type = st.selectbox(
        "Aircraft Type",
        ["B738", "A320", "C172", "B77W", "E190"]
    )
    
    if st.button("Plan Route"):
        with st.spinner("Analyzing route..."):
            # TODO: Implement route planning logic
            st.write("Route planning functionality will be implemented here.")

def ask_question():
    """Question answering mode."""
    st.header("Ask a Question")
    
    question = st.text_input("Enter your question", "What are the current weather conditions at KJFK?")
    
    if st.button("Ask"):
        with st.spinner("Thinking..."):
            # TODO: Implement question answering logic
            st.write("Question answering functionality will be implemented here.")

if __name__ == "__main__":
    main()
