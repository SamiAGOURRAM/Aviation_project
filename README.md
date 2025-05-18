# SkyPilot AI: Advanced Generative AI for Aviation ✈️

SkyPilot AI is an advanced Generative AI-powered agent designed to assist pilots with pre-flight briefings and provide easy access to aeronautical knowledge. It leverages Large Language Models (LLMs), Retrieval Augmented Generation (RAG), real-time data APIs, and intelligent caching to deliver a comprehensive and interactive experience.

**Target Use Cases:** Pre-flight planning, educational support for student pilots, and general aviation knowledge queries. Designed for ground-based use with internet connectivity.

## ✨ Features

*   **Conversational Interface:** Interact with the agent using natural language via a Streamlit web app.
*   **Real-Time Data Integration:**
    *   **Weather:** METARs, TAFs for specific airports (`get_airport_weather`).
    *   **Route Weather:** Conditions between two airports (`get_route_weather`).
    *   **Flight Level Weather:** Weather at specific altitudes and locations (`get_flight_level_weather`).
    *   **Visual Maps:** Links to Windy.com maps for radar, satellite, clouds, etc. (`get_visual_weather_map`).
    *   **Air Traffic:** Live arrivals and departures for airports (`get_airport_traffic`).
*   **Retrieval Augmented Generation (RAG):**
    *   Answers questions based on an internal knowledge base of aviation documents (e.g., Pilot's Handbook of Aeronautical Knowledge - PHAK chapters).
    *   Utilizes `chapter_title` metadata extracted from filenames for improved retrieval relevance.
*   **Intelligent Agent (LangChain):**
    *   Powered by **Llama-4-Scout-17B** (via Groq API) for reasoning and tool selection.
    *   Uses Chain-of-Thought (CoT) prompting for transparent decision-making.
    *   Manages conversation history for contextual understanding.
*   **Dual LLM Invocation Strategy:**
    *   **Agent Control LLM:** Orchestrates tool use and synthesizes final responses.
    *   **RAG Synthesizer LLM:** Dedicated LLM instance within the Haystack RAG pipeline to generate answers *strictly* from retrieved document context, ensuring faithfulness.
*   **Efficient Caching (Redis):**
    *   Global LLM completion cache.
    *   Custom agent query response cache with dynamic TTLs based on data type.
    *   Feedback-driven cache adjustments.
*   **Feedback Loop for Fine-Tuning:**
    *   Collects user feedback (👍/👎, comments).
    *   Stores successful, positively-rated interactions (query, tool calls, response) in Redis for future fine-tuning of smaller, specialized LLMs.
*   **Transparent Reasoning:** Option to view the agent's `intermediate_steps` (thoughts, tool calls, observations).

## 🛠️ Technology Stack

*   **Core AI/LLM:**
    *   **Language Model:** Llama-4-Scout-17B (via **Groq API**)
    *   **Orchestration:** **LangChain**
    *   **RAG & Document Processing:** **Haystack 2.x**
*   **Data Infrastructure:**
    *   **Vector Database (RAG):** **Qdrant** (on-disk persistence)
    *   **Caching & Feedback Store:** **Redis**
*   **External Data APIs:**
    *   **AVWX** (Weather, Airport Info)
    *   **OpenSky Network** (Live Air Traffic)
    *   **Windy** (Visual Weather Map URLs)
*   **Development & UI:**
    *   **Programming Language:** Python 3.12
    *   **Web Application:** **Streamlit**
*   **Key Python Libraries:** `requests`, `pydantic` (for structured tools), `numpy`, `matplotlib` (for evaluation).

## 🚀 Getting Started

### Prerequisites

1.  **Python:** Version 3.10 or higher (3.12 used in development).
2.  **Redis:** A running Redis server instance.
    *   Installation: [Redis Quick Start](https://redis.io/docs/getting-started/)
3.  **Poetry (Recommended for dependency management) or pip:**
    *   If using Poetry, ensure `poetry.lock` and `pyproject.toml` are up-to-date.
    *   If using pip, a `requirements.txt` file should be present.
4.  **API Keys:** You will need API keys for:
    *   Groq (for Llama-4-Scout-17B)
    *   AVWX
    *   Windy API (if your `WindyConnector` uses direct API calls; current setup often constructs URLs)
    *   OpenSky Network (Username/Password)
5.  **Knowledge Base Documents:** Place your aviation documents (e.g., PHAK PDF chapters) in the designated knowledge base directory (default: `aviation_assistant/data/aviation_documents/`).

### Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your_repository_url>
    cd <your_repository_name>
    ```

2.  **Set up Python Environment & Install Dependencies:**

    *   **Using Poetry (Recommended):**
        ```bash
        poetry install
        ```
    *   **Using pip and `requirements.txt`:**
        ```bash
        python -m venv .venv
        source .venv/bin/activate  # On Windows: .venv\Scripts\activate
        pip install -r requirements.txt
        ```

3.  **Configure API Keys & Paths:**
    *   The application primarily uses Streamlit secrets for API keys. Create a file named `.streamlit/secrets.toml` in the project root directory (same level as `streamlit_app.py`).
    *   **Example `.streamlit/secrets.toml`:**
        ```toml
        GROQ_API_KEY = "gsk_YOUR_GROQ_KEY"
        AVWX_API_KEY = "YOUR_AVWX_KEY"
        WINDY_API_KEY = "YOUR_WINDY_KEY" # If used directly
        OPENSKY_USERNAME = "YOUR_OPENSKY_USERNAME"
        OPENSKY_PASSWORD = "YOUR_OPENSKY_PASSWORD"
        REDIS_URL = "redis://localhost:6379" # Adjust if your Redis is elsewhere

        # Optional: Override default paths if needed for the assistant
        # DOC_PATH = "/custom/path/to/aviation_documents"
        # QDRANT_PATH = "/custom/path/to/qdrant_datastore_assistant"
        ```

4.  **Prepare Knowledge Base & Qdrant:**
    *   Ensure your aviation documents (PHAK PDFs, etc.) are in the path specified in the assistant's configuration (default: `aviation_assistant/data/aviation_documents/`).
    *   The Qdrant database will be created (or used if existing) at the path specified in the assistant's configuration (default: `aviation_assistant/data/qdrant_datastore/`).
    *   On the first run, or if `force_reindex` is true in the configuration, the documents will be indexed into Qdrant. This may take some time.

### Running the Application

1.  **Start Redis Server (if not already running):**
    Open a new terminal window and run:
    ```bash
    redis-server --daemonize yes
    ```
    *   `--daemonize yes` runs Redis in the background. If you omit it, Redis will run in the foreground of that terminal.
    *   Verify Redis is running: `redis-cli ping` (should return `PONG`).

2.  **Run the Streamlit Web Application:**
    In your main project directory (where `streamlit_app.py` is located), activate your Python environment (e.g., `source .venv/bin/activate` or `poetry shell`) and then run:
    ```bash
    streamlit run streamlit_app.py
    ```
    This will start the Streamlit development server, and your default web browser should open to the application URL (usually `http://localhost:8501`).

3.  **First Run - Document Indexing:**
    If this is the first time running the application or if the Qdrant store is empty/`force_reindex` is enabled, the system will index your knowledge base documents. This process involves:
    *   Reading PDF/text files.
    *   Splitting them into chunks.
    *   Generating embeddings for each chunk.
    *   Writing the documents and embeddings to the Qdrant vector store.
    This can take several minutes depending on the number and size of your documents. Monitor the console output from the Streamlit application for progress logs from the `EnhancedAviationAssistant`.

4.  **Interact with SkyPilot AI:**
    Once the app is running and indexing (if any) is complete, you can start asking aviation-related questions!

## ⚙️ Key Code Components

*   **`aviation_assistant/agent/enhanced_aviation_assistant.py`:** Contains the main `EnhancedAviationAssistant` class, including:
    *   Initialization of LLMs, tools, RAG pipeline, memory, and agent.
    *   Tool definitions and their underlying Python functions.
    *   Prompt engineering (system prompt).
    *   Caching logic (`_cache_response`, `_check_cache`, `_determine_cache_ttl`).
    *   Feedback processing (`provide_feedback`).
    *   Document indexing for RAG (`index_documents`).
*   **`aviation_assistant/data/connectors/`:** Modules for specific API interactions (e.g., `avwx_connector.py`).
*   **`aviation_assistant/data/services/integration_service.py`:** Service layer that uses connectors to provide abstracted data access to tools.
*   **`streamlit_app.py`:** The Streamlit web application code for the UI and user interaction logic.
*   **`calculate_tool_metrics.py`:** Script for evaluating tool selection performance (requires generated oracle and agent JSON files).

## 📊 Evaluation

The project includes an initial evaluation methodology focusing on the agent's tool selection consistency against an LLM-based oracle.
*   **Metrics:** Simple Intersection, Exact Match, Agent Subset/Correct, and Weighted Intersection Hit Rates.
*   **Process:** Compare tools selected by SkyPilot AI against those selected by a SOTA LLM oracle for a sample set of queries.


---
