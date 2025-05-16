# aviation_assistant/agent/enhanced_aviation_assistant.py

import os
import sys
import logging
import json
import time
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime
import re
import hashlib
from pathlib import Path
import traceback

# LangChain imports
from langchain.agents import Tool
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.cache import RedisCache
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools.render import format_tool_to_openai_function
from langchain.globals import set_llm_cache
from haystack.dataclasses import ChatMessage
from haystack.components.writers import DocumentWriter
from langchain_core.utils.function_calling import convert_to_openai_tool # Updated
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages # Updated
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser # Updated
from langchain.agents import Tool, AgentExecutor # AgentExecutor is now directly imported
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages # Updated
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser # Updated
from langchain.memory import ConversationBufferMemory
# from langchain.callbacks.manager import CallbackManager # Not used in this specific method
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # Used when initializing LLM
# from langchain_community.cache import RedisCache # Used when initializing LLM cache
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.messages import AIMessage, HumanMessage # Not directly used here, but good to know
from langchain_core.utils.function_calling import convert_to_openai_tool # Updated for converting tools
# from langchain.globals import set_llm_cache # Used when initializing LLM cache
from langchain_core.runnables import RunnablePassthrough # For LCEL construction

# OpenAI imports for function calling pattern
from langchain_openai import ChatOpenAI

# Haystack 2.x imports
from haystack import Pipeline, Document as HaystackDocument
from haystack.document_stores.in_memory import InMemoryDocumentStore
# from haystack.components.document_stores.faiss import FAISSDocumentStore  # Uncomment for production
from haystack.components.converters import TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.rankers import SentenceTransformersDiversityRanker  # Added for re-ranking
from haystack.components.builders import PromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore 
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

# Redis for caching
import redis

# For handling connection errors
import requests
from requests.exceptions import ConnectionError, Timeout

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


# Import integration service and connectors (assuming these are in your project structure)
from aviation_assistant.data.services.integration_service import IntegrationService
from aviation_assistant.data.connectors.avwx_connector import AVWXConnector
from aviation_assistant.data.connectors.windy_connector import WindyConnector
from aviation_assistant.data.connectors.opensky_connector import OpenSkyConnector



# aviation_assistant/agent/enhanced_aviation_assistant.py

import os
import sys
import logging
import json
import time
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime
import re
import hashlib
from pathlib import Path
import traceback

# LangChain imports
from langchain.agents import Tool, AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.cache import RedisCache
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage # Keep for potential direct use
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.globals import set_llm_cache
from langchain_core.runnables import RunnablePassthrough

# OpenAI imports for function calling pattern
from langchain_openai import ChatOpenAI

# Haystack 2.x imports
from haystack import Pipeline, Document as HaystackDocument
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import TextFileToDocument
from haystack.components.converters.pypdf import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.rankers import SentenceTransformersDiversityRanker
from haystack.components.builders import PromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.writers import DocumentWriter # Added DocumentWriter
from haystack.utils import Secret


# Redis for caching
import redis

# For handling connection errors
import requests
from requests.exceptions import ConnectionError, Timeout

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from aviation_assistant.data.services.integration_service import IntegrationService
from aviation_assistant.data.connectors.avwx_connector import AVWXConnector
from aviation_assistant.data.connectors.windy_connector import WindyConnector
from aviation_assistant.data.connectors.opensky_connector import OpenSkyConnector


class EnhancedAviationAssistant:
    def __init__(self, config: Dict[str, Any], redis_url: str = "redis://localhost:6379"):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.redis_client = redis.from_url(redis_url)
        self.logger.info("Initialized Redis client connection")
        self.openrouter_headers = {
            "HTTP-Referer": self.config.get("site_url", "http://localhost"),
            "X-Title": self.config.get("site_name", "Aviation Assistant")
        }
        self._init_connectors()
        self._init_llm()
        self._init_rag_pipeline() # RAG pipeline now includes document store init
        self._init_tools_and_agent()
        self.logger.info("Enhanced Aviation Assistant initialized successfully")

    def _init_connectors(self):
        try:
            self.avwx = AVWXConnector(api_key=self.config.get("avwx_api_key"))
            self.windy = WindyConnector(api_key=self.config.get("windy_api_key"))
            self.opensky = OpenSkyConnector(
                username=self.config.get("opensky_username"),
                password=self.config.get("opensky_password")
            )
            self.integration_service = IntegrationService(self.avwx, self.windy, self.opensky)
            self.logger.info("Aviation data connectors initialized")
        except Exception as e:
            self.logger.error(f"Error initializing connectors: {str(e)}")
            raise RuntimeError(f"Failed to initialize aviation data connectors: {str(e)}")

    def _init_llm(self):
        agent_llm_model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
        self.logger.info(f"Initializing agent LLM with model: {agent_llm_model_name} via Groq.")
        try:
            self.llm = ChatOpenAI(
                model=agent_llm_model_name,
                openai_api_key=self.config.get("groq_api_key"),
                openai_api_base="https://api.groq.com/openai/v1",
                temperature=0.1, # Slightly lower for more deterministic agent behavior
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()]
            )
            if self.redis_client:
                try:
                    llm_cache = RedisCache(self.redis_client)
                    set_llm_cache(llm_cache)
                    self.logger.info("LLM initialized with Groq Llama model and Redis cache enabled.")
                except Exception as cache_e:
                    self.logger.error(f"Failed to initialize RedisCache for LLM: {str(cache_e)}")
                    self.logger.warning("LLM caching will be disabled or use default in-memory cache.")
            else:
                self.logger.warning("Redis client not available. LLM caching will be disabled.")
        except Exception as e:
            self.logger.error(f"Error initializing LLM: {str(e)}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize LLM: {str(e)}")

    def _extract_chapter_title_from_filename(self, filename: str) -> Optional[str]:
        """
        Attempts to extract a chapter title from a filename.
        Assumes a convention like 'PHAK_Chapter_XX_Actual_Title_Here.pdf'
        or 'Chapter_XX_Actual_Title_Here.pdf'
        """
        name_part = Path(filename).stem # Get filename without extension
        # Try to match "Chapter_XX_Title" or "PHAK_Chapter_XX_Title"
        match = re.search(r"(?:PHAK_)?Chapter_\d{1,2}_(.*)", name_part, re.IGNORECASE)
        if match:
            title = match.group(1).replace('_', ' ').strip()
            # Capitalize first letter of each word
            return ' '.join(word.capitalize() for word in title.split())
        # Fallback for simpler names like "Aircraft_Systems.pdf"
        name_part = name_part.replace('_', ' ').strip()
        if "chapter" not in name_part.lower(): # Avoid picking up "Chapter" itself if it's the only word
             return ' '.join(word.capitalize() for word in name_part.split())
        return None

    def index_documents(self, document_path_str: str):
        """
        Process and index documents (TXT, MD, PDF) for RAG into QdrantDocumentStore.
        If 'force_reindex' is true in config, the Qdrant collection will be recreated.
        Otherwise, documents are added/updated.
        """
        document_path = Path(document_path_str)
        if not document_path.exists():
            self.logger.warning(f"Document path {document_path} does not exist. RAG will have no custom documents.")
            return

        force_reindex = self.config.get("force_reindex", False)

        if self.document_store.count_documents() > 0 and not force_reindex:
            self.logger.info(
                f"Found {self.document_store.count_documents()} documents in Qdrant collection '{self.document_store.index}'. " # .index holds collection name
                f"Skipping indexing. To re-index, set 'force_reindex: true' in config (this will re-init the store "
                f"if you also set recreate_index=True in QdrantDocumentStore init, or delete the qdrant data directory)."
            )
            return
        
        if force_reindex:
            self.logger.warning(
                f"Force re-index is True. If QdrantDocumentStore was initialized with recreate_index=True, "
                f"the collection '{self.document_store.index}' would have been cleared. "
                f"Otherwise, new documents will be added/updated. For a full wipe with on-disk persistence, "
                f"delete the Qdrant data directory ({self.config.get('qdrant_path', './data/qdrant_datastore')}) and restart."
            )
            # If you truly want to wipe and recreate the collection here when force_reindex is True,
            # you'd need to call Qdrant client methods to delete and recreate the collection,
            # or re-initialize self.document_store with recreate_index=True.
            # For simplicity, we'll assume user manages the physical store deletion or uses recreate_index on init.

        self.logger.info(f"Starting document indexing into Qdrant from: {document_path}")
        try:
            text_converter = TextFileToDocument()
            pdf_converter = PyPDFToDocument()
            document_splitter = DocumentSplitter(split_by="word", split_length=250, split_overlap=50)
            
            # IMPORTANT: The doc_embedder model MUST match what Qdrant expects for embedding_dim
            doc_embedder = SentenceTransformersDocumentEmbedder(
                model="sentence-transformers/all-mpnet-base-v2", # Outputs 768 dim
                progress_bar=True,
                batch_size=32
            )
            doc_embedder.warm_up()

            # DocumentWriter works with various document stores, including Qdrant
            document_writer = DocumentWriter(document_store=self.document_store)

            # --- File Collection (same as before) ---
            files_to_process_by_type: Dict[str, List[Path]] = { "text": [], "pdf": [], "json": [] }
            for root, _, files in os.walk(document_path):
                for file in files:
                    file_path_obj = Path(os.path.join(root, file))
                    ext = file_path_obj.suffix.lower()
                    if ext in (".txt", ".md"): files_to_process_by_type["text"].append(file_path_obj)
                    elif ext == ".pdf": files_to_process_by_type["pdf"].append(file_path_obj)
                    elif ext == ".json": files_to_process_by_type["json"].append(file_path_obj)

            all_converted_documents: List[HaystackDocument] = []

            # --- Process Text/MD Files (same as before) ---
            if files_to_process_by_type["text"]:
                # ... (conversion logic same)
                self.logger.info(f"Processing {len(files_to_process_by_type['text'])} TXT/MD files.")
                conversion_result = text_converter.run(sources=files_to_process_by_type["text"])
                all_converted_documents.extend(conversion_result["documents"])


            # --- Process PDF Files (same as before, with metadata) ---
            if files_to_process_by_type["pdf"]:
                # ... (conversion and metadata logic same)
                self.logger.info(f"Processing {len(files_to_process_by_type['pdf'])} PDF files.")
                pdf_conversion_result = pdf_converter.run(sources=files_to_process_by_type["pdf"])
                processed_pdf_docs = []
                for doc in pdf_conversion_result["documents"]:
                    original_filepath_str = doc.meta.get("source_id", str(doc.id) + ".pdf") 
                    doc.meta["file_name"] = Path(original_filepath_str).name
                    chapter_title = self._extract_chapter_title_from_filename(doc.meta["file_name"])
                    if chapter_title:
                        doc.meta["chapter_title"] = chapter_title
                    processed_pdf_docs.append(doc)
                all_converted_documents.extend(processed_pdf_docs)
            
            # --- Split, Embed, and Write Text & PDF Documents ---
            if all_converted_documents:
                self.logger.info(f"Splitting {len(all_converted_documents)} converted documents.")
                split_docs_result = document_splitter.run(documents=all_converted_documents)
                documents_to_embed_and_write = split_docs_result["documents"]

                # Embed documents before writing for Qdrant, as it stores vectors
                self.logger.info(f"Embedding {len(documents_to_embed_and_write)} document chunks.")
                embedded_docs_result = doc_embedder.run(documents=documents_to_embed_and_write)
                documents_for_qdrant = embedded_docs_result["documents"] # These now have .embedding populated

                self.logger.info(f"Writing {len(documents_for_qdrant)} embedded chunks to QdrantDocumentStore.")
                # DocumentWriter will pass documents with embeddings to QdrantDocumentStore
                document_writer.run(documents=documents_for_qdrant)
            else:
                self.logger.info("No TXT, MD, or PDF documents found to process for general embedding pipeline.")


            self.logger.info(f"Document indexing complete. Total documents in Qdrant collection '{self.document_store.index}': {self.document_store.count_documents()}")

        except Exception as e:
            self.logger.error(f"Critical error during document indexing: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed document indexing: {str(e)}")

    def _init_rag_pipeline(self):
        """Initialize Haystack document store and RAG pipeline with QdrantDocumentStore."""
        try:
            document_path_str = self.config.get("document_path", "./data/aviation_documents")
            self.logger.info(f"Setting up RAG pipeline. Document source path: {document_path_str}")

            # --- QdrantDocumentStore Setup ---
            # Define a path for Qdrant to store its data locally.
            # This should be a directory.
            qdrant_data_path_str = self.config.get("qdrant_path", "/workspaces/Aviation_project/aviation_assistant/data/qdrant_datastore")
            qdrant_data_path = Path(qdrant_data_path_str)
            qdrant_data_path.mkdir(parents=True, exist_ok=True) # Ensure the directory exists
            self.logger.info(f"Qdrant local data path: {qdrant_data_path.resolve()}")

            # For 'sentence-transformers/all-mpnet-base-v2', the embedding dimension is 768.
            embedding_dimension = 768
            # Name for your Qdrant collection (like a table in a database)
            qdrant_collection_name = self.config.get("qdrant_collection_name", "aviation_assistant_docs")

            self.document_store = QdrantDocumentStore(
                path=str(qdrant_data_path.resolve()), # Path for on-disk storage
                # url=":memory:", # Use this for in-memory only, no persistence
                # url="http://localhost:6333", # Use this if running a separate Qdrant server/Docker container
                index=qdrant_collection_name,       # Name of the collection in Qdrant
                embedding_dim=embedding_dimension,
                recreate_index=False, # Set to True only if you want to wipe the collection on every start
                return_embedding=True, # Useful if you want embeddings back with documents
                wait_result_from_api=True, # Ensures write operations are confirmed
                #hnsw_config={"m": 16, "ef_construct": 100} # Optional: HNSW parameters for indexing
            )
            self.logger.info(
                f"Initialized QdrantDocumentStore. Path: {qdrant_data_path.resolve()}, "
                f"Collection: {qdrant_collection_name}, Embedding Dim: {embedding_dimension}"
            )

            # --- Initialize other RAG components ---
            self.text_embedder = SentenceTransformersTextEmbedder(
                model="sentence-transformers/all-mpnet-base-v2" # Should match embedding_dim
            )
            self.text_embedder.warm_up()
            
            # Use QdrantEmbeddingRetriever for QdrantDocumentStore
            self.retriever = QdrantEmbeddingRetriever(
                document_store=self.document_store,
                top_k=10
            )

            self.ranker = SentenceTransformersDiversityRanker(
                model="sentence-transformers/all-MiniLM-L6-v2",
                similarity="cosine",
                top_k=5
            )
            self.ranker.warm_up()
            
            rag_prompt_template = """
            You are an expert aviation assistant. Use the following retrieved context from aviation manuals, regulations, and documents to answer the question accurately and concisely. Prioritize information directly from these documents.

            Context:
            {% for doc in documents %}
                Document Source: {{ doc.meta.get('file_name', 'Unknown File') }}
                {% if doc.meta.get('chapter_title') %}Chapter: {{ doc.meta.chapter_title }}{% endif %}
                Content Snippet: {{ doc.content }}
                ---
            {% endfor %}
            
            Based *only* on the context provided above, answer the following question.
            If the context does not contain the answer, state that the information is not found in the provided documents and recommend consulting official sources. Do not use external knowledge.
            
            Question: {{ query }}
            Answer:
            """
            self.prompt_builder = PromptBuilder(template=rag_prompt_template)

            rag_llm_model_name_on_groq = "meta-llama/llama-4-scout-17b-16e-instruct"
            self.rag_llm_generator = OpenAIChatGenerator(
                api_key=Secret.from_token(self.config.get("groq_api_key")),
                model=rag_llm_model_name_on_groq,
                api_base_url="https://api.groq.com/openai/v1",
                generation_kwargs={"max_tokens": 1500, "temperature": 0.0}
            )
            
            self.rag_pipeline = Pipeline()
            self.rag_pipeline.add_component("text_embedder", self.text_embedder)
            self.rag_pipeline.add_component("retriever", self.retriever) # Now QdrantEmbeddingRetriever
            self.rag_pipeline.add_component("ranker", self.ranker)
            self.rag_pipeline.add_component("prompt_builder", self.prompt_builder)

            self.rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
            self.rag_pipeline.connect("retriever.documents", "ranker.documents")
            self.rag_pipeline.connect("ranker.documents", "prompt_builder.documents")
            
            # Index documents if needed
            self.index_documents(document_path_str) 
            
            self.logger.info("RAG pipeline initialized with QdrantDocumentStore.")
        except Exception as e:
            self.logger.error(f"Error initializing RAG pipeline with QdrantDocumentStore: {str(e)}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize RAG pipeline: {str(e)}")
    def _init_tools_and_agent(self):
        try:
            self.tools = [
                Tool(
                    name="get_airport_weather",
                    func=self._get_airport_weather,
                    description="Get comprehensive weather information for a specific airport. Input should be a valid ICAO airport code (e.g., KJFK, EGLL)."
                ),
                Tool(
                    name="get_route_weather",
                    func=self._get_route_weather,
                    description="Get weather information for a flight route between two airports. Input should be two ICAO codes separated by a comma (e.g., 'KJFK,KLAX')."
                ),
                Tool(
                    name="get_flight_level_weather",
                    func=self._get_flight_level_weather,
                    description="Get weather at a specific flight level and geographic location. Input should be latitude,longitude,flight_level (e.g., '40.7,-74.0,FL350')."
                ),
                Tool(
                    name="get_airport_traffic",
                    func=self._get_airport_traffic,
                    description="Get current air traffic information (arrivals/departures) for an airport. Input should be an ICAO code and optional hours for history (e.g., 'KJFK,2' for 2 hours of data, defaults to recent traffic)."
                ),
                Tool( # RENAMED and description updated
                    name="query_aviation_knowledge_base",
                    func=self._query_aviation_knowledge_base, # Function name updated below
                    description="Query the aviation knowledge base for information on regulations, aircraft systems (engines, controls, etc.), principles of flight, aerodynamics, flight procedures, emergency handling, airport operations, and weather theory from official manuals like the Pilot's Handbook of Aeronautical Knowledge (PHAK). Input should be a specific question."
                ),
                Tool(
                    name="get_visual_weather_map",
                    func=self._get_visual_weather_map,
                    description="Generate a URL for a visual weather map (e.g., radar, satellite) for a specified airport. Input should be ICAO code and map type separated by comma (e.g., 'KJFK,radar' or 'EGLL,satellite')."
                )
            ]
            
            openai_tools = [convert_to_openai_tool(t) for t in self.tools]
            self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            
            # Enhanced system message for the agent
            system_message_content = """You are an advanced Aviation Pre-Flight Briefing & Planning AI Agent. Your primary role is to assist pilots by accurately answering questions and providing information using your specialized tools. Base your responses *primarily* on the information retrieved by these tools.

Available Tools:
- get_airport_weather: Current/forecast weather at an airport. (Input: ICAO code)
- get_route_weather: Weather along a flight route. (Input: ICAO1,ICAO2)
- get_flight_level_weather: Weather at a specific altitude/location. (Input: lat,lon,FLXXX)
- get_airport_traffic: Airport arrival/departure info. (Input: ICAO,optional_hours)
- query_aviation_knowledge_base: For questions about aviation regulations, aircraft systems, aerodynamics, flight procedures (normal/emergency), airport operations, weather theory from official manuals (like PHAK). (Input: specific question)
- get_safety_recommendations_for_conditions: For advice, procedures, or safety considerations regarding specific hazardous weather or flight situations, based on documented best practices. (Input: description of condition/situation)
- get_visual_weather_map: Generates links to weather maps. (Input: ICAO,map_type)

Operational Guidelines:
1.  **Analyze Query:** Carefully understand the user's request, considering conversation history for context.
2.  **Select Tool:** If a tool is directly applicable, use it. Formulate precise input for the tool.
3.  **Evaluate Tool Output:**
    a.  If the tool provides a comprehensive answer, use that to respond directly to the user.
    b.  **Handling RAG "Not Found":** If `query_aviation_knowledge_base` or `get_safety_recommendations_for_conditions` returns a message indicating that the information was *not found in its documents* (e.g., "No relevant information found..." or "The information on X is not found..."), **accept this result.** Present this "not found" information to the user and **do not immediately try another tool or re-query the same RAG tool with slight variations for the same original user intent.** Only re-query if the user provides significant new clarifying information for a subsequent query.
    c.  If any tool returns an error (e.g., API issue, connection problem), clearly state the problem to the user.
4.  **Single Tool Attempt (for RAG):** Reinforce the point above: for RAG tools, one well-formed attempt based on the user's query is usually sufficient. If it doesn't find it, it means the documents likely don't have it.
5.  **Synthesis:** Your final answer should synthesize the most relevant tool output.
6.  **Fallback to General Knowledge (Restricted):** Only use your general knowledge if:
    (a) No tool is appropriate.
    (b) A relevant RAG tool was tried and explicitly stated it could not find the information in its documents.
    (c) A tool call resulted in an unrecoverable error.
    When using general knowledge, clearly state this if there's any ambiguity with tool-sourced info.
7.  **Prioritize Safety & Accuracy:** If unsure after tool use, or if information is critical and not found, clearly state limitations and strongly recommend consulting official FAA publications, flight instructors, or other authoritative sources.
8.  **Clarity:** Be clear, concise, and use appropriate aviation terminology where helpful.
"""
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message_content),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            llm_with_tools = self.llm.bind_tools(tools=openai_tools) # bind_tools is correct
            
            agent_runnable = (
                RunnablePassthrough.assign(
                    agent_scratchpad=lambda x: format_to_openai_tool_messages(x["intermediate_steps"])
                )
                | prompt
                | llm_with_tools
                | OpenAIToolsAgentOutputParser()
            )
            
            self.agent_chain = AgentExecutor(
                agent=agent_runnable,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True, # Good for debugging
                max_iterations=6 # Slightly increased if complex queries need more steps
            )
            self.logger.info("Tools and OpenAI tools-based agent initialized successfully.")
        except Exception as e:
            self.logger.error(f"Error initializing tools and agent: {str(e)}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize tools and agent: {str(e)}")

    def process_query(self, query: str, streaming: bool = False) -> Dict[str, Any]:
        # ... (your existing process_query method - no changes needed here based on the request)
        start_time = time.time()
        query_hash = self._normalize_query(query)
        response_id = f"response:{int(start_time)}_{query_hash[:8]}"
        
        cached_response = self._check_cache(query)
        if cached_response:
            self.logger.info(f"Cache hit for query: {query}")
            return {
                "message": cached_response,
                "source": "cache",
                "response_id": response_id,
                "processing_time": 0 # Or load processing time if stored
            }
        
        try:
            self.logger.info(f"Invoking agent_chain with input: '{query}'")
            self.logger.debug(f"Current chat_history for agent: {self.memory.chat_memory.messages}")
            
            # Ensure chat_history is correctly passed if needed by your specific RunnablePassthrough or prompt
            # The AgentExecutor handles memory automatically if configured.
            response_from_agent = self.agent_chain.invoke({
                "input": query
                # "chat_history" is implicitly handled by memory in AgentExecutor
            })
            
            output_message = response_from_agent.get("output", str(response_from_agent)) # Handle if output key missing
            
            processing_time = time.time() - start_time
            self._cache_response(query, output_message, response_id) # Cache successful responses
            
            return {
                "message": output_message,
                "source": "agent",
                "response_id": response_id,
                "processing_time": processing_time
            }
        except Exception as e:
            self.logger.error(f"Error processing query with agent: {str(e)}", exc_info=True)
            fallback_response = self._create_fallback_response(e)
            processing_time = time.time() - start_time
            # Do not cache error responses in the primary query cache
            return {
                "message": fallback_response,
                "error": str(e), # Include error string for debugging/logging
                "source": "error",
                "response_id": response_id,
                "processing_time": processing_time
            }


    def _create_fallback_response(self, error: Exception) -> str:
        # ... (your existing _create_fallback_response method)
        if isinstance(error, (ConnectionError, Timeout, requests.exceptions.RequestException)):
            return (
                "I'm sorry, I'm currently unable to connect to external aviation data services. "
                "This might be a temporary network issue or the services may be down. "
                "Please check your internet connection and try again in a few moments."
            )
        elif "rate limit" in str(error).lower():
            return (
                "I apologize, but a data provider I rely on is currently experiencing high traffic (rate limit exceeded). "
                "This usually resolves within a short period. Please try your request again shortly."
            )
        elif isinstance(error, (json.JSONDecodeError)):
             return (
                "I encountered an issue processing data from one of the services (invalid data format). "
                "This is likely a temporary issue with the data provider. Please try again later."
            )
        else: # Generic fallback
            self.logger.error(f"Unhandled error type for fallback: {type(error).__name__} - {str(error)}")
            return (
                "I encountered an unexpected issue while processing your request. "
                "The technical team has been notified. Please try again later, or rephrase your question."
            )

    def provide_feedback(self, response_id: str, is_positive: bool) -> None:
        # ... (your existing provide_feedback method)
        try:
            parts = response_id.split("_")
            if len(parts) < 2:
                self.logger.warning(f"Invalid response_id format for feedback: {response_id}")
                return

            query_hash_part = parts[1] # Assuming the hash is the second part
            
            feedback_key = f"feedback:entry:{response_id}" # More specific key for feedback entry
            self.redis_client.hset(feedback_key, mapping={
                "is_positive": "1" if is_positive else "0",
                "timestamp": time.time()
            })
            # Optional: Set an expiry for the feedback entry itself
            self.redis_client.expire(feedback_key, 86400 * 30) # Keep feedback for 30 days

            # Adjust TTL of the cached query response based on feedback
            # The cache key for the query response itself
            cache_key_for_query = f"cache:query:{query_hash_part}" 
            
            if self.redis_client.exists(cache_key_for_query):
                if is_positive:
                    # Extend TTL for positive feedback - e.g., keep for 7 days
                    new_ttl = 86400 * 7
                    self.redis_client.expire(cache_key_for_query, new_ttl)
                    self.logger.info(f"Extended TTL for {cache_key_for_query} to {new_ttl}s due to positive feedback on {response_id}.")
                else:
                    # Significantly reduce TTL or delete for negative feedback - e.g., keep for 5 mins or delete
                    # Deleting might be too aggressive if it was a transient issue, reducing TTL is safer.
                    new_ttl = 300 # 5 minutes
                    self.redis_client.expire(cache_key_for_query, new_ttl)
                    self.logger.info(f"Reduced TTL for {cache_key_for_query} to {new_ttl}s due to negative feedback on {response_id}.")
            else:
                self.logger.info(f"No cached query response found at {cache_key_for_query} to adjust TTL for feedback on {response_id}.")

        except Exception as e:
            self.logger.error(f"Error storing feedback for {response_id}: {str(e)}", exc_info=True)


    def _check_cache(self, query: str) -> Optional[str]:
        # ... (your existing _check_cache method)
        try:
            query_key_hash = self._normalize_query(query)
            cache_key = f"cache:query:{query_key_hash}"
            
            cached_data_json = self.redis_client.get(cache_key)
            if cached_data_json:
                cached_data = json.loads(cached_data_json.decode('utf-8'))
                # Optional: Check for recent negative feedback on this specific query_hash
                # This requires storing feedback linked to query_hash, not just response_id
                # For simplicity, current feedback logic adjusts TTL of the response directly.
                # If you want more aggressive cache invalidation based on any negative feedback for this query:
                # feedback_score = self.redis_client.get(f"feedback:score:{query_key_hash}")
                # if feedback_score and int(feedback_score) < 0:
                #    self.logger.info(f"Cache hit for {query_key_hash}, but ignoring due to negative feedback score.")
                #    return None
                
                self.logger.info(f"Cache hit for query (hash: {query_key_hash}).")
                return cached_data.get("response_content") # Assuming you store content this way
            return None
        except Exception as e:
            self.logger.error(f"Error checking cache for query '{query[:50]}...': {str(e)}", exc_info=True)
            return None # Fail safe, retrieve fresh data

    def _cache_response(self, query: str, response_content: str, response_id: str) -> None:
        # ... (your existing _cache_response method)
        try:
            query_key_hash = self._normalize_query(query)
            cache_key = f"cache:query:{query_key_hash}"
            ttl = self._determine_cache_ttl(query) # Your existing TTL logic

            # Store response content and some metadata together
            cache_payload = json.dumps({
                "response_content": response_content,
                "response_id": response_id, # Link back to the specific response ID
                "original_query": query,    # For easier debugging from cache
                "timestamp": time.time(),
                "ttl_at_caching": ttl
            })
            
            self.redis_client.setex(cache_key, ttl, cache_payload)
            
            # Storing metadata related to the response_id is still useful for feedback linking etc.
            self.redis_client.hset(f"meta:response:{response_id}", mapping={
                "query_hash": query_key_hash, # Link response_id to the query_hash
                "original_query": query,
                "timestamp": time.time(),
                "initial_ttl": ttl
            })
            self.logger.info(f"Cached response for query (hash: {query_key_hash}) with TTL {ttl}s.")
        except Exception as e:
            self.logger.error(f"Error caching response for query '{query[:50]}...': {str(e)}", exc_info=True)

    def _determine_cache_ttl(self, query: str) -> int:
        # ... (your existing _determine_cache_ttl method - looks good)
        query_lower = query.lower()
        if any(term in query_lower for term in ["current weather", "metar", "taf", "notam", "traffic", "active runway"]):
            return 300  # 5 minutes for highly dynamic data
        elif any(term in query_lower for term in ["forecast", "winds aloft", "outlook briefing"]):
            return 1800  # 30 minutes for forecasts
        elif any(term in query_lower for term in ["phak", "handbook", "regulation", "procedure", "aircraft system", "aerodynamics"]):
            return 86400 * 7  # 7 days for relatively static knowledge base queries
        elif "airport information" in query_lower or "frequencies" in query_lower:
            return 86400 # 1 day for airport data that changes less frequently than weather
        return 3600  # 1 hour default

    def _normalize_query(self, query: str) -> str:
        # ... (your existing _normalize_query method - looks good)
        normalized = re.sub(r'\s+', ' ', query.lower().strip())
        # Further normalization: remove punctuation that might not affect meaning
        normalized = re.sub(r'[^\w\s-]', '', normalized) # Keep alphanumeric, spaces, hyphens
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()


    def safe_api_call(self, func: Callable, *args, default_value: Any = None, tool_name: str = "Unknown Tool", **kwargs) -> Any:
        # ... (enhanced safe_api_call)
        try:
            return func(*args, **kwargs)
        except ConnectionError as e:
            self.logger.error(f"Connection error in {tool_name} calling {func.__name__}: {e}")
            return default_value or {"error": f"Connection failed while using {tool_name}. Please check your internet connection and try again."}
        except Timeout as e:
            self.logger.error(f"Timeout error in {tool_name} calling {func.__name__}: {e}")
            return default_value or {"error": f"The request to {tool_name} timed out. The service might be busy or temporarily unavailable."}
        except requests.exceptions.HTTPError as e: # More specific for HTTP errors
            self.logger.error(f"HTTP error in {tool_name} calling {func.__name__}: {e.response.status_code} - {e.response.text[:200]}")
            if e.response.status_code == 401: # Unauthorized
                return default_value or {"error": f"Authentication failed for {tool_name}. Please check API key configuration."}
            elif e.response.status_code == 403: # Forbidden
                 return default_value or {"error": f"Access denied for {tool_name}. You might not have permission for this resource."}
            elif e.response.status_code == 404: # Not Found
                return default_value or {"error": f"The requested resource was not found by {tool_name}."}
            elif e.response.status_code == 429: # Rate limit
                return default_value or {"error": f"Rate limit exceeded for {tool_name}. Please try again later."}
            return default_value or {"error": f"A server-side error occurred with {tool_name} (HTTP {e.response.status_code})."}
        except json.JSONDecodeError as e: # If API returns malformed JSON
            self.logger.error(f"JSON decode error in {tool_name} calling {func.__name__}: {str(e)}")
            return default_value or {"error": f"{tool_name} returned improperly formatted data. This might be a temporary service issue."}
        except Exception as e: # Catch-all for other unexpected errors
            self.logger.error(f"Unexpected error in {tool_name} calling {func.__name__}: {type(e).__name__} - {str(e)}", exc_info=True)
            return default_value or {"error": f"An unexpected error occurred while using {tool_name}."}

    # --- Tool Implementations ---
    # (Make sure to use the enhanced safe_api_call within these)

    def _get_airport_weather(self, icao_code: str) -> str:
        # ... (Use self.safe_api_call with tool_name="AirportWeatherAPI")
        icao_code = icao_code.strip().upper()
        self.logger.info(f"Tool call: _get_airport_weather for {icao_code}")
        # Cache within the tool for raw API data can be beneficial
        raw_data_cache_key = f"api_data:airport_weather:{icao_code}"
        cached_raw_data = self.redis_client.get(raw_data_cache_key)

        if cached_raw_data:
            try:
                airport_weather = json.loads(cached_raw_data.decode('utf-8'))
                self.logger.info(f"Using cached raw API data for airport weather: {icao_code}")
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to decode cached JSON for {raw_data_cache_key}, fetching fresh.")
                airport_weather = None
        else:
            airport_weather = None

        if not airport_weather:
            airport_weather = self.safe_api_call(
                self.integration_service.get_airport_weather,
                icao_code,
                default_value={"error": f"Could not retrieve weather for {icao_code}."},
                tool_name="AirportWeatherService"
            )
            if "error" not in airport_weather and airport_weather: # Cache successful raw API responses
                try:
                    self.redis_client.setex(raw_data_cache_key, 300, json.dumps(airport_weather)) # Cache raw data for 5 mins
                except TypeError as te:
                    self.logger.error(f"TypeError caching raw airport weather data for {icao_code}: {te}")


        if "error" in airport_weather and airport_weather.get("error"):
            return f"Error: {airport_weather['error']}"
        if not airport_weather: # Should be caught by default_value if API fails
            return f"Error: No weather data returned for {icao_code}."

        # Format the response using the enhanced formatter
        # This part should be robust to missing keys in airport_weather
        try:
            formatted_data = self._format_weather_response(airport_weather)
            response_parts = [formatted_data["text"]]
            if formatted_data["hazards"]:
                hazard_text = "\n\nHazardous Conditions:"
                for sev in ["high", "moderate", "low"]: # Order by severity
                    cat_hazards = [h for h in formatted_data["hazards"] if h["severity"] == sev]
                    if cat_hazards:
                        hazard_text += f"\n{sev.upper()} SEVERITY:"
                        for i, hazard in enumerate(cat_hazards, 1):
                             hazard_text += f"\n  {i}. {hazard['description']} (Category: {hazard['category']})"
                response_parts.append(hazard_text)
            if formatted_data["visualization_links"]:
                response_parts.append("\n\nVisualization Links:")
                for name, url in formatted_data["visualization_links"].items():
                    response_parts.append(f"- {name.replace('_', ' ').capitalize()}: {url}")
            if formatted_data["safety_recommendations"]:
                response_parts.append("\n\nSafety Recommendations:")
                for i, rec in enumerate(formatted_data["safety_recommendations"], 1):
                    response_parts.append(f"{i}. {rec}")
            return "\n".join(response_parts)
        except Exception as fmt_e:
            self.logger.error(f"Error formatting weather response for {icao_code}: {fmt_e}", exc_info=True)
            # Fallback to a simpler representation of raw data if formatting fails
            return f"Successfully retrieved weather data for {icao_code}, but encountered an issue during formatting. Raw summary (if available): {airport_weather.get('summary', 'Not available')}"


    def _get_route_weather(self, departure_destination: str) -> str:
        # ... (Use self.safe_api_call with tool_name="RouteWeatherAPI")
        # Similar caching for raw API data can be applied here
        self.logger.info(f"Tool call: _get_route_weather for {departure_destination}")
        try:
            parts = departure_destination.split(",")
            if len(parts) != 2:
                return "Error: Please provide departure and destination ICAO codes separated by a comma (e.g., 'KJFK,KLAX')."
            departure = parts[0].strip().upper()
            destination = parts[1].strip().upper()

            # Raw data caching
            raw_data_cache_key = f"api_data:route_weather:{departure}_{destination}"
            cached_raw_data = self.redis_client.get(raw_data_cache_key)
            route_weather = None
            if cached_raw_data:
                try:
                    route_weather = json.loads(cached_raw_data.decode('utf-8'))
                    self.logger.info(f"Using cached raw API data for route weather: {departure} to {destination}")
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to decode cached JSON for {raw_data_cache_key}, fetching fresh.")
            
            if not route_weather:
                route_weather = self.safe_api_call(
                    self.integration_service.get_route_weather,
                    departure, destination,
                    default_value={"error": f"Could not retrieve route weather for {departure}-{destination}."},
                    tool_name="RouteWeatherService"
                )
                if "error" not in route_weather and route_weather:
                    try:
                        self.redis_client.setex(raw_data_cache_key, 900, json.dumps(route_weather)) # Cache raw for 15 mins
                    except TypeError as te:
                        self.logger.error(f"TypeError caching raw route weather data for {departure}-{destination}: {te}")

            if "error" in route_weather and route_weather.get("error"):
                return f"Error: {route_weather['error']}"
            if not route_weather:
                return f"Error: No route weather data returned for {departure}-{destination}."

            # Formatting logic similar to _get_airport_weather
            try:
                formatted_data = self._format_weather_response(route_weather) # Ensure this can handle route data structure
                response_parts = [f"Route Weather: {departure} to {destination}\n", formatted_data["text"]]
                # ... (add hazards, visualizations, recommendations as in _get_airport_weather)
                if formatted_data["hazards"]:
                    response_parts.append("\n\nHazardous Conditions Along Route:")
                    # (similar hazard formatting as airport weather)
                if formatted_data["visualization_links"]:
                    response_parts.append("\n\nRoute Visualization Links:")
                    # (similar link formatting)
                if formatted_data["safety_recommendations"]:
                    response_parts.append("\n\nSafety Recommendations for Route:")
                    # (similar rec formatting)
                return "\n".join(response_parts)
            except Exception as fmt_e:
                self.logger.error(f"Error formatting route weather for {departure}-{destination}: {fmt_e}", exc_info=True)
                return f"Successfully retrieved route weather for {departure}-{destination}, but encountered formatting issue. Summary: {route_weather.get('summary', 'Not available')}"

        except Exception as e: # General catch for parsing etc.
            self.logger.error(f"Unexpected error in _get_route_weather: {e}", exc_info=True)
            return f"Error processing route weather request: {str(e)}"


    def _get_flight_level_weather(self, location_fl: str) -> str:
        # ... (Use self.safe_api_call with tool_name="FlightLevelWeatherAPI")
        self.logger.info(f"Tool call: _get_flight_level_weather for {location_fl}")
        # Similar raw data caching can be applied
        try:
            parts = location_fl.split(",")
            if len(parts) != 3:
                return "Error: Please provide latitude, longitude, and flight level (e.g., '40.7,-74.0,FL350')."
            latitude = float(parts[0].strip())
            longitude = float(parts[1].strip())
            flight_level_str = parts[2].strip().upper()

            fl_weather = self.safe_api_call(
                self.integration_service.get_flight_level_weather,
                latitude, longitude, flight_level_str,
                default_value={"error": f"Could not retrieve weather at {flight_level_str} for {latitude},{longitude}."},
                tool_name="FlightLevelWeatherService"
            )

            if "error" in fl_weather and fl_weather.get("error"):
                return f"Error: {fl_weather['error']}"
            if not fl_weather:
                 return f"Error: No flight level weather data returned for {location_fl}."
            
            try:
                formatted_data = self._format_weather_response(fl_weather) # Ensure this handles FL data
                response_parts = [f"Flight Level Weather at {flight_level_str} ({latitude}, {longitude}):\n", formatted_data["text"]]
                # Add other relevant parts like temperature, wind if available and formatted
                # (e.g., from formatted_data if _format_weather_response is adapted)
                return "\n".join(response_parts)
            except Exception as fmt_e:
                self.logger.error(f"Error formatting flight level weather for {location_fl}: {fmt_e}", exc_info=True)
                return f"Successfully retrieved flight level weather for {location_fl}, but encountered formatting issue. Summary: {fl_weather.get('summary', 'Not available')}"
        except ValueError:
            return "Error: Invalid format for latitude, longitude, or flight level. Please use numbers for lat/lon."
        except Exception as e:
            self.logger.error(f"Unexpected error in _get_flight_level_weather: {e}", exc_info=True)
            return f"Error processing flight level weather request: {str(e)}"


    def _get_airport_traffic(self, icao_hours: str) -> str:
        # ... (Use self.safe_api_call with tool_name="AirportTrafficAPI")
        self.logger.info(f"Tool call: _get_airport_traffic for {icao_hours}")
        # Similar raw data caching
        try:
            parts = icao_hours.split(",")
            icao_code = parts[0].strip().upper()
            hours = 1 # Default to 1 hour for recent traffic
            if len(parts) > 1:
                try:
                    hours = int(parts[1].strip())
                    if not (0 < hours <= 24): # Reasonable limit for "recent" traffic
                        return "Error: Hours must be between 1 and 24."
                except ValueError:
                    return "Error: Invalid format for hours. Must be a number."
            
            traffic_data = self.safe_api_call(
                self.integration_service.get_airport_traffic,
                icao_code, hours,
                default_value={"error": f"Could not retrieve traffic for {icao_code}."},
                tool_name="AirportTrafficService"
            )

            if "error" in traffic_data and traffic_data.get("error"):
                return f"Error: {traffic_data['error']}"
            if not traffic_data:
                return f"Error: No traffic data returned for {icao_code}."

            # Formatting
            summary = [f"Airport Traffic for {icao_code} (last {hours} hour(s)):\n"]
            arrivals = traffic_data.get("arrivals", [])
            departures = traffic_data.get("departures", [])

            summary.append(f"Arrivals ({len(arrivals)}):")
            if arrivals:
                for arr in arrivals[:5]: # Show top 5
                    summary.append(f"  - {arr.get('callsign', 'N/A')} from {arr.get('estDepartureAirport', 'N/A')} at {datetime.fromtimestamp(arr.get('lastSeen', 0)).strftime('%H:%M')}Z (Altitude: {arr.get('geoAltitude', 'N/A')}m)")
            else:
                summary.append("  No recent arrivals found.")
            
            summary.append(f"\nDepartures ({len(departures)}):")
            if departures:
                for dep in departures[:5]: # Show top 5
                    summary.append(f"  - {dep.get('callsign', 'N/A')} to {dep.get('estArrivalAirport', 'N/A')} departed around {datetime.fromtimestamp(dep.get('firstSeen', 0)).strftime('%H:%M')}Z")
            else:
                summary.append("  No recent departures found.")
            
            return "\n".join(summary)

        except Exception as e:
            self.logger.error(f"Unexpected error in _get_airport_traffic: {e}", exc_info=True)
            return f"Error processing airport traffic request: {str(e)}"

    # RENAMED function to match tool
    def _query_aviation_knowledge_base(self, query: str) -> str:
        """
        Query aviation knowledge base (PHAK, regulations, etc.) using Haystack RAG.
        """
        self.logger.info(f"Tool call: _query_aviation_knowledge_base for query: '{query[:100]}...'")

        try:
            if not self.rag_pipeline or not self.rag_llm_generator:
                self.logger.error("RAG pipeline or RAG LLM generator not initialized for knowledge base query.")
                return "Error: Knowledge Base RAG system is currently unavailable."
            
            pipeline_inputs = {
                "text_embedder": {"text": query},
                "ranker": {"query": query},
                "prompt_builder": {"query": query}
            }
            
            pipeline_result = self.rag_pipeline.run(data=pipeline_inputs)

            final_prompt_string = pipeline_result.get("prompt_builder", {}).get("prompt")
            if not final_prompt_string:
                self.logger.error(f"RAG prompt_builder did not return a 'prompt'. Result: {pipeline_result}")
                retrieved_docs = pipeline_result.get("ranker", {}).get("documents", [])
                if not retrieved_docs:
                    return "Based on the available documents, no specific information was found for your query. You may need to consult official FAA publications or a certified flight instructor."
                return "Error: Could not construct the final prompt for the knowledge base query."

            llm_response = self.rag_llm_generator.run(
                messages=[ChatMessage.from_user(final_prompt_string)] # Haystack's ChatMessage
            )
            
            if llm_response and "replies" in llm_response and llm_response["replies"]:
                answer = llm_response["replies"][0].text # CORRECTED: Use .text
                
                if "not found in the provided documents" in answer.lower() or \
                   "context does not contain the answer" in answer.lower() or \
                   "no relevant information" in answer.lower() :
                    self.logger.info(f"RAG LLM indicated no specific answer found in documents for query: {query}")
                    return answer 
                return answer
            else:
                self.logger.warning(f"No valid 'replies' from RAG LLM for knowledge base query: {query}. Response: {llm_response}")
                return "The knowledge base search did not return a specific answer at this time."
        except Exception as e:
            self.logger.error(f"Error querying aviation knowledge base with RAG: {str(e)}", exc_info=True)
            # Ensure the error message itself doesn't cause another error if it contains special chars for f-string
            error_message = str(e).replace("{", "{{").replace("}", "}}")
            return f"An error occurred while searching the aviation knowledge base: {error_message}"

    def _get_visual_weather_map(self, params: str) -> str:
        # ... (Use self.safe_api_call with tool_name="VisualWeatherMapAPI")
        self.logger.info(f"Tool call: _get_visual_weather_map for '{params}'")
        try:
            parts = params.split(",")
            if not parts or not parts[0].strip():
                return "Error: ICAO code is required for the visual weather map."
            
            icao_code = parts[0].strip().upper()
            map_type = "radar" # Default
            if len(parts) > 1 and parts[1].strip():
                map_type = parts[1].strip().lower()
                if map_type not in ["radar", "satellite", "wind", "temperature", "clouds", "pressure", "rain", "snow"]: # Example valid types
                    return f"Error: Invalid map type '{map_type}'. Valid types include radar, satellite, wind, etc."

            # Get airport info to find coordinates (could be cached)
            airport_info = self.safe_api_call(
                self.integration_service.get_airport_details, 
                icao_code,
                default_value={"error": f"Could not retrieve info for airport {icao_code}."},
                tool_name="AirportDetailService" # Updated tool name for logging
            )

            if "error" in airport_info and airport_info.get("error"):
                return f"Error: {airport_info['error']}"
            
            # The structure now directly matches what the rest of the method expects
            # airport_info will have a "coordinates" key with "lat" and "lon"
            if not airport_info or "coordinates" not in airport_info:
                return f"Error: Could not find coordinates for {icao_code} to generate map."

            coords = airport_info["coordinates"]
            lat = coords.get("lat")
            lon = coords.get("lon")

            if "error" in airport_info and airport_info.get("error"):
                return f"Error: {airport_info['error']}"
            if not airport_info or "coordinates" not in airport_info:
                return f"Error: Could not find coordinates for {icao_code} to generate map."

            coords = airport_info["coordinates"]
            lat = coords.get("lat")
            lon = coords.get("lon")

            if lat is None or lon is None:
                return f"Error: Missing latitude or longitude for {icao_code}."

            map_url_data = self.safe_api_call(
                self.windy.get_visualization_url, # Assuming this returns a direct URL string or dict with URL
                lat=lat, lon=lon, map_type=map_type, zoom=8, # Adjust zoom as needed
                default_value={"error": f"Failed to generate '{map_type}' map URL for {icao_code}."},
                tool_name="WindyMapVisualization"
            )

            # Handle if map_url_data itself is a dict with an error or the URL string
            map_url_str = ""
            if isinstance(map_url_data, dict):
                if "error" in map_url_data and map_url_data.get("error"):
                    return f"Error: {map_url_data['error']}"
                map_url_str = map_url_data.get("url", "") # Assuming windy connector returns a dict with 'url' key
            elif isinstance(map_url_data, str):
                map_url_str = map_url_data
            
            if not map_url_str:
                return f"Error: Could not generate the '{map_type}' map URL for {icao_code}."

            return (f"Visual Weather Map for {icao_code} ({airport_info.get('name', 'Unknown Airport')}):\n"
                    f"Type: {map_type.capitalize()}\n"
                    f"URL: {map_url_str}\n"
                    f"Note: This map provides a {map_type} visualization around {icao_code}.")

        except Exception as e:
            self.logger.error(f"Unexpected error in _get_visual_weather_map for '{params}': {e}", exc_info=True)
            return f"Error generating visual weather map: {str(e)}"

    # --- Helper methods for weather response formatting ---
    def _format_weather_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Ensure this method is robust to missing keys in 'data'
        formatted = {
            "text": "Weather data summary not available.",
            "hazards": [],
            "visualization_links": {},
            "safety_recommendations": []
        }

        if not data: # Handle empty or None data
            return formatted

        formatted["text"] = self._create_human_readable_summary(data)
        # Ensure _extract_categorized_hazards is robust to missing 'hazardous_conditions'
        formatted["hazards"] = self._extract_categorized_hazards(data.get("hazardous_conditions", []))
        formatted["visualization_links"] = self._generate_visualization_links(data)
        # Ensure _generate_safety_recommendations is robust
        formatted["safety_recommendations"] = self._generate_safety_recommendations(data.get("hazardous_conditions", []))
        
        return formatted

    def _create_human_readable_summary(self, data: Dict[str, Any]) -> str:
        # ... (Make this robust to missing keys)
        if not data: return "No summary data provided."
        
        summary_parts = []
        station_info = data.get("station", {})
        if station_info:
            summary_parts.append(f"Airport: {station_info.get('name', 'N/A')} ({station_info.get('icao', 'N/A')})")
        
        if "metar_summary" in data: summary_parts.append(f"Current Conditions: {data['metar_summary']}")
        if "taf_summary" in data: summary_parts.append(f"Forecast: {data['taf_summary']}")
        if data.get("summary"): summary_parts.append(f"Overall Summary: {data['summary']}")

        if not summary_parts: return "Basic weather information not available."
        return "\n".join(summary_parts)


    def _extract_categorized_hazards(self, hazards_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # ... (Make robust)
        if not hazards_list: return []
        
        processed_hazards = []
        for hazard in hazards_list:
            if not isinstance(hazard, dict): continue # Skip malformed entries
            description = hazard.get("description", "No description")
            severity = hazard.get("severity", "unknown").lower()
            category = self._categorize_hazard(description) # Your existing categorization
            processed_hazards.append({
                "description": description,
                "severity": severity if severity in ["high", "moderate", "low"] else "unknown",
                "category": category,
                "source": hazard.get("source", "N/A")
            })
        return processed_hazards

    def _categorize_hazard(self, description: str) -> str:
        # ... (your existing categorization logic - seems fine)
        description_lower = description.lower()
        if any(term in description_lower for term in ["visibility", "fog", "haze", "obscuration", "br", "mist"]): return "visibility"
        if any(term in description_lower for term in ["wind", "gust", "crosswind", "squall"]): return "wind"
        if any(term in description_lower for term in ["thunderstorm", "ts", "lightning", "cb", "cumulonimbus", "convective"]): return "convective"
        if any(term in description_lower for term in ["ice", "icing", "fz", "freez", "snow", "sleet", "hail"]): return "icing_precipitation"
        if any(term in description_lower for term in ["turbulence", "turb", "shear", "chop", "llws"]): return "turbulence"
        if any(term in description_lower for term in ["ceiling", "cigs", "ovc", "bkn", "sct low"]): return "ceiling"
        return "other"

    def _generate_visualization_links(self, data: Dict[str, Any]) -> Dict[str, str]:
        # ... (Make robust)
        if not data: return {}
        links = {}
        if isinstance(data.get("route_visualization"), dict):
            links.update(data["route_visualization"])
        if isinstance(data.get("visualization_url"), str): # For single primary URL
            links["primary_visualization"] = data["visualization_url"]
        # You might want to add logic here to construct a Windy link if lat/lon is in 'data'
        # and no other links are present.
        return links

    def _generate_safety_recommendations(self, hazards_list: List[Dict[str, Any]]) -> List[str]:
        # ... (Make robust and more comprehensive if possible)
        if not hazards_list: return ["General: Always fly with caution and be prepared for changing conditions."]
        
        recommendations = []
        has_high_severity = False
        for hazard_entry in hazards_list: # Ensure this is the list of categorized hazards
            if not isinstance(hazard_entry, dict): continue
            
            description = hazard_entry.get("description", "").lower()
            severity = hazard_entry.get("severity", "unknown").lower() # From _extract_categorized_hazards
            category = hazard_entry.get("category", "other")     # From _extract_categorized_hazards

            if severity == "high": has_high_severity = True

            if category == "visibility":
                if severity == "high": recommendations.append("Severely reduced visibility: Consider delaying flight, filing IFR, or choosing an alternate route/destination. Ensure instrument currency if applicable.")
                else: recommendations.append("Reduced visibility: Be prepared for potential instrument conditions. Maintain situational awareness.")
            elif category == "wind":
                if severity == "high": recommendations.append("Strong/gusty winds: Exercise extreme caution during takeoff/landing. Check aircraft crosswind limits. Consider delaying if limits are approached or exceeded.")
                else: recommendations.append("Moderate winds: Be prepared for potential crosswinds. Review crosswind landing techniques.")
            elif category == "convective":
                recommendations.append("Convective activity (thunderstorms): Avoid by at least 20 NM. Do not attempt to fly under or through. Monitor radar and PIREPs.")
            elif category == "icing_precipitation":
                if severity == "high": recommendations.append("High risk of icing: Avoid flight into known or forecast icing conditions. If encountered, exit icing conditions immediately. Ensure aircraft is equipped for icing if flight is unavoidable and legal.")
                else: recommendations.append("Potential for icing: Be alert for airframe icing. Monitor OAT. Know escape routes.")
            elif category == "turbulence":
                if severity == "high": recommendations.append("Severe turbulence: Avoid area if possible. If encountered, slow to maneuvering speed (Va), maintain wings level, and accept altitude deviations. Ensure all items are secured.")
                else: recommendations.append("Moderate turbulence: Advise passengers, secure cabin. Consider speed adjustments if appropriate.")
            elif category == "ceiling":
                if severity == "high": recommendations.append("Low ceilings: VFR flight likely not possible or marginal. Plan for IFR or delay. Check terrain clearance.")
                else: recommendations.append("Lower ceilings: Monitor trends. Be prepared for deteriorating conditions if VFR.")

        if not recommendations and hazards_list: # If specific recs weren't triggered but hazards exist
            recommendations.append("Hazardous conditions reported. Review all available weather data and exercise caution.")
        
        if has_high_severity:
            recommendations.append("CRITICAL: One or more high-severity hazards reported. Thoroughly assess risks before flight.")
        
        recommendations.append("Always consult official weather briefings and pilot reports (PIREPs).")
        return list(set(recommendations)) # Remove duplicates
import redis
import hashlib
import re

def normalize_query_for_delete(query: str) -> str:
    normalized = re.sub(r'\s+', ' ', query.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()

def clear_specific_query_caches(redis_url: str, queries_to_clear: list):
    r = redis.from_url(redis_url)
    try:
        r.ping()
        print(f"Connected to Redis at {redis_url}")
    except redis.exceptions.ConnectionError as e:
        print(f"Could not connect to Redis: {e}")
        return

    for query in queries_to_clear:
        query_hash = normalize_query_for_delete(query)
        cache_key = f"cache:query:{query_hash}"
        if r.exists(cache_key):
            r.delete(cache_key)
            print(f"Deleted cache key: {cache_key} for query: '{query}'")
        else:
            print(f"Cache key not found: {cache_key} for query: '{query}'")



if __name__ == '__main__':
    # Example Usage (requires API keys and data setup)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    config_example = {
        "groq_api_key" : "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",
        "avwx_api_key": "jLX2iQWBGYttAwYBxiFCUc_4ljrbNOqCNvHe5hjtS8o",
        "windy_api_key": "ymAvnkZvsbZpxgMVlHabWuDIG6ZgHsJd",
        "opensky_username": "sami___ag",
        "opensky_password": "716879Sami.",
        "document_path": "/workspaces/Aviation_project/aviation_assistant/data/aviation_documents",
        "site_url": "http://aviationassistant.com",
        "site_name": "Aviation Assistant"
    }

    # --- Optional: Clear specific query caches before running new tests ---
    # from your_module import clear_specific_query_caches # Assuming you have this function
    # redis_url_for_clear = "redis://localhost:6379"
    # queries_to_clear_before_test = [
    #     "What is the DECIDE model in aeronautical decision making?",
    #     "Explain Bernoulli's Principle as it relates to flight.",
    #     "What are the primary flight controls of an airplane and their functions?",
    #     "Describe the standard VFR cruising altitudes."
    # ]
    # print("\n--- Clearing specific query caches (optional) ---")
    # clear_specific_query_caches(redis_url_for_clear, queries_to_clear_before_test)
    # print("--- Finished clearing caches ---\n")

    try:
        print("Initializing EnhancedAviationAssistant...")
        assistant = EnhancedAviationAssistant(config=config_example)
        print("Assistant initialized.")

        # --- Test Suite Tailored to Your Chapters ---
        test_queries = [
            # Weather Tool Test (keeping one for sanity check)
            {"type": "Weather", "query": "Get weather for CYYZ"},

            # --- PHAK Knowledge Base Tests (Targeting Your Chapters) ---

            # For phak_Chapter_2_Aeronautical_Decision_Making.pdf
            {"type": "PHAK_ADM", "query": "What are the hazardous attitudes in aviation according to the PHAK?"},
            # {"type": "PHAK_ADM", "query": "Explain the 3P model (Perceive, Process, Perform) in ADM from the PHAK."},
            # {"type": "PHAK_ADM", "query": "What is single-pilot resource management as described in PHAK Chapter 2 on ADM?"},


            # # For phak_Chapter_4_Principles_of_Flight.pdf
            # {"type": "PHAK_Principles", "query": "How does an airfoil generate lift according to PHAK Chapter 4?"},
            # {"type": "PHAK_Principles", "query": "What are the four forces acting on an aircraft in flight as explained in the Principles of Flight chapter?"},
            # {"type": "PHAK_Principles", "query": "Describe wingtip vortices based on the PHAK's Principles of Flight chapter."},

            # # For phak_Chapter_6_Flight_Controls.pdf
            # {"type": "PHAK_Controls", "query": "What are the primary flight controls and what does each control, as per PHAK Chapter 6?"},
            # {"type": "PHAK_Controls", "query": "Explain the purpose of trim tabs according to the Flight Controls chapter in the PHAK."},
            # {"type": "PHAK_Controls", "query": "What are flaps and how do they affect flight, based on PHAK Chapter 6?"},

            # # For phak_Chapter_14_Airport_Operations.pdf (Note: PHAK Table of Contents shows Chapter 13 as Airport Operations, Chapter 14 is Airspace)
            # # Please verify chapter content. Assuming your file "phak_Chapter_14_Airport_Operations.pdf" actually contains Airport Operations.
            # {"type": "PHAK_AirportOps", "query": "What do runway threshold markings indicate, according to PHAK Chapter 14 on Airport Operations?"},
            # {"type": "PHAK_AirportOps", "query": "Explain the meaning of different airport signs as described in the Airport Operations chapter of PHAK."},
            # {"type": "PHAK_AirportOps", "query": "How is a standard traffic pattern flown at a non-towered airport, based on PHAK Airport Operations?"},

            # # For phak_Chapter_16_Navigation.pdf (Note: PHAK Table of Contents shows Chapter 15 as Navigation, Chapter 16 is Aeromedical Factors)
            # # Please verify chapter content. Assuming your file "phak_Chapter_16_Navigation.pdf" actually contains Navigation.
            {"type": "PHAK_Navigation", "query": "What is dead reckoning in air navigation, as explained in PHAK Chapter 16?"},
            # {"type": "PHAK_Navigation", "query": "How does a VOR system work for navigation, based on the PHAK Navigation chapter?"},
            # {"type": "PHAK_Navigation", "query": "Explain the difference between true course and magnetic course in navigation from PHAK Chapter 16."},

            # # --- RAG - Safety Recommendations Test (will also use the knowledge base if relevant) ---
            # {"type": "Safety_Rec", "query": "What are recommended procedures for encountering unexpected icing conditions?"},
            {"type": "Safety_Rec", "query": "What should a pilot do if they suspect carbon monoxide poisoning?"}, # This might be in Aeromedical factors, which you don't have listed, so good to see "not found".

            # # --- Test for a query where info might NOT be in your specific PHAK chapters ---
            # {"type": "PHAK_Knowledge_NotFound", "query": "What are the detailed maintenance requirements for a turbine engine after a bird strike?"} # Likely not in these general PHAK chapters
        ]

        for i, test_case in enumerate(test_queries):
            print(f"\n--- Test Case {i+1}: {test_case['type']} ---")
            query = test_case["query"]
            print(f"Q: {query}")
            response = assistant.process_query(query) # streaming=False for simpler console logging
            print(f"A: {response.get('message')}")
            print(f"Source: {response.get('source')}")
            if response.get('error'):
                print(f"Error: {response.get('error')}")
            print("-" * 30)

    except Exception as main_e:
        print(f"\n!!!!!!!!!! ERROR IN MAIN EXECUTION !!!!!!!!!!!")
        print(f"Error: {main_e}")
        traceback.print_exc()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")