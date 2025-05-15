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



class EnhancedAviationAssistant:
    """
    Enhanced Aviation Assistant that leverages LangChain for orchestration,
    Haystack 2.x for RAG, and Redis for caching. This implementation uses
    Llama 3.1 from OpenRouter for both agent and RAG components.
    """

    def __init__(self, config: Dict[str, Any], redis_url: str = "redis://localhost:6379"):
        """
        Initialize the enhanced aviation assistant.
        
        Args:
            config: Configuration with API keys and settings
            redis_url: URL for Redis connection
        """
        self.logger = logging.getLogger(__name__)
        self.config = config

        # Initialize Redis client
        self.redis_client = redis.from_url(redis_url)
        self.logger.info("Initialized Redis client connection")

        # Set up OpenRouter headers first
        self.openrouter_headers = {
            "HTTP-Referer": self.config.get("site_url", "http://localhost"),
            "X-Title": self.config.get("site_name", "Aviation Assistant")
        }

        # Initialize integration service and connectors
        self._init_connectors()
        
        # Set up LLM for agent
        self._init_llm()
        
        # Set up document store and RAG pipeline
        self._init_rag_pipeline()
        
        # Create tools and agent
        self._init_tools_and_agent()
        
        self.logger.info("Enhanced Aviation Assistant initialized successfully")

    def _init_connectors(self):
        """Initialize aviation data connectors and integration service"""
        try:
            self.avwx = AVWXConnector(api_key=self.config.get("avwx_api_key"))
            self.windy = WindyConnector(api_key=self.config.get("windy_api_key"))
            self.opensky = OpenSkyConnector(
                username=self.config.get("opensky_username"),
                password=self.config.get("opensky_password")
            )

            self.integration_service = IntegrationService(
                self.avwx, self.windy, self.opensky
            )
            self.logger.info("Aviation data connectors initialized")
        except Exception as e:
            self.logger.error(f"Error initializing connectors: {str(e)}")
            raise RuntimeError(f"Failed to initialize aviation data connectors: {str(e)}")

    def _init_llm(self):
        """Initialize the LLM for the agent using Llama from OpenRouter"""

        agent_llm_model_name = "meta-llama/llama-4-scout-17b-16e-instruct" 
            
        self.logger.info(f"Initializing agent LLM with model: {agent_llm_model_name} via Groq.")
        try:
            
            # Initialize LLM for agent
            self.llm = ChatOpenAI(
                model=agent_llm_model_name, 
                openai_api_key=self.config.get("groq_api_key"), # Use your Groq API key
                openai_api_base="https://api.groq.com/openai/v1", # Groq's OpenAI-compatible endpoint
                temperature=0.2, # Adjust as needed
                streaming=True,  # If you want streaming responses from the agent
                # default_headers=openrouter_default_headers, # Removed for Groq unless they specify needed headers
                callbacks=[StreamingStdOutCallbackHandler()] # For streaming to console
            )
            
            # Set up Redis cache for LLM
            # Ensure self.redis_client is a connected redis.Redis instance
            if self.redis_client:
                try:
                    # Correctly instantiate RedisCache by passing the redis.Redis client instance
                    # The constructor expects the client as the first argument (often named redis_ internally)
                    llm_cache = RedisCache(self.redis_client)
                    
                    # Correctly set the global LLM cache
                    set_llm_cache(llm_cache) # Use the imported set_llm_cache function
                    
                    self.logger.info("LLM initialized with OpenRouter Llama model and Redis cache enabled.")
                except Exception as cache_e:
                    self.logger.error(f"Failed to initialize RedisCache for LLM: {str(cache_e)}")
                    self.logger.warning("LLM caching will be disabled or use default in-memory cache.")
                    # Optionally, explicitly set to InMemoryCache or None
                    # from langchain.cache import InMemoryCache
                    # set_llm_cache(InMemoryCache())
            else:
                self.logger.warning("Redis client not available. LLM caching will be disabled.")
            
        except Exception as e:
            self.logger.error(f"Error initializing LLM: {str(e)}")
            traceback.print_exc() # Add traceback for more detailed error
            raise RuntimeError(f"Failed to initialize LLM: {str(e)}")

    def _init_rag_pipeline(self):
        """Initialize Haystack document store and RAG pipeline with re-ranking"""
        try:
            document_path = self.config.get("document_path", "./data/aviation_documents")
            self.logger.info(f"Setting up RAG pipeline with documents from: {document_path}")
            
            # 1. Initialize Document Store
            self.document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
            self.logger.info(f"Initialized document store: {type(self.document_store).__name__}")
            
            # 2. Initialize RAG Pipeline Components
            self.text_embedder = SentenceTransformersTextEmbedder(
                model="sentence-transformers/all-mpnet-base-v2"
            )
            self.text_embedder.warm_up()
            
            self.retriever = InMemoryEmbeddingRetriever(document_store=self.document_store, top_k=10)
            
            self.ranker = SentenceTransformersDiversityRanker(
                model="sentence-transformers/all-MiniLM-L6-v2",
                similarity="cosine",
                top_k=5
            )
            self.ranker.warm_up()
            
            rag_prompt_template = """
            You are an aviation expert assistant providing accurate information to pilots.
            Use the following context from aviation manuals and regulations to answer the question.
            Be concise, accurate, and prioritize safety in your responses.
            If the information in the context is insufficient, say so clearly.
            
            Context:
            {% for doc in documents %}
                Document: {{ doc.meta.get('file_name', 'Unknown Source') }}
                Content: {{ doc.content }}
                ---
            {% endfor %}
            
            Question: {{ query }}
            Answer:
            """
            self.prompt_builder = PromptBuilder(
                template=rag_prompt_template,
                required_variables=["documents", "query"] 
            )

            rag_llm_model_name_on_groq = "meta-llama/llama-4-scout-17b-16e-instruct"
            
            self.rag_llm_generator = OpenAIChatGenerator(
                api_key=Secret.from_token(self.config.get("groq_api_key")),
                model=rag_llm_model_name_on_groq,
                api_base_url="https://api.groq.com/openai/v1",
                generation_kwargs={"max_tokens": 2000, "temperature": 0.1}
            )
            
            # 3. Create RAG Pipeline
            self.rag_pipeline = Pipeline()
            self.rag_pipeline.add_component("text_embedder", self.text_embedder)
            self.rag_pipeline.add_component("retriever", self.retriever)
            self.rag_pipeline.add_component("ranker", self.ranker)
            # Add prompt_builder, its output will be used manually
            self.rag_pipeline.add_component("prompt_builder", self.prompt_builder)
            # llm_generator is NOT connected here if its input is List[ChatMessage]
            # and prompt_builder outputs str. It will be called separately.
            
            # 4. Connect RAG Pipeline (up to prompt_builder)
            self.rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
            self.rag_pipeline.connect("retriever.documents", "ranker.documents")
            self.rag_pipeline.connect("ranker.documents", "prompt_builder.documents")
            
            # 5. Index documents
            self.index_documents(document_path) # Ensure this is called after store init
            
            self.logger.info("RAG pipeline (up to prompt_builder) initialized with re-ranking.")
        except Exception as e:
            self.logger.error(f"Error initializing RAG pipeline: {str(e)}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize RAG pipeline: {str(e)}")

    def _init_tools_and_agent(self):
        """Initialize LangChain tools and agent with the newer 'tools' and 'tool_choice' paradigm"""
        try:
            # 1. Create LangChain Tools (your existing tool definitions are fine)
            self.tools = [
                Tool(
                    name="get_airport_weather",
                    func=self._get_airport_weather,
                    description="Get comprehensive weather information for an airport. Input should be a valid ICAO airport code (e.g., KJFK, EGLL)."
                ),
                Tool(
                    name="get_route_weather",
                    func=self._get_route_weather,
                    description="Get weather information for a flight route between two airports. Input should be two ICAO codes separated by comma (e.g., 'KJFK,KLAX')."
                ),
                Tool(
                    name="get_flight_level_weather",
                    func=self._get_flight_level_weather,
                    description="Get weather at a specific flight level and location. Input should be latitude,longitude,flight_level (e.g., '40.7,-74.0,FL350')."
                ),
                Tool(
                    name="get_airport_traffic",
                    func=self._get_airport_traffic,
                    description="Get traffic information for an airport. Input should be an ICAO code and optional hours (e.g., 'KJFK,2' for 2 hours of data)."
                ),
                Tool(
                    name="query_aviation_regulations",
                    func=self._query_aviation_regulations,
                    description="Query aviation regulations, procedures, and recommendations. Input should be a specific question about aviation procedures, regulations, or safety."
                ),
                Tool(
                    name="get_weather_recommendations",
                    func=self._get_weather_recommendations,
                    description="Get specific recommendations for handling particular weather conditions. Input should describe the weather condition (e.g., 'low visibility', 'strong winds')."
                ),
                Tool(
                    name="get_visual_weather_map",
                    func=self._get_visual_weather_map,
                    description="Generate a visual weather map URL for a specified airport and map type. Input should be ICAO code and map type separated by comma (e.g., 'KJFK,radar' or 'EGLL,satellite')."
                )
            ]
            
            # 2. Convert LangChain Tools to OpenAI 'tools' format
            # This replaces the deprecated format_tool_to_openai_function
            openai_tools = [convert_to_openai_tool(t) for t in self.tools]
            
            # 3. Set up Conversation Memory
            self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            
            system_message_content = """You are an Aviation Pre-Flight Briefing & Planning Agent. Your primary purpose is to assist pilots by accurately answering their questions using your available tools. Your responses should be based *primarily* on the information retrieved by these tools when they are relevant.

You have access to the following tools:
- get_airport_weather: For current/forecast weather at a specific airport. (Input: ICAO code)
- get_route_weather: For weather along a flight route. (Input: ICAO1,ICAO2)
- get_flight_level_weather: For weather at a specific altitude/location. (Input: lat,lon,FLXXX)
- get_airport_traffic: For airport arrival/departure info. (Input: ICAO,optional_hours)
- query_aviation_regulations: Use for ANY question about aviation rules, specific regulations (like VFR/IFR/LIFR requirements), or official procedures from aviation documents. (Input: specific question)
- get_weather_recommendations: Use for ANY query asking for advice or what to do about specific weather conditions based on aviation documents. (Input: weather condition description)
- get_visual_weather_map: To generate links to weather maps. (Input: ICAO,map_type)

Decision Process & Tool Usage Strategy:
1. Analyze the user's query, including conversation history for context if it's a follow-up.
2. Determine if the query directly asks for information one of your tools is designed to provide.
3. If a tool is clearly appropriate, use it ONCE to get the information. Formulate the best possible input for the tool based on the user's query and conversation context.
4. Evaluate the tool's output:
    a. If the tool provides a direct and comprehensive answer, use that information to formulate your response to the user.
    b. If the tool indicates the information was not found in its documents BUT provides relevant general knowledge or points to official sources (e.g., "The context does not specify... however, general knowledge suggests... consult FAR X"), present this full output to the user. Do NOT immediately try to re-query the same tool with slight variations unless the tool's response explicitly suggests a more specific query would help and you have new information to form that query.
    c. If the tool returns a clear error message (e.g., "Error retrieving..."), state that you encountered an issue with the tool.

Presenting Information:
- When a tool returns useful information (even if it's to state the documents don't contain the specifics but general knowledge is X), your main goal is to present THAT information clearly and comprehensively.
- Avoid redundant tool calls for the same underlying user intent within a single turn. Trust the tool's first good attempt.
- Your final answer should be a synthesis of the most relevant tool output.

Fallback to General Knowledge:
- Only use your general knowledge if:
    (a) No tool is appropriate for the query.
    (b) A relevant tool was tried and explicitly stated it could not find the information in its documents and provided no other useful general information or pointers.
    (c) A tool call resulted in an unrecoverable error.
- When using general knowledge, clearly state that this information is not from your specialized tools/documents if there's a chance of confusion.

Prioritize safety. If unsure after one or two well-chosen tool attempts, or if your general knowledge is not definitive, clearly state the limitations and recommend consulting official sources.
Do not get stuck in a loop of calling the same tool multiple times with minor variations if the initial results are not satisfactory but indicate the limits of the available documents.
"""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message_content),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad") 
            ])
            
            llm_with_tools = self.llm.bind_tools(tools=openai_tools)
            
            # 6. Construct the Agent Runnable using LCEL (LangChain Expression Language)
            # This defines the flow: input prep -> prompt -> LLM (with tools) -> output parser
            agent_runnable = (
                RunnablePassthrough.assign( # Assigns new keys to the input dict for the next step
                    # agent_scratchpad will hold the history of tool calls and responses
                    agent_scratchpad=lambda x: format_to_openai_tool_messages(x["intermediate_steps"])
                )
                | prompt
                | llm_with_tools # Use the LLM instance that has tools bound to it
                | OpenAIToolsAgentOutputParser() # Use the new output parser for tools
            )
            
            # 7. Create the AgentExecutor
            # The AgentExecutor orchestrates the agent's interaction with tools and memory.
            self.agent_chain = AgentExecutor(
                agent=agent_runnable,     # The core agent logic defined above
                tools=self.tools,         # The original list of LangChain Tool objects
                memory=self.memory,       # The conversation memory
                verbose=True,             # For detailed logging of agent steps
                handle_parsing_errors=True, # Helps in debugging parsing issues
                max_iterations=5          # Prevents overly long loops
            )
            
            self.logger.info("Tools and OpenAI tools-based agent initialized successfully.")
        except Exception as e:
            self.logger.error(f"Error initializing tools and agent: {str(e)}")
            traceback.print_exc() # Prints the full traceback for debugging
            raise RuntimeError(f"Failed to initialize tools and agent: {str(e)}")
        
    
    def index_documents(self, document_path: str):
        """
        Process and index documents for RAG.
        
        Args:
            document_path: Path to document directory
        """
        if not os.path.exists(document_path):
            self.logger.warning(f"Document path {document_path} does not exist. Skipping indexing.")
            return

        try:
            text_file_converter = TextFileToDocument()
            general_doc_splitter = DocumentSplitter(
                split_by="word",
                split_length=300,
                split_overlap=50
            )
            regulation_doc_splitter = DocumentSplitter(
                split_by="sentence",
                split_length=15,
                split_overlap=5
            )
            doc_embedder = SentenceTransformersDocumentEmbedder(
                model="sentence-transformers/all-mpnet-base-v2",
                progress_bar=True, # Good for seeing progress
                batch_size=32
            )
            doc_embedder.warm_up()

            # Initialize DocumentWriter with your document store instance
            document_writer = DocumentWriter(document_store=self.document_store) # <--- INITIALIZE DocumentWriter

            files_to_process = []
            for root, _, files in os.walk(document_path):
                for file in files:
                    file_path = Path(os.path.join(root, file))
                    if file_path.suffix.lower() in (".txt", ".md"):
                        files_to_process.append(file_path)
                    elif file_path.suffix.lower() == ".json":
                        # For JSON files, we are writing directly in _process_json_file
                        # after embedding. This is fine.
                        self._process_json_file(file_path, 
                                               general_doc_splitter, 
                                               regulation_doc_splitter, 
                                               doc_embedder) 
            
            if files_to_process:
                self.logger.info(f"Found {len(files_to_process)} text files to index via pipeline.")
                
                indexing_pipeline = Pipeline()
                indexing_pipeline.add_component("converter", text_file_converter)
                indexing_pipeline.add_component("splitter", general_doc_splitter)
                indexing_pipeline.add_component("embedder", doc_embedder)
                indexing_pipeline.add_component("writer", document_writer) # <--- USE DocumentWriter instance

                indexing_pipeline.connect("converter.documents", "splitter.documents")
                indexing_pipeline.connect("splitter.documents", "embedder.documents")
                indexing_pipeline.connect("embedder.documents", "writer.documents") # Output of embedder goes to writer
                
                indexing_pipeline.run({"converter": {"sources": files_to_process}})
                
            self.logger.info(f"Finished indexing. Total documents in store: {self.document_store.count_documents()}")
        except Exception as e:
            self.logger.error(f"Error during document indexing: {str(e)}")
            traceback.print_exc()
            raise
    def _process_json_file(self, file_path, general_splitter, regulation_splitter, embedder):
        """Process a JSON file for indexing with specialized handling based on content type"""
        try:
            self.logger.info(f"Processing JSON file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            json_docs = []
            
            # Determine content type to choose appropriate splitter
            is_regulation = "regulations" in str(file_path).lower() or "rules" in str(file_path).lower()
            splitter = regulation_splitter if is_regulation else general_splitter
            
            if isinstance(data, dict):
                if "recommendations" in data and isinstance(data["recommendations"], list):
                    for i, rec in enumerate(data["recommendations"]):
                        json_docs.append(HaystackDocument(
                            content=json.dumps(rec, indent=2),
                            meta={
                                "file_path": str(file_path), 
                                "file_name": file_path.name, 
                                "type": "recommendation", 
                                "item_index": i
                            }
                        ))
                else:
                    json_docs.append(HaystackDocument(
                        content=json.dumps(data, indent=2),
                        meta={"file_path": str(file_path), "file_name": file_path.name}
                    ))
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    json_docs.append(HaystackDocument(
                        content=json.dumps(item, indent=2),
                        meta={
                            "file_path": str(file_path), 
                            "file_name": file_path.name, 
                            "item_index": i
                        }
                    ))
            
            if json_docs:
                # Split documents
                split_json_docs = splitter.run(documents=json_docs)["documents"]
                
                # Embed documents
                embedded_json_docs = embedder.run(documents=split_json_docs)["documents"]
                
                # Write to document store
                self.document_store.write_documents(embedded_json_docs)
                
                self.logger.info(f"Indexed {len(embedded_json_docs)} chunks from JSON file {file_path.name}")
        except Exception as e:
            self.logger.error(f"Error processing JSON document {file_path}: {str(e)}")

    def process_query(self, query: str, streaming: bool = False) -> Dict[str, Any]:
        """
        Process a user query using the agent.
        
        Args:
            query: User query
            streaming: Whether to stream the response
            
        Returns:
            Response dictionary
        """
        start_time = time.time()
        query_hash = self._normalize_query(query)
        response_id = f"response:{int(start_time)}_{query_hash[:8]}"
        
        # Check cache first
        cached_response = self._check_cache(query)
        if cached_response:
            self.logger.info(f"Cache hit for query: {query}")
            return {
                "message": cached_response,
                "source": "cache",
                "response_id": response_id,
                "processing_time": 0
            }
        
        try:
            # Run the agent
            self.logger.info(f"DEBUG: Invoking agent_chain with input: '{query}'")
            self.logger.info(f"DEBUG: Current chat_history: {self.memory.chat_memory.messages}")
            response = self.agent_chain.invoke({
                "input": query,
                "chat_history": self.memory.chat_memory.messages
            })
            
            if isinstance(response, dict) and "output" in response:
                output_message = response["output"]
            else:
                output_message = str(response)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Cache the response with dynamic TTL
            self._cache_response(query, output_message, response_id)
            
            return {
                "message": output_message,
                "source": "agent",
                "response_id": response_id,
                "processing_time": processing_time
            }
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            traceback.print_exc()
            
            # Generate a fallback response
            fallback_response = self._create_fallback_response(e)
            
            return {
                "message": fallback_response,
                "error": str(e),
                "source": "error",
                "response_id": response_id,
                "processing_time": time.time() - start_time
            }
    
    def _create_fallback_response(self, error: Exception) -> str:
        """Create an appropriate fallback response based on error type"""
        if isinstance(error, ConnectionError) or isinstance(error, Timeout):
            return (
                "I'm sorry, I'm having trouble connecting to the aviation data services. "
                "This may be due to network issues or service availability. "
                "Please try again in a few moments."
            )
        elif "rate limit" in str(error).lower():
            return (
                "I apologize, but we've reached the rate limit for one of our data providers. "
                "This usually resolves within a minute. Please try again shortly."
            )
        else:
            return (
                f"I encountered an error while processing your request: {str(error)}. "
                "This could be due to API limitations or temporary unavailability. "
                "Please try again with a more specific question or try later."
            )

    def provide_feedback(self, response_id: str, is_positive: bool) -> None:
        """
        Store user feedback for a response and adjust cache TTL.
        
        Args:
            response_id: ID of the response
            is_positive: Whether feedback is positive
        """
        try:
            # Extract query hash from response_id
            parts = response_id.split("_")
            if len(parts) >= 2:
                query_hash_part = parts[1]
                
                # Store feedback
                feedback_key = f"feedback:{response_id}"
                self.redis_client.hset(feedback_key, mapping={
                    "feedback": 1 if is_positive else -1,
                    "timestamp": time.time()
                })
                
                # Find the cache key
                cache_pattern = f"cache:query:{query_hash_part}*"
                cache_keys = self.redis_client.keys(cache_pattern)
                
                # Update TTL for cached responses based on feedback
                for cache_key in cache_keys:
                    if is_positive:
                        # Extend TTL for positive feedback
                        self.redis_client.expire(cache_key, 86400 * 7)  # 7 days
                    else:
                        # Reduce TTL for negative feedback
                        self.redis_client.expire(cache_key, 300)  # 5 minutes
                
                self.logger.info(f"Recorded {'positive' if is_positive else 'negative'} feedback for {response_id}")
            else:
                self.logger.warning(f"Invalid response_id format: {response_id}")
        except Exception as e:
            self.logger.error(f"Error storing feedback: {str(e)}")

    def _check_cache(self, query: str) -> Optional[str]:
        """
        Check if a query response is cached.
        
        Args:
            query: User query
            
        Returns:
            Cached response or None
        """
        try:
            query_key_hash = self._normalize_query(query)
            cache_key = f"cache:query:{query_key_hash}"
            
            cached_content = self.redis_client.get(cache_key)
            if cached_content:
                # Check if has negative feedback
                feedback_keys = self.redis_client.keys(f"feedback:response:*_{query_key_hash[:8]}*")
                
                for key in feedback_keys:
                    feedback = self.redis_client.hget(key, "feedback")
                    if feedback and int(feedback) < 0:
                        # Found negative feedback, don't use cache
                        return None
                
                return cached_content.decode('utf-8')
            
            return None
        except Exception as e:
            self.logger.error(f"Error checking cache: {str(e)}")
            return None

    def _cache_response(self, query: str, response_content: str, response_id: str) -> None:
        """
        Cache a query response with dynamic TTL based on query type.
        
        Args:
            query: User query
            response_content: Response text
            response_id: Response ID
        """
        try:
            query_key_hash = self._normalize_query(query)
            cache_key = f"cache:query:{query_key_hash}"
            
            # Determine TTL based on query type
            ttl = self._determine_cache_ttl(query)
            
            # Store in Redis with determined TTL
            self.redis_client.setex(cache_key, ttl, response_content)
            
            # Store query metadata
            self.redis_client.hset(f"meta:response:{response_id}", mapping={
                "original_query": query,
                "timestamp": time.time(),
                "ttl": ttl
            })
            
            self.logger.info(f"Cached response for query with TTL {ttl}s: {query[:50]}...")
        except Exception as e:
            self.logger.error(f"Error caching response: {str(e)}")

    def _determine_cache_ttl(self, query: str) -> int:
        """
        Determine cache TTL based on query content.
        
        Args:
            query: User query
            
        Returns:
            TTL in seconds
        """
        query_lower = query.lower()
        
        # Short TTL for real-time data
        if any(term in query_lower for term in ["current", "now", "latest", "traffic", "today"]):
            return 300  # 5 minutes
        
        # Medium TTL for weather forecasts
        elif any(term in query_lower for term in ["weather", "forecast", "metar", "taf", "wind"]):
            return 1800  # 30 minutes
        
        # Long TTL for regulations and static information
        elif any(term in query_lower for term in ["regulation", "rule", "procedure", "guideline"]):
            return 86400 * 7  # 7 days
        
        # Default TTL
        return 3600  # 1 hour

    def _normalize_query(self, query: str) -> str:
        """
        Normalize a query for caching by removing whitespace and converting to lowercase.
        
        Args:
            query: Original query
            
        Returns:
            Normalized query hash
        """
        normalized = re.sub(r'\s+', ' ', query.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()

    def safe_api_call(self, func: Callable, *args, default_value: Any = None, **kwargs) -> Any:
        """
        Wrapper for API calls with robust error handling.
        
        Args:
            func: Function to call
            *args: Arguments to pass to the function
            default_value: Default value to return on error
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Function result or default value on error
        """
        try:
            return func(*args, **kwargs)
        except ConnectionError as e:
            self.logger.error(f"Connection error: {e}")
            return default_value or {"error": "Connection failed. Please try again later."}
        except Timeout as e:
            self.logger.error(f"Timeout error: {e}")
            return default_value or {"error": "Request timed out. Service may be busy."}
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return default_value or {"error": f"An unexpected error occurred: {str(e)}"}

    def _format_weather_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format weather data with clear sections and visual indicators.
        
        Args:
            data: Raw weather data
            
        Returns:
            Formatted weather data
        """
        return {
            "text": self._create_human_readable_summary(data),
            "hazards": self._extract_categorized_hazards(data),
            "visualization_links": self._generate_visualization_links(data),
            "safety_recommendations": self._generate_safety_recommendations(data)
        }

    def _create_human_readable_summary(self, data: Dict[str, Any]) -> str:
        """Create a human-readable summary from weather data"""
        if "summary" in data:
            return data["summary"]
        
        # Fallback if no summary exists
        summary_parts = []
        
        if "station" in data:
            station = data["station"]
            summary_parts.append(f"Airport: {station.get('name', 'Unknown')} ({station.get('icao', 'Unknown')})")
        
        if "metar_summary" in data:
            summary_parts.append(f"Current Conditions: {data['metar_summary']}")
        
        if "taf_summary" in data:
            summary_parts.append(f"Forecast: {data['taf_summary']}")
        
        return "\n\n".join(summary_parts)

    def _extract_categorized_hazards(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and categorize hazards from weather data"""
        hazards = data.get("hazardous_conditions", [])
        
        # Group hazards by severity
        categorized = {"high": [], "moderate": [], "low": []}
        
        for hazard in hazards:
            severity = hazard.get("severity", "").lower()
            if severity in categorized:
                categorized[severity].append(hazard)
            else:
                categorized["low"].append(hazard)
        
        # Flatten to a list with severity category
        result = []
        for severity, items in categorized.items():
            for item in items:
                result.append({
                    "description": item.get("description", ""),
                    "severity": severity,
                    "source": item.get("source", ""),
                    "category": self._categorize_hazard(item.get("description", ""))
                })
        
        return result

    def _categorize_hazard(self, description: str) -> str:
        """Categorize hazard based on description"""
        description_lower = description.lower()
        
        if any(term in description_lower for term in ["visibility", "fog", "haze", "obscuration"]):
            return "visibility"
        elif any(term in description_lower for term in ["wind", "gust", "crosswind"]):
            return "wind"
        elif any(term in description_lower for term in ["thunderstorm", "lightning", "cb", "convective"]):
            return "convective"
        elif any(term in description_lower for term in ["ice", "freez", "snow"]):
            return "icing"
        elif any(term in description_lower for term in ["turbulence", "shear", "chop"]):
            return "turbulence"
        else:
            return "other"

    def _generate_visualization_links(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Generate visualization links from weather data"""
        # Return existing links if available
        if "route_visualization" in data and isinstance(data["route_visualization"], dict):
            return data["route_visualization"]
        
        if "visualization_url" in data and isinstance(data["visualization_url"], str):
            return {"primary": data["visualization_url"]}
        
        # No visualization links in data
        return {}

    def _generate_safety_recommendations(self, data: Dict[str, Any]) -> List[str]:
        """Generate safety recommendations based on weather data"""
        recommendations = []
        hazards = data.get("hazardous_conditions", [])
        
        # Look for specific hazard types and generate recommendations
        for hazard in hazards:
            description = hazard.get("description", "").lower()
            severity = hazard.get("severity", "").lower()
            
            if "visibility" in description:
                if severity == "high":
                    recommendations.append("Consider delaying flight or filing IFR due to severely reduced visibility.")
                else:
                    recommendations.append("Be prepared for reduced visibility conditions.")
            
            elif "wind" in description or "gust" in description:
                if severity == "high":
                    recommendations.append("Exercise extreme caution during takeoff and landing due to strong winds.")
                else:
                    recommendations.append("Be prepared for potential crosswind landing conditions.")
            
            elif "thunderstorm" in description or "cb" in description:
                recommendations.append("Avoid thunderstorms by at least 20 nautical miles.")
            
            elif "ice" in description or "freez" in description:
                recommendations.append("Be alert for potential airframe icing conditions.")
        
        # Add general recommendations if specific ones weren't generated
        if not recommendations and hazards:
            recommendations.append("Exercise caution due to reported hazardous conditions.")
        
        return recommendations

    def _get_airport_weather(self, icao_code: str) -> str:
        """
        Get comprehensive weather information for an airport.
        
        Args:
            icao_code: ICAO airport code
            
        Returns:
            Formatted weather information
        """
        icao_code = icao_code.upper()
        cache_key = f"data:airport_weather:{icao_code}"
        
        # Check Redis cache first
        cached_data = self.redis_client.get(cache_key)
        if cached_data:
            return f"[Cached Data] {cached_data.decode('utf-8')}"
        
        try:
            # Get fresh data from API using safe_api_call
            airport_weather = self.safe_api_call(
                self.integration_service.get_airport_weather,
                icao_code,
                default_value={"error": f"Failed to retrieve weather for {icao_code}"}
            )
            
            if "error" in airport_weather:
                return f"Error: {airport_weather['error']}"
            
            # Format the response using the enhanced formatter
            formatted_data = self._format_weather_response(airport_weather)
            
            # Create a human-readable response
            response_parts = [formatted_data["text"]]
            
            # Add hazard information
            if formatted_data["hazards"]:
                hazard_text = "\n\nHazardous Conditions:"
                
                # Group hazards by severity
                hazards_by_severity = {}
                for hazard in formatted_data["hazards"]:
                    severity = hazard["severity"]
                    if severity not in hazards_by_severity:
                        hazards_by_severity[severity] = []
                    hazards_by_severity[severity].append(hazard)
                
                # Add hazards in order of severity
                for severity in ["high", "moderate", "low"]:
                    if severity in hazards_by_severity:
                        hazard_text += f"\n{severity.upper()} SEVERITY:"
                        for i, hazard in enumerate(hazards_by_severity[severity], 1):
                            hazard_text += f"\n{i}. {hazard['description']}"
                
                response_parts.append(hazard_text)
            
            # Add visualization links
            if formatted_data["visualization_links"]:
                viz_text = "\n\nVisualization Links:"
                for name, url in formatted_data["visualization_links"].items():
                    viz_text += f"\n- {name.replace('_', ' ').capitalize()}: {url}"
                response_parts.append(viz_text)
            
            # Add safety recommendations
            if formatted_data["safety_recommendations"]:
                rec_text = "\n\nSafety Recommendations:"
                for i, rec in enumerate(formatted_data["safety_recommendations"], 1):
                    rec_text += f"\n{i}. {rec}"
                response_parts.append(rec_text)
            
            response = "\n".join(response_parts)
            
            # Cache the response for 30 minutes
            self.redis_client.setex(cache_key, 1800, response)
            
            return response
        except Exception as e:
            self.logger.error(f"Error getting airport weather for {icao_code}: {str(e)}")
            return f"Error retrieving weather for {icao_code}: {str(e)}"

    def _get_route_weather(self, departure_destination: str) -> str:
        """
        Get weather information for a flight route.
        
        Args:
            departure_destination: Departure and destination ICAO codes
            
        Returns:
            Formatted route weather information
        """
        try:
            # Parse input
            parts = departure_destination.split(",")
            if len(parts) != 2:
                return "Please provide both departure and destination airports separated by a comma (e.g., 'KJFK,KLAX')."
            
            departure = parts[0].strip().upper()
            destination = parts[1].strip().upper()
            
            cache_key = f"data:route_weather:{departure}_{destination}"
            
            # Check Redis cache first
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return f"[Cached Data] {cached_data.decode('utf-8')}"
            
            # Get fresh data from API with safe_api_call
            route_weather = self.safe_api_call(
                self.integration_service.get_route_weather,
                departure, 
                destination,
                default_value={"error": f"Failed to retrieve route weather for {departure} to {destination}"}
            )
            
            if "error" in route_weather:
                return f"Error: {route_weather['error']}"
            
            # Format using the enhanced formatter
            formatted_data = self._format_weather_response(route_weather)
            
            # Create a detailed response
            response_parts = [f"Flight Route: {departure} to {destination}\n"]
            response_parts.append(formatted_data["text"])
            
            # Add hazard information
            if formatted_data["hazards"]:
                hazard_text = "\n\nHazardous Conditions Along Route:"
                for i, hazard in enumerate(formatted_data["hazards"], 1):
                    hazard_text += f"\n{i}. {hazard['description']} (Severity: {hazard['severity'].upper()})"
                response_parts.append(hazard_text)
            
            # Add visualization links
            if formatted_data["visualization_links"]:
                viz_text = "\n\nVisualization Links:"
                for name, url in formatted_data["visualization_links"].items():
                    viz_text += f"\n- {name.replace('_', ' ').capitalize()}: {url}"
                response_parts.append(viz_text)
            
            # Add safety recommendations
            if formatted_data["safety_recommendations"]:
                rec_text = "\n\nSafety Recommendations:"
                for i, rec in enumerate(formatted_data["safety_recommendations"], 1):
                    rec_text += f"\n{i}. {rec}"
                response_parts.append(rec_text)
            
            response = "\n".join(response_parts)
            
            # Cache the response for 30 minutes
            self.redis_client.setex(cache_key, 1800, response)
            
            return response
        except Exception as e:
            self.logger.error(f"Error getting route weather: {str(e)}")
            return f"Error retrieving route weather information: {str(e)}"

    def _get_flight_level_weather(self, location_fl: str) -> str:
        """
        Get weather at a specific flight level.
        
        Args:
            location_fl: Latitude, longitude, and flight level
            
        Returns:
            Formatted flight level weather information
        """
        try:
            # Parse input
            parts = location_fl.split(",")
            if len(parts) != 3:
                return "Please provide latitude, longitude, and flight level separated by commas (e.g., '40.7,-74.0,FL350')."
            
            try:
                latitude = float(parts[0].strip())
                longitude = float(parts[1].strip())
                flight_level = parts[2].strip().upper()
            except ValueError:
                return "Invalid coordinates. Please provide valid numerical latitude and longitude values."
            
            cache_key = f"data:fl_weather:{latitude}_{longitude}_{flight_level}"
            
            # Check Redis cache first
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return f"[Cached Data] {cached_data.decode('utf-8')}"
            
            # Get fresh data from API with safe_api_call
            fl_weather = self.safe_api_call(
                self.integration_service.get_flight_level_weather,
                latitude, 
                longitude, 
                flight_level,
                default_value={"error": f"Failed to retrieve weather at {flight_level} for coordinates {latitude}, {longitude}"}
            )
            
            if "error" in fl_weather:
                return f"Error: {fl_weather['error']}"
            
            # Format with enhanced formatter
            formatted_data = self._format_weather_response(fl_weather)
            
            # Create a detailed response
            response_parts = [f"Weather at {flight_level} ({latitude}, {longitude}):\n"]
            response_parts.append(formatted_data["text"])
            
            # Add visualization links
            if formatted_data["visualization_links"]:
                viz_text = "\n\nVisualization Links:"
                for name, url in formatted_data["visualization_links"].items():
                    viz_text += f"\n- {name.replace('_', ' ').capitalize()}: {url}"
                response_parts.append(viz_text)
            
            response = "\n".join(response_parts)
            
            # Cache the response for 30 minutes
            self.redis_client.setex(cache_key, 1800, response)
            
            return response
        except Exception as e:
            self.logger.error(f"Error getting flight level weather: {str(e)}")
            return f"Error retrieving flight level weather information: {str(e)}"

    def _get_airport_traffic(self, icao_hours: str) -> str:
        """
        Get airport traffic information.
        
        Args:
            icao_hours: ICAO code and optional hours
            
        Returns:
            Formatted traffic information
        """
        try:
            # Parse input
            parts = icao_hours.split(",")
            icao_code = parts[0].strip().upper()
            hours = 2  # Default
            
            if len(parts) > 1:
                try:
                    hours = int(parts[1].strip())
                except ValueError:
                    return "Invalid hours value. Please provide a valid number."
            
            cache_key = f"data:airport_traffic:{icao_code}_{hours}"
            
            # Check Redis cache first
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return f"[Cached Data] {cached_data.decode('utf-8')}"
            
            # Get fresh data from API with safe_api_call
            traffic = self.safe_api_call(
                self.integration_service.get_airport_traffic,
                icao_code, 
                hours,
                default_value={"error": f"Failed to retrieve traffic for {icao_code}"}
            )
            
            if "error" in traffic:
                return f"Error: {traffic['error']}"
            
            # Format the response
            if "summary" in traffic:
                traffic_summary = traffic["summary"]
            else:
                traffic_summary = f"Traffic at {icao_code} in the past {hours} hours:\n"
                arrivals = traffic.get("arrivals", [])
                departures = traffic.get("departures", [])
                
                traffic_summary += f"\nArrivals: {len(arrivals)}\n"
                for i, flight in enumerate(arrivals[:5], 1):  # Show first 5 arrivals
                    callsign = flight.get("callsign", "Unknown")
                    origin = flight.get("estDepartureAirport", "Unknown")
                    eta = flight.get("estimatedArrivalTime", "Unknown")
                    traffic_summary += f"{i}. {callsign} from {origin} (ETA: {eta})\n"
                
                traffic_summary += f"\nDepartures: {len(departures)}\n"
                for i, flight in enumerate(departures[:5], 1):  # Show first 5 departures
                    callsign = flight.get("callsign", "Unknown")
                    destination = flight.get("estArrivalAirport", "Unknown")
                    etd = flight.get("estimatedDepartureTime", "Unknown")
                    traffic_summary += f"{i}. {callsign} to {destination} (ETD: {etd})\n"
            
            # Cache the response for 15 minutes (traffic data changes frequently)
            self.redis_client.setex(cache_key, 900, traffic_summary)
            
            return traffic_summary
        except Exception as e:
            self.logger.error(f"Error getting airport traffic: {str(e)}")
            return f"Error retrieving airport traffic information: {str(e)}"

# Inside your EnhancedAviationAssistant class

    def _query_aviation_regulations(self, query: str) -> str:
        """
        Query aviation regulations and documentation using Haystack RAG.
        """
        cache_key = f"data:regulations:{self._normalize_query(query)}"
        # No need to check agent-level cache here as this is a tool method.
        # If this method itself caches, that's fine.
        # The agent process_query handles its own caching.

        try:
            self.logger.info(f"Querying aviation regulations RAG for: {query}")
            if not self.rag_pipeline or not self.rag_llm_generator:
                self.logger.error("RAG pipeline or RAG LLM generator not initialized.")
                return "Error: RAG system not available for regulations."
            
            pipeline_inputs = {
                "text_embedder": {"text": query},
                "ranker": {"query": query},
                "prompt_builder": {"query": query} 
            }
            
            pipeline_result = self.rag_pipeline.run(data=pipeline_inputs)

            final_prompt_string = None
            if "prompt_builder" in pipeline_result and "prompt" in pipeline_result["prompt_builder"]:
                final_prompt_string = pipeline_result["prompt_builder"]["prompt"]
            else:
                self.logger.error(f"Could not find 'prompt' in prompt_builder output. Result: {pipeline_result}")
                return "Error: Could not construct the prompt for the LLM from pipeline result."

            if not final_prompt_string:
                 self.logger.error("Generated prompt string from prompt_builder is empty.")
                 return "Error: Generated prompt string is empty."

            llm_response = self.rag_llm_generator.run(
                messages=[ChatMessage.from_user(final_prompt_string)]
            )
            
            if "replies" in llm_response and llm_response["replies"] and len(llm_response["replies"]) > 0:
                answer = llm_response["replies"][0].text # MODIFIED: .text instead of .content
                
                # It's good practice for the RAG tool to return the direct answer,
                # and let the agent decide how to phrase "I don't know" if the answer is empty or unhelpful.
                # However, your current logic of checking the answer content is also fine.
                if not answer or "don't have specific information" in answer.lower() or "couldn't find" in answer.lower():
                    # The RAG didn't find a good answer in the documents.
                    return ( # Return a clear message that the RAG tool couldn't find it.
                        f"My document search for '{query}' did not yield a specific answer. "
                        "You may need to consult official FAA publications or a flight instructor."
                    )
                
                # self.redis_client.setex(cache_key, 86400 * 7, answer) # Caching here is optional, agent also caches.
                return answer
            else:
                self.logger.warning(f"No 'replies' found in RAG LLM generator response for query: {query}")
                return (
                    "My document search did not return a specific answer (no LLM reply from RAG). "
                    "Consider checking official FAA publications."
                )
        except Exception as e:
            self.logger.error(f"Error querying aviation regulations with RAG: {str(e)}")
            traceback.print_exc()
            return f"Error during RAG query for regulations: {str(e)}"

    def _get_weather_recommendations(self, weather_condition: str) -> str:
        """
        Get recommendations for specific weather conditions using Haystack RAG.
        """
        cache_key = f"data:weather_rec:{self._normalize_query(weather_condition)}"
        # No agent-level cache check here.

        try:
            self.logger.info(f"Getting weather recommendations RAG for: {weather_condition}")
            if not self.rag_pipeline or not self.rag_llm_generator:
                self.logger.error("RAG pipeline or RAG LLM generator not initialized for weather recommendations.")
                return "Error: RAG system not available for weather recommendations."
            
            query = f"What are the recommended procedures, considerations, and safety precautions for pilots encountering {weather_condition}?"
            
            pipeline_inputs = {
                "text_embedder": {"text": query},
                "ranker": {"query": query},
                "prompt_builder": {"query": query}
            }
            
            pipeline_result = self.rag_pipeline.run(data=pipeline_inputs)

            final_prompt_string = None
            if "prompt_builder" in pipeline_result and "prompt" in pipeline_result["prompt_builder"]:
                final_prompt_string = pipeline_result["prompt_builder"]["prompt"]
            else:
                self.logger.error(f"Could not find 'prompt' in prompt_builder output for weather rec. Result: {pipeline_result}")
                return "Error: Could not construct the prompt for weather recommendations."

            if not final_prompt_string:
                 self.logger.error("Generated prompt string for weather rec from prompt_builder is empty.")
                 return "Error: Generated prompt string for weather recommendations is empty."

            llm_response = self.rag_llm_generator.run(
                messages=[ChatMessage.from_user(final_prompt_string)]
            )
            
            if "replies" in llm_response and llm_response["replies"] and len(llm_response["replies"]) > 0:
                answer = llm_response["replies"][0].text # MODIFIED: .text instead of .content
                # self.redis_client.setex(cache_key, 86400 * 7, answer) # Optional tool-level caching
                return answer
            else:
                self.logger.warning(f"No 'replies' found in RAG LLM generator response for weather_condition: {weather_condition}")
                # Let the RAG tool indicate it didn't find it, agent can phrase the final response.
                return (
                    f"My document search for recommendations on '{weather_condition}' did not return a specific answer from RAG. "
                    "Always prioritize safety and consult official manuals or instructors."
                )
        except Exception as e:
            self.logger.error(f"Error getting weather recommendations with RAG: {str(e)}")
            traceback.print_exc()
            return f"Error during RAG query for weather recommendations: {str(e)}"

    def _get_visual_weather_map(self, params: str) -> str:
        """
        Generate a visual weather map URL for a specified airport and map type.
        
        Args:
            params: ICAO code and map type
            
        Returns:
            URL for weather visualization
        """
        try:
            parts = params.split(",")
            if len(parts) < 1:
                return "Please provide an ICAO code and optional map type (e.g., 'KJFK,radar' or 'EGLL,satellite')."
            
            icao_code = parts[0].strip().upper()
            map_type = "radar"  # Default
            
            if len(parts) > 1:
                map_type = parts[1].strip().lower()
            
            cache_key = f"data:visual_map:{icao_code}_{map_type}"
            
            # Check cache first
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return cached_data.decode('utf-8')
            
            # Get airport coordinates
            airport_info = self.safe_api_call(
                self.integration_service.get_airport_info,
                icao_code,
                default_value={"error": f"Failed to retrieve information for {icao_code}"}
            )
            
            if "error" in airport_info:
                return f"Error: {airport_info['error']}"
            
            if "coordinates" not in airport_info:
                return f"Error: Could not find coordinates for {icao_code}."
            
            # Get visualization URL from Windy connector
            coords = airport_info["coordinates"]
            
            map_url = self.safe_api_call(
                self.windy.get_visualization_url,
                lat=coords.get("lat", 0),
                lon=coords.get("lon", 0),
                map_type=map_type,
                zoom=8,
                default_value={"error": f"Failed to generate visualization for {map_type} at {icao_code}"}
            )
            
            if isinstance(map_url, dict) and "error" in map_url:
                return f"Error: {map_url['error']}"
            
            response = (
                f"Visual weather map for {icao_code} ({airport_info.get('name', 'Unknown')}):\n\n"
                f"Map type: {map_type.capitalize()}\n"
                f"URL: {map_url}\n\n"
                f"This map provides a {map_type} visualization of current weather conditions at {icao_code}."
            )
            
            # Cache the response for 30 minutes
            self.redis_client.setex(cache_key, 1800, response)
            
            return response
        except Exception as e:
            self.logger.error(f"Error generating visual weather map: {str(e)}")
            return f"Error generating visual weather map: {str(e)}"
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

    # Create a configuration for testing
    # Ensure document path exists
    if not os.path.exists("./data/aviation_documents"):
        os.makedirs("./data/aviation_documents")
        with open("./data/aviation_documents/sample_rule.txt", "w") as f:
            f.write("Visual Flight Rules (VFR) require good visibility. Instrument Flight Rules (IFR) are used in low visibility.")
        with open("./data/aviation_documents/sample_recommendation.json", "w") as f:
            json.dump({
                "recommendations": [
                    {"condition": "icing", "severity": "high", "action": "Avoid flight into known icing conditions. Divert or descend if encountered."},
                    {"condition": "thunderstorm", "severity": "extreme", "action": "Avoid by at least 20 nautical miles."}
                ]
            }, f)

    config_example = {
        "groq_api_key" : "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",
        "avwx_api_key": "jLX2iQWBGYttAwYBxiFCUc_4ljrbNOqCNvHe5hjtS8o",
        "windy_api_key": "ymAvnkZvsbZpxgMVlHabWuDIG6ZgHsJd",
        "opensky_username": "sami___ag",
        "opensky_password": "716879Sami.",
        "openrouter_api_key": "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",
        "document_path": "./data/aviation_documents",
        "site_url": "http://aviationassistant.com",
        "site_name": "Aviation Assistant"
    }

    # Check if critical OpenRouter API key is set
    if config_example["openrouter_api_key"] == "YOUR_OPENROUTER_KEY":
        print("WARNING: OPENROUTER_API_KEY is not set. LLM functionalities will fail.")
        # exit() # Optionally exit if key is missing

        # --- TEMPORARY: Clear specific caches before running tests ---
    redis_url_for_clear = "redis://localhost:6379" # Your Redis URL
    queries_that_failed = [
        "What are VFR visibility requirements?",
        "What should I do in icing conditions?",
        "Tell me more about LIFR conditions.",
        "And what are the visibility requirements for that?"
    ]
    print("\n--- Clearing specific query caches ---")
    clear_specific_query_caches(redis_url_for_clear, queries_that_failed)
    print("--- Finished clearing caches ---\n")

    try:
        assistant = EnhancedAviationAssistant(config=config_example)

        print("\n--- Testing Weather Recommendations ---")
        rec_query = "Get KJFK to KLAX route weather"
        response = assistant.process_query(rec_query)
        print(f"Q: {rec_query}\nA: {response.get('message')}")

    except Exception as main_e:
        print(f"Error in example usage: {main_e}")
        traceback.print_exc()
