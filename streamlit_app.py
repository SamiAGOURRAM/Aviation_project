import streamlit as st
import time
import json
from pathlib import Path
import os
import logging # For assistant's internal logging
from typing import Tuple, List # For type hinting if using the method above


# Add the project root to sys.path to allow imports from aviation_assistant
import sys

# If streamlit_app.py is in the project root:
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
# print(f"Project root added to sys.path: {PROJECT_ROOT}") # For debugging
# print(f"Current sys.path: {sys.path}") # For debugging

# Now try to import the assistant
try:
    from aviation_assistant.agent.aviation_assistant_code import EnhancedAviationAssistant
    from aviation_assistant.agent.aviation_assistant_code import timedelta # Ensure timedelta is available
except ImportError as e:
    st.error(
        f"Failed to import EnhancedAviationAssistant. Error: {e}\n"
        f"PROJECT_ROOT was: {PROJECT_ROOT}\n"
        f"sys.path is: {sys.path}\n"
        "Please ensure streamlit_app.py is in the project root directory and the 'aviation_assistant' module is present there."
    )
    st.stop()

# --- Configuration ---
# Default paths assuming streamlit_app.py is at the project root
DEFAULT_DOC_PATH = str(PROJECT_ROOT / "aviation_assistant" / "data" / "aviation_documents")
DEFAULT_QDRANT_PATH_STREAMLIT_APP = str(PROJECT_ROOT / "streamlit_qdrant_datastore") # Separate Qdrant for app if desired, or use assistant's
DEFAULT_QDRANT_PATH_ASSISTANT = str(PROJECT_ROOT / "aviation_assistant" / "data" / "qdrant_datastore") # Assistant's default

# For local Redis, this is standard
LOCAL_REDIS_URL = "redis://localhost:6379"

APP_CONFIG = {
    # API Keys: Prioritize Streamlit secrets, then environment variables, then placeholders (which will require user input or error)
    "groq_api_key": st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE_IF_NO_SECRET")),
    "avwx_api_key": st.secrets.get("AVWX_API_KEY", os.environ.get("AVWX_API_KEY", "YOUR_AVWX_API_KEY_HERE_IF_NO_SECRET")),
    "windy_api_key": st.secrets.get("WINDY_API_KEY", os.environ.get("WINDY_API_KEY", "YOUR_WINDY_API_KEY_HERE_IF_NO_SECRET")),
    "opensky_username": st.secrets.get("OPENSKY_USERNAME", os.environ.get("OPENSKY_USERNAME", "YOUR_OPENSKY_USERNAME_IF_NO_SECRET")),
    "opensky_password": st.secrets.get("OPENSKY_PASSWORD", os.environ.get("OPENSKY_PASSWORD", "YOUR_OPENSKY_PASSWORD_IF_NO_SECRET")),

    # Document and Qdrant Paths for the Assistant
    # The assistant will use these paths. Ensure they are correct.
    "document_path": os.environ.get("DOC_PATH", DEFAULT_DOC_PATH), # Path for assistant to find documents
    "qdrant_path": os.environ.get("QDRANT_PATH", DEFAULT_QDRANT_PATH_ASSISTANT), # Path for assistant's Qdrant store
    "qdrant_collection_name": "aviation_docs_main_assistant", # Collection name for the assistant
    "force_reindex": False, # Set to True in assistant config if you want it to re-index on its init

    "site_url": "http://localhost:8501", # Streamlit's default URL
    "site_name": "SkyPilot AI Streamlit"
}
# Redis URL for the assistant
ASSISTANT_REDIS_URL = st.secrets.get("REDIS_URL", os.environ.get("REDIS_URL", LOCAL_REDIS_URL))


# --- Helper Functions ---

@st.cache_resource(ttl=3600) # Cache the assistant instance
def get_assistant():
    """Initializes and returns the EnhancedAviationAssistant instance."""
    # Ensure Qdrant path for the assistant exists (the assistant itself should also do this)
    Path(APP_CONFIG["qdrant_path"]).mkdir(parents=True, exist_ok=True)

    try:
        assistant_config = APP_CONFIG.copy()

        required_keys = ["groq_api_key", "avwx_api_key", "windy_api_key"] # Add others if strictly needed for init
        missing_secrets = []
        for key in required_keys:
            value = assistant_config.get(key, "")
            if "YOUR_" in str(value) or not value:
                missing_secrets.append(key)

        if missing_secrets:
            st.error(
                f"Missing API Key(s) for the assistant: {', '.join(missing_secrets)}. "
                f"Please configure them in `.streamlit/secrets.toml` or as environment variables."
            )
            st.stop()

        with st.spinner(" SKYNET IS BOOTING UP... Initializing Aviation Assistant... 🛫"):
            # Pass the specific Redis URL to the assistant
            assistant = EnhancedAviationAssistant(config=assistant_config, redis_url=ASSISTANT_REDIS_URL)
        st.success("Aviation Assistant is ready to help! 🧑‍✈️")
        return assistant
    except RuntimeError as e:
        st.error(f"Error initializing Aviation Assistant: {e}")
        st.exception(e) # Shows full traceback in Streamlit for debugging
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during assistant initialization: {e}")
        st.exception(e)
        st.stop()

def display_chat_message(message_obj, assistant_ref, message_idx: int): # Added message_idx
    """Displays a single chat message with sender, content, and feedback options."""
    role = message_obj["role"]
    content = message_obj["content"]
    avatar_map = {"user": "🧑‍✈️", "assistant": "✈️"}

    with st.chat_message(role, avatar=avatar_map.get(role)):
        st.markdown(content) # Use markdown for rich text

        if role == "assistant" and "metadata" in message_obj:
            metadata = message_obj["metadata"]
            response_id = metadata.get("response_id", f"unknown_resp_id_msg{message_idx}") # Fallback if response_id is missing
            source = metadata.get("source", "N/A")
            processing_time = metadata.get("processing_time", 0)
            api_error = metadata.get("error") # Error from the API call itself
            intermediate_steps = metadata.get("intermediate_steps") # List of (AgentAction, observation) OR List of Dicts

            # Display source, processing time, and any API error
            caption_parts = [f"Source: {source}"]
            if processing_time > 0:
                caption_parts.append(f"Time: {processing_time:.2f}s")
            # Use the potentially fallbacked response_id for display if original was None
            caption_parts.append(f"ID: {response_id}")
            st.caption(" | ".join(caption_parts))

            if api_error and source == "error":
                pass # Error message is already in 'content'
            elif api_error: # If there was an error within a tool but agent provided fallback
                st.warning(f"Tool Warning/Error: {api_error}")


            # Expander for "Agent Thoughts / Tool Usage"
            if intermediate_steps:
                with st.expander("🧠 Show Agent Reasoning & Tool Usage"):
                    for i, step_data in enumerate(intermediate_steps): # step_data can be a tuple or a dict
                        try:
                            action_tool = "Unknown Tool"
                            action_tool_input = "No Input"
                            action_log = "Log not available."
                            observation_content = "Observation not available."

                            # Check the type of step_data to handle both live and cached formats
                            if isinstance(step_data, tuple) and len(step_data) == 2:
                                action_obj, obs_obj = step_data
                                action_tool = getattr(action_obj, 'tool', action_tool)
                                action_tool_input = getattr(action_obj, 'tool_input', action_tool_input)
                                action_log = getattr(action_obj, 'log', action_log)
                                observation_content = str(obs_obj)
                            elif isinstance(step_data, dict):
                                action_tool = step_data.get('tool', action_tool)
                                action_tool_input = step_data.get('tool_input', action_tool_input)
                                action_log = step_data.get('log', action_log)
                                observation_content = step_data.get('observation', observation_content)
                            else:
                                st.error(f"Error displaying step {i+1}: Unknown step data format: {type(step_data)}")
                                st.json({"raw_step_data": str(step_data)})
                                continue

                            st.markdown(f"**Step {i+1}: Tool Call**")
                            
                            tool_input_display_str = ""
                            if isinstance(action_tool_input, dict):
                                try:
                                    tool_input_display_str = json.dumps(action_tool_input, indent=2)
                                except TypeError:
                                    tool_input_display_str = str(action_tool_input)
                            elif isinstance(action_tool_input, str):
                                tool_input_display_str = action_tool_input
                            else:
                                tool_input_display_str = str(action_tool_input)
                            
                            lang_for_code = "json" if isinstance(action_tool_input, dict) or \
                                                     (isinstance(action_tool_input, str) and \
                                                      (action_tool_input.strip().startswith(("{","[")) and \
                                                       action_tool_input.strip().endswith(("}","")))) \
                                                  else "text"
                            st.code(f"Tool: {action_tool}\nInput:\n{tool_input_display_str}", language=lang_for_code)
                            
                            st.markdown(f"**Agent Log (Thought Process):**\n```\n{action_log.strip()}\n```")
                            st.markdown(f"**Observation (Tool Output):**")
                            try:
                                obs_parsed = json.loads(observation_content)
                                st.json(obs_parsed)
                            except (json.JSONDecodeError, TypeError):
                                st.text(observation_content)

                        except Exception as e_step_display:
                            st.error(f"Error displaying detail for step {i+1}: {e_step_display}")
                            raw_data_str_on_error = "Could not parse step data for detailed display."
                            try: raw_data_str_on_error = str(step_data)
                            except: pass
                            st.json({"raw_step_data_on_error": raw_data_str_on_error})
                        st.markdown("---")
            
            # Feedback section
            # Ensure response_id is not None before creating keys for feedback
            if response_id and source != "error":
                # MODIFICATION: Incorporate message_idx into the key for guaranteed uniqueness per render cycle
                # Clean up the response_id for use in a key, then add message_idx
                clean_response_id_part = response_id.replace(":", "_").replace("-","_").replace(".","_")
                feedback_base_key = f"feedback_controls_{clean_response_id_part}_msg{message_idx}"
                
                # These keys are now unique for each message's feedback controls in the current render
                feedback_given_key = f"feedback_given_status_{feedback_base_key}"
                comment_key = f"comment_text_for_{feedback_base_key}"
                positive_button_key = f"positive_btn_{feedback_base_key}"
                negative_button_key = f"negative_btn_{feedback_base_key}"


                if not st.session_state.get(feedback_given_key, False):
                    current_comment_value = st.session_state.get(comment_key, "")
                    
                    cols_feedback = st.columns([1, 1, 3, 5])
                    with cols_feedback[0]:
                        if st.button("👍", key=positive_button_key, help="Good response!"):
                            comment_to_submit = st.session_state.get(comment_key, "")
                            # Pass the original response_id to the backend, not the modified widget key
                            assistant_ref.provide_feedback(response_id, True, comment_to_submit)
                            st.session_state[feedback_given_key] = True
                            st.toast("Thanks for your positive feedback! 😊", icon="👍")
                            st.rerun()
                    with cols_feedback[1]:
                        if st.button("👎", key=negative_button_key, help="Needs improvement."):
                            comment_to_submit = st.session_state.get(comment_key, "")
                            # Pass the original response_id to the backend
                            assistant_ref.provide_feedback(response_id, False, comment_to_submit)
                            st.session_state[feedback_given_key] = True
                            st.toast("Thanks! We'll use this to improve. 🛠️", icon="👎")
                            st.rerun()
                    
                    with cols_feedback[3]:
                         st.text_area(
                             "Optional comment:",
                             value=current_comment_value,
                             key=comment_key, # This key now includes message_idx via feedback_base_key
                             height=75,
                             help="Your comment will be submitted with your feedback.",
                             disabled=st.session_state.get(feedback_given_key, False)
                         )
                else:
                    st.caption("✔️ Feedback submitted.")
# --- Streamlit App ---
st.set_page_config(page_title="SkyPilot AI", page_icon="✈️", layout="wide")

# Initialize assistant (cached)
# This will run when the script first executes or if the cache expires/code changes
assistant_instance = get_assistant()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.title("SkyPilot AI")
    st.markdown("Your AI Co-Pilot for Pre-Flight Briefings & Aviation Knowledge.")

    if st.button("🔄 Clear Chat & Reset", use_container_width=True):
        st.session_state.messages = []
        if hasattr(assistant_instance, 'memory') and hasattr(assistant_instance.memory, 'clear'):
            assistant_instance.memory.clear() # Clear agent's conversational memory
        st.toast("Chat history and agent memory cleared!", icon="🗑️")
        # Clear any feedback states
        for key in list(st.session_state.keys()):
            if key.startswith("feedback_given_") or key.startswith("comment_text_"):
                del st.session_state[key]
        st.rerun()
        st.markdown("---") # Separator

    # New Clear Cache Button
    if st.button("🗑️ Clear Assistant Cache", use_container_width=True, type="secondary",
                  help="Clears all cached query responses from Redis. Useful for testing or forcing fresh data."):
        if 'assistant_instance' in locals() and hasattr(assistant_instance, 'clear_all_query_caches'):
            with st.spinner("Clearing cache..."):
                try:
                    num_deleted, deleted_sample = assistant_instance.clear_all_query_caches()
                    if num_deleted > 0:
                        st.success(f"Successfully cleared {num_deleted} cached items!")
                        if deleted_sample:
                            with st.expander("See sample of cleared keys"):
                                for key_name in deleted_sample:
                                    st.text(key_name)
                    else:
                        st.info("No cached items found to clear, or an error occurred (check logs).")
                except Exception as e:
                    st.error(f"Error clearing cache: {e}")
            st.rerun() # Rerun to reflect any changes if needed
        else:
            st.warning("Assistant instance not available or doesn't support cache clearing method.")


    st.markdown("---") # Separator before example prompts or other content

    st.markdown("---")
    st.subheader("💡 Example Prompts:")
    example_prompts = [
        "Weather at KJFK?",
        "Route weather KLAX to KORD",
        "Radar map for EDDF",
        "Hazardous attitudes in ADM?",
        "Primary flight controls?"
    ]
    for prompt_text in example_prompts:
        st.text(prompt_text) # st.text makes it selectable
    st.markdown("---")

    st.info(f"""
    **Assistant Data Sources:**
    - Real-time Weather, Traffic
    - Aviation Docs (PHAK, etc.)

    **Assistant Document Path:**
    `{Path(APP_CONFIG['document_path']).name}`
    **Assistant Qdrant DB:**
    `{Path(APP_CONFIG['qdrant_path']).name}/{APP_CONFIG['qdrant_collection_name']}`
    """)

    if st.button("FORCE RE-INDEX DOCS (Slow)", use_container_width=True, type="secondary",
                  help="Tell the assistant to re-process all documents in its configured document_path. This can take a very long time."):
        with st.spinner("Sending re-index command to assistant... This might take a while. Monitor assistant logs."):
            try:
                # The assistant's config should determine if it actually re-indexes.
                # We are calling its index_documents method.
                # To truly force it, the assistant's internal 'force_reindex' config flag
                # would need to be true, or its Qdrant init should have recreate_index=True.
                # This button basically just triggers the assistant's indexing logic again.
                original_force_reindex_config = assistant_instance.config.get("force_reindex", False)
                assistant_instance.config["force_reindex"] = True # Temporarily set for this call

                st.write(f"Calling assistant's index_documents for path: {assistant_instance.config['document_path']}")
                assistant_instance.index_documents(assistant_instance.config['document_path'])

                assistant_instance.config["force_reindex"] = original_force_reindex_config # Reset
                st.success("Document re-indexing process invoked in the assistant. Check assistant's console/logs for detailed status.")
            except Exception as e_reindex:
                st.error(f"Error during re-indexing command: {e_reindex}")
        st.rerun()

    st.markdown("---")
    st.caption("Built with LangChain, Haystack & Streamlit.")


# --- Main Chat Interface ---
st.header("✈️ Enhanced Aviation Assistant", anchor=False)
st.markdown("Ask me anything about aviation weather, traffic, regulations, or procedures!")

# Display chat messages from history
for idx, msg  in enumerate(st.session_state.messages):
    display_chat_message(msg, assistant_instance, idx)

# Accept user input
if user_query := st.chat_input("How can I assist your flight planning today?"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="🧑‍✈️"): # Display user message immediately
        st.markdown(user_query)

    # Get assistant response
    with st.chat_message("assistant", avatar="✈️"):
        message_placeholder = st.empty() # For streaming-like effect or final message
        with st.spinner("Thinking..."):
            try:
                response_payload = assistant_instance.process_query(user_query)

                assistant_response_content = response_payload.get("message", "Sorry, I couldn't process that.")
                metadata_for_display = {
                    "response_id": response_payload.get("response_id"),
                    "source": response_payload.get("source", "agent"),
                    "processing_time": response_payload.get("processing_time", 0),
                    "error": response_payload.get("error"), # API error or None
                    "intermediate_steps": response_payload.get("intermediate_steps", [])
                }

                message_placeholder.markdown(assistant_response_content) # Display final content

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_response_content,
                    "metadata": metadata_for_display
                })
                st.rerun()


            except Exception as e_process:
                st.error(f"Critical error processing your query: {e_process}")
                st.exception(e_process) # For debugging
                error_response_id = f"error_{int(time.time())}"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"An unexpected application error occurred: {e_process}",
                    "metadata": {"response_id": error_response_id, "source": "app_error"}
                })
                st.rerun()