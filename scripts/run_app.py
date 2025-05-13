#!/usr/bin/env python3
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Run the Streamlit app
if __name__ == "__main__":
    import streamlit.web.cli as stcli
    from aviation_assistant.ui.app import main
    
    sys.argv = ["streamlit", "run", "aviation_assistant/ui/app.py"]
    sys.exit(stcli.main())
