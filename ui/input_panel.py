"""
Input panel UI components for the HotSwapPII.
"""
import logging
from typing import Dict, Tuple

import streamlit as st

from utils.data_processing import load_demo_text

logger = logging.getLogger(__name__)

def render_input_panel(demo_text_path: str) -> Tuple[str, Dict]:
    """
    Render the input panel for text input and analysis.
    
    Args:
        demo_text_path: Path to the demo text file
        
    Returns:
        Tuple of (input text, input panel state)
    """
    st.subheader("Input")

    # Additional input information
    with st.expander("About PII Detection", expanded=False):
        st.info(
            """
            This system identifies personally identifiable information (PII) in text. 
            PII includes names, addresses, phone numbers, emails, IDs, and other 
            sensitive information.
            
            **How it works:**
            1. Enter or upload text in the input panel.
            2. Configure model and detection settings in the sidebar.
            3. View the detected PII and anonymized output below.
            """
        )
    
    # Load demo text
    demo_text = load_demo_text(demo_text_path)
    
    # Tab options for input
    tab_titles = ["Text Input", "File Upload", "Examples"]
    tabs = st.tabs(tab_titles)
    
    # Input state
    input_state = {"active_tab": 0, "input_source": "text"}
    
    # Text Input Tab
    with tabs[0]:
        text = st.text_area(
            label="Enter text to analyze",
            value=demo_text,
            height=250,
            key="text_input",
        )
        input_state["active_tab"] = 0
        input_state["input_source"] = "text"
    
    # File Upload Tab
    with tabs[1]:
        uploaded_file = st.file_uploader(
            "Upload a text file",
            type=["txt", "md", "html", "json", "csv"],
            help="Upload a text file to analyze for PII",
        )
        
        if uploaded_file is not None:
            # Read the file
            try:
                file_text = uploaded_file.getvalue().decode("utf-8")
                st.success(f"File uploaded: {uploaded_file.name}")
                
                # Display a preview of the file
                with st.expander("File Preview", expanded=False):
                    st.text(file_text[:1000] + ("..." if len(file_text) > 1000 else ""))
                
                # Use the file text as input
                text = file_text
                input_state["active_tab"] = 1
                input_state["input_source"] = "file"
                input_state["file_name"] = uploaded_file.name
            except Exception as e:
                st.error(f"Error reading file: {e}")
                text = demo_text
        else:
            # No file uploaded, use the default text
            text = demo_text if input_state["active_tab"] == 1 else text
    
    
    # Show a loading indicator if needed
    if st.session_state.get("loading", False):
        with st.spinner("Analyzing text..."):
            pass
    
    # Process button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        process_button = st.button(
            "Analyze Text",
            type="primary",
            use_container_width=True,
        
        )
        
        if process_button:
            st.session_state["loading"] = True
            input_state["process_clicked"] = True
        else:
            input_state["process_clicked"] = False
    
    return text, input_state
