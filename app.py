"""
HotSwappable - Main Application

A streamlined system for detecting and anonymizing personally identifiable information (PII)
in text documents using various NER models.
"""
import logging
import os
from pathlib import Path

import streamlit as st

from config.config import APP_DESCRIPTION, APP_TITLE, DEFAULT_DEMO_TEXT_PATH
from core.detector import analyze_text, filter_overlapping_entities
from ui.evaluation_panel import render_evaluation_panel
from ui.input_panel import render_input_panel
from ui.results_panel import render_results_panel
from ui.sidebar import render_sidebar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pii-detection")

# Set page config
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    """Main application entry point."""
    # Application title
    st.title(APP_TITLE)
    st.caption(APP_DESCRIPTION)
    
    # Initialize session state if needed
    if "detection_results" not in st.session_state:
        st.session_state["detection_results"] = []
    if "show_results" not in st.session_state:
        st.session_state["show_results"] = False
    if "loading" not in st.session_state:
        st.session_state["loading"] = False
    
    # Render sidebar and get settings
    settings = render_sidebar()
    
    # Render tabs for main content
    tab_titles = ["PII Detection", "Model Evaluation"]
    tabs = st.tabs(tab_titles)
    
    # PII Detection tab
    with tabs[0]:
        # Render input panel
        text, input_state = render_input_panel(DEFAULT_DEMO_TEXT_PATH)
        
        # Process detection if button clicked or settings changed
        if input_state.get("process_clicked", False):
            with st.spinner("Analyzing text..."):
                # Get settings from sidebar
                model_family = settings["model_family"]
                model_path = settings["model_path"]
                selected_entities = settings["selected_entities"]
                threshold = settings["threshold"]
                exclude_overlaps = settings["exclude_overlaps"]
                overlap_tolerance = settings["overlap_tolerance"]
                return_decision_process = settings["return_decision_process"]
                allow_list = settings["allow_list"]
                deny_list = settings["deny_list"]
                regex_pattern = settings["regex_pattern"]
                regex_entity_type = settings["regex_entity_type"]
                regex_score = settings["regex_score"]
                regex_context = settings["regex_context"]
                
                # Run detection
                results = analyze_text(
                    model_family=model_family,
                    model_path=model_path,
                    text=text,
                    entities=selected_entities,
                    language="en",
                    score_threshold=threshold,
                    return_decision_process=return_decision_process,
                    allow_list=allow_list,
                    deny_list=deny_list,
                    regex_pattern=regex_pattern,
                    regex_entity_type=regex_entity_type,
                    regex_score=regex_score,
                    regex_context=regex_context,
                )
                
                # Filter overlapping entities if needed
                if exclude_overlaps:
                    results = filter_overlapping_entities(
                        results=results,
                        exclude_overlaps=exclude_overlaps,
                        overlap_tolerance=overlap_tolerance,
                    )
                
                # Store results in session state
                st.session_state["detection_results"] = results
                st.session_state["show_results"] = True
                st.session_state["loading"] = False
                
                # Log results
                logger.info(f"Detected {len(results)} PII entities")
        
        # Render results panel
        render_results_panel(
            text=text,
            detection_results=st.session_state["detection_results"],
            settings=settings,
            show_results=st.session_state["show_results"],
        )
    
    # Model Evaluation tab
    with tabs[1]:
        render_evaluation_panel(settings)
    
    # Footer
    st.markdown("---")
    st.caption(
        """
        HotSwapPII | Using Microsoft Presidio | 
        [GitHub](https://github.com/microsoft/presidio) | 
        [Documentation](https://microsoft.github.io/presidio/)
        """
    )

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Create demo text file if it doesn't exist
    demo_text_path = Path(DEFAULT_DEMO_TEXT_PATH)
    if not demo_text_path.exists():
        demo_text = """Hello, my name is John Smith. I live in New York City and work at Acme Corporation.
My email is john.smith@example.com and my phone number is (555) 123-4567.
My credit card number is 4111-1111-1111-1111 with expiration date 05/25.
My social security number is 123-45-6789 and my passport number is X12345678.
Please contact me at my home address: 123 Main Street, Apt 4B, New York, NY 10001."""
        
        demo_text_path.parent.mkdir(exist_ok=True)
        with open(demo_text_path, "w", encoding="utf-8") as f:
            f.write(demo_text)
    
    # Run the application
    main()
