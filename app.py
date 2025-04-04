"""
HotSwappable - Main Application

A streamlined system for detecting and anonymizing personally identifiable information (PII)
in text documents using various NER models.
"""
import logging
import os
from pathlib import Path

import streamlit as st
from django.templatetags.i18n import language

from config.config import APP_DESCRIPTION, APP_TITLE, DEFAULT_DEMO_TEXT_PATH
from core.detector import (analyze_text, filter_overlapping_entities, 
                          initialize_analyzer_engine, process_with_custom_pipeline)
from models.model_factory import get_independent_model

from ui.evaluation_panel import render_evaluation_panel
from ui.benchmarks_panel import render_benchmarks_panel
from ui.model_comparison_panel import render_model_comparison_panel
from ui.custom_pipeline_panel import render_custom_pipeline_panel
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
    if "use_custom_pipeline" not in st.session_state:
        st.session_state["use_custom_pipeline"] = False
    
    # Render sidebar and get settings
    settings = render_sidebar()
    
    # Render tabs for main content
    tab_titles = ["PII Detection", "Model Evaluation", "Model Dataset Benchmarks", "Model Comparison", "Custom Pipeline"]
    tabs = st.tabs(tab_titles)
    
    # PII Detection tab
    with tabs[0]:
        # Add custom pipeline status indicator
        if st.session_state.get("use_custom_pipeline", False):
            st.markdown("""
                <div style="background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
                           padding: 0.5rem 1rem;
                           border-radius: 4px;
                           margin-bottom: 1rem;
                           display: flex;
                           align-items: center;
                           gap: 0.5rem;">
                    <span style="font-size: 1.2rem;">⚙️</span>
                    <span style="color: #4A4A4A; font-weight: 500;">Custom Pipeline Active</span>
                </div>
            """, unsafe_allow_html=True)
        
        # Render input panel
        text, input_state = render_input_panel(DEFAULT_DEMO_TEXT_PATH)
        
        # Process detection if button clicked or settings changed
        if input_state.get("process_clicked", False):
            with st.spinner("Analyzing text..."):
                # Get settings from sidebar
                base_model = settings["base_model"]
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

                # Check if custom pipeline is selected
                if "use_custom_pipeline" in st.session_state and st.session_state["use_custom_pipeline"] and "custom_pipeline" in settings:
                    st.info("Using custom pipeline for entity detection")
                    
                    # Get the custom pipeline configuration
                    custom_pipeline = settings["custom_pipeline"]
                    
                    # Process text with the custom pipeline
                    results = process_with_custom_pipeline(
                        text=text,
                        custom_pipeline=custom_pipeline,
                        settings=settings
                    )
                elif base_model == "presidio":
                    analyzer, ad_hoc_recognizers = initialize_analyzer_engine(
                        model_family=model_family,
                        model_path=model_path,
                        deny_list=deny_list,
                        regex_pattern=regex_pattern,
                        regex_entity_type=regex_entity_type,
                        regex_score=regex_score,
                        regex_context=regex_context
                    )

                    # Run detection
                    results = analyze_text(
                        _analyzer=analyzer,
                        text=text,
                        entities=selected_entities,
                        language="en",
                        score_threshold=threshold,
                        return_decision_process=return_decision_process,
                        allow_list=allow_list,
                        ad_hoc_recognizers=ad_hoc_recognizers,
                    )
                else:
                    model = get_independent_model(
                        model_family=model_family,
                        model_path=model_path
                    )
                    results = model.analyze(text=text, entities=selected_entities, language='en')
                
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
        # Add custom pipeline status indicator
        if st.session_state.get("use_custom_pipeline", False):
            st.markdown("""
                <div style="background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
                           padding: 0.5rem 1rem;
                           border-radius: 4px;
                           margin-bottom: 1rem;
                           display: flex;
                           align-items: center;
                           gap: 0.5rem;">
                    <span style="font-size: 1.2rem;">⚙️</span>
                    <span style="color: #4A4A4A; font-weight: 500;">Custom Pipeline Active - Evaluation Results May Vary</span>
                </div>
            """, unsafe_allow_html=True)
        
        render_evaluation_panel(settings)

    # Model Benchmarks tab
    with tabs[2]:
        render_benchmarks_panel(settings)

    # Model Comparison tab
    with tabs[3]:
        render_model_comparison_panel(settings)
    
    # Custom Pipeline tab
    with tabs[4]:
        render_custom_pipeline_panel(settings)
    
    # Footer
    st.markdown("---")
    st.caption(
        """
        HotSwapPII | Using Microsoft Presidio | 
        [GitHub](https://github.com/microsoft/presidio) | 
        [Documentation](https://microsoft.github.io/presidio/)
        """
    )

    # Custom CSS for RBC branding
    st.markdown("""
        <style>
        /* Main background and container styles */
        .stApp {
            background-color: #FFFFFF;
            color: #4A4A4A;
        }
        
        /* Change st-c2 class background to use RBC yellow */
        .st-c2 {
            background-color: #FFD700 !important;
        }
        
        /* Header styling */
        .stApp header {
            background: linear-gradient(135deg, #0052CC 0%, #014da4 100%);
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Button styling */
        .stButton button {
            background: linear-gradient(135deg, #0052CC 0%, #014da4 100%);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            background: linear-gradient(135deg, #014da4 0%, #0052CC 100%);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #4A4A4A;
            border-bottom: 2px solid transparent;
            transition: all 0.3s ease;
            font-weight: 500;
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #0052CC;
            border-bottom: 2px solid #0052CC;
            font-weight: 600;
        }
        
        /* Remove secondary underline */
        .stTabs [data-baseweb="tab-list"] button:nth-child(2) {
            border-bottom: none !important;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #F8F9FA;
            border-right: 1px solid #E9ECEF;
        }
        
        /* Sidebar text and elements */
        .css-1d391kg .stMarkdown {
            color: #4A4A4A;
        }
        
        .css-1d391kg .stSelectbox label,
        .css-1d391kg .stCheckbox label,
        .css-1d391kg .stRadio label,
        .css-1d391kg .stSlider label {
            color: #4A4A4A;
        }
        
        .css-1d391kg .stSelectbox select,
        .css-1d391kg .stCheckbox input,
        .css-1d391kg .stRadio input {
            background-color: #FFFFFF;
            border: 1px solid #E9ECEF;
            color: #4A4A4A;
        }
        
        /* Checkbox styling */
        .stCheckbox input[type="checkbox"] {
            accent-color: #FFD700;
        }
        
        /* Radio button styling */
        .stRadio input[type="radio"] {
            accent-color: #FFD700;
        }
        
        /* Card and container styling */
        .stCard {
            background-color: #FFFFFF;
            border: 1px solid #E9ECEF;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        /* Input fields */
        .stTextInput input, 
        .stTextArea textarea,
        .stTextInput label,
        .stTextArea label {
            border: 1px solid #E9ECEF;
            border-radius: 0.25rem;
            padding: 0.5rem;
            color: #4A4A4A;
            background-color: #FFFFFF;
        }
        
        .stTextInput input:focus, 
        .stTextArea textarea:focus {
            border-color: #0052CC;
            box-shadow: 0 0 0 2px rgba(0,82,204,0.1);
        }
        
        /* Select boxes */
        .stSelectbox select {
            border: 1px solid #E9ECEF;
            border-radius: 0.25rem;
            padding: 0.5rem;
            color: #4A4A4A;
            background-color: #FFFFFF;
        }
        
        .stSelectbox select:focus {
            border-color: #0052CC;
            box-shadow: 0 0 0 2px rgba(0,82,204,0.1);
        }
        
        /* Sliders */
        .stSlider [data-baseweb="slider"] {
            background-color: #E9ECEF;
        }
        
        .stSlider [data-baseweb="slider"] [data-baseweb="thumb"] {
            background-color: #0052CC;
        }
        
        /* Footer styling */
        footer {
            background-color: #F8F9FA;
            color: #4A4A4A;
            padding: 1rem;
            text-align: center;
            border-top: 1px solid #E9ECEF;
        }
        
        /* Info and success messages */
        .stAlert {
            border-radius: 0.25rem;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        .stAlert[data-baseweb="notification"] {
            background-color: #E3F2FD;
            border-left: 4px solid #0052CC;
            color: #014da4;
        }
        
        .stSuccess {
            background-color: #E8F5E9;
            border-left: 4px solid #2E7D32;
            color: #1B5E20;
        }
        
        /* Error messages */
        .stError {
            background-color: #FFF8E1;
            border-left: 4px solid #FFD700;
            color: #4A4A4A;
        }
        
        /* Warning messages */
        .stWarning {
            background-color: #FFF8E1;
            border-left: 4px solid #FFD700;
            color: #4A4A4A;
        }
        
        /* Note sections and info boxes */
        .stMarkdown .stAlert,
        .stMarkdown .stInfo {
            background-color: #FFF8E1;
            border-left: 4px solid #FFD700;
            color: #4A4A4A;
            padding: 1rem;
            border-radius: 0.25rem;
            margin: 1rem 0;
        }
        
        /* Divider styling */
        .stDivider {
            background: linear-gradient(90deg, #0052CC 0%, #FFD700 100%);
            height: 2px;
            margin: 1rem 0;
        }
        
        /* Metric cards */
        .stMetric {
            background-color: #FFFFFF;
            border: 1px solid #E9ECEF;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        .stMetric:hover {
            border-color: #0052CC;
            box-shadow: 0 2px 4px rgba(0,82,204,0.1);
        }
        
        /* Dataframe styling */
        .stDataFrame {
            border: 1px solid #E9ECEF;
            border-radius: 0.5rem;
            overflow: hidden;
        }
        
        .stDataFrame thead th {
            background-color: #F8F9FA;
            color: #4A4A4A;
            font-weight: 600;
        }
        
        .stDataFrame tbody tr:hover {
            background-color: #F8F9FA;
        }
        
        /* Custom pipeline status indicator */
        .custom-pipeline-indicator {
            background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
            padding: 0.5rem 1rem;
            border-radius: 4px;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .custom-pipeline-indicator span {
            color: #4A4A4A;
            font-weight: 500;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #F8F9FA;
            border: 1px solid #E9ECEF;
            border-radius: 0.25rem;
            padding: 0.5rem 1rem;
            color: #4A4A4A;
            font-weight: 500;
        }
        
        .streamlit-expanderHeader:hover {
            background-color: #E9ECEF;
        }
        
        /* Caption styling */
        .stCaption {
            color: #6C757D;
            font-size: 0.875rem;
        }
        
        /* Label styling */
        .stLabel {
            color: #4A4A4A;
            font-weight: 500;
        }
        
        /* Error text styling */
        .stMarkdown .stError {
            color: #4A4A4A;
        }
        
        /* Warning text styling */
        .stMarkdown .stWarning {
            color: #4A4A4A;
        }
        </style>
    """, unsafe_allow_html=True)

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
