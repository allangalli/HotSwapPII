"""
Results panel UI components for the HotSwapPII.
"""
import logging
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from annotated_text import annotated_text
from presidio_analyzer import RecognizerResult

from core.anonymizer import anonymize_text, create_annotated_tokens
from utils.data_processing import format_results_as_json
from utils.synthetic_data import OpenAIParams, generate_synthetic_text
from utils.visualization import (create_confidence_histogram,
                                 create_entity_barchart,
                                 create_entity_confidence_boxplot)

logger = logging.getLogger(__name__)

def render_results_panel(
    text: str,
    detection_results: List[RecognizerResult],
    settings: Dict,
    show_results: bool = False
) -> None:
    """
    Render the results panel with PII detection results.
    
    Args:
        text: The original text
        detection_results: List of detected PII entities
        settings: Dictionary with application settings
        show_results: Whether to show the results
    """
    if not show_results:
        return
    
    if not detection_results:
        st.info("No PII entities detected in the text.")
        return
    
    # Display results in tabs
    tab_titles = ["Detection Results", "Anonymized Text", "Visualizations", "JSON Output"]
    tabs = st.tabs(tab_titles)
    
    # Detection Results tab
    with tabs[0]:
        render_detection_results(text, detection_results, settings)
    
    # Anonymized Text tab
    with tabs[1]:
        render_anonymized_text(text, detection_results, settings)
    
    # Visualizations tab
    with tabs[2]:
        render_visualizations(detection_results)
    
    # JSON Output tab
    with tabs[3]:
        render_json_output(text, detection_results)


def render_detection_results(
    text: str,
    detection_results: List[RecognizerResult],
    settings: Dict
) -> None:
    """
    Render the detection results table and highlighted text.
    
    Args:
        text: The original text
        detection_results: List of detected PII entities
        settings: Dictionary with application settings
    """
    st.subheader("Found PII Entities")
    
    # Create DataFrame for display
    df = pd.DataFrame([r.to_dict() for r in detection_results])
    
    if not df.empty:
        # Add the extracted text
        df["text"] = [text[res.start : res.end] for res in detection_results]
        
        # Rename columns for display
        df_display = df[["entity_type", "text", "start", "end", "score"]].rename(
            columns={
                "entity_type": "Entity Type",
                "text": "Text",
                "start": "Start",
                "end": "End",
                "score": "Confidence",
            }
        )
        
        # Add decision process if requested
        if settings.get("return_decision_process", False) and "analysis_explanation" in df.columns:
            analysis_df = pd.DataFrame([r.analysis_explanation.to_dict() for r in detection_results])
            df_display = pd.concat([df_display, analysis_df], axis=1)
        
        # Show the table
        st.dataframe(
            df_display.reset_index(drop=True),
            use_container_width=True,
            column_config={
                "Confidence": st.column_config.NumberColumn(
                    "Confidence",
                    format="%.2f",
                    help="Confidence score (0-1)",
                ),
            },
        )
    

    # Summary statistics
    if not df.empty:
        st.subheader("Detection Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total PII Entities", len(detection_results))
        
        with col2:
            # Most common entity type
            if len(detection_results) > 0:
                entity_counts = df["entity_type"].value_counts()
                most_common = entity_counts.index[0]
                st.metric("Most Common Type", f"{most_common} ({entity_counts[0]})")
            else:
                st.metric("Most Common Type", "None")
        
        with col3:
            # Average confidence
            if len(detection_results) > 0:
                avg_confidence = df["score"].mean()
                st.metric("Avg. Confidence", f"{avg_confidence:.2f}")
            else:
                st.metric("Avg. Confidence", "N/A")

    # Show highlighted text
    st.subheader("Highlighted Text")
    annotated_tokens = create_annotated_tokens(text, detection_results)
    annotated_text(*annotated_tokens)
    



def render_anonymized_text(
    text: str,
    detection_results: List[RecognizerResult],
    settings: Dict
) -> None:
    """
    Render the anonymized text output.
    
    Args:
        text: The original text
        detection_results: List of detected PII entities
        settings: Dictionary with application settings
    """
    anonymization_method = settings.get("anonymization_method", "replace")
    
    st.subheader(f"Anonymized Output ({anonymization_method})")
    
    # Process synthetic data separately
    if anonymization_method == "synthesize":
        # First create replaced text
        anonymized = anonymize_text(
            text=text,
            entities=detection_results,
            anonymization_method="replace",
        )
        
        # Then generate synthetic text
        openai_params = settings.get("openai_params")
        if openai_params and openai_params.api_key:
            with st.spinner("Generating synthetic text..."):
                synthetic_text = generate_synthetic_text(
                    anonymized_text=anonymized["text"],
                    openai_params=openai_params,
                )
            
            st.text_area(
                label="Text with synthetic PII values",
                value=synthetic_text,
                height=250,
                key="synthetic_text",
            )
        else:
            st.warning("OpenAI API key is required for synthetic data generation.")
            st.text_area(
                label="Anonymized text (with placeholders)",
                value=anonymized["text"],
                height=250,
                key="anonymized_text",
            )
    else:
        # Handle other anonymization methods
        anonymized = anonymize_text(
            text=text,
            entities=detection_results,
            anonymization_method=anonymization_method,
            mask_char=settings.get("mask_char", "*"),
            mask_chars_count=settings.get("mask_chars_count", 15),
            encrypt_key=settings.get("encrypt_key"),
        )
        
        st.text_area(
            label="Anonymized text",
            value=anonymized["text"],
            height=250,
            key="anonymized_text",
        )
    
    # Download button for anonymized text
    if anonymization_method == "synthesize" and 'synthetic_text' in locals():
        download_text = synthetic_text
        filename = "synthetic_text.txt"
    else:
        download_text = anonymized["text"]
        filename = f"anonymized_{anonymization_method}.txt"
    
    st.download_button(
        label="Download Anonymized Text",
        data=download_text,
        file_name=filename,
        mime="text/plain",
    )


def render_visualizations(
    detection_results: List[RecognizerResult]
) -> None:
    """
    Render visualizations of detection results.
    
    Args:
        detection_results: List of detected PII entities
    """
    st.subheader("PII Detection Visualizations")
    
    if not detection_results:
        st.info("No entities detected to visualize.")
        return
    
    # Entity type distribution
    entity_counts = {}
    for result in detection_results:
        entity_type = result.entity_type
        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
    
    # Create entity distribution chart
    entity_chart = create_entity_barchart(entity_counts, "Entity Type Distribution")
    if entity_chart:
        st.altair_chart(entity_chart, use_container_width=True)
    
    # Split visualizations into columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Confidence histogram
        confidence_chart = create_confidence_histogram(
            detection_results, 
            bins=min(10, len(detection_results)),
            title="Confidence Score Distribution"
        )
        if confidence_chart:
            st.altair_chart(confidence_chart, use_container_width=True)
    
    with col2:
        # Entity confidence boxplot
        if len(entity_counts) > 1:  # Only show if multiple entity types
            boxplot_chart = create_entity_confidence_boxplot(
                detection_results,
                title="Confidence by Entity Type"
            )
            if boxplot_chart:
                st.altair_chart(boxplot_chart, use_container_width=True)


def render_json_output(
    text: str,
    detection_results: List[RecognizerResult]
) -> None:
    """
    Render the JSON output of detected entities.
    
    Args:
        text: The original text
        detection_results: List of detected PII entities
    """
    st.subheader("JSON Output")
    
    # Format results as JSON
    json_text = format_results_as_json(text, detection_results)
    
    # Display JSON
    st.code(json_text, language="json")
    
    # Download button for JSON
    st.download_button(
        label="Download JSON",
        data=json_text,
        file_name="pii_detection_results.json",
        mime="application/json",
    )    