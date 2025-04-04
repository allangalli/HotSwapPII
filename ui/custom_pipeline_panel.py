"""
Custom Pipeline panel UI components for HotSwapPII.
Allows users to select specific models for each entity type based on performance.
"""
import json
import logging
import os
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from config.config import MODEL_DETAILS, MODEL_OPTIONS, MODEL_OPTIONS_TO_BENHCMARKS_KEY, DATASET_OPTIONS, DATASET_BENCHMARK_RESULTS_FILE_PATH

logger = logging.getLogger(__name__)

def get_entity_types_from_benchmark_results(file_path: str) -> List[str]:
    """
    Extract all entity types from a benchmark results file.
    
    Args:
        file_path: Path to the benchmark results JSON file
        
    Returns:
        List of entity type names
    """
    try:
        if not os.path.exists(file_path):
            return []
            
        with open(file_path, 'r') as f:
            data = json.load(f)

        def filter_out_singular_metrics(key):
            unwanted_keys = ['rows', 'time_taken']
            if key in unwanted_keys:
                return False
            else:
                return True

        filtered_keys = filter(filter_out_singular_metrics, data.keys())

        # The first model key doesn't matter, we just want entity types
        model_key = list(filtered_keys)[0]
        # Remove 'overall' from entity types
        entity_types = [et for et in data[model_key].keys() if et not in ['overall', 'rows', 'time_taken']]
        return entity_types
    except Exception as e:
        logger.error(f"Error loading entity types from {file_path}: {e}")
        return []

def get_entity_performance_from_benchmark(file_path: str) -> pd.DataFrame:
    """
    Extract performance metrics for each model and entity type from benchmark results.
    
    Args:
        file_path: Path to the benchmark results JSON file
        
    Returns:
        DataFrame with model performance by entity type
    """
    try:
        if not os.path.exists(file_path):
            return pd.DataFrame()
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Create a list to store performance data
        performance_data = []

        def filter_out_singular_metrics(pair):
            unwanted_keys = ['rows', 'time_taken']
            key, value = pair
            if key in unwanted_keys:
                return False
            else:
                return True

        filtered_dictionary = dict(filter(filter_out_singular_metrics, data.items()))

        # Loop through each model in the benchmark data
        for model_name, model_results in filtered_dictionary.items():

            filtered_results_dictionary = dict(filter(filter_out_singular_metrics, model_results.items()))
            # Loop through each entity type (excluding 'overall')
            for entity_type, metrics in filtered_results_dictionary.items():
                if entity_type != 'overall':
                    # Extract recall metric for entity type using type validation
                    recall = metrics.get('partial', {}).get('recall', 0)
                    performance_data.append({
                        'model': model_name,
                        'entity_type': entity_type,
                        'recall': recall
                    })
        
        # Convert to DataFrame
        df = pd.DataFrame(performance_data)
        return df
    except Exception as e:
        logger.error(f"Error loading performance data from {file_path}: {e}")
        return pd.DataFrame()

def get_best_model_for_entity_types(file_path: str) -> Dict[str, str]:
    """
    Determine the best model for each entity type based on recall score.
    
    Args:
        file_path: Path to the benchmark results JSON file
        
    Returns:
        Dictionary mapping entity types to their best model
    """
    performance_df = get_entity_performance_from_benchmark(file_path)
    
    if performance_df.empty:
        return {}
    
    # Get the best model for each entity type
    best_models = {}
    for entity_type in performance_df['entity_type'].unique():
        entity_df = performance_df[performance_df['entity_type'] == entity_type]
        if not entity_df.empty:
            # Get the model with the highest recall
            best_model = entity_df.loc[entity_df['recall'].idxmax()]
            best_models[entity_type] = {
                'model': best_model['model'],
                'recall': best_model['recall']
            }
    
    return best_models

def get_model_display_name(benchmark_key: str) -> str:
    """
    Convert a benchmark key to a user-friendly model display name.
    
    Args:
        benchmark_key: The benchmark key for the model
        
    Returns:
        The display name for the model
    """
    # Create reverse mapping
    reverse_mapping = {v: k for k, v in MODEL_OPTIONS_TO_BENHCMARKS_KEY.items()}
    return reverse_mapping.get(benchmark_key, benchmark_key)

def render_custom_pipeline_panel(settings: Dict) -> None:
    """
    Render the custom pipeline panel UI.
    
    Args:
        settings: Dictionary with application settings
    """
    st.header("Custom Pipeline Builder")
    
    with st.expander("About Custom Pipeline", expanded=False):
        st.info(
            """
            **Custom Pipeline Builder**
            
            Build a custom pipeline by selecting the best model for each PII entity type.
            
            The custom pipeline allows you to:
            1. See which model performs best for each entity type
            2. Select specific models for detecting each entity type
            3. Create a hybrid detection system that uses the best model for each entity
            
            When text is processed through the custom pipeline, each model will detect 
            its assigned entity types, and the results will be combined.
            """
        )
    
    # Show custom pipeline status
    if "use_custom_pipeline" in st.session_state and st.session_state["use_custom_pipeline"]:
        st.success("✅ Custom pipeline is currently ACTIVE")
        
        # Button to disable custom pipeline
        if st.button("Disable Custom Pipeline", type="secondary"):
            st.session_state.use_custom_pipeline = False
            if "custom_pipeline" in settings:
                del settings["custom_pipeline"]
            st.rerun()
    else:
        st.info("⚠️ Custom pipeline is currently INACTIVE")

    # Select dataset for benchmark results
    selected_dataset_index = st.selectbox(
        "Select benchmark dataset",
        options=range(len(DATASET_OPTIONS)),
        format_func=lambda i: DATASET_OPTIONS[i],
        help="Choose which dataset to use for determining the best models"
    )
    
    # Get the file path for the selected dataset benchmark results
    benchmark_file_path = DATASET_BENCHMARK_RESULTS_FILE_PATH[selected_dataset_index]
    
    if not os.path.exists(benchmark_file_path):
        st.warning(f"No benchmark results found for {DATASET_OPTIONS[selected_dataset_index]}")
        return
    
    # Get entity types from benchmark results
    entity_types = get_entity_types_from_benchmark_results(benchmark_file_path)
    
    if not entity_types:
        st.warning("No entity types found in benchmark results")
        return
    
    # Get best models for each entity type
    best_models = get_best_model_for_entity_types(benchmark_file_path)
    
    # Create a table for best models
    st.subheader("Best Performing Models by Entity Type")

    def filter_out_singular_metrics(pair):
        unwanted_keys = ['rows', 'time_taken']
        key, value = pair
        if key in unwanted_keys:
            return False
        else:
            return True
    filtered_best_models = dict(filter(filter_out_singular_metrics, best_models.items()))

    best_model_data = []
    for entity_type, model_info in filtered_best_models.items():
        model_display_name = get_model_display_name(model_info['model'])
        best_model_data.append({
            "Entity Type": entity_type,
            "Best Model": model_display_name,
            "Recall Score": f"{model_info['recall']:.2%}"
        })
    
    best_df = pd.DataFrame(best_model_data)
    st.dataframe(best_df, use_container_width=True, hide_index=True)
    
    # Add download button for best models
    csv_best_models = best_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Best Models by Entity Type (CSV)",
        data=csv_best_models,
        file_name="best_models_by_entity.csv",
        mime="text/csv",
    )
    
    # Create custom pipeline configuration
    st.subheader("Custom Pipeline Configuration")

    # Display entity types with model selection dropdowns
    st.write("Select the model to use for each entity type:")

    # Calculate columns based on number of entity types
    num_cols = 5
    cols = st.columns(num_cols)

    # Store selected models in session state if not already there
    if "custom_pipeline_models" not in st.session_state:
        st.session_state.custom_pipeline_models = {}

        # Initialize with best models
        for entity_type, model_info in filtered_best_models.items():
            model_display_name = get_model_display_name(model_info['model'])
            st.session_state.custom_pipeline_models[entity_type] = model_display_name

    # Render dropdowns for each entity type
    for i, entity_type in enumerate(entity_types):
        col_idx = i % num_cols
        with cols[col_idx]:
            # Get best model for this entity type
            best_model_name = get_model_display_name(filtered_best_models.get(entity_type, {}).get('model', MODEL_OPTIONS[0]))

            # Create dropdown with model options
            selected_model = st.selectbox(
                f"{entity_type}",
                options=MODEL_OPTIONS,
                index=MODEL_OPTIONS.index(best_model_name) if best_model_name in MODEL_OPTIONS else 0,
                key=f"model_select_{entity_type}"
            )

            # Store selection in session state
            st.session_state.custom_pipeline_models[entity_type] = selected_model

    # Add button to use the custom pipeline
    if st.button("Apply Custom Pipeline", type="primary"):
        # Store custom pipeline config in settings
        settings["custom_pipeline"] = st.session_state.custom_pipeline_models
        
        # Update session state and rerun
        st.session_state.use_custom_pipeline = True
        st.success("Custom pipeline configured! Use the PII Detection tab to process text with this pipeline.")
        st.rerun()

    # Display the current custom pipeline configuration
    st.subheader("Current Custom Pipeline Configuration")

    if "custom_pipeline_models" in st.session_state and st.session_state.custom_pipeline_models:
        pipeline_data = []
        for entity_type, model in st.session_state.custom_pipeline_models.items():
            pipeline_data.append({
                "Entity Type": entity_type,
                "Selected Model": model
            })

        pipeline_df = pd.DataFrame(pipeline_data)
        st.dataframe(pipeline_df, use_container_width=True, hide_index=True)
        
        # Add download button for custom pipeline configuration
        csv_pipeline = pipeline_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Custom Pipeline Configuration (CSV)",
            data=csv_pipeline,
            file_name="custom_pipeline_config.csv",
            mime="text/csv",
        )
        
        # Add a warning if pipeline is active but configuration has changed
        if "custom_pipeline" in settings and settings["custom_pipeline"] != st.session_state.custom_pipeline_models:
            st.warning("⚠️ Custom pipeline configuration has changed. Click 'Apply Custom Pipeline' to update.")
    else:
        st.info("No custom pipeline configured yet. Select models for each entity type above.")
