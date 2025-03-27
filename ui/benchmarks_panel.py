"""
Evaluation panel UI components for the HotSwapPII.
"""
import logging
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from core.evaluator import (analyze_threshold_sensitivity, evaluate_model,
                            process_validation_data)
from utils.visualization import (create_f1_comparison_chart,
                                 create_metrics_heatmap)
from config.config import DATASET_OPTIONS, DATASET_DESCRIPTIONS, DATASET_SAMPLE_FILE_PATH, DATASET_BENCHMARK_RESULTS_FILE_PATH, MODEL_OPTIONS_TO_BENHCMARKS_KEY

logger = logging.getLogger(__name__)


def render_benchmarks_panel(settings: Dict) -> None:
    """
    Render the model benchmarks panel.

    Args:
        settings: Dictionary with application settings
    """
    st.header("Model Dataset Benchmarks")

    with st.expander("About Model Dataset Benchmarks", expanded=False):
        st.info('Placeholder')

    # Model selection
    dataset_help = """
    Select which dataset benchmark metrics to view. The metrics correspond to the performance of the selected model in
    the sidebar on this particular dataset along with other important metrics. Descriptions of the dataset and it's distinguishing features
    are available.
    """

    dataset_selection = st.selectbox(
        "Dataset",
        options=DATASET_OPTIONS,
        index=None,
        placeholder="Select dataset...",
        help=dataset_help
    )

    if dataset_selection:
        selected_dataset_index = DATASET_OPTIONS.index(dataset_selection)

        with st.expander("About Dataset", expanded=False):
            st.info(DATASET_DESCRIPTIONS[selected_dataset_index])

            st.subheader('Dataset Sample')
            try:
                sample_df = pd.read_csv(DATASET_SAMPLE_FILE_PATH[selected_dataset_index])
                st.dataframe(sample_df.head())
            except Exception as e:
                st.error(f"Could not display sample of benchmark dataset: {e}")

        st.subheader('Dataset Sample')

        metric_evaluation_schemas = ["Type", "Partial", "Exact", "Strict"]
        metric_evaluation_schemas_keys = ["ent_type", "partial", "exact", "strict"]
        metric_evaluation_schema_tabs = st.tabs(metric_evaluation_schemas)

        with open(DATASET_BENCHMARK_RESULTS_FILE_PATH[selected_dataset_index]) as f:
            benchmark_results = json.load(f)[MODEL_OPTIONS_TO_BENHCMARKS_KEY[settings["model_selection"]]]
            for i in range(4):
                with metric_evaluation_schema_tabs[i]:
                    def filter_out_singular_metrics(pair):
                        unwanted_keys = ['rows', 'time_taken']
                        key, value = pair
                        if key in unwanted_keys:
                            return False
                        else:
                            return True

                    filtered_benchmark_results = dict(filter(filter_out_singular_metrics, benchmark_results.items()))
                    entity_types = filtered_benchmark_results.keys()
                    data = []

                    for entity_type in entity_types:
                        data.append([
                            entity_type,
                            f"{filtered_benchmark_results[entity_type][metric_evaluation_schemas_keys[i]]["precision"]:.2%}",
                            f"{filtered_benchmark_results[entity_type][metric_evaluation_schemas_keys[i]]["recall"]:.2%}",
                            f"{filtered_benchmark_results[entity_type][metric_evaluation_schemas_keys[i]]["f1"]:.2%}",
                            filtered_benchmark_results[entity_type][metric_evaluation_schemas_keys[i]]["correct"],
                            filtered_benchmark_results[entity_type][metric_evaluation_schemas_keys[i]]["incorrect"],
                            filtered_benchmark_results[entity_type][metric_evaluation_schemas_keys[i]]["partial"],
                            filtered_benchmark_results[entity_type][metric_evaluation_schemas_keys[i]]["missed"],
                            filtered_benchmark_results[entity_type][metric_evaluation_schemas_keys[i]]["spurious"],
                            filtered_benchmark_results[entity_type][metric_evaluation_schemas_keys[i]]["possible"],
                            filtered_benchmark_results[entity_type][metric_evaluation_schemas_keys[i]]["actual"],
                        ])

                    # Rename columns
                    display_df = pd.DataFrame(
                        data=data,
                        columns=[
                            "Entity Type",
                            "Precision",
                            "Recall",
                            "F1 Score",
                            "Correct",
                            "Incorrect",
                            "Partial",
                            "Missed",
                            "Spurious",
                            "Possible",
                            "Actual"
                        ]
                    )

                    # Display the table
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                    )