"""
Evaluation panel UI components for the HotSwapPII.
"""
import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from core.evaluator import (analyze_threshold_sensitivity, evaluate_model,
                            process_validation_data)
from utils.visualization import (create_f1_comparison_chart,
                                 create_metrics_heatmap)
from config.config import MODEL_OPTIONS_TO_BENHCMARKS_KEY, DEFAULT_ADD_TO_BENCHMARK

logger = logging.getLogger(__name__)

def render_evaluation_panel(settings: Dict) -> None:
    """
    Render the model evaluation panel.
    
    Args:
        settings: Dictionary with application settings
    """
    st.header("Model Evaluation")
    
    with st.expander("About Model Evaluation", expanded=False):
        st.info(
            """
            **Model Evaluation**
            
            Upload a CSV file with labeled PII data to evaluate the model's performance.
            
            **Expected CSV format:**
            - Must contain a 'text' column with the text to evaluate
            - Must contain a 'label' column with JSON data in this format:
              ```
              [{"start":58,"end":89,"text":"John Smith","labels":["name"]},...]
              ```
              
            The evaluation will calculate:
            - Precision: What percentage of detected entities are correct
            - Recall: What percentage of actual entities were detected
            - F1 Score: Harmonic mean of precision and recall
            - Performance by entity type
            
            You can adjust confidence threshold and other settings to see how they affect performance.
            """
        )
    
    # File uploader for validation data
    uploaded_file = st.file_uploader(
        "Upload validation data (CSV)",
        type=["csv"],
        help="CSV file with 'text' and 'label' columns",
    )

    add_to_benchmark = st.checkbox(
        "Add results to benchmarks data",
        value=DEFAULT_ADD_TO_BENCHMARK,
        help="Filter out entities that overlap with higher confidence entities",
    )
    
    if uploaded_file is not None:
        with st.spinner("Processing validation data..."):
            validation_df, error_message = process_validation_data(uploaded_file)
            
            if error_message:
                st.error(error_message)
                
                # Show expected format
                st.info(
                    """
                    **Expected data format:**
                    
                    Your CSV file must contain:
                    1. A 'text' column with the text to evaluate
                    2. A 'label' column with JSON data in this format:
                    ```
                    [{"start":58,"end":89,"text":"John Smith","labels":["name"]},...]
                    ```
                    
                    If your data is in a different format, you may need to preprocess it.
                    """
                )
                
                # Show sample of the file for debugging
                try:
                    sample_df = pd.read_csv(uploaded_file)
                    st.subheader("Sample of uploaded data")
                    st.dataframe(sample_df.head())
                except Exception as e:
                    st.error(f"Could not display sample of uploaded file: {e}")
            else:
                st.success(f"Successfully loaded validation data with {len(validation_df)} records")
                
                # Show sample of processed data
                with st.expander("View sample data", expanded=False):
                    if "processed_annotations" in validation_df.columns:
                        # Create a display version with sample text and annotation count
                        display_df = validation_df[["text"]].copy()
                        display_df["annotations_count"] = validation_df["processed_annotations"].apply(len)
                        st.dataframe(display_df.head())
                    else:
                        st.dataframe(validation_df[["text", "label"]].head())
                
                # Evaluation settings
                st.subheader("Evaluation Settings")
                
                # Split into tabs for basic and advanced settings
                settings_tabs = st.tabs(["Basic Settings", "Advanced Analysis"])
                
                with settings_tabs[0]:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        eval_threshold = st.slider(
                            "Confidence threshold",
                            min_value=0.0,
                            max_value=1.0,
                            value=settings.get("threshold", 0.35),
                            step=0.05,
                            help="Minimum confidence score for entity detection",
                        )
                    
                    with col2:
                        match_threshold = st.slider(
                            "Match threshold",
                            min_value=0.1,
                            max_value=1.0,
                            value=0.5,
                            step=0.1,
                            help="IoU threshold for matching predicted and ground truth entities",
                        )
                
                with settings_tabs[1]:
                    run_threshold_analysis = st.checkbox(
                        "Run threshold sensitivity analysis",
                        value=False,
                        help="Analyze how different confidence thresholds affect performance",
                    )
                    
                    if run_threshold_analysis:
                        num_thresholds = st.slider(
                            "Number of thresholds to test",
                            min_value=5,
                            max_value=20,
                            value=10,
                            step=1,
                            help="Number of evenly spaced thresholds between 0 and 1",
                        )
                        
                        # Generate thresholds
                        thresholds = np.linspace(0.05, 0.95, num_thresholds).tolist()
                
                # Evaluation button
                if st.button("Evaluate Model", type="primary"):
                    # Get settings from sidebar
                    base_model = settings.get("base_model")
                    model_family = settings.get("model_family")
                    model_path = settings.get("model_path")
                    selected_entities = settings.get("selected_entities")
                    exclude_overlaps = settings.get("exclude_overlaps", True)
                    overlap_tolerance = settings.get("overlap_tolerance", 1)
                    
                    # Set up progress tracking
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                    
                    def update_progress(progress):
                        progress_bar.progress(progress)
                        progress_text.text(f"Evaluated {int(progress * 100)}% of samples...")
                    
                    try:
                        # Run evaluation
                        with st.spinner("Evaluating model..."):
                            # Get custom pipeline if active
                            custom_pipeline = None
                            if st.session_state.get("use_custom_pipeline", False) and "custom_pipeline_models" in st.session_state:
                                custom_pipeline = st.session_state.custom_pipeline_models
                                st.info("Using custom pipeline for evaluation")
                            
                            # Run evaluation with or without custom pipeline
                            results_df, overall_metrics, entity_metrics, nervaluate_overall_metrics, nervaluate_entity_metrics = evaluate_model(
                                df=validation_df,
                                base_model=base_model,
                                model_family=model_family,
                                model_path=model_path,
                                entities=selected_entities,
                                threshold=eval_threshold,
                                exclude_overlaps=exclude_overlaps,
                                overlap_tolerance=overlap_tolerance,
                                overlap_threshold=match_threshold,
                                progress_callback=update_progress,
                                custom_pipeline=custom_pipeline
                            )
                            
                            # Run threshold analysis if requested (only for non-custom pipeline)
                            threshold_results = None
                            if run_threshold_analysis and 'thresholds' in locals() and not custom_pipeline:
                                with st.spinner("Running threshold sensitivity analysis..."):
                                    threshold_results = analyze_threshold_sensitivity(
                                        df=validation_df,
                                        model_family=model_family,
                                        model_path=model_path,
                                        thresholds=thresholds,
                                        exclude_overlaps=False,
                                        overlap_tolerance=overlap_tolerance,
                                        overlap_threshold=match_threshold,
                                    )

                            if (add_to_benchmark):
                                model_selection = settings.get("model_selection")
                                add_to_benchmark_panel(uploaded_file, nervaluate_overall_metrics, nervaluate_entity_metrics, model_selection)
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        progress_text.empty()

                        # Results
                        st.subheader("Results")

                        # Display results
                        render_evaluation_results(
                            results_df, 
                            overall_metrics, 
                            entity_metrics,
                            nervaluate_overall_metrics,
                            nervaluate_entity_metrics,
                            threshold_results
                        )
                    except Exception as e:
                        st.error(f"Error during evaluation: {e}")
                        logger.exception("Evaluation error")

def add_to_benchmark_panel(uploaded_file, nervaluate_overall_metrics, nervaluate_entity_metrics, model_selection: str):
    file_name = uploaded_file.name.split('.')[0] + '_results.json'
    file_path = f"./data/benchmark_results/{file_name}"
    nervaluate_entity_metrics['overall'] = nervaluate_overall_metrics

    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    data[MODEL_OPTIONS_TO_BENHCMARKS_KEY[model_selection]] = nervaluate_entity_metrics

    # Write the updated data back to the JSON file
    with open(file_path, "w") as file:
        json.dump(data, file)


def render_evaluation_results(
    results_df: pd.DataFrame,
    overall_metrics: Dict,
    entity_metrics: pd.DataFrame,
    nervaluate_overall_metrics: Dict,
    nervaluate_entity_metrics: Dict,
    threshold_results: pd.DataFrame = None  # noqa: F821
) -> None:
    """
    Render the evaluation results.
    
    Args:
        results_df: DataFrame with per-document results
        overall_metrics: Dictionary with overall metrics
        entity_metrics: DataFrame with per-entity-type metrics
        threshold_results: Optional DataFrame with threshold sensitivity results
    """
    # Create tabs for different metric calculation algorithms
    metric_evaluation_tabs = st.tabs(["NERvaluate", "In House"])

    with metric_evaluation_tabs[0]:
        # Create tabs for different result views
        result_tabs = st.tabs(["Summary", "Entity Performance", "Document Results"])

        # Summary tab
        with result_tabs[0]:

            st.subheader("Overall Performance")
            metric_evaluation_schemas = ["Type", "Partial", "Exact", "Strict"]
            metric_evaluation_schemas_keys = ["ent_type", "partial", "exact", "strict"]
            metric_evaluation_schema_tabs = st.tabs(metric_evaluation_schemas)

            for i in range(4):
                with metric_evaluation_schema_tabs[i]:
                    st.text(
                        f"Metric evaluation schema: {metric_evaluation_schemas[i]}",
                        help="Percentage of detected entities that are correct",
                    )


                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Precision",
                            f"{nervaluate_overall_metrics[metric_evaluation_schemas_keys[i]]['precision']:.2%}",
                            help="Percentage of detected entities that are correct",
                        )

                    with col2:
                        st.metric(
                            "Recall",
                            f"{nervaluate_overall_metrics[metric_evaluation_schemas_keys[i]]['recall']:.2%}",
                            help="Percentage of actual entities that were detected",
                        )

                    with col3:
                        st.metric(
                            "F1 Score",
                            f"{nervaluate_overall_metrics[metric_evaluation_schemas_keys[i]]['f1']:.2%}",
                            help="Harmonic mean of precision and recall",
                        )

                    # Additional metrics
                    st.subheader("Detailed Metrics")

                    col1, col2, col3, col4 = st.columns(4)
                    col5, col6, col7 = st.columns(3)

                    with col1:
                        st.metric(
                            "Correct",
                            nervaluate_overall_metrics[metric_evaluation_schemas_keys[i]]['correct'],
                            help="Correctly identified entities",
                        )

                    with col2:
                        st.metric(
                            "Incorrect",
                            nervaluate_overall_metrics[metric_evaluation_schemas_keys[i]]['incorrect'],
                            help="Incorrectly identified entities",
                        )

                    with col3:
                        st.metric(
                            "Partial",
                            nervaluate_overall_metrics[metric_evaluation_schemas_keys[i]]['partial'],
                            help="Partially correct identified entities",
                        )

                    with col4:
                        st.metric(
                            "Missed",
                            nervaluate_overall_metrics[metric_evaluation_schemas_keys[i]]['missed'],
                            help="Number of entities in ground truth that weren't detected by the model",
                        )

                    with col5:
                        st.metric(
                            "Spurious",
                            nervaluate_overall_metrics[metric_evaluation_schemas_keys[i]]['spurious'],
                            help="Number of entities predicted by the model that weren't in the ground truth",
                        )

                    with col6:
                        st.metric(
                            "Possible",
                            nervaluate_overall_metrics[metric_evaluation_schemas_keys[i]]['possible'],
                            help="Total number of entities in the ground truth",
                        )

                    with col7:
                        st.metric(
                            "Actual",
                            nervaluate_overall_metrics[metric_evaluation_schemas_keys[i]]['actual'],
                            help="Total number of entities predicted by the model",
                        )

        # Entity Performance tab
        with result_tabs[1]:
            st.subheader("Performance by Tag")
            metric_evaluation_schemas = ["Type", "Partial", "Exact", "Strict"]
            metric_evaluation_schemas_keys = ["ent_type", "partial", "exact", "strict"]
            metric_evaluation_schema_tabs = st.tabs(metric_evaluation_schemas)
            st.subheader("Performance by Entity Type")

            for i in range(4):
                with metric_evaluation_schema_tabs[i]:
                    entity_types = nervaluate_entity_metrics.keys()
                    data = []

                    for entity_type in entity_types:
                        data.append([
                            entity_type,
                            f"{nervaluate_entity_metrics[entity_type][metric_evaluation_schemas_keys[i]]["precision"]:.2%}",
                            f"{nervaluate_entity_metrics[entity_type][metric_evaluation_schemas_keys[i]]["recall"]:.2%}",
                            f"{nervaluate_entity_metrics[entity_type][metric_evaluation_schemas_keys[i]]["f1"]:.2%}",
                            nervaluate_entity_metrics[entity_type][metric_evaluation_schemas_keys[i]]["correct"],
                            nervaluate_entity_metrics[entity_type][metric_evaluation_schemas_keys[i]]["incorrect"],
                            nervaluate_entity_metrics[entity_type][metric_evaluation_schemas_keys[i]]["partial"],
                            nervaluate_entity_metrics[entity_type][metric_evaluation_schemas_keys[i]]["missed"],
                            nervaluate_entity_metrics[entity_type][metric_evaluation_schemas_keys[i]]["spurious"],
                            nervaluate_entity_metrics[entity_type][metric_evaluation_schemas_keys[i]]["possible"],
                            nervaluate_entity_metrics[entity_type][metric_evaluation_schemas_keys[i]]["actual"],
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
                    
                    # Add download button for entity metrics
                    csv_entity = display_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label=f"Download {metric_evaluation_schemas[i]} Entity Metrics (CSV)",
                        data=csv_entity,
                        file_name=f"entity_metrics_{metric_evaluation_schemas_keys[i]}.csv",
                        mime="text/csv",
                    )

        # Document Results tab
        with result_tabs[2]:
            st.subheader("Results by Document")

            if not results_df.empty:
                # Create display version with key columns
                display_columns = [
                    "text", "precision", "recall", "f1",
                    "true_positives", "false_positives", "false_negatives"
                ]

                if all(col in results_df.columns for col in display_columns):
                    doc_display = results_df[display_columns].copy()

                    # Format percentages
                    for col in ["precision", "recall", "f1"]:
                        doc_display[col] = doc_display[col].map("{:.2%}".format)

                    # Truncate long text
                    doc_display["text"] = doc_display["text"].apply(
                        lambda x: x[:100] + "..." if len(x) > 100 else x
                    )

                    # Rename columns
                    doc_display = doc_display.rename(
                        columns={
                            "text": "Text",
                            "precision": "Precision",
                            "recall": "Recall",
                            "f1": "F1 Score",
                            "true_positives": "TP",
                            "false_positives": "FP",
                            "false_negatives": "FN",
                        }
                    )

                    st.dataframe(doc_display, use_container_width=True, hide_index=True)
                    
                    # Add download button for document results
                    csv_doc_results = doc_display.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Document Results (CSV)",
                        data=csv_doc_results,
                        file_name="document_results_nervaluate.csv",
                        mime="text/csv",
                    )
                else:
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Add download button for full results
                    csv_full_results = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Full Results (CSV)",
                        data=csv_full_results,
                        file_name="full_results_nervaluate.csv",
                        mime="text/csv",
                    )
            else:
                st.info("No per-document results available.")

    with metric_evaluation_tabs[1]:
        # Create tabs for different result views
        result_tabs = st.tabs(["Summary", "Entity Performance", "Document Results", "Threshold Analysis"])

        # Summary tab
        with result_tabs[0]:
            st.subheader("Overall Performance")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Precision",
                    f"{overall_metrics['precision']:.2%}",
                    help="Percentage of detected entities that are correct",
                )

            with col2:
                st.metric(
                    "Recall",
                    f"{overall_metrics['recall']:.2%}",
                    help="Percentage of actual entities that were detected",
                )

            with col3:
                st.metric(
                    "F1 Score",
                    f"{overall_metrics['f1']:.2%}",
                    help="Harmonic mean of precision and recall",
                )

            # Additional metrics
            st.subheader("Detailed Metrics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "True Positives",
                    overall_metrics["true_positives"],
                    help="Correctly identified entities",
                )

            with col2:
                st.metric(
                    "False Positives",
                    overall_metrics["false_positives"],
                    help="Incorrectly identified entities",
                )

            with col3:
                st.metric(
                    "False Negatives",
                    overall_metrics["false_negatives"],
                    help="Missed entities",
                )

            with col4:
                st.metric(
                    "Total Ground Truth",
                    overall_metrics["total_ground_truth"],
                    help="Total number of entities in ground truth",
                )

        # Entity Performance tab
        with result_tabs[1]:
            st.subheader("Performance by Entity Type")

            if not entity_metrics.empty:
                # Format percentages for display
                display_df = entity_metrics.copy()

                # Add percentage formatting
                display_df["precision"] = display_df["precision"].map("{:.2%}".format)
                display_df["recall"] = display_df["recall"].map("{:.2%}".format)
                display_df["f1"] = display_df["f1"].map("{:.2%}".format)

                # Rename columns
                display_df = display_df.rename(
                    columns={
                        "entity_type": "Entity Type",
                        "precision": "Precision",
                        "recall": "Recall",
                        "f1": "F1 Score",
                        "true_positives": "TP",
                        "false_positives": "FP",
                        "false_negatives": "FN",
                        "total_ground_truth": "GT Count",
                        "total_predicted": "Pred Count",
                    }
                )

                # Display the table
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                )

                # Create visualizations
                st.subheader("Entity Performance Visualizations")

                # F1 score heatmap
                f1_heatmap = create_metrics_heatmap(
                    entity_metrics,
                    metric="f1",
                    title="F1 Score by Entity Type"
                )
                if f1_heatmap:
                    st.altair_chart(f1_heatmap, use_container_width=True)

                # Comparison chart
                comparison_chart = create_f1_comparison_chart(entity_metrics)
                if comparison_chart:
                    st.altair_chart(comparison_chart, use_container_width=True)

                # Download buttons
                st.subheader("Export Results")

                col1, col2 = st.columns(2)

                with col1:
                    csv_entity = entity_metrics.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Entity Metrics (CSV)",
                        data=csv_entity,
                        file_name="entity_type_metrics.csv",
                        mime="text/csv",
                    )

                with col2:
                    csv_results = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Full Results (CSV)",
                        data=csv_results,
                        file_name="evaluation_results.csv",
                        mime="text/csv",
                    )
            else:
                st.info("No entity-type metrics available.")

        # Document Results tab
        with result_tabs[2]:
            st.subheader("Results by Document")

            if not results_df.empty:
                # Create display version with key columns
                display_columns = [
                    "text", "precision", "recall", "f1",
                    "true_positives", "false_positives", "false_negatives"
                ]

                if all(col in results_df.columns for col in display_columns):
                    doc_display = results_df[display_columns].copy()

                    # Format percentages
                    for col in ["precision", "recall", "f1"]:
                        doc_display[col] = doc_display[col].map("{:.2%}".format)

                    # Truncate long text
                    doc_display["text"] = doc_display["text"].apply(
                        lambda x: x[:100] + "..." if len(x) > 100 else x
                    )

                    # Rename columns
                    doc_display = doc_display.rename(
                        columns={
                            "text": "Text",
                            "precision": "Precision",
                            "recall": "Recall",
                            "f1": "F1 Score",
                            "true_positives": "TP",
                            "false_positives": "FP",
                            "false_negatives": "FN",
                        }
                    )

                    st.dataframe(doc_display, use_container_width=True, hide_index=True)
                    
                    # Add download button for document results
                    csv_doc_results = doc_display.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Document Results (CSV)",
                        data=csv_doc_results,
                        file_name="document_results_in_house.csv",
                        mime="text/csv",
                    )
                else:
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Add download button for full results
                    csv_full_results = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Full Results (CSV)",
                        data=csv_full_results,
                        file_name="full_results_in_house.csv",
                        mime="text/csv",
                    )
            else:
                st.info("No per-document results available.")

        # Threshold Analysis tab
        with result_tabs[3]:
            st.subheader("Threshold Sensitivity Analysis")

            if threshold_results is not None and not threshold_results.empty:
                # Display threshold results
                st.write("Performance metrics at different confidence thresholds:")

                # Format percentages for display
                display_df = threshold_results.copy()

                # Format columns
                display_df["precision"] = display_df["precision"].map("{:.2%}".format)
                display_df["recall"] = display_df["recall"].map("{:.2%}".format)
                display_df["f1"] = display_df["f1"].map("{:.2%}".format)
                display_df["threshold"] = display_df["threshold"].map("{:.2f}".format)

                # Rename columns
                display_df = display_df.rename(
                    columns={
                        "threshold": "Threshold",
                        "precision": "Precision",
                        "recall": "Recall",
                        "f1": "F1 Score",
                        "true_positives": "TP",
                        "false_positives": "FP",
                        "false_negatives": "FN",
                    }
                )

                # Display table
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Add download button for threshold results
                csv_threshold_display = display_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Threshold Results (CSV)",
                    data=csv_threshold_display,
                    file_name="threshold_analysis_results.csv",
                    mime="text/csv",
                )
                
                # Create line chart of metrics vs threshold
                threshold_chart_data = threshold_results.melt(
                    id_vars=["threshold"],
                    value_vars=["precision", "recall", "f1"],
                    var_name="Metric",
                    value_name="Value"
                )

                # Make first letter uppercase
                threshold_chart_data["Metric"] = threshold_chart_data["Metric"].str.capitalize()

                # Create chart
                chart = alt.Chart(threshold_chart_data).mark_line(point=True).encode(
                    x=alt.X("threshold:Q", title="Confidence Threshold"),
                    y=alt.Y("Value:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color(
                        "Metric:N",
                        scale=alt.Scale(
                            domain=["Precision", "Recall", "F1"],
                            range=["#1f77b4", "#ff7f0e", "#2ca02c"]
                        )
                    ),
                    tooltip=["threshold", "Metric", "Value"]
                ).properties(
                    title="Performance Metrics vs. Confidence Threshold",
                    width=600,
                    height=400
                ).interactive()

                st.altair_chart(chart, use_container_width=True)

                # Optimal threshold suggestion
                best_f1_idx = threshold_results["f1"].idxmax()
                best_threshold = threshold_results.loc[best_f1_idx, "threshold"]

                st.info(
                    f"Based on this analysis, the optimal confidence threshold for maximizing F1 score "
                    f"is approximately {best_threshold:.2f}, which yields an F1 score of "
                    f"{threshold_results.loc[best_f1_idx, 'f1']:.2%}."
                )

                # Download original threshold results data
                csv_threshold_raw = threshold_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Raw Threshold Data (CSV)",
                    data=csv_threshold_raw,
                    file_name="threshold_analysis_raw.csv",
                    mime="text/csv",
                )
            else:
                st.info(
                    "Threshold sensitivity analysis was not run. Enable it in the Advanced Analysis "
                    "tab before evaluation to see how different confidence thresholds affect performance."
                )