import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

def load_specialty_data(file_path):
    """Load evaluation results for a single specialty"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Extract scores from dictionary fields
        metric_columns = [
            'accuracy', 'thoroughness', 'organised', 'synthesized',
            'hallucination', 'speciality_appropriateness', 'usefulness',
            'fairness', 'comprehensible', 'internally_consistent', 'succinct'
        ]
        
        for col in metric_columns:
            if col in df.columns:
                df[f'{col}_score'] = df[col].apply(lambda x: x['score'] if isinstance(x, dict) else x)
        
        # Convert action types
        if 'final_action_type' in df.columns:
            df['final_action_type_binary'] = df['final_action_type'].map({1: 0, 2: 1})
            
        return df
        
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")
        return pd.DataFrame()

def analyze_specialty(df, specialty_name):
    """Create analysis section for a single specialty"""
    st.header(f"{specialty_name} Analysis")
    
    # Define score columns
    score_columns = [col for col in df.columns if col.endswith('_score')]
    
    # Metrics Overview
    st.subheader("Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_accuracy = df['accuracy_score'].mean()
        st.metric("Average Accuracy", f"{avg_accuracy:.2f}")
    
    with col2:
        avg_thoroughness = df['thoroughness_score'].mean()
        st.metric("Average Thoroughness", f"{avg_thoroughness:.2f}")
    
    with col3:
        avg_specialty = df['speciality_appropriateness_score'].mean()
        st.metric("Appropriate for Speciality", f"{avg_specialty:.2f}")

    with col4:
        avg_hallucination = df['hallucination_score'].mean()
        st.metric("Hallucination", f"{avg_hallucination:.2f}")
    
    with col5:
        total_evaluations = len(df)
        st.metric("Total Evaluations", total_evaluations)
    
    # Score Distributions
    st.subheader("Score Distributions")
    main_metrics = ['hallucination_score', 'accuracy_score', 'thoroughness_score', 'organised_score', 'synthesized_score']
    
    fig = go.Figure()
    for metric in main_metrics:
        if metric in df.columns:
            fig.add_trace(go.Box(
                y=df[metric],
                name=metric.replace('_score', ''),
                boxmean=True
            ))
    
    fig.update_layout(
        title=f"Score Distribution - {specialty_name}",
        yaxis_title="Score",
        boxmode='group'
    )
    st.plotly_chart(fig)
    
    # Classification Analysis
    st.subheader("Classification Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        if all(col in df.columns for col in ['classification', 'final_action_type_binary']):
            conf_matrix = confusion_matrix(
                df['final_action_type_binary'],
                df['classification']
            )
            
            fig = go.Figure(data=go.Heatmap(
                z=conf_matrix,
                x=['Predicted Accept', 'Predicted Reject'],
                y=['Actual Accept', 'Actual Reject'],
                text=conf_matrix,
                texttemplate='%{text}',
                textfont={"size": 12},
                colorscale='RdBu'
            ))
            
            fig.update_layout(
                title=f"Confusion Matrix - {specialty_name}",
                width=400,
                height=400
            )
            st.plotly_chart(fig)
            
            # Calculate metrics
            tn, fp, fn, tp = conf_matrix.ravel()
            metrics = {
                'Accuracy': (tp + tn) / (tp + tn + fp + fn),
                'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0
            }
            metrics['F1-Score'] = 2 * (metrics['Precision'] * metrics['Recall']) / (metrics['Precision'] + metrics['Recall']) if (metrics['Precision'] + metrics['Recall']) > 0 else 0
            
            st.write("Classification Metrics:")
            metrics_df = pd.DataFrame({
                'Metric': list(metrics.keys()),
                'Value': list(metrics.values())
            })
            st.dataframe(metrics_df)
    
    with col2:
        if 'classification_probability' in df.columns:
            # ROC Curve
            fpr, tpr, _ = roc_curve(
                df['final_action_type_binary'],
                df['classification_probability']
            )
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f'ROC curve (AUC = {roc_auc:.2f})',
                mode='lines'
            ))
            
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                name='Random',
                mode='lines',
                line=dict(dash='dash')
            ))
            
            fig.update_layout(
                title=f'ROC Curve - {specialty_name}',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                width=400,
                height=400
            )
            st.plotly_chart(fig)
    
    # Error Analysis
    st.subheader("Error Analysis")
    if all(col in df.columns for col in ['classification', 'final_action_type_binary']):
        misclassified = df[df['classification'] != df['final_action_type_binary']]
        
        st.write(f"Number of misclassified cases: {len(misclassified)}")
        
        if len(misclassified) > 0:
            display_cols = [
                'request_id', 'classification_probability',
                'accuracy_score', 'thoroughness_score', 'speciality_appropriateness_score'
            ]
            st.dataframe(misclassified[display_cols])
    
    # Raw Data
    with st.expander("View Raw Data"):
        display_columns = score_columns + ['request_id', 'final_action_type', 'classification', 'classification_probability']
        st.dataframe(df[display_columns])
        
        # Download button
        csv = df[display_columns].to_csv(index=False)
        st.download_button(
            label=f"Download {specialty_name} Data",
            data=csv,
            file_name=f'evaluation_data_{specialty_name}.csv',
            mime='text/csv',
        )

def create_dashboard():
    st.set_page_config(page_title="Specialty Evaluation Dashboard", layout="wide")
    st.title("Specialty-wise Note Evaluation Dashboard")
    
    # Find all specialty files
    data_dir = Path('../test/final_output/')
    eval_files = list(data_dir.glob('auto_eval_*.json'))
    
    if not eval_files:
        st.error("No evaluation files found!")
        return
    
    # Create tabs for each specialty
    specialty_names = [f.stem.replace('auto_eval_', '') for f in eval_files]
    tabs = st.tabs(specialty_names)
    
    # Analyze each specialty in its own tab
    for tab, specialty_file in zip(tabs, eval_files):
        with tab:
            specialty_name = specialty_file.stem.replace('auto_eval_', '')
            df = load_specialty_data(specialty_file)
            
            if not df.empty:
                analyze_specialty(df, specialty_name)
            else:
                st.error(f"No data available for {specialty_name}")

if __name__ == "__main__":
    create_dashboard()
