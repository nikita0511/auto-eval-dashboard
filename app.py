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
            'accuracy', 'thoroughness', 'organized', 'synthesized',
            'hallucination', 'speciality_appropriateness', 'usefulness',
            'fairness', 'comprehensible', 'internally_consistent', 'succinct'
        ]
        
        for col in metric_columns:
            if col in df.columns:
                df[f'{col}_score'] = df[col].apply(lambda x: x['score'] if isinstance(x, dict) else x)
                df[f'{col}_justification'] = df[col].apply(lambda x: x['justification'] if isinstance(x, dict) else '')
        
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
    metrics = {
        'Accuracy': 'accuracy_score',
        'Thoroughness': 'thoroughness_score',
        'Organization': 'organized_score',
        'Synthesis': 'synthesized_score',
        'Hallucination': 'hallucination_score',
        'Specialty Appropriateness': 'speciality_appropriateness_score',
        'Usefulness': 'usefulness_score',
        'Fairness': 'fairness_score',
        'Comprehensibility': 'comprehensible_score',
        'Internal Consistency': 'internally_consistent_score',
        'Succinctness': 'succinct_score'
    }
    
    # Create metric columns dynamically
    cols = st.columns(4)
    for i, (metric_name, metric_col) in enumerate(metrics.items()):
        if metric_col in df.columns:
            with cols[i % 4]:
                avg_value = df[metric_col].mean()
                st.metric(metric_name, f"{avg_value:.2f}")
    
    # Score Distributions
    st.subheader("Score Distributions")
    
    # Box plot for all metrics
    fig = go.Figure()
    for metric_name, metric_col in metrics.items():
        if metric_col in df.columns:
            fig.add_trace(go.Box(
                y=df[metric_col],
                name=metric_name,
                boxmean=True
            ))
    
    fig.update_layout(
        title=f"Score Distribution - {specialty_name}",
        yaxis_title="Score",
        boxmode='group',
        height=600
    )
    st.plotly_chart(fig)
    
    # Correlation Matrix
    st.subheader("Metric Correlations")
    available_metrics = [col for col in metrics.values() if col in df.columns]
    corr = df[available_metrics].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr,
        x=[metric_name for metric_name, metric_col in metrics.items() if metric_col in df.columns],
        y=[metric_name for metric_name, metric_col in metrics.items() if metric_col in df.columns],
        text=corr.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        colorscale='RdBu'
    ))
    
    fig.update_layout(
        title=f"Correlation Matrix - {specialty_name}",
        height=800,
        width=800
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
                'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
            }
            metrics['F1-Score'] = 2 * (metrics['Precision'] * metrics['Recall']) / (metrics['Precision'] + metrics['Recall']) if (metrics['Precision'] + metrics['Recall']) > 0 else 0
            
            st.write("Classification Metrics:")
            metrics_df = pd.DataFrame({
                'Metric': list(metrics.keys()),
                'Value': [f"{v:.3f}" for v in metrics.values()]
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
            # Show detailed analysis of misclassified cases
            st.write("### Misclassified Cases Analysis")
            
            # Average scores for misclassified vs correctly classified
            correct = df[df['classification'] == df['final_action_type_binary']]
            
            # Use score columns for comparison
            score_metrics = [col for col in df.columns if col.endswith('_score')]
            metric_names = [col.replace('_score', '').title() for col in score_metrics]
            
            comparison_df = pd.DataFrame({
                'Metric': metric_names,
                'Misclassified': [misclassified[col].mean() for col in score_metrics],
                'Correctly Classified': [correct[col].mean() for col in score_metrics]
            })
            
            fig = go.Figure(data=[
                go.Bar(name='Misclassified', x=comparison_df['Metric'], y=comparison_df['Misclassified']),
                go.Bar(name='Correctly Classified', x=comparison_df['Metric'], y=comparison_df['Correctly Classified'])
            ])
            
            fig.update_layout(
                title='Score Comparison: Misclassified vs Correctly Classified Cases',
                barmode='group',
                height=400,
                xaxis_tickangle=-45  # Angle the x-axis labels for better readability
            )
            st.plotly_chart(fig)
            
            # Display misclassified cases
            display_cols = ['request_id', 'classification_probability'] + score_metrics
            st.dataframe(misclassified[display_cols])

    
    # Justification Analysis
    st.subheader("Justification Examples")
    with st.expander("View Random Justifications"):
        justification_cols = [col for col in df.columns if col.endswith('_justification')]
        if justification_cols:
            sample_row = df.sample(n=1).iloc[0]
            for col in justification_cols:
                metric_name = col.replace('_justification', '')
                st.write(f"**{metric_name.title()}**: {sample_row[col]}")
    
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
    data_dir = Path('final_output/')
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
