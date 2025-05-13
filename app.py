import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any

def load_and_process_data(file_path: str):
    """Load and process the evaluation data."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to pandas dataframes for easier analysis
    auto_eval_data = []
    human_eval_data = []
    
    for sample in data:
        specialty = sample.get('specialties',[])[0]
        request_id = sample.get('request_id')
        timestamp = pd.to_datetime(sample.get('timestamp'))  # Add timestamp processing
        
        # Process auto eval data
        if sample.get("auto_eval_info"):
            for score in sample["auto_eval_info"][0]["scores"]:
                auto_eval_data.append({
                    'request_id': request_id,
                    'specialty': specialty,
                    'metric': score['metric'],
                    'score': float(score['score']),
                    'justification': score['justification'],
                    'timestamp': timestamp
                })
        
        # Process human eval data
        if sample.get("human_eval_info") and sample["human_eval_info"].get("metric_scores"):
            for score in sample["human_eval_info"]["metric_scores"]:
                if score["score"] is not None:
                    human_eval_data.append({
                        'request_id': request_id,
                        'specialty': specialty,
                        'metric': score['metric'],
                        'score': float(score['score']),
                        'justification': score.get('justification', ''),
                        'timestamp': timestamp
                    })
    
    auto_df = pd.DataFrame(auto_eval_data)
    human_df = pd.DataFrame(human_eval_data)
    
    return auto_df, human_df

def create_specialty_analysis(auto_df, human_df, selected_metrics, selected_specialties):
    """Create specialty-wise analysis visualizations."""
    
    st.subheader("Specialty-wise Analysis")
    
    # Specialty-wise mean scores
    auto_specialty_means = auto_df[auto_df['metric'].isin(selected_metrics)].groupby(
        ['specialty', 'metric'])['score'].mean().reset_index()
    
    # Create heatmap of average scores by specialty
    fig = go.Figure(data=go.Heatmap(
        z=auto_specialty_means.pivot(index='specialty', columns='metric', values='score').values,
        x=selected_metrics,
        y=auto_specialty_means.pivot(index='specialty', columns='metric', values='score').index,
        colorscale='RdYlGn',
        text=np.round(auto_specialty_means.pivot(index='specialty', columns='metric', values='score').values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
    ))
    
    fig.update_layout(
        title="Average Scores by Specialty and Metric",
        height=400 + (len(selected_specialties) * 20)
    )
    st.plotly_chart(fig)
    
    # Box plots for score distribution by specialty
    col1, col2 = st.columns(2)
    
    with col1:
        metric_for_box = st.selectbox("Select metric for distribution", selected_metrics)
        fig = px.box(
            auto_df[auto_df['metric'] == metric_for_box],
            x='specialty',
            y='score',
            title=f'{metric_for_box} Score Distribution by Specialty'
        )
        st.plotly_chart(fig)
    
    with col2:
        # Correlation matrix for selected specialty
        selected_specialty = st.selectbox("Select specialty for correlation analysis", selected_specialties)
        specialty_data = auto_df[auto_df['specialty'] == selected_specialty]
        
        pivot_data = specialty_data.pivot(columns='metric', values='score')
        corr_matrix = pivot_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
        ))
        fig.update_layout(
            title=f"Metric Correlations for {selected_specialty}",
            height=500,
            width=500
        )
        st.plotly_chart(fig)
    
    # Detailed statistics table
    st.subheader("Detailed Statistics by Specialty")
    specialty_stats = auto_df[auto_df['metric'].isin(selected_metrics)].groupby(
        ['specialty', 'metric']).agg({
            'score': ['count', 'mean', 'std', 'min', 'max']
        }).round(3)
    st.dataframe(specialty_stats)

def create_dashboard():
    st.title("Note Evaluation Analytics Dashboard")
    
    # Load data
    file_path = "ap10-may13.json"
    auto_df, human_df = load_and_process_data(file_path)
    
    # Sidebar filters
    st.sidebar.header("Filters")
    selected_metrics = st.sidebar.multiselect(
        "Select Metrics",
        options=sorted(auto_df['metric'].unique()),
        default=['accuracy', 'hallucination', 'usefulness']
    )
    
    selected_specialties = st.sidebar.multiselect(
        "Select Specialties",
        options=sorted(auto_df['specialty'].unique()),
        default=sorted(auto_df['specialty'].unique())
    )
    
    # Filter data based on selections
    auto_df_filtered = auto_df[
        (auto_df['metric'].isin(selected_metrics)) & 
        (auto_df['specialty'].isin(selected_specialties))
    ]
    human_df_filtered = human_df[
        (human_df['metric'].isin(selected_metrics)) & 
        (human_df['specialty'].isin(selected_specialties))
    ]
    
    # create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overall Analysis", 
        "Specialty Analysis", 
        "Auto vs Human Comparison",
        "Detailed Metrics",
        "Weekly Trend Analysis"
    ])
    
    with tab1:
        st.header("Overall Analysis")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Notes Evaluated", len(auto_df['request_id'].unique()))
        with col2:
            st.metric("Notes with Human Evaluation", len(human_df['request_id'].unique()))
        with col3:
            st.metric("Number of Specialties", len(selected_specialties))
        
        # Overall score distributions for both auto and human evals
        col1, col2 = st.columns(2)
        
        with col1:
            fig_auto = px.box(
                auto_df_filtered,
                x='metric',
                y='score',
                color='specialty',
                title='Auto Evaluation Score Distribution'
            )
            st.plotly_chart(fig_auto)
        
        with col2:
            fig_human = px.box(
                human_df_filtered,
                x='metric',
                y='score',
                color='specialty',
                title='Human Evaluation Score Distribution'
            )
            st.plotly_chart(fig_human)
        
        # Mean scores comparison
        st.subheader("Mean Scores Comparison")
        
        # Calculate statistics for both auto and human evals
        auto_means = auto_df_filtered.groupby('metric')['score'].mean().round(2)
        auto_stds = auto_df_filtered.groupby('metric')['score'].std().round(2)
        auto_counts = auto_df_filtered.groupby('metric')['score'].count()
        
        human_means = human_df_filtered.groupby('metric')['score'].mean().round(2)
        human_stds = human_df_filtered.groupby('metric')['score'].std().round(2)
        human_counts = human_df_filtered.groupby('metric')['score'].count()
        
        # Create comparison table
        comparison_data = []
        for metric in selected_metrics:
            comparison_data.append({
                'Metric': metric,
                'Auto Mean': f"{auto_means.get(metric, 0):.2f} ± {auto_stds.get(metric, 0):.2f} (n={auto_counts.get(metric, 0)})",
                'Human Mean': f"{human_means.get(metric, 0):.2f} ± {human_stds.get(metric, 0):.2f} (n={human_counts.get(metric, 0)})",
                'Auto Mean (raw)': auto_means.get(metric, 0),  # For sorting
                'Human Mean (raw)': human_means.get(metric, 0)  # For sorting
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Plot means comparison
        fig_means = go.Figure()
        fig_means.add_trace(go.Bar(
            name='Auto Evaluation',
            x=comparison_df['Metric'],
            y=comparison_df['Auto Mean (raw)'],
            error_y=dict(
                type='data',
                array=auto_stds[selected_metrics],
                visible=True
            )
        ))
        fig_means.add_trace(go.Bar(
            name='Human Evaluation',
            x=comparison_df['Metric'],
            y=comparison_df['Human Mean (raw)'],
            error_y=dict(
                type='data',
                array=human_stds[selected_metrics],
                visible=True
            )
        ))
        
        fig_means.update_layout(
            title='Mean Scores: Auto vs Human Evaluation',
            barmode='group',
            xaxis_title='Metric',
            yaxis_title='Score'
        )
        st.plotly_chart(fig_means)
        
        # Display comparison table
        st.dataframe(
        comparison_df[['Metric', 'Auto Mean', 'Human Mean', 'Auto Mean (raw)']]
        .sort_values('Auto Mean (raw)', ascending=False)
        .drop(columns=['Auto Mean (raw)'])  # Drop the sorting column after sorting
        .set_index('Metric')
)
    
    with tab2:
        create_specialty_analysis(auto_df_filtered, human_df_filtered, selected_metrics, selected_specialties)
    
    with tab3:
        st.header("Auto vs Human Evaluation Comparison")
        
        # Calculate agreement statistics
        agreement_data = []
        for specialty in selected_specialties:
            for metric in selected_metrics:
                # Get scores for the metric and specialty
                auto_subset = auto_df[
                    (auto_df['metric'] == metric) & 
                    (auto_df['specialty'] == specialty)
                ][['request_id', 'score']]
                
                human_subset = human_df[
                    (human_df['metric'] == metric) & 
                    (human_df['specialty'] == specialty)
                ][['request_id', 'score']]
                
                # Merge on request_id to get matching pairs
                merged_df = pd.merge(auto_subset, human_subset, 
                                   on='request_id', 
                                   suffixes=('_auto', '_human'))
                
                if not merged_df.empty:
                    matches = sum(merged_df['score_auto'] == merged_df['score_human'])
                    total = len(merged_df)
                    agreement = (matches / total) * 100 if total > 0 else 0
                    
                    agreement_data.append({
                        'specialty': specialty,
                        'metric': metric,
                        'agreement_rate': agreement,
                        'matches': matches,
                        'total': total,
                        'mean_auto': merged_df['score_auto'].mean(),
                        'mean_human': merged_df['score_human'].mean()
                    })
        
        if agreement_data:
            agreement_df = pd.DataFrame(agreement_data)
            
            # Agreement heatmap
            fig = px.density_heatmap(
                agreement_df,
                x='metric',
                y='specialty',
                z='agreement_rate',
                title='Agreement Rate Heatmap',
                labels={'agreement_rate': 'Agreement Rate (%)'}
            )
            st.plotly_chart(fig)
            
            # Mean difference analysis
            agreement_df['score_diff'] = agreement_df['mean_auto'] - agreement_df['mean_human']
            fig = px.bar(
                agreement_df,
                x='metric',
                y='score_diff',
                color='specialty',
                title='Mean Score Difference (Auto - Human)',
                barmode='group'
            )
            st.plotly_chart(fig)
            
            # Detailed statistics
            st.dataframe(agreement_df.round(2))
    
    with tab4:
        st.header("Detailed Metrics Analysis")
        
        # Select metric for detailed analysis
        selected_metric = st.selectbox("Select metric for detailed analysis", selected_metrics)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Score distribution
            fig = px.histogram(
                auto_df_filtered[auto_df_filtered['metric'] == selected_metric],
                x='score',
                color='specialty',
                title=f'Score Distribution for {selected_metric}',
                marginal='box'
            )
            st.plotly_chart(fig)
        
        with col2:
            # Sample justifications
            st.subheader("Sample Justifications")
            specialty = st.selectbox("Select specialty", selected_specialties)
            samples = auto_df_filtered[
                (auto_df_filtered['metric'] == selected_metric) & 
                (auto_df_filtered['specialty'] == specialty)
            ][['score', 'justification']].head()
            st.dataframe(samples)
        
        # Correlation analysis
        st.subheader("Correlation with Other Metrics")
        pivot_data = auto_df_filtered.pivot_table(
            index='request_id',
            columns='metric',
            values='score'
        )
        corr_matrix = pivot_data.corr()
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            title="Metric Correlations"
        )
        st.plotly_chart(fig)

    with tab5:
        st.header("Trend Analysis")
        
        # Time range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=auto_df_filtered['timestamp'].min().date(),
                min_value=auto_df_filtered['timestamp'].min().date(),
                max_value=auto_df_filtered['timestamp'].max().date()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=auto_df_filtered['timestamp'].max().date(),
                min_value=auto_df_filtered['timestamp'].min().date(),
                max_value=auto_df_filtered['timestamp'].max().date()
            )
        
        # Filter data by date range
        mask = (auto_df_filtered['timestamp'].dt.date >= start_date) & (auto_df_filtered['timestamp'].dt.date <= end_date)
        date_filtered_auto = auto_df_filtered[mask]
        
        mask = (human_df_filtered['timestamp'].dt.date >= start_date) & (human_df_filtered['timestamp'].dt.date <= end_date)
        date_filtered_human = human_df_filtered[mask]
        
        # Weekly aggregation
        weekly_auto = (
            date_filtered_auto
            .set_index('timestamp')
            .groupby(['metric', pd.Grouper(freq='W')])
            ['score']
            .agg(['mean', 'count', 'std'])
            .reset_index()
        )
        
        weekly_human = (
            date_filtered_human
            .set_index('timestamp')
            .groupby(['metric', pd.Grouper(freq='W')])
            ['score']
            .agg(['mean', 'count', 'std'])
            .reset_index()
        )
        
        # Trend visualization
        st.subheader("Weekly Score Trends")
        
        # Metric selector for detailed trend
        metric_for_trend = st.selectbox(
            "Select metric for trend analysis",
            selected_metrics,
            key="trend_metric"
        )
        
        # Create trend plot
        fig = go.Figure()
        
        # Add auto evaluation trend
        auto_metric_data = weekly_auto[weekly_auto['metric'] == metric_for_trend]
        fig.add_trace(go.Scatter(
            x=auto_metric_data['timestamp'],
            y=auto_metric_data['mean'],
            name='Auto Evaluation',
            mode='lines+markers',
            error_y=dict(
                type='data',
                array=auto_metric_data['std'],
                visible=True
            )
        ))
        
        # Add human evaluation trend
        human_metric_data = weekly_human[weekly_human['metric'] == metric_for_trend]
        fig.add_trace(go.Scatter(
            x=human_metric_data['timestamp'],
            y=human_metric_data['mean'],
            name='Human Evaluation',
            mode='lines+markers',
            error_y=dict(
                type='data',
                array=human_metric_data['std'],
                visible=True
            )
        ))
        
        fig.update_layout(
            title=f'Weekly {metric_for_trend} Score Trends',
            xaxis_title='Week',
            yaxis_title='Score',
            hovermode='x unified'
        )
        st.plotly_chart(fig)
        
        # Volume analysis
        st.subheader("Evaluation Volume Trends")
        
        fig_volume = px.bar(
            weekly_auto[weekly_auto['metric'] == metric_for_trend],
            x='timestamp',
            y='count',
            title=f'Weekly Number of Evaluations for {metric_for_trend}'
        )
        st.plotly_chart(fig_volume)
        
        # Specialty-wise trends
        st.subheader("Specialty-wise Trends")
        
        # Add specialty to weekly aggregation
        specialty_weekly = (
            date_filtered_auto
            .set_index('timestamp')
            .groupby(['specialty', 'metric', pd.Grouper(freq='W')])
            ['score']
            .mean()
            .reset_index()
        )
        
        specialty_trend = px.line(
            specialty_weekly[specialty_weekly['metric'] == metric_for_trend],
            x='timestamp',
            y='score',
            color='specialty',
            title=f'Weekly Trends by Specialty for {metric_for_trend}'
        )
        st.plotly_chart(specialty_trend)
        
        # Statistical summary
        st.subheader("Weekly Statistics")
        weekly_stats = weekly_auto[weekly_auto['metric'] == metric_for_trend].round(3)
        st.dataframe(weekly_stats)

if __name__ == "__main__":
    st.set_page_config(page_title="Note Evaluation Analytics", layout="wide")
    create_dashboard()
