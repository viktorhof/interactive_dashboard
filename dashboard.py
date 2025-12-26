"""
Interactive Energy-Pollution Analysis Dashboard
Visual Data Science Project - Report Stage
Created: December 2025
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    layout="wide",
    page_title="Energy-Pollution Analysis Dashboard",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_data():
    """Load and prepare all datasets"""
    # Main dataset
    df = pd.read_csv('data/wrangled/energy_air_quality.csv')

    # Cluster assignments
    clusters = pd.read_csv('data/wrangled/cluster_assignments.csv')

    # Merge cluster info
    df_full = df.merge(clusters[['country', 'cluster', 'cluster_label']],
                       on='country', how='left')

    # Convert per_capita_energy to MWh for readability
    if 'per_capita_energy' in df_full.columns:
        df_full['per_capita_energy_mwh'] = df_full['per_capita_energy'] * 1_000_000

    # Regression results
    reg_coef = pd.read_csv('data/wrangled/regression_coefficients.csv')
    reg_perf = pd.read_csv('data/wrangled/regression_performance.csv')

    # Country-level aggregates (for some visualizations)
    country_avg = df_full.groupby('country').agg({
        'pm25_concentration': 'mean',
        'pm10_concentration': 'mean',
        'no2_concentration': 'mean',
        'per_capita_energy_mwh': 'mean',
        'primary_energy_consumption': 'mean',
        'population': 'mean',
        'gdp': 'mean',
        'cluster': 'first',
        'cluster_label': 'first'
    }).reset_index()

    return df_full, country_avg, reg_coef, reg_perf

# Load data
df, country_avg, reg_coef, reg_perf = load_data()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_country_iso_mapping():
    """Map country names to ISO alpha-3 codes for choropleth"""
    # Common mappings for countries in dataset
    mapping = {
        'Afghanistan': 'AFG', 'Albania': 'ALB', 'Algeria': 'DZA', 'Argentina': 'ARG',
        'Armenia': 'ARM', 'Australia': 'AUS', 'Austria': 'AUT', 'Azerbaijan': 'AZE',
        'Bahrain': 'BHR', 'Bangladesh': 'BGD', 'Belarus': 'BLR', 'Belgium': 'BEL',
        'Bolivia': 'BOL', 'Bosnia And Herzegovina': 'BIH', 'Brazil': 'BRA',
        'Bulgaria': 'BGR', 'Canada': 'CAN', 'Chile': 'CHL', 'China': 'CHN',
        'Colombia': 'COL', 'Croatia': 'HRV', 'Cyprus': 'CYP', 'Czechia': 'CZE',
        'Denmark': 'DNK', 'Ecuador': 'ECU', 'Egypt': 'EGY', 'Estonia': 'EST',
        'Finland': 'FIN', 'France': 'FRA', 'Georgia': 'GEO', 'Germany': 'DEU',
        'Greece': 'GRC', 'Hungary': 'HUN', 'India': 'IND', 'Indonesia': 'IDN',
        'Iran': 'IRN', 'Iraq': 'IRQ', 'Ireland': 'IRL', 'Israel': 'ISR',
        'Italy': 'ITA', 'Japan': 'JPN', 'Jordan': 'JOR', 'Kazakhstan': 'KAZ',
        'Kuwait': 'KWT', 'Latvia': 'LVA', 'Lithuania': 'LTU', 'Malaysia': 'MYS',
        'Mexico': 'MEX', 'Mongolia': 'MNG', 'Morocco': 'MAR', 'Nepal': 'NPL',
        'Netherlands': 'NLD', 'New Zealand': 'NZL', 'Norway': 'NOR', 'Oman': 'OMN',
        'Pakistan': 'PAK', 'Peru': 'PER', 'Philippines': 'PHL', 'Poland': 'POL',
        'Portugal': 'PRT', 'Qatar': 'QAT', 'Romania': 'ROU', 'Russia': 'RUS',
        'Saudi Arabia': 'SAU', 'Serbia': 'SRB', 'Slovakia': 'SVK', 'Slovenia': 'SVN',
        'South Africa': 'ZAF', 'South Korea': 'KOR', 'Spain': 'ESP', 'Sweden': 'SWE',
        'Switzerland': 'CHE', 'Thailand': 'THA', 'Turkey': 'TUR', 'Ukraine': 'UKR',
        'United Arab Emirates': 'ARE', 'United Kingdom': 'GBR', 'United States': 'USA',
        'Uruguay': 'URY', 'Uzbekistan': 'UZB', 'Vietnam': 'VNM'
    }
    return mapping

def get_cluster_colors():
    """Consistent cluster colors across all visualizations"""
    return {
        0: '#FF6B6B',  # Red - Oil-rich Middle East
        1: '#95D5B2',  # Green - Typical countries
        2: '#4ECDC4',  # Teal - Developed economies
        3: '#C77DFF',  # Purple - Dense developing
        -1: '#FFD166'  # Yellow - Outliers
    }

def create_world_map(data, pollutant_col, iso_mapping):
    """Create choropleth world map"""
    # Add ISO codes
    data_with_iso = data.copy()
    data_with_iso['iso_alpha'] = data_with_iso['country'].map(iso_mapping)

    fig = px.choropleth(
        data_with_iso,
        locations='iso_alpha',
        color=pollutant_col,
        hover_name='country',
        hover_data={
            'iso_alpha': False,
            pollutant_col: ':.1f',
            'per_capita_energy_mwh': ':.1f',
            'cluster_label': True
        },
        color_continuous_scale='RdYlGn_r',
        labels={
            'pm25_concentration': 'PM2.5 (μg/m³)',
            'pm10_concentration': 'PM10 (μg/m³)',
            'no2_concentration': 'NO2 (μg/m³)',
            'per_capita_energy_mwh': 'Energy (MWh/person)'
        },
        projection='natural earth'
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        height=350,
        title=dict(text='Global Distribution', font=dict(size=14, weight='bold'))
    )

    return fig

def create_energy_scatter(data, pollutant_col, cluster_colors):
    """Create energy vs pollution scatter plot"""

    fig = px.scatter(
        data,
        x='per_capita_energy_mwh',
        y=pollutant_col,
        color='cluster',
        size='population',
        hover_name='country',
        hover_data={
            'per_capita_energy_mwh': ':.1f',
            pollutant_col: ':.1f',
            'cluster_label': True,
            'year': True,
            'cluster': False,
            'population': ':,.0f'
        },
        color_discrete_map=cluster_colors,
        labels={
            'per_capita_energy_mwh': 'Per-Capita Energy (MWh/person)',
            'pm25_concentration': 'PM2.5 (μg/m³)',
            'pm10_concentration': 'PM10 (μg/m³)',
            'no2_concentration': 'NO2 (μg/m³)'
        },
        opacity=0.6
    )

    fig.update_layout(
        title=dict(text='Energy Consumption vs Air Pollution', font=dict(size=14, weight='bold')),
        height=350,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
        legend=dict(title='Cluster', orientation='v', x=1.02, y=1)
    )

    return fig

def create_time_series(data, selected_countries, pollutant_col, cluster_colors):
    """Create time series for selected countries"""

    if not selected_countries:
        # Show top 5 polluters if no selection
        top_countries = data.groupby('country')[pollutant_col].mean().nlargest(5).index.tolist()
        selected_countries = top_countries

    # Limit to 10 countries for readability
    selected_countries = selected_countries[:10]

    filtered = data[data['country'].isin(selected_countries)]

    fig = go.Figure()

    for country in selected_countries:
        country_data = filtered[filtered['country'] == country].sort_values('year')
        cluster = country_data['cluster'].iloc[0] if len(country_data) > 0 else 1

        fig.add_trace(go.Scatter(
            x=country_data['year'],
            y=country_data[pollutant_col],
            name=country,
            mode='lines+markers',
            line=dict(color=cluster_colors.get(cluster, '#999999'), width=2),
            marker=dict(size=6)
        ))

    # Add global average as reference
    global_avg = data.groupby('year')[pollutant_col].mean()
    fig.add_trace(go.Scatter(
        x=global_avg.index,
        y=global_avg.values,
        name='Global Average',
        mode='lines',
        line=dict(color='#FF6B35', width=3, dash='dash'),
        opacity=0.9
    ))

    pollutant_labels = {
        'pm25_concentration': 'PM2.5 (μg/m³)',
        'pm10_concentration': 'PM10 (μg/m³)',
        'no2_concentration': 'NO2 (μg/m³)'
    }

    fig.update_layout(
        title=dict(
            text=f'Pollution Trends Over Time ({len(selected_countries)} countries)',
            font=dict(size=14, weight='bold')
        ),
        xaxis_title='Year',
        yaxis_title=pollutant_labels.get(pollutant_col, pollutant_col),
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=-0.2, xanchor='center', x=0.5)
    )

    return fig

def create_cluster_heatmap(data, cluster_colors):
    """Create cluster comparison heatmap with z-scores"""

    # Define features to compare (only using available columns)
    features = [
        'per_capita_energy_mwh',
        'primary_energy_consumption',
        'population',
        'pm25_concentration',
        'pm10_concentration',
        'no2_concentration'
    ]

    # Calculate mean by cluster
    cluster_means = data.groupby('cluster')[features].mean()

    # Calculate z-scores
    z_scores = (cluster_means - cluster_means.mean()) / cluster_means.std()

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z_scores.T.values,
        x=[f'C{i}' if i >= 0 else 'Out' for i in z_scores.index],
        y=['Per-capita Energy', 'Total Energy', 'Population', 'PM2.5', 'PM10', 'NO2'],
        colorscale='RdYlGn_r',
        zmid=0,
        zmin=-2,
        zmax=2,
        text=np.round(z_scores.T.values, 1),
        texttemplate='%{text}σ',
        textfont={"size": 10},
        colorbar=dict(title='Z-score')
    ))

    fig.update_layout(
        title=dict(text='Cluster Characteristics (Z-scores)', font=dict(size=14, weight='bold')),
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig

def create_top_polluters_bar(data, pollutant_col, selected_countries, n=15):
    """Create bar chart of top polluters"""

    top_data = data.groupby('country')[pollutant_col].mean().nlargest(n).reset_index()
    top_data['selected'] = top_data['country'].isin(selected_countries)

    fig = px.bar(
        top_data,
        x=pollutant_col,
        y='country',
        orientation='h',
        color='selected',
        color_discrete_map={True: '#FF6B6B', False: '#999999'},
        labels={
            'pm25_concentration': 'PM2.5 (μg/m³)',
            'pm10_concentration': 'PM10 (μg/m³)',
            'no2_concentration': 'NO2 (μg/m³)',
            'selected': 'Selected'
        }
    )

    pollutant_labels = {
        'pm25_concentration': 'PM2.5',
        'pm10_concentration': 'PM10',
        'no2_concentration': 'NO2'
    }

    fig.update_layout(
        title=dict(
            text=f'Top {n} Countries by {pollutant_labels.get(pollutant_col, "Pollution")}',
            font=dict(size=14, weight='bold')
        ),
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )

    return fig

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'selected_countries' not in st.session_state:
    st.session_state.selected_countries = []

# ============================================================================
# SIDEBAR - FILTERS & CONTROLS
# ============================================================================

with st.sidebar:
    st.title("Filters & Controls")

    st.markdown("---")

    # Country selection
    st.subheader("Country Selection")

    # Get available countries with valid data (at least one pollutant measurement)
    valid_countries = df[
        df['pm25_concentration'].notna() |
        df['pm10_concentration'].notna() |
        df['no2_concentration'].notna()
    ]['country'].unique()
    all_countries = sorted(valid_countries)

    # Multi-select for countries
    selected_countries = st.multiselect(
        "Select countries to analyze:",
        options=all_countries,
        default=st.session_state.selected_countries,
        help="Select up to 10 countries for comparison"
    )

    # Update session state
    st.session_state.selected_countries = selected_countries[:10]  # Limit to 10

    # Quick selection buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Top 5 Polluters"):
            # Only select from countries with cluster assignments (used in analysis)
            clustered_countries = country_avg[country_avg['cluster'].notna()]
            top_5 = clustered_countries.nlargest(5, 'pm25_concentration')['country'].tolist()
            st.session_state.selected_countries = top_5
            st.rerun()

    with col_btn2:
        if st.button("Clear All"):
            st.session_state.selected_countries = []
            st.rerun()

    st.markdown("---")

    # Cluster filter
    st.subheader("Cluster Filter")

    cluster_options = {
        0: "C0: Oil-rich Middle East",
        1: "C1: Typical countries",
        2: "C2: Developed economies",
        3: "C3: Dense developing",
        -1: "Outliers (China, India)"
    }

    selected_clusters = []
    for cluster_id, cluster_name in cluster_options.items():
        if st.checkbox(cluster_name, value=True, key=f'cluster_{cluster_id}'):
            selected_clusters.append(cluster_id)

    st.markdown("---")

    # Year range
    st.subheader("Time Period")
    year_range = st.slider(
        "Select year range:",
        min_value=int(df['year'].min()),
        max_value=int(df['year'].max()),
        value=(int(df['year'].min()), int(df['year'].max())),
        help="Filter data by year range"
    )

    st.markdown("---")

    # Pollutant selection
    st.subheader("Pollutant Type")
    pollutant = st.radio(
        "Select pollutant to display:",
        options=['PM2.5', 'PM10', 'NO2'],
        index=0,
        help="PM2.5 = Fine particles, PM10 = Coarse particles, NO2 = Nitrogen dioxide"
    )

    pollutant_col_map = {
        'PM2.5': 'pm25_concentration',
        'PM10': 'pm10_concentration',
        'NO2': 'no2_concentration'
    }
    pollutant_col = pollutant_col_map[pollutant]

    st.markdown("---")

    # Summary statistics
    st.subheader("Summary")

    # Filter data based on selections
    filtered_df = df[
        (df['cluster'].isin(selected_clusters)) &
        (df['year'].between(year_range[0], year_range[1]))
    ]

    if selected_countries:
        display_df = filtered_df[filtered_df['country'].isin(selected_countries)]
    else:
        display_df = filtered_df

    st.metric("Total Countries", len(display_df['country'].unique()))
    st.metric("Observations", len(display_df))

    if len(display_df) > 0:
        avg_pollution = display_df[pollutant_col].mean()
        st.metric(f"Avg {pollutant}", f"{avg_pollution:.1f} μg/m³")

        avg_energy = display_df['per_capita_energy_mwh'].mean()
        st.metric("Avg Energy", f"{avg_energy:.1f} MWh/p")

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

# Header
st.title("Global Energy Consumption & Air Pollution Analysis")
st.markdown("**Interactive Dashboard** | 67 Countries | 2010-2021 | Random Forest + K-Means Clustering")

st.markdown("---")

# Get data for visualizations
cluster_colors = get_cluster_colors()
iso_mapping = get_country_iso_mapping()

# Prepare data for country-level visualizations
country_display = country_avg[country_avg['cluster'].isin(selected_clusters)].copy()

# ============================================================================
# ROW 1: MAP + SCATTER (40% of screen height)
# ============================================================================

row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    # VIZ 1: World Map
    map_fig = create_world_map(country_display, pollutant_col, iso_mapping)
    st.plotly_chart(map_fig, use_container_width=True, key='world_map')

with row1_col2:
    # VIZ 2: Energy vs Pollution Scatter

    scatter_display = filtered_df.copy()
    if selected_countries:
        # Highlight selected countries
        scatter_display['opacity'] = scatter_display['country'].apply(
            lambda x: 1.0 if x in selected_countries else 0.3
        )

    scatter_fig = create_energy_scatter(
        scatter_display,
        pollutant_col,
        cluster_colors
    )
    st.plotly_chart(scatter_fig, use_container_width=True, key='scatter_plot')

# ============================================================================
# ROW 2: TIME SERIES (30% of screen height)
# ============================================================================

st.markdown("---")

# VIZ 3: Time Series
time_series_fig = create_time_series(
    filtered_df,
    selected_countries,
    pollutant_col,
    cluster_colors
)

st.plotly_chart(time_series_fig, use_container_width=True, key='time_series')

# ============================================================================
# ROW 3: HEATMAP + BAR CHART + MODEL DETAILS (30% of screen height)
# ============================================================================

st.markdown("---")

row3_col1, row3_col2, row3_col3 = st.columns([1.2, 1.2, 1])

with row3_col1:
    # VIZ 4: Cluster Heatmap
    heatmap_fig = create_cluster_heatmap(filtered_df, cluster_colors)
    st.plotly_chart(heatmap_fig, use_container_width=True, key='heatmap')

with row3_col2:
    # VIZ 5: Top Polluters Bar
    bar_fig = create_top_polluters_bar(
        country_display,
        pollutant_col,
        selected_countries,
        n=15
    )
    st.plotly_chart(bar_fig, use_container_width=True, key='bar_chart')

with row3_col3:
    # Expandable Model Details
    with st.expander("Model Details", expanded=False):
        st.subheader("Random Forest Performance")

        # Model performance table
        perf_display = reg_perf[['target', 'r2_cv_mean', 'rmse_cv_mean']].copy()
        perf_display.columns = ['Pollutant', 'CV R²', 'CV RMSE']
        perf_display['Pollutant'] = perf_display['Pollutant'].str.replace('_concentration', '')
        perf_display['CV R²'] = perf_display['CV R²'].apply(lambda x: f"{x:.3f}")
        perf_display['CV RMSE'] = perf_display['CV RMSE'].apply(lambda x: f"{x:.2f}")

        st.dataframe(perf_display, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.subheader("Top 5 Feature Importance")

        # Feature importance
        top_features = reg_coef.groupby('feature')['importance'].mean().nlargest(5)

        for feature, importance in top_features.items():
            st.metric(
                label=feature.replace('_', ' ').title(),
                value=f"{importance*100:.1f}%"
            )

        st.markdown("---")
        st.caption("Model: Random Forest with GridSearchCV (108 parameter combinations)")
        st.caption("K-Means: 4 clusters + outlier detection")
        st.caption("Cross-validation: 5-fold CV")

