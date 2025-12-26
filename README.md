# Global Energy Consumption & Air Pollution Analysis

Interactive dashboard analyzing the relationship between energy consumption and air pollution across 67 countries (2010-2021).

## Live Dashboard

**Access the interactive dashboard here:** [Streamlit App URL will be added after deployment]

## Project Overview

This project uses machine learning (Random Forest Regression + K-Means Clustering) to analyze:
- How energy consumption patterns predict air pollution levels
- Country clusters based on energy-pollution profiles
- Trends in PM2.5, PM10, and NO2 concentrations

## Data Sources

- **Energy Data**: Our World in Data - Energy Dataset
- **Air Quality Data**: WHO Ambient Air Quality Database

## Key Findings

- Per-capita energy consumption is the strongest predictor of pollution (52.7% feature importance)
- 4 distinct country clusters identified:
  - C0: Oil-rich Middle East (high energy, high pollution)
  - C1: Typical countries (moderate energy, moderate pollution)
  - C2: Developed economies (high energy, low pollution)
  - C3: Dense developing nations (medium energy, high pollution)
- China and India classified as outliers due to mega-scale characteristics

## Technologies

- **Python**: pandas, numpy, scikit-learn
- **Visualization**: Plotly, Streamlit
- **Machine Learning**: Random Forest (GridSearchCV), K-Means Clustering

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run dashboard.py
```

## Project Structure

```
data_vis/
├── dashboard.py              # Interactive Streamlit dashboard
├── requirements.txt          # Python dependencies
├── data/
│   └── wrangled/
│       ├── energy_air_quality.csv
│       ├── cluster_assignments.csv
│       ├── regression_coefficients.csv
│       └── regression_performance.csv
└── README.md
```

## Course

Visual Data Science - Report Stage
December 2025
