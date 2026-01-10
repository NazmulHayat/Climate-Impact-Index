Check it out: https://climate-impact-index.streamlit.app !

# 🌍 Global Climate Impact Index

A machine learning-powered web application that predicts climate impact vulnerability for countries worldwide. The project analyzes data from the four most common climate-caused disasters: **floods, droughts, storms, and extreme temperatures**.

## Overview

This analysis uses data derived from natural disasters sourced from EM-DAT, CRED / UCLouvain. We worked with approximately **16 raw data files** containing over **100,000 rows** of data, which were engineered, cleaned, merged, and transformed into a unified dataset. The final processed dataset contains **7,000+ country-year observations** spanning multiple decades.

## Features

- **Interactive World Map**: Visualize climate impact predictions by country
- **2026 Predictions**: ML-powered forecasts for future climate impacts
- **Top 20 Rankings**: Countries with highest predicted climate impact
- **Country Deep-Dive**: Explore individual country trends and hazard distributions
- **Feature Importance**: Understand which factors drive predictions

## Data Pipeline

The project involved extensive data engineering work:

1. **Raw Data Collection**: 16 CSV files from EM-DAT covering:
   - Per-capita impact data (per 100,000 population)
   - Total count data (deaths and affected populations)
   - Economic damage data (% of GDP)

2. **Data Cleaning Pipelines**: Multiple Jupyter notebooks were used to clean and process the data:
   - `data_cleanup_pipeline_for_capita.ipynb` - Per-capita data processing
   - `data_cleanup_pipeline_for_total.ipynb` - Total count data processing
   - `flood_data_clean.ipynb` - Flood-specific data cleaning
   - `drought_data_clean.ipynb` - Drought-specific data cleaning
   - `economic_data_cleanup.ipynb` - Economic damage data processing
   - `Final_cleaup.ipynb` - Final data integration and validation

3. **Feature Engineering**: Created 18 features including:
   - Individual hazard impacts (flood, drought, storm, extreme temperature)
   - Temporal trends and rolling averages
   - Country-specific baselines
   - Log-transformed absolute measures

4. **Final Dataset**: Processed data stored in `data/processed/` directory

## Climate Impact Index (CLI)

The CLI combines four normalized components:

- **Human Burden (50%)**: Total people affected by disasters using 3-year rolling averages
- **Persistence (25%)**: Temporal consistency of climate impacts
- **Climate Intensity (15%)**: Per-capita normalized impact across all hazards
- **Structural Vulnerability (10%)**: Historical average impact per country

*The component structure and weighting percentages were inspired by Germanwatch Climate Risk Index, though the specific implementation has been adapted and improved for this analysis.*

## Machine Learning Model

- **Algorithm**: Random Forest Regressor
- **Performance**: Cross-validated R² score of **0.55**
- **Features**: 18 engineered features
- **Training**: Data through 2023, predicting 2026
- **Model Comparison**: Tested Random Forest, Gradient Boosting, and XGBoost - Random Forest performed best

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                          # Main Streamlit application
├── ml_model_cont_improved.py       # ML model training and prediction
├── create_continuous_analysis_data.py  # Data processing script
├── data/
│   ├── raw/                        # Raw data files (16 CSV files)
│   └── processed/                  # Cleaned and processed datasets
├── notebook/                       # Data cleaning pipeline notebooks
└── requirements.txt                # Python dependencies
```

## Data Sources

- **EM-DAT**: The Emergency Events Database
- **CRED / UCLouvain**: Centre for Research on the Epidemiology of Disasters

## Technologies Used

- Python
- Streamlit (Web Interface)
- Scikit-learn (Machine Learning)
- Pandas (Data Processing)
- Plotly (Visualizations)
- NumPy (Numerical Computing)

## Author

Developed by Nazmul Hayat
