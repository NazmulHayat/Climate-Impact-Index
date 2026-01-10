import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    layout="wide",
    page_title="Global Climate Impact Index",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        font-size: 2.5rem;
        font-weight: 300;
        letter-spacing: -0.02em;
        margin-bottom: 1rem;
    }
    .stPlotlyChart {
        margin: 0 auto;
    }
    footer {
        margin-top: 4rem;
        padding-top: 2rem;
        border-top: 1px solid #e0e0e0;
        text-align: center;
        color: #666;
        font-size: 0.9rem;
        line-height: 1.8;
    }
    .data-info {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌍 Global Climate Impact Index")

st.markdown("This analysis focuses on the **four most common climate-caused disasters**: floods, storms, extreme temperatures, and droughts. We worked with data spanning **1900-2025**.")

df = pd.read_csv("data/processed/analysis_data_set.csv")

hazard_cols = [
    "flood_impact",
    "drought_impact",
    "storms_impact",
    "extreme_temp_impact"
]

hazard_color_map = {
    "Floods": "#1f77b4",
    "Droughts": "#d62728",
    "Storms": "#9467bd",
    "Extreme Temperature": "#ff7f0e"
}

dominant_color_map = {
    "Flood-prone": "#1f77b4",
    "Drought-prone": "#d62728",
    "Storm-prone": "#9467bd",
    "Extreme-temperature-prone": "#ff7f0e"
}

def rebase_impact(df, col="climate_impact_index"):
    return df[col] - df[col].min()

periods = sorted(df["period"].unique())
max_period = min(2020, int(max(periods)))
min_period = max(1960, int(min(periods)))

selected_period = st.slider(
    "Select 5-year period",
    min_value=min_period,
    max_value=max_period,
    step=5,
    value=max_period
)

df_p = df[df["period"] == selected_period].copy()
df_p["impact_rebased"] = rebase_impact(df_p)

period_end = selected_period + 4

df_p["impact_rank"] = df_p["impact_rebased"].rank(method='dense', ascending=False).astype(int)

def get_risk_category(rank, total_countries):
    if rank <= 10:
        return "1-10"
    elif rank <= 20:
        return "11-20"
    elif rank <= 50:
        return "21-50"
    elif rank <= 100:
        return "51-100"
    else:
        return ">100"

df_p["risk_category"] = df_p["impact_rank"].apply(lambda x: get_risk_category(x, len(df_p)))

category_colors = {
    "1-10": "#8B0000",
    "11-20": "#CC0000",
    "21-50": "#FF4444",
    "51-100": "#FF9999",
    ">100": "#FFCCCC"
}

fig_map = px.choropleth(
    df_p,
    locations="country",
    locationmode="country names",
    color="risk_category",
    hover_name="country",
    hover_data={
        "impact_rank": True,
        "impact_rebased": ":.3f"
    },
    color_discrete_map=category_colors,
    category_orders={"risk_category": ["1-10", "11-20", "21-50", "51-100", ">100"]},
    title=f"Climate Impact Severity ({selected_period} - {period_end})"
)

fig_map.update_traces(
    hovertemplate="<b>%{hovertext}</b><br>" +
                  "Rank: %{customdata[0]}<br>" +
                  "Impact Severity: %{customdata[1]:.3f}<br>" +
                  "<extra></extra>"
)

fig_map.update_layout(
    margin=dict(l=0, r=0, t=60, b=0),
    height=700,
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type='natural earth',
        bgcolor='rgba(0,0,0,0)'
    ),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    title_font_size=18
)

st.plotly_chart(fig_map, use_container_width=True)

st.caption(
    "Impact Severity reflects relative, per-capita human impact from climate disasters. "
    "Darker red indicates higher relative impact. "
    "This is a comparative measure — not total damage, hazard intensity, or event frequency."
)

st.subheader("Dominant Climate Hazard by Country")

country_features = (
    df.groupby("country")[hazard_cols]
    .mean()
    .abs()
    .reset_index()
)

country_features["hazard_prone"] = country_features[hazard_cols].idxmax(axis=1)

country_features["hazard_prone"] = country_features["hazard_prone"].map({
    "flood_impact": "Flood-prone",
    "drought_impact": "Drought-prone",
    "storms_impact": "Storm-prone",
    "extreme_temp_impact": "Extreme-temperature-prone"
})

fig_dom = px.choropleth(
    country_features,
    locations="country",
    locationmode="country names",
    color="hazard_prone",
    hover_name="country",
    title="Dominant Climate Hazard (Peak Human Impact)",
    color_discrete_map=dominant_color_map
)

fig_dom.update_layout(
    margin=dict(l=0, r=0, t=50, b=0),
    height=600,
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type='natural earth',
        bgcolor='rgba(0,0,0,0)'
    ),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

st.plotly_chart(fig_dom, use_container_width=True)

st.caption(
    "Countries are labeled by the hazard with the highest average impact severity per period. "
    "This highlights peak-risk hazards, not cumulative contribution or hazard frequency."
)

st.subheader("Country Deep-Dive")

country = st.selectbox(
    "Select a country",
    sorted(df["country"].dropna().unique())
)

df_c = (
    df[df["country"] == country]
    .sort_values("period")
    .copy()
)

df_c["impact_rebased"] = rebase_impact(df_c)

col1, col2 = st.columns(2)
with col1:
    fig_trend = px.line(
        df_c,
        x="period",
        y="impact_rebased",
        markers=True,
        title=f"Climate Impact Severity Over Time — {country}",
        labels={
            "period": "Year (5-year periods)",
            "impact_rebased": "Impact Severity (relative)"
        }
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with col2:
    fig_res = px.line(
        df_c,
        x="period",
        y="economic_damage_pct_gdp",
        markers=True,
        title=f"Economic Damage from Disasters Over Time — {country}",
        labels={
            "period": "Year (5-year periods)",
            "economic_damage_pct_gdp": "Economic Damage (% of GDP)"
        }
    )

    fig_res.update_traces(line=dict(color="red", width=3))
    fig_res.update_yaxes(rangemode="tozero")

    st.plotly_chart(fig_res, use_container_width=True)

total_df = pd.read_csv("data/processed/final_total_data.csv")
total_df_c = total_df[total_df["country"] == country]

hazard_cols_total = {
    "Floods": "flood_total_affected",
    "Droughts": "drought_total_affected",
    "Storms": "storm_total_affected",
    "Extreme Temperature": "extreme_temp_total_affected",
}

hazard_contrib = {
    hazard: total_df_c[col].sum()
    for hazard, col in hazard_cols_total.items()
}

bar_df = (
    pd.DataFrame.from_dict(hazard_contrib, orient="index", columns=["Contribution"])
      .reset_index()
      .rename(columns={"index": "Hazard"})
)

bar_df["Contribution"] = bar_df[ "Contribution"] / bar_df["Contribution"].sum()

fig_bar = px.bar(
    bar_df,
    x="Hazard",
    y="Contribution",
    text_auto=".0%",
    title=f"Distribution of People Affected by Disaster Type - {country}",
    color="Hazard",
    color_discrete_map=hazard_color_map
)

st.plotly_chart(fig_bar, use_container_width=True)

st.caption(
    "Bar chart shows each disaster's share of total affected people "
    "in the selected country over the full period."
)

st.divider()
st.subheader("🎯 Climate Impact Predictions for 2026")

from ml_model_cont_improved import train_and_predict, create_cli_map

@st.cache_data
def load_improved_model():
    return train_and_predict()

with st.spinner("Training model and generating predictions... This may take a minute."):
    model_improved, pred_df, feature_importance, features = load_improved_model()

if pred_df.empty:
    st.error("No prediction data available. Please check the data.")
    st.stop()

fig_cli, pred_df_with_rank = create_cli_map(pred_df)
st.plotly_chart(fig_cli, use_container_width=True)

st.caption(
    "**CLI Formula:** 0.50 × Human Burden + 0.25 × Persistence + 0.15 × Climate Intensity + 0.10 × Structural Vulnerability. "
    "**Lower rank number = Higher predicted impact** (darker red indicates higher predicted impact). "
    "Please refer to the Methodology section below for detailed information."
)

st.subheader("Top 20 Countries by Climate Impact (2026 Prediction)")
top_20_improved = pred_df_with_rank.nsmallest(20, "CLI_rank")[
    ["country", "CLI_rank", "CLI", "risk_category"]
].copy()
top_20_improved = top_20_improved.sort_values("CLI_rank", ascending=True)
top_20_improved = top_20_improved.rename(columns={
    "country": "Country",
    "CLI_rank": "Rank",
    "CLI": "CLI Score",
    "risk_category": "Risk Category"
})
top_20_improved["Rank"] = top_20_improved["Rank"].astype(int)

max_rank = top_20_improved["Rank"].max()
min_rank = top_20_improved["Rank"].min()

try:
    if max_rank > min_rank:
        st.dataframe(
            top_20_improved.style.format({
                "CLI Score": "{:.3f}"
            }).background_gradient(
                subset=["Rank"], 
                cmap="Reds_r"
            ),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.dataframe(
            top_20_improved.style.format({
                "CLI Score": "{:.3f}"
            }),
            use_container_width=True,
            hide_index=True
        )
except ImportError:
    # Fallback if matplotlib is not available
    st.dataframe(
        top_20_improved.style.format({
            "CLI Score": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

st.subheader("Feature Importance")
fig_feat = px.bar(
    feature_importance.head(10),
    x='importance',
    y='feature',
    orientation='h',
    title='Top 10 Most Important Features',
    labels={'importance': 'Importance', 'feature': 'Feature'}
)
fig_feat.update_layout(
    height=400, 
    yaxis={'categoryorder': 'total ascending'},
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)
st.plotly_chart(fig_feat, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("### Data Sources & Methodology")

st.markdown("**Data Sources:** This analysis uses data derived from natural disasters including floods, droughts, storms, and extreme temperatures. We worked with approximately **16 raw data files** from EM-DAT, CRED / UCLouvain, analyzing over **100,000 rows of data** in total. These raw files were engineered, cleaned, merged, and transformed into a unified dataset tailored for our analysis. The final processed dataset contains **7,000+ country-year observations** spanning multiple decades.")

with st.expander("**Methodology Details**"):
    st.markdown("""
    **Climate Impact Index (CLI) Components:**
    
    The CLI combines four normalized components. The component structure and weighting percentages were inspired by Germanwatch Climate Risk Index, though the specific implementation and feature engineering have been adapted and improved for this analysis:
    - **Human Burden (50%)**: Total people affected by disasters (deaths and affected populations), using 3-year rolling averages and log transformation for fair comparison across countries
    - **Persistence (25%)**: Temporal consistency of climate impacts, combining 3-year averages and previous year's impact to identify sustained vs. isolated events
    - **Climate Intensity (15%)**: Per-capita normalized impact across floods, droughts, storms, and extreme temperatures
    - **Structural Vulnerability (10%)**: Historical average impact per country, reflecting baseline vulnerability factors
    
    **Machine Learning Model:**
    
    Multiple machine learning algorithms were tested including Random Forest, Gradient Boosting, and XGBoost. Random Forest achieved the best cross-validated R² score of **0.55** and was selected for final predictions. The model is trained on historical data through 2023 and uses 18 features including individual hazard impacts, temporal trends, country baselines, and rolling statistics. The predicted per-capita climate impact for 2026 is converted to CLI scores using the weighted component formula above.
    
    **Ranking:** Lower rank numbers indicate higher predicted climate impact. Countries require at least 5 years of historical data for predictions. Results are comparative indicators based on historical patterns, not absolute forecasts.
    """)

st.markdown("<div style='text-align: right; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e0e0e0;'><p style='color: #666; font-size: 0.9rem; font-style: italic;'>Developed by Nazmul Hayat</p></div>", unsafe_allow_html=True)
