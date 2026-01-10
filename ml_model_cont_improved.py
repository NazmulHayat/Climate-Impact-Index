import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


def filter_valid_countries(df):
    invalid_countries = [
        'Asia', 'Africa', 'Europe', 'North America', 'South America', 'Oceania',
        'USSR', 'Soviet Union', 'Czechoslovakia', 'Yugoslavia',
        'World', 'Low-income countries', 'Lower-middle-income countries',
        'Upper-middle-income countries', 'High-income countries',
        'European Union', 'EU'
    ]
    return df[~df['country'].isin(invalid_countries)].copy()


def calculate_trend_5yr(series):
    result = []
    for i in range(len(series)):
        start_idx = max(0, i - 4)
        window = series.iloc[start_idx:i+1].values
        valid_window = window[~np.isnan(window)]
        if len(valid_window) >= 2:
            x = np.arange(len(valid_window))
            trend = np.polyfit(x, valid_window, 1)[0]
            result.append(trend)
        else:
            result.append(0.0)
    return pd.Series(result, index=series.index)


def create_cli_map(pred_df):
    if pred_df.empty or pred_df["CLI"].isna().all():
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        fig.update_layout(title="Climate Impact Index (CLI): Most Affected Countries (2026 Prediction)")
        return fig, pred_df
    
    pred_df = pred_df.copy()
    pred_df = pred_df[pred_df["CLI"].notna()].copy()
    
    if pred_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        fig.update_layout(title="Climate Impact Index (CLI): Most Affected Countries (2026)")
        return fig, pred_df
    
    pred_df["CLI_rank"] = pred_df["CLI"].rank(method='dense', ascending=False).astype(int)
    
    def get_risk_category(rank):
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
    
    pred_df["risk_category"] = pred_df["CLI_rank"].apply(get_risk_category)
    
    category_colors = {
        "1-10": "#8B0000",
        "11-20": "#CC0000",
        "21-50": "#FF4444",
        "51-100": "#FF9999",
        ">100": "#FFCCCC"
    }
    
    fig = px.choropleth(
        pred_df,
        locations="country",
        locationmode="country names",
        color="risk_category",
        hover_name="country",
        hover_data={
            "CLI_rank": True,
            "CLI": ":.3f"
        },
        color_discrete_map=category_colors,
        category_orders={"risk_category": ["1-10", "11-20", "21-50", "51-100", ">100"]},
        title="Climate Impact Index (CLI): Most Affected Countries (2026)",
        labels={"risk_category": "Risk Category"}
    )
    
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>" +
                      "Rank: %{customdata[0]}<br>" +
                      "CLI: %{customdata[1]:.3f}<br>" +
                      "<extra></extra>"
    )
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=60, b=0),
        height=700,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        ),
        title_font_size=18
    )
    
    return fig, pred_df


def train_and_predict():
    df = pd.read_csv("data/processed/analysis_data_set_continuous.csv")
    df = filter_valid_countries(df)
    df = df.sort_values(["country", "period"])

    df["impact_rebased_next"] = df.groupby("country")["impact_rebased"].shift(-1)

    train_df_temp = df[df["period"] <= 2023].copy()
    
    country_mean_impact = train_df_temp.groupby("country")["impact_rebased"].mean()
    df["country_mean_impact"] = df["country"].map(country_mean_impact)
    df["country_mean_impact"].fillna(train_df_temp["impact_rebased"].mean(), inplace=True)

    df["impact_lag1"] = df.groupby("country")["impact_rebased"].shift(1)
    df["impact_lag1"].fillna(df["country_mean_impact"], inplace=True)

    df["log_total_affected"] = np.log1p(df["total_affected"].fillna(0))
    df["log_total_death"] = np.log1p(df["total_deaths"].fillna(0))

    df["total_affected_3yr_avg"] = df.groupby("country")["total_affected"].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean().fillna(0)
    )
    df["log_total_affected_3yr_avg"] = np.log1p(df["total_affected_3yr_avg"])

    df["total_death_3yr_avg"] = df.groupby("country")["total_deaths"].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean().fillna(0)
    )
    df["log_total_death_3yr_avg"] = np.log1p(df["total_death_3yr_avg"])

    df["impact_3yr_avg"] = df.groupby("country")["climate_impact_index"].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )

    df["impact_trend_5yr"] = df.groupby("country")["climate_impact_index"].apply(
        calculate_trend_5yr
    ).reset_index(level=0, drop=True)

    df["absolute_impact_trend"] = df.groupby("country")["log_total_affected"].apply(
        calculate_trend_5yr
    ).reset_index(level=0, drop=True)

    df["impact_std_5yr"] = df.groupby("country")["climate_impact_index"].transform(
        lambda x: x.rolling(window=5, min_periods=1).std().fillna(0)
    )

    country_long_term_mean = train_df_temp.groupby("country")["climate_impact_index"].mean()
    df["country_long_term_mean"] = df["country"].map(country_long_term_mean)
    df["country_long_term_mean"].fillna(train_df_temp["climate_impact_index"].mean(), inplace=True)
    df["country_recent_deviation"] = df["climate_impact_index"] - df["country_long_term_mean"]

    train_df_temp = df[df["period"] <= 2023].copy()

    human_burden_components = train_df_temp[["log_total_affected", "log_total_death", "log_total_affected_3yr_avg", "log_total_death_3yr_avg"]].mean(axis=1)
    human_burden_mean = human_burden_components.mean()
    human_burden_std = human_burden_components.std() or 1
    df["human_burden"] = (df[["log_total_affected", "log_total_death", "log_total_affected_3yr_avg", "log_total_death_3yr_avg"]].mean(axis=1) - human_burden_mean) / human_burden_std

    persistence_train = train_df_temp[["impact_3yr_avg", "impact_lag1"]].mean(axis=1)
    persistence_mean = persistence_train.mean()
    persistence_std = persistence_train.std() or 1
    df["persistence"] = (df[["impact_3yr_avg", "impact_lag1"]].mean(axis=1) - persistence_mean) / persistence_std

    climate_intensity_mean = train_df_temp["climate_impact_index"].mean()
    climate_intensity_std = train_df_temp["climate_impact_index"].std() or 1
    df["climate_intensity"] = (df["climate_impact_index"] - climate_intensity_mean) / climate_intensity_std

    structural_mean = train_df_temp["country_mean_impact"].mean()
    structural_std = train_df_temp["country_mean_impact"].std() or 1
    df["structural_vulnerability"] = (df["country_mean_impact"] - structural_mean) / structural_std

    df["CLI"] = (
        0.50 * df["human_burden"] +
        0.25 * df["persistence"] +
        0.15 * df["climate_intensity"] +
        0.10 * df["structural_vulnerability"]
    )

    df["impact_rebased_next"] = df.groupby("country")["impact_rebased"].shift(-1)

    FEATURES = [
        "flood_impact",
        "drought_impact",
        "storms_impact",
        "extreme_temp_impact",
        "climate_impact_index",
        "country_mean_impact",
        "impact_lag1",
        "impact_3yr_avg",
        "impact_trend_5yr",
        "impact_std_5yr",
        "country_recent_deviation",
        "log_total_affected",
        "log_total_death",
        "log_total_affected_3yr_avg",
        "log_total_death_3yr_avg",
        "absolute_impact_trend",
        "hazard_count",
        "economic_damage_pct_gdp",
    ]

    TARGET = "impact_rebased_next"

    train_df = df[df["period"] <= 2023].copy()
    all_countries = train_df['country'].unique()
    pred_df_list = []
    
    for country in all_countries:
        country_data = df[df['country'] == country].copy()
        for year in [2025, 2024, 2023, 2022, 2021, 2020]:
            year_data = country_data[country_data['period'] == year].copy()
            if not year_data.empty:
                pred_df_list.append(year_data.iloc[[0]])
                break
    
    if not pred_df_list:
        raise ValueError("No prediction data available")
    
    pred_df = pd.concat(pred_df_list, ignore_index=True)
    
    country_years_count = train_df.groupby('country')['period'].nunique()
    valid_countries = []
    for country in pred_df['country'].unique():
        total_years = country_years_count.get(country, 0)
        if total_years >= 5:
            valid_countries.append(country)
    
    pred_df = pred_df[pred_df['country'].isin(valid_countries)].copy()
    
    train_df = train_df.dropna(subset=FEATURES + [TARGET])

    for feat in FEATURES:
        if feat in ['flood_impact', 'drought_impact', 'storms_impact', 'extreme_temp_impact']:
            pred_df[feat] = pred_df[feat].fillna(0)
        elif feat == 'climate_impact_index':
            pred_df[feat] = pred_df[feat].fillna(
                pred_df[['flood_impact', 'drought_impact', 'storms_impact', 'extreme_temp_impact']].mean(axis=1, skipna=True)
            )
            pred_df[feat] = pred_df[feat].fillna(0)
        elif feat in ['log_total_affected_3yr_avg', 'log_total_death_3yr_avg']:
            if feat not in pred_df.columns:
                if feat == 'log_total_affected_3yr_avg':
                    pred_df[feat] = pred_df['log_total_affected']
                else:
                    pred_df[feat] = pred_df['log_total_death']
            pred_df[feat] = pred_df[feat].fillna(pred_df[feat].median() if pred_df[feat].notna().any() else 0)
        elif feat == 'absolute_impact_trend':
            if feat not in pred_df.columns:
                pred_df[feat] = 0
            pred_df[feat] = pred_df[feat].fillna(0)
        else:
            pred_df[feat] = pred_df[feat].fillna(pred_df[feat].median() if pred_df[feat].notna().any() else 0)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_pred = pred_df[FEATURES]

    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    search_rf = RandomizedSearchCV(
        rf,
        param_distributions=param_grid,
        n_iter=20,
        scoring="r2",
        cv=3,
        random_state=42,
        n_jobs=-1
    )

    search_rf.fit(X_train, y_train)
    best_model_rf = search_rf.best_estimator_

    y_pred_2026 = best_model_rf.predict(X_pred)
    pred_df["predicted_impact_rebased_2026"] = y_pred_2026

    train_df_temp = df[df["period"] <= 2023].copy()
    
    human_burden_components = train_df_temp[["log_total_affected", "log_total_death", "log_total_affected_3yr_avg", "log_total_death_3yr_avg"]].mean(axis=1)
    human_burden_mean = human_burden_components.mean()
    human_burden_std = human_burden_components.std() or 1
    pred_df["human_burden"] = (pred_df[["log_total_affected", "log_total_death", "log_total_affected_3yr_avg", "log_total_death_3yr_avg"]].mean(axis=1) - human_burden_mean) / human_burden_std

    persistence_train = train_df_temp[["impact_3yr_avg", "impact_lag1"]].mean(axis=1)
    persistence_mean = persistence_train.mean()
    persistence_std = persistence_train.std() or 1
    pred_df["persistence"] = (pred_df[["impact_3yr_avg", "impact_lag1"]].mean(axis=1) - persistence_mean) / persistence_std

    climate_intensity_mean = train_df_temp["climate_impact_index"].mean()
    climate_intensity_std = train_df_temp["climate_impact_index"].std() or 1
    pred_df["climate_intensity"] = (pred_df["climate_impact_index"] - climate_intensity_mean) / climate_intensity_std

    structural_mean = train_df_temp["country_mean_impact"].mean()
    structural_std = train_df_temp["country_mean_impact"].std() or 1
    pred_df["structural_vulnerability"] = (pred_df["country_mean_impact"] - structural_mean) / structural_std

    pred_df["CLI"] = (
        0.50 * pred_df["human_burden"] +
        0.25 * pred_df["persistence"] +
        0.15 * pred_df["climate_intensity"] +
        0.10 * pred_df["structural_vulnerability"]
    )

    pred_df = filter_valid_countries(pred_df)
    pred_df = pred_df[pred_df["CLI"].notna()].copy()

    feature_importance = pd.DataFrame({
        'feature': FEATURES,
        'importance': best_model_rf.feature_importances_
    }).sort_values('importance', ascending=False)

    return best_model_rf, pred_df, feature_importance, FEATURES


if __name__ == "__main__":
    model, pred_df, feature_importance, FEATURES = train_and_predict()
    
    print(f"Predictions for 2026 (using 2025 data):")
    print(f"Countries predicted: {len(pred_df)}")
    print(f"Mean predicted impact: {pred_df['predicted_impact_2026'].mean():.4f}")
    
    print("\nFeature Importance:")
    print(feature_importance.to_string(index=False))
    
    print("\nTop 10 Countries Predicted Highest Impact in 2026:")
    top_10 = pred_df.nlargest(10, "predicted_impact_2026")[["country", "predicted_impact_2026", "CLI"]]
    print(top_10.to_string(index=False))
