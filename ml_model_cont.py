import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

df = pd.read_csv("data/processed/analysis_data_set_continuous.csv")

print(df.shape)

df = df.sort_values(["country", "period"])

df["impact_rebased_next"] = (
    df.groupby("country")["impact_rebased"].shift(-1)
)

# spliting data
train_df = df[df["period"] < 2015].copy()
test_df = df[df["period"] >= 2015].copy()

country_mean = (
    train_df.groupby("country")["impact_rebased"].mean()
)

train_df["country_mean_impact"] = train_df["country"].map(country_mean)
test_df["country_mean_impact"] = test_df["country"].map(country_mean)

# fallback for unseen / missing
global_mean = train_df["impact_rebased"].mean()
train_df["country_mean_impact"].fillna(global_mean, inplace=True)
test_df["country_mean_impact"].fillna(global_mean, inplace=True)


FEATURES = [
    "country_mean_impact",
    "flood_impact",
    "drought_impact",
    "storms_impact",
    "extreme_temp_impact",
    "hazard_count",
    "economic_damage_pct_gdp",
]

TARGET = "impact_rebased_next"

train_df = train_df.dropna(subset=FEATURES + [TARGET])
test_df  = test_df.dropna(subset=FEATURES + [TARGET])
X_train = train_df[FEATURES]
y_train = train_df[TARGET]
X_test = test_df[FEATURES]
y_test = test_df[TARGET]

#random forest model
#hyperparameter tuning 
param_grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

search_rf = RandomizedSearchCV(
    rf,
    param_distributions=param_grid,
    n_iter=20,              # keep small
    scoring="r2",
    cv=3,                   # simple CV
    random_state=42,
    n_jobs=-1
)
search_rf.fit(X_train, y_train)
best_model_rf = search_rf.best_estimator_
print("Random Forest Model")
print("Best params:", search_rf.best_params_)
print("Best CV R²:", search_rf.best_score_)

y_pred_rf = best_model_rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
print("RMSE:", rmse_rf)
print("R²:", r2_rf)

# Gradient Boosting Regressor Model
print("\n" + "="*60)
print("Gradient Boosting Regressor Model")
print("="*60)
param_grid_gbr = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [2, 3, 4],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 3, 5],
    "subsample": [0.6, 0.8, 1.0]
}

gbr = GradientBoostingRegressor(
    random_state=42
) 

search_gbr = RandomizedSearchCV(
    gbr,
    param_distributions=param_grid_gbr,
    n_iter=20,
    scoring="r2",
    cv=3,
    random_state=42,
    n_jobs=-1
)
search_gbr.fit(X_train, y_train)
best_model_gbr = search_gbr.best_estimator_
print("Best params:", search_gbr.best_params_) 
print("Best CV R²:", search_gbr.best_score_)

y_pred_gbr = best_model_gbr.predict(X_test)
rmse_gbr = np.sqrt(mean_squared_error(y_test, y_pred_gbr))
r2_gbr = r2_score(y_test, y_pred_gbr)
print("RMSE:", rmse_gbr)
print("R²:", r2_gbr)

# XGBoost Model
print("\n" + "="*60)
print("XGBoost Model")
print("="*60)
param_grid_xgb = {
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 4, 5, 6, 7],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.2],
    "reg_alpha": [0, 0.1, 0.5],
    "reg_lambda": [1, 1.5, 2]
}

xgb = XGBRegressor(
    random_state=42,
    n_jobs=-1,
    tree_method='hist'  # faster for large datasets
)

search_xgb = RandomizedSearchCV(
    xgb,
    param_distributions=param_grid_xgb,
    n_iter=50,  # more iterations for XGBoost
    scoring="r2",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search_xgb.fit(X_train, y_train)
best_model_xgb = search_xgb.best_estimator_
print("Best params:", search_xgb.best_params_)
print("Best CV R²:", search_xgb.best_score_)

y_pred_xgb = best_model_xgb.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)
print("RMSE:", rmse_xgb)
print("R²:", r2_xgb)

# Model Comparison Summary
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print(f"{'Model':<25} {'CV R²':<12} {'Test R²':<12} {'Test RMSE':<12}")
print("-"*60)
print(f"{'Random Forest':<25} {search_rf.best_score_:<12.4f} {r2_rf:<12.4f} {rmse_rf:<12.4f}")
print(f"{'Gradient Boosting':<25} {search_gbr.best_score_:<12.4f} {r2_gbr:<12.4f} {rmse_gbr:<12.4f}")
print(f"{'XGBoost':<25} {search_xgb.best_score_:<12.4f} {r2_xgb:<12.4f} {rmse_xgb:<12.4f}")
print("="*60)

# Find best model
best_models = [
    ("Random Forest", r2_rf, rmse_rf, best_model_rf),
    ("Gradient Boosting", r2_gbr, rmse_gbr, best_model_gbr),
    ("XGBoost", r2_xgb, rmse_xgb, best_model_xgb)
]
best_models.sort(key=lambda x: x[1], reverse=True)  # Sort by R²
print(f"\n🏆 Best Model: {best_models[0][0]} (R² = {best_models[0][1]:.4f}, RMSE = {best_models[0][2]:.4f})")
