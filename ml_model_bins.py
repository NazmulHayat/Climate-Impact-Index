import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv("data/processed/analysis_data_set.csv")

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


#adding lag features
train_df["impact_lag1"] = (
    train_df.groupby("country")["impact_rebased"].shift(1)
)
test_df["impact_lag1"] = (
    test_df.groupby("country")["impact_rebased"].shift(1)
)

FEATURES = [
    "country_mean_impact",
    "impact_lag1",
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

search = RandomizedSearchCV(
    rf,
    param_distributions=param_grid,
    n_iter=20,              # keep small
    scoring="r2",
    cv=3,                   # simple CV
    random_state=42,
    n_jobs=-1
)
search.fit(X_train, y_train)
best_model = search.best_estimator_
print("Random Forest Model")
print("Best params:", search.best_params_)
print("Best CV R²:", search.best_score_)


y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("RMSE:", rmse)
print("R²:", r2)

#Gradientboostingregressor model -> bad R2: only 0.188
param_grid = {
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

search = RandomizedSearchCV(
    gbr,
    param_distributions=param_grid,
    n_iter=20,
    scoring="r2",
    cv=3,
    random_state=42,
    n_jobs=-1
)
search.fit(X_train, y_train)
best_model = search.best_estimator_
print("Gradient Boosting Regressor Model")
print("Best params:", search.best_params_) 
print("Best CV R²:", search.best_score_)

y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("RMSE:", rmse)
print("R²:", r2)