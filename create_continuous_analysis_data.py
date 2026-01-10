"""
Create continuous (non-binned) analysis_data_set from raw data.
This script processes all raw data files without 5-year binning, using individual years instead.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def build_hazard_impact_continuous(df, hazard_name, min_events=1):
    """
    no binning
    Required columns:
    country | year | affected | death
    """
    df = df.copy()
    
    # Drop rows where both affected & death are missing
    df = df.dropna(subset=["affected", "death"], how="all")
    df = df.sort_values(["country", "year"])
    
    df["period"] = df["year"]
    
    agg = (
        df.groupby(["country", "period"])
          .agg(
              affected=("affected", "mean"),
              death=("death", "mean"),
              events=("year", "count")
          )
          .reset_index()
    )
    
    # Filter by minimum events
    agg = agg[agg["events"] >= min_events]
    
    # LOG TRANSFORM (reduce skew)
    agg["affected_log"] = np.log1p(agg["affected"])
    agg["death_log"] = np.log1p(agg["death"])
    
    # STANDARDIZE (z-score) - fit on all data
    scaler = StandardScaler()
    agg[["affected_z", "death_z"]] = scaler.fit_transform(
        agg[["affected_log", "death_log"]]
    )
    
    # FINAL IMPACT SCORE
    agg[f"{hazard_name}_impact"] = agg["affected_z"] + agg["death_z"]
    
    return agg[["country", "period", f"{hazard_name}_impact"]]


def build_hazard_total_continuous(df, hazard_name, min_events=1):
    """
    Process total hazard data WITHOUT binning - uses individual years.
    
    Required columns:
    country | year | affected | death
    """
    df = df.copy()
    
    df = df.dropna(subset=["affected", "death"], how="all")
    df = df.sort_values(["country", "year"])
    
    df["period"] = df["year"]
    
    agg = (
        df.groupby(["country", "period"])
          .agg(
              events=("year", "count"),
              total_deaths=("death", "sum"),
              total_affected=("affected", "sum")
          )
          .reset_index()
    )
    
    agg = agg[agg["events"] >= min_events]
    
    agg = agg.rename(columns={
        "total_affected": f"{hazard_name}_total_affected",
        "total_deaths": f"{hazard_name}_total_deaths"
    })
    
    return agg[["country", "period", f"{hazard_name}_total_affected", f"{hazard_name}_total_deaths"]]


print("=" * 60)
print("Creating Continuous (Non-Binned) Analysis Dataset")
print("=" * 60)

# PART 1: Process per-capita impact data (for final_data)
print("\n[1/4] Processing per-capita impact data...")

# Drought
print("  - Processing drought data...")
drought_aff = pd.read_csv('data/raw/per_100k/total-affected-by-drought/affected.csv')
drought_death = pd.read_csv('data/raw/per_100k/death-rate-from-drought/death.csv')
drought_aff.rename(columns={
    'Country name': 'country',
    'Year': 'year',
    'Total number of people affected by drought per 100,000': 'affected'
}, inplace=True)
drought_death.rename(columns={
    'Country name': 'country',
    'Year': 'year',
    'Death rates from drought': 'death'
}, inplace=True)
drought_df = pd.merge(drought_aff, drought_death, on=['country', 'year'], how='outer')
drought_final = build_hazard_impact_continuous(drought_df, "drought")

# Flood
print("  - Processing flood data...")
flood_aff = pd.read_csv('data/raw/per_100k/total-affected-by-floods/affected.csv')
flood_death = pd.read_csv('data/raw/per_100k/death-rate-from-floods/death.csv')
flood_aff.rename(columns={
    'Country name': 'country',
    'Year': 'year',
    'Total number of people affected by floods per 100,000': 'affected'
}, inplace=True)
flood_death.rename(columns={
    'Country name': 'country',
    'Year': 'year',
    'Death rates from floods': 'death'
}, inplace=True)
flood_df = pd.merge(flood_aff, flood_death, on=['country', 'year'], how='outer')
flood_final = build_hazard_impact_continuous(flood_df, "flood")

# Storms
print("  - Processing storms data...")
storms_aff = pd.read_csv('data/raw/per_100k/total-affected-by-storms/affected.csv')
storms_death = pd.read_csv('data/raw/per_100k/death-rate-from-storms/death.csv')
storms_aff.rename(columns={
    'Country name': 'country',
    'Year': 'year',
    'Total number of people affected by storms per 100,000': 'affected'
}, inplace=True)
storms_death.rename(columns={
    'Country name': 'country',
    'Year': 'year',
    'Death rates from storms': 'death'
}, inplace=True)
storms_df = pd.merge(storms_aff, storms_death, on=['country', 'year'], how='outer')
storms_final = build_hazard_impact_continuous(storms_df, "storms")

# Extreme temperatures
print("  - Processing extreme temperature data...")
extreme_temp_aff = pd.read_csv('data/raw/per_100k/total-affected-by-extreme-temperatures/affected.csv')
extreme_temp_death = pd.read_csv('data/raw/per_100k/death-rate-from-extreme-temperatures/death.csv')
extreme_temp_aff.rename(columns={
    'Country name': 'country',
    'Year': 'year',
    'Total number of people affected by extreme temperatures per 100,000': 'affected'
}, inplace=True)
extreme_temp_death.rename(columns={
    'Country name': 'country',
    'Year': 'year',
    'Death rates from extreme temperatures': 'death'
}, inplace=True)
extreme_temp_df = pd.merge(extreme_temp_aff, extreme_temp_death, on=['country', 'year'], how='outer')
extreme_temp_final = build_hazard_impact_continuous(extreme_temp_df, "extreme_temp")

# Merge all impact data
print("  - Merging impact data...")
final_data_processed = (
    flood_final
    .merge(drought_final, on=["country", "period"], how="outer")
    .merge(storms_final, on=["country", "period"], how="outer")
    .merge(extreme_temp_final, on=["country", "period"], how="outer")
)

# Calculate climate impact index
impact_cols = ["flood_impact", "drought_impact", "storms_impact", "extreme_temp_impact"]
final_data_processed["climate_impact_index"] = final_data_processed[impact_cols].mean(axis=1, skipna=True)
final_data_processed["hazard_count"] = final_data_processed[impact_cols].notna().sum(axis=1)

# Rebase impact
min_val = final_data_processed["climate_impact_index"].min()
final_data_processed["impact_rebased"] = final_data_processed["climate_impact_index"] - min_val

print(f"  ✓ Final data shape: {final_data_processed.shape}")

# PART 2: Process total counts data (for final_total_data)
print("\n[2/4] Processing total counts data...")

# Drought totals
print("  - Processing drought totals...")
drought_aff_total = pd.read_csv('data/raw/total_count/total-affected-by-drought/affected.csv')
drought_death_total = pd.read_csv('data/raw/total_count/deaths-from-drought/deaths.csv')
drought_aff_total.rename(columns={'total_affected_drought': 'affected'}, inplace=True)
drought_death_total.rename(columns={'deaths_drought': 'death'}, inplace=True)
drought_total_df = pd.merge(drought_aff_total, drought_death_total, on=['country', 'year'], how='outer')
drought_total_final = build_hazard_total_continuous(drought_total_df, "drought")

# Flood totals
print("  - Processing flood totals...")
flood_aff_total = pd.read_csv('data/raw/total_count/total-affected-by-floods/affected.csv')
flood_death_total = pd.read_csv('data/raw/total_count/deaths-from-floods/deaths.csv')
flood_aff_total.rename(columns={'total_affected_flood': 'affected'}, inplace=True)
flood_death_total.rename(columns={'deaths_flood': 'death'}, inplace=True)
flood_total_df = pd.merge(flood_aff_total, flood_death_total, on=['country', 'year'], how='outer')
flood_total_final = build_hazard_total_continuous(flood_total_df, "flood")

# Storm totals
print("  - Processing storm totals...")
storm_aff_total = pd.read_csv('data/raw/total_count/total-affected-by-storms/affected.csv')
storm_death_total = pd.read_csv('data/raw/total_count/deaths-from-storms/deaths.csv')
storm_aff_total.rename(columns={'total_affected_storm': 'affected'}, inplace=True)
storm_death_total.rename(columns={'deaths_storm': 'death'}, inplace=True)
storm_total_df = pd.merge(storm_aff_total, storm_death_total, on=['country', 'year'], how='outer')
storm_total_final = build_hazard_total_continuous(storm_total_df, "storm")

# Extreme temp totals
print("  - Processing extreme temperature totals...")
extreme_temp_aff_total = pd.read_csv('data/raw/total_count/total-affected-by-extreme-temperatures/affected.csv')
extreme_temp_death_total = pd.read_csv('data/raw/total_count/deaths-from-extreme-temperatures/deaths.csv')
extreme_temp_aff_total.rename(columns={'total_affected_temperature': 'affected'}, inplace=True)
extreme_temp_death_total.rename(columns={'deaths_temperature': 'death'}, inplace=True)
extreme_temp_total_df = pd.merge(extreme_temp_aff_total, extreme_temp_death_total, on=['country', 'year'], how='outer')
extreme_temp_total_final = build_hazard_total_continuous(extreme_temp_total_df, "extreme_temp")

# Merge all total data
print("  - Merging total data...")
final_total_processed = extreme_temp_total_final.copy()
final_total_processed = final_total_processed.merge(drought_total_final, on=['country', 'period'], how='outer')
final_total_processed = final_total_processed.merge(flood_total_final, on=['country', 'period'], how='outer')
final_total_processed = final_total_processed.merge(storm_total_final, on=['country', 'period'], how='outer')

# Fill NaN with 0
final_total_processed = final_total_processed.fillna(0)

# Calculate totals
final_total_processed['total_deaths'] = (
    final_total_processed['extreme_temp_total_deaths'] +
    final_total_processed['drought_total_deaths'] +
    final_total_processed['flood_total_deaths'] +
    final_total_processed['storm_total_deaths']
)

final_total_processed['total_affected'] = (
    final_total_processed['extreme_temp_total_affected'] +
    final_total_processed['drought_total_affected'] +
    final_total_processed['flood_total_affected'] +
    final_total_processed['storm_total_affected']
)

print(f"  ✓ Final total data shape: {final_total_processed.shape}")

# PART 3: Process economic damage data (without binning)
print("\n[3/4] Processing economic damage data...")

economic_damage = pd.read_csv('data/raw/total_count/economic-damages-from-disasters/economic_damage.csv')

# Melt to long format
economic_long = economic_damage.melt(
    id_vars=['Entity', 'Code', 'Year'],
    value_vars=['drought', 'flood', 'storm', 'extreme_temp'],
    var_name='hazard',
    value_name='economic_damage_pct_gdp'
)

# Use Year directly as period (no binning)
economic_long['period'] = economic_long['Year']
economic_long['country'] = economic_long['Entity']

# Aggregate economic damage to country-year using mean across hazards
hazard_cols = ['drought', 'flood', 'storm', 'extreme_temp']
economic_agg = (
    economic_long
    .groupby(['country', 'period'])['economic_damage_pct_gdp']
    .mean()
    .reset_index()
    .rename(columns={'economic_damage_pct_gdp': 'economic_damage_pct_gdp'})
)

economic_processed = economic_agg[['country', 'period', 'economic_damage_pct_gdp']].copy()
print(f"  ✓ Economic data shape: {economic_processed.shape}")


# PART 4: Merge everything into analysis dataset
print("\n[4/4] Merging all datasets...")

# Merge impact data with total data
analysis_dataset = final_data_processed.merge(
    final_total_processed[['country', 'period', 'total_deaths', 'total_affected']],
    on=['country', 'period'],
    how='outer'
)

# Merge with economic data
analysis_dataset = analysis_dataset.merge(
    economic_processed,
    on=['country', 'period'],
    how='outer'
)

# Calculate resilience metrics
analysis_dataset['resilience_rate'] = np.where(
    analysis_dataset['total_affected'] > 0,
    analysis_dataset['total_deaths'] / analysis_dataset['total_affected'],
    0
)
analysis_dataset['resilience_per_100k'] = analysis_dataset['resilience_rate'] * 100000

# Fill economic damage NaN with 0
analysis_dataset['economic_damage_pct_gdp'] = analysis_dataset['economic_damage_pct_gdp'].fillna(0)

# Select final columns (same as binned version)
final_columns = [
    'country', 'period',
    'flood_impact', 'drought_impact', 'storms_impact', 'extreme_temp_impact',
    'climate_impact_index', 'impact_rebased', 'hazard_count',
    'total_deaths', 'total_affected', 'resilience_rate', 'resilience_per_100k',
    'economic_damage_pct_gdp'
]

analysis_data_set = analysis_dataset[final_columns].copy()

# Sort by country and period
analysis_data_set = analysis_data_set.sort_values(['country', 'period']).reset_index(drop=True)


output_path = 'data/processed/analysis_data_set_continuous.csv'
analysis_data_set.to_csv(output_path, index=False)

print(f"\n{'='*60}")
print(f"✓ Continuous analysis dataset created successfully!")
print(f"{'='*60}")
print(f"Output: {output_path}")
print(f"Shape: {analysis_data_set.shape}")
print(f"Columns: {list(analysis_data_set.columns)}")
print(f"\nFirst few rows:")
print(analysis_data_set.head(10))
print(f"\nPeriod range: {analysis_data_set['period'].min()} - {analysis_data_set['period'].max()}")
print(f"Unique countries: {analysis_data_set['country'].nunique()}")
print(f"Unique periods (years): {analysis_data_set['period'].nunique()}")

