import pandas as pd
import numpy as np
import os

def assign_stage(smw):
    """
    Assigns crop stage based on Standard Meteorological Week (SMW).
    Approximate stages for Kharif Rice:
    - Germination/Seedling: Weeks 22-23 (Early June)
    - Vegetative: Weeks 24-30 (Mid June - Late July)
    - Reproductive (Panicle Initiation/Flowering): Weeks 31-36 (August - Mid Sept)
    - Ripening/Maturity: Weeks 37-44 (Late Sept - Oct)
    """
    if 22 <= smw <= 23:
        return 'Germination'
    elif 24 <= smw <= 30:
        return 'Vegetative'
    elif 31 <= smw <= 36:
        return 'Reproductive'
    elif 37 <= smw <= 44:
        return 'Ripening'
    else:
        return 'Other'

def calculate_stage_features(group):
    # Sort by week to ensure correct ordering if needed (though we aggregating)
    # group = group.sort_values('week')
    
    features = {}
    
    # Define stages
    stages = ['Germination', 'Vegetative', 'Reproductive', 'Ripening']
    
    for stage in stages:
        stage_data = group[group['Stage'] == stage]
        
        prefix = f"{stage}_"
        
        if stage_data.empty:
            # Fill with NaNs or 0s? 0s might be safer for rainfall, but NaNs for temp.
            # Let's use 0 for rain-related, and mean of the whole season for temp (imputation) or just NaN.
            # For now, let's put NaN and handle later, or 0.
            features[prefix + 'Rainfall_Total'] = 0.0
            features[prefix + 'Rainy_Weeks'] = 0.0
            features[prefix + 'Max_Weekly_Rain'] = 0.0
            features[prefix + 'Mean_Temp'] = group['temp'].mean() # Fallback
            features[prefix + 'Heat_Stress_Weeks'] = 0.0
            features[prefix + 'Night_Stress_Weeks'] = 0.0
            features[prefix + 'Mean_Humidity'] = group['humidity'].mean() # Fallback
            features[prefix + 'Mean_Solar'] = group['solarradiation'].mean() # Fallback
            continue

        # --- Rain Features ---
        features[prefix + 'Rainfall_Total'] = stage_data['precip'].sum()
        # Proxy for Rainy Days: Weeks with > 10mm rain (Weekly aggregation)
        # 2.5mm is a rainy day. A rainy week might have > 10mm? 
        # Let's stick to the user's "Rainy Days" request by using "Rainy Weeks" > 5mm as a proxy for "One significant rain".
        features[prefix + 'Rainy_Weeks'] = (stage_data['precip'] > 5.0).sum()
        features[prefix + 'Max_Weekly_Rain'] = stage_data['precip'].max()
        
        # --- Temp Features ---
        features[prefix + 'Mean_Temp'] = stage_data['temp'].mean()
        # Heat Stress: Weeks where max temp > 35
        features[prefix + 'Heat_Stress_Weeks'] = (stage_data['tempmax'] > 35.0).sum()
        # Night Stress: Weeks where min temp > 25
        features[prefix + 'Night_Stress_Weeks'] = (stage_data['tempmin'] > 25.0).sum()
        
        # --- Other Features ---
        features[prefix + 'Mean_Humidity'] = stage_data['humidity'].mean()
        features[prefix + 'Mean_Solar'] = stage_data['solarradiation'].mean()
        
    return pd.Series(features)

def main():
    print("Loading data...")
    try:
        real_df = pd.read_csv('d:/Rice/rice.csv')
        syn_df = pd.read_csv('d:/Rice/synthetic_weekly_rice.csv')
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # 1. Standardize Columns
    # Real data might not have 'Is_Synthetic'
    real_df['Is_Synthetic'] = 0
    real_df['Synthetic_ID'] = 'Real_' + real_df['District'] + '_' + real_df['Year'].astype(str)
    
    # Synthetic data cols check
    # Ensure generated synthetic data has the same meteo columns
    required_cols = ['District', 'Year', 'week', 'precip', 'temp', 'tempmax', 'tempmin', 'humidity', 'solarradiation', 'Yield (Ton./Ha.)', 'Is_Synthetic']
    
    # Rename Yield if needed
    if 'Yield' in syn_df.columns and 'Yield (Ton./Ha.)' not in syn_df.columns:
        syn_df.rename(columns={'Yield': 'Yield (Ton./Ha.)'}, inplace=True)
        
    # Combine
    # Select only needed columns to avoid mismatches
    cols_to_use = required_cols + ['Season'] # Add Season if available
    
    # Check if 'Season' exists, if not assume Kharif or derived
    if 'Season' not in real_df.columns:
        real_df['Season'] = 'Kharif' # Assumption based on context
        
    # Filter for columns that exist
    common_cols = list(set(real_df.columns) & set(syn_df.columns))
    combined_df = pd.concat([real_df[common_cols], syn_df[common_cols]], axis=0, ignore_index=True)
    
    print(f"Combined Data Shape: {combined_df.shape}")
    
    # 2. Assign Stages
    combined_df['Stage'] = combined_df['week'].apply(assign_stage)
    
    # 3. Aggregate Features
    print("Aggregating Stage-wise features...")
    # Group by Unique Identifier. 
    # For Real: District + Year. 
    # For Synthetic: Synthetic_ID is unique per generated season.
    # We need a unique ID for grouping.
    
    if 'Synthetic_ID' not in combined_df.columns:
        # Create a unique ID
        combined_df['ID'] = combined_df.apply(lambda x: f"{x['District']}_{x['Year']}_{x['Is_Synthetic']}", axis=1)
    else:
        combined_df['ID'] = combined_df['Synthetic_ID'].fillna(combined_df['District'] + '_' + combined_df['Year'].astype(str))

    # We also need the Target (Yield) and Meta info (District, Year, Is_Synthetic) for the final table
    # Since Yield is constant for the whole season, we can just take the first value.
    
    # Function to apply to the grouped DF
    # Use apply on the groupby object
    grouped = combined_df.groupby('ID')
    
    feature_df = grouped.apply(calculate_stage_features).reset_index()
    
    # Extract Metadata and Target
    meta_df = grouped.agg({
        'Yield (Ton./Ha.)': 'first',
        'District': 'first',
        'Year': 'first',
        'Is_Synthetic': 'first'
    }).reset_index()
    
    # Merge
    final_df = pd.merge(meta_df, feature_df, on='ID')
    
    # 4. Add Interaction Features
    # Example: Rain * Temp during Reproductive (High rain + High Temp might be bad/good?)
    # User asked for: Rain * Temperature, Heat Stress * Humidity
    
    stages = ['Germination', 'Vegetative', 'Reproductive', 'Ripening']
    for stage in stages:
        prefix = f"{stage}_"
        # Rain * Temp
        final_df[f'{prefix}Rain_x_Temp'] = final_df[f'{prefix}Rainfall_Total'] * final_df[f'{prefix}Mean_Temp']
        # Heat Stress * Humidity (Humid Heat)
        final_df[f'{prefix}Heat_x_Humidity'] = final_df[f'{prefix}Heat_Stress_Weeks'] * final_df[f'{prefix}Mean_Humidity']

    print("Features Calculated.")
    print(final_df.head())
    
    # Save
    output_path = 'd:/Rice/advanced_features.csv'
    final_df.to_csv(output_path, index=False)
    print(f"Saved advanced features to {output_path}")

if __name__ == "__main__":
    main()
