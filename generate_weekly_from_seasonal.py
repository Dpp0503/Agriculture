
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import os

def load_data():
    print("--- Loading Data ---")
    
    # 1. Load Real Data (Weekly)
    real_path = "rice.csv"
    if not os.path.exists(real_path):
        print(f"Error: {real_path} not found.")
        return None, None

    real_df = pd.read_csv(real_path)
    real_df.columns = real_df.columns.str.strip()
    
    # Ensure ordered by week
    if 'week' in real_df.columns:
        real_df = real_df.sort_values(['District', 'Year', 'Season', 'week'])
    
    # 2. Load Synthetic Seasonal Data
    syn_path = "best_synthetic_rice.csv"
    if not os.path.exists(syn_path):
        print(f"Error: {syn_path} not found.")
        return None, None
        
    syn_df = pd.read_csv(syn_path)
    
    print(f"Real Weekly Samples: {len(real_df)}")
    print(f"Synthetic Seasonal Samples: {len(syn_df)}")
    
    return real_df, syn_df

def aggregate_real_seasonally(real_df):
    # We need to aggregate real data exactly like we did for the synthetic generator
    # So we can match them in the same feature space.
    
    def custom_agg(x):
        res = pd.Series({
            'Mean_Temp': x['temp'].mean() if 'temp' in x.columns else 0,
            'Total_Rain': x['precip'].sum() if 'precip' in x.columns else 0,
            'Mean_Humidity': x['humidity'].mean() if 'humidity' in x.columns else 0,
             # Simplified matching features
        })
        return res

    print("Aggregating real weekly data for pattern matching...")
    real_seasonal = real_df.groupby(['District', 'Season', 'Year', 'Crop']).apply(custom_agg).reset_index()
    return real_seasonal

def generate_weekly_series(real_df, real_seasonal, syn_df, output_file="synthetic_weekly_rice.csv"):
    print("\n--- Generating Weekly Time-Series ---")
    
    # Matching Features
    match_cols = ['Mean_Temp', 'Total_Rain', 'Mean_Humidity']
    
    # Fit Nearest Neighbors on Real Seasonal Data
    X_real = real_seasonal[match_cols].values
    scaler = StandardScaler()
    X_real_scaled = scaler.fit_transform(X_real)
    
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X_real_scaled)
    
    # Find neighbors for Synthetic Data
    X_syn = syn_df[match_cols].values
    X_syn_scaled = scaler.transform(X_syn)
    
    distances, indices = nbrs.kneighbors(X_syn_scaled)
    
    synthetic_weekly_list = []
    
    print("Reconstructing weekly patterns...")
    
    for i in range(len(syn_df)):
        syn_row = syn_df.iloc[i]
        
        # 1. Get Nearest Real Neighbor
        neighbor_idx = indices[i][0]
        neighbor_meta = real_seasonal.iloc[neighbor_idx]
        
        # 2. Extract Weekly Pattern of Neighbor
        mask = (real_df['District'] == neighbor_meta['District']) & \
               (real_df['Year'] == neighbor_meta['Year']) & \
               (real_df['Season'] == neighbor_meta['Season']) & \
               (real_df['Crop'] == neighbor_meta['Crop'])
        
        neighbor_weekly = real_df[mask].copy()
        
        if len(neighbor_weekly) == 0:
            continue
            
        # 3. Rescale Weekly Values to match Synthetic Totals
        
        # --- Rain Rescaling ---
        real_total_rain = neighbor_weekly['precip'].sum()
        syn_total_rain = syn_row['Total_Rain']
        
        if real_total_rain > 0:
            scale_factor_rain = syn_total_rain / real_total_rain
            neighbor_weekly['precip'] = neighbor_weekly['precip'] * scale_factor_rain
        else:
            # If real neighbor had 0 rain, distribute synthetic rain evenly
            n_weeks = len(neighbor_weekly)
            neighbor_weekly['precip'] = syn_total_rain / n_weeks
            
        # --- Temp Rescaling ---
        real_mean_temp = neighbor_weekly['temp'].mean()
        syn_mean_temp = syn_row['Mean_Temp']
        
        if real_mean_temp > 0:
             scale_factor_temp = syn_mean_temp / real_mean_temp
             neighbor_weekly['temp'] = neighbor_weekly['temp'] * scale_factor_temp
             # Also scale min/max roughly
             if 'tempmax' in neighbor_weekly.columns:
                 neighbor_weekly['tempmax'] = neighbor_weekly['tempmax'] * scale_factor_temp
             if 'tempmin' in neighbor_weekly.columns:
                 neighbor_weekly['tempmin'] = neighbor_weekly['tempmin'] * scale_factor_temp
        
        # --- Humidity Rescaling ---
        real_mean_hum = neighbor_weekly['humidity'].mean()
        syn_mean_hum = syn_row['Mean_Humidity']
        
        if real_mean_hum > 0:
            scale_factor_hum = syn_mean_hum / real_mean_hum
            neighbor_weekly['humidity'] = (neighbor_weekly['humidity'] * scale_factor_hum).clip(0, 100)

        # 4. assign Metadata
        # We give it a unique ID based on the index
        neighbor_weekly['Synthetic_ID'] = i
        neighbor_weekly['Is_Synthetic'] = 1
        neighbor_weekly['Yield'] = syn_row['Yield'] # Assign the synthetic yield (Label)
        
        # Append
        synthetic_weekly_list.append(neighbor_weekly)
        
    # Combine
    syn_weekly_df = pd.concat(synthetic_weekly_list, axis=0)
    
    # Save
    syn_weekly_df.to_csv(output_file, index=False)
    print(f"Saved {len(syn_weekly_df)} weekly rows to '{output_file}'")
    
    # Validation Plot
    plt.figure(figsize=(12, 6))
    
    # Plot a few random synthetic rain curves vs real
    sample_syn_ids = np.random.choice(syn_weekly_df['Synthetic_ID'].unique(), 3)
    
    for sid in sample_syn_ids:
        subset = syn_weekly_df[syn_weekly_df['Synthetic_ID'] == sid]
        plt.plot(subset['week'].values, subset['precip'].values, linestyle='--', label=f'Syn Season {sid}')
        
    plt.title("Sample Synthetic Weekly Rain Patterns")
    plt.xlabel("Week")
    plt.ylabel("Precipitation (mm)")
    plt.legend()
    plt.savefig("weekly_pattern_check.png")
    print("Saved 'weekly_pattern_check.png'")

if __name__ == "__main__":
    real_df, syn_df = load_data()
    if real_df is not None and syn_df is not None:
        real_seasonal = aggregate_real_seasonally(real_df)
        generate_weekly_series(real_df, real_seasonal, syn_df)
