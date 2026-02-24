
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings
import os
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')
# Set random seed
np.random.seed(42)
torch.manual_seed(42)

def load_and_aggregate_data(file_path):
    print("\n--- 1. LOADING & AGGREGATING DATA ---")
    try:
        if not os.path.exists(file_path):
             raise FileNotFoundError(f"File not found: {file_path}")
             
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        yield_col = next((c for c in df.columns if 'Yield' in c), None)
        if yield_col:
            df.rename(columns={yield_col: 'Yield'}, inplace=True)
        else:
            raise ValueError("Yield column not found")

        # GDD Calculation
        if all(c in df.columns for c in ['tempmax', 'tempmin']):
            df['GDD'] = (((df['tempmax'] + df['tempmin']) / 2) - 10).clip(lower=0)
        else:
            df['GDD'] = 0 

        def custom_agg(x):
            if 'week' in x.columns:
                x = x.sort_values('week')
            
            n = len(x)
            split_idx = int(n * 0.3)
            
            # Expanded Feature Set
            early_rain = x['precip'].iloc[:split_idx].sum() if 'precip' in x.columns else 0
            late_rain = x['precip'].iloc[-split_idx:].sum() if split_idx > 0 and 'precip' in x.columns else 0
            
            # Variability features
            temp_var = x['temp'].std() if 'temp' in x.columns else 0
            rain_days = (x['precip'] > 0.1).sum() if 'precip' in x.columns else 0
            
            res = pd.Series({
                'Mean_Temp': x['temp'].mean() if 'temp' in x.columns else 0,
                'Max_Temp': x['tempmax'].max() if 'tempmax' in x.columns else 0,
                'Total_Rain': x['precip'].sum() if 'precip' in x.columns else 0,
                'Rain_Days': rain_days,
                'Temp_Variability': temp_var,
                'Mean_Humidity': x['humidity'].mean() if 'humidity' in x.columns else 0,
                'Total_GDD': x['GDD'].sum(),
                'Mean_Solar': x['solarradiation'].mean() if 'solarradiation' in x.columns else 0,
                'Early_Rain_30pct': early_rain,
                'Late_Rain_30pct': late_rain,
                'Duration_Weeks': len(x)
            })
            res['Yield'] = x['Yield'].iloc[0]
            return res

        print("Aggregating weekly data to seasonal...")
        aggregated = df.groupby(['District', 'Season', 'Year', 'Crop']).apply(custom_agg).reset_index()
        aggregated = aggregated.dropna()
        print(f"Aggregated Shape: {aggregated.shape}")
        return aggregated

    except Exception as e:
        print(f"Error: {e}")
        return None

# --- VAE Definition ---
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=4): # Slightly larger latent dim
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim) 
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Beta-VAE aspect: weight KLD slightly to encourage disentanglement?
    return MSE + 1.2 * KLD

def train_vae(model, dataloader, epochs=300, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data,) in enumerate(dataloader):
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
    return model

def run_best_generation(df, output_file="best_synthetic_rice.csv"):
    print("\n--- 2. RUNNING ADVANCED GENERATION PIPELINE ---")
    
    # 1. Inspect Columns
    feature_cols = [c for c in df.columns if c not in ['District', 'Season', 'Year', 'Crop', 'Yield']]
    print(f"Using Features: {feature_cols}")
    
    X = df[feature_cols].values
    y = df['Yield'].values
    
    # 2. Train Weather VAE
    print("\n[Step A] Training VAE on Weather Patterns...")
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    X_tensor = torch.FloatTensor(X_scaled)
    
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    input_dim = X_scaled.shape[1]
    vae = VAE(input_dim, latent_dim=4)
    train_vae(vae, dataloader)
    print("  VAE Trained.")

    # 3. Probabilistic Yield Modeling
    print("\n[Step B] Training Probabilistic Yield Model...")
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, oob_score=True)
    rf.fit(X, y)
    print(f"  RF OOB R² Score: {rf.oob_score_:.4f}")
    
    # Calculate Residuals (The "Noise" Distribution)
    y_pred_train = rf.predict(X)
    residuals = y - y_pred_train
    residuals_std = np.std(residuals)
    print(f"  Residual Std Dev: {residuals_std:.4f}")
    
    # 4. Generate Novel Data
    print("\n[Step C] Generating and Validating Samples...")
    n_samples = 2000 # Generate extra, filter later
    
    vae.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, 4)
        syn_weather_scaled = vae.decode(z).numpy()
    
    syn_weather = scaler_X.inverse_transform(syn_weather_scaled)
    syn_df = pd.DataFrame(syn_weather, columns=feature_cols)
    
    # Clean up physical constraints
    for col in syn_df.columns:
        syn_df[col] = syn_df[col].clip(lower=0) 
    if 'Mean_Humidity' in syn_df.columns:
        syn_df['Mean_Humidity'] = syn_df['Mean_Humidity'].clip(upper=100)
    
    # Predict Mean Yield
    syn_mean_yield = rf.predict(syn_df.values)
    
    # Add Noise (Probabilistic Step)
    # Allow some variation based on residual distribution
    noise = np.random.normal(0, residuals_std, size=n_samples)
    syn_final_yield = syn_mean_yield + noise
    syn_df['Yield'] = syn_final_yield
    syn_df['Yield'] = syn_df['Yield'].clip(lower=0) # No negative yield
    
    # 5. Novelty Validation (Distance to Nearest Neighbor)
    print("\n[Step D] Checking Novelty (Privacy check)...")
    # Normalize features for distance calculation
    # We compare Synthetic Points vs Real Points
    
    # Use only reduced feature set for distance to avoid curse of dimensionality affects
    # Or use all scaled features
    
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X_scaled)
    
    # Scale synthetic data using the SAME scaler
    syn_X_scaled = scaler_X.transform(syn_df[feature_cols].values)
    
    distances, indices = nbrs.kneighbors(syn_X_scaled)
    min_distances = distances.flatten()
    
    syn_df['Distance_to_Real'] = min_distances
    
    # Filter "Too Close" points (Potential Memorization)
    dist_threshold = 0.1 # Arbitrary threshold, tune based on distribution
    novel_mask = syn_df['Distance_to_Real'] > dist_threshold
    print(f"  Total Generated: {n_samples}")
    print(f"  Dropped (Too Close to Real): {(~novel_mask).sum()}")
    
    final_syn_df = syn_df[novel_mask].copy()
    print(f"  Final Novel Samples: {len(final_syn_df)}")
    
    # Plot DCR Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(min_distances, kde=True, color='purple', label='Distance to Closest Real Record')
    plt.axvline(dist_threshold, color='red', linestyle='--', label='Privacy Threshold')
    plt.title('Novelty Check: Distance to Nearest Real Neighbor')
    plt.xlabel('Euclidean Distance (Scaled Space)')
    plt.legend()
    plt.savefig('dcr_plot.png')
    print("  Saved 'dcr_plot.png'")
    
    # Save Final Dataset
    # Take top 1000 novel samples if available, or all
    if len(final_syn_df) > 1000:
        final_syn_df = final_syn_df.sample(1000, random_state=42)
    
    final_syn_df.to_csv(output_file, index=False)
    print(f"\nSaved {len(final_syn_df)} high-quality samples to '{output_file}'")

if __name__ == "__main__":
    file_path = "rice.csv"
    if os.path.exists(file_path):
        agg_df = load_and_aggregate_data(file_path)
        if agg_df is not None:
            run_best_generation(agg_df)
    else:
        print(f"Error: {file_path} not found.")
