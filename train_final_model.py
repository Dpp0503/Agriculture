
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Try importing XGBoost
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not found. Training Random Forest only.")

# Suppress warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

import math

# --- 1. Deep Learning Models ---

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# Bidirectional LSTM
class RiceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(RiceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Last time step, both directions concatenated
        return self.fc(out)

class RiceCNN1D(nn.Module):
    def __init__(self, input_dim, seq_len):
        super(RiceCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool  = nn.AdaptiveMaxPool1d(1)
        self.fc    = nn.Linear(128, 1)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(0.2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.drop(self.relu(self.conv1(x)))
        x = self.drop(self.relu(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class RiceTransformer(nn.Module):
    def __init__(self, input_dim, seq_len):
        super(RiceTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, 32)
        self.pos_enc   = PositionalEncoding(d_model=32, max_len=seq_len + 1)
        encoder_layer  = nn.TransformerEncoderLayer(d_model=32, nhead=4,
                                                    batch_first=True, dropout=0.1,
                                                    dim_feedforward=128)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Pool over time dim instead of flatten
        self.fc   = nn.Linear(32, 1)

    def forward(self, x):
        x = self.embedding(x)          # (B, T, 32)
        x = self.pos_enc(x)            # Add positional info
        x = self.transformer(x)        # (B, T, 32)
        x = self.pool(x.permute(0, 2, 1)).squeeze(-1)  # (B, 32)
        return self.fc(x)

# ------------------------------------------------------------------------
# Improved training: early stopping + LR scheduler + two-phase fine-tuning
# ------------------------------------------------------------------------
def train_dl_model(model, X_train, y_train, X_test, y_test,
                   epochs=150, batch_size=32, lr=0.001,
                   patience=15, finetune_on_real=True,
                   X_real=None, y_real=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = model.to(device)

    # --- Scale ---
    scaler = StandardScaler()
    n, seq, feat = X_train.shape
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, feat)).reshape(n, seq, feat)
    X_test_scaled  = scaler.transform(X_test.reshape(-1, feat)).reshape(X_test.shape[0], seq, feat)

    def _make_tensors(X, y):
        return (torch.FloatTensor(X).to(device),
                torch.FloatTensor(y).view(-1, 1).to(device))

    # Internal val split (10% of train) for early stopping
    val_size = max(1, int(0.1 * n))
    X_tr, X_val = X_train_scaled[val_size:], X_train_scaled[:val_size]
    y_tr, y_val = y_train[val_size:],        y_train[:val_size]

    Xt, yt = _make_tensors(X_tr, y_tr)
    Xval, yval = _make_tensors(X_val, y_val)
    Xv_full, _ = _make_tensors(X_test_scaled, y_test)

    def _run_training(model, Xt, yt, Xval, yval, epochs, lr, patience):
        loader    = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        best_val, patience_count, best_state = float('inf'), 0, None

        for epoch in range(epochs):
            # --- Train ---
            model.train()
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # --- Val ---
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(Xval), yval).item()
            scheduler.step(val_loss)

            # --- Early stopping ---
            if val_loss < best_val - 1e-5:
                best_val, patience_count = val_loss, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= patience:
                    break

        if best_state:
            model.load_state_dict(best_state)
        return model

    # --- Phase 1: Train on full train set (real + synthetic) ---
    model = _run_training(model, Xt, yt, Xval, yval, epochs, lr, patience)

    # --- Phase 2: Fine-tune on real data only (if provided) ---
    if finetune_on_real and X_real is not None and len(X_real) > 1:
        X_real_scaled = scaler.transform(X_real.reshape(-1, feat)).reshape(len(X_real), seq, feat)
        Xr, yr = _make_tensors(X_real_scaled, y_real)
        # Fine-tune with very low LR, short run, no split needed (few samples)
        ft_loader = DataLoader(TensorDataset(Xr, yr), batch_size=max(4, len(X_real)//4), shuffle=True)
        ft_opt    = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        criterion = nn.MSELoss()
        model.train()
        for _ in range(30):
            for Xb, yb in ft_loader:
                ft_opt.zero_grad()
                criterion(model(Xb), yb).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                ft_opt.step()

    # --- Final prediction ---
    model.eval()
    with torch.no_grad():
        pred = model(Xv_full).cpu().numpy().flatten()

    return model, pred

# --- 2. Data Processing ---

def load_and_prepare_data(real_file, synthetic_file, synthetic_weekly_file):
    print("\n--- 1. LOADING & PREPARING DATA ---")
    
    # A. Load Real Data
    try:
        real_raw = pd.read_csv(real_file)
        real_raw.columns = real_raw.columns.str.strip()
        
        yield_col = next((c for c in real_raw.columns if 'Yield' in c), None)
        if yield_col:
            real_raw.rename(columns={yield_col: 'Yield'}, inplace=True)
        else:
            raise ValueError("Yield column not found in real data")

        # GDD Calculation
        if all(c in real_raw.columns for c in ['tempmax', 'tempmin']):
            real_raw['GDD'] = (((real_raw['tempmax'] + real_raw['tempmin']) / 2) - 10).clip(lower=0)
        else:
            real_raw['GDD'] = 0 
        
        # Add ID for matching
        real_raw['ID'] = 'Real_' + real_raw['District'] + '_' + real_raw['Year'].astype(str) + '_' + real_raw['Season']
        
    except Exception as e:
        print(f"Error processing real data: {e}")
        return None, None

    # B. Load Synthetic Data (Weekly)
    # We need weekly data for DL sequences and then aggregate it for Tabular RF
    try:
        syn_weekly = pd.read_csv(synthetic_weekly_file)
        syn_weekly['Is_Synthetic'] = 1
        # Add ID 
        if 'ID' not in syn_weekly.columns:
             syn_weekly['ID'] = 'Syn_' + syn_weekly['Synthetic_ID'].astype(str)
             
        # GDD for Syn
        if 'GDD' not in syn_weekly.columns:
             if all(c in syn_weekly.columns for c in ['tempmax', 'tempmin']):
                syn_weekly['GDD'] = (((syn_weekly['tempmax'] + syn_weekly['tempmin']) / 2) - 10).clip(lower=0)
             else:
                syn_weekly['GDD'] = 0
                
    except Exception as e:
        print(f"Error loading synthetic weekly data: {e}")
        return None, None

    # --- C. Prepare Sequences (DL Input) ---
    print("preparing sequences...")
    # Sequence Features
    seq_cols = ['temp', 'precip', 'humidity', 'solarradiation', 'GDD']
    final_seq_cols = [c for c in seq_cols if c in real_raw.columns and c in syn_weekly.columns]
    SEQ_LEN = 20
    
    def get_sequences(df):
        seqs = []
        ids = []
        targets = []
        
        # Sort by week
        if 'week' in df.columns:
            df = df.sort_values(['ID', 'week'])
            
        grouped = df.groupby('ID')
        for name, group in grouped:
            # Extract features
            vals = group[final_seq_cols].values
            
            # Pad/Truncate
            if len(vals) < SEQ_LEN:
                pad = np.zeros((SEQ_LEN - len(vals), len(final_seq_cols)))
                vals = np.vstack([vals, pad])
            elif len(vals) > SEQ_LEN:
                 vals = vals[:SEQ_LEN]
            
            seqs.append(vals)
            ids.append(name)
            targets.append(group['Yield'].iloc[0])
            
        return np.array(seqs), np.array(targets), np.array(ids)

    # Get sequences for both
    real_seq, real_y, real_ids = get_sequences(real_raw)
    syn_seq, syn_y, syn_ids = get_sequences(syn_weekly)
    
    # Combine
    X_seq = np.concatenate([real_seq, syn_seq], axis=0)
    y_target = np.concatenate([real_y, syn_y], axis=0)
    all_ids = np.concatenate([real_ids, syn_ids], axis=0)
    
    # Indicate Source
    # Real IDs start with "Real_"
    is_synthetic = np.array([0 if 'Real_' in x else 1 for x in all_ids])
    
    print(f"Total Sequences: {X_seq.shape}")
    
    # --- D. Prepare Tabular Features (RF Input) ---
    print("preparing tabular features...")
    
    # Re-use the aggregation logic
    def custom_agg(x):
        if 'week' in x.columns:
            x = x.sort_values('week')
        n = len(x)
        split_idx = int(n * 0.3)
        
        early_rain = x['precip'].iloc[:split_idx].sum() if 'precip' in x.columns else 0
        late_rain = x['precip'].iloc[-split_idx:].sum() if split_idx > 0 and 'precip' in x.columns else 0
        
        res = pd.Series({
            'Mean_Temp': x['temp'].mean() if 'temp' in x.columns else 0,
            'Max_Temp': x['tempmax'].max() if 'tempmax' in x.columns else 0,
            'Total_Rain': x['precip'].sum() if 'precip' in x.columns else 0,
            'Rain_Days': (x['precip'] > 0.1).sum() if 'precip' in x.columns else 0,
            'Temp_Variability': x['temp'].std() if 'temp' in x.columns else 0,
            'Mean_Humidity': x['humidity'].mean() if 'humidity' in x.columns else 0,
            'Total_GDD': x['GDD'].sum(),
            'Mean_Solar': x['solarradiation'].mean() if 'solarradiation' in x.columns else 0,
            'Early_Rain_30pct': early_rain,
            'Late_Rain_30pct': late_rain,
            'Duration_Weeks': len(x)
        })
        return res

    # Aggregate Real
    real_tab = real_raw.groupby('ID').apply(custom_agg)
    # Aggregate Syn
    syn_tab = syn_weekly.groupby('ID').apply(custom_agg)
    
    # Align with sequences using IDs
    # Create DataFrame indexed by ID
    df_tab_real = pd.DataFrame(real_tab)
    df_tab_syn = pd.DataFrame(syn_tab)
    
    df_combined = pd.concat([df_tab_real, df_tab_syn])
    
    # Reindex to match the sequence order EXACTLY
    X_tab = df_combined.loc[all_ids].reset_index(drop=True)
    
    # Final check
    assert len(X_tab) == len(X_seq)
    
    # Add meta columns for split logic
    X_tab['Is_Synthetic'] = is_synthetic
    X_tab['Yield'] = y_target # To keep consistent with original structure
    
    return X_tab, X_seq

def evaluate_predictions(y_true, y_pred, name):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return r2, mae

def train_and_select_best(df_tab, X_seq, output_model="final_best_model.joblib"):
    print("\n--- 2. TRAINING & MODEL SELECTION ---")
    
    feature_cols = [c for c in df_tab.columns if c not in ['Yield', 'Is_Synthetic']]
    
    y = df_tab['Yield'].values
    is_syn = df_tab['Is_Synthetic'].values
    
    # Split Indices
    indices = np.arange(len(df_tab))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Prepare Data Subsets
    # 1. Tabular
    X_tab_train = df_tab.iloc[train_idx][feature_cols].values
    X_tab_test = df_tab.iloc[test_idx][feature_cols].values
    
    # 2. Sequential
    X_seq_train = X_seq[train_idx]
    X_seq_test = X_seq[test_idx]
    
    # Targets
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    # Identify Real Test Data
    test_is_real = is_syn[test_idx] == 0
    
    if test_is_real.sum() == 0:
        print("Warning: No real data in test set.")
        real_mask = np.ones(len(y_test), dtype=bool) # Fallback
    else:
        real_mask = test_is_real
        print(f"Evaluating primarily on {real_mask.sum()} HELD-OUT REAL samples.")

    best_model = None
    best_score = -np.inf
    best_name = ""
    results = []

    # Extract real-only training sequences for Phase 2 fine-tuning
    train_real_mask = is_syn[train_idx] == 0
    X_real_train = X_seq_train[train_real_mask]
    y_real_train = y_train[train_real_mask]
    print(f"  Real train samples for fine-tuning: {len(X_real_train)}")

    # --- Model 1: Random Forest ---
    print("\nTraining Random Forest...")
    rf = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(X_tab_train, y_train)
    y_pred_rf = rf.predict(X_tab_test)
    r2_rf, mae_rf = evaluate_predictions(y_test[real_mask], y_pred_rf[real_mask], "Random Forest")
    print(f"  RF Real Data R²: {r2_rf:.4f} (MAE: {mae_rf:.4f})")
    
    if r2_rf > best_score:
        best_score = r2_rf
        best_model = rf
        best_name = "Random Forest"
    
    results.append({"Model": "Random Forest", "R2": r2_rf})

    # --- Model 2: XGBoost ---
    if HAS_XGB:
        print("\nTraining XGBoost...")
        xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
        xgb.fit(X_tab_train, y_train)
        y_pred_xgb = xgb.predict(X_tab_test)
        r2_xgb, mae_xgb = evaluate_predictions(y_test[real_mask], y_pred_xgb[real_mask], "XGBoost")
        print(f"  XGB Real Data R²: {r2_xgb:.4f} (MAE: {mae_xgb:.4f})")
        
        if r2_xgb > best_score:
            best_score = r2_xgb
            best_model = xgb
            best_name = "XGBoost"
        results.append({"Model": "XGBoost", "R2": r2_xgb})

    # --- Model 3: LSTM ---
    print("\nTraining LSTM...")
    input_dim = X_seq.shape[2]
    lstm = RiceLSTM(input_dim)
    lstm, y_pred_lstm = train_dl_model(lstm, X_seq_train, y_train, X_seq_test, y_test,
                                        X_real=X_real_train, y_real=y_real_train)
    r2_lstm, mae_lstm = evaluate_predictions(y_test[real_mask], y_pred_lstm[real_mask], "LSTM")
    print(f"  LSTM Real Data R²: {r2_lstm:.4f} (MAE: {mae_lstm:.4f})")
    
    if r2_lstm > best_score:
        best_score = r2_lstm
        best_model = lstm
        best_name = "LSTM"
    results.append({"Model": "LSTM", "R2": r2_lstm})

    # --- Model 4: 1D-CNN ---
    print("\nTraining 1D-CNN...")
    cnn = RiceCNN1D(input_dim, seq_len=X_seq.shape[1])
    cnn, y_pred_cnn = train_dl_model(cnn, X_seq_train, y_train, X_seq_test, y_test,
                                      finetune_on_real=False)  # CNN unstable with tiny fine-tune batches
    r2_cnn, mae_cnn = evaluate_predictions(y_test[real_mask], y_pred_cnn[real_mask], "1D-CNN")
    print(f"  CNN Real Data R²: {r2_cnn:.4f} (MAE: {mae_cnn:.4f})")
    
    if r2_cnn > best_score:
        best_score = r2_cnn
        best_model = cnn
        best_name = "1D-CNN"
    results.append({"Model": "1D-CNN", "R2": r2_cnn, "MAE": mae_cnn})

    # --- Model 5: Transformer ---
    print("\nTraining Transformer...")
    transformer = RiceTransformer(input_dim, seq_len=X_seq.shape[1])
    transformer, y_pred_trans = train_dl_model(transformer, X_seq_train, y_train, X_seq_test, y_test,
                                                X_real=X_real_train, y_real=y_real_train)
    r2_trans, mae_trans = evaluate_predictions(y_test[real_mask], y_pred_trans[real_mask], "Transformer")
    print(f"  Transformer Real Data R\u00b2: {r2_trans:.4f} (MAE: {mae_trans:.4f})")

    if r2_trans > best_score:
        best_score = r2_trans
        best_model = transformer
        best_name = "Transformer"
    results.append({"Model": "Transformer", "R2": r2_trans, "MAE": mae_trans})

    print(f"\n*** WINNER: {best_name} with R2 {best_score:.4f} on Real Data ***")
    
    # Save Model
    if "Random Forest" in best_name or "XGBoost" in best_name:
        joblib.dump(best_model, output_model)
        # Plot Feature Importance
        importances = best_model.feature_importances_
        imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(10, 6))
        top10 = imp_df.head(10)
        plt.barh(top10['Feature'][::-1], top10['Importance'][::-1], color='steelblue')
        plt.title(f'Feature Importance ({best_name})')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
    else:
        # Save PyTorch Model
        torch.save(best_model.state_dict(), "final_best_model.pth")
        print("Saved PyTorch model to final_best_model.pth")

    print(f"\nSaved best model.")

if __name__ == "__main__":
    real_file = "rice.csv"
    # Note: We now need the WEEKLY synthetic data for DL
    synthetic_weekly_file = "synthetic_weekly_rice.csv" 
    # Use best_synthetic_rice for Tabular if needed, but we can generate tabular from weekly to keep alignment
    
    df_tab, X_seq = load_and_prepare_data(real_file, None, synthetic_weekly_file)
    if df_tab is not None:
        train_and_select_best(df_tab, X_seq)

