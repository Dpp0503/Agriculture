import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# --- ONE: Data Preparation ---
def load_and_prep_data():
    print("Loading data...")
    # 1. Load Tabular Data (Target & Stage Features)
    df_tab = pd.read_csv('d:/Rice/advanced_features.csv')
    df_tab = df_tab.sort_values('ID').reset_index(drop=True)
    
    # 2. Load Raw Sequence Data
    real_df = pd.read_csv('d:/Rice/rice.csv')
    syn_df = pd.read_csv('d:/Rice/synthetic_weekly_rice.csv')
    
    # Standardize & Combine Raw Data
    real_df['Is_Synthetic'] = 0
    # Replicate ID logic from advanced_feature_engineering.py
    # Note: real_df['District'] + '_' + real_df['Year']
    real_df['ID'] = 'Real_' + real_df['District'] + '_' + real_df['Year'].astype(str) + '_0'
    
    # Synthetic ID
    # In earlier script: 
    # combined_df['ID'] = combined_df['Synthetic_ID'].fillna(combined_df['District'] + '_' + combined_df['Year'].astype(str))
    # Let's check columns first
    if 'Yield' in syn_df.columns and 'Yield (Ton./Ha.)' not in syn_df.columns:
        syn_df.rename(columns={'Yield': 'Yield (Ton./Ha.)'}, inplace=True)

    # We need to reconstruct the ID EXACTLY as in advanced_features.csv
    # In advanced_feature_engineering.py:
    # combined_df['ID'] = combined_df['Synthetic_ID'].fillna(...)
    # Let's apply the same logic.
    
    # Combine first to be safe
    common_cols = list(set(real_df.columns) & set(syn_df.columns))
    # Ensure ID is not in common yet if it's not in synthetic
    if 'ID' in common_cols: common_cols.remove('ID')
    
    raw_df = pd.concat([real_df, syn_df], axis=0, ignore_index=True)
    
    # Re-create ID
    if 'Synthetic_ID' not in raw_df.columns:
        raw_df['Synthetic_ID'] = np.nan
        
    # Logic from advanced_feature_engineering.py lines 120-124:
    # combined_df['ID'] = combined_df['Synthetic_ID'].fillna(combined_df['District'] + '_' + combined_df['Year'].astype(str))
    # Wait, the previous script used a lambda with Is_Synthetic for the fillna part?
    # Let's stick to the resulting 'ID' in df_tab. We just need to attach Sequence to it.
    
    # The safest way is to generate ID using the exact same columns if possible.
    # Or, we can do a merge?
    # Let's try to reconstruct ID logic carefully.
    
    # Real ID: Real_{District}_{Year} (Wait, check generated file)
    # The previous script did: 
    # real_df['Synthetic_ID'] = 'Real_' + ...
    # combined['ID'] = combined['Synthetic_ID'].fillna(...)
    
    # So for Real data, ID came from Synthetic_ID column which was manually created.
    # For Synthetic data, it came from Synthetic_ID column.
    
    # Let's repeat that:
    raw_df['Seq_ID'] = raw_df['Synthetic_ID']
    # If Synthetic_ID is NaN (Real Data maybe?), fill it.
    # But wait, we need to distinguish Real vs Synthetic rows easily.
    
    # Let's verify what IDs look like in df_tab
    print(f"Sample IDs in Tabular Data: {df_tab['ID'].head().tolist()}")
    
    # We can create a mapping key in raw_df: District + Year + Is_Synthetic.
    # And same in df_tab. Then map.
    
    # Actually, let's just create a composite key for joining.
    # Key: District, Year, Is_Synthetic (if available), or just District, Year, Yield?
    # Unique ID logic:
    # Real: District, Year
    # Synthetic: Synthetic_ID
    
    # Let's try to pass the raw data through the same pipeline? No, too slow.
    
    # Quick fix: Re-construct ID for raw_df rows.
    # For Real rows: ID = 'Real_' + District + '_' + Year
    # For Synthetic rows: ID = Synthetic_ID (assuming it exists)
    
    # Indices for Real Data in raw_df
    # We loaded real_df then syn_df.
    # real_df['ID'] = ...
    
    # Let's process separately then concat.
    real_df['ID'] = 'Real_' + real_df['District'] + '_' + real_df['Year'].astype(str)
    
    # Synthetic
    # If Synthetic_ID exists use it, else make one? 
    # The previous script: combined_df['ID'] = combined_df['Synthetic_ID'].fillna(...)
    # and for real df it set real_df['Synthetic_ID'] = 'Real_' ...
    # So basically ID IS Synthetic_ID.
    
    if 'Synthetic_ID' in syn_df.columns:
        syn_df['ID'] = syn_df['Synthetic_ID']
    else:
        # Fallback
        syn_df['ID'] = 'Syn_' + syn_df['District'] + '_' + syn_df['Year'].astype(str)
        
    # Concat
    cols_needed = ['ID', 'week', 'precip', 'temp', 'tempmax', 'tempmin', 'humidity', 'solarradiation', 'District', 'Year', 'Is_Synthetic']
    raw_df = pd.concat([real_df[cols_needed], syn_df[cols_needed]], axis=0)
    
    # Filter to only IDs in df_tab - actually we don't need this filter if we match by keys later
    # valid_ids = set(df_tab['ID'].unique())
    # raw_df = raw_df[raw_df['ID'].isin(valid_ids)]
    
    # Pivot to Sequences
    # We need (N_samples, Sequence_Length, N_features)
    # Sort by ID, week
    raw_df = raw_df.sort_values(['ID', 'week'])
    
    # Group and pad
    features = ['precip', 'temp', 'tempmax', 'tempmin', 'humidity', 'solarradiation']
    seq_len = 20
    
    X_seq_list = []
    X_tab_list = [] # New list for tabular data
    y_list = []
    ids_list = []
    
    # Drop columns to prepare feature row
    drop_cols = ['ID', 'District', 'Year', 'Yield (Ton./Ha.)', 'Yield', 'Synthetic_ID', 'Is_Synthetic', 'Season', 'Crop', 'duration', 'Area', 'Production']
    tab_cols = [c for c in df_tab.columns if c not in drop_cols]
    
    # Create Composite Key for matching
    # Key: (District, Year, Is_Synthetic)
    
    # 3. Create Keys matching
    # df_tab has these columns.
    # raw_df has these columns.
    
    # Group raw_df by these 3 columns
    # Ensure types match. Year is int. Is_Synthetic is int. District is str.
    keys = ['District', 'Year', 'Is_Synthetic']
    
    grouped = raw_df.groupby(keys)
    
    print(f"Number of groups in Raw Data: {len(grouped)}")
    print(f"Number of rows in Tabular Data: {len(df_tab)}")
    
    X_seq_list = []
    X_tab_list = [] 
    y_list = []
    
    # Drop columns to prepare feature row
    drop_cols = ['ID', 'District', 'Year', 'Yield (Ton./Ha.)', 'Yield', 'Synthetic_ID', 'Is_Synthetic', 'Season', 'Crop', 'duration', 'Area', 'Production']
    tab_cols = [c for c in df_tab.columns if c not in drop_cols]
    
    count_matched = 0
    
    for _, row in df_tab.iterrows():
        # Create key tuple
        key = (row['District'], row['Year'], row['Is_Synthetic'])
        
        if key in grouped.groups:
            group = grouped.get_group(key)
            
            # Sort by week
            group = group.sort_values('week')
            
            # Extract array
            arr = group[features].values
            
            # Pad/Truncate
            if len(arr) < seq_len:
                # Pad with zeros
                pad = np.zeros((seq_len - len(arr), len(features)))
                arr = np.concatenate([arr, pad], axis=0)
            elif len(arr) > seq_len:
                # Truncate
                arr = arr[:seq_len]
                
            X_seq_list.append(arr)
            
            # Extract tabular features
            tab_row = row[tab_cols].values
            tab_row = np.nan_to_num(pd.to_numeric(tab_row, errors='coerce'))
            X_tab_list.append(tab_row)
            
            y_list.append(row['Yield (Ton./Ha.)'])
            count_matched += 1
            
    print(f"Matched {count_matched} samples out of {len(df_tab)}")
            
    X_seq = np.array(X_seq_list)
    X_tab = np.array(X_tab_list) # Now strictly aligned
    y = np.array(y_list)
    
    print(f"Sequence Shape: {X_seq.shape}")
    print(f"Tabular Shape: {X_tab.shape}")
    print(f"Target Shape: {y.shape}")
    
    return X_seq, X_tab, y, tab_cols

# --- TWO: Models ---



# 2. Deep Learning Wrappers
class RiceRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(RiceRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        _, h_n = self.rnn(x)
        return self.fc(h_n[-1]) # Last hidden state

class RiceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(RiceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

class RiceCNN(nn.Module):
    def __init__(self, input_dim, seq_len):
        super(RiceCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        # Flatten size calculation:
        # L=20 -> Pool -> 10 -> Pool -> 5
        self.flat_dim = 32 * (seq_len // 4)
        self.fc = nn.Linear(self.flat_dim, 1)
        
    def forward(self, x):
        # Swap dims for Conv1d: (Batch, Seq, Feat) -> (Batch, Feat, Seq)
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

class RiceTransformer(nn.Module):
    def __init__(self, input_dim, seq_len):
        super(RiceTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, 32)
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(32 * seq_len, 1) # Flatten all
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

def train_dl_model(model_class, X_train, y_train, X_test, y_test, name, input_dim, seq_len):
    print(f"Training {name}...")
    
    # Scale Inputs
    # For sequences, scale per feature across all time/samples
    N, L, F = X_train.shape
    X_train_flat = X_train.reshape(-1, F)
    X_test_flat = X_test.reshape(-1, F)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(N, L, F)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape[0], L, F)
    
    # Tensor conversion
    Xt = torch.FloatTensor(X_train_scaled)
    yt = torch.FloatTensor(y_train).view(-1, 1)
    Xv = torch.FloatTensor(X_test_scaled)
    yv = torch.FloatTensor(y_test).view(-1, 1)
    
    dataset = TensorDataset(Xt, yt)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Init Model
    if name == 'CNN' or name == 'Transformer':
        model = model_class(input_dim, seq_len)
    else:
        model = model_class(input_dim)
        
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    epochs = 50
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            
    # Eval
    model.eval()
    with torch.no_grad():
        pred = model(Xv).numpy().flatten()
        
    r2 = r2_score(y_test, pred)
    return r2, pred

# --- THREE: Main Comparison ---
def main():
    X_seq, X_tab, y, tab_features = load_and_prep_data()
    
    # Split Indices
    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)
    
    y_test = y[idx_test]
    
    results = {}
    predictions = {}
    

    
    # DL Params
    input_dim = X_seq.shape[2]
    seq_len = X_seq.shape[1]
    
    # 2. RNN
    r2_rnn, pred_rnn = train_dl_model(RiceRNN, X_seq[idx_train], y[idx_train], X_seq[idx_test], y[idx_test], 'RNN', input_dim, seq_len)
    results['RNN'] = r2_rnn
    predictions['RNN'] = pred_rnn
    print(f"RNN R2: {r2_rnn:.4f}")
    
    # 3. LSTM
    r2_lstm, pred_lstm = train_dl_model(RiceLSTM, X_seq[idx_train], y[idx_train], X_seq[idx_test], y[idx_test], 'LSTM', input_dim, seq_len)
    results['LSTM'] = r2_lstm
    predictions['LSTM'] = pred_lstm
    print(f"LSTM R2: {r2_lstm:.4f}")
    
    # 4. CNN
    r2_cnn, pred_cnn = train_dl_model(RiceCNN, X_seq[idx_train], y[idx_train], X_seq[idx_test], y[idx_test], 'CNN', input_dim, seq_len)
    results['CNN'] = r2_cnn
    predictions['CNN'] = pred_cnn
    print(f"CNN R2: {r2_cnn:.4f}")
    
    # 5. Transformer
    r2_trans, pred_trans = train_dl_model(RiceTransformer, X_seq[idx_train], y[idx_train], X_seq[idx_test], y[idx_test], 'Transformer', input_dim, seq_len)
    results['Transformer'] = r2_trans
    predictions['Transformer'] = pred_trans
    print(f"Transformer R2: {r2_trans:.4f}")
    
    # --- Plotting ---
    # Bar Chart
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    scores = list(results.values())
    colors = plt.cm.viridis([i / len(names) for i in range(len(names))])
    plt.bar(names, scores, color=colors)
    plt.title('Model Comparison ($R^2$ Score)')
    plt.ylabel('$R^2$')
    plt.ylim(-0.5, 1.0)
    plt.savefig('d:/Rice/comprehensive_model_comparison.png')
    plt.close()
    
    # Scatter Plot of Best DL vs Actual
    # Find best DL
    dl_scores = {k:v for k,v in results.items()}
    best_dl_name = max(dl_scores, key=dl_scores.get)
    best_dl_pred = predictions[best_dl_name]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, best_dl_pred, alpha=0.6, label=f'{best_dl_name} ($R^2={results[best_dl_name]:.2f}$)', color='green', marker='x')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Yield')
    plt.ylabel('Predicted Yield')
    plt.title(f'Actual vs Predicted: Best DL ({best_dl_name})')
    plt.legend()
    plt.savefig('d:/Rice/best_dl_vs_actual.png')
    plt.close()
    
    # Save Report
    with open('d:/Rice/comprehensive_report.txt', 'w') as f:
        f.write("Model Comparison Results:\n")
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")

if __name__ == "__main__":
    main()
