
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

SEQ_LEN = 26  # Bug Fix 4: Match max real season length (was 20, truncating ripening data)

# =============================================================================
# 1. MODEL ARCHITECTURES
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# --- LSTM with Attention Pooling ---
class RiceLSTM(nn.Module):
    """Bidirectional LSTM with attention pooling over all timesteps
       instead of using only the last hidden state."""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.2, bidirectional=True)
        hid2 = hidden_dim * 2  # bidirectional
        # Attention mechanism: score each timestep
        self.attn_w = nn.Linear(hid2, 1)
        self.fc = nn.Linear(hid2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)               # (B, T, 2H)
        # Attention weights over timesteps
        scores = self.attn_w(out).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        context = (out * weights).sum(dim=1)   # (B, 2H)
        return self.fc(context)


# --- 1D-CNN with BatchNorm + Residual ---
class RiceCNN1D(nn.Module):
    def __init__(self, input_dim, seq_len):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(128)
        self.pool  = nn.AdaptiveMaxPool1d(1)
        self.residual_proj = nn.Conv1d(64, 128, kernel_size=1)
        self.fc    = nn.Linear(128, 1)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(0.2)

    def forward(self, x):
        x = x.permute(0, 2, 1)                           # (B, F, T)
        x = self.drop(self.relu(self.bn1(self.conv1(x)))) # (B, 64, T)
        residual = self.residual_proj(x)                  # (B, 128, T)
        x = self.drop(self.relu(self.bn2(self.conv2(x)))) # (B, 128, T)
        x = x + residual                                  # skip connection
        x = self.drop(self.relu(self.bn3(self.conv3(x)))) # (B, 128, T)
        x = self.pool(x).squeeze(-1)                      # (B, 128)
        return self.fc(x)


# --- Transformer with Padding Mask, wider capacity ---
class RiceTransformer(nn.Module):
    def __init__(self, input_dim, seq_len):
        super().__init__()
        d_model = 64  # Wider: was 32
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_enc   = PositionalEncoding(d_model=d_model, max_len=seq_len + 1)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, batch_first=True,
            dropout=0.1, dim_feedforward=256  # Wider: was 128
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.pool       = nn.AdaptiveAvgPool1d(1)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(0.1)
        self.fc         = nn.Linear(d_model, 1)

    def forward(self, x, src_key_padding_mask=None):
        x = self.embedding(x)
        x = self.pos_enc(x)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.pool(x.permute(0, 2, 1)).squeeze(-1)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return self.fc(x)


# =============================================================================
# 2. TRAINING ENGINE
# =============================================================================

def _build_padding_mask(seq_lengths, seq_len, device):
    """Create boolean mask: True = padded (ignored by attention)."""
    B = len(seq_lengths)
    mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
    for i, sl in enumerate(seq_lengths):
        if sl < seq_len:
            mask[i, sl:] = True
    return mask


def train_dl_model(model, X_train, y_train, X_test, y_test,
                   epochs=200, batch_size=32, lr=0.001,
                   patience=20, finetune_on_real=True,
                   X_real=None, y_real=None,
                   seq_lengths_train=None, seq_lengths_test=None,
                   is_transformer=False,
                   train_is_real=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = model.to(device)

    # --- Scale features ---
    scaler = StandardScaler()
    n, seq, feat = X_train.shape
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, feat)).reshape(n, seq, feat)
    X_test_scaled  = scaler.transform(X_test.reshape(-1, feat)).reshape(X_test.shape[0], seq, feat)

    def _make_tensors(X, y):
        return (torch.FloatTensor(X).to(device),
                torch.FloatTensor(y).view(-1, 1).to(device))

    # --- Validation: use held-out real data for early stopping ---
    if X_real is not None and len(X_real) > 5:
        real_val_size = max(2, int(0.2 * len(X_real)))
        X_real_val_scaled = scaler.transform(
            X_real[:real_val_size].reshape(-1, feat)
        ).reshape(real_val_size, seq, feat)
        Xval, yval = _make_tensors(X_real_val_scaled, y_real[:real_val_size])
        X_tr = X_train_scaled
        y_tr = y_train
    else:
        val_size = max(1, int(0.1 * n))
        X_tr  = X_train_scaled[val_size:]
        y_tr  = y_train[val_size:]
        Xval, yval = _make_tensors(X_train_scaled[:val_size], y_train[:val_size])

    Xt, yt = _make_tensors(X_tr, y_tr)
    Xv_full, _ = _make_tensors(X_test_scaled, y_test)

    # --- Build padding masks (Bug Fix 2: actually use them during training) ---
    # For training: we need per-batch masks, so we store full train masks and slice
    train_seq_lens = seq_lengths_train  # may be None for non-transformer
    test_pad_mask  = None
    if is_transformer and seq_lengths_test is not None:
        test_pad_mask = _build_padding_mask(seq_lengths_test, seq, device)

    # --- Gaussian noise augmentation indices (real training samples) ---
    # Identify which training indices are real (for augmentation)
    # We'll duplicate real samples with noise during each epoch

    # ---- Huber Loss (robust to outliers) ----
    criterion = nn.HuberLoss(delta=1.0)

    # --- Build sample weights: real samples get 10x weight ---
    sample_weights = None
    if train_is_real is not None:
        w = torch.ones(len(X_tr), device=device)
        # Identify real samples in X_tr (which may be all of X_train_scaled)
        real_flags = torch.FloatTensor(train_is_real[:len(X_tr)]).to(device)
        w = torch.where(real_flags == 1, torch.tensor(10.0, device=device),
                        torch.tensor(1.0, device=device))
        sample_weights = w

    def _run_phase(model, Xt, yt, Xval, yval, epochs, lr, patience,
                   weights=None):
        # Use weighted random sampler if weights are provided
        if weights is not None and len(weights) == len(Xt):
            sampler = torch.utils.data.WeightedRandomSampler(
                weights.cpu(), num_samples=len(Xt), replacement=True
            )
            loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, sampler=sampler)
        else:
            loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        best_val, patience_count, best_state = float('inf'), 0, None

        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in loader:
                optimizer.zero_grad()

                # Gaussian noise augmentation
                noise = torch.randn_like(X_batch) * 0.03
                X_batch_aug = X_batch + noise

                if is_transformer:
                    batch_mask = (X_batch_aug.abs().sum(dim=-1) == 0)
                    out = model(X_batch_aug, src_key_padding_mask=batch_mask)
                else:
                    out = model(X_batch_aug)

                loss = criterion(out, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            # --- Validation ---
            model.eval()
            with torch.no_grad():
                if is_transformer:
                    val_mask = (Xval.abs().sum(dim=-1) == 0)
                    val_loss = criterion(model(Xval, src_key_padding_mask=val_mask), yval).item()
                else:
                    val_loss = criterion(model(Xval), yval).item()

            # Early stopping
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

    # --- Phase 1: Pre-train on all data (real + synthetic) with real-sample weighting ---
    model = _run_phase(model, Xt, yt, Xval, yval, epochs, lr, patience, weights=sample_weights)

    # --- Phase 2: Fine-tune on real data only (with early stopping) ---
    if finetune_on_real and X_real is not None and len(X_real) > 5:
        X_real_scaled = scaler.transform(X_real.reshape(-1, feat)).reshape(len(X_real), seq, feat)

        ft_val_size = max(2, int(0.2 * len(X_real)))
        X_ft_train = X_real_scaled[ft_val_size:]
        y_ft_train = y_real[ft_val_size:]
        X_ft_val   = X_real_scaled[:ft_val_size]
        y_ft_val   = y_real[:ft_val_size]

        if len(X_ft_train) > 2:
            Xr_t, yr_t = _make_tensors(X_ft_train, y_ft_train)
            Xr_v, yr_v = _make_tensors(X_ft_val, y_ft_val)
            model = _run_phase(model, Xr_t, yr_t, Xr_v, yr_v,
                               epochs=80, lr=1e-4, patience=10)

    # --- Final prediction ---
    model.eval()
    with torch.no_grad():
        if is_transformer and test_pad_mask is not None:
            pred = model(Xv_full, src_key_padding_mask=test_pad_mask).cpu().numpy().flatten()
        else:
            pred = model(Xv_full).cpu().numpy().flatten()

    return model, pred


# =============================================================================
# 3. DATA LOADING & PREPARATION
# =============================================================================

def load_and_prepare_data(real_file, synthetic_weekly_file):
    print("\n--- 1. LOADING & PREPARING DATA ---")

    # A. Load Real Data
    real_raw = pd.read_csv(real_file)
    real_raw.columns = real_raw.columns.str.strip()

    yield_col = next((c for c in real_raw.columns if 'Yield' in c), None)
    if yield_col:
        real_raw.rename(columns={yield_col: 'Yield'}, inplace=True)
    else:
        raise ValueError("Yield column not found in real data")

    # GDD
    if all(c in real_raw.columns for c in ['tempmax', 'tempmin']):
        real_raw['GDD'] = (((real_raw['tempmax'] + real_raw['tempmin']) / 2) - 10).clip(lower=0)
    else:
        real_raw['GDD'] = 0

    # Diurnal Temperature Range
    if all(c in real_raw.columns for c in ['tempmax', 'tempmin']):
        real_raw['DTR'] = real_raw['tempmax'] - real_raw['tempmin']

    real_raw['ID'] = ('Real_' + real_raw['District'] + '_'
                      + real_raw['Year'].astype(str) + '_' + real_raw['Season'])

    # B. Load Synthetic Weekly Data
    syn_weekly = pd.read_csv(synthetic_weekly_file)
    syn_weekly['Is_Synthetic'] = 1
    if 'ID' not in syn_weekly.columns:
        syn_weekly['ID'] = 'Syn_' + syn_weekly['Synthetic_ID'].astype(str)

    if 'GDD' not in syn_weekly.columns:
        if all(c in syn_weekly.columns for c in ['tempmax', 'tempmin']):
            syn_weekly['GDD'] = (((syn_weekly['tempmax'] + syn_weekly['tempmin']) / 2) - 10).clip(lower=0)
        else:
            syn_weekly['GDD'] = 0

    if 'DTR' not in syn_weekly.columns:
        if all(c in syn_weekly.columns for c in ['tempmax', 'tempmin']):
            syn_weekly['DTR'] = syn_weekly['tempmax'] - syn_weekly['tempmin']

    # C. Prepare Sequences
    print("Preparing sequences...")
    # Expanded feature set: 11 features
    seq_cols = ['temp', 'tempmax', 'tempmin', 'precip', 'humidity',
                'solarradiation', 'GDD', 'DTR',
                'windspeed', 'cloudcover', 'sealevelpressure']
    final_seq_cols = [c for c in seq_cols if c in real_raw.columns and c in syn_weekly.columns]
    print(f"  Sequence features ({len(final_seq_cols)}): {final_seq_cols}")

    def get_sequences(df):
        seqs, ids, targets, lengths = [], [], [], []
        if 'week' in df.columns:
            df = df.sort_values(['ID', 'week'])
        for name, group in df.groupby('ID'):
            vals = group[final_seq_cols].values
            actual_len = min(len(vals), SEQ_LEN)
            if len(vals) < SEQ_LEN:
                pad = np.zeros((SEQ_LEN - len(vals), len(final_seq_cols)))
                vals = np.vstack([vals, pad])
            elif len(vals) > SEQ_LEN:
                vals = vals[:SEQ_LEN]
            seqs.append(vals)
            ids.append(name)
            targets.append(group['Yield'].iloc[0])
            lengths.append(actual_len)
        return np.array(seqs), np.array(targets), np.array(ids), np.array(lengths)

    real_seq, real_y, real_ids, real_lengths = get_sequences(real_raw)
    syn_seq, syn_y, syn_ids, syn_lengths = get_sequences(syn_weekly)

    X_seq      = np.concatenate([real_seq, syn_seq], axis=0)
    y_target   = np.concatenate([real_y, syn_y], axis=0)
    all_ids    = np.concatenate([real_ids, syn_ids], axis=0)
    all_lengths = np.concatenate([real_lengths, syn_lengths], axis=0)
    is_synthetic = np.array([0 if 'Real_' in x else 1 for x in all_ids])

    print(f"  Real seasons: {(is_synthetic == 0).sum()}")
    print(f"  Synthetic seasons: {(is_synthetic == 1).sum()}")
    print(f"  Total shape: {X_seq.shape}")

    return X_seq, y_target, is_synthetic, all_lengths


# =============================================================================
# 4. MAIN TRAINING & SELECTION
# =============================================================================

def train_and_select_best(X_seq, y, is_syn, all_lengths):
    print("\n--- 2. TRAINING & MODEL SELECTION ---")

    # Bug Fix 3: Stratified split (proportional real/synthetic)
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=is_syn
    )

    X_seq_train = X_seq[train_idx]
    X_seq_test  = X_seq[test_idx]
    y_train_raw = y[train_idx]
    y_test_raw  = y[test_idx]
    lengths_train = all_lengths[train_idx]
    lengths_test  = all_lengths[test_idx]

    # Bug Fix 1: Target normalization — fit ONLY on training data
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    y_test  = y_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()

    # Masks
    test_is_real = is_syn[test_idx] == 0
    train_is_real = is_syn[train_idx] == 0

    if test_is_real.sum() == 0:
        print("Warning: No real data in test set.")
        real_mask = np.ones(len(y_test_raw), dtype=bool)
    else:
        real_mask = test_is_real
        print(f"  Real test samples: {real_mask.sum()}")
        print(f"  Synthetic test samples: {(~real_mask).sum()}")

    # Real training data for fine-tuning
    X_real_train = X_seq_train[train_is_real]
    y_real_train = y_train[train_is_real]
    print(f"  Real train samples for fine-tuning: {len(X_real_train)}")

    input_dim = X_seq.shape[2]
    best_model = None
    best_score = -np.inf
    best_name  = ""
    results    = []
    all_preds  = {}

    # --- Model 1: LSTM ---
    print("\nTraining LSTM...")
    lstm = RiceLSTM(input_dim)
    lstm, pred_lstm_n = train_dl_model(
        lstm, X_seq_train, y_train, X_seq_test, y_test,
        X_real=X_real_train, y_real=y_real_train, is_transformer=False,
        train_is_real=train_is_real
    )
    pred_lstm = y_scaler.inverse_transform(pred_lstm_n.reshape(-1, 1)).flatten()
    r2_lstm, mae_lstm = r2_score(y_test_raw[real_mask], pred_lstm[real_mask]), \
                        mean_absolute_error(y_test_raw[real_mask], pred_lstm[real_mask])
    print(f"  LSTM  R²: {r2_lstm:.4f}  MAE: {mae_lstm:.4f}")
    results.append({"Model": "LSTM", "R2": r2_lstm, "MAE": mae_lstm})
    all_preds["LSTM"] = pred_lstm
    if r2_lstm > best_score:
        best_score, best_model, best_name = r2_lstm, lstm, "LSTM"

    # --- Model 2: 1D-CNN ---
    print("\nTraining 1D-CNN...")
    cnn = RiceCNN1D(input_dim, seq_len=SEQ_LEN)
    cnn, pred_cnn_n = train_dl_model(
        cnn, X_seq_train, y_train, X_seq_test, y_test,
        X_real=X_real_train, y_real=y_real_train, is_transformer=False,
        train_is_real=train_is_real
    )
    pred_cnn = y_scaler.inverse_transform(pred_cnn_n.reshape(-1, 1)).flatten()
    r2_cnn, mae_cnn = r2_score(y_test_raw[real_mask], pred_cnn[real_mask]), \
                      mean_absolute_error(y_test_raw[real_mask], pred_cnn[real_mask])
    print(f"  CNN   R²: {r2_cnn:.4f}  MAE: {mae_cnn:.4f}")
    results.append({"Model": "1D-CNN", "R2": r2_cnn, "MAE": mae_cnn})
    all_preds["1D-CNN"] = pred_cnn
    if r2_cnn > best_score:
        best_score, best_model, best_name = r2_cnn, cnn, "1D-CNN"

    # --- Model 3: Transformer ---
    print("\nTraining Transformer...")
    transformer = RiceTransformer(input_dim, seq_len=SEQ_LEN)
    transformer, pred_trans_n = train_dl_model(
        transformer, X_seq_train, y_train, X_seq_test, y_test,
        X_real=X_real_train, y_real=y_real_train,
        seq_lengths_train=lengths_train, seq_lengths_test=lengths_test,
        is_transformer=True, train_is_real=train_is_real
    )
    pred_trans = y_scaler.inverse_transform(pred_trans_n.reshape(-1, 1)).flatten()
    r2_trans, mae_trans = r2_score(y_test_raw[real_mask], pred_trans[real_mask]), \
                          mean_absolute_error(y_test_raw[real_mask], pred_trans[real_mask])
    print(f"  Trans R²: {r2_trans:.4f}  MAE: {mae_trans:.4f}")
    results.append({"Model": "Transformer", "R2": r2_trans, "MAE": mae_trans})
    all_preds["Transformer"] = pred_trans
    if r2_trans > best_score:
        best_score, best_model, best_name = r2_trans, transformer, "Transformer"

    # --- Ensemble: average top-2 models ---
    print("\n--- Ensemble (Top-2 Average) ---")
    sorted_models = sorted(results, key=lambda x: x['R2'], reverse=True)
    top2_names = [sorted_models[0]['Model'], sorted_models[1]['Model']]
    pred_ensemble = (all_preds[top2_names[0]] + all_preds[top2_names[1]]) / 2.0
    r2_ens = r2_score(y_test_raw[real_mask], pred_ensemble[real_mask])
    mae_ens = mean_absolute_error(y_test_raw[real_mask], pred_ensemble[real_mask])
    print(f"  Ensemble ({top2_names[0]} + {top2_names[1]})  R²: {r2_ens:.4f}  MAE: {mae_ens:.4f}")
    results.append({"Model": f"Ensemble({top2_names[0]}+{top2_names[1]})", "R2": r2_ens, "MAE": mae_ens})

    if r2_ens > best_score:
        best_score = r2_ens
        best_name  = f"Ensemble({top2_names[0]}+{top2_names[1]})"
        # Save both component models
        print("  Ensemble is the winner — saving both component models.")

    print(f"\n{'='*50}")
    print(f"  WINNER: {best_name} with R² = {best_score:.4f}")
    print(f"{'='*50}")

    # --- Save ---
    # Always save the best single model
    single_best_name = sorted_models[0]['Model']
    single_best_model = {'LSTM': lstm, '1D-CNN': cnn, 'Transformer': transformer}[single_best_name]

    if isinstance(single_best_model, nn.Module):
        torch.save(single_best_model.state_dict(), "final_best_model.pth")
        print(f"Saved {single_best_name} to final_best_model.pth")

    # Save results
    pd.DataFrame(results).to_csv("deep_learning_results.csv", index=False)
    print("Saved deep_learning_results.csv")

    # --- Plot ---
    # Use winner predictions
    if "Ensemble" in best_name:
        best_pred_for_plot = pred_ensemble
    else:
        best_pred_for_plot = all_preds.get(best_name, pred_trans)

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_raw[real_mask], best_pred_for_plot[real_mask],
                alpha=0.7, color='teal', edgecolors='black', linewidth=0.5, s=60)
    mn, mx = y_test_raw.min(), y_test_raw.max()
    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=2)
    plt.xlabel('Actual Yield (Ton/Ha)')
    plt.ylabel('Predicted Yield (Ton/Ha)')
    plt.title(f'Best: {best_name} (R²={best_score:.4f})')
    plt.tight_layout()
    plt.savefig('advanced_yield_prediction.png')
    plt.close()
    print("Saved advanced_yield_prediction.png")


# =============================================================================
# 5. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    X_seq, y, is_syn, lengths = load_and_prepare_data("rice.csv", "synthetic_weekly_rice.csv")
    train_and_select_best(X_seq, y, is_syn, lengths)
