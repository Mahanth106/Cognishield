# src/models.py
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# 1. Deep Autoencoder for Reconstruction Error
class BehaviorAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 4):
        super(BehaviorAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(X: np.ndarray, epochs: int = 60, batch_size: int = 16, lr: float = 0.005) -> BehaviorAutoencoder:
    """
    Trains unsupervised Autoencoder on normalized behavioral features.
    """
    input_dim = X.shape[1]
    model = BehaviorAutoencoder(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    tensor_data = torch.tensor(X, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(tensor_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for _ in range(epochs):
        for batch in loader:
            batch_x = batch[0]
            if batch_x.size(0) <= 1:
                continue
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()

    return model

def compute_autoencoder_scores(model: BehaviorAutoencoder, X: np.ndarray) -> np.ndarray:
    model.eval()
    tensor_data = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        recon = model(tensor_data)
        # Mean Squared Error across feature dimensions per sample
        mse = torch.mean((tensor_data - recon) ** 2, dim=1).numpy()
    
    # Min-max scale into 0 to 1 range
    scaler = MinMaxScaler()
    norm_scores = scaler.fit_transform(mse.reshape(-1, 1)).flatten()
    return norm_scores

# 2. Sequence LSTM Model for Temporal Modeling
class TemporalLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super(TemporalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.regressor = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        # Predict the next time step's feature vector
        last_step = lstm_out[:, -1, :]
        prediction = self.regressor(last_step)
        return prediction

def compute_lstm_sequence_scores(X: np.ndarray, seq_len: int = 3) -> np.ndarray:
    """
    Creates rolling multi-day sequences and computes transition prediction deviations.
    """
    n_samples, n_features = X.shape
    if n_samples <= seq_len:
        return np.zeros(n_samples)

    sequences = []
    targets = []
    for i in range(n_samples - seq_len):
        sequences.append(X[i:i+seq_len])
        targets.append(X[i+seq_len])

    seq_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
    target_tensor = torch.tensor(np.array(targets), dtype=torch.float32)

    lstm_model = TemporalLSTM(input_dim=n_features)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    lstm_model.train()
    for _ in range(35):
        optimizer.zero_grad()
        preds = lstm_model(seq_tensor)
        loss = criterion(preds, target_tensor)
        loss.backward()
        optimizer.step()

    lstm_model.eval()
    with torch.no_grad():
        preds = lstm_model(seq_tensor)
        seq_mse = torch.mean((target_tensor - preds) ** 2, dim=1).numpy()

    scaler = MinMaxScaler()
    norm_seq_scores = scaler.fit_transform(seq_mse.reshape(-1, 1)).flatten()
    
    # Pad initial days that didn't have enough history
    padded_scores = np.concatenate([np.zeros(seq_len), norm_seq_scores])
    return padded_scores

# 3. Isolation Forest Boundary Anomaly Detector
def compute_isolation_forest_scores(X: np.ndarray, contamination: float = 0.08):
    """
    Fits an Isolation Forest and returns normalized anomaly severity scores (0 to 1)
    along with the fitted model (for SHAP explainability).
    
    Returns:
        tuple: (norm_scores: np.ndarray, iso_model: IsolationForest)
    """
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(X)
    # score_samples returns negative anomaly score (more negative = more anomalous)
    raw_scores = -iso.score_samples(X)
    scaler = MinMaxScaler()
    norm_scores = scaler.fit_transform(raw_scores.reshape(-1, 1)).flatten()
    return norm_scores, iso


# ── Phase 1: Explainability Functions ──────────────────────────────

def compute_shap_attributions(iso_model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    Computes per-feature SHAP attributions for the Isolation Forest model
    using TreeExplainer for exact, fast Shapley value computation.
    
    Each value represents how much that feature contributed to the
    anomaly score — positive values push toward anomaly, negative toward normal.
    
    Args:
        iso_model: Fitted IsolationForest instance
        X: Scaled feature matrix (n_samples, n_features)
    
    Returns:
        np.ndarray of shape (n_samples, n_features) — SHAP values per feature
    """
    import shap
    explainer = shap.TreeExplainer(iso_model)
    shap_values = explainer.shap_values(X)
    return shap_values


def compute_autoencoder_feature_deltas(model: BehaviorAutoencoder, X: np.ndarray) -> np.ndarray:
    """
    Computes per-feature reconstruction error as feature-level attributions
    for the Deep Autoencoder. Each value represents how much that specific
    behavioral dimension deviated from the learned normal profile.
    
    This provides interpretability analogous to SHAP values — showing which
    features the model struggled to reconstruct (i.e., which behavioral
    dimensions shifted from baseline norms).
    
    Args:
        model: Trained BehaviorAutoencoder instance
        X: Scaled feature matrix (n_samples, n_features)
    
    Returns:
        np.ndarray of shape (n_samples, n_features) — normalized [0, 1] per-feature deltas
    """
    model.eval()
    tensor_data = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        recon = model(tensor_data)
        # Per-feature squared reconstruction error
        deltas = (tensor_data - recon).pow(2).numpy()

    # Normalize each feature column independently to [0, 1]
    if deltas.max() > 0:
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(deltas)
    else:
        normalized = deltas
    return normalized