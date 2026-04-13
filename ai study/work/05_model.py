import torch
import torch.nn as nn

class MultiTaskModel(nn.Module):
    def __init__(self, n_features=18, window_size=30):
        super(MultiTaskModel, self).__init__()
        
        # Option A: Bi-LSTM encoder
        # The architecture requests a Shared representation of 128-d.
        # Since it's a Bi-directional LSTM, the outputs are concatenated.
        # Setting hidden_size=64 makes the concatenated output 64 + 64 = 128.
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # RUL head: Linear(128->64) -> ReLU -> Linear(64->1)
        self.rul_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Alert head: Linear(128->64) -> ReLU -> Linear(64->1) -> Sigmoid
        self.alert_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x expected shape: (batch_size, window_size, n_features)
        
        # lstm_out: (batch, seq_len, num_directions * hidden_size)
        # hn: (num_layers * num_directions, batch, hidden_size)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # We need the 128-d shared representation for the entire sequence.
        # We can extract the final hidden state of the last layer for both forward and backward directions.
        # hn[-2,:,:] is the final forward state | hn[-1,:,:] is the final backward state
        shared_rep = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1) # shape: (batch_size, 128)
        
        # Pass the shared representation through the two heads
        rul_pred = self.rul_head(shared_rep)
        alert_pred = self.alert_head(shared_rep)
        
        # Squeeze down to (batch_size,) to match PyTorch target shapes
        return rul_pred.squeeze(-1), alert_pred.squeeze(-1)

def combined_loss(rul_pred, rul_true, alert_pred, alert_true, alpha=1.0, beta=1.0):
    """
    Combined loss: L = α * MSE(rul) + β * BCE(alert)
    """
    mse_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCELoss() # Sigmoid is built into the alert head
    
    mse = mse_loss_fn(rul_pred, rul_true)
    bce = bce_loss_fn(alert_pred, alert_true)
    
    total_loss = alpha * mse + beta * bce
    return total_loss, mse, bce

if __name__ == "__main__":
    print("Testing Step 5: Multi-Task Model Architecture...\n")
    
    # We use 18 features because Step 1-4 left us with 18 sensor + operational columns
    # instead of the 15 mentioned generically in the diagram.
    model = MultiTaskModel(n_features=18, window_size=30)
    
    print("Model Architecture Graph:")
    print(model)
    
    # Test with dummy batch: 64 samples, 30 cycles, 18 features
    dummy_input = torch.randn(64, 30, 18)
    dummy_rul_true = torch.empty(64).uniform_(0, 125) # random true RULs
    dummy_alert_true = torch.empty(64).random_(2)     # random binary alerts (0 or 1)
    
    # Pass through model
    rul_pred, alert_pred = model(dummy_input)
    
    print("\nDummy Forward Pass Check:")
    print(f"Input shape (batch, W, features): {dummy_input.shape}")
    print(f"RUL output shape:   {rul_pred.shape}")
    print(f"Alert output shape: {alert_pred.shape}")
    
    # Calculate loss
    total_loss, mse, bce = combined_loss(rul_pred, dummy_rul_true, alert_pred, dummy_alert_true, alpha=1.0, beta=1.0)
    print(f"\nLoss Check (α=1.0, β=1.0):")
    print(f"MSE (RUL):    {mse.item():.4f}")
    print(f"BCE (Alert):  {bce.item():.4f}")
    print(f"Total Loss:   {total_loss.item():.4f}")
    
    print("\nStep 5 Model implementation is complete and functioning!")
