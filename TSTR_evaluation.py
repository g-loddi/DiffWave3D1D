import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import os
from torchsummary import summary

folder_path = '3_services_data'

seed_list = [42, 21, 7]
model_name_list = ['diffwave1d', 'diffwave3d', 'diffwave3d1d_64', 'csdi']



class ServiceDataset(Dataset):

    def __init__(self, df: pd.DataFrame):
        labels = df["service"]
        self.y = torch.tensor(labels.values, dtype=torch.long)  

        # Time series columns as features
        ts_cols = [str(i) for i in range(96)]
        self.X = torch.tensor(df[ts_cols].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLPClassifier(nn.Module):
    """
    A simple feed-forward network for 2-class classification.
    Input: 96-dim
    Output: 3-dim (logits for 3 services)
    """
    def __init__(self, hidden_dim=128, num_classes=3): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(96, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)  
        )
        
    def forward(self, x):
        # Returns logits of shape (batch_size, 2)
        return self.net(x)
    
class CNNClassifier(nn.Module):
    """
    A simple 1D CNN classifier for 96-step time series.
    Output: 3-dim (logits for 3 services)
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Output shape: (batch_size, 64, 1)
            nn.Flatten(),             # Shape: (batch_size, 64)
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Input shape: (batch_size, 96)
        x = x.unsqueeze(1)  # Convert to (batch_size, 1, 96) for Conv1d
        return self.net(x)


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=False):
        self.patience  = patience
        self.min_delta = min_delta
        self.verbose   = verbose
        self.best_loss = float('inf')
        self.bad_epochs = 0
        self.should_stop = False

    def __call__(self, val_loss):
        if val_loss + self.min_delta < self.best_loss:
            self.best_loss  = val_loss
            self.bad_epochs = 0
            if self.verbose:
                print(f"  ↳ New best val loss: {val_loss:.4f}")
        else:
            self.bad_epochs += 1
            if self.verbose:
                print(f"  ↳ No improvement ({self.bad_epochs}/{self.patience})")
            if self.bad_epochs >= self.patience:
                if self.verbose:
                    print("  ↳ Early stopping triggered.")
                self.should_stop = True

def train_and_evaluate(df_train, df_test, 
                       model_type="mlp",
                       epochs=100, 
                       batch_size=64, 
                       lr=1e-4,
                       device='cuda',
                       patience=5,
                       min_delta=0.0,
                       val_fraction=0.2,
                       seed=42):
    # 1) Create full train Dataset, then split into train/val
    full_train_ds = ServiceDataset(df_train)
    n_total = len(full_train_ds)
    n_val   = int(n_total * val_fraction)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        full_train_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    # 2) Instantiate model, optimizer, criterion
    if model_type == "mlp":
        model = MLPClassifier(hidden_dim=128, num_classes=3).to(device)
    elif model_type == "cnn":
        model = CNNClassifier(num_classes=3).to(device)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta, verbose=True)
    best_wts = model.state_dict()

    # 3) Training + Validation loop
    for epoch in range(1, epochs+1):
        # — train —
        model.train()
        running_loss = 0.0
        for Xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * Xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch} — train loss: {train_loss:.4f}")

        # — validate —
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                val_running += criterion(model(Xb), yb).item() * Xb.size(0)
        val_loss = val_running / len(val_loader.dataset)
        print(f"Epoch {epoch} — val   loss: {val_loss:.4f}")

        # — early stopping check —
        early_stopper(val_loss)
        if val_loss == early_stopper.best_loss:
            best_wts = model.state_dict()
        if early_stopper.should_stop:
            print(f"Stopping early at epoch {epoch}")
            break

    # restore best weights
    model.load_state_dict(best_wts)

    # 4) Final test on df_test
    test_ds   = ServiceDataset(df_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds = torch.argmax(model(Xb), dim=1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return model, accuracy

if __name__ == "__main__":

    seed = 1
    random.seed(seed)           # Python's random
    np.random.seed(seed)        # NumPy
    torch.manual_seed(seed)     # PyTorch CPU
    torch.cuda.manual_seed(seed)        # PyTorch GPU (single)
    torch.cuda.manual_seed_all(seed)    # PyTorch GPU (all)
    torch.backends.cudnn.deterministic = True   # For reproducibility
    
    log_accuracy_list = []

    for seed in seed_list:
        real_path = os.path.join(folder_path, f"real_data_seed{seed}.parquet")
        df_real = pd.read_parquet(real_path)

        for model_name in model_name_list:
            gen_file = f"generated_data_{model_name}_seed{seed}.parquet"
            gen_path = os.path.join(folder_path, gen_file)

            if not os.path.exists(gen_path):
                print(f"Missing generated file: {gen_file}, skipping...")
                continue

            df_generated = pd.read_parquet(gen_path)

            model_type = 'mlp'

            print(f"Training on {gen_path} and evaluating on real data...")

            model, accuracy = train_and_evaluate(
                df_train=df_generated,
                df_test=df_real,
                model_type=model_type,
                epochs=100,
                batch_size=8192,
                lr=1e-4,
                device='cuda'  # or 'cuda'
                )
            log_accuracy_list.append(('mlp',gen_path,accuracy))
            print(f"Accuracy on real data: {accuracy:.4f}")
            
 

            model_type = 'cnn'
       

            print(f"Training on {gen_path} and evaluating on real data...")

            model, accuracy = train_and_evaluate(
                df_train=df_generated,
                df_test=df_real,
                model_type=model_type,
                epochs=100,
                batch_size=8192,
                lr=1e-4,
                device='cuda'  # or 'cuda'
            )
            log_accuracy_list.append(('cnn',gen_path, accuracy))
            print(f"Accuracy on real data: {accuracy:.4f}")
            
    # Save the accuracies to a CSV file
    log_accuracy_df = pd.DataFrame(log_accuracy_list, columns=["Method","Path", "Accuracy"])
    log_accuracy_df.to_csv(folder_path+f"/accuracy_results.csv", index=False)
    print(f"Accuracy results saved to accuracy_results.csv")
    
    
    # device = 'cpu'
    # model = MLPClassifier(hidden_dim=128, num_classes=3).to(device)
    # summary(model=model, device="cpu")

    # model = CNNClassifier(num_classes=3).to(device)   
    # summary(model=model, device="cpu")
