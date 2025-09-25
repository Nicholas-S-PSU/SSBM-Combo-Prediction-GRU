# train_combo_gru.py
import os
import glob
import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# -------------------------
# Utility: reproducibility
# -------------------------
def set_seed(seed: int = 1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# -------------------------
# Dataset
# -------------------------
class SlippiDataset(Dataset):
    def __init__(self, dir_path):
        self.shard_paths = sorted(glob.glob(os.path.join(dir_path, "*.npz")))
        self.shard_offsets = list()
        self.shard_sizes = list()
        self.total_size = 0
        self._build_index()
    
    def _build_index(self):
        offset = 0
        for path in self.shard_paths:
            with np.load(path) as data:
                n = data['X'].shape[0]
                self.shard_sizes.append(n)
                self.shard_offsets.append(offset)
                offset += n
        self.total_size = offset
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, index):
        for shard_index, offset in enumerate(self.shard_offsets):
            if index < offset + self.shard_sizes[shard_index]:
                local_index = index - offset
                with np.load(self.shard_paths[shard_index]) as data:
                    X = data['X'][local_index]
                    stage = data["stage"][local_index]
                    comboer_state = data["comboer_state"][local_index]
                    comboer_char = data["comboer_char"][local_index]
                    comboee_state = data["comboee_state"][local_index]
                    comboee_char = data["comboee_char"][local_index]
                    hits = data["hits"][local_index]
                    y = data['y'][local_index]
                X = torch.from_numpy(X).float()
                stage = torch.from_numpy(stage).int()
                comboer_state = torch.from_numpy(comboer_state).int()
                comboer_char = torch.from_numpy(comboer_char).int()
                comboee_state = torch.from_numpy(comboee_state).int()
                comboee_char = torch.from_numpy(comboee_char).int()
                hits = torch.from_numpy(hits).int()
                y = torch.tensor(y, dtype=torch.float32)
                return {"X":X, "stage":stage, "comboer_state":comboer_state, "comboer_char":comboer_char, 
                        "comboee_state":comboee_state, "comboee_char":comboee_char, "hits":hits, "y":y}
        raise IndexError



# -------------------------
# Model
# -------------------------
class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.2,
        use_input_projection: bool = True
    ):
        """
        input_dim: per-timestep continuous features (not including any embeddings)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        proj_in_dim = input_dim
        if use_input_projection:
            # small input projection often helps
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, max(64, input_dim)),
                nn.ReLU(),
                nn.LayerNorm(max(64, input_dim))
            )
            proj_in_dim = max(64, input_dim)
        else:
            self.input_proj = None
        #set up embeddings
        self.action_embedding = nn.Embedding(387, 6) #action state embedding
        self.char_embedding = nn.Embedding(33, 4) #character embedding
        self.stage_embedding = nn.Embedding(6, 4) #stage embedding (could switch to one hot)
        self.hit_embedding = nn.Embedding(93, 6) #previous hit embedding

        self.gru = nn.GRU(
            input_size=proj_in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0)
        )

        # classification head
        final_dim = hidden_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(final_dim),
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, 1)
        )

    def forward(self, X, stage, comboer_state, comboer_char, comboee_state, comboee_char, hits):
        """
        X (B, time, features) of standard features not to be embedded
        stage (B, time) of stages
        comboer_state (B, time) of comboer states
        comboer_char (B, time) of comboer char
        comboee_state (B, time) of comboee states
        comboee_char (B, time) of comboee char
        hits (B, time) of previous hit id's
        returns: logits (B,) (not passed through sigmoid)
        """
        x = X
        comboer_state_emb = self.action_embedding(comboer_state)
        comboer_char_emb = self.char_embedding(comboer_char)
        comboee_state_emb = self.action_embedding(comboee_state)
        comboee_char_emb = self.char_embedding(comboee_char)
        stage_emb = self.stage_embedding(stage) #yields (B, time, 4)
        hit_emb = self.hit_embedding(hits)

        packed = torch.cat((X, stage_emb, comboer_state_emb, comboer_char_emb, comboee_state_emb, comboee_char_emb, hit_emb), dim = -1) #(B, T, D+30)

        if self.input_proj is not None:
            # flatten then project per timestep
            packed = self.input_proj(packed)  # (B, T, Dproj)

        packed_out, h_n = self.gru(packed)  # h_n: (num_layers, B, H)

        final_h = h_n[self.num_layers - 1, :, :]  # (B, H)

        logits = self.classifier(final_h).squeeze(-1)  # (B,)
        return logits

# -------------------------
# Training / evaluation
# -------------------------
def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch, cfg):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=cfg.get('pos_weight', None))
    pbar = tqdm(dataloader, desc=f"Train epoch {epoch}", leave=False)
    running_loss = 0.0
    all_preds = []
    all_targets = []
    for batch in pbar:
        X = batch['X'].to(device)         # (B, Tmax, D)
        stage = batch["stage"].to(device)
        cr_state = batch["comboer_state"].to(device)
        cr_char = batch["comboer_char"].to(device)
        ce_state = batch["comboee_state"].to(device)
        ce_char = batch["comboee_char"].to(device)
        hits = batch["hits"].to(device)
        y = batch['y'].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=cfg.get('use_amp', True)):
            logits = model(X, stage, cr_state, cr_char, ce_state, ce_char, hits)
            loss = loss_fn(logits, y)

        scaler.scale(loss).backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.get('grad_clip', 1.0))
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * X.shape[0]
        preds = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.append(preds)
        all_targets.append(y.detach().cpu().numpy())
        pbar.set_postfix({'loss': loss.item()})

    avg_loss = running_loss / len(dataloader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    auc = None
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except Exception:
        auc = None
    return avg_loss, auc

def eval_model(model, dataloader, device, cfg):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=cfg.get('pos_weight', None))
    running_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Valid", leave=False):
            X = batch["X"].to(device)
            stage = batch["stage"].to(device)
            cr_state = batch["comboer_state"].to(device)
            cr_char = batch["comboer_char"].to(device)
            ce_state = batch["comboee_state"].to(device)
            ce_char = batch["comboee_char"].to(device)
            hits = batch["hits"].to(device)
            y = batch["y"].to(device)

            logits = model(X, stage, cr_state, cr_char, ce_state, ce_char, hits)
            loss = loss_fn(logits, y)

            running_loss += loss.item() * X.shape[0]
            preds = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y.detach().cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    auc = None
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except Exception:
        auc = None

    # Also compute accuracy at threshold 0.5
    acc = ((all_preds >= 0.5) == (all_targets >= 0.5)).mean()
    return avg_loss, auc, acc

# -------------------------
# Main training routine
# -------------------------
def main_training(
    train_path,
    val_path,
    model_save_path = "best_model.pt",
    batch_size = 64,
    num_epochs = 30,
    lr = 1e-3,
    hidden_dim = 256,
    input_dim = None,        # must set or inferred from dataset
    use_amp = True,
    num_workers = 4,
    seed = 1
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = SlippiDataset(train_path)
    val_ds = SlippiDataset(val_path)

    # infer input dim from first sample if not provided
    if input_dim is None:
        s0 = train_ds[0]
        input_dim = s0["X"].shape[1] + 30 #30 embedding dimensions + the normal X dims

    print(f"Assumed input_dim={input_dim}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=max(0, num_workers//2), pin_memory=True)

    # Initialize model
    model = GRUClassifier(
        input_dim=input_dim,  #normal feature dims plus embeddings
        hidden_dim=hidden_dim,
        num_layers=1,
        dropout=0.2,
        use_input_projection=True
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_auc = -math.inf
    best_val_loss = math.inf
    for epoch in range(1, num_epochs+1):
        train_cfg = {
            'use_amp': use_amp,
            'grad_clip': 1.0
        }
        train_loss, train_auc = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, train_cfg)
        val_loss, val_auc, val_acc = eval_model(model, val_loader, device, train_cfg)
        print(f"Epoch {epoch} | train_loss {train_loss:.4f} | train_auc {train_auc} | val_loss {val_loss:.4f} | val_auc {val_auc} | val_acc {val_acc:.4f}")

        # scheduler step: prefer val_auc if available
        if (val_auc is not None):
            scheduler.step(val_auc)
            is_best = (val_auc > best_val_auc)
            if is_best:
                best_val_auc = val_auc
        else:
            scheduler.step(val_loss)
            is_best = (val_loss < best_val_loss)
            if is_best:
                best_val_loss = val_loss

        if is_best:
            print("Saving new best model to", model_save_path)
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_auc': val_auc
            }, model_save_path)

    print("Training complete. Best val_auc:", best_val_auc, "best val_loss:", best_val_loss)

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    TRAIN_PATH = "train"     #looks in directory for .npz
    VAL_PATH = "test"
    MODEL_SAVE_PATH = "model"

    main_training(
        train_path=TRAIN_PATH,
        val_path=VAL_PATH,
        model_save_path=MODEL_SAVE_PATH,
        batch_size=64,
        num_epochs=30,
        lr=1e-3,
        hidden_dim=256,
        input_dim=None,          # inferred from dataset
        use_amp=True,
        num_workers=4,
        seed=42
    )
