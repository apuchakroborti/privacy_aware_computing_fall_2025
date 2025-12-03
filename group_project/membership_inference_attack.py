import numpy as np
import pandas as pd
from datasets import Dataset
# import matplotlib.pyplot as plt
# import seaborn as sns
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

device = "cuda" if torch.cuda.is_available() else "cpu"

import utils as UTIL
import torch.nn.functional as F


# Loading data and model
# BASE_DIR = '/content/drive/MyDrive/PrivacyAwareComputing/ProjectMIALM'
BASE_DIR = "./ProjectMIALM"
model_path = os.path.join(BASE_DIR, 'victim_model_distilbert_agnews')


model, tokenizer = UTIL.load_victim('victim_model_distilbert_agnews')
print(f"Loaded victim_model")
print(f"from victim_model, model:\n{model}")
print(f"from victim_model, tokenizer:\n{tokenizer}")

# 
df = pd.read_csv(os.path.join(BASE_DIR, 'validation_samples.csv'))  # your uploaded file
raw_ds = Dataset.from_pandas(df.rename(columns={"label": "labels"})[["text", "labels"]])
print(f"\nLoaded raw_ds:\n{raw_ds}")


def tokenize_batch(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

shadow_source_ds = raw_ds.map(tokenize_batch, batched=True)
print(f"\nshadow_source_ds:\n{shadow_source_ds}")

shadow_source_ds = shadow_source_ds.remove_columns(["text"])  # only model inputs
print(f"After removing text column, shadow_source_ds:\n{shadow_source_ds}")


membership_data = np.loadtxt(os.path.join(BASE_DIR, 'validation_results.txt'), dtype=int)
print(f"\nmembership_data:\n{membership_data}")

membership_data = membership_data[membership_data[:,0].argsort()]  # sort by index
print(f"After sort by index, membership_data:\n{membership_data}")

membership_labels = membership_data[:,1]  # array of 0/1
print("Loaded membership labels:", len(membership_labels))
print(f"membership_labels:\n{membership_labels}")

assert len(membership_labels) == len(raw_ds)

member_idx = np.where(membership_labels == 1)[0].tolist()
print(f"member_idx:\n{member_idx}")

nonmember_idx = np.where(membership_labels == 0)[0].tolist()
print(f"nonmember_idx:\n{nonmember_idx}")

target_member_ds = shadow_source_ds.select(member_idx)
print(f"target_member_ds:\n{target_member_ds}")

target_nonmember_ds = shadow_source_ds.select(nonmember_idx)
print(f"target_nonmember_ds:\n{target_nonmember_ds}")

def make_shadow_splits(dataset, num_shadows=5, shadow_train_frac=0.5, seed=0):
    """
    dataset: Tokenized Dataset
    Returns: list of (train_ds, out_ds) for each shadow
    """
    rng = np.random.default_rng(seed)
    n = len(dataset)
    indices = np.arange(n)

    shadow_pairs = []
    for s in range(num_shadows):
        rng.shuffle(indices)
        cut = int(shadow_train_frac * n)
        train_idx = indices[:cut]
        out_idx   = indices[cut:]

        train_ds = dataset.select(train_idx.tolist())
        out_ds   = dataset.select(out_idx.tolist())
        shadow_pairs.append((train_ds, out_ds))

    return shadow_pairs


def make_loader(ds, batch_size=16, shuffle=True):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=UTIL.collate_fn)


def train_shadow_model(victim_dir, train_ds, val_ds=None,
                       epochs=2, lr=2e-5, batch_size=16, noise_rate=0.2):
    """
    Each shadow shares same architecture & init as victim.
    """
    model = DistilBertForSequenceClassification.from_pretrained(os.path.join(BASE_DIR, victim_dir)).to(device)
    model.train()

    train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=batch_size, shuffle=False) if val_ds else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # loss_fn = nn.CrossEntropyLoss(reduction="mean")
    loss_fn = UTIL.NoisyLabelCrossEntropy(noise_rate=noise_rate, num_classes=4)

    for ep in range(epochs):
        running = 0.0
        for batch in train_loader:
            batch = {k:v.to(device) for k,v in batch.items()}
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"])
            loss = loss_fn(out.logits, batch["labels"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running += loss.item()

        print(f"[shadow] epoch {ep+1}/{epochs} loss={running/len(train_loader):.4f}")

    model.eval()
    return model



def build_attack_dataset(victim_dir, shadow_pairs,
                         shadow_epochs=2, batch_size=16, shadow_noise_rate=0.2, shadow_lr=2e-5):
    """
    For each shadow:
      - train on shadow train split (members)
      - compute losses on members and non-members
    Returns: X (losses), y (membership labels)
    """
    X_list, y_list = [], []

    for i, (in_ds, out_ds) in enumerate(shadow_pairs):
        print(f"\n=== Training shadow {i+1}/{len(shadow_pairs)} ===")
        shadow_model = train_shadow_model(
            victim_dir, in_ds, epochs=shadow_epochs, batch_size=batch_size, noise_rate=shadow_noise_rate, lr=shadow_lr
        )

        in_features  = UTIL.extract_features(shadow_model, in_ds, batch_size=batch_size)
        out_features = UTIL.extract_features(shadow_model, out_ds, batch_size=batch_size)

        X_list.append(in_features)
        X_list.append(out_features)

        y_list.append(np.ones(len(in_features)))      # members = 1
        y_list.append(np.zeros(len(out_features)))      # members = 1

    X = np.concatenate(X_list, axis=0).reshape(-1, in_features.shape[1])  # (M,1)
    y = np.concatenate(y_list, axis=0).astype(np.int64)

    return X, y



def train_attack_model(
        X, y,
        epochs=20,
        lr=1e-4,
        batch_size=128,
        lbfgs_steps=10
    ):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    ds = torch.utils.data.TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    attack = UTIL.AttackMLP(in_dim=X.shape[1]).to(device)

    # ---- Adam optimizer ----
    opt = torch.optim.Adam(attack.parameters(), lr=lr)

    # ---- LR Scheduler ----
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=2,
        threshold=1e-3
    )

    bce = nn.BCEWithLogitsLoss()

    print(f"\n=== Training attack model (Adam + LBFGS) ===")

    for ep in range(epochs):

        attack.train()
        running = 0.0

        # ----------------------------
        #   PHASE 1 — Adam optimizer
        # ----------------------------
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)

            logits = attack(xb)
            loss = bce(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item()

        # LR scheduler step
        scheduler.step(running / len(dl))

        # Print every few epochs
        if ep % max(1, epochs // 10) == 0:
            print(f"[Adam {ep+1}/{epochs}] loss = {running/len(dl):.4f}")

    # =====================================
    #   PHASE 2 — LBFGS refinement
    # =====================================

    print("\n=== Refining with LBFGS ===")

    # Define LBFGS optimizer (must use FULL batch!)
    lbfgs = torch.optim.LBFGS(
        attack.parameters(),
        lr=0.1,               # LBFGS uses a different internal rule
        max_iter=20,
        history_size=50,
        line_search_fn="strong_wolfe"
    )

    full_x = X_t.to(device)
    full_y = y_t.to(device)

    def closure():
        lbfgs.zero_grad()
        logits = attack(full_x)
        loss = bce(logits, full_y)
        loss.backward()
        return loss

    # Perform LBFGS refinement steps
    for i in range(lbfgs_steps):
        loss_val = lbfgs.step(closure)
        print(f"  LBFGS step {i+1}/{lbfgs_steps}: loss={loss_val.item():.6f}")

    attack.eval()
    return attack

@torch.no_grad()
def attack_victim(attack_model, victim_model, member_ds, nonmember_ds, batch_size=32):
    m_features = UTIL.extract_features(victim_model, member_ds, batch_size=batch_size)
    nm_features = UTIL.extract_features(victim_model, nonmember_ds, batch_size=batch_size)

    X_test = np.concatenate([m_features, nm_features], axis=0)
    y_true = np.concatenate([
        np.ones(len(m_features)),
        np.zeros(len(nm_features))
    ])

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    probs = torch.sigmoid(attack_model(X_test_t)).cpu().numpy().squeeze()

    y_pred = (probs > 0.5).astype(np.int64)
    results = {}
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['roc_auc'] = float(roc_auc_score(y_true, probs))
    results['precision'] = precision_score(y_true, y_pred)
    results['recall'] = recall_score(y_true, y_pred)
    results['f1'] = f1_score(y_true, y_pred)
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    # results['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

    print(f"\nVictim attack accuracy: {results['accuracy']*100:.2f}% ROC AUC: {results['roc_auc']*100:.2f}%")
    return results, probs, y_true



def run_loss_based_mia(
    victim_dir,
    shadow_source_ds,      # fake data pool for shadows
    target_member_ds,      # fake members for evaluation
    target_nonmember_ds,   # fake non-members for evaluation
    num_shadows=5,
    shadow_train_frac=0.5,
    shadow_noise_rate=0.2,
    shadow_epochs=10,
    batch_size=16,
    attack_model_epochs=20,
    attack_lr=1e-4,
    mode='validation'
                    ):
    print(f'Inside run_loss_based_mia, victim_dir: {victim_dir}, shadow_source_ds: {shadow_source_ds}, target_member_ds: {target_member_ds}, target_nonmember_ds: {target_nonmember_ds}')
    print(f'num_shadows: {num_shadows}, shadow_train_frac: {shadow_train_frac}, shadow_noise_rate: {shadow_noise_rate}, shadow_epochs: {shadow_epochs}')
    print(f'shadow_epochs: {shadow_epochs}, batch_size: {batch_size}, attack_model_epochs: {attack_model_epochs}, attack_lr: {attack_lr}, mode: {mode}')
   
    victim_model, tokenizer = UTIL.load_victim(victim_dir)    

    # print(f'victim_model: {victim_model}')
    # print(f'tokenizer: {tokenizer}')

    shadow_pairs = make_shadow_splits(
        shadow_source_ds,
        num_shadows=num_shadows,
        shadow_train_frac=shadow_train_frac
    )
    print(f'shadow_pairs: {shadow_pairs}')

    X_attack, y_attack = build_attack_dataset(
        victim_dir, shadow_pairs,
        shadow_epochs=shadow_epochs,
        batch_size=batch_size,
        shadow_noise_rate=shadow_noise_rate,
        shadow_lr=1e-4
    )
    print(f'X_attack: {X_attack}')
    print(f'y_attack: {y_attack}')
    
    data = {'X_attack_shape_1': X_attack.shape[1]}
    UTIL.save_key_value_pair('./saved_data/x_attack_shape_1.txt', data)

    attack_model = train_attack_model(X_attack, y_attack, epochs=attack_model_epochs, lr=attack_lr)
    
    if mode == 'validation':
      results, probs, y_true = attack_victim(
          attack_model, victim_model,
          target_member_ds, target_nonmember_ds,
          batch_size=batch_size
        )
      return attack_model, results
    else:
      return attack_model, None
    

# mode = 'inference'
mode = 'validation'
attack_model, results = run_loss_based_mia(
    victim_dir="victim_model_distilbert_agnews",
    shadow_source_ds=shadow_source_ds,
    target_member_ds=target_member_ds,
    target_nonmember_ds=target_nonmember_ds,
    # num_shadows=5,
    num_shadows=3,
    shadow_train_frac=0.5,
    shadow_noise_rate=0.02,
    # shadow_epochs=5,
    shadow_epochs=10,
    batch_size=32,
    attack_model_epochs=500,
    attack_lr=1e-1,
    mode=mode
)

# print(f'attack_model: {attack_model}')
print(f'results: {results}')


# Define the path to save the model
SAVE_PATH = f'./saved_data/attack_model_state_dict.pth'

# Save only the model's state dictionary
torch.save(attack_model.state_dict(), f'{SAVE_PATH}')

print(f"Model saved to {SAVE_PATH}")
