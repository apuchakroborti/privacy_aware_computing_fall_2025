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
from torch.nn.utils.rnn import pad_sequence

device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = "./ProjectMIALM"
model_path = os.path.join(BASE_DIR, 'victim_model_distilbert_agnews')



class AttackMLP(nn.Module):
    def __init__(self, in_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),

            nn.Linear(16, 1)
        )


    def forward(self, x):
        return self.net(x)

class NoisyLabelCrossEntropy(nn.Module):
    def __init__(self, noise_rate, num_classes):
        super().__init__()
        self.η = noise_rate
        self.C = num_classes

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            # construct noise-aware target distribution
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.η / (self.C - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1 - self.η)

        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))
    
def collate_fn(batch):
    # print(f'inside collate_fn, batch: {batch}')
    input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    # print(f'input_ids: {input_ids}')

    attention_mask = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch]
    # print(f'attention_mask: {attention_mask}')

    labels = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
    # print(f'labels: {labels}')

    # Pad sequences
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

@torch.no_grad()
def extract_features(model, ds, batch_size=32):
    # print(f'Inside extract_features, model: {model}, ds: {ds}, batch_size: {batch_size}')
    ds = ds.with_format("python")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # print(f'loader: {loader}')

    all_features = []

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    # print(f'loss_fn: {loss_fn}')

    # for batch in loader:
    #     print(f'batch in loader: {batch}')
    for batch in loader:
        # print(f'\n==========={batch}================')
        input_ids = batch["input_ids"].to(device)
        # print(f'input_ids: {input_ids}')
        
        attn = batch["attention_mask"].to(device)
        # print(f'attn: {attn}')
        
        labels = batch["labels"].to(device)
        # print(f'labels: {labels}')
        
        outputs = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)

        labels = batch["labels"].to(device)
        # print(f'labels: {labels}')
        
        logits = outputs.logits
        # print(f'logits: {logits}')

        probs = F.softmax(logits, dim=-1)
        # print(f'probs: {probs}')

        hidden = outputs.hidden_states[-1]  # last layer hidden state
        # print(f'hidden: {hidden}')

        cls_embed = hidden[:,0,:]
        # print(f'cls_embed: {cls_embed}')

        # 1. Loss
        losses = loss_fn(logits, labels)
        # print(f'losses: {losses}')

        # 2. Confidence
        conf = probs.max(dim=-1)[0]
        # print(f'conf: {conf}')

        # 3. Margin
        top2 = torch.topk(probs, k=2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]

        # 4. Entropy
        entr = -(probs * probs.log()).sum(dim=-1)

        # 5. Correctness
        preds = probs.argmax(dim=-1)
        correct = (preds == labels).float()

        # 6. Logit margin
        top2 = torch.topk(probs, k=2, dim=-1).values
        logit_margin = top2[:, 0] - top2[:, 1]

        # 7. Logit norm
        logit_norm = logits.norm(dim=-1)

        # 8. Class norm

        cls_norm = cls_embed.norm(dim=1)

        feats = torch.stack([
            losses,
            conf,
            # margin,
            entr,
            correct,
            logit_margin,
            # logit_norm,
            # p_true,
            # wrong_conf,
            # cls_norm
        ], dim=1)


        # collect
        all_features.append(feats.cpu().numpy())
        # print(f'all_features: {all_features}')

        # print("losses shape:", losses.shape)
        # print("conf shape:", conf.shape)
        # print("entr shape:", entr.shape)
        # print("correct shape:", correct.shape)
        # print("logit_margin shape:", logit_margin.shape)


    return np.concatenate(all_features, axis=0)
    


def load_victim(model_dir="victim_model"):
    model_path = os.path.join(BASE_DIR, model_dir)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path, local_files_only=True)
    model = DistilBertForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(device)
    
    # tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    # model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
    
    model.eval()
    return model, tokenizer



# file handling
def save_key_value_pair(filename, data):
    with open(filename, 'w') as f:
        for key, value in data.items():
            # Format: key:value
            f.write(f"{key}:{value}\n")

    print(f"Key-value pairs saved to {filename}")

def load_key_value_pair(filename):
    loaded_data = {}
    with open(filename, 'r') as f:
        for line in f:
            # Remove whitespace and the newline character
            line = line.strip()
            if line:
                # Split the line into key and value
                key, value = line.split(':', 1)
                loaded_data[key] = value

    print("Loaded Data:", loaded_data)
    # Output: {'user_id': '12345', 'setting': 'high', 'last_login': '2025-12-01'}
    return loaded_data