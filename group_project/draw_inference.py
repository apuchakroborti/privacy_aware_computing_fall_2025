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
# In your reloading script:
# from models import AttackMLP

def load_attack_model(path, in_dim):
    # 1. Instantiate the model with the correct input dimension
    # (Assuming AttackMLP is defined and in_dim is the feature count, X.shape[1])
    reloaded_model = UTIL.AttackMLP(in_dim=in_dim).to(device)

    # 2. Load the state dictionary
    state_dict = torch.load(path)

    # 3. Load the parameters into the instantiated model
    reloaded_model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    reloaded_model.eval() 
    
    return reloaded_model

# Example of calling the function:
# LOAD_PATH = 'attack_model_state_dict.pth'
# You must know the input dimension (in_dim) used during training
# Let's assume the training input dimension was 512
# input_dim_used_in_training = 512 

# reloaded_attack = load_attack_model(LOAD_PATH, input_dim_used_in_training)

# print(f"Model successfully reloaded from {LOAD_PATH}.")

# Draw Inference
def mia_predict_from_df(df, tokenizer, victim_model, attack_model, batch_size=32):
    print(f'Inside mia_predict_from_df')
    # Step 1 — Convert DataFrame to HuggingFace Dataset
    ds = Dataset.from_pandas(df[["text", "label"]].rename(columns={"label": "labels"}))
    print(f'from pandas ds: {ds}')

    # Step 2 — Tokenize
    ds = ds.map(lambda batch: tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    ), batched=True)
    print(f'from map ds: {ds}')

    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    # Step 3 — Extract multi-feature MIA vectors
    feats = UTIL.extract_features(victim_model, ds, batch_size=batch_size)
    print(f'feats: {feats}')
    # feats.shape = (N, feature_dim)

    # Step 4 — Convert features to tensor
    X = torch.tensor(feats, dtype=torch.float32).to(device)
    print(f'X: {X}')

    # Step 5 — Predict membership probability
    attack_model.eval()
    # probs = torch.sigmoid(attack_model(X)).cpu().numpy().squeeze()
    probs = torch.sigmoid(attack_model(X)).detach().cpu().numpy().squeeze()

    # Step 6 — Convert to binary membership prediction
    preds = (probs > 0.5).astype(int)

    return probs, preds

BASE_DIR = "./ProjectMIALM"
test_df = pd.read_csv(os.path.join(BASE_DIR, 'sampled.csv'))
print(f'test_df: {test_df}')

model, tokenizer = UTIL.load_victim('victim_model_distilbert_agnews')
print(f'model: {model}')
print(f'tokenizer: {tokenizer}')

in_dim = UTIL.load_key_value_pair('./saved_data/x_attack_shape_1.txt')['X_attack_shape_1']
in_dim = int(in_dim)
print(f'in_dim: {in_dim}')

attack_model = load_attack_model('./saved_data/attack_model_state_dict.pth', in_dim)
print(f'attack_model: {attack_model}')

print(f'Calling mia_predict_from_df ...')
_, preds = mia_predict_from_df(test_df, tokenizer, model, attack_model, batch_size=32)
print(f'preds: {preds}')