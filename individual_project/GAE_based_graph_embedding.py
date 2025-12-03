import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import math
import os


class GCNEncoder(nn.Module):
    """
    Two-layer GCN encoder implementing:
    Z = Ahat * ReLU( Ahat * X * W0 ) * W1
    where * denotes matrix multiply and X is (N x F).
    """
    def __init__(self, in_feats: int, hidden_feats: int, out_feats: int):
        super().__init__()
        # We'll use nn.Linear for W0 and W1; they operate on node features
        self.lin0 = nn.Linear(in_feats, hidden_feats, bias=True)
        self.lin1 = nn.Linear(hidden_feats, out_feats, bias=True)

    def forward(self, X: torch.Tensor, Ahat: torch.Tensor) -> torch.Tensor:
        # X: (N, F), Ahat: (N, N)
        # hidden = ReLU( Ahat @ (X @ W0) )
        h = self.lin0(X)         # (N, hidden)
        h = torch.matmul(Ahat, h)  # (N, hidden)
        h = F.relu(h)
        z = self.lin1(h)         # (N, out)
        z = torch.matmul(Ahat, z)   # (N, out)
        return z



def read_edge_list(path: str) -> Tuple[List[Tuple[str,str]], List[str]]:
    """
    Reads whitespace-separated edge list file lines 'u v'.
    Returns list of edges (u,v) and sorted list of unique nodes.
    """
    edges = []
    nodes = set()
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            edges.append((u, v))
            nodes.add(u); nodes.add(v)
    nodes = sorted(nodes)
    return edges, nodes

def build_adj_matrix(edges: List[Tuple[str,str]], nodes: List[str]) -> np.ndarray:
    """
    Builds undirected adjacency matrix A with self-loops (Aii = 1).
    Returns numpy array A of shape (N, N), dtype float32.
    """
    idx = {n:i for i,n in enumerate(nodes)}
    N = len(nodes)
    A = np.zeros((N, N), dtype=np.float32)
    for u,v in edges:
        i, j = idx[u], idx[v]
        A[i,j] = 1.0
        A[j,i] = 1.0  # undirected
    # self-loops
    for i in range(N):
        A[i,i] = 1.0
    return A

def symmetric_normalize(A: np.ndarray) -> np.ndarray:
    
    """
    Returns symmetric normalized adjacency: D^{-1/2} A D
    Handles isolated nodes (deg=0) by leaving their inv sqrt as 0.
    """
    deg = A.sum(axis=1)
    print(f'len D: {len(deg)}')
    print(f'D: {deg}')
    N = A.shape[0]
    print(f'N: {N}')
    D = np.zeros((N, N), dtype=np.float32)
    for i in range(0, N):
        for j in range(0, N):
            if i==j:
                D[i][j]=deg[i]

    print(f'D: {D}')

    # avoid division by zero
    deg_inv_sqrt = np.zeros_like(deg, dtype=np.float32)
    print(f'deg_inv_sqrt:\n{deg_inv_sqrt}')

    nonzero = deg > 0
    print(f'nonzero:\n{nonzero}')

    deg_inv_sqrt[nonzero] = 1.0 / np.sqrt(deg[nonzero])
    print(f'deg_inv_sqrt[nonzero]:\n{deg_inv_sqrt[nonzero]}')

    D_inv_sqrt = np.diag(deg_inv_sqrt)
    print(f'D_inv_sqrt:\n{D_inv_sqrt}')

    # A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    # print(f'\nPrevious Ahat:\n{A_hat}')

    A_hat = D_inv_sqrt @ A @ D
    print(f'Ahat:\n{A_hat}')
    return A_hat

def adjacency_reconstruction(Z: torch.Tensor) -> torch.Tensor:
    """
    Reconstruction via inner product and sigmoid: sigma( Z Z^T )
    Returns (N, N) matrix of probabilities.
    """
    logits = torch.matmul(Z, Z.t())   # shape (N, N)
    return torch.sigmoid(logits)

def train_gae(A: np.ndarray,
              hidden_dim: int = 32,
              embed_dim: int = 16,
              lr: float = 0.01,
              weight_decay: float = 0.0,
              epochs: int = 500,
              device: str = None,
              verbose: bool = True) -> Tuple[np.ndarray, List[float]]:
    """
    Train GAE on adjacency A (numpy float32).
    Returns embeddings Z (N, embed_dim) numpy array and list of losses.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    N = A.shape[0]
    # X = identity (N x N)
    X = np.eye(N, dtype=np.float32)

    A_hat = symmetric_normalize(A)  # numpy
    # Convert to torch
    X_t = torch.from_numpy(X).to(device)          # (N,N)
    Ahat_t = torch.from_numpy(A_hat).to(device)   # (N,N)
    A_t = torch.from_numpy(A).to(device)          # (N,N)

    model = GCNEncoder(in_feats=N, hidden_feats=hidden_dim, out_feats=embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []
    for ep in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        Z = model(X_t, Ahat_t)  # (N, embed)
        A_pred = adjacency_reconstruction(Z)  # (N,N) probabilities between 0..1

        # Binary cross-entropy per matrix entry, averaged over N^2 as specified
        # Flatten and compute BCE
        loss = F.binary_cross_entropy(A_pred.view(-1), A_t.view(-1), reduction='mean')
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if verbose and (ep % max(1, epochs//10) == 0 or ep <= 10):
            print(f"Epoch {ep:4d}/{epochs}  loss={loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        Z_final = model(X_t, Ahat_t).cpu().numpy()  # (N, embed_dim)
    return Z_final, losses

def save_embeddings(nodes: List[str], Z: np.ndarray, out_path: str):
    """
    Save embeddings to a text file: node \t z0 z1 ... z_{d-1}
    """
    with open(out_path, 'w') as f:
        for i, node in enumerate(nodes):
            emb = " ".join(map(str, Z[i].tolist()))
            f.write(f"{node}\t{emb}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a Graph Auto-Encoder (GAE) on an edgelist")
    base_path = "/home/achakroborti1/PWAC_FALL25/individual_project"
    # common parameters
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--embed", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--no-cuda", action="store_true", help="disable cuda even if available")

    #for Ga
    parser.add_argument("--edgesa", default=f"{base_path}/data/validation_dataset/validation_G1.edgelist.txt", help="Path to edge list .txt file (u v per line)")
    # parser.add_argument("--edgesa", default=f"{base_path}/data/seed_free/unseed_G1.edgelist", help="Path to edge list .txt file (u v per line)")

    parser.add_argument("--out_dir", default="./GAE_based_outputs")

    # for Gu
    parser.add_argument("--edges_u", default=f"{base_path}/data/validation_dataset/validation_G2.edgelist.txt", help="Path to edge list .txt file (u v per line)")
    # parser.add_argument("--edges_u", default=f"{base_path}/data/seed_free/unseed_G2.edgelist", help="Path to edge list .txt file (u v per line)")

    args = parser.parse_args()

    # create the our directory
    os.makedirs(args.out_dir, exist_ok=True)

    edges, nodes = read_edge_list(args.edgesa)
    A = build_adj_matrix(edges, nodes)
    device = "cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Read {len(edges)} edges, {len(nodes)} nodes. Training on device={device}")

    Za, losses = train_gae(A,
                         hidden_dim=args.hidden,
                         embed_dim=args.embed,
                         lr=args.lr,
                         epochs=args.epochs,
                         device=device,
                         verbose=True)

    save_embeddings(nodes, Za, f"{args.out_dir}/Za_embeddings.txt")
    print(f"Za:: Saved embeddings to {args.out_dir}/Za_embeddings.txt")

 
    edges_u, nodes_u = read_edge_list(args.edges_u)
    Au = build_adj_matrix(edges_u, nodes_u)
    device = "cpu" if args.no_cuda else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Read {len(edges_u)} edges, {len(nodes_u)} nodes. Training on device={device}")

    Zu, losses = train_gae(Au,
                         hidden_dim=args.hidden,
                         embed_dim=args.embed,
                         lr=args.lr,
                         epochs=args.epochs,
                         device=device,
                         verbose=True)

    save_embeddings(nodes_u, Zu, f"{args.out_dir}/Zu_embeddings.txt")
    print(f"Zu:: Saved embeddings to {args.out_dir}/Zu_embeddings.txt")