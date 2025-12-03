import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Tuple, List, Dict
import math
import os

import seed_based_propagation as SB
import copy
import utils as UTIL
# ---------- Model components ----------
class Discriminator(nn.Module):
    """
    Simple discriminator: input d -> hidden -> sigmoid output (prob vector from Za).
    We use a small MLP; you can change hidden dim.
    """
    def __init__(self, d: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2 if hidden//2>0 else 1),
            nn.ReLU(),
            nn.Linear(hidden//2 if hidden//2>0 else 1, 1)
        )

    def forward(self, x):
        logits = self.net(x).squeeze(-1)  # (batch,)
        probs = torch.sigmoid(logits)
        return probs, logits  # return both if you want logits


class LinearMapper(nn.Module):
    """
    Learnable linear mapping W (no bias) that maps source embedding dim d -> d.
    """
    def __init__(self, d: int):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.W(x)





# ---------- I/O helpers ----------
def load_embeddings(path: str) -> Tuple[List[str], np.ndarray]:
    print(f'load_embeddings:\n{path}')

    """
    Load embeddings from a text file with lines:
       node_id \t v0 v1 ... v{d-1}
    Returns (node_ids_list, numpy array shape (N, d)).
    """
    nodes = []
    vecs = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            node = parts[0]
            vals = list(map(float, parts[1:]))
            nodes.append(node)
            vecs.append(vals)
    
    print(f'load_embeddings, nodes:\n{nodes}')
    print(f'load_embeddings, vecs:\n{vecs}')
    return nodes, np.array(vecs, dtype=np.float32)


def read_edge_list(path: str) -> List[Tuple[str, str]]:
    edges = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            edges.append((parts[0], parts[1]))
    return edges


def compute_degrees_from_edges(edges: List[Tuple[str, str]]) -> Dict[str, int]:
    deg = {}
    for u, v in edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1  # undirected
    return deg


def degrees_for_nodes(node_list: List[str], deg_dict: Dict[str,int]) -> np.ndarray:
    """
    Return degree array aligned to node_list order (0 if missing).
    """
    return np.array([deg_dict.get(n, 0) for n in node_list], dtype=np.int32)


# ---------- CSLS similarity helpers ----------
# this is the cos(Wzia, zj) method
def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix between rows of A (nxd) and B (mxd).
    Returns shape (n, m).
    """
    # normalize
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A_norm @ B_norm.T  # (n, m)

# this method is for the equation 7
# 𝐶𝑆𝐿𝑆(Wzai ,zuj )= 2𝑐𝑜𝑠(Wza𝑖,zuj )−𝐷𝑢(Wzai )−𝐷𝑎(zuj )... (7)
def csls_scores(source_vecs: np.ndarray, target_vecs: np.ndarray, K: int = 10) -> np.ndarray:
    """
    Compute CSLS score matrix for all pairs: CSLS(s_i, t_j)
    Returns matrix (n_source, n_target).
    Implementation following:
       D_u(s_i) = avg_{k in KNN_u(s_i)} cos(s_i, t_k)
       D_a(t_j) = avg_{k in KNN_a(t_j)} cos(t_j, s_k)
       CSLS = 2 * cos(s_i, t_j) - D_u(s_i) - D_a(t_j)
    """
    # this is the cos(Wzia, zj) method, source_vecs=Wzia, target_vecs=zj
    sims = cosine_sim_matrix(source_vecs, target_vecs)  # (n, m)

    # this part is for the equation 6
    # 𝐷𝑢(Wzai )=(1/𝐾) sum of zj∈𝑆𝑢(Wzai ) 𝑐𝑜𝑠(Wzai ,zj), ...(6)
    # D_u: for each source row, average top K values in that row
    K_u = min(K, sims.shape[1])
    topk_u = np.partition(-sims, K_u-1, axis=1)[:, :K_u]  # negative partition trick
    topk_u = -topk_u
    D_u = topk_u.mean(axis=1)  # (n,)

    # this part is for the equation 6
    # need to change for Da_a(zju): # 𝐷𝑢(Wzai )=(1/𝐾) sum of zj∈𝑆𝑢(Wzai ) 𝑐𝑜𝑠(Wzai ,zj), ...(6)
    # For D_a compute cos(target, source) -> transpose of sims
    sims_t = sims.T  # (m, n)
    K_a = min(K, sims_t.shape[1])
    topk_a = np.partition(-sims_t, K_a-1, axis=1)[:, :K_a]
    topk_a = -topk_a
    D_a = topk_a.mean(axis=1)  # (m,)

    # CSLS = 2*cos - D_u[:,None] - D_a[None,:]
    cs = 2.0 * sims - D_u[:, None] - D_a[None, :]
    return cs


# ---------- Training routine ----------
def adversarial_alignment(
    za_nodes: List[str],
    Za: np.ndarray,         # (n, d)
    zu_nodes: List[str],
    Zu: np.ndarray,         # (m, d)
    edges_a_path: str = None,   # optional: to compute degrees for Za nodes
    edges_u_path: str = None,   # optional: to compute degrees for Zu nodes
    Nadv: int = 200,
    batch_size: int = 32,
    disc_hidden: int = 128,
    disc_lr: float = 0.1,
    W_lr: float = 0.01,
    W_epochs: int = 50,
    device: str = None,
    K_csls: int = 10,
    seed: int = 1234,
    verbose: bool = True
):
    """
    Perform adversarial alignment between Za and Zu following the paper description.
    Returns:
       best_W_state_dict, final_W_state_dict, mapping (dict: source_node -> matched target node),
       history dict with losses and metric
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    n, d = Za.shape
    m, d2 = Zu.shape
    assert d == d2, "Embedding dimensions must match"

    # degrees
    deg_a = np.zeros(n, dtype=np.int32)
    deg_u = np.zeros(m, dtype=np.int32)
    if edges_a_path is not None:
        edges_a = read_edge_list(edges_a_path)
        deg_dict_a = compute_degrees_from_edges(edges_a)
        deg_a = degrees_for_nodes(za_nodes, deg_dict_a)
        print(f'degree a: {deg_a}')

    if edges_u_path is not None:
        edges_u = read_edge_list(edges_u_path)
        deg_dict_u = compute_degrees_from_edges(edges_u)
        deg_u = degrees_for_nodes(zu_nodes, deg_dict_u)
        print(f'degree u: {deg_u}')

    # If degrees are all zero (no edges provided), fallback to using available nodes directly
    if deg_a.sum() == 0:
        deg_a = np.arange(1, n+1)  # fallback ranking
    if deg_u.sum() == 0:
        deg_u = np.arange(1, m+1)

    # select top Nadv by degree (cap by n,m)
    Nadv = min(Nadv, n, m)
    # this returns the indicies of sorted deg_a in descendingly
    top_idx_a = np.argsort(-deg_a)[:Nadv]
    top_idx_u = np.argsort(-deg_u)[:Nadv]
    print(f'Showing the top degree nodes:')
    print(f'Top degree Za Nodes indices: {top_idx_a}')
    print(f'Top degree Zu Nodes indices: {top_idx_u}')

    # top degree nodes
    # dict: index, node value
    Za_top_degree_Adv_Node = {}
    for index in top_idx_a:
        Za_top_degree_Adv_Node[index]=za_nodes[index]
    print(f'Za_top_degree_Adv_Node:\n{Za_top_degree_Adv_Node}')

    Zu_top_degree_Adv_Node = {}
    for index in top_idx_u:
        Zu_top_degree_Adv_Node[index]=zu_nodes[index]
    print(f'Zu_top_degree_Adv_Node:\n{Zu_top_degree_Adv_Node}')
    

  

    # Build tensors
    Za_t = torch.from_numpy(Za).to(device)   # (n, d)
    Zu_t = torch.from_numpy(Zu).to(device)   # (m, d)

    # Only training uses top nodes as inputs; but for CSLS we need full matrices
    Za_top = Za[top_idx_a]   # numpy (Nadv, d)
    Zu_top = Zu[top_idx_u]   # numpy (Nadv, d)
    print(f'Top degree Za Nodes: {Za_top}')
    print(f'Top degree Zu Nodes: {Zu_top}')

    # models
    discriminator = Discriminator(d, hidden=disc_hidden).to(device)
    # this the linear transform W(dxd) matrix
    mapper = LinearMapper(d).to(device)

    # optimizers
    opt_D = torch.optim.SGD(discriminator.parameters(), lr=disc_lr)
    opt_W = torch.optim.Adam(mapper.parameters(), lr=W_lr)

    # BCE loss (reduction mean) will equal the average over the batch; this maps to the formulas.
    bce = nn.BCELoss(reduction='mean')

    history = {
        "disc_loss": [],
        "W_loss": [],
        "cs_avg_cos": [],   # selection metric per epoch
    }

    # # Save best-mapped embeddings of top nodes
    Za_mapped_top_best = None
    # You also need Zu for refinement
    Zu_full = None

    best_metric = -1e9
    best_W_state = None
    best_mapping = None

    num_batches = math.ceil(Nadv / batch_size)

    # Pre-convert top arrays to tensors for fast sampling
    Za_top_t = torch.from_numpy(Za_top).to(device)  # (Nadv, d)
    Zu_top_t = torch.from_numpy(Zu_top).to(device)  # (Nadv, d)

    for epoch in range(1, W_epochs + 1):
        # We'll shuffle indices each epoch
        perm_a = np.random.permutation(Nadv)
        perm_u = np.random.permutation(Nadv)

        epoch_disc_loss = 0.0
        epoch_W_loss = 0.0

        discriminator.train()
        mapper.train()

        for b in range(num_batches):
            # batch indices
            idx_a = perm_a[b*batch_size:(b+1)*batch_size]
            idx_u = perm_u[b*batch_size:(b+1)*batch_size]
            if len(idx_a) == 0 or len(idx_u) == 0:
                continue

            # ---------- Discriminator update ----------
            opt_D.zero_grad()
            # Positive examples: mapped Za_top -> label 1
            Za_batch = Za_top_t[idx_a]                 # (bs, d)
            Za_mapped = mapper(Za_batch)               # (bs, d)
            p_pos, _ = discriminator(Za_mapped)                 # probs for Za_mapped being from Za
            labels_pos = torch.ones_like(p_pos)        # 1

            # Negative examples: Zu_top -> label 0 (since they are from Zu)
            Zu_batch = Zu_top_t[idx_u]
            p_neg, _ = discriminator(Zu_batch)
            labels_neg = torch.zeros_like(p_neg)

            # Build combined batch for BCE (or compute two BCE terms)
            loss_pos = bce(p_pos, labels_pos)
            loss_neg = bce(p_neg, labels_neg)
            loss_D = loss_pos + loss_neg
            loss_D.backward()
            opt_D.step()
            epoch_disc_loss += loss_D.item()

            # ---------- Mapper (W) update ----------
            # We update W to **confuse** the discriminator according to eq (5):
            # L_W = - (1/n) sum log(1 - P(W zia)) - (1/m) sum log(P(zju))
            # Which corresponds to: for Za_mapped we want discriminator to output 0 (label 0),
            # and for Zu_batch we want discriminator to output 1 (label 1).
            opt_W.zero_grad()
            Za_batch = Za_top_t[idx_a]
            Za_mapped = mapper(Za_batch)
            p_pos_w, _ = discriminator(Za_mapped)   # probs under current D

            Zu_batch = Zu_top_t[idx_u]
            p_neg_w, _ = discriminator(Zu_batch)

            # labels inverted compared to D training
            labels_for_W_pos = torch.zeros_like(p_pos_w)  # want D(WZa)=0
            labels_for_W_neg = torch.ones_like(p_neg_w)   # want D(Zu)=1

            loss_W_pos = bce(p_pos_w, labels_for_W_pos)
            loss_W_neg = bce(p_neg_w, labels_for_W_neg)
            loss_W = loss_W_pos + loss_W_neg
            loss_W.backward()
            opt_W.step()
            epoch_W_loss += loss_W.item()

        # average losses for epoch
        epoch_disc_loss /= max(1, num_batches)
        epoch_W_loss /= max(1, num_batches)
        history['disc_loss'].append(epoch_disc_loss)
        history['W_loss'].append(epoch_W_loss)

        # ---------- Model selection via CSLS ----------
        mapper.eval()
        discriminator.eval()
        with torch.no_grad():
            # map all Za (or at least the top selection) into target space
            Za_mapped_all = mapper(torch.from_numpy(Za).to(device)).cpu().numpy()  # (n,d)
            # compute CSLS between mapped Za_top (use top_idx_a) and all Zu
            Za_mapped_top = Za_mapped_all[top_idx_a]  # (Nadv, d)
            # this method is for the equation 7
            # 𝐶𝑆𝐿𝑆(Wzai ,zuj )= 2𝑐𝑜𝑠(Wza𝑖,zuj )−𝐷𝑢(Wzai )−𝐷𝑎(zuj )... (7)
            cs = csls_scores(Za_mapped_top, Zu, K=K_csls)  # (Nadv, m)
            # for each source top node pick argmax j
            best_j = np.argmax(cs, axis=1)   # (Nadv,)
            # compute average cosine similarity of these matched pairs (metric for selection)
            # only cosine similarity matrix
            # this whole mean process is the equation 6
            # like: # 𝐷𝑢(Wzai )=(1/𝐾) sum of zj∈𝑆𝑢(Wzai ) 𝑐𝑜𝑠(Wzai ,zj), ...(6)
            sims = cosine_sim_matrix(Za_mapped_top, Zu)   # (Nadv, m)
            matched_cos = sims[np.arange(len(best_j)), best_j]
            avg_cos = float(np.mean(matched_cos))
            history['cs_avg_cos'].append(avg_cos)

        if verbose:
            print(f"Epoch {epoch}/{W_epochs} | D_loss={epoch_disc_loss:.6f} | W_loss={epoch_W_loss:.6f} | avg_cos={avg_cos:.6f}")

        # save best model by avg cosine
        if avg_cos > best_metric:
            best_metric = avg_cos
            best_W_state = {k: v.cpu().clone() for k, v in mapper.state_dict().items()}
            # Save mapping (source node -> best target node)
            mapping = {}
            for i_src, j_tgt in enumerate(best_j):
                src_node = za_nodes[top_idx_a[i_src]]
                tgt_node = zu_nodes[j_tgt]
                mapping[src_node] = tgt_node
            best_mapping = mapping
            
            # Save best-mapped embeddings of top nodes
            Za_mapped_top_best = Za_mapped_top.copy()

            # You also need Zu for refinement
            Zu_full = Zu.copy()
            
            
            if verbose:
                print(f"  -> New best avg_cos={best_metric:.6f} (saved)")

    # best W
    key = "W.weight"
    print(f'Length of best_W_state: {len(best_W_state)}')
    print(f'Content of best_W_state:\n{best_W_state}')
    print(f'Shape of best_W_state: {best_W_state[key].shape}')
    print(f'Content of best_W_state[{key}]:\n{best_W_state[key]}')
    # print(f'Key of final_W_state: {final_W_state.keys()}')
    # print(f'Length of value from final_W_state: {len(final_W_state.values())}')
    
    
    # final W
    final_W_state = {k: v.cpu().clone() for k, v in mapper.state_dict().items()}
    print(f'Length of final_W_state: {len(final_W_state)}')
    print(f'Key of final_W_state: {final_W_state.keys()}')
    print(f'Length of value from final_W_state: {len(final_W_state.values())}')
    print(f'Shape of final_W_state: {final_W_state[key].shape}')    
    print(f'Values of final_W_state: {final_W_state[key]}')

    print(f'Number of best_mapping: {len(best_mapping)}')
    print(f'Content of best_mapping:\n{best_mapping}')

    return {
        "best_W_state": best_W_state,
        "final_W_state": final_W_state,
        "best_mapping": best_mapping,
        "history": history,
        "top_src_nodes": [za_nodes[i] for i in top_idx_a],
        "top_tgt_nodes": [zu_nodes[i] for i in top_idx_u],
        "best_metric": best_metric,
         # Important for refinement
        "Za_mapped_top_best": Za_mapped_top_best,
        "Zu_full": Zu_full,
        "top_idx_a": top_idx_a,
        "top_idx_u": top_idx_u,
    }

import networkx as nx

def load_graph_from_edgelist(path, delimiter=None):
    """
    Load an undirected graph from an edge list file.
    path: edge list text file
    delimiter: specify ' ' or ',' or '\t' etc. If None, NetworkX auto-detects whitespace.
    """
    G = nx.read_edgelist(path, delimiter=delimiter, nodetype=str)
    return G

def read_mapping(path: str) -> Dict[str,str]:
    mapping = {}
    if not path:
        return mapping
    with open(path, 'r') as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            parts = ln.split()
            if len(parts) < 2:
                continue
            a, b = parts[0], parts[1]
            mapping[str(a)] = str(b)
    return mapping
# ---------- Example usage ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Adversarial alignment of Za -> Zu")
    base_path = "/home/achakroborti1/PWAC_FALL25/individual_project"
    
    parser.add_argument("--za", default=f"{base_path}/GAE_based_outputs/Za_embeddings.txt", help="path to Za embeddings (node \\t vec...)")
    parser.add_argument("--zu", default=f"{base_path}/GAE_based_outputs/Zu_embeddings.txt", help="path to Zu embeddings")
    parser.add_argument("--edges_a", default=f"{base_path}/data/validation_dataset/validation_G1.edgelist.txt", help="optional edges file for Ga to compute degrees")
    parser.add_argument("--edges_u", default=f"{base_path}/data/validation_dataset/validation_G2.edgelist.txt", help="optional edges file for Gu to compute degrees")
    # parser.add_argument("--edges_a", default=f"{base_path}/data/seed_free/unseed_G1.edgelist", help="optional edges file for Ga to compute degrees")
    # parser.add_argument("--edges_u", default=f"{base_path}/data/seed_free/unseed_G2.edgelist", help="optional edges file for Gu to compute degrees")

    parser.add_argument("--validation_pairs", default=f"{base_path}/data/validation_dataset/validation_seed_mapping.txt", help="optional edges file for Gu to compute degrees")    
    parser.add_argument("--Nadv", type=int, default=100)#2000, 500
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--out_dir", default="./GAE_based_outputs")
    args = parser.parse_args()
    print(f'All arguments: {args}')
    
    Ga_set = set()
    Za_set = set()

    Ga = load_graph_from_edgelist(args.edges_a)
    print(f'Loaded Ga graph')
    print(Ga.number_of_nodes(), Ga.number_of_edges())
    UTIL.print_top_nodes_by_degree_with_nodes_original_indices(Ga)
 
    Gu = load_graph_from_edgelist(args.edges_u)
    print(f'Loaded Gu graph')
    print(Gu.number_of_nodes(), Gu.number_of_edges())
    UTIL.print_top_nodes_by_degree_with_nodes_original_indices(Gu)
    
    za_nodes, Za = load_embeddings(args.za)
    print(f'Number of za_nodes: \n{len(za_nodes)}')
    print(f'za_nodes: \n{za_nodes}')
    print(f'Za: \n{Za}')

    # print(f'================Ga nodes and Za nodes are equal: {Ga_set==Za_set}')

    zu_nodes, Zu = load_embeddings(args.zu)
    print(f'\nNumber of zu_nodes: \n{len(zu_nodes)}')
    print(f'\nzu_nodes: \n{zu_nodes}')
    print(f'Zu: \n{Zu}')

    result = adversarial_alignment(
        za_nodes=za_nodes,
        Za=Za,
        zu_nodes=zu_nodes,
        Zu=Zu,
        edges_a_path=args.edges_a,
        edges_u_path=args.edges_u,
        Nadv=args.Nadv,
        batch_size=args.batch,
        W_epochs=args.epochs,
        K_csls=args.K,
        verbose=True
    )
    np.savez(
        f"{args.out_dir}/adv_align_out.npz",
        best_metric=result["best_metric"],
        best_mapping=result["best_mapping"],     # dict OK
        history=result["history"],               # dict OK
        best_W_state=result["best_W_state"],     # matrix OK
        final_W_state=result["final_W_state"],   # matrix OK
        top_src_nodes=result["top_src_nodes"],   # list or np array OK
        top_tgt_nodes=result["top_tgt_nodes"],   # list or np array OK
        Za_mapped_top_best=result["Za_mapped_top_best"],
        Zu_full=result["Zu_full"],
        top_idx_a = result["top_idx_a"],
        top_idx_u = result["top_idx_u"],
        allow_pickle=True
        )

    validation_pairs = read_mapping(args.validation_pairs)
    print(f"validation_pairs: {validation_pairs}")

    best_mapping = result["best_mapping"]
    match_count = 0

    for best_mapping_key in best_mapping:
        if best_mapping_key in validation_pairs.keys() and best_mapping[best_mapping_key]==validation_pairs[best_mapping_key]:
            match_count+=1

    print(f'Best mapping match count with validation pairs: {match_count}')
    
    

    


# ### Notes & clarifications

# * **Loss implementation**: I implemented the discriminator loss as standard BCE where `Za_mapped` are labeled `1` (from (G_a)) and `Zu` labeled `0` — that yields the same per-sample summed terms as equation (4). For (W)'s objective (equation (5)) I inverted labels (Za_mapped -> 0; Zu -> 1) and used BCE; that implements the negative-log terms you specified and updates only (W).
# * **Batching**: both discriminator and (W) updates use the same batch size and sampled indices; each batch does one D step and one W step as in many adversarial training loops.
# * **Degrees / Top nodes**: The script expects optional edge lists to compute degrees and pick `Nadv` highest-degree nodes. If you don't provide edge lists it falls back to a deterministic ranking (so the code still runs).
# * **CSLS**: Uses the standard K-nearest averaging of cosine similarities. The `cs_avg_cos` used for model selection is the *average cosine similarity of matched pairs* (your described selection metric).
# * **Device (GPU)**: Code automatically uses GPU if available.
# * **Mapping output**: The script returns `best_mapping` as a dict mapping top source nodes → best target node found by CSLS at the best epoch.
# * **Hyperparams to match paper**:

#   * discriminator: SGD lr=0.1 (set),
#   * W optimizer: Adam lr=0.01 (set),
#   * batch size default 32 (set).
# * **Possible improvements**:

#   * Add negative-sampling strategies or temperature-scaled scoring.
#   * Use label smoothing or gradient penalty to stabilize training.
#   * Use orthogonalization or Procrustes post-processing for W.
#   * Evaluate matching quality with an external ground-truth seed set if available.

# ---

# If you want, I can:

# * adapt the code to accept embeddings saved in NumPy `.npy` or `.npz` formats,
# * add an evaluation routine (AUC / precision@k) if you have ground-truth node correspondences,
# * implement Procrustes/orthogonal mapping and compare,
# * or convert models to use PyTorch Geometric tensors for large-scale efficiency.

# Which one next?
