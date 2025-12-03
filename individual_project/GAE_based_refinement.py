#!/usr/bin/env python3
import argparse
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
import math
import sys
import warnings

# ---------------------------
# Utilities
# ---------------------------
def safe_item(x):
    """If x is a 0-d numpy array containing a Python object, return .item(); otherwise return x."""
    if isinstance(x, np.ndarray) and x.shape == ():
        return x.item()
    return x

def load_npz_state(path):
    data = np.load(path, allow_pickle=True)
    # Convert to dictionary-like with safe_item applied where possible
    state = {}
    for k in data.files:
        state[k] = safe_item(data[k])
    return state

def load_embeddings_txt(path):
    """
    Load node list and embedding matrix from 'node <vals...>' text file.
    Returns: nodes (list of str), embeddings (np.ndarray float32)
    """
    nodes = []
    vecs = []
    with open(path, 'r') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            node = parts[0]
            vals = list(map(float, parts[1:]))
            nodes.append(node)
            vecs.append(vals)
    return nodes, np.array(vecs, dtype=np.float32)

def get_W_matrix(possible_W):
    """
    Accept several stored formats and return a numpy 2D array W (d x d).
    Formats handled:
      - numpy array (d x d)
      - torch state_dict saved as a dict-like mapping keys->arrays (e.g., {'W.weight': array(...)})
      - a Python dict with a single matrix entry
    """
    if possible_W is None:
        return None
    # If it's already numpy 2D
    if isinstance(possible_W, np.ndarray):
        if possible_W.ndim == 2:
            return possible_W.astype(np.float64)
        # If it's an object array with one element (rare), try .item()
        if possible_W.shape == ():
            return get_W_matrix(possible_W.item())

    # dict-like
    if isinstance(possible_W, dict):
        # Common key names to check
        for key in ["W.weight", "W.weight.cpu()", "weight", "matrix", "W"]:
            if key in possible_W:
                mat = possible_W[key]
                if isinstance(mat, np.ndarray):
                    return mat.astype(np.float64)
                # else maybe it's a list
                return np.array(mat, dtype=np.float64)
        # If any value is a 2D array, take the first 2D value
        for v in possible_W.values():
            if isinstance(v, np.ndarray) and v.ndim == 2:
                return v.astype(np.float64)
            # handle nested 0-d object arrays
            if isinstance(v, np.ndarray) and v.shape == ():
                try:
                    vv = v.item()
                    if isinstance(vv, np.ndarray) and vv.ndim == 2:
                        return vv.astype(np.float64)
                except Exception:
                    pass
    # If it's a list/tuple that can be converted
    if isinstance(possible_W, (list, tuple)):
        arr = np.asarray(possible_W, dtype=np.float64)
        if arr.ndim == 2:
            return arr
    raise ValueError("Unrecognized W format; gave type: {}".format(type(possible_W)))

# ---------------------------
# Cosine and CSLS
# ---------------------------
def normalize_rows(X, eps=1e-8):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < eps] = eps
    return X / norms

def cosine_sim_matrix(A, B):
    """
    Compute cosine similarity A @ B.T where rows are vectors.
    """
    A_n = normalize_rows(A)
    B_n = normalize_rows(B)
    return np.dot(A_n, B_n.T)

def knn_indices(target_matrix, query_matrix, K):
    """
    Compute indices of K nearest neighbors (by cosine similarity) in target_matrix for each row of query_matrix.
    Returns indices: shape (n_query, K)
    """
    # sklearn NearestNeighbors works with distance metrics; use cosine distance
    nn = NearestNeighbors(n_neighbors=min(K, target_matrix.shape[0]), metric='cosine')
    nn.fit(target_matrix)
    dists, idx = nn.kneighbors(query_matrix, return_distance=True)
    return idx, dists

def csls_scores_batch(source, target, K=10):
    """
    Vectorized CSLS computation between source (n x d) and target (m x d).
    Returns CSLS matrix (n x m).
    """
    # pairwise cosine matrix
    sims = cosine_sim_matrix(source, target)  # n x m

    # D_u: for each source row, average of top-K similarities to target
    K_u = min(K, sims.shape[1])
    # use partial sort
    topk_u = -np.partition(-sims, K_u-1, axis=1)[:, :K_u]
    Du = topk_u.mean(axis=1)  # (n,)

    # D_a: for each target row, average of top-K similarities to source
    sims_t = sims.T  # m x n
    K_a = min(K, sims_t.shape[1])
    topk_a = -np.partition(-sims_t, K_a-1, axis=1)[:, :K_a]
    Da = topk_a.mean(axis=1)  # (m,)

    # broadcast
    cs = 2.0 * sims - Du[:, None] - Da[None, :]
    return cs

# =================
def build_one_to_one_seeds(candidate_sets, cs_matrix):
    """
    candidate_sets: dict {src → [top 10 target idx]}
    cs_matrix: full CSLS matrix (n_src × n_tgt)
    """
    pairs = []

    # collect all candidate edges with weights
    for src, tgts in candidate_sets.items():
        for tgt in tgts:
            score = cs_matrix[src, tgt]
            pairs.append((score, src, tgt))

    # sort by descending score
    pairs.sort(reverse=True)

    used_src = set()
    used_tgt = set()
    seeds = {}

    for score, src, tgt in pairs:
        if src in used_src:
            continue
        if tgt in used_tgt:
            continue
        seeds[src] = tgt
        used_src.add(src)
        used_tgt.add(tgt)

    return seeds



# ---------------------------
# Refinement main
# ---------------------------
def refinement_from_state(
    Za, Zu,
    alignment_state,
    top_idx_a=None, top_idx_u=None,
    use_best=True,
    K_csls=10,
    top_M=500,
    anchor_selection_mutual=True
):
    """
    Za: (n, d) numpy
    Zu: (m, d) numpy
    alignment_state: dict loaded from adv npz (with keys like best_W_state, final_W_state, Za_mapped_top_best, etc)
    top_idx_a, top_idx_u: arrays of indices of top-degree nodes (if saved in state they will be used)
    anchor_selection_mutual: if True require mutual nearest (optional)
    """
    # 1) Extract W
    best_W_raw = alignment_state.get("best_W_state", None)
    final_W_raw = alignment_state.get("final_W_state", None)

    if use_best:
        W_raw = best_W_raw if best_W_raw is not None else final_W_raw
    else:
        W_raw = final_W_raw if final_W_raw is not None else best_W_raw

    W = get_W_matrix(W_raw)
    d = W.shape[0]
    print(f"[info] Loaded W shape: {W.shape}")

    # 2) top indices (if not passed)
    if top_idx_a is None:
        top_idx_a = alignment_state.get("top_idx_a", None)
    if top_idx_u is None:
        top_idx_u = alignment_state.get("top_idx_u", None)
    if top_idx_a is None or top_idx_u is None:
        # attempt to use top_src_nodes / top_tgt_nodes arrays of node ids if provided
        top_idx_a = alignment_state.get("top_src_nodes", None)
        top_idx_u = alignment_state.get("top_tgt_nodes", None)
    # If they are node lists, we must have mapping from node->index (not available here), so expect indices.
    if isinstance(top_idx_a, np.ndarray) or isinstance(top_idx_a, list):
        top_idx_a = np.asarray(top_idx_a, dtype=int)
    else:
        raise ValueError("top_idx_a must be provided as array/list of indices in Za order.")
    if isinstance(top_idx_u, np.ndarray) or isinstance(top_idx_u, list):
        top_idx_u = np.asarray(top_idx_u, dtype=int)
    else:
        raise ValueError("top_idx_u must be provided as array/list of indices in Zu order.")

    # 3) Produce Za_mapped_top: prefer saved Za_mapped_top_best; else compute
    Za_mapped_top_best = alignment_state.get("Za_mapped_top_best", None)
    if Za_mapped_top_best is not None:
        Za_mapped_top = np.asarray(Za_mapped_top_best)
        print("[info] Using saved Za_mapped_top_best.")
    else:
        # Map Za via W
        Za_mapped_all = (Za @ W.T)
        Za_mapped_top = Za_mapped_all[top_idx_a]
        print("[info] Computed Za_mapped_top from W.")

    # 4) Zu subset for anchors (we use Zu[top_idx_u])
    Zu_anchor = Zu[top_idx_u]

    # 5) Compute CSLS between Za_mapped_top and Zu_anchor
    # Hungarian
    cs = csls_scores_batch(Za_mapped_top, Zu_anchor, K=K_csls)

    # Hungarian algorithm requires a *cost* matrix. 
    # Higher CSLS = better, so we negate it to convert to cost.
    from scipy.optimize import linear_sum_assignment

    cost_matrix = -cs

    # Compute global optimal 1-to-1 matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build anchor pairs from matching (global, unique)
    raw_anchor_pairs = [(top_idx_a[i], top_idx_u[j]) for i, j in zip(row_ind, col_ind)]

    # Sort anchors by similarity strength
    similarities = cs[row_ind, col_ind]
    sorted_idx = np.argsort(-similarities)

    raw_anchor_pairs = [raw_anchor_pairs[k] for k in sorted_idx]
    similarities = similarities[sorted_idx]

    # Optional: keep top-N anchors (recommended)
    N_anchor = min(300, len(raw_anchor_pairs))   # You can tune 100–300
    anchor_pairs = raw_anchor_pairs[:N_anchor]

    print(f"[info] Global Hungarian anchor selection produced {len(anchor_pairs)} unique anchors.")
   
    # 6) Build Zhat matrices (d x k) for procrustes: we use shape (d, k) or (k, d); follow paper: columns = anchor vectors
    Za_hat = np.stack([Za[i] for i, _ in anchor_pairs], axis=1)  # shape (d, k)
    Zu_hat = np.stack([Zu[j] for _, j in anchor_pairs], axis=1)  # shape (d, k)

    # 7) Procrustes (orthogonal)
    M = Zu_hat @ Za_hat.T  # (d x d)
    U, S, Vt = np.linalg.svd(M)
    W_refined = U @ Vt
    print("[info] Procrustes done. W_refined shape:", W_refined.shape)

    # 8) Recompute candidate sets: for each of top_M source-high-degree nodes (passed as top_idx_a[:top_M]), return top-10 CSLS matches in Zu
    top_M_eff = min(top_M, len(top_idx_a))
    source_indices_for_candidates = top_idx_a[:top_M_eff]
    Za_mapped_refined = (Za @ W_refined.T)
    cs_full = csls_scores_batch(Za_mapped_refined[source_indices_for_candidates], Zu, K=K_csls)
    candidate_sets = {}
    for idx_local, src_idx in enumerate(source_indices_for_candidates):
        top10_local = np.argsort(-cs_full[idx_local])[:10]
        # top10_local are indices in Zu (global)
        candidate_sets[int(src_idx)] = [int(x) for x in top10_local]

    return {
        "W_refined": W_refined,
        "anchor_pairs": anchor_pairs,
        "candidate_sets": candidate_sets
    }
def get_first_one(candidate_sets):
    already_mapped= set()
    seeds = {}
    for key in candidate_sets.keys():
        if candidate_sets[key][0] in already_mapped:
            continue
        else:
            already_mapped.add(candidate_sets[key][0])
            seeds[key]=candidate_sets[key][0]
        
    print(f'number of keys: {len(seeds.keys())}')
    return seeds
# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    base_path = "/home/achakroborti1/PWAC_FALL25/individual_project"

    parser.add_argument("--za", default=f"{base_path}/seed_free/chat_gpt_embedings/Za_embeddings.txt", help="path to Za embeddings (node \\t vec...)")
    parser.add_argument("--zu", default=f"{base_path}/seed_free/chat_gpt_embedings/Zu_embeddings.txt", help="path to Zu embeddings")
    parser.add_argument("--state", default=f"{base_path}/seed_free/chat_gpt_embedings/adv_align_out.npz", help="optional edges file for Gu to compute degrees")
   
    parser.add_argument("--out_dir", default="./GAE_based_outputs")
    parser.add_argument("--use_best", action="store_true", help="use best W (default True if present)")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--top_M", type=int, default=2000)#500
    parser.add_argument("--mutual", action="store_true", help="require mutual nearest for anchors")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    za_nodes, Za = load_embeddings_txt(args.za)
    zu_nodes, Zu = load_embeddings_txt(args.zu)

    state = load_npz_state(args.state)

    # get top_idx_a / top_idx_u from saved state (prefer index arrays)
    top_idx_a = state.get("top_idx_a", None)
    top_idx_u = state.get("top_idx_u", None)
    # if state saved top_src_nodes as names, try to map to indices using node lists
    if top_idx_a is None and "top_src_nodes" in state:
        # top_src_nodes may be list of node ids; convert to indices
        saved_ids = state["top_src_nodes"]
        if isinstance(saved_ids, (list, np.ndarray)):
            id_to_idx = {n:i for i,n in enumerate(za_nodes)}
            try:
                top_idx_a = np.array([id_to_idx[x] for x in saved_ids], dtype=int)
            except Exception:
                top_idx_a = None
    if top_idx_u is None and "top_tgt_nodes" in state:
        saved_ids = state["top_tgt_nodes"]
        if isinstance(saved_ids, (list, np.ndarray)):
            id_to_idx = {n:i for i,n in enumerate(zu_nodes)}
            try:
                top_idx_u = np.array([id_to_idx[x] for x in saved_ids], dtype=int)
            except Exception:
                top_idx_u = None

    if top_idx_a is None or top_idx_u is None:
        print("[warning] top_idx_a / top_idx_u not found; you must pass arrays of indices in the state file.")
        sys.exit(1)
    
    # run refinement
    refinement_result = refinement_from_state(
        Za=Za, Zu=Zu,
        alignment_state=state,
        top_idx_a=top_idx_a, top_idx_u=top_idx_u,
        use_best=args.use_best,
        K_csls=args.K,
        top_M=args.top_M,
        anchor_selection_mutual=args.mutual
    )
    print(f'Done!')
    # save outputs
    np.save(os.path.join(args.out_dir, "W_refined.npy"), refinement_result["W_refined"])
    print(f'Saved W_refined!')
    
    np.save(os.path.join(args.out_dir, "anchor_pairs.npy"), np.array(refinement_result["anchor_pairs"], dtype=np.int64))
    print(f'Saved anchor_pairs!')

    np.save(os.path.join(args.out_dir, "candidate_sets.npy"), refinement_result["candidate_sets"])
    print(f'Saved candidate_sets!')
    candidate_sets = refinement_result["candidate_sets"]
    print(f'Length of candidate_sets: {len(candidate_sets)}')
    print(f'candidate_sets:\n{candidate_sets}')

if __name__ == "__main__":
    main()