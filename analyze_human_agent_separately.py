import os
import sys
import numpy as np
import pandas as pd
import torch
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModel
import time
import pickle
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import matplotlib.lines as mlines
from mutual_adaptation_analysis_individual import (
    compute_mutual_adaptation_metrics_individual,
    visualize_mutual_adaptation_individual,
)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils as ut


def load_dataframe(fname):
    try:
        dataframe = pickle.load(open(fname, 'rb'))
    except:
        raise ValueError('NO DATAFILE FOUND', fname)
    dataframe.pop('rules', None)
    return dataframe


@torch.no_grad()
def encode_sentences(texts, tokenizer, model, device):
    """
    """
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    outputs = model(**inputs)
    # mean pooling
    attention_mask = inputs["attention_mask"]
    emb = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
    # L2 normalize
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu()


def pairwise_similarity_matrix(embeddings: torch.Tensor) -> np.ndarray:
    """
    """
    return (embeddings @ embeddings.T).numpy()


def bootstrap_consensus_from_pairs(sim_matrix, sample_ratio, n_repeats):
    """
    """
    vals = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    n = len(vals)
    boot_means = []
    sample_size = int(n * sample_ratio)

    for _ in range(n_repeats):
        sample = np.random.choice(vals, sample_size, replace=True)
        boot_means.append(np.mean(sample))

    return boot_means


def compute_human_agent_consensus_separated_bootstrap(agent_ratio, agent_style, simulation, tokenizer, model, device, seed=None):
    """
    """

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        # GPU
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    any_user = next(iter(simulation))
    n_rounds = len(simulation[any_user]["my_history"])
    user_ids = list(simulation.keys())

    is_pure_human = (agent_ratio=="Agents 0%")

    results = []
    for r in range(n_rounds):
        human_exprs, agent_exprs = [], []
        for uid in user_ids:
            history = simulation[uid].get("my_history", [])
            if len(history) > r:
                if str(uid).startswith("agent_"):
                    agent_exprs.append(history[r])
                else:
                    human_exprs.append(history[r])
        if is_pure_human:
            embeddings = encode_sentences(human_exprs, tokenizer, model, device)
            sim_h = pairwise_similarity_matrix(embeddings)
            h_samples = bootstrap_consensus_from_pairs(sim_h, sample_ratio=0.6, n_repeats=100)
            for val in h_samples:
                results.append({"agent_ratio": agent_ratio, "agent_style": agent_style, "round": r, "human_consensus": val, "agent_consensus": np.nan, "divergence": np.nan})
            continue
        if not human_exprs or not agent_exprs:
            continue

        all_exprs = human_exprs + agent_exprs
        embeddings = encode_sentences(all_exprs, tokenizer, model, device)
        n_h = len(human_exprs)
        human_emb = embeddings[:n_h]
        agent_emb = embeddings[n_h:]

        # human only
        h_samples = []
        if len(human_emb) > 1:
            sim_h = pairwise_similarity_matrix(human_emb)
            h_samples = bootstrap_consensus_from_pairs(sim_h, sample_ratio=0.6, n_repeats=100)

        # agent only
        a_samples = []
        if len(agent_emb) > 1:
            sim_a = pairwise_similarity_matrix(agent_emb)
            a_samples = bootstrap_consensus_from_pairs(sim_a, sample_ratio=0.6, n_repeats=100)

        # human vs agent
        h_centroid = torch.mean(human_emb, dim=0, keepdim=True)
        a_centroid = torch.mean(agent_emb, dim=0, keepdim=True)
        h_centroid = torch.nn.functional.normalize(h_centroid, p=2, dim=1)
        a_centroid = torch.nn.functional.normalize(a_centroid, p=2, dim=1)

        full_divergence = float(1 - torch.mm(h_centroid, a_centroid.T).item())

        # Bootstrap n out of n
        n_h, n_a = human_emb.shape[0], agent_emb.shape[0]
        div_repeats = 100
        div_samples = []
        for _ in range(div_repeats):
            idx_h = np.random.choice(n_h, size=n_h, replace=True)
            idx_a = np.random.choice(n_a, size=n_a, replace=True)
            h_c = torch.mean(human_emb[idx_h], dim=0, keepdim=True)
            a_c = torch.mean(agent_emb[idx_a], dim=0, keepdim=True)
            h_c = torch.nn.functional.normalize(h_c, p=2, dim=1)
            a_c = torch.nn.functional.normalize(a_c, p=2, dim=1)
            div_samples.append(1 - torch.mm(h_c, a_c.T).item())

        max_len = max(len(h_samples), len(a_samples), len(div_samples))
        for i in range(max_len):
            results.append({
                "agent_ratio": agent_ratio, "agent_style": agent_style, "round": r,
                "human_consensus": h_samples[i] if i < len(h_samples) else np.nan,
                "agent_consensus": a_samples[i] if i < len(a_samples) else np.nan,
                "divergence": div_samples[i] if i < len(div_samples) else full_divergence
            })

    return pd.DataFrame(results)



def compute_human_agent_consensus_separated(agent_ratio, agent_style, simulation, tokenizer, model, device):
    """
    """
    any_user = next(iter(simulation))
    n_rounds = len(simulation[any_user]["my_history"])
    user_ids = list(simulation.keys())

    is_pure_human = (agent_ratio=="Agents 0%")

    results = []
    for r in range(n_rounds):
        human_exprs, agent_exprs = [], []
        for uid in user_ids:
            history = simulation[uid].get("my_history", [])
            if len(history) > r:
                if str(uid).startswith("agent_"):
                    agent_exprs.append(history[r])
                else:
                    human_exprs.append(history[r])
        if is_pure_human:
            embeddings = encode_sentences(human_exprs, tokenizer, model, device)
            sim_h = pairwise_similarity_matrix(embeddings)
            hvals = sim_h[np.triu_indices_from(sim_h, k=1)]
            human_cons = float(np.mean(hvals))
            results.append({ "agent_ratio": agent_ratio, "agent_style": agent_style, "round": r, "human_consensus": human_cons, "agent_consensus": np.nan, "divergence": np.nan})
            continue
        if not human_exprs or not agent_exprs:
            continue

        all_exprs = human_exprs + agent_exprs
        embeddings = encode_sentences(all_exprs, tokenizer, model, device)
        n_h = len(human_exprs)
        human_emb = embeddings[:n_h]
        agent_emb = embeddings[n_h:]

        # human only
        if len(human_emb) > 1:
            sim_h = pairwise_similarity_matrix(human_emb)
            hvals = sim_h[np.triu_indices_from(sim_h, k=1)]
            human_cons = float(np.mean(hvals))
        else:
            human_cons = np.nan

        # agent only
        if len(agent_emb) > 1:
            sim_a = pairwise_similarity_matrix(agent_emb)
            avals = sim_a[np.triu_indices_from(sim_a, k=1)]
            agent_cons = float(np.mean(avals))
        else:
            agent_cons = np.nan

        # human vs agent
        h_centroid = torch.mean(human_emb, dim=0, keepdim=True)
        a_centroid = torch.mean(agent_emb, dim=0, keepdim=True)
        h_centroid = torch.nn.functional.normalize(h_centroid, p=2, dim=1)
        a_centroid = torch.nn.functional.normalize(a_centroid, p=2, dim=1)
        divergence = float(1 - torch.mm(h_centroid, a_centroid.T).item())

        results.append({"agent_ratio": agent_ratio, "agent_style": agent_style, "round": r, "human_consensus": human_cons, "agent_consensus": agent_cons, "divergence": divergence})
    return pd.DataFrame(results)


def classify_ratio(x):
    if x in ["Agents 0%", "Agents 12.5%", "Agents 33%", "Agents 50%", "Agents 75%"]:
        return x
    else: return None


# round
def roll(g):
    g = g.sort_values("round")
    for m in ["human_mean","human_sd","agent_mean","agent_sd","div_mean","div_sd"]:
        g[m] = g[m].rolling(smooth_window, center=True, min_periods=1).mean()
    return g


def analyze_semantic_drift(simulation, tokenizer, model, device, agent_ratio, agent_style, reduce_dim=2, normalize=True):
    """
    """
    any_user = next(iter(simulation))
    n_rounds = len(simulation[any_user]["my_history"])
    user_ids = list(simulation.keys())

    human_centroids, agent_centroids = [], []
    for r in range(n_rounds):
        human_exprs, agent_exprs = [], []
        for uid in user_ids:
            history = simulation[uid].get("my_history", [])
            if len(history) > r:
                if str(uid).startswith("agent_"):
                    agent_exprs.append(history[r])
                else:
                    human_exprs.append(history[r])

        if not human_exprs or not agent_exprs:
            continue

        all_exprs = human_exprs + agent_exprs
        embeddings = encode_sentences(all_exprs, tokenizer, model, device)
        n_h = len(human_exprs)
        human_emb = embeddings[:n_h]
        agent_emb = embeddings[n_h:]

        if normalize:
            human_emb = torch.nn.functional.normalize(human_emb, p=2, dim=1)
            agent_emb = torch.nn.functional.normalize(agent_emb, p=2, dim=1)

        h_c = torch.mean(human_emb, dim=0).cpu().numpy()
        a_c = torch.mean(agent_emb, dim=0).cpu().numpy()

        human_centroids.append(h_c)
        agent_centroids.append(a_c)

    all_centroids = np.vstack(human_centroids + agent_centroids)
    pca = PCA(n_components=reduce_dim)
    coords_2d = pca.fit_transform(all_centroids)

    h_coords = coords_2d[:len(human_centroids)]
    a_coords = coords_2d[len(human_centroids):]

    df_list = []
    df_list.append(pd.DataFrame({"round": np.arange(len(human_centroids)), "group": "Human", "x": h_coords[:, 0], "y": h_coords[:, 1], "agent_ratio": agent_ratio, "agent_style": agent_style}))
    df_list.append(pd.DataFrame({"round": np.arange(len(agent_centroids)), "group": "Agent", "x": a_coords[:, 0], "y": a_coords[:, 1], "agent_ratio": agent_ratio, "agent_style": agent_style}))
    return pd.concat(df_list, ignore_index=True)




def adaptation_attribution(H, A, eps=1e-9, return_per_round=False):
    """
    """
    assert H.shape == A.shape and H.shape[0] >= 2
    T = H.shape[0] - 1
    pH_list, pA_list, sH_list, sA_list = [], [], [], []
    d_list = []

    for t in range(T):
        d = A[t] - H[t]
        dist = np.linalg.norm(d)
        if dist < eps:
            u = np.zeros_like(d)
        else:
            u = d / dist
        dH = H[t+1] - H[t]
        dA = A[t+1] - A[t]

        pH = float(np.dot(dH, u))
        pA = float(np.dot(dA, -u))
        pH_list.append(pH); pA_list.append(pA)

        sH = float(np.linalg.norm(dH - pH * u))
        sA = float(np.linalg.norm(dA - pA * (-u)))
        sH_list.append(sH); sA_list.append(sA)
        d_list.append(dist)

    C_H, C_A = np.sum(pH_list), np.sum(pA_list)
    S_H, S_A = np.sum(sH_list), np.sum(sA_list)
    total_parallel = C_H + C_A + eps

    ai_index = (C_H - C_A) / total_parallel
    human_share = C_H / total_parallel

    out = {
        "C_H": C_H, "C_A": C_A,
        "AI_index": ai_index,
        "Human_share": human_share,
        "Side_H": S_H, "Side_A": S_A,
        "mean_distance": float(np.mean(d_list)),
    }

    if return_per_round:
        per_round = pd.DataFrame({"round": np.arange(T), "pH": pH_list, "pA": pA_list, "sH": sH_list, "sA": sA_list, "distance": d_list})
        return out, per_round
    return out



def parse_ratio(x):
    if x == "Agents 0%":
        return 0.0
    return float(x.replace("Agents ", "").replace("%", ""))


def _centroid_cosine_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    x, y: shape (d,) or (1, d), assumed float tensors on CPU or GPU.
    return cosine distance = 1 - cosine_similarity
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(0)
    x = torch.nn.functional.normalize(x, p=2, dim=1)
    y = torch.nn.functional.normalize(y, p=2, dim=1)
    return float(1 - torch.mm(x, y.T).item())


def compute_final_consensus_distance_to_t0_centroids(
    agent_ratio: str,
    agent_style: str,
    simulation: dict,
    tokenizer,
    model,
    device,
    t0_round: int = 0,
    final_round: int | None = None,
    consensus_group: str = "all",   # "all" | "human" | "agent"
):
    """
    2) t=0 human centroid
    3) t=0 agent centroid

    """
    any_user = next(iter(simulation))
    n_rounds = len(simulation[any_user]["my_history"])
    user_ids = list(simulation.keys())

    if final_round is None:
        final_round = n_rounds - 1

    def collect_exprs(round_idx: int):
        human_exprs, agent_exprs = [], []
        for uid in user_ids:
            history = simulation[uid].get("my_history", [])
            if len(history) > round_idx:
                if str(uid).startswith("agent_"):
                    agent_exprs.append(history[round_idx])
                else:
                    human_exprs.append(history[round_idx])
        return human_exprs, agent_exprs

    h0_exprs, a0_exprs = collect_exprs(t0_round)

    h0_centroid = None
    a0_centroid = None

    if len(h0_exprs) > 0:
        h0_emb = encode_sentences(h0_exprs, tokenizer, model, device)  # already normalized per sentence
        h0_centroid = torch.mean(h0_emb, dim=0)  # (d,)
    if len(a0_exprs) > 0:
        a0_emb = encode_sentences(a0_exprs, tokenizer, model, device)
        a0_centroid = torch.mean(a0_emb, dim=0)

    # final consensus centroid
    hf_exprs, af_exprs = collect_exprs(final_round)

    final_centroid = None
    if consensus_group == "all":
        all_exprs = hf_exprs + af_exprs
        if len(all_exprs) > 0:
            all_emb = encode_sentences(all_exprs, tokenizer, model, device)
            final_centroid = torch.mean(all_emb, dim=0)
    elif consensus_group == "human":
        if len(hf_exprs) > 0:
            hf_emb = encode_sentences(hf_exprs, tokenizer, model, device)
            final_centroid = torch.mean(hf_emb, dim=0)
    elif consensus_group == "agent":
        if len(af_exprs) > 0:
            af_emb = encode_sentences(af_exprs, tokenizer, model, device)
            final_centroid = torch.mean(af_emb, dim=0)
    else:
        raise ValueError(f"Unknown consensus_group={consensus_group}, choose from all/human/agent.")

    # distances
    dist_final_to_h0 = np.nan
    dist_final_to_a0 = np.nan
    dist_t0_h0_to_a0 = np.nan

    if final_centroid is not None and h0_centroid is not None:
        dist_final_to_h0 = _centroid_cosine_distance(final_centroid, h0_centroid)

    if final_centroid is not None and a0_centroid is not None:
        dist_final_to_a0 = _centroid_cosine_distance(final_centroid, a0_centroid)

    if h0_centroid is not None and a0_centroid is not None:
        dist_t0_h0_to_a0 = _centroid_cosine_distance(h0_centroid, a0_centroid)

    return {
        "agent_ratio": agent_ratio,
        "agent_style": agent_style,
        "t0_round": t0_round,
        "final_round": final_round,
        "consensus_group": consensus_group,
        "dist_final_to_t0_human": dist_final_to_h0,
        "dist_final_to_t0_agent": dist_final_to_a0,
        "dist_t0_human_to_t0_agent": dist_t0_h0_to_a0,
        "n_human_t0": len(h0_exprs),
        "n_agent_t0": len(a0_exprs),
        "n_human_final": len(hf_exprs),
        "n_agent_final": len(af_exprs),
    }



def extract_t0_final_embeddings(simulation, tokenizer, model, device, t0_round: int = 0, final_round: int | None = None):
    """
    H0 A0 Hf Af
    """
    any_user = next(iter(simulation))
    n_rounds = len(simulation[any_user]["my_history"])
    user_ids = list(simulation.keys())

    if final_round is None:
        final_round = n_rounds - 1

    def collect_exprs(round_idx: int):
        human_exprs, agent_exprs = [], []
        for uid in user_ids:
            history = simulation[uid].get("my_history", [])
            if len(history) > round_idx:
                if str(uid).startswith("agent_"):
                    agent_exprs.append(history[round_idx])
                else:
                    human_exprs.append(history[round_idx])
        return human_exprs, agent_exprs

    h0_exprs, a0_exprs = collect_exprs(t0_round)
    hf_exprs, af_exprs = collect_exprs(final_round)

    H0 = encode_sentences(h0_exprs, tokenizer, model, device) if len(h0_exprs) else None
    A0 = encode_sentences(a0_exprs, tokenizer, model, device) if len(a0_exprs) else None
    Hf = encode_sentences(hf_exprs, tokenizer, model, device) if len(hf_exprs) else None
    Af = encode_sentences(af_exprs, tokenizer, model, device) if len(af_exprs) else None

    return {
        "t0_round": t0_round, "final_round": final_round,
        "h0_exprs": h0_exprs, "a0_exprs": a0_exprs, "hf_exprs": hf_exprs, "af_exprs": af_exprs,
        "H0": H0, "A0": A0, "Hf": Hf, "Af": Af,
    }



def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()

def _centroid_np(Z: np.ndarray) -> np.ndarray:
    return Z.mean(axis=0, keepdims=True)

def mean_sem(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, 0
    m = float(np.mean(x))
    sem = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return m, sem, n


@torch.no_grad()
def compute_final_round_human_dispersion(agent_ratio, agent_style, simulation, tokenizer, model, device):
    """
    """
    any_user = next(iter(simulation))
    n_rounds = len(simulation[any_user]["my_history"])
    final_r = n_rounds - 1

    user_ids = list(simulation.keys())
    human_exprs = []
    for uid in user_ids:
        history = simulation[uid].get("my_history", [])
        if len(history) > final_r:
            if not str(uid).startswith("agent_"):
                human_exprs.append(history[final_r])

    if len(human_exprs) == 0:
        return {
            "agent_ratio": agent_ratio,
            "agent_style": agent_style,
            "final_round": final_r,
            "mean_dispersion": np.nan,
            "sem_dispersion": np.nan,
            "N_human": 0,
        }

    emb = encode_sentences(human_exprs, tokenizer, model, device)  # shape (N, D)

    # centroid + normalize
    c = torch.mean(emb, dim=0, keepdim=True)
    c = torch.nn.functional.normalize(c, p=2, dim=1)  # shape (1, D)

    # cosine distance to centroid: 1 - cos sim
    dists = (1.0 - torch.mm(emb, c.T).squeeze(1)).cpu().numpy()

    m, s, n = mean_sem(dists)
    return {
        "agent_ratio": agent_ratio,
        "agent_style": agent_style,
        "final_round": final_r,
        "mean_dispersion": m,
        "sem_dispersion": s,
        "N_human": n,
    }


if __name__ == "__main__":

    smooth_window = 3
    print("Preparing Fig. 1c...")

    # Load similarity evaluator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'all-MiniLM-L6-v2'
    MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
    MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", f"processed_data/model_cache/{model_name}")
    snapshot_download(MODEL_ID, cache_dir=MODEL_CACHE_DIR)
    if not os.path.exists(MODEL_CACHE_DIR):
        model_dir = snapshot_download(MODEL_ID, cache_dir=MODEL_CACHE_DIR)
    else:
        # HF
        model_dir = ut.find_hf_model_dir(MODEL_CACHE_DIR)
    # tokenizer model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).to(device)
    model.eval()

    fnames = [
        "processed_data/A4.pkl",  # 12.5
        "processed_data/A3.pkl",  # 33.3
        "processed_data/A2.pkl",  # 50
        "processed_data/A5.pkl",  # 75
    ]

    dataframes = []
    for fname in fnames:
        frame = load_dataframe(fname)
        frame.pop('rules', None)
        dataframes.append(frame)

    agent_ratio_list = ["Agents 12.5%", "Agents 33%", "Agents 50%", "Agents 75%"]
    agent_style_list = ['Neutral',   'Neutral',      'Neutral',     'Neutral']
    agent_label_list = ["12.5%", "33.3%", "50%", "75%"]
    run_id_list = [0, 0, 0, 0]

    packs = []
    for i, dataframe in enumerate(dataframes):
        run_id = run_id_list[i]
        simulation = dataframe[run_id]["simulation"]
        pack = extract_t0_final_embeddings(
            simulation=simulation,
            tokenizer=tokenizer,
            model=model,
            device=device,
            t0_round=0,
            final_round=None,
        )
        packs.append(pack)

    print("Computing metrics...")
    metrics_list = []
    for i, pack in enumerate(packs):
        metrics = compute_mutual_adaptation_metrics_individual(
            agent_ratio=agent_ratio_list[i], pack=pack, normalize=True
        )
        metrics_list.append(metrics)

    df_metrics = pd.DataFrame(metrics_list)

    print("Generating Fig. 1c...")
    visualize_mutual_adaptation_individual(
        df_metrics, output_dir='figures', agent_label_list=agent_label_list,
        error_bar_type="sem"
    )

    raise SystemExit
