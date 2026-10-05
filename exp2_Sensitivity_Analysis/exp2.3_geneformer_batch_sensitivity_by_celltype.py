#!/usr/bin/env python3
# Dataset-specific Geneformer ISP sensitivity analysis.

# Purpose
# -------
# For one requested cell type, this script:
#   1. Reads gse_comparisons_to_test.csv.
#   2. Finds every eligible tissue/GSE/control_state/target_state comparison for that cell type.
#   3. For DOWN and UP separately, reads the matching discovery *_sig_only.csv.
#   4. Restricts the recurrent-gene list to hits from the requested cell type.
#   5. Subsets the cell-type h5ad to the requested GSE and the two requested states.
#   6. Recreates Geneformer embeddings and checks state separation.
#   7. Runs each recurrent gene INDIVIDUALLY (one-gene genes_to_perturb list), using
#      the same donor-aware sampling strategy as the discovery pipeline.
#   8. Computes the dataset-specific median cosine shift and compares its sign with
#      the pooled discovery median cosine shift.
#   9. Writes per-experiment results and one combined sensitivity table.

# Example
# -------
# python Geneformer_batch_sensitivity_by_celltype.py "Cholangiocytes"
# python Geneformer_batch_sensitivity_by_celltype.py "CD16+ NK cells" --max-isp-cells 500

import os
import sys
import re
import glob
import json
import pickle
import shutil
import hashlib
import tempfile
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

# ---- Cluster-specific Geneformer import path ---------------------------------
GENEFORMER_REPO = "/path/to/Geneformer"
if GENEFORMER_REPO not in sys.path:
    sys.path.insert(0, GENEFORMER_REPO)

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import loompy
import torch
import datasets as hf_datasets
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from scipy import stats

from geneformer import TranscriptomeTokenizer, EmbExtractor, InSilicoPerturber
import geneformer.perturber_utils as pu

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1


# =============================================================================
# Arguments
# =============================================================================
parser = argparse.ArgumentParser(
    description="Run dataset-specific Geneformer ISP sensitivity analysis for one cell type."
)
parser.add_argument("celltype", help='Cell type exactly as used in the discovery CSVs, e.g. "Cholangiocytes"')
parser.add_argument(
    "--comparisons-csv",
    default="path/to/gse_comparisons_to_test.csv",
    help="CSV with cell_type,tissue,GSE_to_test,control_state,target_state",
)
parser.add_argument(
    "--sig-dir",
    default="/path/to/sig_only",
    help="Directory containing *_sig_only.csv discovery results",
)
parser.add_argument(
    "--output-dir",
    default="/path/to/geneformer_batch_sensitivity",
    help="Root output directory",
)
parser.add_argument(
    "--model-dir",
    default=GENEFORMER_REPO,
    help="Geneformer model directory",
)
parser.add_argument("--max-isp-cells", type=int, default=500)
parser.add_argument("--min-cells-per-donor", type=int, default=5)
parser.add_argument(
    "--min-cells-state",
    type=int,
    default=20,
    help="Minimum cells in EACH state required for dataset-specific sensitivity separation testing (default: 20).",
)
parser.add_argument("--min-gene-cells", type=int, default=20)
parser.add_argument("--forward-batch-size", type=int, default=64)
parser.add_argument(
    "--skip-separation-gate",
    action="store_true",
    help="Still calculate separation, but run ISP even if the separation test fails.",
)
parser.add_argument(
    "--force",
    action="store_true",
    help="Rerun an experiment even if its per-experiment sensitivity_results.csv already exists.",
)
args = parser.parse_args()

CELL_TYPE = args.celltype
COMPARISONS_CSV = args.comparisons_csv
SIG_DIR = args.sig_dir
OUTPUT_ROOT = args.output_dir
GENEFORMER_MODEL_DIR = args.model_dir
MAX_ISP_CELLS_TOTAL = args.max_isp_cells
MIN_CELLS_PER_DONOR_ISP = args.min_cells_per_donor
MIN_CELLS_PER_STATE = args.min_cells_state
MIN_CELLS_PER_GENE = args.min_gene_cells
FORWARD_BATCH_SIZE = args.forward_batch_size

H5AD_DIRS = {
    "liver": "/path/to/liver_cell_type_adata/",
    "immune": "/path/to/immune_cell_type_adata/",
}
MAP_PATHS = {
    "liver": "/path/to/gene_symbol_to_ensembl_liver.csv",
    "immune": "/path/to/gene_symbol_to_ensembl_immune.csv",
}

CONDITION_CANDIDATES = ["broad_condition", "condition", "disease", "Status"]
GSE_CANDIDATES = [
    "GSE", "gse", "GSE_id", "gse_id", "dataset", "Dataset", "dataset_id",
    "source_dataset", "source", "study", "study_id", "geo_accession",
]
DONOR_CANDIDATES = ["patient_id", "donor", "Donor", "donor_id", "sample_id"]
CELLTYPE_CANDIDATES = ["cell_type", "celltype", "CellType"]

MODE_TO_PERTURB = {"down": "delete", "up": "overexpress"}


# =============================================================================
# GPU setup and Geneformer patch copied from the working discovery pipeline
# =============================================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    raise RuntimeError("No GPU detected. This script intentionally requires a GPU for Geneformer ISP.")

torch.cuda.set_device(0)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print(f"GPU: {torch.cuda.get_device_name(0)}")

if not getattr(pu, "_load_model_gpu_patched", False):
    _REAL_LOAD_MODEL = pu.load_model

    def _gpu_load_model(model_type, num_classes, model_directory, *a, **kw):
        model = _REAL_LOAD_MODEL(model_type, num_classes, model_directory, *a, **kw)
        model = model.to("cuda:0").eval()
        print(f"   Model device: {next(model.parameters()).device}")
        return model

    pu.load_model = _gpu_load_model
    pu._load_model_gpu_patched = True


# =============================================================================
# Helpers
# =============================================================================
def safe_name(x):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(x)).strip("_")


def normalize_celltype(x):
    return re.sub(r"\s+", " ", str(x).replace("_", " ").strip()).casefold()


def normalize_gse(x):
    return str(x).strip().casefold()


def base_gse(x):
    """GSE192740_liver -> gse192740; leaves ordinary accessions unchanged."""
    s = normalize_gse(x)
    return re.sub(r"_(liver|immune)$", "", s)


def find_celltype_h5ad(tissue, cell_type):
    root = H5AD_DIRS[tissue]
    files = sorted(glob.glob(os.path.join(root, "*.h5ad")))
    target = normalize_celltype(cell_type)
    matches = []
    for p in files:
        stem = os.path.splitext(os.path.basename(p))[0]
        if normalize_celltype(stem) == target:
            matches.append(p)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple h5ad files matched {cell_type!r} in {root}: {matches}")
    raise FileNotFoundError(
        f"No h5ad filename in {root} matched cell type {cell_type!r}. "
        "Expected filename stem to equal the cell type (underscores/spaces are normalized)."
    )


def detect_column(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find {label} column. Tried: {candidates}. Available: {list(df.columns)}")


def subset_to_gse_and_states(adata, requested_gse, control_state, target_state):
    condition_col = detect_column(adata.obs, CONDITION_CANDIDATES, "condition")
    gse_col = detect_column(adata.obs, GSE_CANDIDATES, "GSE/dataset")

    cond = adata.obs[condition_col].astype(str)
    gse_vals = adata.obs[gse_col].astype(str)

    req = normalize_gse(requested_gse)
    exact = gse_vals.map(normalize_gse) == req

    # Fallback only if exact matching finds nothing. This handles cases like
    # comparison CSV='GSE192740_liver' but obs='GSE192740'.
    if exact.sum() == 0:
        req_base = base_gse(requested_gse)
        fallback = gse_vals.map(base_gse) == req_base
        if fallback.sum() > 0:
            print(
                f"   NOTE: no exact GSE match for {requested_gse!r}; using base-accession match "
                f"on {req_base!r} ({fallback.sum()} cells)."
            )
            exact = fallback

    state_mask = cond.isin([str(control_state), str(target_state)])
    mask = exact & state_mask
    sub = adata[mask].copy()

    # Standardized metadata used downstream.
    sub.obs["_condition"] = sub.obs[condition_col].astype(str)
    sub.obs["_cell_type"] = CELL_TYPE
    sub.obs["_gse"] = sub.obs[gse_col].astype(str)

    donor_col = next((c for c in DONOR_CANDIDATES if c in sub.obs.columns), None)
    if donor_col:
        sub.obs["_patient_id"] = sub.obs[donor_col].astype(str)
    else:
        sub.obs["_patient_id"] = "unknown"

    if "assay_type" in sub.obs.columns:
        sub.obs["_assay_type"] = sub.obs["assay_type"].astype(str)
    else:
        sub.obs["_assay_type"] = "unknown"

    print(f"   Dataset column: {gse_col!r}; condition column: {condition_col!r}")
    print(f"   Subset shape: {sub.n_obs:,} cells x {sub.n_vars:,} genes")
    print(f"   State counts: {sub.obs['_condition'].value_counts().to_dict()}")
    if donor_col:
        print(f"   Donor column: {donor_col!r}; unique donors: {sub.obs['_patient_id'].nunique()}")
    return sub


def ensure_raw_counts(adata):
    if "raw_counts" not in adata.layers:
        raise ValueError('"raw_counts" layer not found.')
    vals = adata.layers["raw_counts"].data if sp.issparse(adata.layers["raw_counts"]) else adata.layers["raw_counts"]
    if not np.allclose(vals, np.round(vals), atol=1e-3):
        raise ValueError('"raw_counts" does not appear to contain integer-like raw counts.')


def add_ensembl_mapping(adata, tissue):
    mapping_df = pd.read_csv(MAP_PATHS[tissue])
    gene_map = dict(zip(mapping_df["gene_symbol"].astype(str), mapping_df["ensembl_id"].astype(str)))
    adata.var["ensembl_id"] = pd.Index(adata.var_names.astype(str)).map(gene_map)
    mapped = adata.var["ensembl_id"].notna()
    print(f"   Mapped {int(mapped.sum()):,}/{adata.n_vars:,} genes to Ensembl IDs")
    adata = adata[:, mapped].copy()
    if adata.n_vars == 0:
        raise RuntimeError("No genes mapped to Ensembl IDs.")
    return adata


def load_recurrent_genes(sig_dir, tissue, control_state, target_state, mode, cell_type):
    filename = f"{mode.upper()}_{control_state}_{target_state}_{tissue}_all_isp_results_sig_only.csv"
    path = os.path.join(sig_dir, filename)
    if not os.path.exists(path):
        print(f"   No discovery file: {path}")
        return pd.DataFrame(), path

    df = pd.read_csv(path)
    needed = {"gene_symbol", "ensembl_id", "median_cosine_shift", "cell_type"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    ct_mask = df["cell_type"].astype(str).map(normalize_celltype) == normalize_celltype(cell_type)
    out = df.loc[ct_mask].copy()
    if "significant" in out.columns:
        # Input directory is already sig_only, but keep this defensive.
        out = out[out["significant"].astype(str).str.casefold().isin(["true", "1"])]

    out = out.drop_duplicates(subset=["ensembl_id"], keep="first").reset_index(drop=True)
    out = out.rename(columns={"median_cosine_shift": "pooled_median_cosine_shift"})
    return out, path


def direction(x, eps=1e-12):
    if pd.isna(x):
        return "missing"
    if x > eps:
        return "positive"
    if x < -eps:
        return "negative"
    return "zero"


def patient_aware_sample(dataset, donor_key="patient_id", max_cells_total=500, min_cells_per_donor=5, random_state=42):
    rng = np.random.default_rng(random_state)
    n_total = len(dataset)
    if max_cells_total is None or max_cells_total >= n_total:
        return list(range(n_total)), {"all_cells": n_total}

    if donor_key not in dataset.column_names:
        idx = rng.choice(n_total, size=min(max_cells_total, n_total), replace=False).tolist()
        return idx, {"random_fallback": len(idx)}

    donor_to_indices = defaultdict(list)
    for i, d in enumerate(dataset[donor_key]):
        donor_to_indices[d].append(i)

    eligible = {d: idxs for d, idxs in donor_to_indices.items() if len(idxs) >= min_cells_per_donor}
    if not eligible:
        idx = rng.choice(n_total, size=min(max_cells_total, n_total), replace=False).tolist()
        return idx, {"random_fallback": len(idx)}

    cells_per_donor = max(min_cells_per_donor, int(np.floor(max_cells_total / len(eligible))))
    sampled = []
    report = {}
    for donor, idxs in eligible.items():
        n_take = min(cells_per_donor, len(idxs))
        chosen = rng.choice(idxs, size=n_take, replace=False).tolist()
        sampled.extend(chosen)
        report[str(donor)] = n_take

    if len(sampled) > max_cells_total:
        sampled = rng.choice(sampled, size=max_cells_total, replace=False).tolist()
    return sampled, report


def trim_sequences_safe(dataset, max_len=1024, protected_token_ids=None):
    protected_token_ids = set() if protected_token_ids is None else set(protected_token_ids)
    if len(dataset) == 0:
        return dataset
    max_orig = int(np.max(dataset["length"]))
    if max_len >= max_orig:
        return dataset

    sample_ids = dataset["input_ids"][0]
    cls_token = int(sample_ids[0])
    eos_token = int(sample_ids[-1])
    fully_protected = protected_token_ids | {cls_token, eos_token}

    def trim_example(example):
        ids = list(example["input_ids"])
        if len(ids) <= max_len:
            return example
        cls, eos = ids[0], ids[-1]
        middle = ids[1:-1]
        target_middle = max_len - 2
        keep_middle = middle[:target_middle]
        tail_middle = middle[target_middle:]
        protected_in_tail = [tid for tid in tail_middle if int(tid) in protected_token_ids]
        if protected_in_tail:
            swap_positions = [
                i for i in range(len(keep_middle) - 1, -1, -1)
                if int(keep_middle[i]) not in fully_protected
            ]
            for swap_pos, prot_tid in zip(swap_positions, protected_in_tail):
                keep_middle[swap_pos] = prot_tid
        example["input_ids"] = [cls] + keep_middle + [eos]
        example["length"] = len(example["input_ids"])
        return example

    return dataset.map(trim_example, num_proc=None)


def run_separation_test(embeddings, control_state, target_state, emb_feature_cols, donor_col=None):
    ctrl_mask = embeddings["condition"].astype(str) == str(control_state)
    tgt_mask = embeddings["condition"].astype(str) == str(target_state)
    n_ctrl, n_tgt = int(ctrl_mask.sum()), int(tgt_mask.sum())

    empty = {
        "passes": False, "p_value": 1.0, "sep_score": 0.0, "cohens_d": None,
        "balanced_accuracy": 0.5, "n_ctrl": n_ctrl, "n_tgt": n_tgt,
        "n_donors_ctrl": 0, "n_donors_tgt": 0,
    }
    if n_ctrl < MIN_CELLS_PER_STATE or n_tgt < MIN_CELLS_PER_STATE:
        return empty

    ctrl_embs = embeddings.loc[ctrl_mask, emb_feature_cols].values.astype(np.float32)
    tgt_embs = embeddings.loc[tgt_mask, emb_feature_cols].values.astype(np.float32)
    ctrl_centroid = ctrl_embs.mean(axis=0, keepdims=True)
    tgt_centroid = tgt_embs.mean(axis=0, keepdims=True)

    ctrl_margins = (
        cosine_similarity(ctrl_embs, ctrl_centroid).ravel()
        - cosine_similarity(ctrl_embs, tgt_centroid).ravel()
    )
    tgt_margins = (
        cosine_similarity(tgt_embs, tgt_centroid).ravel()
        - cosine_similarity(tgt_embs, ctrl_centroid).ravel()
    )

    _, p_ctrl = stats.mannwhitneyu(ctrl_margins, np.zeros(len(ctrl_margins)), alternative="greater")
    _, p_tgt = stats.mannwhitneyu(tgt_margins, np.zeros(len(tgt_margins)), alternative="greater")
    p_combined = max(float(p_ctrl), float(p_tgt))
    sep_score = float((ctrl_margins.mean() + tgt_margins.mean()) / 2)

    n_donors_ctrl = n_donors_tgt = 0
    cohens_d = None
    if donor_col and donor_col in embeddings.columns:
        ctrl_donors = embeddings.loc[ctrl_mask, donor_col].astype(str).values
        tgt_donors = embeddings.loc[tgt_mask, donor_col].astype(str).values
        n_donors_ctrl = len(np.unique(ctrl_donors))
        n_donors_tgt = len(np.unique(tgt_donors))
        ctrl_dm = pd.Series(ctrl_margins, index=ctrl_donors).groupby(level=0).mean()
        tgt_dm = pd.Series(tgt_margins, index=tgt_donors).groupby(level=0).mean()
        if len(ctrl_dm) >= 2 and len(tgt_dm) >= 2:
            pooled_sd = np.sqrt((ctrl_dm.std() ** 2 + tgt_dm.std() ** 2) / 2 + 1e-10)
            cohens_d = float((tgt_dm.mean() - ctrl_dm.mean()) / pooled_sd)

    X_all = np.vstack([ctrl_embs, tgt_embs])
    y_all = np.array([0] * len(ctrl_embs) + [1] * len(tgt_embs))
    ba_scores = []
    n_splits = min(3, min(n_ctrl, n_tgt) // 10)
    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_idx, test_idx in skf.split(X_all, y_all):
            clf = LogisticRegression(max_iter=500, C=0.1, random_state=42, solver="lbfgs")
            clf.fit(X_all[train_idx], y_all[train_idx])
            ba_scores.append(balanced_accuracy_score(y_all[test_idx], clf.predict(X_all[test_idx])))

    bal_acc = float(np.mean(ba_scores)) if ba_scores else 0.5
    return {
        "passes": bool((p_combined < 0.05) and (sep_score > 0)),
        "p_value": p_combined,
        "sep_score": sep_score,
        "cohens_d": cohens_d,
        "balanced_accuracy": bal_acc,
        "n_ctrl": n_ctrl,
        "n_tgt": n_tgt,
        "n_donors_ctrl": n_donors_ctrl,
        "n_donors_tgt": n_donors_tgt,
    }


def make_tokenized_dataset(adata, tissue, work_dir, tag):
    """Create a Geneformer HF dataset from a dataset/state-filtered AnnData."""
    tok_input_dir = os.path.join(work_dir, "loom")
    tok_output_dir = os.path.join(work_dir, "tokenized")
    os.makedirs(tok_input_dir, exist_ok=True)
    os.makedirs(tok_output_dir, exist_ok=True)

    X = adata.layers["raw_counts"]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    X = X.astype(np.float32, copy=False)

    ensembl_ids = adata.var["ensembl_id"].astype(str).values
    n_counts = X.sum(axis=1).astype(np.float32)
    matrix = X.T

    row_attrs = {"ensembl_id": ensembl_ids, "gene_name": ensembl_ids}
    col_attrs = {
        "CellID": np.asarray(adata.obs_names.astype(str)),
        "n_counts": n_counts,
        "_cell_type": np.asarray(adata.obs["_cell_type"].astype(str)),
        "_condition": np.asarray(adata.obs["_condition"].astype(str)),
        "_assay_type": np.asarray(adata.obs["_assay_type"].astype(str)),
        "_patient_id": np.asarray(adata.obs["_patient_id"].astype(str)),
    }

    loom_path = os.path.join(tok_input_dir, f"{tag}.loom")
    loompy.create(loom_path, matrix, row_attrs, col_attrs)

    tk.tokenize_data(
        data_directory=tok_input_dir,
        output_directory=tok_output_dir,
        output_prefix=tag,
        file_format="loom",
    )
    return os.path.join(tok_output_dir, f"{tag}.dataset")


def extract_embeddings(dataset_path, work_dir, tag):
    emb_dir = os.path.join(work_dir, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)
    embex = EmbExtractor(
        model_type="Pretrained",
        num_classes=0,
        emb_mode="cls",
        cell_emb_style="mean_pool",
        max_ncells=None,
        emb_layer=-1,
        emb_label=["cell_type", "condition", "patient_id"],
        nproc=4,
    )
    embeddings = embex.extract_embs(
        model_directory=GENEFORMER_MODEL_DIR,
        input_data_file=dataset_path,
        output_directory=emb_dir,
        output_prefix=f"embs_{tag}",
    )
    embeddings.to_parquet(os.path.join(emb_dir, f"embs_{tag}.parquet"), index=True)
    return embeddings


def get_emb_feature_cols(embeddings):
    meta = {"cell_type", "condition", "cell_id", "donor", "patient_id", "sample_id", "index"}
    return [
        c for c in embeddings.columns
        if c not in meta and pd.api.types.is_numeric_dtype(embeddings[c])
    ]


def compute_baseline_cosine(embeddings, control_state, target_state, emb_feature_cols):
    ctrl = embeddings.loc[embeddings["condition"].astype(str) == str(control_state), emb_feature_cols].values
    tgt = embeddings.loc[embeddings["condition"].astype(str) == str(target_state), emb_feature_cols].values
    if len(ctrl) == 0 or len(tgt) == 0:
        return np.nan
    ctrl_mean = ctrl.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    return float(
        np.dot(ctrl_mean / (np.linalg.norm(ctrl_mean) + 1e-12),
               tgt_mean / (np.linalg.norm(tgt_mean) + 1e-12))
    )


def safe_write_patch():
    if getattr(pu, "_safe_write_patched_for_sensitivity", False):
        return
    original_write = pu.write_perturbation_dictionary

    def safe_write(data, path):
        basename = os.path.basename(path)
        if len(basename) > 200:
            dir_part = os.path.dirname(path)
            safe_prefix = basename[:60]
            token_hash = hashlib.md5(basename.encode()).hexdigest()[:16]
            path = os.path.join(dir_part, f"{safe_prefix}_{token_hash}")
        original_write(data, path)

    pu.write_perturbation_dictionary = safe_write
    pu._safe_write_patched_for_sensitivity = True


def load_gene_result_from_pickles(isp_dir, target_state, token_id, baseline_cos_sim):
    """Return raw per-cell values and a median shift using the same baseline heuristic as v4."""
    pickle_files = [
        f for f in os.listdir(isp_dir)
        if f.endswith(".pickle") and "cell_embs_dict" in f
    ]
    merged = defaultdict(lambda: defaultdict(list))
    for fname in pickle_files:
        with open(os.path.join(isp_dir, fname), "rb") as f:
            batch_dict = pickle.load(f)
        if isinstance(batch_dict, list):
            batch_dict = batch_dict[0] if batch_dict else {}
        for state_label, inner in batch_dict.items():
            for key, val in inner.items():
                if isinstance(val, list):
                    merged[state_label][key].extend(val)
                else:
                    merged[state_label][key].append(val)

    if target_state not in merged:
        return np.array([]), np.nan, np.nan, 0

    state_dict = merged[target_state]
    candidate_keys = []
    for key in state_dict.keys():
        if isinstance(key, tuple):
            first = key[0]
            emb_key = key[1] if len(key) > 1 else None
            if emb_key == "cell_emb":
                if first == token_id or first == (token_id,) or (isinstance(first, tuple) and token_id in first):
                    candidate_keys.append(key)
        elif key == token_id:
            candidate_keys.append(key)

    # For one-gene group perturbation the key can vary by Geneformer version.
    # If there is only one cell-embedding key, it is unambiguous.
    if not candidate_keys:
        cell_keys = [k for k in state_dict if isinstance(k, tuple) and len(k) > 1 and k[1] == "cell_emb"]
        if len(cell_keys) == 1:
            candidate_keys = cell_keys

    vals = []
    for key in candidate_keys:
        vals.extend(state_dict[key] if isinstance(state_dict[key], list) else [state_dict[key]])

    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return arr, np.nan, np.nan, 0

    # Preserve the discovery script's compatibility heuristic. Some Geneformer
    # versions return already-centered small shifts; others return cosine sims.
    effective_baseline = baseline_cos_sim
    if abs(float(np.median(arr[: min(10, len(arr))]))) < 0.1:
        effective_baseline = 0.0

    shifts = arr - effective_baseline
    return shifts, float(np.median(shifts)), float(np.mean(shifts)), len(shifts)


def run_one_gene_isp(
    full_dataset,
    embeddings,
    emb_feature_cols,
    control_state,
    target_state,
    mode,
    gene_symbol,
    ensembl_id,
    token_id,
    experiment_dir,
):
    gene_tag = safe_name(f"{gene_symbol}_{ensembl_id}")
    gene_dir = os.path.join(experiment_dir, "isp", mode, gene_tag)
    os.makedirs(gene_dir, exist_ok=True)

    control_ds = full_dataset.filter(
        lambda x: str(x["condition"]) == str(control_state), num_proc=None
    )
    if len(control_ds) == 0:
        return {"status": "no_control_cells", "n_sampled_control": 0, "n_gene_cells": 0}

    donor_key = next((c for c in DONOR_CANDIDATES if c in control_ds.column_names), None)
    sampled_idx, donor_report = patient_aware_sample(
        control_ds,
        donor_key=donor_key or "patient_id",
        max_cells_total=MAX_ISP_CELLS_TOTAL,
        min_cells_per_donor=MIN_CELLS_PER_DONOR_ISP,
    )
    sampled = control_ds.select(sampled_idx)

    # Critical for matching discovery behavior: in the original genes_to_perturb='all'
    # run, a gene is only perturbed in cells where that token is represented. Apply
    # the same criterion explicitly now that we run one named gene at a time.
    gene_ds = sampled.filter(
        lambda x: int(token_id) in {int(t) for t in x["input_ids"]}, num_proc=None
    )
    n_gene_cells = len(gene_ds)

    pd.DataFrame(
        list(donor_report.items()), columns=["donor", "n_cells_sampled"]
    ).to_csv(os.path.join(gene_dir, "patient_sampling.csv"), index=False)

    if n_gene_cells < MIN_CELLS_PER_GENE:
        return {
            "status": "too_few_gene_cells",
            "n_sampled_control": len(sampled),
            "n_gene_cells": n_gene_cells,
        }

    gene_ds = trim_sequences_safe(
        gene_ds, max_len=1024, protected_token_ids={int(token_id)}
    ).flatten_indices()

    baseline = compute_baseline_cosine(
        embeddings, control_state, target_state, emb_feature_cols
    )

    ctrl_embs = embeddings.loc[
        embeddings["condition"].astype(str) == str(control_state), emb_feature_cols
    ].values
    tgt_embs = embeddings.loc[
        embeddings["condition"].astype(str) == str(target_state), emb_feature_cols
    ].values
    state_embs_dict = {
        str(control_state): torch.tensor(ctrl_embs.mean(axis=0), dtype=torch.float32).to(DEVICE),
        str(target_state): torch.tensor(tgt_embs.mean(axis=0), dtype=torch.float32).to(DEVICE),
    }

    tmp_dir = tempfile.mkdtemp(prefix="geneformer_sensitivity_")
    try:
        input_path = os.path.join(tmp_dir, "isp_input.dataset")
        gene_ds.save_to_disk(input_path)

        # A one-element list is deliberately used. Geneformer treats a list with
        # multiple genes as a JOINT perturbation; one gene per call yields individual effects.
        isp = InSilicoPerturber(
            perturb_type=MODE_TO_PERTURB[mode],
            perturb_rank_shift=None,
            genes_to_perturb=[str(ensembl_id)],
            combos=0,
            anchor_gene=None,
            model_type="Pretrained",
            num_classes=0,
            emb_mode="cls",
            cell_emb_style="mean_pool",
            filter_data=None,
            cell_states_to_model={
                "state_key": "condition",
                "start_state": str(control_state),
                "goal_state": str(target_state),
                "alt_states": [],
            },
            state_embs_dict=state_embs_dict,
            max_ncells=None,
            emb_layer=-1,
            forward_batch_size=FORWARD_BATCH_SIZE,
            nproc=1,
        )

        # IMPORTANT: Hugging Face Dataset.map/filter with num_proc=1 still
        # launches a worker subprocess. Geneformer stores CUDA tensors in
        # state_embs_dict, so that worker attempts to deserialize CUDA state
        # after a fork and crashes with:
        #   "Cannot re-initialize CUDA in forked subprocess".
        # The constructor requires an integer nproc, so initialize with 1 and
        # then switch the internal Dataset operations to true single-process
        # mode by setting nproc=None before perturb_data(). GPU forward passes
        # are unaffected and still run on the assigned H100.
        isp.nproc = None
        isp.perturb_data(
            model_directory=GENEFORMER_MODEL_DIR,
            input_data_file=input_path,
            output_directory=gene_dir,
            output_prefix=f"isp_{safe_name(gene_symbol)}",
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        torch.cuda.empty_cache()

    shifts, med, mean, n = load_gene_result_from_pickles(
        gene_dir, str(target_state), int(token_id), baseline
    )
    if n == 0:
        return {
            "status": "no_isp_output",
            "n_sampled_control": len(sampled),
            "n_gene_cells": n_gene_cells,
            "baseline_cos_sim": baseline,
        }

    # Descriptive one-sample signed-rank p-value is included only as a secondary
    # diagnostic. It is NOT the original discovery null and is not used to call hits.
    try:
        if np.allclose(shifts, 0):
            p_vs_zero = 1.0
        else:
            p_vs_zero = float(stats.wilcoxon(shifts, alternative="two-sided").pvalue)
    except Exception:
        p_vs_zero = np.nan

    return {
        "status": "ok",
        "n_sampled_control": len(sampled),
        "n_gene_cells": n_gene_cells,
        "n_shift_values": n,
        "baseline_cos_sim": baseline,
        "dataset_median_cosine_shift": med,
        "dataset_mean_cosine_shift": mean,
        "dataset_std_cosine_shift": float(np.std(shifts)),
        "p_vs_zero_descriptive": p_vs_zero,
    }


# =============================================================================
# Load requested comparisons and initialize tokenizer
# =============================================================================
comparisons = pd.read_csv(COMPARISONS_CSV)
required_cols = {"cell_type", "tissue", "GSE_to_test", "control_state", "target_state"}
missing = required_cols - set(comparisons.columns)
if missing:
    raise ValueError(f"Comparisons CSV missing required columns: {sorted(missing)}")

comparisons = comparisons[
    comparisons["cell_type"].astype(str).map(normalize_celltype) == normalize_celltype(CELL_TYPE)
].copy()
comparisons = comparisons.drop_duplicates(
    subset=["tissue", "GSE_to_test", "control_state", "target_state"]
).reset_index(drop=True)

if comparisons.empty:
    raise RuntimeError(f"No rows in {COMPARISONS_CSV} matched cell_type={CELL_TYPE!r}")

bad_tissues = sorted(set(comparisons["tissue"].astype(str).str.lower()) - set(H5AD_DIRS))
if bad_tissues:
    raise ValueError(f"Unsupported tissue value(s) in comparisons CSV: {bad_tissues}")

print("\nComparisons to run:")
print(comparisons.to_string(index=False))

os.makedirs(OUTPUT_ROOT, exist_ok=True)
celltype_root = os.path.join(OUTPUT_ROOT, safe_name(CELL_TYPE))
os.makedirs(celltype_root, exist_ok=True)

# One tokenizer reused for all experiments.
tk = TranscriptomeTokenizer(
    custom_attr_name_dict={
        "_cell_type": "cell_type",
        "_condition": "condition",
        "_assay_type": "assay_type",
        "_patient_id": "patient_id",
    },
    nproc=4,
)
safe_write_patch()

all_rows = []
experiment_status = []


# =============================================================================
# Main comparison loop
# =============================================================================
for comp_idx, comp in comparisons.iterrows():
    tissue = str(comp["tissue"]).lower()
    gse = str(comp["GSE_to_test"])
    control_state = str(comp["control_state"])
    target_state = str(comp["target_state"])

    experiment_tag = safe_name(f"{tissue}__{gse}__{control_state}_to_{target_state}")
    experiment_dir = os.path.join(celltype_root, experiment_tag)
    os.makedirs(experiment_dir, exist_ok=True)
    results_csv = os.path.join(experiment_dir, "sensitivity_results.csv")

    print("\n" + "=" * 90)
    print(f"[{comp_idx + 1}/{len(comparisons)}] {CELL_TYPE} | {tissue} | {gse} | {control_state} -> {target_state}")
    print("=" * 90)

    if os.path.exists(results_csv) and not args.force:
        old = pd.read_csv(results_csv)

        # Do not mistake a crashed/partial ISP run for a completed experiment.
        # Previous versions caught infrastructure failures per gene and wrote
        # status=error rows, so those experiments must be rerun automatically.
        has_errors = (
            "status" in old.columns
            and old["status"].astype(str).str.lower().eq("error").any()
        )
        if has_errors:
            print(f"   Existing results contain error rows; rerunning experiment: {results_csv}")
        else:
            print(f"   Existing valid results found, loading instead of rerunning: {results_csv}")
            all_rows.extend(old.to_dict("records"))
            experiment_status.append({
                "cell_type": CELL_TYPE, "tissue": tissue, "GSE_to_test": gse,
                "control_state": control_state, "target_state": target_state,
                "status": "loaded_existing",
            })
            continue

    # Determine recurrent genes BEFORE doing expensive tokenization.
    recurrent_by_mode = {}
    for mode in ["down", "up"]:
        rec, source_path = load_recurrent_genes(
            SIG_DIR, tissue, control_state, target_state, mode, CELL_TYPE
        )
        recurrent_by_mode[mode] = rec
        print(f"   {mode.upper()}: {len(rec)} recurrent discovery genes from {os.path.basename(source_path)}")

    if all(df.empty for df in recurrent_by_mode.values()):
        print("   No recurrent genes for either mode; skipping experiment.")
        experiment_status.append({
            "cell_type": CELL_TYPE, "tissue": tissue, "GSE_to_test": gse,
            "control_state": control_state, "target_state": target_state,
            "status": "no_recurrent_genes",
        })
        continue

    # Load only this cell-type h5ad, then restrict to requested dataset and states.
    h5ad_path = find_celltype_h5ad(tissue, CELL_TYPE)
    print(f"   Loading h5ad: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    if adata.var_names.duplicated().any():
        adata.var_names_make_unique()
    ensure_raw_counts(adata)
    adata = subset_to_gse_and_states(adata, gse, control_state, target_state)

    n_ctrl_raw = int((adata.obs["_condition"] == control_state).sum())
    n_tgt_raw = int((adata.obs["_condition"] == target_state).sum())
    if n_ctrl_raw == 0 or n_tgt_raw == 0:
        print("   Missing one requested state after GSE filtering; skipping.")
        experiment_status.append({
            "cell_type": CELL_TYPE, "tissue": tissue, "GSE_to_test": gse,
            "control_state": control_state, "target_state": target_state,
            "status": "missing_state", "n_control": n_ctrl_raw, "n_target": n_tgt_raw,
        })
        continue

    adata = add_ensembl_mapping(adata, tissue)

    # Tokenize/embedding extraction once for this GSE + direction, shared by DOWN and UP.
    work_dir = os.path.join(experiment_dir, "prepared")
    os.makedirs(work_dir, exist_ok=True)
    dataset_path = make_tokenized_dataset(adata, tissue, work_dir, experiment_tag)
    full_dataset = hf_datasets.load_from_disk(dataset_path)

    embeddings = extract_embeddings(dataset_path, work_dir, experiment_tag)
    emb_feature_cols = get_emb_feature_cols(embeddings)
    donor_emb_col = next((c for c in DONOR_CANDIDATES if c in embeddings.columns), None)

    sep = run_separation_test(
        embeddings, control_state, target_state, emb_feature_cols, donor_emb_col
    )
    pd.DataFrame([{**sep, "control_state": control_state, "target_state": target_state}]).to_csv(
        os.path.join(experiment_dir, "separation_test.csv"), index=False
    )
    print(
        f"   Separation: passes={sep['passes']} sep={sep['sep_score']:.6g} "
        f"p={sep['p_value']:.3g} n=({sep['n_ctrl']},{sep['n_tgt']})"
    )

    if (not sep["passes"]) and (not args.skip_separation_gate):
        print("   Separation gate failed; skipping ISP for this dataset-specific comparison.")
        # Still write rows for all planned genes so the missing sensitivity result is explicit.
        failed_rows = []
        for mode, rec_df in recurrent_by_mode.items():
            for r in rec_df.itertuples(index=False):
                failed_rows.append({
                    "cell_type": CELL_TYPE,
                    "tissue": tissue,
                    "GSE_to_test": gse,
                    "control_state": control_state,
                    "target_state": target_state,
                    "perturb_mode": mode,
                    "gene_symbol": r.gene_symbol,
                    "ensembl_id": r.ensembl_id,
                    "pooled_median_cosine_shift": r.pooled_median_cosine_shift,
                    "pooled_direction": direction(r.pooled_median_cosine_shift),
                    "dataset_median_cosine_shift": np.nan,
                    "dataset_direction": "missing",
                    "direction_consistent": np.nan,
                    "status": "separation_failed",
                    "separation_passes": False,
                    "separation_score": sep["sep_score"],
                    "separation_p_value": sep["p_value"],
                    "n_control_state": sep["n_ctrl"],
                    "n_target_state": sep["n_tgt"],
                })
        failed_df = pd.DataFrame(failed_rows)
        failed_df.to_csv(results_csv, index=False)
        all_rows.extend(failed_rows)
        experiment_status.append({
            "cell_type": CELL_TYPE, "tissue": tissue, "GSE_to_test": gse,
            "control_state": control_state, "target_state": target_state,
            "status": "separation_failed",
        })
        del adata, embeddings, full_dataset
        torch.cuda.empty_cache()
        continue

    # Map recurrent genes to Geneformer tokens. Discovery CSV already supplies Ensembl IDs.
    experiment_rows = []
    for mode in ["down", "up"]:
        rec_df = recurrent_by_mode[mode]
        if rec_df.empty:
            continue

        print(f"\n   Running {mode.upper()} ISP for {len(rec_df)} recurrent genes...")
        for gene_i, r in enumerate(rec_df.itertuples(index=False), start=1):
            gene_symbol = str(r.gene_symbol)
            ensembl_id = str(r.ensembl_id)
            pooled_shift = float(r.pooled_median_cosine_shift)
            token_id = tk.gene_token_dict.get(ensembl_id)

            print(f"      [{gene_i}/{len(rec_df)}] {gene_symbol} ({ensembl_id})")

            base_row = {
                "cell_type": CELL_TYPE,
                "tissue": tissue,
                "GSE_to_test": gse,
                "control_state": control_state,
                "target_state": target_state,
                "perturb_mode": mode,
                "gene_symbol": gene_symbol,
                "ensembl_id": ensembl_id,
                "pooled_median_cosine_shift": pooled_shift,
                "pooled_direction": direction(pooled_shift),
                "separation_passes": bool(sep["passes"]),
                "separation_score": sep["sep_score"],
                "separation_p_value": sep["p_value"],
                "separation_balanced_accuracy": sep["balanced_accuracy"],
                "n_control_state": sep["n_ctrl"],
                "n_target_state": sep["n_tgt"],
                "n_donors_control": sep["n_donors_ctrl"],
                "n_donors_target": sep["n_donors_tgt"],
            }

            if token_id is None:
                result = {"status": "not_in_geneformer_token_dict", "n_gene_cells": 0}
            else:
                try:
                    result = run_one_gene_isp(
                        full_dataset=full_dataset,
                        embeddings=embeddings,
                        emb_feature_cols=emb_feature_cols,
                        control_state=control_state,
                        target_state=target_state,
                        mode=mode,
                        gene_symbol=gene_symbol,
                        ensembl_id=ensembl_id,
                        token_id=int(token_id),
                        experiment_dir=experiment_dir,
                    )
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    print(f"         ERROR: {msg}")

                    # Infrastructure-level multiprocessing/CUDA failures should
                    # stop the job immediately rather than producing hundreds of
                    # per-gene errors and incorrectly marking the experiment done.
                    fatal_markers = (
                        "Cannot re-initialize CUDA in forked subprocess",
                        "subprocesses has abruptly died during map operation",
                    )
                    if any(marker in msg for marker in fatal_markers):
                        raise RuntimeError(
                            "Fatal Geneformer multiprocessing/CUDA failure during ISP. "
                            "Dataset preprocessing inside ISP should be running with "
                            "multiprocessing disabled."
                        ) from e

                    result = {
                        "status": "error",
                        "error": msg,
                    }

            row = {**base_row, **result}
            ds_shift = row.get("dataset_median_cosine_shift", np.nan)
            row["dataset_direction"] = direction(ds_shift)
            if pd.notna(ds_shift) and direction(ds_shift) != "zero" and direction(pooled_shift) != "zero":
                row["direction_consistent"] = bool(direction(ds_shift) == direction(pooled_shift))
            else:
                row["direction_consistent"] = np.nan
            if pd.notna(ds_shift):
                row["absolute_shift_difference"] = abs(float(ds_shift) - pooled_shift)
                row["shift_ratio_dataset_to_pooled"] = (
                    float(ds_shift) / pooled_shift if abs(pooled_shift) > 1e-12 else np.nan
                )
            else:
                row["absolute_shift_difference"] = np.nan
                row["shift_ratio_dataset_to_pooled"] = np.nan

            experiment_rows.append(row)

            # Incremental checkpointing after every gene.
            pd.DataFrame(experiment_rows).to_csv(results_csv, index=False)

    exp_df = pd.DataFrame(experiment_rows)
    exp_df.to_csv(results_csv, index=False)
    all_rows.extend(experiment_rows)
    experiment_status.append({
        "cell_type": CELL_TYPE, "tissue": tissue, "GSE_to_test": gse,
        "control_state": control_state, "target_state": target_state,
        "status": "completed", "n_rows": len(exp_df),
        "n_direction_consistent": int((exp_df["direction_consistent"] == True).sum()) if len(exp_df) else 0,
        "n_direction_inconsistent": int((exp_df["direction_consistent"] == False).sum()) if len(exp_df) else 0,
    })

    del adata, embeddings, full_dataset
    torch.cuda.empty_cache()


# =============================================================================
# Combined sensitivity outputs
# =============================================================================
combined_path = os.path.join(celltype_root, "sensitivity_summary_all_experiments.csv")
status_path = os.path.join(celltype_root, "experiment_status.csv")

combined = pd.DataFrame(all_rows)
if not combined.empty:
    sort_cols = [
        c for c in ["tissue", "GSE_to_test", "control_state", "target_state", "perturb_mode", "gene_symbol"]
        if c in combined.columns
    ]
    combined = combined.sort_values(sort_cols).reset_index(drop=True)
    combined.to_csv(combined_path, index=False)

    valid = combined[combined["direction_consistent"].isin([True, False])].copy()
    if not valid.empty:
        small = (
            valid.groupby(
                ["cell_type", "tissue", "GSE_to_test", "control_state", "target_state", "perturb_mode"],
                dropna=False,
            )
            .agg(
                n_genes_tested=("gene_symbol", "count"),
                n_direction_consistent=("direction_consistent", "sum"),
            )
            .reset_index()
        )
        small["n_direction_inconsistent"] = small["n_genes_tested"] - small["n_direction_consistent"]
        small["proportion_direction_consistent"] = (
            small["n_direction_consistent"] / small["n_genes_tested"]
        )
        small.to_csv(os.path.join(celltype_root, "sensitivity_small_table.csv"), index=False)

pd.DataFrame(experiment_status).to_csv(status_path, index=False)

print("\n" + "=" * 90)
print("DONE")
print(f"Combined gene-level results: {combined_path}")
print(f"Experiment status:           {status_path}")
small_path = os.path.join(celltype_root, "sensitivity_small_table.csv")
if os.path.exists(small_path):
    print(f"Small sensitivity table:     {small_path}")
print("=" * 90)

