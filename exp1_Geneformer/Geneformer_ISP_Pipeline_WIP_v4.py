# Geneformer In-Silico Perturbation (ISP) Pipeline
### Based on Liu et al. — *Evaluating Foundation Models for In-Silico Perturbation*


# **Pipeline overview:**
# 1. Install dependencies (done so skipped)
# 2. Load and QC `.h5ad` files
# 3–13. **Per-cell-type loop** (gene mapping → tokenize → embeddings → separation test → ISP → stats → figures → enrichment)
# 14. Export final results

# Configure data, model, and gene-set paths for your environment before running.

# Setup
import os
import sys
import json
if '/path/to/Geneformer' not in sys.path:
    sys.path.insert(0, '/path/to/Geneformer')
import scanpy as sc

import argparse
_parser = argparse.ArgumentParser(description='Geneformer ISP Pipeline')
_parser.add_argument('dataset', choices=['liver', 'immune'],
                     help="Dataset to run on: 'liver' or 'immune'")
_parser.add_argument('--control', default='Healthy',
                     help="Control condition label (default: 'Healthy')")
_parser.add_argument('--target', default='MASH',
                     help="Target condition label (default: 'MASH')")
_parser.add_argument('--celltype',
                     help="Cell Type")
_parser.add_argument('--mode', choices=['down', 'up'], default='down',
                     help="Perturbation mode: 'down' or 'up' (default: 'down')")
_parser.add_argument('--validation', choices=['Y', 'N'], default='N',
                     help="Validation Mode: 'Y' or 'N' (default: 'N'")
_args = _parser.parse_args()

dataset_choice = _args.dataset.lower()
if dataset_choice == 'liver':
    H5AD_PATH_DIR = '/path/to/liver/adata/'
elif dataset_choice == 'immune':
    H5AD_PATH_DIR = '/path/to/immune/adata/'

CAND_GENE_PATH = '/path/to/candidate_gene.csv'

validation_mode = _args.validation
if validation_mode == 'Y':
    print("RUNNING VALIDATION")
    if dataset_choice == 'liver':
        H5AD_PATH_DIR = '/path/to/valid_liver'
        CAND_GENE_PATH = '/path/to/Liver_Master_Gene_List.csv'
    if dataset_choice == 'immune':
        H5AD_PATH_DIR = '/path/to/valid_immune'
        CAND_GENE_PATH = '/path/to/Immune_Healthy_MASLD_Master_Gene_List.csv'


CONDITION_COL  = 'broad_condition'  # obs column that holds Healthy/MASH labels
CELLTYPE_COL   = 'cell_type'        # obs column storing cell-type label (used to infer name from file)
CONTROL_LABEL  = _args.control#'Healthy'
TARGET_LABEL   = _args.target#'MASH'
ALT_STATES = []
FOCUS_CELLTYPES = [_args.celltype] if _args.celltype else []


DONOR_COL = 'patient_id'

MAX_ISP_CELLS_TOTAL     = 500
MIN_CELLS_PER_DONOR_ISP = 5

SAVED_EMBEDDINGS_PATH = None

GENE_ID_TYPE   = 'symbol'
SPECIES        = 'human'
PERTURB_MODE   = _args.mode
N_CANDIDATE_GENES = 200
MIN_CELLS_PER_STATE = 100
OUTPUT_DIR = f'/path/to/validation_output/geneformer/{dataset_choice}/{CONTROL_LABEL}_{TARGET_LABEL}/{PERTURB_MODE}'
os.makedirs(OUTPUT_DIR, exist_ok=True)

GENEFORMER_MODEL_DIR = "/path/to/Geneformer"

print(' Configuration set.')
print(f'   H5AD directory  : {H5AD_PATH_DIR}')
print(f'   Control        : {CONTROL_LABEL}')
print(f'   Target         : {TARGET_LABEL}')
print(f'   Perturb        : {PERTURB_MODE}-regulation')
print(f'   Donor col      : {DONOR_COL}')
print(f'   Saved embeddings: {SAVED_EMBEDDINGS_PATH}')


# Load and QC .h5ad files
import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import glob
import warnings
warnings.filterwarnings('ignore')
sc.settings.verbosity = 1


def detect_condition_col(adata):
    candidates = [CONDITION_COL] if CONDITION_COL else []
    candidates += ['broad_condition', 'condition', 'disease', 'Status']
    for c in candidates:
        if c and c in adata.obs.columns:
            print(f'   Auto-detected condition column: "{c}"')
            print(f'   Values: {adata.obs[c].unique().tolist()}')
            return c
    return None


def infer_cell_type_from_adata(adata, path):
    """Infer the cell-type name from the adata.

    Priority:
      1. CELLTYPE_COL obs column — take the unique value (there should be exactly one).
      2. Stem of the filename (e.g. 'Memory_B_cells.h5ad' → 'Memory B cells').
    """
    if CELLTYPE_COL and CELLTYPE_COL in adata.obs.columns:
        unique_cts = adata.obs[CELLTYPE_COL].unique().tolist()
        if len(unique_cts) == 1:
            return str(unique_cts[0])
        elif len(unique_cts) > 1:
            print(f'   WARNING: {len(unique_cts)} unique values in "{CELLTYPE_COL}" — '
                  f'using filename stem as cell-type name.')
    # Fallback: filename stem, underscores → spaces
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.replace('_', ' ')


def load_and_qc(path):
    print(f'\n Loading: {path}')
    adata = sc.read_h5ad(path)
    print(f'   Shape  : {adata.shape[0]:,} cells × {adata.shape[1]:,} genes')
    print(f'   obs    : {list(adata.obs.columns)}')

    if adata.var_names.duplicated().any():
        print('     Duplicate gene names detected — making unique.')
        adata.var_names_make_unique()

    # ── Raw counts layer ─────────────────────────────────────────────────
    if 'raw_counts' not in adata.layers:
        # Exit if raw_counts layer is missing, because some downstream code expects it
        print('    ERROR: "raw_counts" layer not found in adata. Please check the dataset.')
        raise ValueError('"raw_counts" layer not found in adata.')

    # Check the raw_counts layer is truly raw (non-log-transformed)
    vals = adata.layers['raw_counts'].data if sp.issparse(adata.layers['raw_counts']) else adata.layers['raw_counts']
    if np.allclose(vals, np.round(vals), atol=1e-3):
            print('raw_counts appears to be raw counts')
    else:
        print('    WARNING: "raw_counts" layer does not appear to be raw counts (may be log-transformed).')
        raise ValueError('"raw_counts" layer does not appear to be raw counts (may be log-transformed). Please check the dataset.')

    # ── Condition label (_condition) ─────────────────────────────────────
    detected = detect_condition_col(adata)
    if detected:
        adata.obs['_condition'] = adata.obs[detected].astype(str)
    else:
        print('    Could not detect condition column. Throwing error')
        raise ValueError('Condition column not found. Please check the dataset and CONDITION_COL setting.')
    
    print(f'   Conditions: {adata.obs["_condition"].value_counts().to_dict()}')

    # ── Cell-type label (_cell_type) ─────────────────────────────────────
    cell_type = infer_cell_type_from_adata(adata, path)
    adata.obs['_cell_type'] = cell_type
    print(f'   Cell type: "{cell_type}"')

    return adata, cell_type


# ── Discover all .h5ad files in the directory ─────────────────────────────────
h5ad_files = sorted(glob.glob(os.path.join(H5AD_PATH_DIR, '*.h5ad')))
if not h5ad_files:
    raise FileNotFoundError(f'No .h5ad files found in: {H5AD_PATH_DIR}')
print(f'Found {len(h5ad_files)} .h5ad file(s) in {H5AD_PATH_DIR}:')
for p in h5ad_files:
    print(f'  {os.path.basename(p)}')

# ── Optional: restrict to FOCUS_CELLTYPES ────────────────────────────────────
# Filtering is done by FILENAME STEM before any file is opened, so that
# non-matching cell types are never read into memory at all. This relies on
# filenames matching cell-type names directly (e.g. "Hepatocytes.h5ad" ->
# "Hepatocytes"), which is the same fallback infer_cell_type_from_adata()
# uses internally. If a file's filename-derived name doesn't match but its
# CELLTYPE_COL obs value would have, that file is still skipped here —
# acceptable tradeoff since avoiding the full read is the whole point.
if FOCUS_CELLTYPES:
    _focus_set = set(FOCUS_CELLTYPES)
    _prefiltered_files = []
    for p in h5ad_files:
        stem = os.path.splitext(os.path.basename(p))[0].replace('_', ' ')
        if stem in _focus_set:
            _prefiltered_files.append(p)
        else:
            print(f'   Skipping "{os.path.basename(p)}" (filename stem "{stem}" not in FOCUS_CELLTYPES) — not opening file.')
    h5ad_files = _prefiltered_files
    if not h5ad_files:
        raise RuntimeError(
            f'No .h5ad filenames matched FOCUS_CELLTYPES={FOCUS_CELLTYPES} by filename stem. '
            f'Check that --celltype matches the filename (underscores -> spaces), e.g. "Hepatocytes.h5ad" -> "Hepatocytes".'
        )
    print(f'\n Pre-filtered to {len(h5ad_files)} file(s) matching FOCUS_CELLTYPES before loading: '
          f'{[os.path.basename(p) for p in h5ad_files]}')

# ── Load every (already filtered) file into adata_by_celltype dict ─────────────
adata_by_celltype = {}   # {cell_type_name: adata}
for path in h5ad_files:
    try:
        adata_ct, cell_type = load_and_qc(path)
    except Exception as e:
        print(f'  ERROR loading {path}: {e}')
        continue

    # Re-check against FOCUS_CELLTYPES using the obs-column-derived name too,
    # in case it differs from the filename stem used for pre-filtering above.
    if FOCUS_CELLTYPES and cell_type not in FOCUS_CELLTYPES:
        print(f'   Skipping "{cell_type}" (not in FOCUS_CELLTYPES after loading).')
        del adata_ct
        continue

    if cell_type in adata_by_celltype:
        print(f'   WARNING: duplicate cell type "{cell_type}" — second file will overwrite the first!')
    adata_by_celltype[cell_type] = adata_ct

if not adata_by_celltype:
    raise RuntimeError('No cell types were loaded. Check H5AD_PATH_DIR and FOCUS_CELLTYPES.')
print(f'\n Loaded {len(adata_by_celltype)} cell type(s): {list(adata_by_celltype.keys())}')


# Per-cell-type loop (gene mapping → tokenize → embeddings → separation test → ISP → stats → figures → enrichment)

# ════════════════════════════════════════════════════════════════════════════
# SHARED IMPORTS & ONE-TIME SETUP
# (run once before the per-cell-type loop)
# ════════════════════════════════════════════════════════════════════════════

import sys, os, time, hashlib, shutil, tempfile, subprocess, pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import loompy
import datasets as hf_datasets
import mygene
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
from collections import defaultdict
from scipy import stats
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import gseapy as gp

warnings.filterwarnings('ignore')
matplotlib.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8,
})

if '/content/Geneformer' not in sys.path:
    sys.path.insert(0, '/content/Geneformer')

from geneformer import TranscriptomeTokenizer, EmbExtractor, InSilicoPerturber
import geneformer.perturber_utils as pu

mg = mygene.MyGeneInfo()

# GPU
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['TOKENIZERS_PARALLELISM']   = 'false'
    print(f' GPU: {torch.cuda.get_device_name(0)}')
else:
    # Exit because it will take a very long time to run on CPU and may cause memory issues
    print('  No GPU — ISP will be slow. Exiting.')
    sys.exit(1)

# Patch pu.load_model for GPU (once per session)
if not getattr(pu, '_load_model_gpu_patched', False):
    _REAL_LOAD_MODEL = pu.load_model
    def _gpu_load_model(model_type, num_classes, model_directory, *args, **kwargs):
        model = _REAL_LOAD_MODEL(model_type, num_classes, model_directory, *args, **kwargs)
        if torch.cuda.is_available():
            model = model.to('cuda:0').eval()
            print(f'   Model device: {next(model.parameters()).device}')
        return model
    pu.load_model = _gpu_load_model
    pu._load_model_gpu_patched = True
    print(' pu.load_model patched for GPU.')

# Constants
MODE_MAP        = {'down': 'delete', 'delete': 'delete', 'up': 'overexpress'}
gf_perturb_type = MODE_MAP.get(PERTURB_MODE, 'delete')
MIN_CELLS_PER_GENE    = 20
EFFECT_SIZE_THRESHOLD = 2.5e-4
N_BOOTSTRAP_SPLITS    = 10
SPLIT_FRACTION        = 0.5
MIN_CELLS_HALF        = 5
_alt_states           = ALT_STATES if 'ALT_STATES' in dir() and ALT_STATES else []
states_to_run         = [TARGET_LABEL] + list(_alt_states)

# Colour scheme
SIG_GREEN         = '#2ecc71'
FAIL_RED          = '#e74c3c'
SIG_COLOR         = '#C0392B'
NSIG_COLOR        = '#BDC3C7'
ALT_COLOR         = '#2980B9'
PAN_DISEASE_COLOR = '#8E44AD'
GOAL_LIGHT        = '#E8A49C'
ALT_LIGHT         = '#A9C4D9'
AMBER             = '#F5A623'

# Accumulators (filled during the loop, consumed in Step 14)
all_results            = []   # list of scored DataFrames
all_sep_rows           = []   # list of separation test summary rows
separation_results_all = {}   # {cell_type: {comp_name: result_dict}}
passing_celltypes      = []   # cell types that passed all sep tests
embeddings_all         = []   # list of per-cell-type embedding DataFrames

print(' Shared setup complete.')

# ════════════════════════════════════════════════════════════════════════════
# MASH / CANDIDATE GENE LIST  (updated with genes from Elison et al. preprint supplementary table 8 and Hong et al. 2025 supplementary table 11)
# ════════════════════════════════════════════════════════════════════════════
import pandas as pd
df = pd.read_csv(CAND_GENE_PATH)
MASH_GENES = df['Gene'].tolist()
_seen = set()
_deduped = []
for g in MASH_GENES:
    if g not in _seen:
        _deduped.append(g)
        _seen.add(g)
MASH_GENES = _deduped
print(f' {len(MASH_GENES)} unique MASH candidate genes loaded.')

# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS  (separation test, ISP scoring, etc.)
# Defined once here; called inside the per-cell-type loop below.
# ════════════════════════════════════════════════════════════════════════════

primary_comp = f'{CONTROL_LABEL} vs {TARGET_LABEL}'
state_pairs  = [(CONTROL_LABEL, TARGET_LABEL, primary_comp)]
for alt_state in _alt_states:
    state_pairs.append((CONTROL_LABEL, alt_state, f'{CONTROL_LABEL} vs {alt_state}'))
    state_pairs.append((TARGET_LABEL,  alt_state, f'{TARGET_LABEL} vs {alt_state}'))


def run_separation_test_publishable(
        embeddings_df, cell_type, control_label, target_label,
        emb_cols=None, donor_col='donor'):
    ct_mask   = embeddings_df['cell_type'] == cell_type
    ctrl_mask = ct_mask & (embeddings_df['condition'] == control_label)
    tgt_mask  = ct_mask & (embeddings_df['condition'] == target_label)
    n_ctrl, n_tgt = ctrl_mask.sum(), tgt_mask.sum()
    empty = {'passes': False, 'p_value': 1.0, 'sep_score': 0.0,
             'cohens_d': None, 'balanced_accuracy': 0.5,
             'n_ctrl': n_ctrl, 'n_tgt': n_tgt,
             'n_donors_ctrl': 0, 'n_donors_tgt': 0,
             'ctrl_margins': np.array([]), 'tgt_margins': np.array([])}
    if n_ctrl < MIN_CELLS_PER_STATE or n_tgt < MIN_CELLS_PER_STATE:
        print(f'   {cell_type} [{control_label} vs {target_label}]: '
              f'insufficient cells (ctrl={n_ctrl}, tgt={n_tgt}). Skipping.')
        return empty
    if emb_cols is None:
        meta = {'cell_type', 'condition', 'donor', 'cell_id', 'index'}
        emb_cols = [c for c in embeddings_df.columns
                    if c not in meta and pd.api.types.is_numeric_dtype(embeddings_df[c])]
    ctrl_embs = embeddings_df.loc[ctrl_mask, emb_cols].values.astype(np.float32)
    tgt_embs  = embeddings_df.loc[tgt_mask,  emb_cols].values.astype(np.float32)
    ctrl_centroid = ctrl_embs.mean(axis=0, keepdims=True)
    tgt_centroid  = tgt_embs.mean(axis=0,  keepdims=True)
    ctrl_margins = (cosine_similarity(ctrl_embs, ctrl_centroid).flatten() -
                    cosine_similarity(ctrl_embs, tgt_centroid).flatten())
    tgt_margins  = (cosine_similarity(tgt_embs,  tgt_centroid).flatten() -
                    cosine_similarity(tgt_embs,  ctrl_centroid).flatten())
    _, p_ctrl = stats.mannwhitneyu(ctrl_margins, np.zeros(len(ctrl_margins)), alternative='greater')
    _, p_tgt  = stats.mannwhitneyu(tgt_margins,  np.zeros(len(tgt_margins)),  alternative='greater')
    sep_score  = (ctrl_margins.mean() + tgt_margins.mean()) / 2
    p_combined = max(p_ctrl, p_tgt)
    n_donors_ctrl = n_donors_tgt = 0
    cohens_d = None
    actual_donor_col = None
    for dc in ([donor_col] if donor_col else []) + ['donor', 'Donor', 'patient_id']:
        if dc and dc in embeddings_df.columns:
            actual_donor_col = dc
            break
    if actual_donor_col:
        ctrl_donors = embeddings_df.loc[ctrl_mask, actual_donor_col].values
        tgt_donors  = embeddings_df.loc[tgt_mask,  actual_donor_col].values
        n_donors_ctrl = len(np.unique(ctrl_donors))
        n_donors_tgt  = len(np.unique(tgt_donors))
        ctrl_donor_means = pd.Series(ctrl_margins, index=ctrl_donors).groupby(level=0).mean()
        tgt_donor_means  = pd.Series(tgt_margins,  index=tgt_donors).groupby(level=0).mean()
        if len(ctrl_donor_means) >= 2 and len(tgt_donor_means) >= 2:
            pooled_sd = np.sqrt((ctrl_donor_means.std()**2 + tgt_donor_means.std()**2) / 2 + 1e-10)
            cohens_d  = float((tgt_donor_means.mean() - ctrl_donor_means.mean()) / pooled_sd)
    X_all = np.vstack([ctrl_embs, tgt_embs])
    y_all = np.array([0]*len(ctrl_embs) + [1]*len(tgt_embs))
    ba_scores = []
    n_splits  = min(3, min(n_ctrl, n_tgt) // 10)
    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_idx, test_idx in skf.split(X_all, y_all):
            clf = LogisticRegression(max_iter=500, C=0.1, random_state=42, solver='lbfgs')
            clf.fit(X_all[train_idx], y_all[train_idx])
            ba_scores.append(balanced_accuracy_score(y_all[test_idx], clf.predict(X_all[test_idx])))
    balanced_accuracy = float(np.mean(ba_scores)) if ba_scores else 0.5
    passes = (p_combined < 0.05) and (sep_score > 0)
    return {
        'passes': passes, 'p_value': float(p_combined), 'sep_score': float(sep_score),
        'cohens_d': cohens_d, 'balanced_accuracy': balanced_accuracy,
        'n_ctrl': int(n_ctrl), 'n_tgt': int(n_tgt),
        'n_donors_ctrl': int(n_donors_ctrl), 'n_donors_tgt': int(n_donors_tgt),
        'ctrl_margins': ctrl_margins, 'tgt_margins': tgt_margins,
    }


def all_comparisons_pass(ct_results):
    for label_a, label_b, comp_name in state_pairs:
        res = ct_results.get(comp_name, {})
        if res.get('n_ctrl', 0) == 0 or res.get('n_tgt', 0) == 0:
            return False
        if not res.get('passes', False):
            return False
    return True


def patient_aware_sample_for_isp(
        dataset, donor_key='donor',
        max_cells_total=None, min_cells_per_donor=5, random_state=42):
    """Donor-stratified sampling.

    When max_cells_total is None or >= len(dataset) all cells are returned
    directly without any sampling — this is the correct behaviour when the
    caller wants to use every available cell.
    """
    rng = np.random.default_rng(random_state)
    n_total = len(dataset)

    # None means no limit — return all cells immediately
    if max_cells_total is None or max_cells_total >= n_total:
        print(f'   Patient-aware sampling: returning all {n_total} cells (no limit).')
        return list(range(n_total)), {'all_cells': n_total}

    if donor_key not in dataset.column_names:
        n = min(max_cells_total, n_total)
        return rng.choice(n_total, size=n, replace=False).tolist(), {'all_cells': n}

    donors = dataset[donor_key]
    donor_to_indices = {}
    for i, d in enumerate(donors):
        donor_to_indices.setdefault(d, []).append(i)
    eligible = {d: idxs for d, idxs in donor_to_indices.items()
                if len(idxs) >= min_cells_per_donor}
    if not eligible:
        n = min(max_cells_total, n_total)
        print(f'     No donors with ≥{min_cells_per_donor} cells — random sampling ({n} cells).')
        return rng.choice(n_total, size=n, replace=False).tolist(), {'all_cells': n}

    n_donors = len(eligible)
    cells_per_donor = max(min_cells_per_donor, int(np.floor(max_cells_total / n_donors)))
    sampled, report = [], {}
    for donor, idxs in eligible.items():
        n_take = min(cells_per_donor, len(idxs))
        chosen = rng.choice(idxs, size=n_take, replace=False).tolist()
        sampled.extend(chosen)
        report[donor] = n_take

    # Only downsample further if we overshot the budget
    if len(sampled) > max_cells_total:
        sampled = rng.choice(sampled, size=max_cells_total, replace=False).tolist()

    print(f'   Patient-aware sampling: {len(sampled)} cells from {n_donors} donors '
          f'(~{cells_per_donor} cells/donor).')
    return sampled, report


def trim_sequences_safe(dataset, max_len=1024, protected_token_ids=None):
    if protected_token_ids is None:
        protected_token_ids = set()
    lengths  = dataset['length']
    max_orig = int(np.max(lengths))
    if max_len >= max_orig:
        return dataset
    sample_ids    = dataset['input_ids'][0]
    cls_token     = int(sample_ids[0])
    eos_token     = int(sample_ids[-1])
    fully_protected = protected_token_ids | {cls_token, eos_token}
    def trim_example(example):
        ids = list(example['input_ids'])
        if len(ids) <= max_len:
            return example
        cls    = ids[0]
        eos    = ids[-1]
        middle = ids[1:-1]
        target_middle = max_len - 2
        if len(middle) <= target_middle:
            return example
        keep_middle = middle[:target_middle]
        tail_middle = middle[target_middle:]
        protected_in_tail = [tid for tid in tail_middle if int(tid) in protected_token_ids]
        if protected_in_tail:
            swap_positions = [i for i in range(len(keep_middle)-1, -1, -1)
                              if int(keep_middle[i]) not in fully_protected]
            for swap_pos, prot_tid in zip(swap_positions, protected_in_tail):
                keep_middle[swap_pos] = prot_tid
        example['input_ids'] = [cls] + keep_middle + [eos]
        example['length']    = len(example['input_ids'])
        return example
    return dataset.map(trim_example, num_proc=1)


def score_one_state(cell_type, state_label, candidate_df, candidate_token_ids_set,
                    merged_by_state, embeddings, emb_feature_cols):
    if state_label not in merged_by_state:
        print(f'   [{state_label}] Not in pickles — skipping.')
        return None
    state_dict = merged_by_state[state_label]
    ctrl_mask  = ((embeddings['cell_type'] == cell_type) &
                  (embeddings['condition'] == CONTROL_LABEL))
    state_mask = ((embeddings['cell_type'] == cell_type) &
                  (embeddings['condition'] == state_label))
    ctrl_mean  = embeddings.loc[ctrl_mask, emb_feature_cols].values.mean(axis=0)
    state_mean = (embeddings.loc[state_mask, emb_feature_cols].values.mean(axis=0)
                  if state_mask.sum() > 0 else ctrl_mean)
    ctrl_norm  = ctrl_mean  / (np.linalg.norm(ctrl_mean)  + 1e-12)
    state_norm = state_mean / (np.linalg.norm(state_mean) + 1e-12)
    baseline_cos_sim = float(np.dot(ctrl_norm, state_norm))
    _sample_key = next(((tid, ek) for (tid, ek) in state_dict.keys() if ek == 'cell_emb'), None)
    if _sample_key:
        _raw = np.array(state_dict[_sample_key][:10], dtype=np.float64)
        _raw = _raw[np.isfinite(_raw)]
        if len(_raw) > 0 and abs(float(np.median(_raw))) < 0.1:
            baseline_cos_sim = 0.0
    token_to_ensembl = {v: k for k, v in tk.gene_token_dict.items()}
    all_gene_shifts  = {}
    all_gene_n_cells = {}
    rng_score = np.random.default_rng(42)
    for (token_id, emb_key), cos_sims in state_dict.items():
        if emb_key != 'cell_emb':
            continue
        arr = np.array(cos_sims, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        all_gene_n_cells[token_id] = len(arr)
        if len(arr) >= MIN_CELLS_PER_GENE:
            all_gene_shifts[token_id] = arr - baseline_cos_sim
    candidate_gene_shifts = {tid: s for tid, s in all_gene_shifts.items() if tid in candidate_token_ids_set}
    other_gene_shifts     = {tid: s for tid, s in all_gene_shifts.items() if tid not in candidate_token_ids_set}
    if not candidate_gene_shifts and not candidate_token_ids_set:
        return None
    if other_gene_shifts:
        other_pool = np.concatenate(list(other_gene_shifts.values()))
        other_pool = other_pool[np.isfinite(other_pool)]
    else:
        other_pool = (np.concatenate(list(candidate_gene_shifts.values()))
                      if candidate_gene_shifts else np.zeros(10))
        other_pool = other_pool[np.isfinite(other_pool)]
    rows = []
    for token_id in candidate_token_ids_set:
        ensembl_id  = token_to_ensembl.get(token_id, str(token_id))
        match       = candidate_df.loc[candidate_df['ensembl_id'] == ensembl_id, 'gene_symbol']
        gene_symbol = match.values[0] if len(match) > 0 else ensembl_id
        n_cells     = all_gene_n_cells.get(token_id, 0)
        if token_id in candidate_gene_shifts:
            shifts_a = candidate_gene_shifts[token_id]
            n_a = len(shifts_a)
            other_for_this = np.concatenate([v for tid, v in other_gene_shifts.items()
                                              if tid != token_id]) if other_gene_shifts else other_pool
            other_for_this = other_for_this[np.isfinite(other_for_this)]
            if len(other_for_this) >= n_a:
                sample_b = rng_score.choice(other_for_this, size=n_a, replace=False)
            elif len(other_for_this) > 0:
                sample_b = rng_score.choice(other_for_this, size=n_a, replace=True)
            else:
                sample_b = np.zeros(n_a)
            if np.median(sample_b) < 0:
                sample_b = np.maximum(sample_b, 0.0)
            try:
                _, pval = stats.ranksums(shifts_a, sample_b)
            except Exception:
                pval = 1.0
            if not np.isfinite(pval):
                pval = 1.0
            median_shift = float(np.nanmedian(shifts_a))
            mean_shift   = float(np.nanmean(shifts_a))
            std_shift    = float(np.nanstd(shifts_a))
        else:
            pval = 1.0
            median_shift = mean_shift = std_shift = 0.0
        rows.append({
            'gene_symbol': gene_symbol, 'ensembl_id': ensembl_id,
            'median_cosine_shift': median_shift, 'mean_cosine_shift': mean_shift,
            'std_cosine_shift': std_shift,
            'median_cos_sim': float(median_shift + baseline_cos_sim),
            'mean_cos_sim':   float(mean_shift   + baseline_cos_sim),
            'n_cells': n_cells, 'pval_raw': float(pval),
            'cell_type': cell_type, 'control_state': CONTROL_LABEL,
            'target_state': state_label, 'perturb_mode': PERTURB_MODE,
            'baseline_cos_sim': baseline_cos_sim,
        })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df['median_cosine_shift'] = df['median_cosine_shift'].fillna(0.0)
    df['mean_cosine_shift']   = df['mean_cosine_shift'].fillna(0.0)
    df['pval_raw']            = df['pval_raw'].fillna(1.0)
    _, pval_adj, _, _ = multipletests(df['pval_raw'], alpha=0.05, method='fdr_bh')
    df['pval_adj'] = pval_adj
    if gf_perturb_type == 'delete':
        df['significant'] = ((df['pval_adj'] < 0.05) &
                             (df['median_cosine_shift'].abs() > EFFECT_SIZE_THRESHOLD) &
                             #(df['median_cosine_shift'] < 0) &
                             (df['n_cells'] >= MIN_CELLS_PER_GENE))
    else:
        df['significant'] = ((df['pval_adj'] < 0.05) &
                             (df['median_cosine_shift'].abs() > EFFECT_SIZE_THRESHOLD) &
                             #(df['median_cosine_shift'] > 0) &
                             (df['n_cells'] >= MIN_CELLS_PER_GENE))
    df.loc[df['n_cells'] < MIN_CELLS_PER_GENE, 'significant'] = False
    ascending = (gf_perturb_type == 'delete')
    df = df.sort_values('median_cosine_shift', ascending=ascending).reset_index(drop=True)
    return df


print(' Helper functions defined.')

# Output sub-directories (created once)
TOKENIZED_DIR     = os.path.join(OUTPUT_DIR, 'tokenized')
EMB_DIR           = os.path.join(OUTPUT_DIR, 'embeddings')
SEP_FIG_DIR       = os.path.join(OUTPUT_DIR, 'separation_test_figures')
ISP_OUTPUT_DIR    = os.path.join(OUTPUT_DIR, 'isp_output')
STATS_OUTPUT_DIR  = os.path.join(OUTPUT_DIR, 'isp_stats')
ISP_FIG_DIR       = os.path.join(OUTPUT_DIR, 'isp_figures')
for d in [TOKENIZED_DIR, EMB_DIR, SEP_FIG_DIR, ISP_OUTPUT_DIR, STATS_OUTPUT_DIR, ISP_FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# Tokenizer (built once, reused across cell types)
tk = TranscriptomeTokenizer(
    custom_attr_name_dict={'_cell_type': 'cell_type', '_condition': 'condition',
                           '_assay_type': 'assay_type', '_patient_id': 'patient_id'},
    nproc=4
)

# ISP write helper
original_write = pu.write_perturbation_dictionary
def safe_write(data, path):
    basename = os.path.basename(path)
    if len(basename) > 200:
        dir_part    = os.path.dirname(path)
        safe_prefix = basename[:60]
        token_hash  = hashlib.md5(basename.encode()).hexdigest()[:16]
        path        = os.path.join(dir_part, f'{safe_prefix}_{token_hash}')
    original_write(data, path)
pu.write_perturbation_dictionary = safe_write

# PatchedISP class (identical to v4)
class PatchedISP(InSilicoPerturber):
    """Thin wrapper around InSilicoPerturber.

    All cell/gene filtering is done by the caller before handing off
    `input_data_file`, so this class just ensures `genes_to_perturb='all'`
    and delegates directly to the base class via an unambiguous explicit call.

    Crucially we do NOT override `perturb_data` with `super().perturb_data()`
    because in Python's MRO that resolves back to this subclass and causes
    infinite recursion.  Instead we call `InSilicoPerturber.perturb_data`
    (the concrete base implementation) explicitly.
    """
    def __init__(self, **kwargs):
        kwargs['genes_to_perturb'] = 'all'
        super().__init__(**kwargs)

# ── MAIN LOOP ────────────────────────────────────────────────────────────────
for cell_type, adata_ct in adata_by_celltype.items():
    safe_ct = cell_type.replace(' ', '_').replace('/', '_')
    print(f'\n{"═"*65}')
 
    # Skip if this cell type is already "resolved":
    #   (a) ISP completed -> isp_figures/<ct>/isp_goal_<ct>.pdf exists, OR
    #   (b) separation test already failed -> separation_test_<ct>.csv exists
    #       and shows at least one comparison did not pass (ISP was
    #       deliberately never run, so no .pdf will ever appear)
    expected_isp_fig = os.path.join(ISP_FIG_DIR, safe_ct, f'isp_goal_{safe_ct}.pdf')
    sep_csv_path      = os.path.join(OUTPUT_DIR, safe_ct, f'separation_test_{safe_ct}.csv')
 
    if os.path.exists(expected_isp_fig):
        print(f'     Results for {cell_type} already exist — skipping.')
        continue
 
    if os.path.exists(sep_csv_path):
        try:
            _prior_sep_df = pd.read_csv(sep_csv_path)
            if not _prior_sep_df.empty and not _prior_sep_df['passes'].all():
                print(f'     {cell_type} already failed separation test (see '
                      f'{sep_csv_path}) — skipping re-run.')
                continue
        except Exception as e:
            print(f'    Could not read existing {sep_csv_path}: {e} — will re-run separation test.')
    print(f'CELL TYPE: {cell_type}  ({adata_ct.n_obs:,} cells)')
    print(f'{"═"*65}')

    ct_out = os.path.join(OUTPUT_DIR, safe_ct)
    os.makedirs(ct_out, exist_ok=True)

    # ── Step 4: Gene mapping ─────────────────────────────────────────────
    print(f'\n[{cell_type}] Step 4: Gene mapping...')

    gene_names = adata_ct.var_names.tolist()

    if dataset_choice == "liver":
        MAP_PATH = "/path/to/gene_symbol_to_ensembl_liver.csv"
    elif dataset_choice == "immune":
        MAP_PATH = "/path/to/gene_symbol_to_ensembl_immune.csv"
    else:
        raise ValueError(f"Unknown dataset_choice: {dataset_choice}")

    mapping_df = pd.read_csv(MAP_PATH)
    gene_map = dict(zip(mapping_df["gene_symbol"].astype(str), mapping_df["ensembl_id"].astype(str)))

    adata_ct.var["ensembl_id"] = adata_ct.var_names.astype(str).map(gene_map)


    n_mapped = adata_ct.var['ensembl_id'].notna().sum()
    print(f'   Mapped {n_mapped:,}/{len(gene_names):,} genes to Ensembl IDs.')
    adata_ct = adata_ct[:, adata_ct.var['ensembl_id'].notna()].copy()
    if adata_ct.n_vars == 0:
        print(f'     No genes mapped for {cell_type} — skipping.')
        continue

    # ── Step 5: Tokenize ─────────────────────────────────────────────────
    print(f'\n[{cell_type}] Step 5: Tokenizing...')
    ct_tok_dir  = os.path.join(TOKENIZED_DIR, safe_ct)
    ct_tok_data = os.path.join(OUTPUT_DIR, 'tokenized_data', safe_ct)
    os.makedirs(ct_tok_dir, exist_ok=True)
    os.makedirs(ct_tok_data, exist_ok=True)

    X = adata_ct.layers['raw_counts']
    if sp.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)
    X = X.astype(np.float32, copy=False)

    # X = adata_ct.layers['raw_counts'].copy()
    # if sp.issparse(X):
    #     X = X.toarray()
    # X = X.astype(np.float32)
    ensembl_ids = adata_ct.var['ensembl_id'].values.astype(str)
    n_counts    = X.sum(axis=1).astype(np.float32)
    matrix      = X.T
    row_attrs = {'ensembl_id': ensembl_ids, 'gene_name': ensembl_ids}
    _assay_arr   = (adata_ct.obs['assay_type'].values.tolist()
                    if 'assay_type' in adata_ct.obs.columns
                    else ['unknown'] * adata_ct.n_obs)
    _patient_arr = (adata_ct.obs['patient_id'].values.tolist()
                    if 'patient_id' in adata_ct.obs.columns
                    else ['unknown'] * adata_ct.n_obs)
    col_attrs = {
        'CellID'     : np.array(adata_ct.obs_names.tolist()),
        'n_counts'   : n_counts,
        '_cell_type' : np.array(adata_ct.obs['_cell_type'].values.tolist()),
        '_condition' : np.array(adata_ct.obs['_condition'].values.tolist()),
        '_assay_type': np.array(_assay_arr),
        '_patient_id': np.array(_patient_arr),
    }
    loom_path = os.path.join(ct_tok_dir, f'{safe_ct}.loom')
    loompy.create(loom_path, matrix, row_attrs, col_attrs)
    print(f'   Loom saved: {loom_path}')

    tk.tokenize_data(
        data_directory=ct_tok_dir,
        output_directory=ct_tok_data,
        output_prefix=f'geneformer_{safe_ct}',
        file_format='loom'
    )
    DATASET_PATH_CT = os.path.join(ct_tok_data, f'geneformer_{safe_ct}.dataset')
    print(f'   Tokenized dataset: {DATASET_PATH_CT}')

    # Sanitise cell type names in token dataset
    tok_dataset = hf_datasets.load_from_disk(DATASET_PATH_CT)
    tok_dataset = tok_dataset.map(
        lambda ex: {**ex, 'cell_type': ex['cell_type'].replace('/', '_')}, num_proc=1
    ).flatten_indices()

    # Fix: Save to a new temporary path to avoid PermissionError
    temp_dataset_path = os.path.join(ct_tok_data, f'geneformer_{safe_ct}_temp.dataset')
    tok_dataset.save_to_disk(temp_dataset_path)
    # Update DATASET_PATH_CT to the new temporary path
    DATASET_PATH_CT = temp_dataset_path
    print(f'   Sanitized dataset saved to: {DATASET_PATH_CT}')

    # ── Step 6: Extract embeddings ────────────────────────────────────────
    print(f'\n[{cell_type}] Step 6: Extracting embeddings...')
    ct_emb_dir  = os.path.join(EMB_DIR, safe_ct)
    os.makedirs(ct_emb_dir, exist_ok=True)

    embex = EmbExtractor(
        model_type='Pretrained', num_classes=0, emb_mode='cls',
        cell_emb_style='mean_pool',
        # filter_data={'cell_type': [safe_ct]}, # Removed redundant 'cell_type' filter
        max_ncells=None, emb_layer=-1,
        emb_label=['cell_type', 'condition', 'patient_id'], nproc=4,
    )
    embeddings = embex.extract_embs(
        model_directory=GENEFORMER_MODEL_DIR,
        input_data_file=DATASET_PATH_CT,
        output_directory=ct_emb_dir,
        output_prefix=f'embs_{safe_ct}',
    )
    emb_save_path = os.path.join(ct_emb_dir, f'embs_{safe_ct}.parquet')
    embeddings.to_parquet(emb_save_path, index=True)
    embeddings_all.append(embeddings)
    print(f'   Embeddings: {embeddings.shape}')

    emb_meta_cols    = {'cell_type', 'condition', 'cell_id', 'donor', 'patient_id'}
    emb_feature_cols = [c for c in embeddings.columns
                        if c not in emb_meta_cols
                        and pd.api.types.is_numeric_dtype(embeddings[c])]

    _donor_col_in_emb = None
    for _dc in ['donor', 'Donor', 'donor_id', 'patient_id', 'sample_id']:
        if _dc and _dc in embeddings.columns:
            _donor_col_in_emb = _dc
            break

    # ── Step 7: Separation test ───────────────────────────────────────────
    print(f'\n[{cell_type}] Step 7: Separation test...')
    ct_sep_results = {}

    for label_a, label_b, comp_name in state_pairs:
        has_a = ((embeddings['cell_type'] == cell_type) &
                 (embeddings['condition'] == label_a)).sum() >= MIN_CELLS_PER_STATE
        has_b = ((embeddings['cell_type'] == cell_type) &
                 (embeddings['condition'] == label_b)).sum() >= MIN_CELLS_PER_STATE
        if not has_a or not has_b:
            print(f'   {cell_type} [{comp_name}]: missing condition data — skipping.')
            ct_sep_results[comp_name] = {
                'passes': False, 'p_value': 1.0, 'sep_score': 0.0,
                'cohens_d': None, 'balanced_accuracy': 0.5,
                'n_ctrl': 0, 'n_tgt': 0, 'n_donors_ctrl': 0, 'n_donors_tgt': 0,
                'ctrl_margins': np.array([]), 'tgt_margins': np.array([])
            }
            continue
        res = run_separation_test_publishable(
            embeddings, cell_type, label_a, label_b,
            emb_cols=emb_feature_cols, donor_col=_donor_col_in_emb
        )
        ct_sep_results[comp_name] = res
        d_str  = f"{res['cohens_d']:.3f}" if res['cohens_d'] is not None else 'N/A'
        status = ' PASS' if res['passes'] else 'FAIL'
        print(f'   [{comp_name}]: {status}  sep={res["sep_score"]:.4f}  '
              f'p={res["p_value"]:.2e}  d={d_str}  '
              f'bal_acc={res["balanced_accuracy"]:.3f}  '
              f'n=({res["n_ctrl"]},{res["n_tgt"]})')
        all_sep_rows.append({
            'cell_type': cell_type, 'comparison': comp_name,
            'label_a': label_a, 'label_b': label_b,
            'passes': res['passes'],
            'sep_score': round(res['sep_score'], 5),
            'p_value': res['p_value'],
            'cohens_d': round(res['cohens_d'], 3) if res['cohens_d'] is not None else None,
            'balanced_accuracy': round(res['balanced_accuracy'], 3),
            'n_a': res['n_ctrl'], 'n_b': res['n_tgt'],
            'n_donors_a': res['n_donors_ctrl'], 'n_donors_b': res['n_donors_tgt'],
        })

    separation_results_all[cell_type] = ct_sep_results

    # Save per-cell-type separation results
    ct_sep_df = pd.DataFrame([
        r for r in all_sep_rows if r['cell_type'] == cell_type
    ])
    ct_sep_df.to_csv(os.path.join(ct_out, f'separation_test_{safe_ct}.csv'), index=False)

    # ── Separation test gate ──────────────────────────────────────────────
    ct_passes_sep = all_comparisons_pass(ct_sep_results)
    if ct_passes_sep:
        passing_celltypes.append(cell_type)
        print(f'\n    {cell_type} passes all separation tests → proceeding to ISP.')
    else:
        print(f'\n    {cell_type} FAILED separation test → skipping ISP.')
        print('      (Results will still appear in cross-cell-type summary plots.)')
        continue   # ← skip Steps 8–13 for this cell type

    # ── Step 8: Candidate gene selection ─────────────────────────────────
    print(f'\n[{cell_type}] Step 8: Building candidate gene list...')
    sym_to_ensembl = dict(zip(adata_ct.var_names, adata_ct.var['ensembl_id']))

    import pickle as _pkl
    vocab_path = '/content/Geneformer/geneformer/gene_median_dictionary.pkl'
    gf_vocab = None
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            gf_vocab = set(_pkl.load(f).keys())

    rows_cand = []
    for symbol in MASH_GENES:
        eid = sym_to_ensembl.get(symbol)
        if eid is None or (isinstance(eid, float) and np.isnan(eid)):
            continue
        if gf_vocab and eid not in gf_vocab:
            continue
        token_id = tk.gene_token_dict.get(eid)
        if token_id is None:
            continue
        rows_cand.append({'gene_symbol': symbol, 'ensembl_id': eid, 'token_id': token_id})

    candidate_df = pd.DataFrame(rows_cand)
    print(f'   {len(candidate_df)} candidate genes mapped.')
    candidate_df.to_csv(os.path.join(ct_out, f'candidate_genes_{safe_ct}.csv'), index=False)

    if len(candidate_df) == 0:
        print(f'     No candidate genes for {cell_type} — skipping ISP.')
        continue

    candidate_token_ids = [int(row.token_id) for row in candidate_df.itertuples()]

    # ── Step 9: In-Silico Perturbation ────────────────────────────────────
    print(f'\n[{cell_type}] Step 9: Running ISP...')
    ct_isp_dir = os.path.join(ISP_OUTPUT_DIR, safe_ct)
    os.makedirs(ct_isp_dir, exist_ok=True)

    # Load & filter tokenised dataset to this cell type
    full_dataset    = hf_datasets.load_from_disk(DATASET_PATH_CT)
    ct_cond_dataset = full_dataset.filter(
        lambda x: x['condition'] == CONTROL_LABEL,
        num_proc=1
    )
    print(f'   {len(ct_cond_dataset)} control cells.')

    _donor_key_isp = None
    for _dk in ['donor', 'Donor', 'donor_id', 'patient_id', 'sample_id']:
        if _dk and _dk in ct_cond_dataset.column_names:
            _donor_key_isp = _dk
            break

    # Patient-aware sampling
    if _donor_key_isp and len(ct_cond_dataset) > 0:
        sampled_idx, donor_report = patient_aware_sample_for_isp(
            ct_cond_dataset, donor_key=_donor_key_isp,
            max_cells_total=MAX_ISP_CELLS_TOTAL, min_cells_per_donor=MIN_CELLS_PER_DONOR_ISP,
        )
        pd.DataFrame(list(donor_report.items()), columns=['donor', 'n_cells']).to_csv(
            os.path.join(ct_isp_dir, f'patient_sampling_{safe_ct}.csv'), index=False
        )
    else:
        rng_isp = np.random.default_rng(42)
        n_take  = len(ct_cond_dataset) if MAX_ISP_CELLS_TOTAL is None else min(MAX_ISP_CELLS_TOTAL, len(ct_cond_dataset))
        sampled_idx  = list(range(n_take)) if n_take == len(ct_cond_dataset) else rng_isp.choice(len(ct_cond_dataset), size=n_take, replace=False).tolist()
        donor_report = {'random_fallback': n_take}

    isp_input_dataset = ct_cond_dataset.select(sampled_idx)
    candidate_set_int = set(int(t) for t in candidate_token_ids)
    isp_input_dataset = isp_input_dataset.filter(
        lambda x: any(int(t) in candidate_set_int for t in x['input_ids']), num_proc=1
    )
    print(f'   {len(isp_input_dataset)} cells after candidate filter.')

    if len(isp_input_dataset) == 0:
        print(f'     No cells with candidate genes — skipping ISP for {cell_type}.')
        continue

    isp_input_dataset = trim_sequences_safe(
        isp_input_dataset, max_len=1024, protected_token_ids=candidate_set_int
    ).flatten_indices()

    isp_tmp_dir    = tempfile.mkdtemp(prefix='isp_gpu_')
    isp_input_path = os.path.join(isp_tmp_dir, 'isp_input.dataset')
    isp_input_dataset.save_to_disk(isp_input_path)

    # State embeddings
    ctrl_mask = ((embeddings['cell_type'] == cell_type) &
                 (embeddings['condition'] == CONTROL_LABEL))
    tgt_mask  = ((embeddings['cell_type'] == cell_type) &
                 (embeddings['condition'] == TARGET_LABEL))
    control_embs = embeddings.loc[ctrl_mask, emb_feature_cols].values
    target_embs  = embeddings.loc[tgt_mask,  emb_feature_cols].values
    state_embs_dict = {
        CONTROL_LABEL: torch.tensor(control_embs.mean(axis=0), dtype=torch.float32).to(DEVICE),
        TARGET_LABEL:  torch.tensor(target_embs.mean(axis=0),  dtype=torch.float32).to(DEVICE),
    }
    for alt_state in _alt_states:
        alt_mask = ((embeddings['cell_type'] == cell_type) &
                    (embeddings['condition'] == alt_state))
        if alt_mask.sum() > 0:
            state_embs_dict[alt_state] = torch.tensor(
                embeddings.loc[alt_mask, emb_feature_cols].values.mean(axis=0),
                dtype=torch.float32
            ).to(DEVICE)

    try:
        # Input dataset is already pre-filtered (control cells only,
        # candidate-gene-containing cells only, trimmed to 1024 tokens).
        # filter_data=None avoids a second internal pass that would try to
        # match 'cell_type' strings and drop all cells.
        # InSilicoPerturber.perturb_data is called directly (no override)
        # to avoid the super() → self recursion that occurred in v4.
        isp = PatchedISP(
            perturb_type=gf_perturb_type, perturb_rank_shift=None,
            combos=0, anchor_gene=None,
            model_type='Pretrained', num_classes=0,
            emb_mode='cls', cell_emb_style='mean_pool',
            filter_data=None,
            cell_states_to_model={
                'state_key':   'condition',
                'start_state': CONTROL_LABEL,
                'goal_state':  TARGET_LABEL,
                'alt_states':  _alt_states
            },
            state_embs_dict=state_embs_dict,
            max_ncells=None, emb_layer=-1,
            forward_batch_size=64, nproc=1,
        )
        InSilicoPerturber.perturb_data(
            isp,
            model_directory=GENEFORMER_MODEL_DIR,
            input_data_file=isp_input_path,
            output_directory=ct_isp_dir,
            output_prefix=f'isp_{safe_ct}'
        )
        print(f'    ISP complete for {cell_type}.')
    finally:
        shutil.rmtree(isp_tmp_dir, ignore_errors=True)
        torch.cuda.empty_cache()

    # ── Step 10: Cosine shift & stats ─────────────────────────────────────
    print(f'\n[{cell_type}] Step 10: Computing cosine shifts...')
    ct_stats_dir = os.path.join(STATS_OUTPUT_DIR, safe_ct)
    os.makedirs(ct_stats_dir, exist_ok=True)

    pickle_files = [
        f for f in os.listdir(ct_isp_dir)
        if f.endswith('.pickle') and 'dict_cell_embs_' in f
    ]
    print(f'   {len(pickle_files)} pickle files found.')

    if not pickle_files:
        print('   No pickle files — skipping stats for this cell type.')
        continue

    merged_by_state = defaultdict(lambda: defaultdict(list))
    for fname in pickle_files:
        try:
            with open(os.path.join(ct_isp_dir, fname), 'rb') as f:
                batch_dict = pickle.load(f)
            if isinstance(batch_dict, list):
                batch_dict = batch_dict[0] if batch_dict else {}
            for sl, inner_dict in batch_dict.items():
                for key, val in inner_dict.items():
                    if isinstance(val, list):
                        merged_by_state[sl][key].extend(val)
                    else:
                        merged_by_state[sl][key].append(val)
        except Exception as e:
            print(f'   Failed to load {fname}: {e}')

    candidate_token_ids_set = {
        tk.gene_token_dict[eid]
        for eid in candidate_df['ensembl_id'].dropna()
        if eid in tk.gene_token_dict
    }

    for state_label in states_to_run:
        df_state = score_one_state(
            cell_type, state_label, candidate_df,
            candidate_token_ids_set, merged_by_state,
            embeddings, emb_feature_cols
        )
        if df_state is None:
            continue
        # Override cell_type to use the display name
        df_state['cell_type'] = cell_type
        safe_state = state_label.replace(' ', '_')
        out_csv    = os.path.join(ct_stats_dir, f'stats_{safe_ct}_{safe_state}.csv')
        df_state.to_csv(out_csv, index=False)
        all_results.append(df_state)
        n_sig = df_state['significant'].sum()
        print(f'   [{state_label}] {len(df_state)} genes scored, {n_sig} significant.')

    # ── Step 11: Rank & summarize (per-cell-type printout) ────────────────
    print(f'\n[{cell_type}] Step 11: Summary...')
    for state_label in states_to_run:
        ct_state_results = [df for df in all_results
                            if df['cell_type'].iloc[0] == cell_type
                            and df['target_state'].iloc[0] == state_label]
        if not ct_state_results:
            continue
        ct_df_state = ct_state_results[-1]
        ct_sig = ct_df_state[ct_df_state['significant']]
        gene_col = 'gene_symbol' if 'gene_symbol' in ct_df_state.columns else 'ensembl_id'
        print(f'   [{state_label}] {len(ct_sig)} significant hits:')
        if len(ct_sig) > 0:
            print(ct_sig[[gene_col, 'median_cosine_shift', 'pval_adj', 'n_cells']].head(10).to_string())

    # ── Step 12: Per-cell-type figures ────────────────────────────────────
    print(f'\n[{cell_type}] Step 12: Saving per-cell-type figures...')
    ct_fig_dir = os.path.join(ISP_FIG_DIR, safe_ct)
    os.makedirs(ct_fig_dir, exist_ok=True)

    # Only use results for this cell type
    ct_all_results = [df for df in all_results if df['cell_type'].iloc[0] == cell_type]
    if not ct_all_results:
        print(f'   No results to plot for {cell_type}.')
        continue

    results_df_ct = pd.concat(ct_all_results, ignore_index=True)
    shift_col = 'median_cosine_shift'
    padj_col  = 'pval_adj'
    gene_col  = 'gene_symbol' if 'gene_symbol' in results_df_ct.columns else 'ensembl_id'

    goal_df_ct = results_df_ct[results_df_ct['target_state'] == TARGET_LABEL].copy()

    # Volcano + bar + distribution for this cell type
    if not goal_df_ct.empty:
        goal_df_ct['-log10_padj'] = -np.log10(goal_df_ct[padj_col].clip(lower=1e-300))
        n_sig = goal_df_ct['significant'].sum()
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [2, 1.5, 1.5]})
        fig.suptitle(
            f'ISP: {cell_type}\n{CONTROL_LABEL} → {TARGET_LABEL}  |  '
            f'Mode: {PERTURB_MODE}  |  n={n_sig} significant',
            fontsize=12, fontweight='bold', y=1.02
        )
        # Panel A: Volcano
        ax = axes[0]
        colors_map = goal_df_ct['significant'].map({True: SIG_COLOR, False: NSIG_COLOR})
        ax.scatter(goal_df_ct[shift_col], goal_df_ct['-log10_padj'],
                   c=colors_map, alpha=0.7, s=20, linewidths=0, rasterized=True)
        ax.axvline(0, color='black', lw=0.8, ls='--', alpha=0.6)
        ax.axhline(-np.log10(0.05), color='#7F8C8D', lw=0.8, ls=':', alpha=0.7)
        top_sig = goal_df_ct[goal_df_ct['significant']].nsmallest(8, shift_col)
        for _, row in top_sig.iterrows():
            ax.annotate(str(row.get(gene_col, '')),
                        xy=(row[shift_col], row['-log10_padj']),
                        xytext=(4, 4), textcoords='offset points', fontsize=7.5,
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        ax.set_xlabel('Median Cosine Shift', fontsize=11)
        ax.set_ylabel('-log₁₀(FDR p-value)', fontsize=11)
        ax.set_title('A. Volcano', fontsize=11, fontweight='bold', loc='left')
        ax.legend(handles=[
            mpatches.Patch(color=SIG_COLOR, label=f'Significant (n={n_sig})'),
            mpatches.Patch(color=NSIG_COLOR, label='Not significant'),
        ], fontsize=9)
        # Panel B: Ranked bar
        ax2 = axes[1]
        top20 = goal_df_ct.nsmallest(20, shift_col).copy()
        bar_colors = [SIG_COLOR if s else NSIG_COLOR for s in top20['significant']]
        labels = top20[gene_col].tolist()
        ax2.barh(range(len(top20)), top20[shift_col].values[::-1],
                 color=bar_colors[::-1], edgecolor='none', height=0.7)
        ax2.set_yticks(range(len(top20)))
        ax2.set_yticklabels(labels[::-1], fontsize=9)
        ax2.axvline(0, color='black', lw=0.8)
        ax2.set_xlabel('Median Cosine Shift', fontsize=10)
        ax2.set_title('B. Top 20 Hits', fontsize=11, fontweight='bold', loc='left')
        # Panel C: Distribution
        ax3 = axes[2]
        sig_shifts  = goal_df_ct.loc[goal_df_ct['significant'],  shift_col].dropna()
        nsig_shifts = goal_df_ct.loc[~goal_df_ct['significant'], shift_col].dropna()
        if len(sig_shifts) > 1 and sig_shifts.nunique() > 1:
            ax3.hist(sig_shifts.values, bins=min(20, len(sig_shifts)),
                     color=SIG_COLOR, alpha=0.7, label=f'Sig (n={len(sig_shifts)})', density=True)
        if len(nsig_shifts) > 1 and nsig_shifts.nunique() > 1:
            ax3.hist(nsig_shifts.values, bins=min(20, len(nsig_shifts)),
                     color=NSIG_COLOR, alpha=0.5, label=f'Not sig (n={len(nsig_shifts)})', density=True)
        ax3.axvline(0, color='black', lw=0.8, ls='--')
        ax3.set_xlabel('Median Cosine Shift', fontsize=10)
        ax3.set_ylabel('Density', fontsize=10)
        ax3.set_title('C. Distribution', fontsize=11, fontweight='bold', loc='left')
        ax3.legend(fontsize=8)
        plt.tight_layout()
        fig_path = os.path.join(ct_fig_dir, f'isp_goal_{safe_ct}.pdf')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.savefig(fig_path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
        plt.show()
        print(f'   Saved: {fig_path}')

    # ── Step 13: Pathway enrichment (offline) ───────────────────────────────────
    print(f'\n[{cell_type}] Step 13: Pathway enrichment (offline)...')

    GENESET_DIR = '/path/to/genesets'
    GENE_SET_LIBS = ['KEGG_2021_Human', 'Reactome_2022', 'GO_Biological_Process_2023']

    # Load gene sets once per cell-type loop iteration is wasteful but cheap (small JSON files);
    # better to load once globally before the main loop if you want to optimize.
    gene_set_dicts = {}
    for lib in GENE_SET_LIBS:
        path = os.path.join(GENESET_DIR, f'{lib}.json')
        if os.path.exists(path):
            with open(path) as f:
                gene_set_dicts[lib] = json.load(f)
        else:
            print(f'     Gene set file missing: {path} — skipping {lib}.')

    # Background gene list — all genes detected in this cell type (recommended for accurate stats)
    background_genes = adata_ct.var_names.tolist()

    for state_label in states_to_run:
        state_df_ct = results_df_ct[
            results_df_ct['target_state'] == state_label
        ] if 'target_state' in results_df_ct.columns else results_df_ct
        ct_sig_genes = state_df_ct.loc[
            state_df_ct['significant'], gene_col
        ].dropna().unique().tolist()
        if len(ct_sig_genes) < 5:
            print(f'   [{state_label}] Fewer than 5 significant genes — skipping enrichment.')
            continue

        enr_outdir = os.path.join(ct_out, f'enrichment_{state_label.replace(" ", "_")}')
        os.makedirs(enr_outdir, exist_ok=True)

        all_top_paths = []
        for lib_name, gene_dict in gene_set_dicts.items():
            try:
                enr = gp.enrich(
                    gene_list=ct_sig_genes,
                    gene_sets=gene_dict,
                    background=background_genes,
                    outdir=None,  # set to enr_outdir if you want gseapy's own files too
                )
                res = enr.results.copy()
                res['Gene_set'] = lib_name
                all_top_paths.append(res)
            except Exception as e:
                print(f'   Offline enrichment failed [{state_label}] [{lib_name}]: {e}')

        if not all_top_paths:
            continue

        combined = pd.concat(all_top_paths, ignore_index=True)
        top_paths = combined.nsmallest(10, 'Adjusted P-value')[
            ['Gene_set', 'Term', 'Adjusted P-value', 'Overlap', 'Genes']
        ]
        print(top_paths.to_string())
        top_paths.to_csv(
            os.path.join(ct_out, f'enrichment_{safe_ct}_{state_label.replace(" ", "_")}_top10.csv'),
            index=False
        )
        combined.to_csv(
            os.path.join(enr_outdir, f'enrichment_full_{safe_ct}_{state_label.replace(" ", "_")}.csv'),
            index=False
        )
# Restore original write function
pu.write_perturbation_dictionary = original_write

print(f'\n{"═"*65}')
print(f'Per-cell-type loop complete for this job.')
print(f'Passing cell types (this run): {passing_celltypes}')
print(f'Total ISP result rows (this run): {sum(len(df) for df in all_results)}')

# ════════════════════════════════════════════════════════════════════════════
# Check whether ALL cell types in this dataset (liver/immune) are done yet.
# "Done" mirrors the per-cell-type skip-check used at the top of the loop:
#   {ISP_FIG_DIR}/<safe_ct>/isp_goal_<safe_ct>.pdf exists
# Cell types that FAILED the separation test never produce this file, so we
# also treat a cell type as "resolved" if its separation_test_<ct>.csv shows
# it failed — otherwise a single failing cell type would block the summary
# forever.
# ════════════════════════════════════════════════════════════════════════════

def _safe_name(ct):
    return ct.replace(' ', '_').replace('/', '_')


def cell_type_is_resolved(cell_type):
    """A cell type is 'resolved' if it either:
      (a) completed ISP and has a saved figure, or
      (b) failed the separation test (so ISP was deliberately skipped).
    Returns (resolved: bool, completed_isp: bool).
    """
    safe_ct = _safe_name(cell_type)
    isp_fig = os.path.join(ISP_FIG_DIR, safe_ct, f'isp_goal_{safe_ct}.pdf')
    if os.path.exists(isp_fig):
        return True, True

    sep_csv = os.path.join(OUTPUT_DIR, safe_ct, f'separation_test_{safe_ct}.csv')
    if os.path.exists(sep_csv):
        try:
            sep_df = pd.read_csv(sep_csv)
            if not sep_df.empty and not sep_df['passes'].all():
                # Failed separation test on at least one comparison —
                # this cell type was correctly skipped, not pending.
                return True, False
        except Exception as e:
            print(f'   Could not read {sep_csv}: {e}')

    return False, False


all_celltype_names = list(adata_by_celltype.keys())  # all cell types this dataset/job was configured to know about
resolved_flags  = {ct: cell_type_is_resolved(ct) for ct in all_celltype_names}
unresolved      = [ct for ct, (resolved, _) in resolved_flags.items() if not resolved]
all_done        = (len(unresolved) == 0)

print(f'\nCell-type completion check ({len(all_celltype_names)} total):')
for ct in all_celltype_names:
    resolved, completed_isp = resolved_flags[ct]
    if not resolved:
        status = 'PENDING'
    elif completed_isp:
        status = 'done (ISP complete)'
    else:
        status = 'done (failed separation test)'
    print(f'   [{status:30s}] {ct}')

# ── Update runs_completed.txt ─────────────────────────────────────────────
# Uses the resolved_flags dict already computed above — no extra file
# reads. A cell type is written once it's "resolved" per
# cell_type_is_resolved(): either ISP completed, or it permanently failed
# the separation test (so it'll never be retried anyway).
runs_completed_path = os.path.join(OUTPUT_DIR, 'runs_completed.txt')

existing_completed = set()
if os.path.exists(runs_completed_path):
    with open(runs_completed_path, 'r') as f:
        existing_completed = {line.strip() for line in f if line.strip()}

newly_completed = [
    ct for ct in all_celltype_names
    if resolved_flags[ct][0] and ct not in existing_completed
]

if newly_completed:
    with open(runs_completed_path, 'a') as f:
        for ct in newly_completed:
            f.write(f'{ct}\n')
    print(f'\n Updated {runs_completed_path} with {len(newly_completed)} newly resolved cell type(s):')
    for ct in newly_completed:
        resolved, completed_isp = resolved_flags[ct]
        reason = 'ISP complete' if completed_isp else 'failed separation test'
        print(f'    - {ct} ({reason})')
else:
    print(f'\n No new cell types to add to {runs_completed_path}.')

if not all_done:
    print(f'\n{len(unresolved)} cell type(s) still pending: {unresolved}')
    print('Skipping cross-cell-type summary — re-run once all cell types are complete.')

print('\nKey output files per cell type (under OUTPUT_DIR/<CellType>/):')
print('  candidate_genes_<ct>.csv          — genes tested')
print('  separation_test_<ct>.csv          — separation test result')
print('  isp_stats/<ct>/stats_<ct>_*.csv   — cosine shift scores')
print('  isp_figures/<ct>/isp_goal_<ct>.pdf — per-cell-type ISP figure')
print('  enrichment_*/                      — pathway enrichment')
