#!/usr/bin/env python3

# Geneformer ISP — Cross-Cell-Type Summary Visualization (standalone)

# Runs independently of the main ISP pipeline. Reads per-cell-type results
# that are already saved on disk (isp_stats/<ct>/stats_<ct>_*.csv and
# <ct>/separation_test_<ct>.csv) and produces:
#   - separation_test_summary.csv (all cell types)
#   - isp_stats/all_isp_results.csv (all cell types, ISP-completed only)
#   - isp_figures/isp_dotplot_ctrl_vs_<state>.pdf (one per target/alt state)
#   - a zip of everything under OUTPUT_DIR

# Completion is gated on OUTPUT_DIR/runs_completed.txt: every cell type
# expected for this dataset (inferred from the source .h5ad files) must
# appear in that file before summary plots are built, unless --force is
# passed.

# Usage:
#     python geneformer_isp_summary.py liver --control Healthy --target MASH --mode down
#     python geneformer_isp_summary.py immune --target MASH --alt-states Fibrotic --force


import os
import sys
import glob
import argparse
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless — HPC nodes have no display
import matplotlib.pyplot as plt


# ════════════════════════════════════════════════════════════════════════
# CLI / configuration
# ════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description='Geneformer ISP cross-cell-type summary')
parser.add_argument('dataset', choices=['liver', 'immune'],
                     help="Dataset to summarize: 'liver' or 'immune'")
parser.add_argument('--control', default='Healthy',
                     help="Control condition label (default: 'Healthy')")
parser.add_argument('--target', default='MASH',
                     help="Target condition label (default: 'MASH')")
parser.add_argument('--mode', choices=['down', 'up'], default='down',
                     help="Perturbation mode: 'down' or 'up' (default: 'down')")
parser.add_argument('--alt-states', nargs='*', default=[],
                     help='Additional target states (besides --target) to also plot, space-separated')
parser.add_argument('--force', action='store_true',
                     help='Build the summary from whatever cell types are done, '
                          'even if some are still pending')
args = parser.parse_args()

dataset_choice = args.dataset.lower()
CONTROL_LABEL  = args.control
TARGET_LABEL   = args.target
PERTURB_MODE   = args.mode
_alt_states    = args.alt_states

if dataset_choice == 'liver':
    H5AD_PATH_DIR = '/path/to/valid_liver/'
elif dataset_choice == 'immune':
    H5AD_PATH_DIR = '/path/to/valid_immune/'

OUTPUT_DIR = (
    f'/path/to/validation_output/geneformer/'
    f'{dataset_choice}/{CONTROL_LABEL}_{TARGET_LABEL}/{PERTURB_MODE}'
)
STATS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'isp_stats')
ISP_FIG_DIR      = os.path.join(OUTPUT_DIR, 'isp_figures')
os.makedirs(ISP_FIG_DIR, exist_ok=True)

SIG_COLOR  = '#C0392B'
NSIG_COLOR = '#BDC3C7'
ALT_COLOR  = '#2980B9'

print(' Configuration set.')
print(f'   H5AD directory : {H5AD_PATH_DIR}')
print(f'   OUTPUT_DIR     : {OUTPUT_DIR}')
print(f'   Control        : {CONTROL_LABEL}')
print(f'   Target         : {TARGET_LABEL}')
print(f'   Perturb        : {PERTURB_MODE}-regulation')
print(f'   Alt states     : {_alt_states}')


def _safe_name(ct):
    return ct.replace(' ', '_').replace('/', '_')


# ════════════════════════════════════════════════════════════════════════
# Determine the FULL expected cell-type list for this dataset from the
# source .h5ad files — NOT from anything in memory, since this script never
# loads any adata. This mirrors infer_cell_type_from_adata()'s filename-stem
# fallback in the main pipeline (underscores -> spaces).
# ════════════════════════════════════════════════════════════════════════
h5ad_files = sorted(glob.glob(os.path.join(H5AD_PATH_DIR, '*.h5ad')))
if not h5ad_files:
    print(f'ERROR: no .h5ad files found in {H5AD_PATH_DIR}')
    sys.exit(1)

all_celltype_names = [
    os.path.splitext(os.path.basename(f))[0].replace('_', ' ')
    for f in h5ad_files
]
print(f'\nExpected cell types ({len(all_celltype_names)}) from source .h5ad files:')
for ct in all_celltype_names:
    print(f'   - {ct}')


# ════════════════════════════════════════════════════════════════════════
# Completion check via runs_completed.txt
# ════════════════════════════════════════════════════════════════════════
runs_completed_path = os.path.join(OUTPUT_DIR, 'runs_completed.txt')
if not os.path.exists(runs_completed_path):
    print(f'\nERROR: {runs_completed_path} not found — '
          f'no cell types have been marked complete for this dataset/control/target/mode yet.')
    sys.exit(1)

with open(runs_completed_path, 'r') as f:
    completed_set = {line.strip() for line in f if line.strip()}

unresolved = [ct for ct in all_celltype_names if ct not in completed_set]

print(f'\nCell-type completion check ({len(all_celltype_names)} total, '
      f'per {runs_completed_path}):')
for ct in all_celltype_names:
    status = 'done' if ct in completed_set else 'PENDING'
    print(f'   [{status:10s}] {ct}')

if unresolved and not args.force:
    print(f'\n{len(unresolved)} cell type(s) still pending: {unresolved}')
    print('Re-run once all cell types are complete, or pass --force to '
          'summarize just the subset that is done.')
    sys.exit(0)

if unresolved and args.force:
    print(f'\n--force set: proceeding with '
          f'{len(all_celltype_names) - len(unresolved)}/{len(all_celltype_names)} '
          f'cell types; skipping pending: {unresolved}')
else:
    print('\nAll expected cell types resolved — building cross-cell-type '
          'summary from saved results on disk.')

ready_celltypes = [ct for ct in all_celltype_names if ct in completed_set]


# ════════════════════════════════════════════════════════════════════════
# Re-read all per-cell-type results from disk. A cell type in
# runs_completed.txt may still have no ISP stats if it failed the
# separation test (deliberately skipped) — those contribute to the
# separation summary but not to the ISP dot plots.
# ════════════════════════════════════════════════════════════════════════
disk_sep_rows = []
disk_results  = []
completed_celltypes = []  # ready cell types that actually have ISP stats

for ct in ready_celltypes:
    safe_ct = _safe_name(ct)

    sep_csv = os.path.join(OUTPUT_DIR, safe_ct, f'separation_test_{safe_ct}.csv')
    if os.path.exists(sep_csv):
        try:
            disk_sep_rows.append(pd.read_csv(sep_csv))
        except Exception as e:
            print(f'   Could not read {sep_csv}: {e}')
    else:
        print(f'   NOTE: no separation_test csv for "{ct}" at {sep_csv}')

    ct_stats_dir = os.path.join(STATS_OUTPUT_DIR, safe_ct)
    stats_files = glob.glob(os.path.join(ct_stats_dir, f'stats_{safe_ct}_*.csv'))
    if not stats_files:
        print(f'   "{ct}" is marked complete but has no ISP stats CSVs '
              f'(likely failed separation test) — excluded from ISP dot plots.')
        continue

    completed_celltypes.append(ct)
    for f in stats_files:
        try:
            disk_results.append(pd.read_csv(f))
        except Exception as e:
            print(f'   Could not read {f}: {e}')

passing_celltypes = completed_celltypes


# ════════════════════════════════════════════════════════════════════════
# Separation test summary (all ready cell types)
# ════════════════════════════════════════════════════════════════════════
if disk_sep_rows:
    sep_summary = pd.concat(disk_sep_rows, ignore_index=True)
    sep_summary_path = os.path.join(OUTPUT_DIR, 'separation_test_summary.csv')
    sep_summary.to_csv(sep_summary_path, index=False)
    print(f'\nSeparation Test Summary saved: {sep_summary_path}')
    print(sep_summary[['cell_type', 'comparison', 'passes', 'sep_score',
                        'p_value', 'cohens_d', 'balanced_accuracy',
                        'n_a', 'n_b']].sort_values(['cell_type', 'comparison']).to_string(index=False))
else:
    print('\nNo separation test results found on disk.')


# ════════════════════════════════════════════════════════════════════════
# Cross-cell-type ISP results + dot plots
# ════════════════════════════════════════════════════════════════════════
def _dot_plot(df, state_label, color, out_name, title_color_note):
    """Build one gene x cell-type dot plot for a given target/alt state."""
    if df.empty or len(passing_celltypes) < 2:
        return
    sig = df[df['significant']].dropna(subset=[shift_col])
    if len(sig) == 0:
        print(f'No significant hits for {state_label} — dot plot skipped.')
        return

    top_genes = (sig.groupby(gene_col)[shift_col].mean()
                 .nsmallest(20).index.tolist())
    pivot_shift = df[df[gene_col].isin(top_genes)].pivot_table(
        index=gene_col, columns='cell_type', values=shift_col, aggfunc='mean'
    ).fillna(0)
    pivot_padj = df[df[gene_col].isin(top_genes)].pivot_table(
        index=gene_col, columns='cell_type', values=padj_col, aggfunc='mean'
    ).fillna(1.0)

    x_labels = pivot_shift.columns.tolist()
    y_labels = pivot_shift.index.tolist()
    fig, ax = plt.subplots(figsize=(max(6, len(x_labels) * 2.5), max(6, len(y_labels) * 0.5)))
    for xi, ct in enumerate(x_labels):
        for yi, gene in enumerate(y_labels):
            pv  = pivot_padj.loc[gene, ct] if ct in pivot_padj.columns else 1.0
            sz  = max(20, -np.log10(pv + 1e-300) * 15)
            col = color if pv < 0.05 else NSIG_COLOR
            ax.scatter(xi, yi, s=sz, c=[col], alpha=0.85, linewidths=0.5, edgecolors='black')
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=35, ha='right', fontsize=9)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel('Cell Type', fontsize=11)
    ax.set_ylabel('Gene', fontsize=11)
    ax.set_title(
        f'Top Perturbation Hits Across Cell Types — Control → {state_label}\n'
        f'Dot size ∝ −log₁₀(FDR p-value)  |  {title_color_note}',
        fontsize=11, fontweight='bold'
    )
    ax.grid(True, alpha=0.2, linewidth=0.5)
    plt.tight_layout()
    out_path = os.path.join(ISP_FIG_DIR, out_name)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


if not disk_results:
    print('\nNo ISP results found on disk to summarize.')
else:
    results_df = pd.concat(disk_results, ignore_index=True)
    results_df = results_df.dropna(subset=['median_cosine_shift'])
    results_df['pval_adj'] = results_df['pval_adj'].fillna(1.0)

    final_path = os.path.join(STATS_OUTPUT_DIR, 'all_isp_results.csv')
    results_df.to_csv(final_path, index=False)
    print(f'\nAll results saved: {final_path}  ({len(results_df)} rows)')

    gene_col  = 'gene_symbol' if 'gene_symbol' in results_df.columns else 'ensembl_id'
    shift_col = 'median_cosine_shift'
    padj_col  = 'pval_adj'

    goal_df = results_df[results_df['target_state'] == TARGET_LABEL].copy()
    _dot_plot(goal_df, TARGET_LABEL, SIG_COLOR,
              'isp_dotplot_ctrl_vs_goal.pdf', 'Red = significant')

    for alt_state in _alt_states:
        alt_df = results_df[results_df['target_state'] == alt_state].copy()
        _dot_plot(alt_df, alt_state, ALT_COLOR,
                  f'isp_dotplot_ctrl_vs_{alt_state.replace(" ", "_")}.pdf',
                  'Blue = significant')

print('\nCross-cell-type summary plots complete.')


# ════════════════════════════════════════════════════════════════════════
# Zip everything under OUTPUT_DIR
# ════════════════════════════════════════════════════════════════════════
zip_path = os.path.join(OUTPUT_DIR, 'geneformer_isp_results.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, fnames in os.walk(OUTPUT_DIR):
        for fn in fnames:
            fp = os.path.join(root, fn)
            if fp == zip_path:
                continue
            zf.write(fp, os.path.relpath(fp, OUTPUT_DIR))

print(f'\nAll results zipped: {zip_path}')
print('\nKey output files per cell type (under OUTPUT_DIR/<CellType>/):')
print('  separation_test_<ct>.csv          — separation test result')
print('  isp_stats/<ct>/stats_<ct>_*.csv   — cosine shift scores')
print('\nCross-cell-type outputs (under OUTPUT_DIR/):')
print('  separation_test_summary.csv        — all cell types')
print('  isp_stats/all_isp_results.csv       — all ISP results combined')
print('  isp_figures/isp_dotplot_*.pdf       — cross-cell-type dot plots')
