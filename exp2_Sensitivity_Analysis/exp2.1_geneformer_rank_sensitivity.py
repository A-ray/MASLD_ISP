#!/usr/bin/env python3
# Baseline Geneformer rank versus ISP shift

# 
# Example:
#   python geneformer_rank_sensitivity.py --run-dir /path/liver/Healthy_MASH/down
#   python geneformer_rank_sensitivity.py --root /path/output/geneformer

# Defaults:
# 500 control-state cells, seed 42, minimum 5 cells/donor, 1024-token trimming,
# V2 CLS/EOS boundaries, and at least 20 evaluable cells/gene. 

from pathlib import Path
import argparse
import json
import re
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def safe_name(value):
    return str(value).replace(' ', '_').replace('/', '_')


def sample_indices(dataset, maximum, minimum, seed):
    """Mirror patient_aware_sample_for_isp and its fallback in supplied pipeline."""
    n = len(dataset)
    rng = np.random.default_rng(seed)
    if maximum is None or maximum >= n:
        return list(range(n))
    donor_key = next((k for k in ['donor', 'Donor', 'donor_id', 'patient_id', 'sample_id']
                      if k in dataset.column_names), None)
    if donor_key is None:
        return rng.choice(n, size=maximum, replace=False).tolist()
    groups = {}
    for i, donor in enumerate(dataset[donor_key]):
        groups.setdefault(donor, []).append(i)
    eligible = {d: ix for d, ix in groups.items() if len(ix) >= minimum}
    if not eligible:
        return rng.choice(n, size=maximum, replace=False).tolist()
    per_donor = max(minimum, int(np.floor(maximum / len(eligible))))
    selected = []
    for ix in eligible.values():
        selected.extend(rng.choice(ix, size=min(per_donor, len(ix)), replace=False).tolist())
    if len(selected) > maximum:
        selected = rng.choice(selected, size=maximum, replace=False).tolist()
    return selected


def trimmed_ids(ids, maximum, candidates, protected):
    """Mirror original trim_sequences_safe, including its tail replacement order."""
    if len(ids) <= maximum:
        return ids
    middle = ids[1:-1]
    keep = middle[:maximum - 2]
    tail = [t for t in middle[maximum - 2:] if t in candidates]
    slots = [i for i in range(len(keep) - 1, -1, -1) if keep[i] not in protected]
    for i, token in zip(slots, tail):
        keep[i] = token
    return [ids[0]] + keep + [ids[-1]]


def rank_table(dataset, candidates, args):
    """Use saved pretrim positions, with two explicitly defined cell populations."""
    if not len(dataset):
        raise ValueError('No cells in requested control state.')
    selected = set(sample_indices(dataset, args.max_isp_cells, args.min_cells_per_donor, args.seed))
    original, matched = {t: [] for t in candidates}, {t: [] for t in candidates}
    eligible_rows = []
    # This pipeline uses the default V2 tokenizer with CLS/EOS. Validate stable
    # boundaries without importing Geneformer or loading a token dictionary.
    first = list(map(int, dataset[0]['input_ids']))
    if len(first) < 2:
        raise ValueError('Expected V2 sequences containing CLS/EOS.')
    cls, eos = first[0], first[-1]
    if cls == eos or cls in candidates or eos in candidates:
        raise ValueError('Cannot validate V2 CLS/EOS boundaries; inspect tokenization.')
    for i, row in enumerate(dataset):
        ids = list(map(int, row['input_ids']))
        if len(ids) < 2 or ids[0] != cls or ids[-1] != eos:
            raise ValueError('Inconsistent CLS/EOS. Script expects unpadded V2 token sequences.')
        if len(set(ids)) != len(ids):
            raise ValueError('Repeated tokens detected; expected unique genes and unpadded sequences.')
        positions = {t: rank for rank, t in enumerate(ids[1:-1], 1) if t in candidates}
        for t, rank in positions.items():
            original[t].append(rank)
        if i in selected and positions:
            eligible_rows.append((ids, positions))
    protected = candidates | {cls, eos}
    for ids, positions in eligible_rows:
        retained = set(trimmed_ids(ids, args.trim_length, candidates, protected))
        for t, rank in positions.items():
            if t in retained:
                matched[t].append(rank)
    rows = []
    for t in sorted(candidates):
        a, b = original[t], matched[t]
        rows.append(dict(token_id=t,
                         median_rank_all_control=np.median(a) if a else np.nan,
                         median_rank_isp_reconstructed=np.median(b) if b else np.nan,
                         n_control_cells=len(dataset), n_control_cells_with_token=len(a),
                         token_presence_fraction=len(a) / len(dataset),
                         n_sampled_control_cells=len(selected),
                         n_sampled_cells_with_any_candidate=len(eligible_rows),
                         n_reconstructed_isp_cells_with_token=len(b)))
    return pd.DataFrame(rows)


def adjust_bh(values):
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid])]
    if len(order):
        q = values[order] * len(order) / np.arange(1, len(order) + 1)
        out[order] = np.minimum(1, np.minimum.accumulate(q[::-1])[::-1])
    return out


def process_run(run, output, args):
    from datasets import load_from_disk
    csv = run / 'isp_stats' / 'all_isp_results.csv'
    results = pd.read_csv(csv, dtype={'ensembl_id': str})
    required = {'gene_symbol', 'ensembl_id', 'median_cosine_shift', 'n_cells',
                'cell_type', 'control_state', 'target_state', 'perturb_mode'}
    if required - set(results):
        raise ValueError(f'{csv}: missing columns {required - set(results)}')
    for col in ['median_cosine_shift', 'n_cells']:
        results[col] = pd.to_numeric(results[col], errors='raise')
    results = results.drop(columns=['token_id'], errors='ignore')
    keys = ['cell_type', 'control_state', 'target_state', 'perturb_mode', 'ensembl_id']
    if results[keys].isna().any().any() or results.duplicated(keys).any():
        raise ValueError(f'{csv}: missing identifiers or duplicate gene/experiment rows.')
    if args.cell_type:
        results = results[results.cell_type.isin(args.cell_type)]
    if results.empty:
        raise ValueError(f'No result rows selected in {csv}')
    output.mkdir(parents=True, exist_ok=True)
    tables, summaries = [], []
    for (cell_type, control), subset in results.groupby(['cell_type', 'control_state'], sort=True):
        ct = safe_name(cell_type)
        candidate_path = run / ct / f'candidate_genes_{ct}.csv'
        mapping = pd.read_csv(candidate_path, dtype={'ensembl_id': str})
        if not {'gene_symbol', 'ensembl_id', 'token_id'} <= set(mapping):
            raise ValueError(f'{candidate_path}: missing mapping columns.')
        mapping = mapping[['gene_symbol', 'ensembl_id', 'token_id']].drop_duplicates()
        numeric = pd.to_numeric(mapping.token_id, errors='raise')
        if numeric.isna().any() or (numeric % 1 != 0).any():
            raise ValueError(f'{candidate_path}: invalid token IDs.')
        mapping['token_id'] = numeric.astype(int)
        if mapping.ensembl_id.isna().any() or mapping.ensembl_id.duplicated().any() or mapping.token_id.duplicated().any():
            raise ValueError(f'{candidate_path}: non-unique Ensembl/token mapping.')
        dataset_dir = run / 'tokenized_data' / ct
        path = dataset_dir / f'geneformer_{ct}_temp.dataset'
        if not path.exists():
            path = dataset_dir / f'geneformer_{ct}.dataset'
        dataset = load_from_disk(str(path))
        if not {'condition', 'input_ids'} <= set(dataset.column_names):
            raise ValueError(f'{path}: missing condition/input_ids.')
        indices = [i for i, condition in enumerate(dataset['condition']) if condition == control]
        dataset = dataset.select(indices)
        ranks = rank_table(dataset, set(mapping.token_id), args)
        ranked = mapping.merge(ranks, on='token_id', validate='one_to_one')
        for (target, mode), group in subset.groupby(['target_state', 'perturb_mode'], sort=True):
            # Outer join retains mapped candidates absent from results, and results
            # absent from the mapping. Neither is silently interpreted as zero.
            table = group.merge(ranked.rename(columns={'gene_symbol': 'mapping_gene_symbol'}),
                                on='ensembl_id', how='outer', validate='one_to_one', indicator=True)
            table['gene_symbol'] = table.gene_symbol.fillna(table.mapping_gene_symbol)
            for col, value in [('cell_type', cell_type), ('control_state', control),
                               ('target_state', target), ('perturb_mode', mode)]:
                table[col] = value
            table['source_run'] = str(run)
            table['rank_dataset'] = str(path)
            table['rank_population'] = args.rank_population
            rank_col = ('median_rank_isp_reconstructed' if args.rank_population == 'isp-reconstructed'
                        else 'median_rank_all_control')
            table['median_baseline_geneformer_rank'] = table[rank_col]
            table['cosine_shift_magnitude'] = table.median_cosine_shift.abs()
            table['isp_cell_count_matches'] = (table.n_cells == table.n_reconstructed_isp_cells_with_token)
            reasons = []
            for row in table.itertuples():
                why = []
                if pd.isna(row.token_id):
                    why.append('missing_token_mapping')
                if not np.isfinite(row.median_cosine_shift):
                    why.append('missing_or_nonfinite_shift')
                if not np.isfinite(row.n_cells) or row.n_cells < args.min_cells_per_gene:
                    why.append('insufficient_isp_cells_placeholder_shift')
                if not np.isfinite(row.median_baseline_geneformer_rank):
                    why.append('rank_unavailable')
                if (args.rank_population == 'isp-reconstructed' and not args.allow_count_mismatch
                        and not row.isp_cell_count_matches):
                    why.append('reconstructed_cell_count_mismatch')
                reasons.append(';'.join(why))
            table['exclusion_reason'] = reasons
            table['included_in_correlation'] = table.exclusion_reason.eq('')
            table = table.drop(columns=['mapping_gene_symbol', '_merge'])
            valid = table.loc[table.included_in_correlation]
            x, y = valid.median_baseline_geneformer_rank, valid.cosine_shift_magnitude
            rho, p = np.nan, np.nan
            status = 'ok'
            if len(valid) < 3:
                status = 'fewer_than_3_evaluable_genes'
            elif x.nunique() < 2 or y.nunique() < 2:
                status = 'constant_rank_or_shift'
            else:
                rho, p = map(float, spearmanr(x, y))
            mismatches = int(((table.n_cells >= args.min_cells_per_gene) & ~table.isp_cell_count_matches).sum())
            if mismatches:
                warnings.warn(f'{run.name}/{ct}/{control}->{target}: {mismatches} evaluable genes have '
                              'cell-count mismatches. Check historical sampling/trim settings.')
            title = f'{cell_type} | {control} → {target} | {mode}'
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(x, y, s=12, alpha=0.35, color='#216E93', linewidths=0, rasterized=True)
            ax.set(xlabel='Median baseline Geneformer rank (1 = highest)',
                   ylabel='Absolute median cosine shift', title=title)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            annotation = f'n = {len(valid):,} genes\nSpearman ρ = {rho:.3f}\np = {p:.3g}' if status == 'ok' else status.replace('_', ' ')
            ax.text(.97, .97, annotation, transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
            ax.spines[['top', 'right']].set_visible(False)
            fig.text(.12, .01, f'Rank population: {args.rank_population}; observed tokens only', fontsize=8)
            fig.tight_layout(rect=(0, .03, 1, 1))
            stem = re.sub(r'[^\w.+-]', '_', f'{ct}__{control}_to_{target}__{mode}')
            fig.savefig(output / f'{stem}.png', dpi=300)
            fig.savefig(output / f'{stem}.pdf')
            plt.close(fig)
            summaries.append(dict(source_run=str(run), cell_type=cell_type, control_state=control,
                                  target_state=target, perturb_mode=mode, rank_population=args.rank_population,
                                  n_genes_total=len(table), n_genes_evaluable=len(valid),
                                  n_cell_count_mismatches=mismatches, spearman_rho=rho, p_value=p, status=status))
            tables.append(table)
            print(f'{title}: {len(valid)}/{len(table)} genes; rho={rho:.4g}; p={p:.4g}', flush=True)
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(output / 'gene_rank_vs_shift.csv', index=False)
    return combined, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument('--run-dir', type=Path, help='One experiment directory containing isp_stats and tokenized_data.')
    location.add_argument('--root', type=Path, help='Recursively find isp_stats/all_isp_results.csv below this directory.')
    parser.add_argument('--output-dir', type=Path, help='Default: <run-dir or root>/rank_sensitivity')
    parser.add_argument('--cell-type', action='append', help='Optional exact cell type, e.g. "T cells". Repeatable.')
    parser.add_argument('--max-isp-cells', type=int, default=500, help='Historical MAX_ISP_CELLS_TOTAL; 0 means all.')
    parser.add_argument('--min-cells-per-donor', type=int, default=5)
    parser.add_argument('--min-cells-per-gene', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trim-length', type=int, default=1024)
    parser.add_argument('--rank-population', choices=['isp-reconstructed', 'all-control'], default='isp-reconstructed')
    parser.add_argument('--allow-count-mismatch', action='store_true', help='Explicitly include reconstructed counts that disagree with n_cells.')
    args = parser.parse_args()
    if args.max_isp_cells < 0 or min(args.min_cells_per_donor, args.min_cells_per_gene) < 1 or args.trim_length < 3:
        parser.error('Invalid cell counts or trim length.')
    if args.max_isp_cells == 0:
        args.max_isp_cells = None
    base = (args.run_dir or args.root).resolve()
    output = (args.output_dir or base / 'rank_sensitivity').resolve()
    runs = [base] if args.run_dir else sorted({p.parent.parent for p in base.rglob('isp_stats/all_isp_results.csv')})
    if not runs:
        parser.error(f'No isp_stats/all_isp_results.csv found below {base}')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'analysis_parameters.json').open('w') as handle:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, handle, indent=2)
    all_tables, summaries, failures = [], [], []
    for run in runs:
        destination = output if args.run_dir else output / run.relative_to(base)
        try:
            table, summary = process_run(run, destination, args)
            all_tables.append(table)
            summaries.extend(summary)
        except Exception as exc:
            failures.append(dict(source_run=str(run), error=f'{type(exc).__name__}: {exc}'))
            print(f'FAILED {run}: {exc}', flush=True)
    if summaries:
        summary = pd.DataFrame(summaries)
        summary['p_adj_bh'] = adjust_bh(summary.p_value)
        summary.to_csv(output / 'spearman_summary.csv', index=False)
    if all_tables and args.root:
        pd.concat(all_tables, ignore_index=True).to_csv(output / 'all_gene_rank_vs_shift.csv', index=False)
    # Write even an empty failure report so an earlier failure log is not stale.
    pd.DataFrame(failures, columns=['source_run', 'error']).to_csv(output / 'failed_runs.csv', index=False)
    print(f'Outputs: {output}', flush=True)
    if failures:
        raise SystemExit(f'{len(failures)} run(s) failed; see failed_runs.csv. Other runs were processed.')


if __name__ == '__main__':
    main()
