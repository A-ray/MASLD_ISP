#!/usr/bin/env python3
# Donor-level sensitivity analysis of existing Geneformer v4 ISP results.

import argparse
import hashlib
import json
import math
import pickle
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
from types import SimpleNamespace
import numpy as np
import pandas as pd

# Lazy loading permits --help and numerical unit checks without datasets installed.
hf = None

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

def valid_donor(value):
    return not pd.isna(value) and str(value).strip().lower() not in {
        '', 'unknown', 'nan', 'none', 'na', 'n/a', 'null', '<na>'}

def validate_run(a):
    a.output_dir.mkdir(parents=True, exist_ok=True)
    # Separate invocation outputs, so one cell type does not overwrite another.
    safe = a.cell_type.replace(' ', '_').replace('/', '_')
    out = a.output_dir / (safe + '_' + a.control + '_' + a.target + '_' + a.mode)
    out.mkdir(parents=True, exist_ok=True)
    mapping_path = out / 'verified_cell_donor_mapping.csv'
    # Remove a prior PASS mapping so a failed rerun cannot leave a stale success artifact.
    if mapping_path.exists():
        mapping_path.unlink()
    report = {'status': 'FAIL', 'errors': [], 'settings': vars(a).copy(),
              'numpy_version': np.__version__, 'datasets_version': hf.__version__}
    errors = report['errors']
    audit = []
    try:
        run = Path(a.run_dir)
        base = run / 'tokenized_data' / safe
        path = a.dataset or base / f'geneformer_{safe}_temp.dataset'
        if not path.exists() and a.dataset is None:
            path = base / f'geneformer_{safe}.dataset'
        report['dataset_path'] = str(path)
        data = hf.load_from_disk(str(path))
        required = {'input_ids', 'length', 'condition', 'patient_id'}
        if not required.issubset(data.column_names):
            raise ValueError(f'Missing dataset columns: {required - set(data.column_names)}')
        if '_audit_source_row' in data.column_names:
            raise ValueError('Reserved audit column already exists')
        data = data.add_column('_audit_source_row', list(range(len(data))))
        control = data.filter(lambda x: x['condition'] == a.control, num_proc=1)
        if not len(control):
            raise ValueError('No starting-condition cells')
        donor_key = next(k for k in ['donor', 'Donor', 'donor_id', 'patient_id', 'sample_id']
                         if k in control.column_names)
        report['sampling_donor_column'] = donor_key
        # Use ORIGINAL candidate list, never the recurrent/significant-only list.
        candidate_path = run / safe / f'candidate_genes_{safe}.csv'
        candidates = pd.read_csv(candidate_path)
        tokens = set(int(x) for x in candidates['token_id'])
        if not tokens:
            raise ValueError('Empty original candidate list')
        ids, sampling_report = patient_aware_sample_for_isp(
            control, donor_key=donor_key,
            max_cells_total=a.max_cells or None,
            min_cells_per_donor=a.min_cells_per_donor, random_state=a.seed)
        selected = control.select(ids)
        selected = selected.filter(lambda x: any(int(t) in tokens for t in x['input_ids']), num_proc=1)
        if not len(selected):
            raise ValueError('No sampled cells after candidate filtering')
        selected = trim_sequences_safe(selected, a.max_length, tokens).flatten_indices()
        # Mirrors apply_additional_filters / downsample_and_sort with max_ncells=None.
        selected = selected.filter(lambda x: x['condition'] == a.control, num_proc=1)
        selected = selected.select(list(range(len(selected)))).sort('length', reverse=True)
        report.update(n_control_cells=len(control), n_sampled_cells=len(ids),
                      n_reconstructed_cells=len(selected), n_candidate_tokens=len(tokens))

        # Build a sequence fingerprint for ALL control cells, including unsampled
        # cells. Cross-donor collisions cannot be resolved from pickle keys alone.
        pool = control.filter(lambda x: any(int(t) in tokens for t in x['input_ids']), num_proc=1)
        pool = trim_sequences_safe(pool, a.max_length, tokens)
        signature_donors = defaultdict(set)
        def signature(row):
            seq = [int(t) for t in row['input_ids'][1:-1]]
            if a.mode == 'up':
                seq = seq[1:]  # Geneformer skips already top-ranked gene.
            return tuple(seq)
        for row in pool:
            donor = str(row['patient_id']) if valid_donor(row['patient_id']) else '<INVALID>'
            signature_donors[signature(row)].add(donor)

        pattern = re.compile(r'^in_silico_(delete|overexpress)_isp_' + re.escape(safe)
                             + r'_dict_cell_embs_(\d+)batch(-?\d+)_raw\.pickle$')
        groups = defaultdict(list)
        kind = 'delete' if a.mode == 'down' else 'overexpress'
        files = list((run / 'isp_output' / safe).glob('*dict_cell_embs_*.pickle'))
        for f in files:
            m = pattern.fullmatch(f.name)
            if not m or m[1] != kind:
                errors.append(f'Unexpected pickle filename: {f.name}')
                continue
            groups[int(m[2])].append((int(m[3]), f))
        report.update(n_pickle_files=len(files), n_saved_cell_indices=len(groups))
        if set(groups) != set(range(len(selected))):
            errors.append('Saved cell indices do not exactly match reconstructed indices; '
                          f'missing={sorted(set(range(len(selected))) - set(groups))[:20]}, '
                          f'extra={sorted(set(groups) - set(range(len(selected))))[:20]}')

        for h, row in enumerate(selected):
            expected = signature(row)
            states = defaultdict(list)
            reasons = []
            seen_batches = set()
            for batch, f in sorted(groups.get(h, [])):
                if batch in seen_batches:
                    reasons.append('duplicate batch index')
                seen_batches.add(batch)
                with f.open('rb') as handle:
                    obj = pickle.load(handle)
                if not isinstance(obj, dict):
                    raise ValueError(f'Unexpected pickle root in {f.name}')
                if not {a.control, a.target}.issubset(obj):
                    reasons.append('required state missing from batch')
                for state, inner in obj.items():
                    if not isinstance(inner, dict):
                        raise ValueError(f'Unexpected state structure in {f.name}')
                    for key, values in inner.items():
                        if not isinstance(key, tuple) or len(key) != 2 or key[1] != 'cell_emb':
                            raise ValueError(f'Unexpected key in {f.name}: {key!r}')
                        if not isinstance(values, list) or len(values) != 1:
                            reasons.append('not exactly one shift per gene per cell')
                        elif not np.isscalar(values[0]) or not np.isfinite(values[0]):
                            reasons.append('nonfinite or nonscalar shift')
                        states[state].append(int(key[0]))
            if not {a.control, a.target}.issubset(states):
                reasons.append('required state missing')
            for state, seq in states.items():
                if len(seq) != len(set(seq)):
                    reasons.append(f'{state}: duplicate gene token across batches')
                if tuple(seq) != expected:
                    reasons.append(f'{state}: ordered tokens do not match reconstructed cell')
            if not valid_donor(row['patient_id']):
                reasons.append('invalid patient_id')
            donors = signature_donors[expected]
            if len(donors) != 1 or '<INVALID>' in donors:
                reasons.append('sequence fingerprint ambiguous across donors')
            audit.append({'isp_cell_index': h, 'source_dataset_row': row['_audit_source_row'],
                          'patient_id': row['patient_id'], 'condition': row['condition'],
                          'n_genes_expected': len(expected), 'n_files': len(groups.get(h, [])),
                          'sequence_donor_count': len(donors), 'passed': not reasons,
                          'reason': '; '.join(sorted(set(reasons)))})
        df = pd.DataFrame(audit)
        df.to_csv(out / 'cell_mapping_audit.csv', index=False)
        failed = int((~df['passed']).sum())
        report.update(n_cells_passed=int(df['passed'].sum()), n_cells_failed=failed,
                      n_donors=int(df['patient_id'].nunique()))
        if failed:
            errors.append(f'{failed} cells failed validation; see cell_mapping_audit.csv')
        if not errors:
            report['status'] = 'PASS'
            df.to_csv(mapping_path, index=False)
        else:
            report['failure_examples'] = df.loc[~df.passed, ['isp_cell_index', 'reason']].head(10).to_dict('records')
    except Exception as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
    report['interpretation'] = (
        'PASS: ordered perturbation tokens and donor mapping validated under the supplied original settings. '
        'Biological uniqueness of patient IDs across source datasets still requires metadata verification.'
        if report['status'] == 'PASS' else
        'FAIL: do not use reconstructed donor assignments. This does not by itself mean ISP must be rerun.')
    text = json.dumps(report, indent=2, default=str)
    (out / 'validation_summary.json').write_text(text + '\n')
    print('\n' + text)
    print('\nOutputs:', out.resolve())
    return report, mapping_path


KEY = ['tissue', 'cell_type', 'control_state', 'target_state', 'perturb_mode']


def safe_name(x):
    return str(x).replace(' ', '_').replace('/', '_')


def truth(x):
    return str(x).strip().lower() in {'true', '1', '1.0', 'yes'}


def normalize_tissue(x):
    value = str(x).strip().lower()
    return value if value in {'immune', 'liver'} else None


def exact_signed_rank_greater(values):
    """Return W+, exact one-sided p, nonzero n. Exact absolute-value ties retained.

    Scale average ranks by two so DP uses integer indices. Every nonzero donor
    contributes an independent fair sign under the symmetric null. This handles
    tied ranks correctly.
    """
    d = np.asarray(values, dtype=float)
    if not np.isfinite(d).all():
        raise ValueError('Nonfinite donor medians passed to test')
    d = d[d != 0]
    if not len(d):
        return 0.0, 1.0, 0
    r2 = np.rint(pd.Series(np.abs(d)).rank(method='average').to_numpy() * 2).astype(int)
    observed = int(r2[d > 0].sum())
    pmf = np.ones(1, dtype=float)
    for weight in r2:
        new = np.zeros(len(pmf) + int(weight), dtype=float)
        new[:len(pmf)] += pmf * 0.5
        new[int(weight):] += pmf * 0.5
        pmf = new
    p = min(1.0, float(pmf[observed:].sum()))
    return observed / 2.0, p, len(d)


def bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if not len(p):
        return p
    if not np.isfinite(p).all():
        raise ValueError('BH input includes nonfinite P-values')
    order = np.argsort(p, kind='stable')
    q_sorted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1]
    q = np.empty_like(p)
    q[order] = np.minimum(q_sorted, 1.0)
    return q


def read_findings(args):
    records, problems = [], []
    files = sorted(f for f in args.sig_dir.rglob('*')
                   if f.is_file() and f.name.lower().endswith('_sig_only.csv'))
    if not files:
        raise ValueError(f'No *_sig_only.csv files in {args.sig_dir}')
    for path in files:
        try:
            frame = pd.read_csv(path, dtype=str)
            frame.columns = frame.columns.str.strip()
            needed = {'cell_type', 'control_state', 'target_state', 'perturb_mode', 'median_cosine_shift'}
            if not needed.issubset(frame):
                raise ValueError(f'Missing columns: {needed - set(frame.columns)}')
            if not {'ensembl_id', 'gene_symbol'}.intersection(frame.columns):
                raise ValueError('Need ensembl_id or gene_symbol')
            numeric = pd.to_numeric(frame['median_cosine_shift'], errors='coerce')
            if numeric.isna().any() or not np.isfinite(numeric).all():
                raise ValueError('Missing/nonfinite/nonnumeric discovery median_cosine_shift')
            frame = frame.loc[numeric > 0].copy()
            if 'significant' in frame:
                frame = frame.loc[frame['significant'].map(truth)]
            filename_tissues = set(re.findall(r'(?:^|[_\W])(immune|liver)(?=[_\W]|$)', path.stem.lower()))
            for index, row in frame.iterrows():
                try:
                    values = {k: str(row[k]).strip() for k in KEY[1:]}
                    if any(v.lower() in {'nan', '', 'none'} for v in values.values()):
                        raise ValueError('Missing comparison metadata')
                    mode = values['perturb_mode'].lower()
                    if mode not in {'up', 'down'}:
                        raise ValueError(f'Unsupported mode {mode}')
                    values['perturb_mode'] = mode
                    tissue_hints = set(filename_tissues)
                    for column in ('tissue', 'Tissue', 'Cell Domain', 'cell_domain', 'dataset_choice'):
                        if column in row and pd.notna(row[column]):
                            hint = normalize_tissue(row[column])
                            if hint:
                                tissue_hints.add(hint)
                    if len(tissue_hints) > 1:
                        raise ValueError('Conflicting tissue hints in columns/filename')
                    safe_ct = safe_name(values['cell_type'])
                    rel = Path(f"{values['control_state']}_{values['target_state']}") / mode
                    if not tissue_hints:
                        tissue_hints = {t for t in ('liver', 'immune')
                                        if (args.results_root / t / rel / safe_ct /
                                            f'candidate_genes_{safe_ct}.csv').exists()}
                    if len(tissue_hints) != 1:
                        raise ValueError('Cannot resolve tissue uniquely; add a tissue column')
                    values['tissue'] = next(iter(tissue_hints))
                    eid = str(row.get('ensembl_id', '')).strip()
                    symbol = str(row.get('gene_symbol', '')).strip()
                    eid = '' if eid.lower() in {'nan', 'none'} else eid
                    symbol = '' if symbol.lower() in {'nan', 'none'} else symbol
                    if not eid and not symbol:
                        raise ValueError('Missing gene identifier')
                    values.update(ensembl_id=eid, gene_symbol=symbol,
                                  discovery_median_cosine_shift=float(row['median_cosine_shift']),
                                  discovery_pval_adj=row.get('pval_adj', ''),
                                  source_csv=str(path), source_row=int(index)+2)
                    records.append(values)
                except Exception as exc:
                    problems.append({'stage': 'discovery_row', 'source': str(path),
                                     'row': int(index)+2, 'reason': str(exc)})
        except Exception as exc:
            problems.append({'stage': 'discovery_file', 'source': str(path), 'reason': str(exc)})
    return records, problems, len(files)


def resolve_run_findings(items, candidate_path):
    """Map selected discoveries to original candidate tokens and deduplicate tests."""
    candidates = pd.read_csv(candidate_path, dtype={'ensembl_id': str, 'gene_symbol': str})
    tests, problems = {}, []
    for row in items:
        try:
            if row['ensembl_id']:
                match = candidates.loc[candidates['ensembl_id'] == row['ensembl_id']]
            else:
                match = candidates.loc[candidates['gene_symbol'] == row['gene_symbol']]
            match = match.drop_duplicates(subset=['ensembl_id', 'token_id'])
            if len(match) != 1:
                raise ValueError(f'Gene maps to {len(match)} original candidates; require exactly one')
            m = match.iloc[0]
            token = int(m['token_id'])
            if token not in tests:
                rec = dict(row)
                rec.update(ensembl_id=str(m['ensembl_id']), gene_symbol=str(m['gene_symbol']),
                           token_id=token, n_discovery_rows=1)
                identity = [str(rec[k]) for k in KEY] + [rec['ensembl_id']]
                rec['finding_id'] = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:20]
                tests[token] = rec
            else:
                tests[token]['n_discovery_rows'] += 1
                sources = set(tests[token]['source_csv'].split('; ')) | {row['source_csv']}
                tests[token]['source_csv'] = '; '.join(sorted(sources))
        except Exception as exc:
            problems.append({'stage': 'gene_mapping', **{k: row[k] for k in KEY},
                             'gene': row['ensembl_id'] or row['gene_symbol'], 'reason': str(exc)})
    return tests, problems


def read_selected_shifts(run_dir, safe_ct, mode, target, tests, mapping):
    kind = 'delete' if mode == 'down' else 'overexpress'
    pattern = re.compile(r'^in_silico_' + kind + '_isp_' + re.escape(safe_ct)
                         + r'_dict_cell_embs_(\d+)batch(-?\d+)_raw\.pickle$')
    cells = mapping.set_index('isp_cell_index')
    rows, seen = [], set()
    for path in sorted((run_dir / 'isp_output' / safe_ct).glob('*dict_cell_embs_*.pickle')):
        m = pattern.fullmatch(path.name)
        if not m:
            raise ValueError(f'Unexpected pickle: {path.name}')
        h = int(m[1])
        if h not in cells.index:
            raise ValueError(f'Unmapped cell {h}')
        with path.open('rb') as f:
            data = pickle.load(f)
        for key, values in data[target].items():
            token, label = key
            token = int(token)
            if label != 'cell_emb' or token not in tests:
                continue
            if (h, token) in seen:
                raise ValueError(f'Duplicate cell/gene result: {h}, {token}')
            seen.add((h, token))
            if len(values) != 1 or not np.isfinite(values[0]):
                raise ValueError('Invalid selected shift')
            rec = tests[token]
            rows.append({'finding_id': rec['finding_id'], 'isp_cell_index': h,
                         'patient_id': str(cells.loc[h, 'patient_id']),
                         'cosine_shift': float(values[0])})
    return pd.DataFrame(rows, columns=['finding_id', 'isp_cell_index', 'patient_id', 'cosine_shift'])


def summarize_tests(tests, cells, mapping, args):
    donor_rows, summary_rows = [], []
    for token, rec in tests.items():
        raw = cells.loc[cells.finding_id == rec['finding_id']]
        agg = raw.groupby('patient_id', sort=True)['cosine_shift'].agg(['median', 'size']).reset_index()
        keep = agg['size'] >= args.min_cells_per_gene_donor
        for _, donor in agg.iterrows():
            donor_rows.append({**{k: rec[k] for k in KEY},
                               'finding_id': rec['finding_id'], 'gene_symbol': rec['gene_symbol'],
                               'ensembl_id': rec['ensembl_id'], 'patient_id': donor['patient_id'],
                               'n_cells': int(donor['size']), 'donor_median_shift': float(donor['median']),
                               'included_in_test': bool(donor['size'] >= args.min_cells_per_gene_donor)})
        values = agg.loc[keep, 'median'].to_numpy(float)
        n = len(values)
        result = {**rec, 'n_donors_in_validated_run': int(mapping.patient_id.nunique()),
                  'n_donors_with_gene': len(agg), 'n_donors_evaluable': n,
                  'n_donors_excluded_low_cells': int((~keep).sum()),
                  'n_cells_with_gene': len(raw), 'n_cells_included': int(agg.loc[keep, 'size'].sum()),
                  'n_positive_donors': int((values > 0).sum()),
                  'n_negative_donors': int((values < 0).sum()),
                  'n_zero_donors': int((values == 0).sum()),
                  'positive_donor_fraction': float((values > 0).mean()) if n else np.nan,
                  'median_donor_shift': float(np.median(values)) if n else np.nan,
                  'donor_shift_q25': float(np.quantile(values, .25)) if n else np.nan,
                  'donor_shift_q75': float(np.quantile(values, .75)) if n else np.nan,
                  'n_nonzero_donors': int((values != 0).sum()),
                  'wilcoxon_W_plus': np.nan, 'pval_raw': np.nan,
                  'test_method': 'exact_conditional_signed_rank_wilcox',
                  'test_status': 'tested' if n >= args.min_donors else 'insufficient_donors'}
        if n >= args.min_donors:
            w, pvalue, nonzero = exact_signed_rank_greater(values)
            result.update(wilcoxon_W_plus=w, pval_raw=pvalue, n_nonzero_donors=nonzero)
        summary_rows.append(result)
    return summary_rows, donor_rows


def classify(frame, args):
    frame = frame.copy()
    frame['pval_adj'] = np.nan
    evaluable = frame.test_status.eq('tested') & frame.pval_raw.notna()
    frame.loc[evaluable, 'pval_adj'] = bh_adjust(frame.loc[evaluable, 'pval_raw'])
    frame['bh_family_size'] = int(evaluable.sum())
    frame['donor_significant'] = evaluable & (frame.pval_adj < args.alpha)
    frame['positive_median'] = frame.median_donor_shift > 0
    frame['consistent_positive'] = (evaluable & frame.positive_median &
                                    (frame.positive_donor_fraction >= args.consistent_fraction))
    frame['all_donors_positive'] = evaluable & (frame.positive_donor_fraction == 1)
    frame['classification'] = 'not_supported_at_donor_level'
    frame.loc[frame.positive_median, 'classification'] = 'positive_median_mixed_directions'
    frame.loc[frame.consistent_positive, 'classification'] = 'consistent_positive_not_BH_significant'
    frame.loc[frame.donor_significant & frame.positive_median, 'classification'] = 'BH_significant_positive'
    frame.loc[frame.donor_significant & ~frame.positive_median, 'classification'] = 'BH_significant_without_positive_median'
    frame.loc[~evaluable, 'classification'] = 'insufficient_donors'
    return frame


def make_plots(summary, donors, out, args, provisional):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    fig_dir = out / 'figures'
    fig_dir.mkdir()
    colors = {'BH_significant_positive': '#007C66',
              'consistent_positive_not_BH_significant': '#2879B9'}
    count = 0
    with PdfPages(fig_dir / 'donor_dotplots.pdf') as pdf:
        for (tissue, mode), group in summary.groupby(['tissue', 'perturb_mode'], sort=True):
            group = group.sort_values(['control_state', 'target_state', 'cell_type', 'gene_symbol']).reset_index(drop=True)
            for start in range(0, len(group), args.plot_rows):
                page = group.iloc[start:start+args.plot_rows]
                fig, ax = plt.subplots(figsize=(15, max(3.6, 1.9 + .48 * len(page))))
                labels = []
                for y, (_, row) in enumerate(page.iterrows()):
                    points = donors.loc[donors.finding_id == row.finding_id].sort_values('patient_id') if len(donors) else donors
                    for _, point in points.iterrows():
                        # Stable jitter for the same donor across rows, no RNG dependence.
                        digest = hashlib.sha256(str(point.patient_id).encode()).digest()
                        jitter = (int.from_bytes(digest[:4], 'big') / (2**32-1) - .5) * .32
                        included = bool(point.included_in_test)
                        ax.scatter(float(point.donor_median_shift), y+jitter, s=22,
                                   color=colors.get(row.classification, '#777777') if included else '#C8C8C8',
                                   marker='o' if included else 'x', alpha=.72, linewidths=.7, zorder=3)
                    if np.isfinite(row.median_donor_shift):
                        ax.scatter(row.median_donor_shift, y, color='black', marker='D', s=20, zorder=4)
                    q = f'{row.pval_adj:.3g}' if np.isfinite(row.pval_adj) else 'NA'
                    labels.append(f'{row.gene_symbol} | {row.cell_type} | {row.control_state} → {row.target_state}'
                                  f'\npositive donors {row.n_positive_donors}/{row.n_donors_evaluable}; q={q}')
                ax.axvline(0, color='#444444', linestyle='--', linewidth=1)
                ax.set_yticks(range(len(page)), labels, fontsize=8)
                ax.set_ylim(len(page)-.45, -.65)
                ax.set_xlabel('Donor median cosine shift toward target state')
                ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3), useMathText=True)
                ax.grid(axis='x', alpha=.15)
                ax.spines[['top', 'right']].set_visible(False)
                ax.set_title(f'{tissue.title()} | {mode.upper()} | donor-level sensitivity'
                             + (' | PROVISIONAL: some inputs failed' if provisional else ''), fontsize=11, pad=12)
                legend = [Line2D([], [], marker='o', linestyle='', color='#007C66', label='BH-significant positive'),
                          Line2D([], [], marker='o', linestyle='', color='#2879B9', label='Consistent positive, not BH-significant'),
                          Line2D([], [], marker='o', linestyle='', color='#777777', label='Other / insufficient donors'),
                          Line2D([], [], marker='D', linestyle='', color='black', label='Median across evaluable donors')]
                if args.min_cells_per_gene_donor > 1:
                    legend.append(Line2D([], [], marker='x', linestyle='', color='#C8C8C8', label='Below cell-count requirement'))
                fig.legend(handles=legend, loc='lower center', ncol=2, frameon=False, fontsize=8)
                fig.tight_layout(rect=(0, .09, 1, 1))
                pdf.savefig(fig)
                fig.savefig(fig_dir / f'{tissue}_{mode}_page_{start//args.plot_rows+1:03d}.png', dpi=180)
                plt.close(fig)
                count += 1
    return count


def main():
    global hf
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sig-dir', type=Path, default=Path('/path/to/sig_only'))
    p.add_argument('--results-root', type=Path, default=Path('/path/to/geneformer'))
    p.add_argument('--output-dir', type=Path, default=Path('/path/to/donor_sensitivity'))
    p.add_argument('--max-cells', type=int, default=500, help='Original sampling cap; 0 means unlimited')
    p.add_argument('--min-cells-per-donor', type=int, default=5, help='Original sampling eligibility requirement')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-length', type=int, default=1024)
    p.add_argument('--min-cells-per-gene-donor', type=int, default=1, help='Minimum observed gene shifts to include a donor median')
    p.add_argument('--min-donors', type=int, default=2, help='Minimum evaluable donors for testing (default 2)')
    p.add_argument('--consistent-fraction', type=float, default=.80, help='Descriptive fraction positive; not a significance threshold')
    p.add_argument('--alpha', type=float, default=.05)
    p.add_argument('--plot-rows', type=int, default=24)
    args = p.parse_args()
    if (args.max_cells < 0 or args.min_cells_per_donor < 1 or args.max_length < 3 or
        args.min_cells_per_gene_donor < 1 or args.min_donors < 2 or args.plot_rows < 1 or
        not .5 < args.consistent_fraction <= 1 or not 0 < args.alpha < 1):
        p.error('Invalid numeric setting')
    import datasets as hf_module
    hf = hf_module
    out = args.output_dir / datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'settings.json').write_text(json.dumps(vars(args), indent=2, default=str)+'\n')
    print('OUTPUT DIRECTORY:', out, flush=True)
    selected, problems, nfiles = read_findings(args)
    if not selected:
        pd.DataFrame(problems).to_csv(out / 'input_issues.csv', index=False)
        raise SystemExit(f'No positive findings found. Check {out}/input_issues.csv')
    pd.DataFrame(selected).to_csv(out / 'selected_discovery_rows.csv', index=False)
    groups = defaultdict(list)
    for row in selected:
        # Normalize directory spellings so space/underscore variants are one run.
        run_key = (row['tissue'], safe_name(row['cell_type']), row['control_state'], row['target_state'], row['perturb_mode'])
        groups[run_key].append(row)
    summary_rows, donor_rows, run_rows = [], [], []
    for i, ((tissue, safe_ct, control, target, mode), items) in enumerate(sorted(groups.items()), 1):
        run_dir = args.results_root / tissue / f'{control}_{target}' / mode
        print(f'\n[{i}/{len(groups)}] {tissue} | {safe_ct} | {control} -> {target} | {mode}', flush=True)
        run_record = dict(tissue=tissue, cell_type=safe_ct, control_state=control, target_state=target,
                          perturb_mode=mode, n_discovery_rows=len(items), status='FAIL')
        try:
            validation_args = SimpleNamespace(run_dir=str(run_dir), cell_type=safe_ct,
                control=control, target=target, mode=mode, max_cells=args.max_cells,
                min_cells_per_donor=args.min_cells_per_donor, seed=args.seed,
                max_length=args.max_length, dataset=None, output_dir=out/'validation'/tissue)
            report, mapping_path = validate_run(validation_args)
            if report['status'] != 'PASS':
                raise ValueError('; '.join(report['errors']))
            mapping = pd.read_csv(mapping_path, dtype={'patient_id': str})
            tests, gene_problems = resolve_run_findings(items, run_dir / safe_ct / f'candidate_genes_{safe_ct}.csv')
            problems.extend(gene_problems)
            if not tests:
                raise ValueError('No selected genes could be mapped to original candidates')
            cells = read_selected_shifts(run_dir, safe_ct, mode, target, tests, mapping)
            stats_rows, median_rows = summarize_tests(tests, cells, mapping, args)
            summary_rows.extend(stats_rows)
            donor_rows.extend(median_rows)
            run_record.update(status='PASS' if not gene_problems else 'PARTIAL', n_findings=len(tests),
                              n_cells=len(mapping), n_donors=int(mapping.patient_id.nunique()))
        except Exception as exc:
            reason = f'{type(exc).__name__}: {exc}'
            run_record['reason'] = reason
            problems.append({'stage': 'run', **{k: run_record[k] for k in KEY}, 'reason': reason})
            print('EXCLUDED RUN:', reason, flush=True)
        run_rows.append(run_record)
    pd.DataFrame(run_rows).to_csv(out / 'run_validation_summary.csv', index=False)
    pd.DataFrame(problems if problems else [], columns=sorted(set().union(*(r.keys() for r in problems)))
                 if problems else ['stage', 'reason']).to_csv(out / 'input_issues.csv', index=False)
    if not summary_rows:
        raise SystemExit(f'No findings analyzed. See {out}/run_validation_summary.csv')
    summary = classify(pd.DataFrame(summary_rows), args)
    donors = pd.DataFrame(donor_rows, columns=KEY + ['finding_id', 'gene_symbol', 'ensembl_id', 'patient_id',
                                                   'n_cells', 'donor_median_shift', 'included_in_test'])
    provisional = bool(problems)
    summary['analysis_complete'] = not provisional
    summary = summary.sort_values(KEY + ['gene_symbol', 'ensembl_id'])
    summary.to_csv(out / 'donor_sensitivity_results.csv', index=False)
    donors.to_csv(out / 'donor_medians.csv', index=False)
    summary.loc[summary.classification.eq('BH_significant_positive')].to_csv(out / 'BH_significant_positive_findings.csv', index=False)
    summary.loc[summary.consistent_positive].to_csv(out / 'consistent_positive_findings.csv', index=False)
    counts = summary.groupby(KEY+['classification'], dropna=False).size().rename('n_findings').reset_index()
    counts.to_csv(out / 'findings_summary_by_comparison.csv', index=False)
    report = {'analysis_complete': not provisional, 'n_discovery_files': nfiles,
              'n_selected_positive_rows': len(selected), 'n_runs_requested': len(groups),
              'n_runs_failed_or_partial': sum(r['status'] != 'PASS' for r in run_rows),
              'n_input_issues': len(problems), 'n_findings_analyzed': len(summary),
              'n_findings_tested': int(summary.test_status.eq('tested').sum()),
              'n_insufficient_donors': int(summary.test_status.eq('insufficient_donors').sum()),
              'n_BH_significant_positive': int(summary.classification.eq('BH_significant_positive').sum()),
              'n_consistent_positive_including_significant': int(summary.consistent_positive.sum()),
              'n_all_donors_positive': int(summary.all_donors_positive.sum()),
              'classification_counts': summary.classification.value_counts().to_dict(),
              'BH_scope': 'All tested findings in this invocation', 'alpha': args.alpha,
              'consistent_positive_fraction': args.consistent_fraction}
    (out / 'analysis_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    pages = make_plots(summary, donors, out, args, provisional)
    print('\n'+json.dumps(report, indent=2))
    print(f'\nCreated {pages} plot pages. Results: {out}', flush=True)
    raise SystemExit(1 if provisional else 0)


if __name__ == '__main__':
    main()

