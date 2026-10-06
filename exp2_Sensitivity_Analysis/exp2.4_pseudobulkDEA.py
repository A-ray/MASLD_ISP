from math import nan

import numpy as np
import scanpy as sc
import decoupler as dc
import anndata as ad
import os
import pandas as pd

director = 'path\\to\\output\\directory'  # Replace with your desired output directory
os.chdir(director)

sc.set_figure_params(figsize=(3,3), frameon=False)

import glob
listr = glob.glob(r"*.h5ad",recursive=False)

for i in listr:
    celltype = i.split(".")[0]
    adata = ad.read_h5ad(i)
    adata.obs_names_make_unique()
    adata.X = adata.layers["raw_counts"]
    pdata = dc.pp.pseudobulk(
        adata=adata,
        sample_col="patient_id",
        groups_col="typist_immune_majority_voting",
        mode="sum"
    )
    pdata.obs_names_make_unique()
    # dc.pl.filter_samples(
    #     adata=pdata,
    #     groupby=["broad_condition", "patient_id", "typist_liver_majority_voting"],
    #     min_cells=10,
    #     min_counts=1000,
    #     figsize=(5, 8),
    # )
    dc.pp.filter_samples(pdata, min_cells=10, min_counts=1000)
    #dc.pl.obsbar(adata=pdata, y="typist_liver_majority_voting", hue="broad_condition", figsize=(6, 3))
    # Store raw counts in layers
    pdata.layers["counts"] = pdata.X.copy()

    # Normalize, scale and compute pca
    sc.pp.normalize_total(pdata, target_sum=1e4)
    sc.pp.log1p(pdata)
    sc.pp.scale(pdata, max_value=10)
    sc.tl.pca(pdata)

    # Return raw counts to X
    dc.pp.swap_layer(adata=pdata, key="counts", inplace=True)
    #dc.tl.rankby_obsm(pdata, key="X_pca")
    # sc.pl.pca_variance_ratio(pdata)
    # dc.pl.obsm(adata=pdata, return_fig=True, nvar=5, titles=["PC scores", "Adjusted p-values"], figsize=(10, 5))
    # sc.pl.pca(
    #     adata=pdata,
    #     color=["broad_condition","patient_id"],
    #     ncols=1,
    #     size=300,
    #     frameon=True,
    # )
    # dc.pl.filter_by_expr(
    #     adata=pdata,
    #     group="broad_condition",
    #     min_count=10,
    #     min_total_count=15,
    #     large_n=10,
    #     min_prop=0.7,
    # )
    # dc.pl.filter_by_prop(
    #     adata=pdata,
    #     min_prop=0.1,
    #     min_smpls=2,
    # )
    dc.pp.filter_by_expr(
        adata=pdata,
        group="broad_condition",
        min_count=10,
        min_total_count=15,
        large_n=10,
        min_prop=0.7,
    )
    dc.pp.filter_by_prop(
        adata=pdata,
        min_prop=0.1,
        min_smpls=2,
    )
    pdata
    pdata = pdata[pdata.obs["broad_condition"].notna()]
    # Import DESeq2
    from pydeseq2.dds import DeseqDataSet, DefaultInference
    from pydeseq2.ds import DeseqStats

    # Build DESeq2 object
    inference = DefaultInference(n_cpus=8)
    dds = DeseqDataSet(
        adata=pdata,
        design="~ broad_condition",
        refit_cooks=True,
        inference=inference,
    )

    # Compute LFCs
    dds.deseq2()

    # Extract contrast between conditions
    listo = list(dds.obs.broad_condition.values.categories)
    if 'Healthy' in listo and 'MASLD' in listo:
        stat_res = DeseqStats(dds, contrast=["broad_condition", "Healthy", "MASLD"], inference=inference)

        # Compute Wald test
        stat_res.summary()

        results_df = stat_res.results_df
        results_df
        filer = director + '\\' + celltype + 'Healthy_MASLD.csv'
        results_df.to_csv(filer)
        filer = director + '\\' + celltype + 'Healthy_MASLD_volcano.png'
        dc.pl.volcano(results_df, x="log2FoldChange", y="pvalue", save=filer)
        data = results_df[["stat"]].T.rename(index={"stat": "Healthy.vs.MASLD"})
        data


    if 'Healthy' in listo and 'MASH' in listo:
        stat_res = DeseqStats(dds, contrast=["broad_condition", "Healthy", "MASH"], inference=inference)

        # Compute Wald test
        stat_res.summary()

        results_df = stat_res.results_df
        results_df
        filer = director + '\\' + celltype + 'Healthy_MASH.csv'
        results_df.to_csv(filer)
        filer = director + '\\' + celltype + 'Healthy_MASH_volcano.png'
        dc.pl.volcano(results_df, x="log2FoldChange", y="pvalue", save=filer)
        data = results_df[["stat"]].T.rename(index={"stat": "Healthy.vs.MASH"})
        data


    if 'MASH' in listo and 'MASLD' in listo:
        stat_res = DeseqStats(dds, contrast=["broad_condition", "MASLD", "MASH"], inference=inference)

        # Compute Wald test
        stat_res.summary()

        results_df = stat_res.results_df
        results_df
        filer = director + '\\' + celltype + 'MASLD_MASH.csv'
        results_df.to_csv(filer)
        filer = director + '\\' + celltype + 'MASLD_MASH_volcano.png'
        dc.pl.volcano(results_df, x="log2FoldChange", y="pvalue", save=filer)
        data = results_df[["stat"]].T.rename(index={"stat": "MASLD.vs.MASH"})
        data





