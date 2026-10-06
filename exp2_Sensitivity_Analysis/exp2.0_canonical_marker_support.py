#CANONICAL EXPRESSION MARKERS
import numpy as np
import scanpy as sc
import decoupler as dc
import anndata as ad
import os
import pandas as pd

director = 'path\\to\\your\\directory'
os.chdir(director)

import glob
listr = glob.glob(r"*.h5ad",recursive=False)
markers = pd.read_csv('immune_canon_markers.csv')

adata = ad.read_h5ad(listr[0])
for i in listr[1:]:
    adata2 = ad.read_h5ad(i)
    adata = ad.concat([adata,adata2])
dc.mt.ora(
    data=adata,
    net=markers.rename(columns={'cell_type': 'source', 'genesymbol': 'target'}),
    tmin=3,
    verbose=True,
    raw=False
)
adata.obsm['score_ora']
acts = dc.pp.get_obsm(adata, 'score_ora')

acts_v = acts.X.ravel()
max_e = np.nanmax(acts_v[np.isfinite(acts_v)])
maxy = float(max_e)
if (acts.X[~np.isfinite(acts.X)].size > 0):
    acts.X[~np.isfinite(acts.X)] = maxy
acts


df = dc.tl.rankby_group(acts, groupby="typist_immune_majority_voting")
df.to_csv('immune_canonical_marker_ranks_comparison_celltypist.csv')
n_ctypes = 3
ctypes_dict = df.groupby('group').head(n_ctypes).groupby('group')['name'].apply(lambda x: list(x)).to_dict()
ctypes_dict
sc.pl.matrixplot(acts, ctypes_dict, 'typist_immune_majority_voting', dendrogram=True, standard_scale='var',
                 colorbar_title='Z-scaled scores', cmap='RdBu_r',save='immune_canonicalmarker_celltypist_matrixplot.png')
annotation_dict = df.groupby('group').head(1).set_index('group')['name'].to_dict()
annotation_dict
adata.obs['cell_type'] = [annotation_dict[clust] for clust in adata.obs['typist_immune_majority_voting']]


sc.pl.pca(
     adata=adata,
     color=["typist_immune_majority_voting","cell_type"],
     ncols=1,
     size=300,
     frameon=True,
    legend_loc='right margin',
    save='pca_immune_canonicalmarkers_celltypist.png'
 )
