# ScCM
### A contrastive learning model for analyzing large-scale CNS scRNA-seq data
# Getting started
### In the first step, a scRNA-seq is required to be downloaded and converted to the h5ad file format.
# Basic Usage (details see the demo)

%run ScCM.py \
--input_h5ad_path={input_path} 

### The trained embedding will be generated in the newly created "results" folder.
count_csv_path = f"./results/scCM/feature_scCM_{name}.csv"

### This embedding can be used for UMAP visualization to cluster cells.

count_frame = pd.read_csv(count_csv_path, index_col=None, header=None, )
adata = sc.AnnData(X=count_frame)
sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca_variance_ratio(adata) 
sc.pp.neighbors(adata) 
sc.tl.umap(adata) 

### The trained visualization results can be combined with the original h5ad file's obsm for downstream analysis.
