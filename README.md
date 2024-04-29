# scCM
### A contrastive learning model for analyzing large-scale CNS scRNA-seq data
#### scCM is constructed based on a momentum contrastive learning framework (MoCo v3 [1]) to learn informative representations of CNS scRNA-seq data. It comprises three modules: an encoder, a momentum encoder, and a predictor head, all constructed using fully connected neural networks. The encoder and momentum encoder receive a pair of gene expression vectors as input. Cells in the same type cluster together as closely as possible, while cells in different types separate as far as possible. After being trained, the embedding vectors produced by the Encoder can be regarded as representations of CNS cells that can be utilized for downstream tasks. 
### Dependent packages
##### (anndata 0.10.6, numpy 1.26.4, pandas 2.2.1, python 3.9, scanpy 1.9.8, scikit-learn 1.4.1, torch 2.2.1)
#
#
## 1. Getting Started
#### Before diving into the powerful capabilities of scCM, you need to prepare your scRNA-seq data. Follow these steps to get started:
### 1.1 Download Data

	wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE214nnn/GSE214979/suppl/GSE214979_filtered_feature_bc_matrix.h5 -O ./application/adat_anderson.h5

### 1.2 Convert Data into H5AD File
#### The initial step requires you to download your scRNA-seq data and convert it into the h5ad file format. This format is essential for scCM to work seamlessly with your data.

	pip install scanpy
 
#### You can use Scanpy for converting and preprocessing.
 	
	import scanpy as sc
	ad = sc.read("adat_anderson.h5") 
 	ad.write("adat_anderson.h5ad")
	
## 2. Basic Usage (details see the demo)
#### Once your scRNA-seq data is ready in the h5ad format, you can utilize ScCM for comprehensive analysis. Below, we outline the basic usage of scCM, with additional details provided in the demo_Frounier section.
### 2.1 Apply scCM

	%run ScCM.py \
	--input_h5ad_path={input_path} \
 	--epochs 10 --lr 0.00001 \

#### Running the command above initiates ScCM's analysis on your data, and it will generate trained embeddings that are saved in a newly created 'results' folder. These embeddings are crucial for downstream analysis. Parameters such as learning rate and batch size can be adjusted as needed.
#### Following are parameters that can be adjusted
	workers=1, epochs=10, start_epoch=0, batch_size=4096, lr=1e-05, momentum=0.9, weight_decay=1e-06, schedule=[100, 120], low_dim=128, pcl_r=1024, moco_dim=128, moco_mlp_dim=512, moco_m=0.999, moco_m_cos=False, moco_t=0.8, cos=False, warmup_epoch=5, aug_prob=0.0, cluster_name='kmeans', num_cluster=-1, seed=0, gpu=1, eval_freq=1, log_freq=10, exp_dir='./experiment_pcl', save_dir='./result', warmup_epochs=10
#### You can run python ScCM.py -h for more information.

### 2.2 Visualizing Results with UMAP
#### Output files are saved in ./result/scCM, including embeddings (feature.csv). To visualize your data and gain insights into cell clustering, you can follow these steps:

#### (1) Load the trained embeddings:

	count_csv_path = f"./results/scCM/feature_scCM_{name}.csv"
	count_frame = pd.read_csv(count_csv_path, index_col=None, header=None, )

#### This embedding can be used for UMAP visualization to cluster cells.

#### (2) Create an AnnData object and perform PCA:

	adata = sc.AnnData(X=count_frame)
	sc.tl.pca(adata, svd_solver='arpack')
	sc.pl.pca_variance_ratio(adata) 
	sc.pp.neighbors(adata) 

#### (3) Apply UMAP for visualizing high-dimensional data in a 2D space:

	sc.tl.umap(adata) 

##### You can then read the embeddings with R and incorperate it to Seurat for computing the neighborhood graph and following clustering.

### Combining Results with Original Data
#### The trained visualization results can be combined with the original h5ad file's obsm (observation-specific features) for downstream analysis. This integration allows you to correlate the clustering results with the biological characteristics of your cells. With ScCM's capabilities, you can effectively analyze, visualize, and interpret large-scale CNS scRNA-seq data, providing valuable insights into the cellular composition and heterogeneity within CNS tissues. By following these steps, you can harness the full potential of ScCM in your research endeavors."

### References
##### [1] Xinlei Chen, Saining Xie, He K. An Empirical Study of Training Self-Supervised Vision Transformers, arXiv preprint arXiv:2104.02057 2021



