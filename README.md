# scCM
### A contrastive learning model for analyzing large-scale CNS scRNA-seq data

# Getting started
#### Before diving into the powerful capabilities of scCM, you need to prepare your scRNA-seq data. Follow these steps to get started:
## Download and Convert Data
#### The initial step requires you to download your scRNA-seq data and convert it into the h5ad file format. This format is essential for scCM to work seamlessly with your data.

# Basic Usage (details see the demo)
#### 	Once your scRNA-seq data is ready in the h5ad format, you can utilize ScCM for comprehensive analysis. Below, we outline the basic usage of scCM, with additional details provided in the demo_Frounier section.

	%run ScCM.py \
	--input_h5ad_path={input_path} 

#### Running the command above initiates ScCM's analysis on your data, and it will generate trained embeddings that are saved in a newly created 'results' folder. These embeddings are crucial for downstream analysis. Parameters such as learning rate and batch size can be adjusted as needed.
### Following are parameters that can be adjusted
	workers=1, epochs=10, start_epoch=0, batch_size=4096, lr=1e-05, momentum=0.9, weight_decay=1e-06, schedule=[100, 120], low_dim=128, pcl_r=1024, moco_dim=128, moco_mlp_dim=512, moco_m=0.999, moco_m_cos=False, moco_t=0.8, cos=False, warmup_epoch=5, aug_prob=0.0, cluster_name='kmeans', num_cluster=-1, seed=0, gpu=1, eval_freq=1, log_freq=10, exp_dir='./experiment_pcl', save_dir='./result', warmup_epochs=10

## Visualizing Results with UMAP
#### To visualize your data and gain insights into cell clustering, you can follow these steps:

#### 1. Load the trained embeddings:

	count_csv_path = f"./results/scCM/feature_scCM_{name}.csv"

	count_frame = pd.read_csv(count_csv_path, index_col=None, header=None, )

#### This embedding can be used for UMAP visualization to cluster cells.

#### 2. Create an AnnData object and perform PCA:

	adata = sc.AnnData(X=count_frame)

	sc.tl.pca(adata, svd_solver='arpack')

	sc.pl.pca_variance_ratio(adata) 

	sc.pp.neighbors(adata) 

#### 3. Apply UMAP for visualizing high-dimensional data in a 2D space:

	sc.tl.umap(adata) 

### Combining Results with Original Data
#### The trained visualization results can be combined with the original h5ad file's obsm (observation-specific features) for downstream analysis. This integration allows you to correlate the clustering results with the biological characteristics of your cells. With ScCM's capabilities, you can effectively analyze, visualize, and interpret large-scale CNS scRNA-seq data, providing valuable insights into the cellular composition and heterogeneity within CNS tissues. By following these steps, you can harness the full potential of ScCM in your research endeavors."

## Contributing
##### If you'd like to contribute, or have any suggestions for these guidelines, you can contact us at farry92@163.com.

