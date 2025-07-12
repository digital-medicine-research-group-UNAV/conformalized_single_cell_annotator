

import os 
import pandas as pd
import scanpy as sc
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns

import anndata as ad 
import scanpy.external as sce
import scrublet as scr
import bbknn
import scanpy.external as sce
from scipy import sparse

import scrublet.helper_functions as hf








###### READ YOUR DATA (adapt this step for your dataset) ######

adata_ref_path = "your_ref_data_path.h5ad"  # Replace with your actual data path
adata_ref = ad.read_h5ad(adata_ref_path)

adata_query_path = "your_query_data_path.h5ad"  # Replace with your actual reference data path
adata_query = ad.read_h5ad(adata_ref_path)


########################################


adata_ref.var["features"] = adata_ref.var_names

# save raw data

adata_ref.raw = ad.AnnData(
    X = adata_ref.X.copy().astype(int),
    obs = adata_ref.obs.copy(),
    var = adata_ref.var.copy())

adata_query.raw = ad.AnnData(
    X = adata_query.X.copy().astype(int),
    obs = adata_query.obs.copy(),
    var = adata_query.var.copy())




#  Functions: QC metrics, filtering, doublet removal, normalization, HVG


eps = 1e-8
def gLog_safe(inp):
    val = inp[1] * np.exp(-inp[0]) + inp[2]
    return np.log(np.clip(val, a_min=eps, a_max=None))

def CV_input_safe(b):
    return np.sqrt(np.clip(b, a_min=eps, a_max=None))

hf.gLog      = gLog_safe
hf.CV_input  = CV_input_safe
scr.gLog     = gLog_safe
scr.CV_input = CV_input_safe


def densify(adata):
    """
    Convert adata.X to a numpy ndarray if it is a sparse matrix or np.matrix.
    """
    X = adata.X
    if sparse.issparse(X):
        adata.X = X.toarray()
    elif isinstance(X, np.matrix):
        adata.X = np.asarray(X)
    # now adata.X is guaranteed to be an ndarray
    return adata

#-------------------------
#  Auxiliary: safe doublet detection with warning suppression and NaN filtering
#-------------------------

def run_scrublet(adata):

    counts = adata.X.copy()  # keep sparse

    if hasattr(counts, 'toarray'):
        counts = counts.toarray().astype(np.float64)

    scrub = scr.Scrublet(counts)
    scores, predicted = scrub.scrub_doublets()

    # filter out NaN scores
    valid = np.isfinite(scores)
    adata = adata[valid].copy()
    adata.obs['doublet_score'] = scores[valid]
    if predicted is None or len(predicted) == 0:
        return adata
    adata.obs['predicted_doublet'] = predicted[valid]
    adata = adata[~adata.obs['predicted_doublet']].copy() # remove doublets

    return adata

def preprocess(adata,
               confounder_key=None,
               batch_key=None,
               min_genes=600,
               min_counts=1000,
               hvg_genes_sel=True,
               mt_threshold=10,
               ribo_threshold=60,
               hb_threshold=60,
               min_cells=10):
    
    
    
    # Annotate mitochondrial, ribosomal, and hemoglobin genes
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    adata.var['ribo'] = adata.var_names.str.startswith(('RPS', 'RPL'))
    adata.var['hb'] = adata.var_names.str.contains('^HB[^(P)]')

    # Calculate QC metrics including percent counts
    sc.pp.calculate_qc_metrics(adata,
                               qc_vars=['mt', 'ribo', 'hb'],
                               inplace=True,
                               percent_top=None)

    # Filter cells by QC thresholds
    qc_filter = (
        (adata.obs['n_genes_by_counts'] >= min_genes) &
        (adata.obs['total_counts'] >= min_counts) &
        (adata.obs['pct_counts_mt'] < mt_threshold) &
        (adata.obs['pct_counts_ribo'] < ribo_threshold) &
        (adata.obs['pct_counts_hb'] < hb_threshold)
    )

    adata = adata[qc_filter].copy()

    print(f"Cells after QC filtering: {adata.shape[0]}")


    # Filter genes expressed in fewer than `min_cells`
    sc.pp.filter_genes(adata, min_cells=min_cells)

    
    # Doublet removal on raw counts
    adata = run_scrublet(adata)

    adata.layers["raw_filtered"] = adata.X.copy()

    # Identify highly variable genes
    if hvg_genes_sel:
        hvg_genes = 2000
        
        try:
            sc.pp.highly_variable_genes(adata,n_top_genes=hvg_genes,layer="counts", flavor='seurat_v3', subset=True) # Use raw counts for HVG selection with Seurat v3 
        except Exception as e:
            sc.pp.highly_variable_genes(adata,n_top_genes=hvg_genes, flavor='seurat_v3', subset=True)

     

    # Normalize and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
     
    adata.layers["lognormalized"] = adata.X.copy()

    # 8. Regress out mitochondrial fraction
    sc.pp.regress_out(adata, ['pct_counts_mt'])                      # Mito regression :contentReference[oaicite:11]{index=11}

     

    # 9. Scale data and clip
    sc.pp.scale(adata, max_value=10)                                 # Scale & clip at ±10 :contentReference[oaicite:12]{index=12}

    adata.layers["scaled"] = adata.X.copy()

    adata = densify(adata)  # Convert sparse matrix to dense

    #  Remove chemistry-driven effects (Optional, maybe better afte concatenation)
    bbknn.ridge_regression(adata,batch_key=batch_key,confounder_key=confounder_key)     


    """
    # This is an optional step to ensure that all layers are numeric and in float32 format. 
    for name, mat in adata.layers.items():
        if sp.issparse(mat):
            # sparse → enforce float32 sparse
            adata.layers[name] = mat.astype(np.float32).tocsc()
        else:
            # dense → force a float32 ndarray
            adata.layers[name] = np.asarray(mat, dtype=np.float32)

    # 3. (Optional) also ensure adata.X itself is numeric
    if sp.issparse(adata.X):
        adata.X = adata.X.astype(np.float32).tocsc()
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)

    """
                                                                                                                               
                                                                                                      
    return adata



#-------------------------
#  Integration pipeline
#-------------------------


def integration_pipeline(adata_concat, query):

    adata_concat = densify(adata_concat)
    
    #  Remove chemistry-driven effects via ridge regression on combined data (Uncomment if needed)
    #bbknn.ridge_regression(adata_concat, batch_key=['Chemistry'],confounder_key=["Manually_curated_celltype"])

    ref_cell_types = adata_concat.obs.loc[adata_concat.obs['Batch'] == "ref",'colum_target_name'].unique().tolist()
    query_cell_types = adata_concat.obs.loc[adata_concat.obs['Batch'] == query,'colum_target_name'].unique().tolist()

    sc.pp.pca(adata_concat, n_comps=40)
    bbknn.bbknn(adata_concat, batch_key='Batch', use_rep='X_pca')
    sc.pp.neighbors(adata_concat, use_rep='X_pca', n_pcs=30)

    sc.tl.umap(adata_concat)
    sc.pl.umap(adata_concat,color="Batch",size=2)
    sc.pl.umap(adata_concat,color='colum_target_name', groups=ref_cell_types,size=2)
    sc.pl.umap(adata_concat,color='colum_target_name', groups=query_cell_types,size=2)

    # Apply PCA
    sc.pp.pca(adata_concat, n_comps=40)
    
    # Run Harmony
    sce.pp.harmony_integrate(adata_concat, key='Batch', basis='X_pca',
                         adjusted_basis='X_pca_harmony',
                         max_iter_harmony=50, theta=1.5)

    # 7. Build neighbor graph on Harmony embedding & final UMAP + clustering
    sc.pp.neighbors(adata_concat,use_rep='X_pca_harmony',n_pcs=40)

    # 5. UMAP and Leiden clustering
    sc.tl.umap(adata_concat)
    sc.pl.umap(adata_concat,color="Batch",size=2)
    sc.pl.umap(adata_concat,color='colum_target_name', groups=ref_cell_types,size=2)
    sc.pl.umap(adata_concat,color='colum_target_name', groups=query_cell_types,size=2)

    return adata_concat




adata_ref = preprocess(adata_ref,
                       hvg_genes_sel=True,
                       batch_key=["batch_1_column", "batch_2_column",...],  # name of the columns in adata.obs that contain batch information (if existing)
                       confounder_key=['colum_target_name'])  # name of the columns in adata.obs that contain confounder information

adata_que = preprocess(adata_query,
                           hvg_genes_sel=False,
                           batch_key=["batch_1_column", "batch_2_column",...],           # name of the columns in adata.obs that contain batch information (if existing)
                           confounder_key=['colum_target_name'])    # name of the columns in adata.obs that contain confounder information

adata_ref.write('your_save_path.h5ad')
adata_que.write('your_save_path.h5ad')



print(f"Reference: {adata_ref.n_obs} cells × {adata_ref.n_vars} genes")
print(f"Query  : {adata_que.n_obs} cells × {adata_que.n_vars} genes")


genes_ref = set(adata_ref.var_names)
genes_q = set(adata_que.var_names)

shared_genes = list(genes_ref & genes_q )


adata_ref_m = adata_ref[:, shared_genes].copy()
adata_myeloid = adata_que[:, shared_genes].copy()


print(f"Reference: {adata_ref.n_obs} cells × {adata_ref.n_vars} genes")
print(f"Query  : {adata_que.n_obs} cells × {adata_que.n_vars} genes")



## Concatenate the reference and query datasets

adata_concat = adata_ref_m.concatenate(
    [adata_myeloid],
    batch_key='Batch',
    batch_categories=['ref', 'query'],
    join='outer',
    index_unique=None)


# Run the integration pipeline

adata_concat = integration_pipeline(adata_concat, query='query')

adata_concat.write('save_path.h5ad')


