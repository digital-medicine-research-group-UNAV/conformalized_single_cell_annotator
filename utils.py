

import os
import anndata as ad
import logging
from contextlib import redirect_stdout, redirect_stderr
import torch
import torch.nn as nn
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
import numpy as np
from celltypist import train, annotate, models
import scanpy as sc
sc.settings.verbosity = 0
os.environ.setdefault('TQDM_DISABLE', '1')

# Silence CellTypist’s own logger
logging.getLogger('celltypist').setLevel(logging.ERROR)
# If other libraries (e.g., scikit-learn) are verbose, silence them similarly:
logging.getLogger('sklearn').setLevel(logging.ERROR)




class ScmapWrapper(nn.Module):
    """
    PyTorch wrapper around scmap sc annotator
    """
    def __init__(
        self,
        n_jobs = -1,
        ct_model = None,
        eps: float = 1e-8,
        cal_adata_genes = None
    ):
        
        super().__init__()

        # ─── Dummy param for nn.mdolue parameters
        self._dummy = nn.Parameter(torch.empty(0))

        # ─── Training/config state ─────────────────────────
        self.ct = ct_model
        self.eps = eps
        self.n_jobs = n_jobs
        self.counter = 0
        self.adata_ref = None
        self.cal_adata_genes =  pd.DataFrame(index=cal_adata_genes)


    # this function constitutes the main logic of the scmap annotator with options "cell" and "random" and "cosine"
    def scmap_annotate_(self, target_data, 
            reference_adata, 
            key_genes, 
            key_annotations,
            n_genes_selected=1000,
            metrics=["cosine"],
            similarity_threshold:float =.7,
            k:int =3,
            unassigned_label:str ="Unassigned"):
        
        """
        Annotate `target_data` using scmap-style KNN and return annotation, probabilities, and logits.

        If `inplace`, adds:
        - annotations to `target_data.obs[key_added]`
        - probabilities to `target_data.obsm[probs_key]` (DataFrame of shape n_cells × n_classes)
        - logits to `target_data.uns[logits_key]` (DataFrame of shape n_cells × n_classes)

        Otherwise returns `(annotation, probs_df, logits_df)`.
        """

        # Determine gene list
        if isinstance(key_genes, (list, np.ndarray, pd.Index)):
            gene_list = np.array(key_genes)
        else:
            if key_genes not in reference_adata.var.columns:
                raise KeyError(f"'{key_genes}' must be a column in reference .var or a list of genes")
            gene_list = reference_adata.var[key_genes].values
        # Ensure target var has gene identifiers if key_genes was list
        if not isinstance(key_genes, (list, np.ndarray, pd.Index)):
            if key_genes not in target_data.var.columns:
                raise KeyError(f"'{key_genes}' must be a column in target .var")
            # use target_data.var[key_genes] to match gene_list
            target_gene_ids = target_data.var[key_genes].values
            ref_gene_ids = reference_adata.var[key_genes].values
        else:
            target_gene_ids = target_data.var_names.values
            ref_gene_ids = reference_adata.var_names.values

        # Subsample genes if needed
        total = len(gene_list)
        n = min(n_genes_selected, total)
        rng = np.random.default_rng(42)
        subset = rng.choice(gene_list, size=n, replace=False)

        # Map to indices
        idx_tar = np.nonzero(np.isin(target_gene_ids, subset))[0]
        idx_ref = np.nonzero(np.isin(ref_gene_ids, subset))[0]

        X_tar = target_data.X[:, idx_tar]
        X_ref = reference_adata.X[:, idx_ref]

        # Dense conversion
        if sparse.issparse(X_ref): X_ref = X_ref.toarray()
        if sparse.issparse(X_tar): X_tar = X_tar.toarray()

        # Fit neighbors
        nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
        nn.fit(X_ref)
        dists, neighbors = nn.kneighbors(X_tar)
        

        labels_ref = reference_adata.obs[key_annotations].astype(str).values
        unique_labels = np.unique(labels_ref)
        n_cells = X_tar.shape[0]
        n_classes = len(unique_labels)

        # Count neighbor votes
        counts = np.zeros((n_cells, n_classes), dtype=int)
        for j in range(k):
            neigh_labels = labels_ref[neighbors[:, j]]
            for idx, label in enumerate(unique_labels):
                counts[:, idx] += (neigh_labels == label)

        

        # Compute probabilities
        probs = counts / k
        # Compute logits (log odds): log(p / (1 - p))
        eps = np.finfo(float).eps
        logits = np.log(probs / (1 - probs + eps) + eps)

        # Assign annotation
        ann = unique_labels[np.argmax(probs, axis=1)]
        ann[np.max(probs, axis=1) < similarity_threshold] = unassigned_label
        ann[dists[:, 0] > similarity_threshold] = unassigned_label

        # Wrap in DataFrames
        #probs_df = pd.DataFrame(probs, index=target_data.obs_names, columns=unique_labels)
        logits_df = pd.DataFrame(logits, index=target_data.obs_names, columns=unique_labels)

        return logits_df
        

    def train_model(self, adata_ref, label_key, layer=None, obsm=None):
        
        if not self.training:
            raise RuntimeError("CellTypist model should be trained in train mode.")
        
        self.adata_ref = adata_ref.copy()
        if layer is not None:
            self.adata_ref.X = self.adata_ref.layers[layer].copy()
            

        #if obsm is not None:
        #    self.adata_ref.X = self.adata_ref.obsm[obsm].copy()
        #    self.gene_names = adata_ref.var_names

        #if layer is None and obsm is None:
        #    self.gene_names = adata_ref.var_names
    
        
        self.gene_names = adata_ref.var_names
        self.column_to_predict = label_key
        
        print("scmap is ready to predict.")
        return self


    def get_probs(self, X: torch.Tensor) -> torch.Tensor:
        
        """
        Get raw per-cell probabilities [N_cells × N_types].
        """
        # Move to CPU NumPy for CellTypist
        X_np = X.detach().cpu().numpy()

        adata_query = ad.AnnData(X_np)
        adata_query.var_names = self.cal_adata_genes.index
        n_genes_selected =  self.adata_ref.shape[1]

        logits_df = self.scmap_annotate_(
                adata_query,
                self.adata_ref,
                self.gene_names,
                self.column_to_predict,
                n_genes_selected=n_genes_selected,
                similarity_threshold=0.7)    
                
        
        probs_np = logits_df.values

        return torch.from_numpy(probs_np).to(self._dummy.device)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        probs = self.get_probs(X)
        return probs

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        probs = self.get_probs(X)
        return probs.argmax(dim=1)
    
    def train(self, mode=True):
        
        super().train(mode)   

    def eval(self):
        self.train(False)

    def get_device(self):
        return self._dummy.device
    
    def _apply(self, fn):
        super()._apply(fn)
        return self






class CellTypistWrapper(nn.Module):
    """
    Wraps CellTypist for training custom references and
    producing logits for external calibration.
    """
    def __init__(
        self,
        n_jobs = -1,
        ct_model: models.Model = None,
        eps: float = 1e-8,
        cal_adata_genes = None
    ):
        
        super().__init__()

        # ─── Dummy param for nn.mdolue parameters
        self._dummy = nn.Parameter(torch.empty(0))

        # ─── Training/config state ─────────────────────────
        self.ct = ct_model
        self.eps = eps
        self.n_jobs = n_jobs
        self.counter = 0
        self.cal_adata_genes =  pd.DataFrame(index=cal_adata_genes) 


    def train_model(self, adata_ref, label_key, layer):
       
        if not self.training:
            raise RuntimeError("CellTypist model should be trained in train mode.")
        
        if layer is not None:
            raw_counts = adata_ref.layers[layer].copy()
            adata_raw = ad.AnnData(
                X=raw_counts,
                obs=adata_ref.obs.copy(),
                var=adata_ref.var.copy()
            )
            adata_ref.raw = adata_raw
            

        self.ct = train(
            adata_ref,
            labels=label_key,
            n_jobs=self.n_jobs,
            check_expression = True)
        print("CellTypist model trained successfully.")
        return self

    def get_probs(self, X: torch.Tensor) -> torch.Tensor:
        
        """
        Get raw per-cell probabilities [N_cells × N_types].
        """

        nan_mask = torch.isnan(X)
        # 2. Check if there are any NaNs at all
        if nan_mask.any():
            # 3. Find which rows contain at least one NaN
            rows_with_nans = nan_mask.any(dim=1).nonzero(as_tuple=False).squeeze()
            print(f"Found NaNs in {rows_with_nans.numel()} rows: {rows_with_nans.tolist()}")

            # 4. (Optional) For each such row, list which columns are NaN
            for r in rows_with_nans.tolist():
                cols = nan_mask[r].nonzero(as_tuple=False).squeeze()
                print(f" Row {r} has NaNs in columns: {cols.tolist()}")
        else:
            print("No NaNs found in X.")


        # Move to CPU NumPy for CellTypist
        X_np = X.detach().cpu().numpy()

        adata_cal = ad.AnnData(X_np)
        adata_cal.var_names = self.cal_adata_genes.index

        #df = pd.DataFrame(X_np, columns=self.ct.features)
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):

                ann = annotate(
                    adata_cal,
                    model=self.ct,
                    majority_voting=False)

        self.counter += 1
        if self.counter % 100 == 0:
            print(f"Processed {self.counter} batches.")     
                
        probs_df = ann.probability_matrix
        probs_np = probs_df.values

        return torch.from_numpy(probs_np).to(self._dummy.device)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        
        probs = self.get_probs(X).clamp(min=self.eps)
        return torch.log(probs)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Hard‐label predictions (highest‐probability index).
        """
        probs = self.get_probs(X)
        return probs.argmax(dim=1)
    
    def train(self, mode=True):        
        super().train(mode)   

    def eval(self):
        self.train(False)

    def get_device(self):
        return self._dummy.device
    
    def _apply(self, fn):
        super()._apply(fn)
        return self

  