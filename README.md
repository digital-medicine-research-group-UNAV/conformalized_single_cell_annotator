# Conformalized single cell annotator


We provide a single-cell annotator based on conformal prediction for robust uncertainty quantification on annotations.

Conformal prediction provides reliable and rigorous uncertainty estimates [1]. Our Conformalized Single Cell Annotator library lets you annotate your single-cell data at various significance levels, ensuring precise and informative cell-type assignments even in noisy or complex datasets. This tool is designed to be fitted with your reference, and is also robust to out-of-distribution samples.

Further details are given in our paper: **Currently under review**.


---




## Setup and Installation

We provide a Conda ([conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)) environment file (`environment.yml`) that specifies minimum required dependencies. You can use either Mamba or Conda to create the environment (**We strongly recommend using Mamba**).


1.  **Clone the repository:**
    ```bash
    git clone https://github.com/digital-medicine-research-group-UNAV/conformalized_single_cell_annotator.git
    cd conformalized_single_cell_annotator
    ```

2.  **Create the environment**
    (This command uses the `environment.yml` file to create a new environment named `annotator-env`):
    ```bash
    mamba env create -f environment.yml
    ```
    or

    ```bash
    conda env create -f environment.yml
    ```




## Quickstart
Below is an integrated guide to getting started with the Conformalized Single Cell Annotator (`conformal_sc_annotator`) and understanding its outputs.

```python
import numpy as np
import pandas as pd
import scanpy as sc

# Import the ConformalSCAnnotator class 
from conformalSC_annotator import  ConformalSCAnnotator

# 1. Load reference and query datasets (must be in .h5ad format)
reference_adata_path = 'path_to_query/your_reference.h5ad'
adata_reference = sc.read_h5ad(reference_adata_path)

query_adata_path = 'path_to_query/your_query.h5ad'
adata_query = sc.read_h5ad(query_adata_path)

# 2. Ensure a .var column with gene names exists
# Often gene names are in adata.var_names. If not, explicitly create a new column.
# In this example, we assume that the column already exists and is named "features".

gene_names_column = "features" 
underlying_model = "torch_net" # Choose between "torch_net", "celltypist" , "scmap"

# 3. Initialize the annotator
annotator = ConformalSCAnnotator(adata_query,
                                var_query_gene_column_name = gene_names_column,
                                underlying_model = underlying_model)    


```
If "torch_net" was selected, we need to define the arquitecture of the underlying classifier.

```python

# Define que network architecture   
network_architecture:dict = {   
                            "hidden_sizes": [ 72,64,32, 32],
                            "dropout_rates": [ 0.15, 0.15, 0.15,  0.15],
                            "learning_rate": 1e-4}

```

The next step is to define the parameters of the out-of-distribution detector. If `alpha` is set to `None`, $\alpha_o$ will be automatically estimated.


```python

OOD_detector_config = {
                        "alpha": None,                              # Significance level for the hyoothesis test. 
                        "delta": 0.1,                               # Only for conditional pvalues
                        "hidden_sizes":  [50, 48, 32, 24],          # AE hidden sizes and topology of the network
                        "dropout_rates": [0.15, 0.15, 0.15, 0.15],  
                        "learning_rate": 1e-4,
                        "batch_size":    72,
                        "n_epochs":      850,
                        "patience":      9,
                        "noise_level":   0.1,
                        "lambda_sparse": 1e-3}


```

Now it's time to configure the model and the conformal predictor. It is possible to fine-tune various components such as the architecture or the conformal predictor.

```python

# 5. Configure model and conformal predictor

from torchcp.classification.score import  APS

do_test = True          # If True, a small fraction of the reference set is reserved as an independent test set.

taxonomy = "standard"       # Choose from: "standard", "mondrian", or "cluster"
cells_OOD = 50              # Exclude cell types with fewer than 50 cells (Optional, it could be an int or a list of cell types)
nc_function = APS()         # Non-conformity function compatible with torchCP
ref_column = "cell_type"    # Column name in the reference data with class labels (e.g., "cell_type", "cell_class", etc.)

annotator.configure(reference_path = adata_reference,                  # Path or AnnData object (.h5ad) for the reference dataset
                    model_architecture = network_architecture,   # Optional: user-defined model; otherwise defaults are used
                    OOD_detector = OOD_detector_config,          # Optional: specify custom OOD detector config
                    CP_predictor = taxonomy,                     
                    cell_names_column = ref_column,              # Column name in reference data with class labels 
                    cell_types_excluded_treshold = cells_OOD,    
                    test =  do_test,                             
                    alpha = [0.01, 0.05, 0.1],                   # List of confidence levels for prediction sets. Can be a single float too; e.g. alpha = 0.1
                    non_conformity_function = nc_function,       # NC-function provided by or compatible with torchCP    
                    epoch = 1000,                                # Only applicable if using "torch_net" as underlying model
                    batch_size = 72,                             # Only applicable if using "torch_net" as underlying model
                    random_state = None)                         # Random seed for reproducibility

```
Once the tool has been configured, the final step is to annotate the query dataset. Here, we specify which parts of the `AnnData` object should be used for inference and out-of-distribution detection.  

```python
# 6. Annotate the query adata 

obsm_layer_ = "obsm"         # Choose from: None (adata.X), "obsm" (adata.obsm), or "layer" (adata.layers)
layer_ = None                # Required only if obsm_layer_ is "layer" — provide the layer name to use
obsm_ = "X_pca_harmony"      # Required if obsm_layer_ is "obsm" — name of the embedding in adata.obsm

obsm_OOD_ = "X_pca_harmony"  # The embedding used by the OOD detector (typically the same as obsm_). If None, adata.X will be used.


annotator.annotate(obsm_layer = obsm_layer_,
                    obsm = obsm_,
                    layer = layer_,
                    obsm_OOD = obsm_OOD_)
```

After running the annotation, the results are stored in the query `AnnData` and the `annotator` object. You can extract predictions, test metrics, and additional metadata as follows:

```python
# 7. Access information

# Retrieve the annotated observations from the query dataset
annotated_cells = annotator.adata_query.obs                     
print("\nPredicted annotations sets: \n" , annotated_cells)

# Internal test results (if do_test = True):
test_results = annotator.test_results

# Unique labels from the reference data
unique_labels = annotator.unique_labels

# Automatically determined alpha (if alpha=None in OOD_detector_config)
alpha_OOD = annotator.alpha_OOD

```
Finally, you can extract the predicted labels and corresponding conformal prediction sets into a traditional pandas DataFrame for inspection, export, or downstream analysis.

```python

# 8. Extract annotation results into a pandas DataFrame


results = []
for pred,cp_pred_001,cp_pred_005, cp_pred_010 in zip(
        annotator.adata_query.obs["predicted_labels"],
        annotator.adata_query.obs["prediction_sets_0.01"],
        annotator.adata_query.obs["prediction_sets_0.05"],
        annotator.adata_query.obs["prediction_sets_0.1"] ):
         
    #print(f"Predicted: {pred} - CP 0.01: {cp_pred_001} - CP 0.05: {cp_pred_005} - CP 0.10: {cp_pred_010}")
        
    results.append({
        "Predicted": pred,
        "CP 0.01": cp_pred_001,
        "CP 0.05": cp_pred_005,
        "CP 0.1": cp_pred_010
    })
    

df_results = pd.DataFrame(results)
#df_results.to_csv("saved_Results.csv", index=False)  # Save to CSV if needed

df_results.head(10)

```


## References 


[1] V. Balasubramanian, S.-S. Ho, and V. Vovk, Conformal Prediction
for Reliable Machine Learning: Theory, Adaptations and Applications,
1st ed. San Francisco, CA, USA: Morgan Kaufmann Publishers Inc.

