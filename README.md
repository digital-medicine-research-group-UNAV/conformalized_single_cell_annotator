# Conformalized single cell annotator


We provide a single-cell annotator based on conformal prediction for robust uncertainty quantification on annotations.

Conformal prediction provides reliable and rigorous uncertainty estimates [1]. Our Conformalized Single Cell Annotator library lets you annotate your single-cell data at various significance levels, ensuring precise and informative cell-type assignments even in noisy or complex datasets. This tool is designed to be fitted with your reference, and is also robust to out-of-distribution samples.


---



## Requirements

Python 3.7 +
Scikit-learnt 1.2.2+
Numpy <2.0.0
Pytorch
TorchCP
torchvision
transformers
Scanpy



## Quickstart
Below is an integrated guide to getting started with the Conformalized Single Cell Annotator and understanding its outputs.





```python
import numpy as np
import pandas as pd
import scanpy as sc

# Import the ConformalSCAnnotator from wherever it lives in your package
from conformal_sc_annotator import ConformalSCAnnotator

# 1. Load your query data using scanpy (.h5ad files required)
query_data_path = 'path_to_query/your_query.h5ad'
adata_query = sc.read_h5ad(query_data_path)

# 2. We need a .var column that contains the gene names (if not created).
## Sometimes this information is on index column adata_query.var_names, but we explicity in a new column if not exist .
## In this case, we suppose that the column is already created and named: "features".

gene_names_column = "features" 

# 3. Initialize the annotator
annotator = ConformalSCAnnotator(
    adata_query,
    var_query_gene_column_name = gene_names_column 
)




```
Now we need to define the arquitecture of the underlying classifier.

```python

# Define que network architecture   
network_architecture:dict = {   
            "hidden_sizes": [128, 128, 64, 64],
            "dropout_rates": [0.4, 0.3, 0.4, 0.25],
            "learning_rate": 0.0001}

```
And the parameters of out out-of-distribution detector.

```python

OOD_detector_config = { "pvalues": "marginal",             # choose between marginal or conditional. Def: "marginal"
                        "alpha": 0.1,                      # Significance level for the hyoothesis test
                        "delta": 0.1,                      # only for conditional pvalues
                        "hidden_sizes": [ 556,  124],      # AE hidden sizes and topology of the network
                        "dropout_rates": [ 0.3,  0.30],
                        "learning_rate": 0.0001,
                        "batch_size": 42,
                        "n_epochs": 200}

```

Let´s load our reference and configurate our tool.

```python

reference_data_path = "path_to_reference/your_reference.h5ad"     # Path to the reference data


# 5. Configure model and conformal predictor
annotator.configure(reference_path = reference_data_path,        # Path to the reference data in format .h5ad
                    model_architecture = network_architecture,   # Optional, if not provided, default values will be used
                    OOD_detector = OOD_detector_config,          # Optional, if not provided, default values will be used
                    CP_predictor = "standard",                   # standard, mondrian or cluster
                    cell_names_column = "celltype",       # class name for fitting the model.  cell_type or celltype_level3 
                    cell_types_excluded_treshold = 45,           # Exclude cell types with less than 50 cells
                    test =  True,                                # Perform internal test of the model
                    alpha = [0.01, 0.05, 0.1],                   # Confidence of the predictions (can be a single element)
                    non_conformity_function = APS(),             # NC-function provided by or compatible with torchCP   (APS, RAPS, THR) 
                    epoch=200,
                    batch_size = 42,
                    random_state = None) 

```
Finally, we only need to annotate the query dataset

```python
# 6. Annotate your data (with batch correction)
# If batch corrected data is stored at .obsm, it can be used .
annotator.annotate(batch_correction="X_pca_harmony")  # batch_correction: None, "X_pca_harmony" or "'X_pca"
```

```python
# 7. View predicted annotations
annotated_cells = annotator.adata_query.obs
print("\nPredicted annotations sets: \n" , annotated_cells)

# And the results of the internal test:

test_results = annotator.test_results

```

```python

# 8. We can get the results from the adata object and store in a classical df:
# predicted labels sntands for the predictions of the underlying model without conformal prediction.

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

