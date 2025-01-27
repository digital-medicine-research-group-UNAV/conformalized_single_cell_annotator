# Conformalized single cell annotator


We provide a single-cell annotator based on conformal prediction for robust uncertainty quantification on annotations.

Conformal prediction provides reliable and rigorous uncertainty estimates [1]. Our Conformalized Single Cell Annotator library lets you annotate your single-cell data at various significance levels, ensuring precise and informative cell-type assignments even in noisy or complex datasets. This tool is designed to be fitted with your reference, and is also robust to out-of-distribution samples.

---

## Features

- **Conformal prediction for single-cell**: Obtain a set of likely cell types for each cell, along with user-defined confidence levels.  
- **Out-of-distribution detection**: Identify those cells not present in the reference data
- **Plug-and-play model usage**: Easily select your reference (`HumanLung_TopMarkersFC_level3`, etc.) for annotation.  
- **Quality control**: Optional data preprocessing ensures your input is in top shape before annotation.  
- **Batch correction**:  `harmony`-based correction to handle technical or batch effects.    
- **Versatile outputs**: Simple access to annotation results, conformal prediction sets, and metrics.

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

# 1. Load your query data
query_data_path = 'path_to_query/your_query.h5ad'
adata_query = sc.read_h5ad(query_data_path)

# 2. Extract needed arrays and metadata
X = adata_query.X.astype(np.float32)               # data matrix (cells x genes)
var_query_list = adata_query.var["features"].tolist()
obs_query = adata_query.obs                         # not mandatory, but can be used for label tracking

# 3. Initialize the annotator
annotator = ConformalSCAnnotator(
    X=X, 
    var_genes=var_query_list, 
    obs_metadata=obs_query  # Optional
)

# 4. (Optional) Quality control
annotator.quality_control()  # If your data is already preprocessed, skip this step.

```
Now we need to define the arquitecture of the Neural network.

```python

# Define que network architecture   
network_architecture:dict = {   
            "hidden_sizes": [128, 128, 64, 64],
            "dropout_rates": [0.4, 0.3, 0.4, 0.25],
            "learning_rate": 0.0001}

```
And the parameters of out out-of-distribution detector.

```python

OOD_detector_config = {
            "alpha": 0.1,
            "delta": 0.1}

```

Let´s load our reference and configurate our tool.

```python

reference_data_path = "path_to_reference/your_reference.h5ad"     # Path to the reference data


# 5. Configure model and conformal predictor
annotator.configure(reference_path = reference_data_path,
                    model_architecture = network_architecture,   # Optional, if not provided, defaul values will be used
                    OOD_detector = OOD_detector_config,          # Optional, if not provided, default values will be used
                    CP_predictor = "cluster",                    # mondrian or cluster
                    cell_type_level = "celltype_level3",         # class name for fitting the model.  
                    cell_types_excluded_treshold = 50,           # Exclude cell types with less than 50 cells
                    test = True,                                 # Perform internal test of the model
                    alpha = [0.01, 0.05, 0.1],                   # Confidence of the predictions
                    non_conformity_function = APS(),             # NC-function provided by or compatible with torchCP    
                    epoch=21,
                    batch_size = 525)

```
Finally, we only need to annotate the query dataset

```python
# 6. Annotate your data (with batch correction)
annotator.annotate(batch_correction="harmony")  # Options: "harmony" or "None"(default) if the data is already integrated 


# 7. View predicted annotations
print("\nPredicted annotations sets:\n", annotator.adata_query.obs)

# 8. Compare to ground truth (if available)
ground_truth_labels_list = obs_query["cell_type"].tolist()
annotator.recover_original_cells(ground_truth_labels_list, similarity_threshold=70)

y_true = annotator._mapped_original_ground_truth_labels

# 9. Inspect and save results
results = []
for pred, cp_pred_001, cp_pred_005, cp_pred_010, true, o_g_t in zip(
    annotator.adata_query.obs["predicted_labels"],
    annotator.adata_query.obs["prediction_sets_0.01"],
    annotator.adata_query.obs["prediction_sets_0.05"],
    annotator.adata_query.obs["prediction_sets_0.1"],
    y_true,
    ground_truth_labels_list
):
    print(f"Predicted: {pred} | CP 0.01: {cp_pred_001} | CP 0.05: {cp_pred_005} | "
          f"CP 0.10: {cp_pred_010} | True: {true} | Original Subtype: {o_g_t}")
    
    results.append({
        "Predicted": pred,
        "CP 0.01": cp_pred_001,
        "CP 0.05": cp_pred_005,
        "CP 0.1": cp_pred_010,
        "True": true,
        "Original_Cell_Subtype": o_g_t
    })

df_results = pd.DataFrame(results)
df_results.to_csv("saves/results_immune.csv", index=False)

```


## Interpreting the Output

- **`predicted_labels`**: The single best label predicted by the model.  
- **`prediction_sets_alpha`**: The conformal prediction set for each significance level (\(\alpha\)). A smaller \(\alpha\) typically means a higher confidence (and potentially smaller sets).  
- **`_mapped_original_ground_truth_labels`**: If you provided ground truth labels, these are mapped to the model’s label space for easy comparison.  
- **`results_immune.csv`**: Example CSV file containing predicted labels, conformal prediction sets, and ground truth side-by-side.


## References 


[1] V. Balasubramanian, S.-S. Ho, and V. Vovk, Conformal Prediction
for Reliable Machine Learning: Theory, Adaptations and Applications,
1st ed. San Francisco, CA, USA: Morgan Kaufmann Publishers Inc.

