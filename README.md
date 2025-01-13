# Conformalized single cell annotator


We provide a single-cell annotator based on conformal prediction for robust uncertainty quantification on annotations.

Conformal prediction provides reliable and rigorous uncertainty estimates. Our Conformalized Single Cell Annotator library lets you annotate single-cell data at various significance levels, ensuring precise and informative cell-type assignments even in noisy or complex datasets.

---

## Features

- **Conformal prediction for single-cell**: Obtain a set of likely cell types for each cell, along with user-defined confidence levels.  
- **Plug-and-play model usage**: Easily switch between different models (`HumanLung_TopMarkersFC_level3`, etc.) for annotation.  
- **Quality control**: Optional data preprocessing ensures your input is in top shape before annotation.  
- **Batch correction**: `combat`, `mnn`, or `harmony`-based correction to handle technical or batch effects.  
- **Ground truth recovery**: Map predicted annotations to your original (true) labels for easy comparison and downstream analysis.  
- **Versatile outputs**: Simple access to annotation results, conformal prediction sets, and metrics.

---



## Requirements

Python 3.7 +
Pytorch
TorchCP
Scanpy
pyod



## Quickstart
Below is an integrated guide to getting started with the Conformalized Single Cell Annotator and understanding its outputs.

---

### 1. Load Query Data

```python
import numpy as np
import pandas as pd
import scanpy as sc

# Import the ConformalSCAnnotator from wherever it lives in your package
# from conformal_sc_annotator import ConformalSCAnnotator

# 1. Load your query data
query_data_path = 'test_data/GSE178360/GSE178360_immune.h5ad'
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
annotator.quality_control()  
# If your data is already preprocessed, feel free to skip this step.

# 5. Configure model and conformal predictor
annotator.configure(
    model="HumanLung_TopMarkersFC_level3",   # Pre-trained or user-provided model
    CP_predictor="mondrian",                # or "cluster"
    cell_type_level="celltype_level3",      # lineage_level2, etc.
    test=True,
    alpha=[0.01, 0.05, 0.1],                # confidence levels
    epoch=20,
    batch_size=525
)

# 6. Annotate your data (with batch correction)
annotator.annotate(batch_correction="combat")  
# Options: "combat", "mnn", or "harmony"

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
df_results.to_csv("saves/results_immune.csv", index=False)```


## Interpreting the Output

- **`predicted_labels`**: The single best label predicted by the model.  
- **`prediction_sets_alpha`**: The conformal prediction set for each significance level (\(\alpha\)). A smaller \(\alpha\) typically means a higher confidence (and potentially smaller sets).  
- **`_mapped_original_ground_truth_labels`**: If you provided ground truth labels, these are mapped to the model’s label space for easy comparison.  
- **`results_immune.csv`**: Example CSV file containing predicted labels, conformal prediction sets, and ground truth side-by-side.
