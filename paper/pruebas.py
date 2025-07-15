



import os
import gc
import numpy as np  
import scanpy as sc
import pandas as pd
import anndata as ad 

from conformalSC_annotator import  ConformalSCAnnotator #be careful with the path
from torchcp.classification.score import  APS,RAPS, THR

# First we read the already preprocessed query data using scanpy. In this case data is coming from a h5ad file.

adata_ref_path = "your_ref_data_path.h5ad"  # Replace with your actual data path
adata_ref = ad.read_h5ad(adata_ref_path)

adata_query_path = "your_query_data_path.h5ad"  # Replace with your actual reference data path
adata_query = ad.read_h5ad(adata_ref_path)

gene_names_column = "features" 
n_queries = [ "1" ] # we only have one query dataset in this case, but you can add more and name them if you want to test different queries



X_np = adata_ref.X
# Create a boolean mask of where values are NaN
nan_mask = np.isnan(X_np)
# Check if there are any NaNs at all
if nan_mask.any():
    # Find the row indices that contain at least one NaN
    rows_with_nans = np.where(nan_mask.any(axis=1))[0]
    print(f"Found NaNs in {len(rows_with_nans)} rows: {rows_with_nans}")
else:
    print("No NaNs found in X_np.")


def in_distribution_mask(df: pd.DataFrame, cells_OOD: list[str]) -> pd.Series:
    
    """
    A row is ‘in-distribution’ iff
      – its ground-truth label is NOT in cells_OOD  AND
      – the model did NOT predict "OOD".
    """

    return (~df['Original_Cell_Subtype'].isin(cells_OOD)) & (df['Predicted'] != 'OOD')


def coverage_and_covgap(df: pd.DataFrame,
                         alpha: float | str,
                         cells_OOD: list[str],
                         unique_labels: list[str]
                         ) -> tuple[float, float, np.ndarray, float]:
    
    """
    Returns:
        coverage_ID        – empirical coverage on strictly in-distribution rows
        cov_gap            – mean |coverage_c − (1−alpha)|
        coverage_per_class – array in the order of unique_labels (NaN if class absent)
        avg_size_ID        – average prediction-set length on in-distribution rows
    """
    a = str(alpha)
    mask_ID = in_distribution_mask(df, cells_OOD)

    # Boolean “true label is inside prediction set” (aligned index ⇒ no indexing error)
    in_set_bool = df.apply(
        lambda row: row['Original_Cell_Subtype'] in row[f'CP {a}'], axis=1
    )

    # Total coverage & average size (strict ID)
    coverage_ID = in_set_bool[mask_ID].mean()
    avg_size_ID = df.loc[mask_ID, f'CP {a}'].apply(len).mean()

    # Per-class coverage
    coverage_per_class = []
    for c in unique_labels:
        class_mask = mask_ID & (df['Original_Cell_Subtype'] == c)
        coverage_per_class.append(
            in_set_bool[class_mask].mean() if class_mask.any() else np.nan
        )

    cov_gap = np.nanmean(np.abs(np.array(coverage_per_class) - (1 - float(alpha))))
    return coverage_ID, cov_gap, np.array(coverage_per_class, dtype=float), avg_size_ID




def overall_success_rate(df: pd.DataFrame, alpha: str) -> float:
    """
    Returns:
        success_rate – fraction of *all non-rejected queries* whose set contains y
        avg_size_all – average prediction-set length on the same rows
    """
    mask = df['Predicted'] != 'OOD'
    in_set_bool = df.apply(
        lambda row, a=alpha: row['Original_Cell_Subtype'] in row[f'CP {a}'], axis=1
    )
    success_rate = in_set_bool[mask].mean()
    avg_size_all = df.loc[mask, f'CP {alpha}'].apply(len).mean()
    return success_rate, avg_size_all




### define the parameters for the experiments
n_seeds = 20
taxonomies =  ["standard", "classwise", "cluster"]
nc_functions =  [APS(), RAPS(penalty=0.01, kreg=1), THR()]


ref_column = "cell_type"  # name of the column in adata_ref.obs with the reference cell types
cells_OOD = ["VL1_LE","I2_NK", "I5_PlasmaCell", "BA_doublets"] #example of OOD cells, you can change this to remove cells from the reference dataset that you do not want to use in the experiments (can be empty too)

all_metrics = []
for idx, query_dataset in enumerate([n_queries]):
    for taxonomy in taxonomies:
        for nc_function in nc_functions:
            for seed in range(n_seeds):
                
                print(f"\nRunning with seed: {seed}, taxonomy: {taxonomy}, nc_function: {nc_function.__class__.__name__}")

                obs_query = adata_query.obs
                annotator = ConformalSCAnnotator(adata_query,
                                                 var_query_gene_column_name = "features",
                                                underlying_model = "torch_net") #"torch_net", "celltypist" , "scmap"
                
                network_architecture:dict = {   
                                            "hidden_sizes": [ 72,64,32, 32],
                                            "dropout_rates": [ 0.15, 0.15, 0.15,  0.15],
                                            "learning_rate": 1e-4}

                OOD_detector_config = {
                                        "alpha": None,
                                        "delta": 0.1,
                                        "hidden_sizes":  [50, 48, 32, 24], 
                                        "dropout_rates": [0.15, 0.15, 0.15, 0.15], 
                                        "learning_rate": 1e-4,
                                        "batch_size":    72,
                                        "n_epochs":      1000,
                                        "patience":      9,
                                        "noise_level":   0.1,
                                        "lambda_sparse": 1e-3} 
                
                do_test = True
                annotator.configure(reference_path = adata_ref,
                                    model_architecture = network_architecture,   # Optional, if not provided, default values will be used
                                    OOD_detector = OOD_detector_config,          # Optional, if not provided, default values will be used
                                    CP_predictor = taxonomy,                         # mondrian or cluster
                                    cell_names_column = ref_column,              # class name for fitting the model.  
                                    cell_types_excluded_treshold = 60,           # Exclude cell types with less than 60 cells or in cells_OOD (e.g. cell_types_excluded_treshold = cells_OOD)
                                    test =  do_test,                                # Perform internal test of the model
                                    alpha = [0.01, 0.05, 0.1],                   # Confidence of the predictions
                                    non_conformity_function = nc_function,             # NC-function provided by or compatible with torchCP    
                                    epoch=1000,
                                    batch_size = 72,
                                    random_state = seed)  
                    
                
                obsm_layer_ = "obsm"
                layer_ = None 
                obsm_ = "X_pca_harmony"
                obsm_OOD_ = "X_pca_harmony"


                annotator.annotate(obsm_layer=obsm_layer_,
                                    obsm = obsm_,
                                    layer = layer_,
                                    obsm_OOD = obsm_OOD_)

                # Get the predictions returning the observations of the query data object
                #print("\nPredicted annotations sets: \n" , annotator.adata_query.obs)

                OOD_performance_scores = annotator.OOD_performance_scores
                unique_labels = annotator.unique_labels

                test_results = annotator.test_results
                ground_truth_labels_list = obs_query[ref_column].tolist()
                
                

                results = []
                for pred,cp_pred_001,cp_pred_005, cp_pred_010, o_g_t in zip(
                        annotator.adata_query.obs["predicted_labels"],
                        annotator.adata_query.obs["prediction_sets_0.01"],
                        annotator.adata_query.obs["prediction_sets_0.05"],
                        annotator.adata_query.obs["prediction_sets_0.1"],
                        ground_truth_labels_list):
                        
                    #print(f"Predicted: {pred} - CP 0.01: {cp_pred_001} - CP 0.05: {cp_pred_005} - CP 0.10: {cp_pred_010} - True: {true}. original cell Subt: {o_g_t}")
                        
                    results.append({
                        "Predicted": pred,
                        "CP 0.01": cp_pred_001,
                        "CP 0.05": cp_pred_005,
                        "CP 0.1": cp_pred_010,
                        "Original_Cell_Subtype": o_g_t
                    })
                    
                df_results = pd.DataFrame(results)

                alphas = ["0.01", "0.05", "0.1"]
                
                # Count the number of OOD predictions
                num_OOD = len(df_results[df_results['Predicted'] == "OOD"])
                
                # Loop over each alpha and compute the desired metrics
                for alpha in alphas:

                    if do_test:
                        coverage_test = test_results[float(alpha)]["Coverage_rate"]
                        CovGap_test = test_results[float(alpha)]["CovGap"]
                        Average_size_test = test_results[float(alpha)]["Average_size"]
                    else:
                        coverage_test = np.nan
                        CovGap_test = np.nan
                        Average_size_test = np.nan

                    acc_OOD = OOD_performance_scores["accuracy"]
                    auroc_OOD = OOD_performance_scores["auroc"]
                    precision_OOD = OOD_performance_scores["precision"]
                    recall_OOD = OOD_performance_scores["recall"]

                    strict_cov, covgap_query, coverage_mondrian, mean_length_ID = coverage_and_covgap(df_results, alpha,cells_OOD, unique_labels)
                    success_rate, mean_length_ALL = overall_success_rate(df_results, alpha)
                    
                    # Append the metrics along with identifying parameters
                    all_metrics.append({
                        'dataset': query_dataset,
                        'taxonomy': taxonomy,
                        'nc_function': nc_function.__class__.__name__,
                        'seed': seed,
                        'alpha': alpha,
                        'coverage_test': coverage_test,
                        'CovGap_test': CovGap_test/100,
                        'Average_size_test': Average_size_test,
                        'Coverage_query_in_dist': strict_cov,
                        'CovGap_query_in_dist': covgap_query,
                        'Average_size_query_in_dist': mean_length_ID,
                        'coverage_per_class_in_dist': coverage_mondrian,
                        'success_rate': success_rate,
                        'Average_size_success_rate': mean_length_ALL,
                        'acc_OOD': acc_OOD,
                        'auroc_OOD': auroc_OOD,
                        'precision_OOD': precision_OOD,
                        'recall_OOD': recall_OOD,
                        "class_names": unique_labels
                    })

                    print(f"Alpha: {alpha}, Strict Coverage: {strict_cov}, CovGap: {covgap_query}, Average Size (ID): {mean_length_ID}, Success Rate: {success_rate}, Average Size (ALL): {mean_length_ALL}")
                # Optional: Clear memory after each seed
                del obs_query
                gc.collect()

 


# Convert the collected results into a DataFrame
df_all_metrics = pd.DataFrame(all_metrics)
df_all_metrics.to_csv('save_path.csv', index=False)




                



