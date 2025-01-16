


import os
import gc
from typing import Union, List, Optional
from scipy.sparse import spmatrix 

import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import anndata as ad
from rapidfuzz import fuzz, process

from conformalSC import SingleCellClassifier
from torchcp.classification.score import THR, RAPS, APS


import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings


class ConformalSCAnnotator:

    """
    A single-cell annotator for the bioinformatics and machine learning community.
    This class prepares data, performs quality control, integrates datasets,
    and annotates cells using conformal prediction.
    """
        
    def __init__(self, X, var_query=None, obs_query=None, var_query_gene_column_name="gene_name"):

        self._is_fitted = False 
        self._is_configured = False

        ## ------ Elements can be accesed through the object  -----

        self.adata_query = self._check_data(X, var_query, obs_query, var_query_gene_column_name)
        self._annotated_cells:list = []
        self._annotated_cells_sets:dict = {}
        self._model_labels:list = []
        self._mapped_original_ground_truth_labels:list = []
        self._mapping:dict = {}



    @staticmethod
    def _prepare_anndata(data: Union[np.ndarray, spmatrix],
                         var_query: Union[pd.DataFrame, List[str]],
                         obs_query: Optional[Union[pd.DataFrame, List, np.ndarray]] = None, 
                         var_query_gene_column_name: Optional[str] = "gene_name") -> ad.AnnData:
         


        """
        Prepares the AnnData object.
        """

        # Validate input data type
        if not isinstance(data, (np.ndarray, spmatrix)):
            raise TypeError("data matrix must be a numpy array (obs x feat).")
        
        if data.dtype != np.float32:
            data = data.astype(np.float32)  # Use 32-bit float for reduced memory usage
        
        # Validate var_query_list
        if not isinstance(var_query, (pd.DataFrame, list)):
            raise TypeError("`var_query` must be a DataFrame or a list.") 
        
        
        # Validate var_query_list
        if var_query_gene_column_name is None:
            var_query_gene_column_name = "gene_name"


        if isinstance(var_query, list):
            var_query = pd.DataFrame({var_query_gene_column_name:var_query })
            
        # Validate obs_query
        if isinstance(var_query, pd.DataFrame):

            if var_query_gene_column_name not in var_query.columns:
                raise KeyError(
                    f"The column '{var_query_gene_column_name}' is missing in var_query. "
                    "Provide the correct column name containing gene names.")      

        if obs_query is not None:

            if not isinstance(obs_query, (pd.DataFrame, list, np.ndarray)):
                raise TypeError("obs_query must be a DataFrame, list or np.ndarray .")
            
            if isinstance(obs_query, (list, np.ndarray)):
                obs_query = pd.DataFrame(index=obs_query)           

        else:
            obs_query = pd.DataFrame(index=np.arange(data.shape[0]))   


        # Create AnnData object
        adata_query = ad.AnnData(
            X=data,        # counts data
            obs=obs_query,  # Rows (samples) as observations
            var=var_query    # Columns (genes) as variables 
        )

        
        adata_query.var_names = adata_query.var[var_query_gene_column_name]
        
        print("Succesfully generated object: ", adata_query.shape)

        return adata_query  


    
    @staticmethod
    def _check_data(data: Union[pd.DataFrame, spmatrix, np.ndarray, ad.AnnData],
                    var_query: Optional[Union[pd.DataFrame, List, np.ndarray, None]] = None,
                    obs_query: Optional[Union[pd.DataFrame, List, np.ndarray, None]] = None, 
                    var_query_gene_column_name: Optional[str] = "gene_name"):
        

        if isinstance(data, ad.AnnData):

            try:
                data.var[var_query_gene_column_name]
                return data

            except KeyError:
                raise KeyError(
                    f"Provide the correct column name containing gene names in -var_query_gene_column_name= -  .")
            
    

        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()


        ann_data = ConformalSCAnnotator._prepare_anndata(data, var_query, obs_query, var_query_gene_column_name)
            
        return ann_data   
    



    @property
    def is_configured(self):

        """
        Check if the configuration method has been executed
        """

        if not self._is_configured:
            print("Annotator is not configured. Please run the `configure` method first.")
            return self._is_configured

        # Print configuration details
        print(f"CP Predictor: {self.CP_predictor}")
        print(f"Cell Type Level: {self.cell_type_level}")
        print(f"Test Mode: {self.do_test}")
        print(f"Alpha: {self.alpha}")
        print(f"Path to reference single cell data: {self.reference_path}")
        print(f"Hidden Sizes: {self.hidden_sizes}")
        print(f"Dropout Rates: {self.dropout_rates}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Excluded Cell Type Threshold: {self.cell_types_excluded_treshold}")

        return self._is_configured


    @property
    def is_fitted(self):
        """
        Check if the fit method has been executed.
        """
        return self._is_fitted


    def quality_control(self,  mito_prefix="MT-", ribo_prefix=("RPS", "RPL"), hb_pattern="^HB[^(P)]"):

        """
        Perform quality control (QC) preprocessing on the AnnData object.

        Parameters:
        mito_prefix : str, default="MT-"
            Prefix identifying mitochondrial genes.
        ribo_prefix : tuple, default=("RPS", "RPL")
            Prefixes identifying ribosomal genes.
        hb_pattern : str, default="^HB[^(P)]"
            Regex pattern identifying hemoglobin genes.
        """

        # This step does not duplicate the AnnData object.
        # It simply creates a new reference to the same object. This is an efficient operation with no additional memory cost.
        adata_query = self.adata_query
        
        print("Filtering low quality cells...")

        # Identify genes of interest
        adata_query.var["mt"] = adata_query.var_names.str.startswith(mito_prefix)
        adata_query.var["ribo"] = adata_query.var_names.str.startswith(ribo_prefix)
        adata_query.var["hb"] = adata_query.var_names.str.contains(hb_pattern, regex=True)

        # Calculate QC metrics
        sc.pp.calculate_qc_metrics(
            adata_query, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
        )

        # Set QC thresholds
        mito_threshold = 10  # Percentage of mitochondrial counts
        total_counts_lower_threshold = 100  # Minimum total counts

        # Apply QC filters
        filter_condition = (
            (adata_query.obs["pct_counts_mt"] < mito_threshold) & 
            (adata_query.obs["total_counts"] > total_counts_lower_threshold)
        )
        
        self.adata_query = adata_query[filter_condition]

        print("Filtered! Post-QC shape: ", self.adata_query.shape)


        return None



    def configure(self,
                    reference_path: str = None,
                    model_architecture: Optional[dict] = None,
                    OOD_detector: Optional[dict] = None,
                    CP_predictor="mondrian",
                    cell_type_level="celltype_level3",
                    cell_types_excluded_treshold = 50,
                    test=False,
                    alpha: Union[float, List[float]] = 0.05, 
                    non_conformity_function = RAPS(),
                    epoch: int = 10,
                    batch_size: int = 1024,
                    verbose=True):   

        """
        Configure the annotator with model-specific settings and conformal prediction parameters.

        Parameters:
        model : str
            Name of the model to load. Supported: 'HumanLung_5K_HVG', 'HumanLung_1K_HVG'.
        CP_predictor : str, default="mondrian"
            Type of conformal predictor to use.
        cell_type_level : str, default="celltype_level3"
            Taxonomy level for cell type prediction.
        test : bool, default=False
            If True, run in test mode (e.g., reduced data or faster execution).
        alpha : float, default=0.05
            Miscoverage level for conformal prediction.
        """



        if isinstance(reference_path, str):
            self.reference_path = reference_path
        else:
            raise ValueError("Please provide a valid path to the reference data.")
        

        if isinstance(alpha, float):
            self.alpha = [alpha]
        else:
            self.alpha = alpha

        self.CP_predictor = CP_predictor
        self.cell_type_level = cell_type_level
        self.do_test = test
        self.epoch = epoch
        self.batch_size = batch_size
        self.non_conformity_function = non_conformity_function


        if model_architecture is None:
            
            model_architecture:dict = {   
                "hidden_sizes": [128, 128, 64, 64],
                "dropout_rates": [0.4, 0.3, 0.4, 0.25],
                "learning_rate": 0.0001}
            
            warnings.warn("You did not configure the model architecture. A generic one will be used.", UserWarning)

           
        # DESIGN OF THE ANNOTATOR
        hidden_sizes:list = model_architecture["hidden_sizes"]
        dropout_rates:list = model_architecture["dropout_rates"]
        learning_rate:list = model_architecture["learning_rate"]


        if OOD_detector is None:
            
            OOD_detector:dict = {
                "n_estimators": 500,
                "max_features": 1,
                "alpha": 0.05}
            
            warnings.warn("You did not configure the OOD_detector. A generic one will be used.", UserWarning)

           
        # DESIGN OF THE ANNOTATOR
        n_estimators:int = OOD_detector["n_estimators"]
        max_features:int = OOD_detector["max_features"]
        alpha_OOD:float = OOD_detector["alpha"]


        # Store model configuration:
        self.reference_path = reference_path

        self.hidden_sizes = hidden_sizes
        self.dropout_rates = dropout_rates
        self.learning_rate = learning_rate

        self.n_estimators = n_estimators
        self.max_features = max_features
        self.alpha_OOD = alpha_OOD

        self.cell_types_excluded_treshold = cell_types_excluded_treshold 

        
        # Reference cell type to be predicted:
        calibration_taxonomy = 'celltype_level2' # TO BE IMPLEMEMNTED



        self._is_configured = True

        
        return None
     

    # Mayby we can define both a fitted model and a model to be fitted    
    def fit(self, epoch , batch_size):   

         
        """
        Fit the single-cell classifier using the provided data and configuration.

        Parameters:
        epoch : int, default=6
             Number of epochs for training the classifier.
        batch_size : int, default=525
            Size of batches for training.
        batch_correction : bool, default=True
            If True, perform Harmony integration of data.
        """


        # Check if the model where configured
        if self._is_configured == False:
            raise ValueError("Please configure the annotator before fitting.")
        
        if self.adata_query is None:
            raise ValueError("Query data (adata_query) is not loaded. Please load the data before fitting.")


         # Initialize classifier with training parameters
        classifier = SingleCellClassifier(epoch=epoch, batch_size=batch_size, do_test=self.do_test) 

        # Load rhe reference data
        self.adata_query  = classifier.load_data(self.reference_path,
                                                        self.adata_query,
                                                        self.cell_type_level,
                                                        self.cell_types_excluded_treshold,
                                                        batch_correction=self.integration_method )

        classifier.fit_OOD_detector(n_estimators=self.n_estimators,
                                    max_features=self.max_features,
                                    alpha_OOD=self.alpha_OOD)


        # Train classifier
        classifier.define_architecture(self.hidden_sizes, self.dropout_rates)
        classifier.fit(lr=self.learning_rate)

        # Calibrate classifier
        classifier.calibrate(non_conformity_function = self.non_conformity_function,
                             alpha = self.alpha,
                             predictors = self.CP_predictor)
            
        if self.do_test:
            classifier.test()

        
        self._is_fitted = True

        return classifier
        
        


    def _annotate(self, classifier):

        
        # Perform prediction
        if self.integration_method == "combat":
            classifier.predict(self.adata_query.X.astype(np.float32))
        
        if self.integration_method == "mnn":
            classifier.predict(self.adata_query.X.astype(np.float32))
            
        if self.integration_method == "harmony":
            classifier.predict(self.adata_query.obsm['X_pca_harmony'].astype(np.float32))

        # Extract predictions and prediction sets
        _annotated_cells = classifier.predicted_labels
        _annotated_cells_sets = classifier.prediction_sets
        self._model_labels = classifier.unique_labels 

        
        # Update observation metadata with predictions
        self.adata_query.obs["predicted_labels"] = _annotated_cells
        for alpha in self.alpha:
            self.adata_query.obs[f"prediction_sets_{alpha}"] = _annotated_cells_sets[alpha]

        return None
    


    def annotate(self, batch_correction="combat"):

        """
        Annotate the dataset using the trained model.
        """

        self.integration_method = batch_correction  

        # Check if the model is already trained and calibrated
        if self._is_fitted == False:
            print("Model not trained yet. Fitting the model first.")
            self.classifier_model = self.fit(epoch=self.epoch, batch_size=self.batch_size)


        print("Starting annotation process...")
        self._annotate(self.classifier_model)
        print("Annotation process completed.")

        # Annotations are now available in `_annotated_cells` and `_annotated_cells_sets`

    


    def recover_original_cells(self, ground_truth_labels, similarity_threshold=70):

        """
        Replace labels in the ground truth DataFrame based on exact and probable matches with reference labels.

        Parameters:
        - model_labels (pd.DataFrame): DataFrame containing reference labels.
        - ground_truth_labels (pd.DataFrame): DataFrame containing labels to be replaced.
        - source_col_model (str): Column name in `model_labels` containing reference labels.
        - source_col_truth (str): Column name in `ground_truth_labels` containing target labels.
        - similarity_threshold (int): Threshold for fuzzy matching (default is 70).

        Returns:
        - ground_truth_labels (pd.DataFrame): Updated DataFrame with replaced labels.
        - mapping (dict): Mapping dictionary used for replacements.
        """

        # Create sets from the specified columns
        set1 = set(self._model_labels)  # Reference set (set1)
        set2 = set(ground_truth_labels)  # Target set (set2)

        # Find exact matches
        exact_matches = set1 & set2

        # Create a mapping dictionary for replacements
        self._mapping = {label: label for label in exact_matches}  # Exact matches map to themselves

        # Find probable matches
        for s1 in set1:  # Exclude exact matches to save processing time
            match = process.extractOne(s1, set2, scorer=fuzz.ratio)
            if match and match[1] > similarity_threshold:  # Threshold for similarity
                self._mapping[match[0]] = s1  # Map target (set2) to reference (set1)

        # Replace elements in the target column of `ground_truth_labels`
        self._mapped_original_ground_truth_labels= [self._mapping.get(label, "OOD") for label in ground_truth_labels]
        
        

        # Print Results
        print(f"\nExact Matches: {len(exact_matches)}")
        print(f"\nProbable Matches: {len(self._mapping) - len(exact_matches)}")
        #print(f"Labels replaced. Remaining 'OOD' labels: {(ground_truth_labels[source_col_truth] == 'OOD').sum()}")

   
        



if __name__ == "__main__":

    

    query_data_path = 'test_data\GSE178360\GSE178360_immune.h5ad'
    adata_query = sc.read_h5ad(query_data_path) 

    ## This is te expected input data: ##

    X = adata_query.X.astype(np.float32)                        # data matrix (cells x genes)
    var_query_list = adata_query.var["features"].tolist()       # This is the case of the list
    #var_query_df = pd.DataFrame({'features':var_query_list })  # Unncomment for testing. This is the case of the df
    obs_query = adata_query.obs                                 # not needed, for ground thruth test


     
    annotator = ConformalSCAnnotator(X, var_query_list, obs_query) # obs_query is optional, it will be used for annotate the predicted cells. 

    annotator.quality_control()  ## This is an optional step to do a basic preprocess the data. If the data is already preprocessed, this step is optional.

    # Define que network architecture   
    network_architecture:dict = {   
                "hidden_sizes": [128, 128, 64, 64],
                "dropout_rates": [0.4, 0.3, 0.4, 0.25],
                "learning_rate": 0.0001}

    OOD_detector_config = {
                "n_estimators": 500,
                "max_features": 1,
                "alpha": 0.02}
        

    reference_data_path = os.path.join("models", "HumanLung_TopMarkersFC_level3.h5ad")     # Path to the reference data

    annotator.configure(reference_path = reference_data_path,
                        model_architecture = network_architecture,   # Optional, if not provided, default values will be used
                        OOD_detector = OOD_detector_config,          # Optional, if not provided, default values will be used
                        CP_predictor = "cluster",                    # mondrian or cluster
                        cell_type_level = "celltype_level3",         # class name for fitting the model.  
                        cell_types_excluded_treshold = 50,           # Exclude cell types with less than 50 cells
                        test = True,                                 # Perform internal test of the model
                        alpha = [0.01, 0.05, 0.1],                   # Confidence of the predictions
                        non_conformity_function = APS(),             # NC-function provided by or compatible with torchCP    
                        epoch=20,
                        batch_size = 525)  
        

    # annotate your data to a given significance level
    annotator.annotate(batch_correction="combat")  # combat, mnn, harmony

    # Get the predictions returning the observations of the query data object
    #print("\nPredicted annotations sets: \n" , annotator.adata_query.obs)

    
    ground_truth_labels_list = obs_query["cell_type"].tolist()
    annotator.recover_original_cells( ground_truth_labels_list, similarity_threshold=70)


    y_true = annotator._mapped_original_ground_truth_labels ## Ground thruth labels mapped to the model labels (predictions)
    #annotator._mapping


    results = []
    for pred,cp_pred_001,cp_pred_005, cp_pred_010, true, o_g_t in zip(
            annotator.adata_query.obs["predicted_labels"],
            annotator.adata_query.obs["prediction_sets_0.01"],
            annotator.adata_query.obs["prediction_sets_0.05"],
            annotator.adata_query.obs["prediction_sets_0.1"],
            y_true,
            ground_truth_labels_list):
        
        print(f"Predicted: {pred} - CP 0.01: {cp_pred_001} - CP 0.05: {cp_pred_005} - CP 0.10: {cp_pred_010} - True: {true}. original cell Subt: {o_g_t}")
        
        results.append({
            "Predicted": pred,
            "CP 0.01": cp_pred_001,
            "CP 0.05": cp_pred_005,
            "CP 0.1": cp_pred_010,
            "True": true,
            "Original_Cell_Subtype": o_g_t
        })
    
    df_results = pd.DataFrame(results)
    df_results.to_csv("saves/results_immune.csv", index=False)  # Save to CSV if needed


    # now compare cell_annotations with annotator._mapped_original_ground_truth_labels
      


