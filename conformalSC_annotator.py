


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
    This is a high-level class that annotates cells using conformal prediction.
    A low level class named SingleCellClassifier contains the core of the method.
    """
        
    def __init__(self, X, var_query=None, obs_query=None, var_query_gene_column_name="gene_name"):

        self._is_fitted = False 
        self._is_configured = False

        ## ------ Elements can be accesed through the object  -----

        self.adata_query, self.var_query_gene_column_name = self._check_data(X, var_query, obs_query, var_query_gene_column_name)
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
            X=data,          # counts data
            obs=obs_query,   # Rows (samples) as observations
            var=var_query    # Columns (genes) as variables 
        )

        
        adata_query.var_names = adata_query.var[var_query_gene_column_name]
        
        print("Succesfully generated object: ", adata_query.shape)

        return adata_query, var_query_gene_column_name


    
    @staticmethod
    def _check_data(data: Union[pd.DataFrame, spmatrix, np.ndarray, ad.AnnData],
                    var_query: Optional[Union[pd.DataFrame, List, np.ndarray, None]] = None,
                    obs_query: Optional[Union[pd.DataFrame, List, np.ndarray, None]] = None, 
                    var_query_gene_column_name: Optional[str] = "gene_name"):
        

        if isinstance(data, ad.AnnData):

            try:
                    
                data.var[var_query_gene_column_name]
                data.var_names = data.var[var_query_gene_column_name]
                return data, var_query_gene_column_name

            except KeyError:
                raise KeyError(
                    f"Provide the correct column name containing gene names in -var_query_gene_column_name= -  .")
            
    

        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()


        ann_data, var_query_gene_column_name = ConformalSCAnnotator._prepare_anndata(data, var_query, obs_query, var_query_gene_column_name)
            
        return ann_data, var_query_gene_column_name   
    



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



    def configure(self,
                    reference_path: str = None,
                    model_architecture: Optional[dict] = None,
                    OOD_detector: Optional[dict] = None,
                    CP_predictor="mondrian",
                    cell_names_column="celltype_level3",
                    cell_types_excluded_treshold = 45,
                    test=False,
                    alpha: Union[float, List[float]] = 0.05, 
                    non_conformity_function = RAPS(),
                    epoch: int = 20,
                    batch_size: int = 64,
                    random_state = None,
                    verbose=True):   
        


        """
        Configure the annotator with model-specific settings and conformal prediction parameters.

        Parameters:
        model : str
            Name of the model to load. Supported: 'HumanLung_5K_HVG', 'HumanLung_1K_HVG'.
        CP_predictor : str, default="mondrian"
            Type of conformal predictor to use.
        cell_names_column : str, default="celltype_level3"
            Name with the cell type in the reference.
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
        self.cell_type_level = cell_names_column
        self.do_test = test
        self.epoch = epoch
        self.batch_size = batch_size
        self.non_conformity_function = non_conformity_function
        self.random_state = random_state


        if model_architecture is None:
            
            model_architecture:dict = {   
                "hidden_sizes": [128,  64, 64],
                "dropout_rates": [0.4, 0.3],
                "learning_rate": 0.0001}
            
            warnings.warn("You did not configure the model architecture. A generic one will be used.", UserWarning)

           
        # DESIGN OF THE ANNOTATOR
        hidden_sizes:list = model_architecture["hidden_sizes"]
        dropout_rates:list = model_architecture["dropout_rates"]
        learning_rate:list = model_architecture["learning_rate"]


        if OOD_detector is None:
            
            OOD_detector:dict = {
                "pvalues": "marginal",
                "alpha": 0.1,
                "delta": 0.1,
                "hidden_sizes": [ 556,  124],     
                "dropout_rates": [ 0.3,  0.30], 
                "learning_rate": 0.001,
                "batch_size": 64,
                "n_epochs": 150}
            
            warnings.warn("You did not configure the OOD_detector. A generic one will be used.", UserWarning)

           
        # DESIGN OF THE ANNOTATOR

        try:
            pvalues:str = OOD_detector["pvalues"]
        except KeyError:
            warnings.warn("You did not configure -pvalues- in the OOD_detector. -marginal- will be used.", UserWarning)
            pvalues:str = "marginal"

        try:   
            alpha_OOD:float = OOD_detector["alpha"]
        except KeyError:
            warnings.warn("You did not configure -alpha- in the OOD_detector. -0.1- will be used.", UserWarning)
            alpha_OOD:float = 0.1

        if pvalues == "conditional":
            
            try:   
                delta_OOD:float = OOD_detector["delta"]

            except KeyError:
                warnings.warn("You did not configure -delta- in the OOD_detector. -0.1- will be used.", UserWarning)
                delta_OOD:float = 0.1
        else:
            delta_OOD:float = 0


        try:

            hidden_sizes_OOD:list = OOD_detector["hidden_sizes"]
            dropout_rates_OOD:list = OOD_detector["dropout_rates"]
            learning_rate_OOD:float = OOD_detector["learning_rate"]
            batch_size_OOD:int = OOD_detector["batch_size"]
            n_epochs_OOD:int = OOD_detector["n_epochs"]

            if len(hidden_sizes_OOD) != len(dropout_rates_OOD):
                raise ValueError("The hidden_sizes and dropout_rates must have the same length.")

        except KeyError:
            raise ValueError("One of the following parameters the OOD detector is missing:\n hidden_sizes,\n dropout_rates,\n learning_rate,\n batch_size,\n n_epochs.")



        # Store model configuration:
        self.reference_path = reference_path

        self.hidden_sizes = hidden_sizes
        self.dropout_rates = dropout_rates
        self.learning_rate = learning_rate

        self.pvalues = pvalues  
        self.alpha_OOD = alpha_OOD
        self.delta_OOD = delta_OOD
        self.hidden_sizes_OOD = hidden_sizes_OOD
        self.dropout_rates_OOD = dropout_rates_OOD
        self.learning_rate_OOD = learning_rate_OOD
        self.batch_size_OOD = batch_size_OOD
        self.n_epochs_OOD = n_epochs_OOD

        self.cell_types_excluded_treshold = cell_types_excluded_treshold 

        
        self._is_configured = True

    
        return None
     


   
    def fit(self, epoch , batch_size):   

         
        """

        Fit the single-cell classifier using the provided data and configuration.

        Parameters:
        epoch : int
             Number of epochs for training the classifier.
        batch_size : int
            Size of batches for training.
        batch_correction : bool
            If True, get the integrated data.

        """


        # Check if the model where configured
        if self._is_configured == False:
            raise ValueError("Please configure the annotator before fitting.")
        
        if self.adata_query is None:
            raise ValueError("Query data (adata_query) is not loaded. Please load the data before fitting.")


         # Initialize classifier with training parameters
        classifier = SingleCellClassifier(epoch=epoch, batch_size=batch_size, do_test=self.do_test, random_state=self.random_state) 

        # Load rhe reference data
        self.adata_query  = classifier.load_data(self.reference_path,
                                                        self.adata_query,
                                                        self.cell_type_level,
                                                        self.cell_types_excluded_treshold,
                                                        batch_correction=self.integration_method,
                                                        gene_column_name = self.var_query_gene_column_name)

        # Fit anomaly detector
        classifier.fit_OOD_detector(pvalues=self.pvalues,
                                    alpha_OOD=self.alpha_OOD,
                                    delta_OOD=self.delta_OOD,
                                    hidden_sizes_OOD=self.hidden_sizes_OOD,
                                    dropout_rates_OOD=self.dropout_rates_OOD,
                                    learning_rate_OOD=self.learning_rate_OOD,
                                    batch_size_OOD=self.batch_size_OOD,
                                    n_epochs_OOD=self.n_epochs_OOD)
        

        # Train classifier
        classifier.define_architecture(self.hidden_sizes, self.dropout_rates)
        classifier.fit(lr=self.learning_rate)


        # Calibrate classifier
        classifier.calibrate(non_conformity_function = self.non_conformity_function,
                             alpha = self.alpha,
                             predictors = self.CP_predictor)
        
        self.test_results = None   
        if self.do_test:
            classifier.test()
            self.test_results = classifier.InDis_results_

        
        self._is_fitted = True

        return classifier
        
        


    def _annotate(self, classifier):

        

        # Perform prediction
        if self.integration_method == None:

            classifier.predict(self.adata_query.X.toarray().astype(np.float32))

        if self.integration_method =='X_pca_harmony':
            classifier.predict(self.adata_query.obsm['X_pca_harmony'].astype(np.float32))
        
        if self.integration_method =='X_pca':
            classifier.predict(self.adata_query.obsm['X_pca'].astype(np.float32))


        # Extract predictions and prediction sets
        _annotated_cells = classifier.predicted_labels
        _annotated_cells_sets = classifier.prediction_sets
        self._model_labels = classifier.unique_labels 

        
        self.unique_labels = classifier.unique_labels
        self.adata_query.labels_encoded = classifier.labels_encoded
        
        
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

    

   
        




      


