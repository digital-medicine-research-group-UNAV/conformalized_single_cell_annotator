


import os
import gc
from typing import Union, List, Optional
from scipy.sparse import spmatrix 

import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import anndata as ad

from conformalSC import SingleCellClassifier
from torchcp.classification.score import THR, RAPS, APS
import os
cwd = os.getcwd()
print("Current kernel CWD:", cwd)

import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings


class ConformalSCAnnotator:

    """
    A single-cell annotator for the bioinformatics and machine learning community.
    This is a high-level class that annotates cells using conformal prediction.
    A low level class named SingleCellClassifier contains the core of the method.
    """
        
    def __init__(self, 
                 X, 
                 var_query=None, 
                 obs_query=None, 
                 var_query_gene_column_name="gene_name", 
                 underlying_model="torch_net"):

        self._is_fitted = False 
        self._is_configured = False

        ## ------ Elements can be accesed through the object  -----

        self.adata_query, self.var_query_gene_column_name = self._check_data(X, var_query, obs_query, var_query_gene_column_name)
        self.underlying_model = underlying_model 
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
                    reference_path: Union[str, ad.AnnData] = None,
                    model_architecture: Optional[dict] = None,
                    OOD_detector: Optional[dict] = None,
                    CP_predictor="standard",
                    cell_names_column=None,
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
            Type of conformal predictor to use.
        cell_names_column : str, default="None"
            Name with the cell type in the reference.
        test : bool, default=False
            If True, run in test mode (e.g., reduced data or faster execution).
        alpha : float, default=0.05
            Miscoverage level for conformal prediction.
        """

        


        if isinstance(reference_path, Union[str, ad.AnnData]):
            self.reference_path = reference_path
        else:
            raise ValueError("Please provide a valid path to the reference data or an adata object.")
        
        # Unificar alpha en lista
        self.alpha = [alpha] if isinstance(alpha, float) else alpha

        self.CP_predictor = CP_predictor
        self.cell_type_level = cell_names_column
        self.do_test = test
        self.non_conformity_function = non_conformity_function
        self.random_state = random_state
        self.epoch = epoch
        self.batch_size = batch_size

        if self.underlying_model not in ["torch_net", "celltypist", "scmap"]:
            raise ValueError(
                "Invalid classifier_model. Supported values are: torch_net, celltipist, SCmap.")
        

        if self.underlying_model == "torch_net":

            if model_architecture is None:
                model_architecture = {
                    "hidden_sizes": [128, 64, 64],
                    "dropout_rates": [0.4, 0.3],
                    "learning_rate": 0.0001
                }
                warnings.warn("No model_architecture specified; usando parámetros genéricos.", UserWarning )


            self.hidden_sizes = model_architecture["hidden_sizes"]
            self.dropout_rates = model_architecture["dropout_rates"]
            self.learning_rate = model_architecture["learning_rate"]

        
 
        # DESIGN OF THE ANNOTATOR
        


        if OOD_detector is None:
            OOD_detector = {
                "pvalues": "marginal",
                "alpha": 0.1,
                "delta": 0.1,
                "hidden_sizes": [556, 124],
                "dropout_rates": [0.3, 0.3],
                "learning_rate": 0.001,
                "batch_size": 64,
                "n_epochs": 150
            }
            warnings.warn(
                "No OOD_detector specified; usando parámetros por defecto.",
                UserWarning
            )

           


        # CONFIGURE OOD DETECTOR

        pvalues = OOD_detector.get("pvalues", "marginal")
        if "pvalues" not in OOD_detector:
            warnings.warn("Clave 'pvalues' faltante en OOD_detector; usando 'marginal'.", UserWarning)

        alpha_OOD = OOD_detector.get("alpha", None)
        if "alpha" not in OOD_detector:
            warnings.warn("'alpha' was not defined in OOD_detector; using None instead and do_test = True.", UserWarning)
            self.do_test = True

        delta_OOD = (
            OOD_detector.get("delta", 0.1) if pvalues == "conditional" else 0
            )
        if pvalues == "conditional" and "delta" not in OOD_detector:
            warnings.warn("Clave 'delta' faltante en OOD_detector; usando 0.1.", UserWarning)


        try:
            self.hidden_sizes_OOD = OOD_detector["hidden_sizes"]
            self.dropout_rates_OOD = OOD_detector["dropout_rates"]
            self.learning_rate_OOD = OOD_detector["learning_rate"]
            self.batch_size_OOD = OOD_detector["batch_size"]
            self.n_epochs_OOD = OOD_detector["n_epochs"]
            if len(self.hidden_sizes_OOD) != len(self.dropout_rates_OOD):
                raise ValueError( "hidden_sizes y dropout_rates must have the same lenght.")
            
        except KeyError as e:
            raise ValueError(
                "One of the following parameters the OOD detector is missing: "
                f"{e.args[0]}"
            )

  

        self.pvalues = pvalues  
        self.alpha_OOD = alpha_OOD
        self.delta_OOD = delta_OOD

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
        classifier = SingleCellClassifier(classifier_model=self.underlying_model,
                                          epoch=epoch,
                                          batch_size=batch_size,
                                          do_test=self.do_test,
                                          random_state=self.random_state) 

        # Load rhe reference data
        self.adata_query  = classifier.load_data(self.reference_path,
                                                        self.adata_query,
                                                        self.cell_type_level,
                                                        self.cell_types_excluded_treshold,
                                                        obsm=self.obsm,
                                                        layer=self.layer,
                                                        gene_column_name = self.var_query_gene_column_name)

        # Fit anomaly detector
        classifier.fit_OOD_detector(pvalues=self.pvalues,
                                    alpha_OOD=self.alpha_OOD,
                                    delta_OOD=self.delta_OOD,
                                    hidden_sizes_OOD=self.hidden_sizes_OOD,
                                    dropout_rates_OOD=self.dropout_rates_OOD,
                                    learning_rate_OOD=self.learning_rate_OOD,
                                    batch_size_OOD=self.batch_size_OOD,
                                    n_epochs_OOD=self.n_epochs_OOD,
                                    obsm_OOD=self.obsm_OOD)
        

        # Train classifier
        if self.underlying_model == "torch_net":
            classifier.define_architecture(self.hidden_sizes, self.dropout_rates)
            classifier.fit_network(lr=self.learning_rate)
        
        elif self.underlying_model == "celltypist":
            classifier.fit_celltypist(self.cell_type_level)
        
        elif self.underlying_model == "scmap":
            classifier.fit_scmap(self.cell_type_level)


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
        
        
        if self.underlying_model != "torch_net": 
            if self.layer is None:
                sc.pp.normalize_total(self.adata_query, target_sum=1e4)
                sc.pp.log1p(self.adata_query)
            else:
                sc.pp.normalize_total(self.adata_query, target_sum=1e4, layer=self.layer)
                sc.pp.log1p(self.adata_query, layer=self.layer)


        if self.obsm_OOD is not None:
            query_data_OOD =  self.adata_query.obsm[self.obsm_OOD].astype(np.float32)
        else:
            query_data_OOD = None
        
        # Perform prediction
        if self.obsm_layer == None:

            if isinstance(self.adata_query.X, np.ndarray):
                classifier.predict(self.adata_query.X.astype(np.float32), query_data_OOD)
            else:
                classifier.predict(self.adata_query.X.toarray().astype(np.float32), query_data_OOD)
        
        else:  
            if self.obsm is not None and self.obsm_layer == "obsm":

                classifier.predict(self.adata_query.obsm[self.obsm].astype(np.float32), query_data_OOD)
            
            elif self.layer is not None and self.obsm_layer == "layer":
                
                classifier.predict(self.adata_query.layers[self.layer].astype(np.float32), query_data_OOD )

        
        # Extract predictions and prediction sets
        _annotated_cells = classifier.predicted_labels
        _annotated_cells_sets = classifier.prediction_sets
        _annotated_cells_scores = classifier.prediction_scores
        self._model_labels = classifier.unique_labels 
        
        # Extract OOD results if ground truth available:
        _accuracy_OOD = classifier.accuracy_OOD
        _precision_OOD = classifier.precision_OOD
        _recall_OOD = classifier.recall_OOD
        _auroc_OOD = classifier.auroc_OOD
        
        self.OOD_performance_scores = { "accuracy": _accuracy_OOD,
                                        "precision": _precision_OOD,
                                        "recall": _recall_OOD,
                                        "auroc": _auroc_OOD}
     
        
        self.unique_labels = classifier.unique_labels
        self.adata_query.labels_encoded = classifier.labels_encoded
        self.alpha_OOD = classifier.alpha_OOD
        self.pvalues_query = classifier.p_values
        
        
        # Update observation metadata with predictions
        self.adata_query.obs["predicted_labels"] = _annotated_cells
        for alpha in self.alpha:
            self.adata_query.obs[f"prediction_sets_{alpha}"] = _annotated_cells_sets[alpha]

        self.adata_query.obs["predicted_NCscores"] = _annotated_cells_scores.tolist()

        return None
    


    def annotate(self, obsm_layer=None, obsm = None, layer=None, obsm_OOD = None):

        """
        Annotate the dataset.
        """

        self.obsm_OOD = obsm_OOD

        if obsm_layer is not None:

            if obsm_layer == "obsm":
                if obsm is None:
                    raise ValueError("Please provide obsm name in obsm=")
                else:
                    self.obsm_layer = obsm_layer
                    self.obsm = obsm
                    self.layer = None

            if obsm_layer == "layer":
                if layer is None:
                    raise ValueError("Please provide layer name in layer=")
                else:
                    self.obsm_layer = obsm_layer
                    self.layer = layer
                    self.obsm = None
                
            if obsm_layer not in ["obsm", "layer"]:
                raise ValueError("obsm_layer must be 'obsm' or 'layer' or None(default).")
            
            print("Using obsm_layer: ", self.obsm_layer, " for annotation.")

        else:
            obsm_layer = None
            self.obsm_layer = None 
            print("obsm_layer is set to None. Data stored in adata.X will be used for annotation.")


        
        if self.obsm is not None and self.obsm not in self.adata_query.obsm.keys():
            raise ValueError(f"The provided obsm '{self.obsm}' is not present in the query AnnData object.")

        if self.layer is not None and self.layer not in self.adata_query.layers.keys():
            raise ValueError(f"The provided layer '{self.layer}' is not present in the query AnnData object.")
        
        if self.obsm_OOD is not None and self.obsm_OOD not in self.adata_query.obsm.keys():
            raise ValueError(f"The provided obsm_OOD '{self.obsm_OOD}' is not present in the query AnnData object.")
        

        if self.underlying_model == "celltypist":
            warnings.warn("Provide raw data for annotation. ",
                            UserWarning)
        




        # Check if the model is already trained and calibrated
        if self._is_fitted == False:
            print("Model not trained yet. Fitting the model first.")
            self.classifier_model = self.fit(epoch=self.epoch, batch_size=self.batch_size)



        print("Starting annotation process...")

        self._annotate(self.classifier_model)

        print("Annotation process completed.")

        # Annotations are now available in `_annotated_cells` and `_annotated_cells_sets`

    

   
        




      


