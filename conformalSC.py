


# Import necessary libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
import scanpy.external as sce
import gc

from sklearn.ensemble import IsolationForest
from sklearn import svm
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.pipeline import make_pipeline

from IPython.display import display

from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from itertools import chain

from torch.utils.data import TensorDataset, DataLoader

from torchcp.classification.predictor import SplitPredictor, ClassWisePredictor, ClusteredPredictor
from torchcp.classification.utils.metrics import Metrics

from annomaly_detector import Annomaly_detector, AEOutlierDetector

#from deel.puncc.api.prediction import BasePredictor
#from deel.puncc.anomaly_detection import SplitCAD
#from sklearn.ensemble import IsolationForest




# Neural Network using pytorch
class  NNClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rates=None):
        super(NNClassifier, self).__init__()
        self.layers = nn.ModuleList()

        if dropout_rates is None:
            dropout_rates = [0.1] * len(hidden_sizes)

        for i in range(len(hidden_sizes)):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            self.layers.append(nn.Dropout(dropout_rates[i]))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
    
    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = torch.relu(layer(x))
            else:
                x = layer(x)
        x = self.output_layer(x)
        return x
    


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # torch tensors
        self.labels = labels  # torch tensors

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



class SingleCellClassifier:

    def __init__(self, epoch=4, batch_size = 64, do_test = True):

        self.epoch:int = epoch
        self.batch_size:int = batch_size
        self.do_test:bool = do_test

        self.alpha_OOD:float = 0.1
        self.delta_OOD:float = 0.1

        self.conformal_prediction = False

        self.excluded_positions:dict = {}
        self.labels_index :list = []
        self.common_gene_names:list = []
        self.unique_labels:list = []

        self._metric = Metrics()

    
    def exclude_cells(self) -> None:

        if not self.cell_types_excluded:
            print("No cell types to exclude.")
            return None
        
        # Ensure cell_types_excluded is a list
        if not isinstance(self.cell_types_excluded, (list, tuple)):
            cell_types = [self.cell_types_excluded]
        else:
            cell_types = self.cell_types_excluded
        
        exclude_indexes = set()
        self.excluded_positions = {}    # Initialize a dictionary to store excluded positions
        self.OOD_data = []              # List to store excluded cell data
        self.OOD_labels = []            # List to store excluded labels
        
        for exclude_cell in cell_types:

            if exclude_cell in self.labels:

                # Find positions of the excluded cell type
                positions = [i for i, label in enumerate(self.labels) if label == exclude_cell]
                self.excluded_positions[exclude_cell] = positions

                # Add excluded data and labels to storage
                self.OOD_data.extend(self.obs_data[i] for i in positions)
                self.OOD_labels.extend(self.labels[i] for i in positions)


                exclude_indexes.update(positions)

            else:
                print(f"Cell-type {exclude_cell} not found in labels")


        # Convert excluded data and labels to numpy arrays 
        self.OOD_data = np.array(self.OOD_data)
        self.OOD_labels = np.array(self.OOD_labels)
        
        # Filter out the excluded cells from labels and data
        self.labels = [label for i, label in enumerate(self.labels) if i not in exclude_indexes]
        self.obs_data = np.array([row for i, row in enumerate(self.obs_data) if i not in exclude_indexes])
        
        # Update labels index to reflect filtered positions
        self.labels_index = [i for i in range(len(self.labels))]

        print(f"Excluded {len(exclude_indexes)} cells. Remaining cells: {len(self.labels)}")
        print(f"Stored {len(self.OOD_data)} excluded cells in a separate dataset.")

        return None



    def integrate_data(self, _adata_ref, adata_query):


        if self.integration_method == "harmony":
            
            # Combine datasets for Harmony correction
            adata_combined = _adata_ref.concatenate(adata_query, batch_categories=["ref", "query"],  batch_key="batch")

            del adata_query, _adata_ref # free memory
            gc.collect()  # Force garbage collection

            # Apply PCA
            sc.pp.pca(adata_combined)
            # Run Harmony
            sce.pp.harmony_integrate(adata_combined, key="batch", max_iter_harmony=10, theta=1.5)

            print("check integration: ")
            #sc.pp.neighbors(adata_combined, use_rep="X_pca_harmony")
            #sc.tl.umap(adata_combined)
            # Visualize UMAP
            #sc.pl.umap(adata_combined, color=["batch"])

            # Split back into reference and query
            _adata_ref = adata_combined[adata_combined.obs["batch"] == "ref"]
            adata_query = adata_combined[adata_combined.obs["batch"] == "query"]
            
            del adata_combined 
            gc.collect()  # Force garbage collection

            return _adata_ref, adata_query
    


        if self.integration_method == "combat":

            adata_combined = _adata_ref.concatenate(adata_query, batch_categories=["ref", "query"],  batch_key="batch")

            adata_combined.X = adata_combined.X.astype(np.float32) # convert to float32

            del adata_query, _adata_ref # free memory

            sc.pp.combat(adata_combined, key='batch')
            
            #print("check integration: ")
            #sc.pp.neighbors(adata_combined)
            #sc.tl.umap(adata_combined)
            #adata_combined.write("saves\integrated_adata_combat.h5ad")

            #umap_coords = pd.DataFrame(
            #    adata_combined.obsm["X_umap"],
            #    columns=["UMAP1", "UMAP2"],
            #    index=adata_combined.obs_names)
            
            #umap_coords.to_csv(r"saves\umap_coordinates_combat.csv")  # Save UMAP coordinates to CSV

            # Visualize UMAP
            #sc.pl.umap(adata_combined, color=["batch"])
            #sc.pl.umap(adata_combined, color=['Cell_subtype'])
            #sc.pl.umap(adata_combined, color=['celltype_level3'])

            # Split back into reference and query
            _adata_ref = adata_combined[adata_combined.obs["batch"] == "ref"]
            adata_query = adata_combined[adata_combined.obs["batch"] == "query"]
            
            del adata_combined 
            gc.collect()  # Force garbage collection

            return _adata_ref, adata_query
        

        if self.integration_method == "mnn":

            adata_combined = _adata_ref.concatenate(adata_query, batch_categories=["ref", "query"],  batch_key="batch")

            del adata_query, _adata_ref # free memory

            sce.pp.mnn_correct(adata_combined, key='batch')

            print("check integration: ")
            sc.pp.neighbors(adata_combined)
            sc.tl.umap(adata_combined)
            #print("check integration: ")
            # Color by batch to see batch correction
            #sc.pl.umap(adata_combined, color="batch")
            
            
            #sc.pp.pca(adata_combined)
            #sc.pp.neighbors(adata_combined, use_rep="X_pca")
            #sc.tl.umap(adata_combined)

            print("check integration: ")

            #sc.pl.umap(adata_combined, color=["batch", "Cell_subtype"], wspace=0.4)

            #sc.tl.leiden(adata_combined, resolution=0.5)
            #sc.pl.umap(adata_combined, color=["batch", "leiden"])

            # Split back into reference and query
            _adata_ref = adata_combined[adata_combined.obs["batch"] == "ref"]
            adata_query = adata_combined[adata_combined.obs["batch"] == "query"]
            
            del adata_combined 
            gc.collect()  # Force garbage collection

            return _adata_ref, adata_query

        return ValueError("Invalid integration method. Choose from False, 'combat', 'harmony', or 'mnn'.")
    

            
    



    def load_data(self, reference_data_path,
                        adata_query, 
                        column_to_predict, 
                        cell_types_excluded_treshold = 0, 
                        batch_correction=None) -> None:


        self.integration_method = batch_correction
        self.cell_types_excluded_treshold:int = cell_types_excluded_treshold
        self.cell_types_excluded:list[str] = []



        print("Loading reference data...")

        #read single cell reference data model:  
        adata = sc.read_h5ad(reference_data_path)
        try:
            adata.var_names =  adata.var["features"]
        except KeyError:
            raise ValueError("please, rename the column in adata.var with the names of the genes as 'features' .")

        # read available cell data:
        #model_features =  adata.var

        ## This is for cheching integration method
        #adata.obs.rename(columns={'column_to_predict': 'Cell_subtype'}, inplace=True)
        #column_to_predict = 'Cell_subtype'
        
        print("Reference data loaded.")

        print("Detecting common genes...")
        # Find intersection of genes between reference and query datasets

        #self.get_common_genes(model_features, list_of_available_features)
       
        common_genes = adata.var_names.intersection(adata_query.var_names)
        if common_genes.empty:
            raise ValueError("No common genes found between reference and query datasets.")
        
        # Subset the datasets to only include common genes
        adata = adata[:, common_genes]
        adata_query = adata_query[:, common_genes]

        
        # Subset both datasets to include only the common genes
        #adata = adata[:, self.common_genes_model_indices]
        #adata_unnanotated = adata_unnanotated[:, common_genes_available_indices]

        print(f"Common genes detected: {len(common_genes)}")

        if self.integration_method != None:

            print("Running integration....")
            
            adata, adata_query = self.integrate_data(adata, adata_query)  

            print("Data integrated!")

            # Extract feature matrix from reference data
            if self.integration_method == "combat":
                self.obs_data = adata.X.astype(np.float32)
            
            if self.integration_method == "mnn":
                self.obs_data = adata.X.astype(np.float32)

            if self.integration_method == "harmony":
                self.obs_data = adata.obsm['X_pca_harmony'].astype(np.float32) # we train over the PCA corrected data

        else:
            self.obs_data = adata.X.astype(np.float32)


        #print(f"Data shape: {self.obs_data.shape}") # (cells, genes)
        
        # If the data is sparse, convert it to dense format (In the future, we can use sparse tensors)
        if not isinstance(self.obs_data, (np.ndarray, torch.Tensor)):
            
            self.obs_data = self.obs_data.toarray().astype(np.float32)

        # Ensure the column to predict exists in metadata
        if column_to_predict not in adata.obs:
            raise ValueError(f"Column '{column_to_predict}' not found in adata.obs.")

        # Extract labels
        self.labels = adata.obs[column_to_predict].values

        

        # Analyze label distribution
        label_distribution = adata.obs[column_to_predict].value_counts()
        print("\nLabel distribution:")
        print(label_distribution)
        

        # Identify and store cell types to exclude
        self.cell_types_excluded = label_distribution[label_distribution < cell_types_excluded_treshold].index.tolist()
        print(f"\nExcluding cell types with fewer than {cell_types_excluded_treshold} cells:")
        print(self.cell_types_excluded)
        

        # Exclude cells
        self.exclude_cells()
        self.num_samples = len(self.labels)
        
        # Encode labels
        self.unique_labels, self.labels_encoded = np.unique(self.labels, return_inverse=True)

        
       # Split the data into train, test, val, cal
        if self.do_test:
            data_remaining, self.data_test, labels_remaining, self.labels_test = train_test_split(
                self.obs_data, self.labels_encoded, stratify=self.labels_encoded, test_size=0.2)

            data_remaining, self.data_val, labels_remaining, self.labels_val = train_test_split(
                data_remaining, labels_remaining, stratify=labels_remaining, test_size=0.2)
              
        else:
            data_remaining, self.data_val, labels_remaining, self.labels_val = train_test_split(
                self.obs_data, self.labels_encoded, stratify=self.labels_encoded, test_size=0.25)



        self.data_train, self.data_cal, self.labels_train, self.labels_cal = train_test_split(
            data_remaining, labels_remaining, stratify=labels_remaining, test_size=0.40)


        print(f"Train data shape: {self.data_train.shape}")
        print(f"Validation data shape: {self.data_val.shape}")
        print(f"Calibration data shape: {self.data_cal.shape}")

        
        if self.do_test:
            print(f"Test data shape: {self.data_test.shape}")

        print("Data loaded")

        return adata_query
    


    def fit_OOD_detector(self, alpha_OOD, delta_OOD ) -> None:

        print("Training OOD detector...")

        self.alpha_OOD = alpha_OOD
        self.delta_OOD = delta_OOD

        network_architecture_ = {
            "hidden_sizes": [164,124,64],
            "dropout_rates": [0.4, 0.3, 0.4],
            "learning_rate": 0.0001,
            "batch_size": 240,
            "n_epochs": 62}

        model_oc = AEOutlierDetector(input_dim=self.data_train.shape[1], network_architecture=network_architecture_)

        self.OOD_detector = Annomaly_detector(oc_model = model_oc, delta=self.delta_OOD)

        self.OOD_detector.fit(self.data_train, self.data_cal )

        print("OOD detector trained!")
        
        return None
    


    def define_architecture(self, hidden_sizes, dropout_rates ) -> None:

        # Define network architecture 
        
        input_size = self.obs_data.shape[1]  # Number of input features (genes)
        output_size = len(np.unique(self.labels_encoded)) # Number of unique classes (cell types)
        print("Input size: ", input_size)
        print("Output size: ", output_size)

        # Initialize model, loss function, and optimizer
        self.model = NNClassifier(input_size, hidden_sizes, output_size, dropout_rates=dropout_rates)
        

        return None
    


    def fit(self, lr=0.001, save_path=None) -> None:


        # Convert data and labels to PyTorch tensors
        self.data_train = torch.from_numpy(self.data_train).float()
        self.labels_train = torch.from_numpy(self.labels_train).long()

        self.data_val = torch.from_numpy(self.data_val).float()
        self.labels_val = torch.from_numpy(self.labels_val).long()


        # Create PyTorch datasets
        self.train_dataset = CustomDataset(self.data_train, self.labels_train)
        self.val_dataset = CustomDataset(self.data_val, self.labels_val)

        # Create PyTorch data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        # Move model to the appropriate device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Compute class weights for handling class imbalance
        classes = np.unique(self.labels_train.numpy())
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=self.labels_train.numpy())
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        # Define loss criterion and optimizer
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        num_epochs = self.epoch  # Assuming self.epoch is defined

        for epoch in range(num_epochs):
            self.model.train()
            train_loss_total = 0

            for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader , 1):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)


                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss_total += loss.item()

            # Compute average training loss
            avg_train_loss = train_loss_total / len(self.train_loader)

            # Validation phase
            self.model.eval()
            val_loss_total = 0
            with torch.no_grad():
                for X_val_batch, y_val_batch in self.val_loader:
                    X_val_batch, y_val_batch = X_val_batch.to(self.device), y_val_batch.to(self.device)

                    val_outputs = self.model(X_val_batch)
                    val_loss = self.criterion(val_outputs, y_val_batch)
                    val_loss_total += val_loss.item()



            avg_val_loss = val_loss_total / len(self.val_loader)

            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

           

        # Save model
        if save_path is not None:
            torch.save(self.model, save_path)
            print(f"Model saved to {save_path}")

        return None

   
    
    def calibrate(self, non_conformity_function, alpha = 0.05, predictors = None) -> None:  

        print("Calibrating the model...")

        self.data_cal = torch.from_numpy(self.data_cal).float()
        self.labels_cal = torch.from_numpy(self.labels_cal).long()  

        self.cal_dataset = CustomDataset(self.data_cal, self.labels_cal)

        self.cal_loader = DataLoader(self.cal_dataset, batch_size=self.batch_size, shuffle=False)
        
        self.conformal_prediction = True   

        if not isinstance(alpha, (list, tuple)):
            self.alphas = [alpha]
        else:
            self.alphas = alpha
    
        
        self.model.eval()
        
        self.conformal_predictors: dict[float, any]  = {}

        # Score function and conformal predictor
        for alpha in self.alphas:
            score_function = non_conformity_function

            if predictors  == "mondrian":
                conformal_predictor =  ClassWisePredictor(score_function, self.model)

            elif predictors == "cluster":
                conformal_predictor =  ClusteredPredictor(score_function, self.model)

            else:                           
                conformal_predictor = SplitPredictor(score_function, self.model)

            conformal_predictor.calibrate(self.cal_loader, alpha)
            self.conformal_predictors[alpha] = conformal_predictor

        print("Model calibrated.")

        return None
    

    def test(self) -> None:

        self.model.eval()   
        
        self.data_test = torch.from_numpy(self.data_test).float()
        self.labels_test = torch.from_numpy(self.labels_test).long()

        self.test_dataset = CustomDataset(self.data_test, self.labels_test)

        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        # Initialize lists to collect results
        true_labels_list = []
        predicted_labels_list = []
        

        with torch.no_grad():
            for test_features, test_labels in self.test_loader:

                test_features = test_features.to(self.device)
                test_labels = test_labels.to(self.device)

                # Make predictions
                test_outputs = self.model(test_features)
                _, predicted = torch.max(test_outputs, 1)  # Get predicted class indices

                true_labels_list.append(test_labels)
                predicted_labels_list.append(predicted)

        

        # Concatenate results
        self.true_labels_ = torch.cat(true_labels_list).cpu().numpy()
        self.predicted_labels_ = torch.cat(predicted_labels_list).cpu().numpy()

        # Overall performance scores
        self.accuracy_ = accuracy_score(self.true_labels_, self.predicted_labels_)
        self.precision_ = precision_score(self.true_labels_, self.predicted_labels_, average='weighted')
        self.recall_ = recall_score(self.true_labels_, self.predicted_labels_, average='weighted')
        self.f1_score_ = f1_score(self.true_labels_, self.predicted_labels_, average='weighted')

        # Metrics by class
        self.precision_per_class_ = precision_score(self.true_labels_, self.predicted_labels_, average=None)
        self.recall_per_class_ = recall_score(self.true_labels_, self.predicted_labels_, average=None)
        self.f1_per_class_ = f1_score(self.true_labels_, self.predicted_labels_, average=None)

        # Get unique class labels
        self.class_labels_ = sorted(set(self.true_labels_))  # Assumes labels are integers

        # Store metrics per class in a dictionary for easy access
        self.metrics_by_class_ = {
            label: {
                "Precision": self.precision_per_class_[i],
                "Recall": self.recall_per_class_[i],
                "F1 Score": self.f1_per_class_[i],
            }
            for i, label in enumerate(self.class_labels_)
        }

        # Print overall scores
        print(f"\nTest Accuracy: {self.accuracy_ * 100:.2f}%")
        print(f"Precision (Weighted Average): {self.precision_ * 100:.2f}%")
        print(f"Recall (Weighted Average): {self.recall_ * 100:.2f}%")
        print(f"F1 Score (Weighted Average): {self.f1_score_ * 100:.2f}%")

        # Print scores per class
        print("\nScores by Class:")
        for label, metrics in self.metrics_by_class_.items():
            print(f"  Class {label}: Precision: {metrics['Precision']:.2f}, Recall: {metrics['Recall']:.2f}, F1 Score: {metrics['F1 Score']:.2f}")


        # Perform conformal prediction if enabled
        if self.conformal_prediction:
            print("\nPerforming conformal prediction...")

            # Evaluate with conformal predictor
            self.InDis_prediction_sets_ = {}
            self.InDis_results_ = {}

            for key in self.conformal_predictors:
                
                prediction_sets = []
                labels_list = []
                with torch.no_grad():
                    for examples in self.test_loader:
                        tmp_x, tmp_label = examples[0].to(self.device), examples[1].to(self.device)
                        prediction_sets_batch = self.conformal_predictors[key].predict(tmp_x)
                        prediction_sets.extend(prediction_sets_batch)
                        labels_list.append(tmp_label)
                
                val_labels = torch.cat(labels_list)
                
                prediction_sets_tensor = torch.stack(prediction_sets).float().to(self.device)

                # Similar to .evalueate() method in the original code but refined
                result = {"coverage_rate": self._metric('coverage_rate')(prediction_sets_tensor, val_labels),
                            "average_size": self._metric('average_size')(prediction_sets_tensor, val_labels),
                            "prediction_set": prediction_sets,
                            "targets": val_labels.tolist()}

                #Calculate size distribution
                
                prediction_set_sizes = [(pred_set == 1).sum().item() for pred_set in result['prediction_set']]
                size_distribution = {size: prediction_set_sizes.count(size) for size in set(prediction_set_sizes)}
                
                self.InDis_results_[key] = {'Coverage_rate': result['coverage_rate'],
                                            'Average_size': result['average_size'],
                                            'Size_distribution': size_distribution}
                
                self.InDis_prediction_sets_[key] = [[set,target] for set,target in zip(result['prediction_set'],result['targets']) ]

                # Print results for debugging
                print(f"\nConformal predictor {key} - Coverage Rate: {result['coverage_rate']}, Average Size: {result['average_size']}")
                print(f"Size Distribution: {size_distribution}")
        

        ## OUT OF DISTRIBUTION PREDICTION



        if len(self.cell_types_excluded) > 0:

            self.OOD_data_test = torch.from_numpy(self.OOD_data).float()
            placeholder_labels = torch.from_numpy(np.full(len(self.OOD_data), -1)).float()

            self.OOD_test_dataset = CustomDataset(self.OOD_data_test, placeholder_labels)

            self.OOD_test_loader = DataLoader(self.OOD_test_dataset, batch_size=self.batch_size, shuffle=False)

            unknown_predicted_labels_list: list = []


            print("\nPerforming out-of-distribution prediction...")

            with torch.no_grad():
                for OOD_test_features, OOD_test_labels in self.OOD_test_loader:

                    OOD_test_features = OOD_test_features.to(self.device)
                    OOD_test_labels = OOD_test_labels.to(self.device)

                    # Make predictions
                    unknown_test_outputs = self.model(OOD_test_features)
                    _, unknown_predicted = torch.max(unknown_test_outputs, 1)  # Get predicted class indices
            
                
                    unknown_predicted_labels_list.append(unknown_predicted) 

            self.unknown_predicted_labels_ = torch.cat(unknown_predicted_labels_list).cpu().numpy()
            
            
            print(f"Classical Predicted labels: {self.unknown_predicted_labels_}")

            
            if self.conformal_prediction:
                
                print("\nPerforming conformal prediction...")
                
                
                # Evaluate with conformal predictor
                self.OOD_prediction_sets_ = {}
                self.OOD_results_ = {}
                for key in self.conformal_predictors:

                    prediction_sets = []
                    labels_list = []
                    with torch.no_grad():
                        for examples in self.OOD_test_loader:
                            tmp_x, tmp_label = examples[0].to(self.device), examples[1].to(self.device)
                            prediction_sets_batch = self.conformal_predictors[key].predict(tmp_x)
                            prediction_sets.extend(prediction_sets_batch)
                            labels_list.append(tmp_label)
                            
                    val_labels = torch.cat(labels_list)
                    
                    prediction_sets_tensor = torch.stack(prediction_sets).float().to(self.device)
                    
                    # Similar to .evalueate() method in the original code but refined 
                    result = {"coverage_rate": self._metric('coverage_rate')(prediction_sets_tensor, val_labels.long()),
                            "average_size": self._metric('average_size')(prediction_sets_tensor, val_labels.long()),
                            "prediction_set": prediction_sets,
                            "targets": val_labels.tolist()}

                    prediction_set_sizes = [(pred_set == 1).sum().item() for pred_set in result['prediction_set']]
                    size_distribution = {size: prediction_set_sizes.count(size) for size in set(prediction_set_sizes)}

                    self.OOD_results_[key] = {'Coverage_rate': result['coverage_rate'],
                                             'Average_size': result['average_size'], 
                                             'Size_distribution': size_distribution}
                    
                    self.OOD_prediction_sets_[key] = [[set,target] for set,target in zip(result['prediction_set'],result['targets']) ]
                    
                    mapped_predictions = [
                                [str(self.unique_labels[idx.item()]) for idx in tensor.cpu().nonzero(as_tuple=True)[0]]
                                    for tensor in result['prediction_set'] ]
                    
                    print("\nConformal predictor" ,key, "\nResults per OOD sample: ",  mapped_predictions)
                    print(f"Size Distribution: {size_distribution}")

            return None
        
        
    def predict(self, data) -> None:

        self.prediction_sets:dict = {}
        self.predicted_labels = None

        print("\nPerforming OOD detection...")
        
        #data_filtered = self.OOD_detector.predict_proba(data).copy()
        #data_OOD_mask = (data_filtered[:, 1] > 0.8).astype(int)

        self.OOD_detector.predict_cond_pvalues(data,  method="MC")
        data_OOD_mask, _ = self.OOD_detector.evaluate_in_batches(alpha=self.alpha_OOD, lambda_par=0.5, use_sbh=True)
        #data_OOD_mask = ().astype(int)
        
        print(f"OOD samples detected: {data_OOD_mask.sum()}")

        
        ## CLASSICAL PREDICTION

        # If the data is sparse, convert it to dense format
        if not isinstance(data, (np.ndarray, torch.Tensor)):
            data = data.toarray().astype(np.float32)

        
        #data = data[:, self.common_genes_available_indices]

        data = torch.from_numpy(data).float()

        data = data.to(self.device)

        # Ensure the model is in evaluation mode
        self.model.eval()

        # Make predictions
        with torch.no_grad():
            predictions = self.model(data)
            
        # Convert predictions to numpy array if needed
        soft_max_predictions = predictions.cpu().numpy()
        predicted_indices = torch.argmax(predictions, dim=1).cpu().numpy()

        # Map predicted indices back to original labels and map the OOD samples to "OOD"
        self.predicted_labels = self.unique_labels[predicted_indices].copy() 
        self.predicted_labels[data_OOD_mask == True] = "OOD"
        
        # CONFORMAL PREDICTION

        placeholder_labels = torch.from_numpy(np.full(len(data), -1)).float()
        data_cust = CustomDataset(data, placeholder_labels)
        data_cp= DataLoader(data_cust, batch_size=self.batch_size, shuffle=False)

        
        print("\nPerforming conformal prediction...")
        if self.conformal_prediction:

            for key in self.conformal_predictors:

                prediction_sets = []
                labels_list = []
                with torch.no_grad():
                    for examples in data_cp:
                        tmp_x, tmp_label = examples[0].to(self.device), examples[1].to(self.device)
                        prediction_sets_batch = self.conformal_predictors[key].predict(tmp_x)
                        prediction_sets.extend(prediction_sets_batch)
                        labels_list.append(tmp_label)
                            
                val_labels = torch.cat(labels_list)
                    
                prediction_sets_tensor = torch.stack(prediction_sets).float().to(self.device)
                    
                # Similar to .evalueate() method in the original code but refined 
                CP_result = {"coverage_rate": self._metric('coverage_rate')(prediction_sets_tensor, val_labels.long()),
                            "average_size": self._metric('average_size')(prediction_sets_tensor, val_labels.long()),
                            "prediction_set": prediction_sets,
                            "targets": val_labels.tolist()}

                #CP_result = self.conformal_predictors[key].evaluate(data_cp)
                

                mapped_predictions = [
                                [str(self.unique_labels[idx.item()]) for idx in tensor.cpu().nonzero(as_tuple=True)[0]]
                                    for tensor in CP_result['prediction_set'] ]
    
                
                
                for i, is_ood in enumerate(data_OOD_mask):
                    if is_ood == True:
                        mapped_predictions[i] = ["OOD"]

                self.prediction_sets[key] = mapped_predictions
                

        print("\nPerforming conformal prediction: Done\n")
   

        return None
        
    


if __name__ == "__main__":


    obs_data_path = 'HumanLung_5K_HVG.h5ad'
    column_to_predict = 'celltype_level3'   # this is the label column
    calibration_taxonomy = 'celltype_level2'

    cell_types_excluded_treshold = 15

    hidden_sizes = [ 256, 128,72, 64]
    dropout_rates = [ 0.4, 0.3, 0.4, 0.25]
    learning_rate = 0.0005

    alphas = [0.01, 0.05, 0.1, 0.2]

    non_conformity_function = None

    ##------

    classifier = SingleCellClassifier(epoch=10, batch_size = 1128)

    classifier.load_data(obs_data_path,column_to_predict, cell_types_excluded_treshold, print_metadata = False)
    classifier.define_architecture(hidden_sizes, dropout_rates)
    classifier.fit(lr=learning_rate, save_path="5K_HVG_model.pth")
    classifier.calibrate(non_conformity_function, alpha = alphas, predictors = "cluster")
    classifier.test()

    #classifier.predict(data = None)