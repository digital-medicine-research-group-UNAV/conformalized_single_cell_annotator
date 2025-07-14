


# Import necessary libraries


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
from sklearn.utils.class_weight import compute_class_weight


from utils import CellTypistWrapper, ScmapWrapper
from typing import Union, List, Optional

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from torchcp.classification.predictor import SplitPredictor, ClassWisePredictor, ClusteredPredictor
from torchcp.classification.utils.metrics import Metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau

from annomaly_detector import Annomaly_detector, AEOutlierDetector

import os
cwd = os.getcwd()
print("Current kernel CWD:", cwd)


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

    def __init__(
        self,
        classifier_model: str = "torch_net",
        epoch: int = 4,
        batch_size: int = 64,
        do_test: bool = True,
        random_state: Optional[int] = None) -> None:

        self.classifier_model = classifier_model
        self.epoch = epoch
        self.batch_size = batch_size
        self.do_test = do_test
        self.random_state = random_state

        self.alpha_OOD = None
        self.delta_OOD = 0.1

        self.conformal_prediction = False

        self.excluded_positions: dict = {}
        self.labels_index: list = []
        self.common_gene_names: list = []
        self.unique_labels: list = []

        self._metric = Metrics()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    def lognormalizate_adata(self) -> None:

       

        if self.layer is None:
            sc.pp.normalize_total(self.adata_train, target_sum=1e4)  
            sc.pp.log1p(self.adata_train)

            if self.do_test:
                sc.pp.normalize_total(self.adata_test, target_sum=1e4)
                sc.pp.log1p(self.adata_test)
            
            sc.pp.normalize_total(self.adata_cal, target_sum=1e4)
            sc.pp.log1p(self.adata_cal)
            print("Data log-normalized.")
        

        else:
            sc.pp.normalize_total(self.adata_train, target_sum=1e4, layer=self.layer)  
            sc.pp.log1p(self.adata_train,layer=self.layer)

            if self.do_test:
                sc.pp.normalize_total(self.adata_test, target_sum=1e4, layer=self.layer)
                sc.pp.log1p(self.adata_test,layer=self.layer)
            
            sc.pp.normalize_total(self.adata_cal, target_sum=1e4, layer=self.layer)
            sc.pp.log1p(self.adata_cal, layer=self.layer)
            print("Data log-normalized.")

        

        return None 


    def load_data(self,
                    reference_data_path:  Union[str, ad.AnnData],
                    adata_query,
                    column_to_predict: str,
                    cell_types_excluded_treshold: Union[int, List[str]] = 0,
                    obsm = None,
                    layer = None,
                    gene_column_name: str = "features") -> None:

        self.column_to_predict = column_to_predict
        self.obsm = obsm
        self.layer = layer
        self.cell_types_excluded_treshold = cell_types_excluded_treshold
        self.cell_types_excluded:list[str] = []

       

        print("Loading reference data...")
        
        #read single cell reference data model: 
        if isinstance(reference_data_path, str):
            adata = sc.read_h5ad(reference_data_path)
        elif isinstance(reference_data_path, ad.AnnData):
            adata = reference_data_path
        else:
            raise TypeError("reference_data must be a file path (str) or an AnnData object.")

        

        try:
            adata.var_names =  adata.var[gene_column_name]
        except KeyError:
            raise ValueError("please, rename the column with gene names with the same name in both query and reference .")

        # Ensure the column to predict exists in metadata
        if self.column_to_predict not in adata.obs:
            raise ValueError(f"Column '{self.column_to_predict}' not found in adata.obs.")
        
        print(f"Reference data shape: {adata.shape}\n") # (cells, genes)
        
        print("Reference data loaded.")

       

        # Extract labels
          
        label_distribution = adata.obs[self.column_to_predict].value_counts()
        self.label_distribution = label_distribution
        print("\nInitial reference data label distribution:")
        print(label_distribution, len(label_distribution))

        try:
            print("\nQuery data label distribution:")
            print(adata_query.obs[self.column_to_predict].value_counts())
        except KeyError:
            raise ValueError(f"Column '{self.column_to_predict}' not found in adata_query.obs.")
    
        

        # Identify and store cell types to exclude
        if isinstance(cell_types_excluded_treshold, int): 
            self.cell_types_excluded = label_distribution[label_distribution < cell_types_excluded_treshold].index.tolist()

        elif isinstance(cell_types_excluded_treshold, list):
            self.cell_types_excluded = cell_types_excluded_treshold
        
        else:
            raise ValueError("Cell_types_excluded_treshold must be an integer of minimum cells to exclude or a list of cell types to exclude.")
        

        print(f"\nExcluding cell types in: {cell_types_excluded_treshold}:")
        print(self.cell_types_excluded)

       

        mask = ~adata.obs[self.column_to_predict].isin(self.cell_types_excluded)
        adata = adata[mask].copy()

        self.labels   = adata.obs[self.column_to_predict].values.tolist()
        self.labels_index = list(range(adata.n_obs))
        
        self.num_samples = len(self.labels)
        
        # Encode labels
        self.unique_labels, self.labels_encoded = np.unique(self.labels, return_inverse=True)
                


        if self.classifier_model != "celltypist":
            print("Detecting common genes...")
        
            common_genes = adata.var_names.intersection(adata_query.var_names)
            if common_genes.empty:
                raise ValueError("No common genes found between reference and query datasets.")
            
            # Subset the datasets to only include common genes
            adata = adata[:, common_genes]
            adata_query = adata_query[:, common_genes]

            print(f"Common genes detected: {len(common_genes)}")
       
        

        all_idx = np.arange(adata.n_obs)  # 0 … n_obs-1 
        

        if self.do_test:
            
            idx_remain, idx_test = train_test_split(
                all_idx,
                test_size=0.10,
                shuffle=True,
                stratify=self.labels_encoded,
                random_state=self.random_state)   

           
            idx_remain, idx_val = train_test_split(
                idx_remain,
                test_size=0.10,
                stratify=self.labels_encoded[idx_remain],
                random_state=self.random_state
            ) 

        else:
            
            idx_remain, idx_val = train_test_split(
                all_idx,
                test_size=0.10,
                stratify=self.labels_encoded,
                random_state=self.random_state
            )
        
        # Finally, calibration set is empty in this case
        
        idx_train, idx_cal = train_test_split(
            idx_remain,
            test_size=0.45,
            stratify=self.labels_encoded[idx_remain],
            random_state=self.random_state
        )
        

        
        # Subsets
        self.adata_train = adata[idx_train].copy()  
        self.adata_val   = adata[idx_val].copy()
        self.adata_test  = adata[idx_test].copy() if self.do_test else None
        self.adata_cal   = adata[idx_cal].copy()
        

        if self.classifier_model == "celltypist":
            self.lognormalizate_adata()
        
        

        self.labels_train  = torch.from_numpy(self.labels_encoded[idx_train]).long()
        self.labels_val    = torch.from_numpy(self.labels_encoded[idx_val]).long()
        self.labels_test   = torch.from_numpy(self.labels_encoded[idx_test]).long() if self.do_test else None
        self.labels_cal    = torch.from_numpy(self.labels_encoded[idx_cal]).long()

        

        print(f"Train data shape: {self.adata_train.shape}")
        print(f"Validation data shape: {self.adata_val.shape}")
        if self.do_test:
            print(f"Test data shape: {self.adata_test.shape}")
        print(f"Calibration data shape: {self.adata_cal.shape}\n")

       
        
        if self.obsm is None and self.layer is None:
            self.data_train = self.adata_train.X.astype(np.float32)
            self.data_val = self.adata_val.X.astype(np.float32)
            if self.do_test:
                self.data_test = self.adata_test.X.astype(np.float32)
            self.data_cal = self.adata_cal.X.astype(np.float32)

        elif self.layer is None and self.obsm is not None:
            self.data_train = self.adata_train.obsm[self.obsm].astype(np.float32)
            self.data_val = self.adata_val.obsm[self.obsm].astype(np.float32)
            if self.do_test:
                self.data_test = self.adata_test.obsm[self.obsm].astype(np.float32)
            self.data_cal = self.adata_cal.obsm[self.obsm].astype(np.float32)
        
        elif self.obsm is None and self.layer is not None:
            self.data_train = self.adata_train.layers[self.layer].astype(np.float32)
            self.data_val = self.adata_val.layers[self.layer].astype(np.float32)
            if self.do_test:
                self.data_test = self.adata_test.layers[self.layer].astype(np.float32)
            self.data_cal = self.adata_cal.layers[self.layer].astype(np.float32)

        else:
            raise ValueError("Fatal internal error: Please, specify either obsm or layer, not both at the same time.")
        

        
        
        # If the data is sparse, convert it to dense format (In the future, we can use sparse tensors)
        if not isinstance(self.data_train, (np.ndarray, torch.Tensor)):  
            self.data_train = self.data_train.toarray().astype(np.float32)

        
        
        if not isinstance(self.data_val, (np.ndarray, torch.Tensor)):
            self.data_val = self.data_val.toarray().astype(np.float32)
        
        if self.do_test and not isinstance(self.data_test, (np.ndarray, torch.Tensor)):
            self.data_test = self.data_test.toarray().astype(np.float32)
        
       
       
        if not isinstance(self.data_cal, (np.ndarray, torch.Tensor)):
            self.data_cal = self.data_cal.toarray().astype(np.float32)

        self.is_outlier = np.array(adata_query.obs[column_to_predict].isin(self.cell_types_excluded))
        print("\nData loaded!")

        return adata_query
    


    def fit_OOD_detector(self, pvalues,  alpha_OOD, delta_OOD,
                        hidden_sizes_OOD, dropout_rates_OOD,
                        learning_rate_OOD, batch_size_OOD, n_epochs_OOD,
                        obsm_OOD ) -> None:

        self.obsm_OOD = obsm_OOD

        print("\nTraining OOD detector with alpha:", alpha_OOD)

        self.alpha_OOD = alpha_OOD
        self.delta_OOD = delta_OOD

        network_architecture_OOD = {
            "hidden_sizes": hidden_sizes_OOD, 
            "dropout_rates": dropout_rates_OOD,
            "learning_rate": learning_rate_OOD,
            "batch_size": batch_size_OOD,
            "n_epochs": n_epochs_OOD,
            "patience":      9,
            "noise_level":   0.1,
            "lambda_sparse": 1e-3}
        

        if self.obsm_OOD is not None:
            self.data_train_OOD = self.adata_train.obsm[self.obsm_OOD].astype(np.float32)
            self.data_val_OOD = self.adata_val.obsm[self.obsm_OOD].astype(np.float32)
            self.data_cal_OOD = self.adata_cal.obsm[self.obsm_OOD].astype(np.float32)
        else:
            self.data_train_OOD = self.adata_train.X.astype(np.float32)
            self.data_val_OOD = self.adata_val.X.astype(np.float32)
            self.data_cal_OOD = self.adata_cal.X.astype(np.float32)


        model_oc = AEOutlierDetector(input_dim=self.data_train_OOD.shape[1], network_architecture=network_architecture_OOD)
        
        self.OOD_detector = Annomaly_detector(pvalues, oc_model = model_oc, delta=self.delta_OOD)

        self.OOD_detector.fit(self.data_train_OOD, self.data_val_OOD, self.data_cal_OOD )

        

        print("OOD detector trained!")
        
        return None
    


    def define_architecture(self, hidden_sizes, dropout_rates ) -> None:

        # Define network architecture 
        
        input_size = self.data_train.shape[1]  # Number of input features (genes)
        output_size = len(np.unique(self.labels_encoded)) # Number of unique classes (cell types)
        print("Input size: ", input_size)
        print("Output size: ", output_size)

        # Initialize model, loss function, and optimizer
        self.model = NNClassifier(input_size, hidden_sizes, output_size, dropout_rates=dropout_rates)
        

        return None
    


    def fit_network(self, lr=0.001, save_path=None) -> None:

        
        # Convert data and labels to PyTorch tensors
        self.data_train = torch.from_numpy(self.data_train).float()
        
        self.data_val = torch.from_numpy(self.data_val).float()
        
        # Create PyTorch datasets
        self.train_dataset = CustomDataset(self.data_train, self.labels_train)
        self.val_dataset = CustomDataset(self.data_val, self.labels_val)


        # Compute sample weights for WeightedRandomSampler
        labels_np = self.labels_train.numpy()
        class_counts = np.bincount(self.labels_train.numpy())
        inv_class_counts = 1.0 / class_counts
        sample_weights = inv_class_counts[self.labels_train.numpy()]  # one weight per training sample
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # Create PyTorch data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler, drop_last=True )
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True )
        
        # Move model to the appropriate device (CPU or GPU)
        self.model.to(self.device)

        # Compute class weights for handling class imbalance
        classes = np.unique(self.labels_train.numpy())
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=self.labels_train.numpy())
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        # Define loss criterion and optimizer
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3)

        num_epochs = self.epoch  # Assuming self.epoch is defined


        # Variables for early stopping
        best_val_loss = float('inf')
        epochs_no_improve = 0
        early_stopping_patience = 6

        

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

            # Step the scheduler based on the validation loss
            scheduler.step(avg_val_loss)

            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping triggered!")
                self._is_fitted = True
                break
           

        # Save model
        if save_path is not None:
            torch.save(self.model, save_path)
            print(f"Model saved to {save_path}")

        return None




    def fit_celltypist(self, column_to_predict) -> None:
        
        print("Training celltypist model...")

        self.model = CellTypistWrapper(cal_adata_genes = self.adata_train.var_names.tolist())

        self.model.to(self.device)

        self.model.train()


        self.model.train_model(self.adata_train, label_key=column_to_predict, layer = self.layer)

        return None
    


    def fit_scmap(self, column_to_predict) -> None:
        
        print("Training scmap model...")

        self.model = ScmapWrapper(cal_adata_genes = self.adata_train.var_names.tolist())

        self.model.to(self.device)

        self.model.train()


        self.model.train_model(self.adata_train, label_key=column_to_predict, layer = self.layer, obsm = self.obsm)

        return None



    def calibrate(self, non_conformity_function, alpha = 0.05, predictors = "standard") -> None:  

        print("Calibrating the model...")

        self.predictors = predictors

        self.data_cal = torch.from_numpy(self.data_cal).float()
        #self.labels_cal = torch.from_numpy(self.labels_cal).long()  

        
        self.cal_dataset = CustomDataset(self.data_cal, self.labels_cal)

        if self.classifier_model != "torch_net":
            self.batch_size = self.data_cal.shape[0]  # Use the entire calibration set as a single batch

        self.cal_loader = DataLoader(self.cal_dataset, batch_size=self.batch_size, shuffle=False)
        
        self.conformal_prediction = True   
        
        

        if not isinstance(alpha, (list, tuple)):
            self.alphas = [alpha]
        else:
            self.alphas = alpha
    
        
        self.model.eval()
        
        self.conformal_predictors: dict[float, any]  = {}

        if not callable(non_conformity_function):
            raise ValueError("non_conformity_function must be callable.")

        # Score function and conformal predictor
        for alpha in self.alphas:

            score_function = non_conformity_function
            
            if self.predictors  == "classwise":
                print("Using classwise taxonomy")
                conformal_predictor =  ClassWisePredictor(score_function, self.model)

            elif self.predictors == "cluster":
                print("Using cluster taxonomy")
                conformal_predictor =  ClusteredPredictor(score_function, self.model, num_clusters=int(len(self.label_distribution)/2))

            elif self.predictors == "standard":    
                print("Using standard taxonomy")                       
                conformal_predictor = SplitPredictor(score_function, self.model)
            
            else:
                raise ValueError("Invalid conformal predictor. Choose from 'classwise', 'cluster', or 'standard")

            conformal_predictor.calibrate(self.cal_loader, alpha)
            self.conformal_predictors[alpha] = conformal_predictor


        print("Model calibrated.")

        return None
    

    def test(self) -> None:

        self.model.eval()   
        
        self.data_test = torch.from_numpy(self.data_test).float()
        #self.labels_test = torch.from_numpy(self.labels_test).long()

        self.test_dataset = CustomDataset(self.data_test, self.labels_test)

        if self.classifier_model != "torch_net":
            self.batch_size = self.data_test.shape[0]

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

                # Similar to .evalueate() method in the original torchcp code but adapted to our needs
                result = {"coverage_rate": self._metric('coverage_rate')(prediction_sets_tensor, val_labels),
                          "CovGap": self._metric('CovGap')(prediction_sets_tensor, val_labels, key, len(np.unique(self.labels_encoded)) ),
                          "average_size": self._metric('average_size')(prediction_sets_tensor, val_labels),
                          "prediction_set": prediction_sets,
                          "targets": val_labels.tolist()}

                #Calculate size distribution
                
                prediction_set_sizes = [(pred_set == 1).sum().item() for pred_set in result['prediction_set']]
                size_distribution = {size: prediction_set_sizes.count(size) for size in set(prediction_set_sizes)}
                
                self.InDis_results_[key] = {'Coverage_rate': result['coverage_rate'],
                                            'CovGap': result['CovGap'],
                                            'Average_size': result['average_size'],
                                            'Size_distribution': size_distribution}
                
                self.InDis_prediction_sets_[key] = [[set,target] for set,target in zip(result['prediction_set'],result['targets']) ]

                # Print results for debugging
                print(f"\nConformal predictor {key} - Coverage Rate: {result['coverage_rate']}, CovGap: {result['CovGap']}, Average Size: {result['average_size']}")
                print(f"Size Distribution: {size_distribution}")
        
        
        return None

        
    def predict(self, data, data_OOD) -> None:

        if not isinstance(data, (np.ndarray, torch.Tensor)):
            data = data.toarray().astype(np.float32)
        

        if data_OOD is None:
            data_OOD = data.copy()  # If no OOD data is provided, use the same data for OOD detection
        else:
            if not isinstance(data_OOD, (np.ndarray, torch.Tensor)):
                data_OOD = data_OOD.toarray().astype(np.float32)


        if self.obsm_OOD is not None:
            arr = self.adata_test.obsm[self.obsm_OOD]
            if isinstance(arr, torch.Tensor):
                # detach() and move to CPU before converting to NumPy
                data_test_X_ID = arr.detach().cpu().numpy().astype(np.float32)
            else:
                # already a NumPy array
                data_test_X_ID = arr.astype(np.float32)
        else:
            data_test_X_ID = self.adata_test.X.astype(np.float32)


        self.prediction_sets:dict = {}
        self.predicted_labels = None


        print("\nPerforming OOD detection...")
        
        self.OOD_detector.predict_pvalues(data_OOD,  X_ID=data_test_X_ID)
        if self.OOD_detector.pvalues == "conditional":

            data_OOD_mask, self.accuracy_OOD, self.precision_OOD, self.recall_OOD, self.auroc_OOD = self.OOD_detector.evaluate_conditional_pvalues(alpha=self.alpha_OOD, lambda_par=0.15, use_sbh=True, is_outlier = self.is_outlier)
        else:
            
            data_OOD_mask, self.accuracy_OOD, self.precision_OOD, self.recall_OOD, self.auroc_OOD = self.OOD_detector.evaluate_marginal_pvalues(alpha=self.alpha_OOD, lambda_par=0.15, use_sbh=True, is_outlier = self.is_outlier)
        
        self.alpha_OOD = self.OOD_detector.alpha_OOD
  
        
        print(f"OOD samples detected: {data_OOD_mask.sum()}")

         
        ## CLASSICAL PREDICTION

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

        # save indices of non-OOD samples for recostructing the prediction sets
        non_ood_mask = torch.from_numpy(~data_OOD_mask).to(data.device)
        filtered_data = data[non_ood_mask]  # filtered in-distribution data
        
        # CONFORMAL PREDICTION

        placeholder_labels = torch.from_numpy(np.full(len(filtered_data), -1)).float()
        data_cust = CustomDataset(filtered_data, placeholder_labels)

        if self.classifier_model != "torch_net":
            self.batch_size = filtered_data.shape[0]

        data_cp = DataLoader(data_cust, batch_size=self.batch_size, shuffle=False)

        
        print("\nPerforming conformal prediction...")
        if self.conformal_prediction:

            
            for key in self.conformal_predictors:
                #print("key: ", key)

                non_conformity_scores = []   # it will be overwritten because doesnt change with alpha
                prediction_sets = []
                labels_list = []
                with torch.no_grad():
                    for examples in data_cp:
                        tmp_x, tmp_label = examples[0].to(self.device), examples[1].to(self.device)
                        prediction_sets_batch = self.conformal_predictors[key].predict(tmp_x)

                        ## Get the nonconformity scores of the query data ##
                        x_batch = self.conformal_predictors[key]._model(tmp_x.to(self.conformal_predictors[key]._device)).float()
                        x_batch = self.conformal_predictors[key]._logits_transformation(x_batch).detach()
                        scores = self.conformal_predictors[key].score_function(x_batch).to(self.conformal_predictors[key]._device)

                        
                        #print("qhat: ", self.conformal_predictors[key].q_hat)
                        #print("prediction_set_scores: ", scores)
                        
                        non_conformity_scores.extend(scores)
                        prediction_sets.extend(prediction_sets_batch)
                        labels_list.append(tmp_label)
                            
                val_labels = torch.cat(labels_list)
                    
                prediction_sets_tensor = torch.stack(prediction_sets).float().to(self.device)
                prediction_scores_tensor = torch.stack(non_conformity_scores).float().to(self.device)

                
                    
                # Similar to .evalueate() method in the original code but refined 
                CP_result = {"coverage_rate": self._metric('coverage_rate')(prediction_sets_tensor, val_labels.long()),
                            "average_size": self._metric('average_size')(prediction_sets_tensor, val_labels.long()),
                            "prediction_set": prediction_sets,
                            "targets": val_labels.tolist()}
                

                mapped_predictions_filtered = [
                                [str(self.unique_labels[idx.item()]) for idx in tensor.cpu().nonzero(as_tuple=True)[0]]
                                    for tensor in CP_result['prediction_set'] ]
    
                
                
                full_mapped_predictions = [None] * len(data_OOD_mask)
                prediction_scores = np.full((len(data_OOD_mask), len(self.unique_labels)), np.nan)  


                non_ood_counter = 0
                for i, is_ood in enumerate(data_OOD_mask):

                    if is_ood:
                        full_mapped_predictions[i] = ["OOD"]
                        

                    else:
                        full_mapped_predictions[i] = mapped_predictions_filtered[non_ood_counter]
                        prediction_scores[i, :] = prediction_scores_tensor[non_ood_counter].cpu()
                        non_ood_counter += 1

                self.prediction_sets[key] = full_mapped_predictions
            
            self.prediction_scores = prediction_scores

            print("\nComputing pvalues.")
            pvalues = self.compute_p_values(data_cp)

            pvalues_full = np.full((len(data_OOD_mask), len(self.unique_labels)), np.nan)
            non_ood_indices = ~np.array(data_OOD_mask)

            pvalues_full[non_ood_indices] = pvalues

           
            self.p_values = pvalues_full

        print("\nPerforming conformal prediction: Done\n")
   

        return None
    

    def compute_p_values(self, data_cp) -> None:

        """
        TorchCP dont have a method to compute p-values for the test samples, so we implement it here.
        This method computes p-values for the test samples based on the calibration scores and the non-conformity scores.
        """
        
        key = list(self.conformal_predictors.keys())[0]  # Use the first predictor's settings
        predictor = self.conformal_predictors[key]

        num_classes = len(self.unique_labels)
        cal_scores_list = []
        cal_labels_list = []

        with torch.no_grad():
            for examples in self.cal_loader:
                tmp_x, tmp_label = examples[0].to(self.device), examples[1].to(self.device)
                
                x_batch = predictor._model(tmp_x.to(predictor._device)).float()
                x_batch = predictor._logits_transformation(x_batch).detach() # these are the logits
                scores_batch = predictor.score_function(x_batch, tmp_label).to(predictor._device)

                
                cal_scores_list.append(scores_batch)
                cal_labels_list.append(tmp_label)

        # Concatenate results from all calibration batches
        calibration_scores = torch.cat(cal_scores_list, dim=0) #These are the non-conformity scores for calibration samples
        calibration_labels = torch.cat(cal_labels_list, dim=0).long()

        # now is the turn to compute p-values for the test samples for each possible label
        all_p_values = []
        with torch.no_grad():
            for examples in data_cp:
                tmp_x, _ = examples[0].to(self.device), examples[1].to(self.device)
                batch_size = tmp_x.shape[0]
                logits_batch = predictor._model(tmp_x.to(predictor._device)).float()
                logits_batch = predictor._logits_transformation(logits_batch).detach()

                p_values_batch = torch.zeros(batch_size, len(self.unique_labels), device=self.device)

                for c in range(num_classes):
                    # Create hypothetical labels for the current class 'c'
                    hypothetical_labels = torch.full((batch_size,), fill_value=c, dtype=torch.long, device=self.device)
                    
                    # Calculate test scores assuming the samples belong to class 'c'
                    test_scores_for_c = predictor.score_function(logits_batch, hypothetical_labels)
                    
                    # If classwise Get the true calibration scores that actually belong to class 'c'
                    if self.predictors == 'classwise':
                        reference_scores = calibration_scores[calibration_labels == c]
                    else: 
                        reference_scores = calibration_scores # Here for standard and cluster taxonomies

                    # Compare test scores with calibration scores to get p-values
                    if len(reference_scores) == 0:
                        p_values_batch[:, c] = 0.0
                        continue

                    comparison = reference_scores.unsqueeze(0) >= test_scores_for_c.unsqueeze(1)
                    p_values_batch[:, c] = (torch.sum(comparison, dim=1) + 1) / (len(reference_scores) + 1)

                all_p_values.append(p_values_batch)



        if all_p_values:
            p_values_tensor = torch.cat(all_p_values, dim=0)
            p_values = p_values_tensor.cpu().numpy()
        else:
            p_values = np.array([])
        
        return p_values   
        
    

