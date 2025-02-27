




import numpy as np
import pandas as pd
import seaborn as sns
import random
import copy
import sys
from tqdm.autonotebook import tqdm

import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.stats.multitest import multipletests
from scipy import stats

from utils_calib import betainv_mc, betainv_simes, find_slope_EB, estimate_fs_correction, betainv_asymptotic

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau






class AEOutlierDetector(nn.Module):
    
    """
    An autoencoder-based outlier (or novelty) detection model.
    Provides methods to train, calibrate a threshold, 
    and score new samples based on reconstruction error.
    """

    def __init__(self, input_dim, network_architecture):
        super(AEOutlierDetector, self).__init__()

        # Extract parameters from the architecture dictionary
        hidden_sizes = network_architecture.get("hidden_sizes", [128, 64, 32, 16])
        dropout_rates = network_architecture.get("dropout_rates", [0.4, 0.3, 0.4, 0.25])
        self.lr = network_architecture.get("learning_rate", 0.0001)
        self.batch_size = network_architecture.get("batch_size", 320)
        self.n_epochs = network_architecture.get("n_epochs", 7)
        self.early_stopping_patience = 7 #5
        self._is_fitted = False

        # Build encoder dynamically based on hidden sizes and dropout rates
        encoder_layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                encoder_layers.append(nn.Linear(input_dim, hidden_sizes[i]))
            else:
                encoder_layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rates[i]))

        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder (reversing encoder structure)
        decoder_layers = []
        rev_hidden_sizes = list(reversed(hidden_sizes))
        for i in range(len(hidden_sizes) - 1, 0, -1):
            decoder_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i - 1]))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout_rates[i - 1]))

        decoder_layers.append(nn.Linear(hidden_sizes[0], input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

        # Internal attributes
        self.threshold_ = None
        self._is_fitted = False

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    


    def fit(self, X, validation_split=0.2):
    
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
    
        # Determine the sizes for training and validation
        total_size = X.size(0)
        val_size = int(total_size * validation_split)
        train_size = total_size - val_size
        
        # Split the dataset
        train_tensor, val_tensor = random_split(X, [train_size, val_size], generator=torch.Generator())
        
        # Create DataLoaders for training and validation
        train_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_tensor, batch_size=self.batch_size, shuffle=False)
        
        
        # Define Loss and Optimizer
        criterion = nn.MSELoss()
        optimizer = Adam(self.parameters(), lr=self.lr)

        #  adaptive learning rate scheduler (monitor validation loss)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

        # Variables for early stopping
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        self.train()  # set the module in training mode
        for epoch in range(self.n_epochs):
            total_train_loss = 0.0
            
            # Training phase
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(batch[0])
                loss = criterion(outputs, batch[0])
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)

            # Validation phase
            self.eval()  # set the module in evaluation mode
            with torch.no_grad():
                total_val_loss = 0.0
                for val_batch in val_loader:
                    val_outputs = self.forward(val_batch[0])
                    val_loss = criterion(val_outputs, val_batch[0])
                    total_val_loss += val_loss.item()
                
                avg_val_loss = total_val_loss / len(val_loader)

            # Step the scheduler based on the validation loss
            scheduler.step(avg_val_loss)

            #print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= self.early_stopping_patience:
                #print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
                print("Early stopping triggered!")
                self._is_fitted = True
                break

        print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")


        self._is_fitted = True
    
    


    def score_samples(self, X):

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(X)
            mse_per_sample = F.mse_loss(reconstructed, X, reduction='none')
            scores =  mse_per_sample.mean(dim=1)
        return scores
    
    def is_fitted(self):

        return self._is_fitted
    




class Annomaly_detector():
    def __init__(self,pvalues, oc_model, delta=0.05):
        
        self.pvalues = pvalues
        self.oc_model = copy.deepcopy(oc_model) ## One-class underlying model
        self.delta = delta
        self.marginal_pvalues = None
        self.conditional_pvalues = None
        self.is_fitted_ = False


    @property
    def fitted(self):
        """check whether the model is fitted."""
        return self.is_fitted_
    

    def fit(self, X_train, X_calib):

        # Fit the black-box one-class classification model
        self.oc_model.fit(X_train, validation_split=0.3)

        print("Fitted the one-class model")
        # Calibrate using conditional conformal p-values
        self.scores_cal = self.oc_model.score_samples(X_calib).numpy().astype(np.float32)

        self.n_cal = self.scores_cal.shape[0]
        print("Calibrated!", self.n_cal)


        self.is_fitted_ = True


    
    def predict_pvalues(self, X_test):

        if not self.is_fitted_:
            raise ValueError("Not fitted yet.  Call 'fit' with appropriate data before using 'predict'.")

        self.scores_test = self.oc_model.score_samples(X_test).numpy().astype(np.float32)

        
        scores_mat = np.tile(self.scores_cal, (self.scores_test.shape[0],1))
        tmp = np.sum(scores_mat >= self.scores_test.reshape(self.scores_test.shape[0],1), 1)
        self.marginal_pvalues = (1.0+tmp)/(1.0+self.n_cal)

        

        if self.pvalues == "confitional":
            if self.fs_correction is None:
                self.fs_correction = estimate_fs_correction(self.delta,self.n_cal)
            self.conditional_pvalues = betainv_mc(self.marginal_pvalues, self.n_cal, self.delta, fs_correction=self.fs_correction)

        
        return None
    


    def evaluate_conditional_pvalues(self, alpha=0.1, lambda_par=0.5, use_sbh=True):    

        if use_sbh:
            pi = (1.0 + np.sum(self.conditional_pvalues>lambda_par)) / (len(self.conditional_pvalues)*(1.0 - lambda_par))
        else:
            pi = 1.0

        alpha_eff = alpha/pi    

        rejections = self.conditional_pvalues <= alpha_eff

    
        return rejections
    
 

    def evaluate_marginal_pvalues(self, alpha=0.1, lambda_par=0.5, use_sbh=True):
        

        if use_sbh:
            pi = (1.0 + np.sum(self.marginal_pvalues>lambda_par)) / (len(self.marginal_pvalues)*(1.0 - lambda_par))
        else:
            pi = 1.0

        alpha_eff = alpha/pi    

        rejections = self.marginal_pvalues <= alpha_eff

    
        return rejections

 