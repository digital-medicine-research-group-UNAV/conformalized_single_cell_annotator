




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


from utils_calib import betainv_mc, betainv_simes, find_slope_EB, estimate_fs_correction, betainv_asymptotic

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split, DataLoader, TensorDataset


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
    def __init__(self, oc_model, delta=0.05):

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

        self.fs_correction = estimate_fs_correction(self.delta, self.n_cal)

        self.is_fitted_ = True

    
    def predict_cond_pvalues(self, X_test, method="MC", simes_kden=2, two_sided=False):

        if not self.is_fitted_:
            raise ValueError("Not fitted yet.  Call 'fit' with appropriate data before using 'predict'.")

        self.scores_test = self.oc_model.score_samples(X_test).numpy().astype(np.float32)
        
        scores_mat = np.tile(self.scores_cal, (self.scores_test.shape[0],1))
        tmp = np.sum(scores_mat >= self.scores_test.reshape(self.scores_test.shape[0],1), 1)
        self.marginal_pvalues = (1.0+tmp)/(1.0+self.n_cal)


        if method=="Simes":
            k = int(self.n_cal/simes_kden)
            self.conditional_pvalues = betainv_simes(self.marginal_pvalues, self.n_cal, k, self.delta)
            two_sided = False


        elif method=="DKWM":
            epsilon = np.sqrt(np.log(2.0/self.delta)/(2.0*self.n_cal))
            if two_sided==True:
                self.conditional_pvalues = np.minimum(1.0, 2.0 * np.minimum(self.marginal_pvalues + epsilon, 1-self.marginal_pvalues + epsilon))
            else:
                self.conditional_pvalues = np.minimum(1.0, self.marginal_pvalues + epsilon)


        elif method=="Linear":
            a = 10.0/self.n_cal #0.005
            b = find_slope_EB(self.n_cal, alpha=a, prob=1.0-self.delta)
            output_1 = np.minimum( (self.marginal_pvalues+a)/(1.0-b), (self.marginal_pvalues+a+b)/(1.0+b) )
            output_2 = np.maximum( (1-self.marginal_pvalues+a+b)/(1.0+b), (1-self.marginal_pvalues+a)/(1.0-b) )
            if two_sided == True:
                self.conditional_pvalues = np.minimum(1.0, 2.0 * np.minimum(output_1, output_2))
            else:
                self.conditional_pvalues = np.minimum(1.0, output_1)


        elif method=="MC":
            if self.fs_correction is None:
                self.fs_correction = estimate_fs_correction(self.delta,self.n_cal)
            self.conditional_pvalues = betainv_mc(self.marginal_pvalues, self.n_cal, self.delta, fs_correction=self.fs_correction)
            two_sided = False


        elif method=="Asymptotic":
            k = int(self.n_cal/simes_kden)
            self.conditional_pvalues = betainv_asymptotic(self.marginal_pvalues, self.n_cal, k, self.delta)
            two_sided = False


        else:
            raise ValueError('Invalid calibration method. Choose "method" in ["Simes", "DKWM", "Linear", "MC", "Asymptotic"]')
        
        
        return None
    

    def evaluate(self, alpha=0.1, lambda_par=0.5, use_sbh=True):
    
        if use_sbh:
            pi = (1.0 + np.sum(self.conditional_pvalues>lambda_par)) / (len(self.conditional_pvalues)*(1.0 - lambda_par))
        else:
            pi = 1.0

        alpha_eff = alpha/pi
        reject, pvals_adj, _, _ = multipletests(self.conditional_pvalues, alpha=alpha_eff, method='fdr_bh')

        return reject, pvals_adj
    

    # Function to process query_data in batches
    def evaluate_in_batches(self, alpha=0.1, lambda_par=0.5, use_sbh=True):

        batch_size = np.sqrt(len(self.scores_cal)).astype(int)

        all_rejects = []
        all_pvals_adj = []

        # Split query_data into batches of size 'batch_size'
        num_batches = int(np.ceil(len(self.conditional_pvalues) / batch_size))
            
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(self.conditional_pvalues))
                
            query_batch = self.conditional_pvalues[start_idx:end_idx]

            if use_sbh:
                pi = (1.0 + np.sum(query_batch>lambda_par)) / (len(query_batch)*(1.0 - lambda_par))
            else:
                pi = 1.0

            alpha_eff = alpha/pi
            #print("alpha_eff: ", alpha_eff, pi)
            reject_batch, pvals_adj_batch, _, _ = multipletests(query_batch, alpha=alpha_eff, method='fdr_bh')
            
            all_rejects.append(reject_batch)
            all_pvals_adj.append(pvals_adj_batch)

        # Concatenate results from all batches
        reject = np.concatenate(all_rejects, axis=0)
        pvals_adj = np.concatenate(all_pvals_adj, axis=0)

        return reject, pvals_adj