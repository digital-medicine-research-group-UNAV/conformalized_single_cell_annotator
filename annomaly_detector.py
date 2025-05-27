




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



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset, random_split



class AEOutlierDetector(nn.Module):
    """
    sparse + denoising autoencoder with
    - LayerNorm + weight normalization
    - SiLU activations
    - Contractive penalties
    - AdamW optimizer + OneCycleLR
    """
    def __init__(self, input_dim, network_architecture, device=None):
        super().__init__()

        # Input dimension
        self._init_args = dict(
            input_dim=input_dim,
            network_architecture=network_architecture,
            device=device
        )

        # Architecture parameters
        hidden_sizes      = network_architecture.get("hidden_sizes", [128, 64, 32, 16])
        dropout_rates     = network_architecture.get("dropout_rates", [0.4, 0.3, 0.4, 0.25])
        self.lr           = network_architecture.get("learning_rate", 1e-4)
        self.weight_decay = network_architecture.get("weight_decay", 1e-5)
        self.batch_size   = network_architecture.get("batch_size", 320)
        self.n_epochs     = network_architecture.get("n_epochs", 50)
        self.patience     = network_architecture.get("patience", 5)
        self.noise_level  = network_architecture.get("noise_level", 0.05)
        self.lambda_sparse    = network_architecture.get("lambda_sparse", 1e-3)
        self.lambda_contractive= network_architecture.get("lambda_contractive", 1e-4)
        

        # Device
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)

        # Encoder
        enc_layers = []
        in_dim = input_dim
        for h, d in zip(hidden_sizes, dropout_rates):
            linear = nn.utils.weight_norm(nn.Linear(in_dim, h))
            enc_layers += [linear, nn.LayerNorm(h), nn.SiLU(), nn.Dropout(d)]
            in_dim = h
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirror)
        dec_layers = []
        for prev, d in zip(reversed(hidden_sizes[:-1]), reversed(dropout_rates[:-1])):
            linear = nn.utils.weight_norm(nn.Linear(in_dim, prev))
            dec_layers += [linear, nn.LayerNorm(prev), nn.SiLU(), nn.Dropout(d)]
            in_dim = prev
        dec_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # Fitting flag
        self._is_fitted = False
        self.to(self.device)

    def forward(self, x, noise=False):
        if noise and self.noise_level > 0:
            x = x + torch.randn_like(x) * self.noise_level
        x_norm = self.input_norm(x)
        z = self.encoder(x_norm)
        x_hat = self.decoder(z)
        return z, x_hat
  
    def _contractive_penalty(self, x, z):
        eps = torch.randn_like(z)
        
        grad = torch.autograd.grad((z * eps).sum(), x, create_graph=True)[0]
        return grad.pow(2).sum(dim=1).mean()

    def fit(self, X, validation_split=0.2):
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        ds = TensorDataset(X)
        n_val = int(len(ds) * validation_split)
        train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        # Loss + optimizer + scheduler
        recon_criterion = nn.SmoothL1Loss()
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.lr,
            steps_per_epoch=len(train_loader),
            epochs=self.n_epochs
        )

        best_val = float('inf')
        epochs_no_improve = 0
        self.train()

        for epoch in range(self.n_epochs):
            train_loss = 0.0
            for (batch,) in train_loader:

                batch = batch.to(self.device)
                batch = batch.clone().detach().requires_grad_(True)

                optimizer.zero_grad()
                z, recon = self.forward(batch, noise=True)

                # Reconstruction + sparse + contractive + orthogonal
                loss_recon = recon_criterion(recon, batch)
                loss_sparse = self.lambda_sparse * z.abs().mean()
                loss_contract = self.lambda_contractive * self._contractive_penalty(batch, z)
                
                loss = loss_recon + loss_sparse + loss_contract 
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
            avg_train = train_loss / len(train_loader)

            # Validation
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(self.device)
                    _, recon = self.forward(batch, noise=False)
                    val_loss += recon_criterion(recon, batch).item()
            avg_val = val_loss / len(val_loader)

            print(f"Epoch {epoch + 1}, Training Loss: {avg_train:.4f}, Validation Loss: {avg_val:.4f}")

            # Early stopping
            if avg_val < best_val:
                best_val = avg_val
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            self.train()

        self._is_fitted = True
        print(f"Finished training: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}")

    def score_samples(self, X):
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before scoring.")
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device)
        self.eval()
        with torch.no_grad():
            _, recon = self.forward(X, noise=False)
            mse = F.mse_loss(recon, X, reduction='none')
            scores = mse.mean(dim=1)
        return scores.cpu()

    def predict(self, X, threshold):
        scores = self.score_samples(X)
        return scores > threshold

    def is_fitted(self):
        return self._is_fitted




class Annomaly_detector():
    def __init__(self,pvalues, oc_model, delta=0.05):

        ModelClass   = oc_model.__class__
        init_kwargs  = getattr(oc_model, "_init_args", {})
        self.oc_model = ModelClass(**init_kwargs)

        self.oc_model.load_state_dict(oc_model.state_dict())
        
        self.pvalues = pvalues
        
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

        

        if self.pvalues == "conditional":
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

 