




import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import  ks_2samp

from utils_calib import betainv_mc, betainv_simes, find_slope_EB, estimate_fs_correction, betainv_asymptotic

from torch.utils.data import random_split
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
        print(f"Using device: {self.device}")

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

    def fit(self, X,X_val = None, validation_split=0.15):
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        ds = TensorDataset(X)
        train_loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        if X_val is None:
            n_val = int(len(ds) * validation_split)
            train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size)
        else:
            if not torch.is_tensor(X_val):
                X_val = torch.tensor(X_val, dtype=torch.float32)
            val_ds = TensorDataset(X_val)
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

    def is_fitted(self):
        return self._is_fitted


 

class Annomaly_detector():
    def __init__(self,pvalues, oc_model, delta=0.05):

        ModelClass   = oc_model.__class__
        init_kwargs  = getattr(oc_model, "_init_args", {})
        self.oc_model = ModelClass(**init_kwargs)

        self.oc_model.load_state_dict(oc_model.state_dict())
        
        self.pvalues = pvalues
        
        self.alpha_OOD = None
        self.delta = delta
        self.marginal_pvalues = None
        self.conditional_pvalues = None
        self.is_fitted_ = False


    @property
    def fitted(self):
        """check whether the model is fitted."""
        return self.is_fitted_
    

    def fit(self, X_train,X_val, X_calib):

        # Fit the black-box one-class classification model
        self.oc_model.fit(X_train,X_val)

        print("Fitted the one-class model")
        # Calibrate using conditional conformal p-values
        self.scores_cal = self.oc_model.score_samples(X_calib).numpy().astype(np.float32)

        jitter_magnitude = 1e-6
        self.scores_cal += np.random.uniform(-jitter_magnitude, jitter_magnitude, self.scores_cal.shape)
        self.n_cal = self.scores_cal.shape[0]
        print("Calibrated!")


        self.is_fitted_ = True


    
    def predict_pvalues(self, X_test, X_ID=None):

        if not self.is_fitted_:
            raise ValueError("Not fitted yet.  Call 'fit' with appropriate data before using 'predict'.")

        self.scores_test = self.oc_model.score_samples(X_test).numpy().astype(np.float32)

        
        scores_mat = np.tile(self.scores_cal, (self.scores_test.shape[0],1))
        tmp = np.sum(scores_mat >= self.scores_test.reshape(self.scores_test.shape[0],1), 1)
        self.marginal_pvalues = (1.0+tmp)/(1.0+self.n_cal)


        #fig, ax = plt.subplots(figsize=(10, 8))
        #ax.hist(self.marginal_pvalues, bins=25, alpha=0.5, color="red",
        #        label="True Outliers", edgecolor='black')


        # Set bold labels and title with increased font sizes
        #ax.set_xlabel("p-values", fontweight='bold', fontsize=28)
        #ax.set_ylabel("Frequency", fontweight='bold', fontsize=28)
        #ax.set_title("Conformal p-values", fontweight='bold', fontsize=30)
        

        # Increase tick label sizes for better readability
        #ax.tick_params(axis='both', which='major', labelsize=24)


        # Add legend, adjust layout, save and display the figure
        #ax.legend(loc="best", fontsize=28, title_fontsize=28 )
        #plt.tight_layout()
        
        #plt.show()
        #plt.clf()

        if self.pvalues == "conditional":
            self.fs_correction = estimate_fs_correction(self.delta,self.n_cal)
            self.conditional_pvalues = betainv_mc(self.marginal_pvalues, self.n_cal, self.delta, fs_correction=self.fs_correction)

        
        if X_ID is not None:
            # Compute conditional p-values for the ID set
            scores_ID = self.oc_model.score_samples(X_ID).numpy().astype(np.float32)
            scores_mat_ID = np.tile(self.scores_cal, (scores_ID.shape[0],1))
            tmp_ID = np.sum(scores_mat_ID >= scores_ID.reshape(scores_ID.shape[0],1), 1)
            self.marginal_pvalues_ID = (1.0+tmp_ID)/(1.0+self.n_cal)

            #fig, ax = plt.subplots(figsize=(10, 8))
            #ax.hist(self.marginal_pvalues_ID, bins=25, alpha=0.5, color="red",
            #        label="True Outliers", edgecolor='black')


            # Set bold labels and title with increased font sizes
            #ax.set_xlabel("p-values", fontweight='bold', fontsize=28)
            #ax.set_ylabel("Frequency", fontweight='bold', fontsize=28)
            #ax.set_title("ID Conformal p-values", fontweight='bold', fontsize=30)
            

            # Increase tick label sizes for better readability
            #ax.tick_params(axis='both', which='major', labelsize=24)


            # Add legend, adjust layout, save and display the figure
            #ax.legend(loc="best", fontsize=28, title_fontsize=28 )
            #plt.tight_layout()
            
            #plt.show()
            #plt.clf()
        
        else:
            self.marginal_pvalues_ID = None

        return None
    


    def evaluate_conditional_pvalues(self, alpha=0.1, lambda_par=0.5, use_sbh=True):    

        if use_sbh:
            pi = (1.0 + np.sum(self.conditional_pvalues>lambda_par)) / (len(self.conditional_pvalues)*(1.0 - lambda_par))
        else:
            pi = 1.0

        alpha_eff = alpha/pi    

        rejections = self.conditional_pvalues <= alpha_eff

    
        return rejections
    
 

    def evaluate_marginal_pvalues(self, alpha=None, lambda_par=0.5, use_sbh=True, is_outlier = None, adata_=None):
        
        self.is_outlier = is_outlier

        if use_sbh:
            pi = (1.0 + np.sum(self.marginal_pvalues>lambda_par)) / (len(self.marginal_pvalues)*(1.0 - lambda_par))
        else:
            pi = 1.0


        if alpha is not None: 

            alpha_eff = alpha/pi  
              

            print(f"Effective alpha: {alpha_eff:.4f} (alpha={alpha:.4f})")

            rejections = self.marginal_pvalues <= alpha_eff
        
        else:
            # hcer if do test is tue
            print("No specific alpha provided, using automatic alpha selection...")

            pvalues = self.marginal_pvalues.copy()
            pvalues_ID = self.marginal_pvalues_ID.copy()

            significance_level = 0.025
            alpha_steps = 50
            num_bins = 25

            alpha_candidates = np.linspace(0, 1, alpha_steps, endpoint=False)
            best_alpha = None
            

            for alpha_ in alpha_candidates:
               
                truncated = pvalues[pvalues >= alpha_]
                id_trunc   = pvalues_ID[pvalues_ID >= alpha_]
                n_remain = len(truncated)
                

                true_scaled = (truncated  - alpha_) / (1 - alpha_)
                id_scaled   = (id_trunc   - alpha_) / (1 - alpha_)

                #bins = np.linspace(0, 1, num_bins + 1)
                #counts, _ = np.histogram(truncated, bins=num_bins, range=(alpha_, 1))
                #id_counts,   _ = np.histogram(id_scaled,   bins=bins)
                #avg_count = counts.mean() 

                #expected = np.full(num_bins, n_remain / num_bins)

                ks_stat, ks_p = ks_2samp(true_scaled, id_scaled)

                
                print(f"α = {alpha_:.2f}    p={ks_p:.3f}")
                if ks_p >= significance_level:
                    best_alpha = alpha_
                    print(f" → Selected α = {alpha_:.2f}\n")
                    break

                if alpha_ >0.5:
                    break
            
            alpha_eff = best_alpha    
            if alpha_eff is None:
                print("WARNING: No suitable alpha found, using default alpha=0.1")
                alpha_eff = 0.1
            else:
                print(f"Authomatic effective alpha: {alpha_eff}")

            rejections = self.marginal_pvalues <= alpha_eff

        self.alpha_OOD = alpha_eff
        
        if adata_ is not None:
            fig, ax = plt.subplots(figsize=(10, 8))
            # Plot histograms with black edges for clarity
            ax.hist(self.marginal_pvalues[is_outlier == 0], bins=25, alpha=0.5, color="blue",
                    label="True inliers", edgecolor='black')
        
            ax.hist(self.marginal_pvalues[is_outlier == 1], bins=25, alpha=0.5, color="red",
                    label="True Outliers", edgecolor='black')
            # Set bold labels and title with increased font sizes
            ax.set_xlabel("p-values", fontweight='bold', fontsize=28)
            ax.set_ylabel("Frequency", fontweight='bold', fontsize=28)
            ax.set_title("Conformal p-values", fontweight='bold', fontsize=30)
            ax.axvline(alpha_eff, color='black', linestyle='dashed', linewidth=2, label=f'Corrected threshold: {alpha_eff:.3f}')
            # Increase tick label sizes for better readability
            ax.tick_params(axis='both', which='major', labelsize=24)


            # Add legend, adjust layout, save and display the figure
            #ax.legend(loc="best", fontsize=28, title_fontsize=28 )
            plt.tight_layout()
            plt.savefig("conformal_pvalues_by_cell_type_heal.svg", dpi=600, bbox_inches='tight')
            plt.show()
            plt.clf()

            cell_types = adata_.obs["cell_type"].astype(str)
            unique_types = cell_types.unique()
            n_types = len(unique_types)
            import math
            n_cols = 3
            n_rows = math.ceil(n_types / n_cols)

            # 3) Create subplots
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharex=True, sharey=True)
            axes = axes.flatten()  # so we can index 0..n_cols*n_rows-1

            # 4) Compute shared bin edges (same for every subplot, for consistent comparison)
            all_pvals = self.marginal_pvalues
            num_bins = 25
            bin_edges = np.linspace(all_pvals.min(), all_pvals.max(), num_bins + 1)

            for idx, ct in enumerate(unique_types):
                ax = axes[idx]
                mask = (cell_types == ct)
                pvals_ct = self.marginal_pvalues[mask]

                # Plot histogram for this cell type
                ax.hist(
                    pvals_ct,
                    bins=bin_edges,
                    color="C{}".format(idx % 10),  # use Matplotlib's default color cycle
                    edgecolor="black",
                    alpha=0.7
                )
                # Vertical line at the corrected threshold
                ax.axvline(alpha_eff, color="black", linestyle="dashed", linewidth=1.5)

                # Title = cell type, with bold font
                ax.set_title(str(ct), fontweight="bold", fontsize=16)
                # Only label tick labels sparsely to avoid overcrowding:
                ax.tick_params(axis="both", which="major", labelsize=12)

            # 5) Hide any unused subplots (if the grid has more slots than types)
            for j in range(n_types, n_rows * n_cols):
                axes[j].axis("off")

            # 6) Add common X/Y labels in the figure
            fig.text(0.5, 0.04, "p-values", ha="center", fontweight="bold", fontsize=18)
            fig.text(0.04, 0.5, "Frequency", va="center", rotation="vertical", fontweight="bold", fontsize=18)
            fig.suptitle("Conformal p-values by Cell Type", fontweight="bold", fontsize=20, y=0.95)

            plt.tight_layout(rect=[0, 0.05, 1, 0.93])  # leave room for the suptitle and shared labels
            
            plt.show()
            plt.clf()


        
        if self.is_outlier is not None:
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
            accuracy = accuracy_score(is_outlier, rejections)
            precision = precision_score(is_outlier, rejections)

            tp = np.sum(np.logical_and(rejections, is_outlier))

            # Count false positives: inliers incorrectly flagged as outliers.
            fp = np.sum(np.logical_and(rejections, np.logical_not(is_outlier)))
            print(f'False Positives: {fp}')
            print(f'True Positives: {tp}')

            # Compute FDR: Handle division by zero if there are no rejections.
            fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
            print(f'FDR: {fdr:.4f}')  

            recall = recall_score(is_outlier, rejections)
            # only compute AUROC if we actually have both classes
            if len(np.unique(is_outlier)) < 2:
                auroc = np.nan
                print("Warning: only one class present in y_true; setting AUROC to NaN")
            else:
                auroc = roc_auc_score(is_outlier, rejections)

    
            #return rejections, accuracy, precision, recall, fdr, auroc, pi, alpha_eff
            return rejections, accuracy, precision, recall, auroc
        
        return rejections
 