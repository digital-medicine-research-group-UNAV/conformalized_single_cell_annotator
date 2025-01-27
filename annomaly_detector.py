




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



class Annomaly_detector():
    def __init__(self, oc_model = IsolationForest(n_jobs=-1), delta=0.05):

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
        self.oc_model.fit(X_train)

        # Calibrate using conditional conformal p-values
        self.scores_cal = self.oc_model.score_samples(X_calib).astype(np.float32)
        self.n_cal = len(self.scores_cal)

        self.fs_correction = estimate_fs_correction(self.delta, self.n_cal)

        self.is_fitted_ = True

    
    def predict_cond_pvalues(self, X_test, method="MC", simes_kden=2, two_sided=False):

        if not self.is_fitted_:
            raise ValueError("Not fitted yet.  Call 'fit' with appropriate data before using 'predict'.")

        scores_test = self.oc_model.score_samples(X_test).astype(np.float32)
        
        scores_mat = np.tile(self.scores_cal, (len(scores_test),1))
        tmp = np.sum(scores_mat <= scores_test.reshape(len(scores_test),1), 1)
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