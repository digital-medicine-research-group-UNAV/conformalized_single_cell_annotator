




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



from utils_calib import betainv_mc, betainv_simes, find_slope_EB, estimate_fs_correction, betainv_asymptotic



class Annomaly_detector():
    def __init__(self, oc_model = IsolationForest(n_jobs=-1), delta=0.05):

        self.oc_model = copy.deepcopy(oc_model) ## One-class underlying model
        self.delta = delta
        self.is_fitted_ = False


    @property
    def fitted(self):
        """check whether the model is fitted."""
        return self.is_fitted_
    

    def fit(self, X_train, X_calib):

        # Fit the black-box one-class classification model
        self.oc_model.fit(X_train)

        # Calibrate using conditional conformal p-values
        self.scores_cal = self.oc_model.score_samples(X_calib)
        self.n_cal = len(self.scores_cal)

        self.fs_correction = estimate_fs_correction(self.delta, self.n_cal)

        self.is_fitted_ = True

    
    def predict(self, X_test, alpha=0.05, method="MC", simes_kden=2, two_sided=False):

        if not self.is_fitted_:
            raise ValueError("Not fitted yet.  Call 'fit' with appropriate data before using 'predict'.")

        scores_test = self.oc_model.score_samples(X_test)
        scores_mat = np.tile(self.scores_cal, (len(scores_test),1))
        tmp = np.sum(scores_mat <= scores_test.reshape(len(scores_test),1), 1)
        pvals = (1.0+tmp)/(1.0+self.n_cal)


        if method=="Simes":
            k = int(self.n_cal/simes_kden)
            pvals = betainv_simes(pvals, self.n_cal, k, self.delta)
            two_sided = False


        elif method=="DKWM":
            epsilon = np.sqrt(np.log(2.0/self.delta)/(2.0*self.n_cal))
            if two_sided==True:
                pvals = np.minimum(1.0, 2.0 * np.minimum(pvals + epsilon, 1-pvals + epsilon))
            else:
                pvals = np.minimum(1.0, pvals + epsilon)


        elif method=="Linear":
            a = 10.0/self.n_cal #0.005
            b = find_slope_EB(self.n_cal, alpha=a, prob=1.0-self.delta)
            output_1 = np.minimum( (pvals+a)/(1.0-b), (pvals+a+b)/(1.0+b) )
            output_2 = np.maximum( (1-pvals+a+b)/(1.0+b), (1-pvals+a)/(1.0-b) )
            if two_sided == True:
                pvals = np.minimum(1.0, 2.0 * np.minimum(output_1, output_2))
            else:
                pvals = np.minimum(1.0, output_1)


        elif method=="MC":
            if self.fs_correction is None:
                self.fs_correction = estimate_fs_correction(self.delta,self.n_cal)
            pvals = betainv_mc(pvals, self.n_cal, self.delta, fs_correction=self.fs_correction)
            two_sided = False


        elif method=="Asymptotic":
            k = int(self.n_cal/simes_kden)
            pvals = betainv_asymptotic(pvals, self.n_cal, k, self.delta)
            two_sided = False


        else:
            raise ValueError('Invalid calibration method. Choose "method" in ["Simes", "DKWM", "Linear", "MC", "Asymptotic"]')
        
        output = (pvals > alpha)


        return output