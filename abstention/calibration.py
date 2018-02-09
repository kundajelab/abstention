from __future__ import division, print_function, absolute_import
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression as IR
from sklearn.linear_model import LogisticRegression as LR


class CalibratorFactory(object):

    def __call__(self, valid_preacts, valid_labels):
        raise NotImplementedError()


class Expit(CalibratorFactory):

    def __call__(self, valid_preacts=None, valid_labels=None):
        def func(preact):
            return expit(preact)
        return func


class PlattScaling(CalibratorFactory):

    def __init__(self, verbose=True):
        self.verbose=verbose

    def __call__(self, valid_preacts, valid_labels):

        lr = LR()                                                       
        #LR needs X to be 2-dimensional
        lr.fit(valid_preacts.reshape(-1,1), valid_labels) 
   
        if (self.verbose): 
            print("Platt scaling coef:", lr.coef_[0][0],
                  "; intercept:",lr.intercept_[0])
    
        def calibration_func(preact):
            return lr.predict_proba(preact.reshape(-1,1))[:,1]
    
        return calibration_func


class IsotonicRegression(CalibratorFactory):

    def __init__(self, verbose=True):
        self.verbose = verbose 

    def __call__(self, valid_preacts, valid_labels):
        ir = IR()
        valid_preacts = valid_preacts.flatten()
        min_valid_preact = np.min(valid_preacts)
        max_valid_preact = np.max(valid_preacts)
        assert len(valid_preacts)==len(valid_labels)
        #sorting to be safe...I think weird results can happen when unsorted
        sorted_valid_preacts, sorted_valid_labels = zip(
            *sorted(zip(valid_preacts, valid_labels), key=lambda x: x[0]))
        y = ir.fit_transform(sorted_valid_preacts, sorted_valid_labels)
    
        def calibration_func(preact):
            preact = np.minimum(preact, max_valid_preact)
            preact = np.maximum(preact, min_valid_preact)
            return ir.transform(preact.flatten())

        return calibration_func
