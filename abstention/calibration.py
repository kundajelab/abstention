from __future__ import division, print_function, absolute_import
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss
import scipy
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


class ImbalanceAdaptationWrapper(CalibratorFactory):

    def __init__(self, base_calibrator_factory, verbose=True):
        self.base_calibrator_factory = base_calibrator_factory
        self.verbose = verbose

    def __call__(self, valid_preacts, valid_labels):
        base_calibration_func = self.base_calibrator_factory(
            valid_preacts=valid_preacts, valid_labels=valid_labels)
        calib_valid_probs = base_calibration_func(valid_preacts) 
        # bandwidth is scotts factor
        valid_kde = KernelDensity(
            kernel='gaussian', bandwidth=len(valid_labels)**(-1./(1+4))).fit(
            list(zip(calib_valid_probs, np.zeros((len(calib_valid_probs))))))

        def calibration_func(preact):
            calib_probs = base_calibration_func(preact)
            valid_densities_at_test_pts = np.exp(valid_kde.score_samples(
                zip(calib_probs, np.zeros((len(calib_probs))))))
            # bandwidth is scotts factor
            kde_test = KernelDensity(kernel='gaussian',
                bandwidth=len(calib_probs)**(-1./(1+4))).fit(
                list(zip(calib_probs, np.zeros((len(calib_probs))))))
            test_densities_at_test_pts = np.exp(kde_test.score_samples(
                zip(calib_probs, np.zeros((len(calib_probs))))))
            neg_densities_at_test_pts =\
                valid_densities_at_test_pts*(1-calib_probs)*\
                len(valid_labels)/(len(valid_labels) - np.sum(valid_labels))
            pos_densities_at_test_pts =\
                valid_densities_at_test_pts*calib_probs*(len(valid_labels)\
                /np.sum(valid_labels))

            def eval_func(x):
                x = x[0]
                differences =\
                    test_densities_at_test_pts-(x*pos_densities_at_test_pts\
                    +(1-x)*neg_densities_at_test_pts)
                loss = np.sum(np.square(differences))
                grad = np.sum(2*(differences)*(neg_densities_at_test_pts\
                    -pos_densities_at_test_pts))
                return loss, np.array([grad])

            alpha = scipy.optimize.minimize(fun=eval_func,
                x0=np.array([0.5]),
                bounds=[(0,1)],
                jac=True,
                method='L-BFGS-B',
                tol=1e-07,
                )['x'][0]

            new_calib_test = pos_densities_at_test_pts*alpha / (
                pos_densities_at_test_pts*alpha + neg_densities_at_test_pts*(1-alpha)
                )

            return new_calib_test

        return calibration_func
