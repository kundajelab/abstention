from __future__ import division, print_function, absolute_import
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss
import scipy
from scipy.special import expit
import scipy.misc
import scipy.optimize
from sklearn.isotonic import IsotonicRegression as IR
from sklearn.linear_model import LogisticRegression as LR


def softmax(preact, temp):
    exponents = np.exp(preact/temp)
    sum_exponents = np.sum(exponents, axis=1) 
    return exponents/sum_exponents[:,None]


#based on https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py#L78
def compute_ece(softmax_out, labels, bins):

    bin_boundaries = np.linspace(0,1,num=bins)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(softmax_out,axis=1)
    is_correct = np.argmax(softmax_out,axis=1)==np.argmax(labels,axis=1)

    ece = 0.0 
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower)*(confidences <= bin_upper)
        prop_in_bin = np.mean((in_bin)) 
        if (prop_in_bin > 0.0):
            accuracy_in_bin = np.mean(is_correct[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin-accuracy_in_bin)*prop_in_bin
    return ece*100


class CalibratorFactory(object):

    def __call__(self, valid_preacts, valid_labels):
        raise NotImplementedError()


class Softmax(CalibratorFactory):

    def __call__(self, valid_preacts=None, valid_labels=None):
        return (lambda x: softmax(preact=x, temp=1.0))


class TempScaling(CalibratorFactory):

    def __init__(self, ece_bins=15, lbfgs_kwargs={}, verbose=True):
        self.lbfgs_kwargs = lbfgs_kwargs
        self.verbose = verbose
        self.ece_bins = ece_bins

    def __call__(self, valid_preacts, valid_labels):

        assert np.max(np.sum(valid_labels,axis=1)==1.0)

        #calculate the temperature scaling parameter
        def eval_func(x):
            x = x[0]
            temp_scaled_valid_preacts = valid_preacts/float(x)
            log_sum_exp = scipy.misc.logsumexp(a = temp_scaled_valid_preacts) 

            exp_result = np.exp(temp_scaled_valid_preacts)
            sum_exp = np.sum(exp_result, axis=1)
            sum_preact_times_exp = np.sum(valid_preacts*
                                          exp_result, axis=1)

            logits_that_matter =\
                np.sum(valid_preacts*valid_labels, axis=1)
            
            log_likelihoods = logits_that_matter/float(x) - log_sum_exp
            nll = -np.mean(log_likelihoods)
            grads = ((sum_preact_times_exp/sum_exp - logits_that_matter)/\
                     (float(x)**2))
            mean_grad = -np.mean(grads)
            return nll, np.array([mean_grad])

        def eval_fprime(x):
            x = x[0]
            temp_scaled_valid_preacts = valid_preacts/float(x)
            log_sum_exp = scipy.misc.logsumexp(a = temp_scaled_valid_preacts) 

            exp_result = np.exp(temp_scaled_valid_preacts)
            sum_exp = np.sum(exp_result, axis=1)
            sum_preact_times_exp = np.sum(temp_scaled_valid_preacts*
                                          exp_result, axis=1)

            logits_that_matter =\
                np.sum(valid_preacts*valid_labels, axis=1)
            
            log_likelihoods = logits_that_matter/float(x) - log_sum_exp
            nll = np.mean(log_likelihoods)
            grads = ((sum_exp/sum_preact_times_exp - logits_that_matter)/\
                     (float(x)**2))
            mean_grad = -np.mean(grads)

            return np.array([mean_grad])

        if (self.verbose):
            original_nll = eval_func(np.array([1.0])) 
            original_ece = compute_ece(
                softmax_out=softmax(preact=valid_preacts, temp=1.0),
                labels=valid_labels, bins=self.ece_bins) 
            print("Original NLL & grad is: ",original_nll)
            print("Original ECE is: ",original_ece)
            
        optimization_result = scipy.optimize.minimize(fun=eval_func,
                                  x0=np.array([1.0]),
                                  bounds=[(0,None)],
                                  jac=True,
                                  method='L-BFGS-B',
                                  tol=1e-07,
                                  **self.lbfgs_kwargs)
        if (self.verbose):
            print(optimization_result)
        optimal_t = optimization_result.x

        if (self.verbose):
            final_nll = eval_func(np.array([optimal_t])) 
            final_ece = compute_ece(
                softmax_out=softmax(preact=valid_preacts, temp=optimal_t),
                labels=valid_labels, bins=self.ece_bins) 
            print("Final NLL & grad is: ",final_nll)
            print("Final ECE is: ",final_ece)

        return (lambda x: softmax(preact=x, temp=optimal_t))


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
                pos_densities_at_test_pts*alpha
                + neg_densities_at_test_pts*(1-alpha))

            return new_calib_test

        return calibration_func


class EMImbalanceAdapter(object):

    def __init__(self, verbose=True,
                       tolerance=1E-3,
                       max_iterations=100):
        self.verbose = verbose
        self.tolerance = tolerance

    def __call__(self, valid_posterior_probs, valid_labels):

        assert valid_posterior_probs.shape==valid_labels.shape
        assert len(valid_posterior_probs.shape)<=2
        assert len(valid_labels.shape)<=2

        #if binary labels were provided, convert to softmax format
        # for consistency
        if (len(valid_labels.shape)==1 or valid_labels.shape[1]==1):
            valid_labels = np.squeeze(valid_labels)
            valid_posterior_probs = np.squeeze(valid_posterior_probs)
            softmax_valid_labels = np.zeros((len(valid_labels),2))
            softmax_valid_labels[:,1] = valid_labels
            softmax_valid_labels[:,0] = 1-valid_labels
            softmax_valid_probs = np.zeros((len(valid_labels),2))
            softmax_valid_probs[:,1] = valid_posterior_probs 
            softmax_valid_probs[:,0] = 1-valid_posterior_probs 
        else:
            softmax_valid_labels = valid_labels
            softmax_valid_probs = valid_posterior_probs

        valid_class_imbalance = np.mean(valid_labels, axis=0)
        if (self.verbose):
            print("Original class imbalance", valid_class_imbalance)
        
        def imbalance_adapter(initial_posterior_probs):
            if (len(initial_posterior_probs.shape)==1
                or initial_posterior_probs.shape[1]==1):
                softmax_posterior_probs = np.zeros(
                    (len(initial_posterior_probs),2)) 
                softmax_posterior_probs[:,0] = initial_posterior_probs
                softmax_posterior_probs[:,1] = 1-initial_posterior_probs

            current_iter_class_imbalance = valid_class_imbalance
            current_iter_posterior_probs = initial_posterior_probs
            next_iter_class_imbalance = None
            next_iter_posterior_probs = None
            iter_number = 0
            if (next_iter_class_imbalance is None
                or (np.sum(np.abs(next_iter_class_imbalance
                                  -current_iter_class_imbalance)
                           <= self.tolerance))):
                if (next_iter_class_imbalance is not None):
                    current_iter_class_imbalance=next_iter_class_imbalance 
                    current_iter_posterior_probs=next_iter_posterior_probs
                next_iter_class_imbalance = np.mean(
                    current_iter_posterior_probs, axis=0) 
                next_iter_posterior_probs_unnorm =(
                    (initial_posterior_probs
                     *next_iter_class_imbalance[None,:])/
                    valid_class_imbalance[None,:])
                next_iter_posterior_probs = (
                    next_iter_posterior_probs_unnorm/
                    np.sum(next_iter_posterior_probs_unnorm,axis=-1)[:,None])
                iter_number += 1
            if (self.verbose):
                print("Finished on iteration",iter_number,"with delta",
                      np.sum(np.abs(current_iter_class_imbalance-
                                    valid_class_imbalance)))
                print("Final imbalance", current_iter_class_imbalance)
            return current_iter_posterior_probs 
        return imbalance_adapter
