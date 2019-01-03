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

        return (lambda preact: softmax(preact=preact, temp=optimal_t))


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


class BiasCorrectionWrapper(CalibratorFactory):

    def __init__(self, base_calibrator_factory,
                       bias_corrector_factory):
        self.base_calibrator_factory = base_calibrator_factory
        self.bias_corrector_factory = bias_corrector_factory

    def __call__(self, valid_preacts, valid_labels):
        base_calibration_func = self.base_calibrator_factory(
            valid_preacts=valid_preacts, valid_labels=valid_labels)
        biased_valid_calib_probs = base_calibration_func(valid_preacts) 
        bias_corrector = self.bias_corrector_factory(
            valid_posterior_probs=biased_valid_calib_probs,
            valid_labels=valid_labels)
        def calibration_func(preact):
            return bias_corrector(
                    unadapted_posterior_probs=
                     base_calibration_func(preact=preact))
        return calibration_func
        

class EMBiasCorrectorFactory(object):

    def __init__(self, tolerance=1E-3, max_iter=100, verbose=False):
        self.tolerance = tolerance
        self.verbose = verbose
        self.max_iter = max_iter

    def __call__(self, valid_posterior_probs, valid_labels):
        
        #idea: figure out what prior distribution valid_posterior_probs
        # originated from to explain the difference between
        # np.mean(valid_labels, axis=0)
        # and np.mean(valid_posterior_probs, axis=0)

        observed_class_freq = np.mean(valid_labels,axis=0)
        current_orig_freq = np.array(observed_class_freq)
        if (self.verbose):
            print("Observed class freq:",observed_class_freq)
        terminate = False

        iterations = 0
        while (terminate==False):
            current_ex_weights =\
                np.sum((current_orig_freq/observed_class_freq)[None,:]
                       *valid_labels,axis=1)
            next_orig_freq =\
                np.mean(current_ex_weights[:,None]*valid_posterior_probs,
                        axis=0)
            iterations += 1
            if ((np.sum(np.abs(next_orig_freq-current_orig_freq)) <
                 self.tolerance) or iterations > self.max_iter):
                terminate = True
            current_orig_freq = next_orig_freq
        if (self.verbose):
            print("Iterations:",iterations)
            print("Est. original class freq:",current_orig_freq)
        return PriorShiftAdapterFunc(
                original_class_freq=current_orig_freq,
                adapted_class_freq=observed_class_freq)


class ImbalanceAdaptationWrapper(CalibratorFactory):

    def __init__(self, base_calibrator_factory,
                       imbalance_adapter,
                       verbose=True):
        self.base_calibrator_factory = base_calibrator_factory
        self.imbalance_adapter = imbalance_adapter
        self.verbose = verbose

    def __call__(self, valid_preacts, valid_labels):
        base_calibration_func = self.base_calibrator_factory(
            valid_preacts=valid_preacts, valid_labels=valid_labels)
        valid_calib_probs = base_calibration_func(valid_preacts) 

        def calibration_func(preact):
            calib_probs = base_calibration_func(preact)
            imbalance_adapter_func = self.imbalance_adapter(
                valid_labels=valid_labels,
                tofit_initial_posterior_probs=calib_probs,
                valid_posterior_probs=valid_calib_probs)
            return imbalance_adapter_func(
                    unadapted_posterior_probs=calib_probs)

        return calibration_func


class AbstractImbalanceAdapterFunc(object):

    def __call__(self, unadapted_posterior_probs):
        raise NotImplementedError()


class PriorShiftAdapterFunc(AbstractImbalanceAdapterFunc):

    def __init__(self, original_class_freq, adapted_class_freq):
        original_class_freq = np.array(original_class_freq)
        adapted_class_freq = np.array(adapted_class_freq) 
        assert len(original_class_freq.shape)==1
        self.original_class_freq = original_class_freq
        self.adapted_class_freq = adapted_class_freq
        assert np.isclose(np.sum(self.original_class_freq), 1.0)
        assert np.isclose(np.sum(self.adapted_class_freq), 1.0)

    def __call__(self, unadapted_posterior_probs):
        #if supplied probs are in binary format, convert to softmax format
        if (len(unadapted_posterior_probs.shape)==1
            or unadapted_posterior_probs.shape[1]==1):
            softmax_unadapted_posterior_probs = np.zeros(
                (len(unadapted_posterior_probs),2)) 
            softmax_unadapted_posterior_probs[:,0] =\
                unadapted_posterior_probs
            softmax_unadapted_posterior_probs[:,1] =\
                1-unadapted_posterior_probs
        else:
            softmax_unadapted_posterior_probs =\
                unadapted_posterior_probs

        adapted_posterior_probs_unnorm =(
            softmax_unadapted_posterior_probs*
             (self.adapted_class_freq[None,:]/
              self.original_class_freq[None,:]))
        adapted_posterior_probs = (
            adapted_posterior_probs_unnorm/
            np.sum(adapted_posterior_probs_unnorm,axis=-1)[:,None])

        #return to binary format if appropriate
        if (len(unadapted_posterior_probs.shape)==1
            or unadapted_posterior_probs.shape[1]==1):
            if (len(unadapted_posterior_probs.shape)==1):
                adapted_posterior_probs =\
                    adapted_posterior_probs[:,1] 
            else:
                if (unadapted_posterior_probs.shape[1]==1):
                    adapted_posterior_probs =\
                        adapted_posterior_probs[:,1:2] 

        return adapted_posterior_probs


class AbstractImbalanceAdapter(object):

    def __call__(self, valid_labels, tofit_initial_posterior_probs,
                       valid_posterior_probs):
        raise NotImplementedError()


class EMImbalanceAdapter(AbstractImbalanceAdapter):

    def __init__(self, verbose=True,
                       tolerance=1E-3,
                       max_iterations=100):
        self.verbose = verbose
        self.tolerance = tolerance

    def __call__(self, valid_labels,
                       tofit_initial_posterior_probs,
                       valid_posterior_probs=None):

        assert len(valid_labels.shape)<=2
        #if binary labels were provided, convert to softmax format
        # for consistency
        if (len(valid_labels.shape)==1 or valid_labels.shape[1]==1):
            valid_labels = np.squeeze(valid_labels)
            valid_posterior_probs = np.squeeze(valid_posterior_probs)
            softmax_valid_labels = np.zeros((len(valid_labels),2))
            softmax_valid_labels[:,1] = valid_labels
            softmax_valid_labels[:,0] = 1-valid_labels
        else:
            softmax_valid_labels = valid_labels
        valid_class_freq = np.mean(softmax_valid_labels, axis=0)
        if (self.verbose):
            print("Original class freq", valid_class_freq)
        
        if (len(tofit_initial_posterior_probs.shape)==1
            or tofit_initial_posterior_probs.shape[1]==1):
            softmax_initial_posterior_probs = np.zeros(
                (len(tofit_initial_posterior_probs),2)) 
            softmax_initial_posterior_probs[:,0] =\
                tofit_initial_posterior_probs
            softmax_initial_posterior_probs[:,1] =\
                1-tofit_initial_posterior_probs
        else:
            softmax_initial_posterior_probs = tofit_initial_posterior_probs

        current_iter_class_freq = valid_class_freq
        current_iter_posterior_probs = softmax_initial_posterior_probs
        next_iter_class_imbalance = None
        next_iter_posterior_probs = None
        iter_number = 0
        while (next_iter_class_imbalance is None
            or (np.sum(np.abs(next_iter_class_imbalance
                              -current_iter_class_freq)
                       > self.tolerance))):
            if (next_iter_class_imbalance is not None):
                current_iter_class_freq=next_iter_class_imbalance 
                current_iter_posterior_probs=next_iter_posterior_probs
            next_iter_class_imbalance = np.mean(
                current_iter_posterior_probs, axis=0) 
            next_iter_posterior_probs_unnorm =(
                (softmax_initial_posterior_probs
                 *next_iter_class_imbalance[None,:])/
                valid_class_freq[None,:])
            next_iter_posterior_probs = (
                next_iter_posterior_probs_unnorm/
                np.sum(next_iter_posterior_probs_unnorm,axis=-1)[:,None])
            iter_number += 1
        if (self.verbose):
            print("Finished on iteration",iter_number,"with delta",
                  np.sum(np.abs(current_iter_class_freq-
                                next_iter_class_imbalance)))
        current_iter_class_freq = next_iter_class_imbalance
        if (self.verbose):
            print("Final freq", current_iter_class_freq)
            print("Multiplier:",current_iter_class_freq/valid_class_freq)

        return PriorShiftAdapterFunc(
                original_class_freq=valid_class_freq,
                adapted_class_freq=current_iter_class_freq)
