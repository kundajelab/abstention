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


def inverse_softmax(preds):
    return np.log(preds) - np.mean(np.log(preds),axis=1)[:,None]


def softmax(preact, temp, biases):
    if (biases is None):
        biases = np.zeros(preact.shape[1])
    exponents = np.exp(preact/temp + biases[None,:])
    sum_exponents = np.sum(exponents, axis=1) 
    return exponents/sum_exponents[:,None]


#based on https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py#L78
def compute_ece_with_bins(softmax_out, labels, bins):

    bin_boundaries = np.linspace(0,1,num=bins)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(softmax_out,axis=1)
    is_correct = np.argmax(softmax_out,axis=1)==np.argmax(labels,axis=1)

    ece = 0.0 
    avg_confidence_bins = []
    accuracy_bins = []
    prop_in_bins = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower)*(confidences <= bin_upper)
        prop_in_bin = np.mean((in_bin)) 
        if (prop_in_bin > 0.0):
            accuracy_in_bin = np.mean(is_correct[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            avg_confidence_bins.append(avg_confidence_in_bin)
            accuracy_bins.append(accuracy_in_bin)
            prop_in_bins.append(prop_in_bin) 
    return (np.array(avg_confidence_bins),
            np.array(accuracy_bins),
            np.array(prop_in_bins),
            np.sum(np.abs(np.array(avg_confidence_bins)
                          -np.array(accuracy_bins))
                   *np.array(prop_in_bins))*100)


#based on https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py#L78
def compute_ece(softmax_out, labels, bins):
    (_, _, _, ece) = compute_ece_with_bins(softmax_out=softmax_out,
                                        labels=labels, bins=bins)
    return ece


class CalibratorFactory(object):

    def __call__(self, valid_preacts, valid_labels):
        raise NotImplementedError()


class Softmax(CalibratorFactory):

    def __call__(self, valid_preacts=None, valid_labels=None):
        return (lambda x: softmax(preact=x, temp=1.0))


class TempScaling(CalibratorFactory):

    def __init__(self, ece_bins=15, lbfgs_kwargs={}, verbose=True,
                       bias_positions=[]):
        self.lbfgs_kwargs = lbfgs_kwargs
        self.verbose = verbose
        self.ece_bins = ece_bins
        #the subset of bias positions that we are allowed to optimize for
        self.bias_positions = bias_positions

    def __call__(self, valid_preacts, valid_labels):

        assert np.max(np.sum(valid_labels,axis=1)==1.0)

        #calculate the temperature scaling parameter
        def eval_func(x):
            t = x[0]
            bs = np.zeros(valid_labels.shape[1])
            for bias_pos_idx, bias_pos in enumerate(self.bias_positions):
                bs[bias_pos] = x[1+bias_pos_idx] 
            #tsb = temp_scaled_biased
            tsb_valid_preacts = valid_preacts/float(t) + bs[None,:]
            log_sum_exp = scipy.misc.logsumexp(a = tsb_valid_preacts,axis=1) 

            exp_tsb_logits = np.exp(tsb_valid_preacts)
            sum_exp = np.sum(exp_tsb_logits, axis=1)
            sum_preact_times_exp = np.sum(valid_preacts*
                                          exp_tsb_logits, axis=1)

            notsb_logits_trueclass =\
                np.sum(valid_preacts*valid_labels, axis=1)
            tsb_logits_trueclass =\
                np.sum(tsb_valid_preacts*valid_labels, axis=1)
            
            log_likelihoods = tsb_logits_trueclass - log_sum_exp
            nll = -np.mean(log_likelihoods)
            grads_t = ((sum_preact_times_exp/sum_exp
                        - notsb_logits_trueclass)/\
                        (float(t)**2))
            grads_b =(
                (valid_labels +
                 ((1-valid_labels)/np.exp(tsb_logits_trueclass)[:,None]))
                + (exp_tsb_logits/sum_exp[:,None])) 
            mean_grad_t = -np.mean(grads_t)
            mean_grads_b = -np.mean(grads_b, axis=0)
            #only supply the gradients for the bias positions that
            # we are allowed to optimize for
            mean_grads_b_masked = []
            for bias_pos_idx, bias_pos in enumerate(self.bias_positions):
                mean_grads_b_masked.append(mean_grads_b[bias_pos])
            return nll, np.array([mean_grad_t]+mean_grads_b_masked)

        if (self.verbose):
            original_nll = eval_func(np.array([1.0]+[0.0 for x in
                                                     self.bias_positions])) 
            original_ece = compute_ece(
                softmax_out=softmax(preact=valid_preacts,
                                    temp=1.0, biases=None),
                labels=valid_labels, bins=self.ece_bins) 
            print("Original NLL & grad is: ",original_nll)
            print("Original ECE is: ",original_ece)
            
        optimization_result = scipy.optimize.minimize(
                                  #fun=eval_func,
                                  fun=lambda x: eval_func(x)[0],
                                  x0=np.array([1.0]+[0.0 for x in
                                                     self.bias_positions]),
                                  bounds=[(0,None)]+[(None,None) for x in
                                                     self.bias_positions],
                                  #jac=True,
                                  jac=False,
                                  method='L-BFGS-B',
                                  tol=1e-07,
                                  **self.lbfgs_kwargs)
        if (self.verbose):
            print(optimization_result)
        biases = np.zeros(valid_labels.shape[1])
        if (hasattr(optimization_result.x, '__iter__')):
            optimal_t = optimization_result.x[0]
            for bias_pos_idx,bias_pos in enumerate(self.bias_positions):
               biases[bias_pos] = optimization_result.x[1+bias_pos_idx] 
            final_nll = eval_func(np.array(optimization_result.x)) 
        else:
            optimal_t = optimization_result.x
            final_nll = eval_func(np.array([optimal_t])) 

        if (self.verbose):
            final_ece = compute_ece(
                softmax_out=softmax(preact=valid_preacts,
                                    temp=optimal_t, biases=biases),
                labels=valid_labels, bins=self.ece_bins) 
            print("Final NLL & grad is: ",final_nll)
            print("Final ECE is: ",final_ece)

        return (lambda preact: softmax(preact=preact,
                                       temp=optimal_t,
                                       biases=biases))


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
                       bias_corrector_factory,
                       num_loops=1):
        self.base_calibrator_factory = base_calibrator_factory
        self.bias_corrector_factory = bias_corrector_factory
        self.num_loops = num_loops

    def __call__(self, valid_preacts, valid_labels):
        base_calibration_funcs = []
        bias_corrector_funcs = []
        valid_posterior_probs = softmax(preact=valid_preacts, temp=1)
        current_valid_probs = valid_posterior_probs
        for loop_num in range(self.num_loops):
            base_calibration_func = self.base_calibrator_factory(
                valid_preacts=inverse_softmax(preds=current_valid_probs),
                valid_labels=valid_labels)
            base_calibration_funcs.append(base_calibration_func)
            current_valid_probs = base_calibration_func(
                preact=inverse_softmax(preds=current_valid_probs)) 
            bias_corrector_func = self.bias_corrector_factory(
                valid_posterior_probs=current_valid_probs,
                valid_labels=valid_labels)
            bias_corrector_funcs.append(bias_corrector_func)
            current_valid_probs = bias_corrector_func(
                unadapted_posterior_probs=current_valid_probs)
        def calibration_func(preact):
            probs = softmax(preact=preact, temp=1)
            for loop_num in range(self.num_loops):
                probs = base_calibration_funcs[loop_num](
                    preact=inverse_softmax(preds=probs))
                probs = bias_corrector_funcs[loop_num](
                    unadapted_posterior_probs=probs)
            return probs
        return calibration_func
        

class EMBiasCorrectorFactory(object):

    def __init__(self, tolerance=1E-3,
                       max_iter=100,
                       num_bootstrap_samples=50,
                       ece_bins=15,
                       verbose=False,
                       subset_to_adjust=None):
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.num_bootstrap_samples = num_bootstrap_samples
        self.ece_bins = ece_bins
        self.verbose = verbose
        self.subset_to_adjust = subset_to_adjust

    def __call__(self, valid_posterior_probs, valid_labels):
        
        #idea: figure out what prior distribution valid_posterior_probs
        # originated from to explain the difference between
        # np.mean(valid_labels, axis=0)
        # and np.mean(valid_posterior_probs, axis=0)

        if (self.verbose):
            print("Observed class freq:",np.mean(valid_labels,axis=0))

        multiplier_samples = []
        ece_delta_samples = []
        bootstrap_rng = np.random.RandomState() 
        for bootstrap_sample_num in range(self.num_bootstrap_samples): 
            bootstrap_rng.seed(100*bootstrap_sample_num)
            heldout_valid_indices = np.random.choice(
                np.array(list(range(len(valid_labels)))),
                size=len(valid_labels), replace=True) 
            heldin_valid_indices = np.random.choice(
                np.array(list(range(len(valid_labels)))),
                size=len(valid_labels), replace=True) 
            heldout_valid_labels = np.array([
                valid_labels[x] for x in heldout_valid_indices])
            heldout_valid_posterior_probs = np.array([
                valid_posterior_probs[x] for x in heldout_valid_indices]) 
            heldin_valid_labels = np.array([
                valid_labels[x] for x in heldin_valid_indices])
            heldin_valid_posterior_probs = np.array([
                valid_posterior_probs[x] for x in heldin_valid_indices]) 

            observed_class_freq = np.mean(heldin_valid_labels,axis=0)
            current_orig_freq = np.array(observed_class_freq)
            terminate = False
            iterations = 0
            while (terminate==False):
                current_ex_weights =\
                    np.sum((current_orig_freq/observed_class_freq)[None,:]
                           *heldin_valid_labels,axis=1)
                next_orig_freq =\
                    np.mean(current_ex_weights[:,None]
                            *heldin_valid_posterior_probs,
                            axis=0)
                iterations += 1
                if ((np.sum(np.abs(next_orig_freq-current_orig_freq)) <
                     self.tolerance) or iterations > self.max_iter):
                    terminate = True
                current_orig_freq = next_orig_freq

            if (self.subset_to_adjust is None):
                multiplier = (observed_class_freq/current_orig_freq)
            else:
                multiplier = np.zeros(len(observed_class_freq))
                for a_class in self.subset_to_adjust:
                    multiplier[a_class] = (observed_class_freq[a_class]/
                                           current_orig_freq[a_class])
                obs_freq_rest = 1-sum([observed_class_freq[a_class]
                                       for a_class in self.subset_to_adjust])
                orig_freq_rest = 1-sum([current_orig_freq[a_class]
                                       for a_class in self.subset_to_adjust])
                if (orig_freq_rest > 0):
                    for a_class in range(len(observed_class_freq)):
                        if (a_class not in self.subset_to_adjust):
                            multiplier[a_class] = (obs_freq_rest/
                                                   orig_freq_rest) 

            adapted_heldout_valid_posterior_probs =\
                PriorShiftAdapterFunc(multipliers=multiplier)(
                 unadapted_posterior_probs=heldout_valid_posterior_probs) 
            unadapted_ece = compute_ece(
                softmax_out=heldout_valid_posterior_probs,
                labels=heldout_valid_labels,
                bins=self.ece_bins)
            adapted_ece = compute_ece(
                softmax_out=adapted_heldout_valid_posterior_probs,
                labels=heldout_valid_labels,
                bins=self.ece_bins) 
            ece_delta = adapted_ece-unadapted_ece

            ece_delta_samples.append(ece_delta)
            multiplier_samples.append(multiplier)

        multiplier_samples = np.array(multiplier_samples)
        multiplier_samples_lower_percentile = np.percentile(
            a=multiplier_samples, q=5, axis=0)
        multiplier_samples_upper_percentile = np.percentile(
            a=multiplier_samples, q=95, axis=0)
        geometric_mean_multiplier_samples =(
            np.exp(np.mean(np.log(multiplier_samples),axis=0)))
        if (self.verbose):
            print("Ece delta samples",
                  "mean:",np.mean(ece_delta_samples),
                  "5th percentile:", np.percentile(ece_delta_samples, 5),
                  "95th percentile:",np.percentile(ece_delta_samples, 95))
            print("Geometric mean est. multipliers:",
                  geometric_mean_multiplier_samples)
            print("lower percentile:",
                  multiplier_samples_lower_percentile)
            print("upper percentile:",
                  multiplier_samples_upper_percentile)
        return PriorShiftAdapterFunc(
                multipliers=geometric_mean_multiplier_samples)


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

    def __init__(self, multipliers):
        self.multipliers = multipliers

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
            softmax_unadapted_posterior_probs*self.multipliers[None,:])
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
                multipliers=(current_iter_class_freq/valid_class_freq))
