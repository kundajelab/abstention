from __future__ import division, print_function, absolute_import
import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss
import scipy
from scipy.special import expit
import scipy.misc
import scipy.optimize
from sklearn.isotonic import IsotonicRegression as IR
from sklearn.linear_model import LogisticRegression as LR


def map_to_softmax_format_if_appropriate(values):
    if (len(values.shape)==1 or values.shape[1]==1):
        values = np.squeeze(values)
        softmax_values = np.zeros((len(values),2))
        softmax_values[:,1] = values
        softmax_values[:,0] = 1-values
    else:
        softmax_values = values
    return softmax_values


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


def get_hard_preds(softmax_preds):
    hard_preds = np.zeros(softmax_preds.shape)
    hard_preds[list(range(softmax_preds.shape[0])),
               np.argmax(softmax_preds, axis=-1)] = 1.0 
    return hard_preds


class ConfusionMatrix(CalibratorFactory):

    def __call__(self, valid_preacts, valid_labels, posterior_supplied=None):

        valid_hard_preds = get_hard_preds(softmax_preds=valid_preacts) 
        denom = (1E-7*((np.sum(valid_hard_preds,axis=0) > 0)==False)
                 + np.sum(valid_hard_preds,axis=0))
        confusion_matrix = (np.sum(valid_hard_preds[:,:,None]
                                   *valid_labels[:,None,:], axis=0)/denom[:,None])
        return (lambda preact: confusion_matrix[np.argmax(preact,axis=-1)]) 


class TempScaling(CalibratorFactory):

    def __init__(self, ece_bins=15, lbfgs_kwargs={}, verbose=False,
                       bias_positions=[]):
        self.lbfgs_kwargs = lbfgs_kwargs
        self.verbose = verbose
        self.ece_bins = ece_bins
        #the subset of bias positions that we are allowed to optimize for
        self.bias_positions = bias_positions

    def __call__(self, valid_preacts, valid_labels, posterior_supplied=False):

        if (posterior_supplied):
            valid_preacts = inverse_softmax(valid_preacts)
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
            grads_b = valid_labels - (exp_tsb_logits/(sum_exp[:,None]))
            #multiply by -1 because we care about *negative* log likelihood
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
                                  fun=eval_func,
                                  #fun=lambda x: eval_func(x)[0],
                                  x0=np.array([1.0]+[0.0 for x in
                                                     self.bias_positions]),
                                  bounds=[(0,None)]+[(None,None) for x in
                                                     self.bias_positions],
                                  jac=True,
                                  #jac=False,
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

        return (lambda preact: softmax(preact=(inverse_softmax(preact)
                                               if posterior_supplied else
                                               preact),
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
