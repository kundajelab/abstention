from __future__ import division, print_function
import numpy as np
from .calibration import (inverse_softmax, get_hard_preds,
                          map_to_softmax_format_if_appropriate)
from scipy import linalg


class AbstractImbalanceAdapterFunc(object):

    def __call__(self, unadapted_posterior_probs):
        raise NotImplementedError()


class PriorShiftAdapterFunc(AbstractImbalanceAdapterFunc):

    def __init__(self, multipliers, calibrator_func=lambda x: x):
        self.multipliers = multipliers
        self.calibrator_func = calibrator_func

    def __call__(self, unadapted_posterior_probs):

        unadapted_posterior_probs =\
            self.calibrator_func(unadapted_posterior_probs)

        #if supplied probs are in binary format, convert to softmax format
        softmax_unadapted_posterior_probs =\
            map_to_softmax_format_if_appropriate(
                values=unadapted_posterior_probs) 
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


class NoAdaptation(object):

    def __call__(self, valid_labels, tofit_initial_posterior_probs,
                      valid_posterior_probs):
        return lambda unadapted_posterior_probs: unadapted_posterior_probs


class AbstractShiftWeightEstimator(object):
    
    # Should return the ratios of the weights for each class 
    def __call__(self, valid_labels,
                       tofit_initial_posterior_probs,
                       valid_posterior_probs):
        raise NotImplementedError()


class NoWeightShift(AbstractShiftWeightEstimator):

    def __call__(self, valid_labels, tofit_initial_posterior_probs,
                       valid_posterior_probs):
        return np.ones(valid_posterior_probs.shape[1])


class EMImbalanceAdapter(AbstractImbalanceAdapter):

    def __init__(self, estimate_priors_from_valid_labels=False,
                       verbose=False,
                       tolerance=1E-6,
                       max_iterations=100,
                       calibrator_factory=None,
                       initialization_weight_ratio=NoWeightShift()):
        self.estimate_priors_from_valid_labels =\
            estimate_priors_from_valid_labels
        self.verbose = verbose
        self.tolerance = tolerance
        self.calibrator_factory = calibrator_factory
        self.max_iterations = max_iterations
        self.initialization_weight_ratio = initialization_weight_ratio

    #valid_labels are only needed if calibration is to be performed
    # or if self.estimate_priors_from_valid_labels is True
    def __call__(self, tofit_initial_posterior_probs,
                       valid_posterior_probs,
                       valid_labels=None):

        #if binary labels were provided, convert to softmax format
        # for consistency
        softmax_valid_posterior_probs =\
            map_to_softmax_format_if_appropriate(
                values=valid_posterior_probs)
        softmax_initial_posterior_probs =\
            map_to_softmax_format_if_appropriate(
                values=tofit_initial_posterior_probs)
        if (valid_labels is not None):
            softmax_valid_labels =\
                map_to_softmax_format_if_appropriate(
                    values=valid_labels)
        else:
            softmax_valid_labels = None
      
        #fit calibration if needed
        if (self.calibrator_factory is not None):
            assert softmax_valid_posterior_probs is not None 
            calibrator_func = self.calibrator_factory(
                valid_preacts=softmax_valid_posterior_probs,
                valid_labels=softmax_valid_labels,
                posterior_supplied=True) 
        else:
            calibrator_func = lambda x: x
        softmax_valid_posterior_probs = calibrator_func(softmax_valid_posterior_probs)
        tofit_initial_posterior_probs = calibrator_func(
            tofit_initial_posterior_probs)

        if (self.estimate_priors_from_valid_labels):
            valid_class_freq = np.mean(valid_labels, axis=0)
        else:
            #compute the class frequencies based on the posterior probs to ensure
            # that if the valid posterior probs are supplied for "to fit", then
            # no shift is estimated
            valid_class_freq = np.mean(softmax_valid_posterior_probs, axis=0)

        if (self.verbose):
            print("Original class freq", valid_class_freq)

        #initialization_weight_ratio is a method that can be used to
        # estimate the ratios between the label frequencies in the
        # validation set and the to_fit set; it can be used to obtain a
        # better initialization for the class frequencies
        #We normalize the frequencies to sum to 1 because methods like BBSE
        # are not guaranteed to return weights that give probs that are valid
        first_iter_class_freq = (
         valid_class_freq*self.initialization_weight_ratio(
            valid_labels = softmax_valid_labels,
            tofit_initial_posterior_probs = softmax_initial_posterior_probs,
            valid_posterior_probs = softmax_valid_posterior_probs))
        first_iter_class_freq = (first_iter_class_freq/
                                 np.sum(first_iter_class_freq))

        current_iter_class_freq = first_iter_class_freq
        current_iter_posterior_probs = softmax_initial_posterior_probs
        next_iter_class_freq = None
        next_iter_posterior_probs = None
        iter_number = 0
        while ((next_iter_class_freq is None
            or (np.sum(np.abs(next_iter_class_freq
                              -current_iter_class_freq)
                       > self.tolerance)))
            and iter_number < self.max_iterations):

            if (next_iter_class_freq is not None):
                current_iter_class_freq=next_iter_class_freq 
                current_iter_posterior_probs=next_iter_posterior_probs
            current_iter_posterior_probs_unnorm =(
                (softmax_initial_posterior_probs
                 *current_iter_class_freq[None,:])/
                valid_class_freq[None,:])
            current_iter_posterior_probs = (
                current_iter_posterior_probs_unnorm/
                np.sum(current_iter_posterior_probs_unnorm,axis=-1)[:,None])

            next_iter_class_freq = np.mean(
                current_iter_posterior_probs, axis=0) 
            if (self.verbose):
                print("next it",next_iter_class_freq)
            iter_number += 1
        if (self.verbose):
            print("Finished on iteration",iter_number,"with delta",
                  np.sum(np.abs(current_iter_class_freq-
                                next_iter_class_freq)))
        current_iter_class_freq = next_iter_class_freq
        if (self.verbose):
            print("Final freq", current_iter_class_freq)
            print("Multiplier:",current_iter_class_freq/valid_class_freq)

        return PriorShiftAdapterFunc(
                    multipliers=(current_iter_class_freq/valid_class_freq),
                    calibrator_func=calibrator_func)


class RLLSImbalanceAdapter(AbstractImbalanceAdapter):

    def __init__(self, soft=False,
                       calibrator_factory=None,
                       #default value of alpha comes from https://github.com/Angela0428/labelshift/blob/5bbe517938f4e3f5bd14c2c105de973dcc2e0917/label_shift.py#L455
                       alpha=0.01,
                       verbose=False):
        self.soft = soft
        self.alpha = alpha
        self.calibrator_factory = calibrator_factory
        self.verbose = verbose

    #based on: https://github.com/Angela0428/labelshift/blob/5bbe517938f4e3f5bd14c2c105de973dcc2e0917/label_shift.py#L184
    def compute_3deltaC(self, n_class, n_train, delta):
        rho = 3*(2*np.log(2*n_class/delta)/(3*n_train)
                 + np.sqrt(2*np.log(2*n_class/delta)/n_train))
        return rho 

    #based on: https://github.com/Angela0428/labelshift/blob/5bbe517938f4e3f5bd14c2c105de973dcc2e0917/label_shift.py#L123
    def compute_w_opt(self, C_yy, mu_y, mu_train_y, rho):
        import cvxpy as cp
        n = C_yy.shape[0]
        theta = cp.Variable(n)
        b = mu_y - mu_train_y
        objective = cp.Minimize(cp.pnorm(C_yy*theta - b) + rho* cp.pnorm(theta))
        constraints = [-1 <= theta]
        prob = cp.Problem(objective, constraints)

        # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve()
        # The optimal value for x is stored in `x.value`.
        # print(theta.value)
        w = 1 + theta.value
        if (self.verbose):
            print('Estimated w is', w)
        return w 

    def __call__(self, valid_labels, tofit_initial_posterior_probs,
                       valid_posterior_probs):
 
        if (self.calibrator_factory is not None):
            calibrator_func = self.calibrator_factory(
                valid_preacts=valid_posterior_probs,
                valid_labels=valid_labels,
                posterior_supplied=True) 
        else:
            calibrator_func = lambda x: x

        valid_posterior_probs =\
            calibrator_func(valid_posterior_probs)
        tofit_initial_posterior_probs =\
            calibrator_func(tofit_initial_posterior_probs)

        #hard_tofit_preds binarizes tofit_initial_posterior_probs
        # according to the argmax predictions
        hard_tofit_preds = get_hard_preds(
            softmax_preds=tofit_initial_posterior_probs)
        hard_valid_preds = get_hard_preds(
            softmax_preds=valid_posterior_probs)

        if (self.soft):
            mu_y = np.mean(tofit_initial_posterior_probs, axis=0) 
        else:
            mu_y = np.mean(hard_tofit_preds, axis=0) 

        if (self.soft):
            mu_train_y = np.mean(valid_posterior_probs, axis=0) 
        else:
            mu_train_y = np.mean(hard_valid_preds, axis=0) 

        #prepare the "confusion" matrix (confusingly named as confusion
        # matrices are usually normalized, but theirs isn't)
        if (self.soft):
            C_yy = np.mean((
                valid_posterior_probs[:,:,None]*
                valid_labels[:,None,:]), axis=0)
        else:
            C_yy = np.mean((hard_valid_preds[:,:,None]*
                            valid_labels[:,None,:]),axis=0) 

        n_class = C_yy.shape[0]  
        m_train = len(valid_posterior_probs)
        #from https://github.com/Angela0428/labelshift/blob/5bbe517938f4e3f5bd14c2c105de973dcc2e0917/label_shift.py#L453
        rho = self.compute_3deltaC(n_class=n_class,
                                   n_train=m_train,
                                   delta=0.05)
        weights = self.compute_w_opt(
            C_yy=C_yy,
            mu_y=mu_y,
            mu_train_y=mu_train_y,
            rho=self.alpha*rho)
         
        return PriorShiftAdapterFunc(
                    multipliers=weights,
                    calibrator_func=calibrator_func)
        

class BBSEImbalanceAdapter(AbstractImbalanceAdapter):

    def __init__(self, soft=False, calibrator_factory=None, verbose=False):
        self.soft = soft
        self.calibrator_factory = calibrator_factory
        self.verbose = verbose

    def __call__(self, valid_labels, tofit_initial_posterior_probs,
                       valid_posterior_probs):

        if (self.calibrator_factory is not None):
            calibrator_func = self.calibrator_factory(
                valid_preacts=valid_posterior_probs,
                valid_labels=valid_labels,
                posterior_supplied=True) 
        else:
            calibrator_func = lambda x: x

        valid_posterior_probs =\
            calibrator_func(valid_posterior_probs)
        tofit_initial_posterior_probs =\
            calibrator_func(tofit_initial_posterior_probs)

        #hard_tofit_preds binarizes tofit_initial_posterior_probs
        # according to the argmax predictions
        hard_tofit_preds = get_hard_preds(
            softmax_preds=tofit_initial_posterior_probs)
        hard_valid_preds = get_hard_preds(
            softmax_preds=valid_posterior_probs)

        if (self.soft):
            muhat_yhat = np.mean(tofit_initial_posterior_probs, axis=0) 
        else:
            muhat_yhat = np.mean(hard_tofit_preds, axis=0) 

        #prepare the "confusion" matrix (confusingly named as confusion
        # matrices are usually normalized, but theirs isn't)
        if (self.soft):
            confusion_matrix = np.mean((
                valid_posterior_probs[:,:,None]*
                valid_labels[:,None,:]), axis=0)
        else:
            confusion_matrix = np.mean((hard_valid_preds[:,:,None]*
                                        valid_labels[:,None,:]),axis=0) 
        inv_confusion = linalg.inv(confusion_matrix)
        weights = inv_confusion.dot(muhat_yhat)
        if (self.verbose):
            if (np.sum(weights < 0) > 0):
                print("Heads up - some estimated weights were negative")
        weights = 1.0*(weights*(weights >= 0)) #mask out negative weights

        return PriorShiftAdapterFunc(
                    multipliers=weights,
                    calibrator_func=calibrator_func)


#effectively a wrapper around an ImbalanceAdapter
class ShiftWeightFromImbalanceAdapter(AbstractShiftWeightEstimator):

    def __init__(self, imbalance_adapter):
        self.imbalance_adapter = imbalance_adapter 

    def __call__(self, valid_labels, tofit_initial_posterior_probs,
                      valid_posterior_probs): 
        prior_shift_adapter_func = self.imbalance_adapter(
            valid_labels=valid_labels,
            tofit_initial_posterior_probs=tofit_initial_posterior_probs,
            valid_posterior_probs=valid_posterior_probs)
        return prior_shift_adapter_func.multipliers

