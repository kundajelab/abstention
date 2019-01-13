from __future__ import division, print_function
import numpy as np
from .calibration import inverse_softmax
from scipy import linalg


class AbstractImbalanceAdapterFunc(object):

    def __call__(self, unadapted_posterior_probs):
        raise NotImplementedError()


class PriorShiftAdapterFunc(AbstractImbalanceAdapterFunc):

    def __init__(self, multipliers, calibrator_func=lambda x: x):
        self.multipliers = multipliers
        self.calibrator_func = calibrator_func

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

        softmax_unadapted_posterior_probs =\
            calibrator_func(softmax_unadapted_posterior_probs)

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


def map_to_softmax_format_if_approrpiate(values):
    if (len(values.shape)==1 or values.shape[1]==1):
        values = np.squeeze(values)
        softmax_values = np.zeros((len(values),2))
        softmax_values[:,1] = values
        softmax_values[:,0] = 1-values
    else:
        softmax_values = values
    return softmax_values


class EMImbalanceAdapter(AbstractImbalanceAdapter):

    def __init__(self, verbose=False,
                       tolerance=1E-3,
                       max_iterations=100,
                       calibrator_factory=None):
        self.verbose = verbose
        self.tolerance = tolerance
        self.calibrator_factory = calibrator_factory

    def __call__(self, valid_labels,
                       tofit_initial_posterior_probs,
                       valid_posterior_probs=None):

        assert len(valid_labels.shape)<=2
        softmax_valid_labels = map_to_softmax_format_if_approrpiate(
                                  values=valid_labels)
        if (valid_posterior_probs is not None):
            softmax_valid_posterior_probs =\
                map_to_softmax_format_if_approrpiate(
                    values=valid_posterior_probs)
        #if binary labels were provided, convert to softmax format
        # for consistency
        if (self.calibrator_factory is not None):
            assert valid_posterior_probs is not None 
            calibrator_func = self.calibrator_factory(
                valid_preacts=softmax_valid_posterior_probs,
                valid_labels=softmax_valid_labels,
                posterior_supplied=True) 
        else:
            calibrator_func = lambda x: x

        if (valid_posterior_probs is not None):
            valid_posterior_probs = calibrator_func(valid_posterior_probs)

        valid_class_freq = np.mean(softmax_valid_labels, axis=0)

        if (self.verbose):
            print("Original class freq", valid_class_freq)
       
        softmax_initial_posterior_probs =\
            calibrator_func(map_to_softmax_format_if_approrpiate(
                values=tofit_initial_posterior_probs))

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
                    multipliers=(current_iter_class_freq/valid_class_freq),
                    calibrator_func=calibrator_func)


class AbstractShiftWeightEstimator(object):
    
    # Should return the ratios of the weights for each class 
    def __call__(self, valid_labels,
                       tofit_initial_posterior_probs,
                       valid_posterior_probs):
        raise NotImplementedError()


#effectively a wrapper around EMImbalanceAdapter
class EMShiftWeightEstimator(AbstractShiftWeightEstimator):

    def __init__(self, **em_imbalance_adapter_kwargs):
        self.imbalance_adapter = EMImbalanceAdapter(
            **em_imbalance_adapter_kwargs) 

    def __call__(self, valid_labels, tofit_initial_posterior_probs,
                      valid_posterior_probs=None): 
        prior_shift_adapter_func = self.imbalance_adapter(
            valid_labels=valid_labels,
            tofit_initial_posterior_probs=tofit_initial_posterior_probs,
            valid_posterior_probs=valid_posterior_probs)
        return prior_shift_adapter_func.multipliers
        

class BBSE(AbstractShiftWeightEstimator):

    def __init__(self, soft=False, calibrator_factory=None):
        self.soft = soft
        self.calibrator_factory = calibrator_factory

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
        hard_tofit_preds = []
        for pred in tofit_initial_posterior_probs:
            to_append = np.zeros(len(pred))
            to_append[np.argmax(pred)] = 1
            hard_tofit_preds.append(to_append)
        hard_tofit_preds = np.array(hard_tofit_preds)
        hard_valid_preds = []
        for pred in valid_posterior_probs:
            to_append = np.zeros(len(pred))
            to_append[np.argmax(pred)] = 1
            hard_valid_preds.append(to_append)
        hard_valid_preds = np.array(hard_valid_preds)

        if (self.soft):
            muhat_yhat = np.mean(tofit_initial_posterior_probs, axis=0) 
        else:
            muhat_yhat = np.mean(hard_tofit_preds, axis=0) 

        #prepare the confusion matrix
        if (self.soft):
            confusion_matrix = np.mean((
                valid_posterior_probs[:,:,None]*
                valid_labels[:,None,:]), axis=0)
        else:
            confusion_matrix = np.mean((hard_valid_preds[:,:,None]*
                                        valid_labels[:,None,:]),axis=0) 
        inv_confusion = linalg.inv(confusion_matrix)
        weights = inv_confusion.dot(muhat_yhat)
        return weights

