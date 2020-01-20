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


def vector_scaled_softmax(preact, ws, biases):
    exponents = np.exp(preact*ws[None,:] + biases[None,:])
    sum_exponents = np.sum(exponents, axis=1)
    return (exponents/sum_exponents[:,None])


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
        return (lambda preact: softmax(preact=preact,
                                       temp=1.0, biases=None))


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


def compute_nll(labels, preacts, t, bs):
    tsb_preacts = preacts/float(t) + bs[None,:]
    return compute_nll_given_preacts(labels=labels, preacts=tsb_preacts)


def compute_nll_given_preacts(labels, preacts):
    log_sum_exp = scipy.special.logsumexp(a=preacts, axis=1) 
    tsb_logits_trueclass = np.sum(preacts*labels, axis=1)
    log_likelihoods = tsb_logits_trueclass - log_sum_exp
    nll = -np.mean(log_likelihoods)
    return nll


def do_regularized_tempscale_optimization(labels, preacts, beta, verbose,
                                          lbfgs_kwargs):
    #beta is the regularization parameter
    def eval_func(x):
        t = x[0]
        bs = np.array(x[1:])
        #tsb = temp_scaled_biased
        tsb_preacts = preacts/float(t) + bs[None,:]
        log_sum_exp = scipy.special.logsumexp(a=tsb_preacts, axis=1) 

        exp_tsb_logits = np.exp(tsb_preacts)
        sum_exp = np.sum(exp_tsb_logits, axis=1)
        sum_preact_times_exp = np.sum(preacts*exp_tsb_logits, axis=1)

        notsb_logits_trueclass =\
            np.sum(preacts*labels, axis=1)
        tsb_logits_trueclass =\
            np.sum(tsb_preacts*labels, axis=1)
        
        log_likelihoods = tsb_logits_trueclass - log_sum_exp
        objective = -np.mean(log_likelihoods) + beta*np.sum(np.square(bs))
        grads_t = ((sum_preact_times_exp/sum_exp
                    - notsb_logits_trueclass)/\
                    (float(t)**2))
        grads_b = labels - (exp_tsb_logits/(sum_exp[:,None]))
        #multiply by -1 because we care about *negative* log likelihood
        mean_grad_t = -np.mean(grads_t)
        mean_grads_b = ((-np.mean(grads_b, axis=0)) + (2*bs*beta))
        return objective, np.array([mean_grad_t]+list(mean_grads_b))

    if (verbose):
        original_nll = compute_nll(labels=labels, preacts=preacts,
                                   t=1.0, bs=np.zeros(labels.shape[1]))
        print("Original NLL is: ",original_nll)
        
    optimization_result = scipy.optimize.minimize(
                              fun=eval_func,
                              #fun=lambda x: eval_func(x)[0],
                              x0=np.array([1.0]+[0.0 for x in
                                                 range(labels.shape[1])]),
                              bounds=[(0,None)]+[(None,None) for x in
                                                 range(labels.shape[1])],
                              jac=True,
                              method='L-BFGS-B',
                              tol=1e-07,
                              **lbfgs_kwargs)
    if (verbose):
        print("Optimization Result:")
        print(optimization_result)
    assert optimization_result.success==True, optimization_result
    optimal_t = optimization_result.x[0]
    biases = np.array(optimization_result.x[1:])
    final_nll = compute_nll(labels=labels, preacts=preacts,
                            t=optimal_t, bs=biases)
    if (verbose):
        print("Final NLL & grad is: ",final_nll)

    return (optimal_t, biases)


def do_tempscale_optimization(labels, preacts, bias_positions, verbose,
                              lbfgs_kwargs):  
    if (bias_positions=='all'):
        bias_positions = np.arange(labels.shape[1])
    def eval_func(x):
        t = x[0]
        bs = np.zeros(labels.shape[1])
        for bias_pos_idx, bias_pos in enumerate(bias_positions):
            bs[bias_pos] = x[1+bias_pos_idx] 
        #tsb = temp_scaled_biased
        tsb_preacts = preacts/float(t) + bs[None,:]
        log_sum_exp = scipy.special.logsumexp(a=tsb_preacts, axis=1) 

        exp_tsb_logits = np.exp(tsb_preacts)
        sum_exp = np.sum(exp_tsb_logits, axis=1)
        sum_preact_times_exp = np.sum(preacts*exp_tsb_logits, axis=1)

        notsb_logits_trueclass =\
            np.sum(preacts*labels, axis=1)
        tsb_logits_trueclass =\
            np.sum(tsb_preacts*labels, axis=1)
        
        log_likelihoods = tsb_logits_trueclass - log_sum_exp
        nll = -np.mean(log_likelihoods)
        grads_t = ((sum_preact_times_exp/sum_exp
                    - notsb_logits_trueclass)/\
                    (float(t)**2))
        grads_b = labels - (exp_tsb_logits/(sum_exp[:,None]))
        #multiply by -1 because we care about *negative* log likelihood
        mean_grad_t = -np.mean(grads_t)
        mean_grads_b = -np.mean(grads_b, axis=0)
        #only supply the gradients for the bias positions that
        # we are allowed to optimize for
        mean_grads_b_masked = []
        for bias_pos_idx, bias_pos in enumerate(bias_positions):
            mean_grads_b_masked.append(mean_grads_b[bias_pos])
        return nll, np.array([mean_grad_t]+mean_grads_b_masked)

    if (verbose):
        original_nll = compute_nll(labels=labels, preacts=preacts,
                                   t=1.0, bs=np.zeros(labels.shape[1]))
        print("Original NLL is: ",original_nll)
        
    optimization_result = scipy.optimize.minimize(
                              fun=eval_func,
                              #fun=lambda x: eval_func(x)[0],
                              x0=np.array([1.0]+[0.0 for x in
                                                 bias_positions]),
                              bounds=[(0,None)]+[(None,None) for x in
                                                 bias_positions],
                              jac=True,
                              method='L-BFGS-B',
                              tol=1e-07,
                              **lbfgs_kwargs)
    if (verbose):
        print(optimization_result)
    assert optimization_result.success==True, optimization_result
    biases = np.zeros(labels.shape[1])
    if (hasattr(optimization_result.x, '__iter__')):
        optimal_t = optimization_result.x[0]
        for bias_pos_idx,bias_pos in enumerate(bias_positions):
           biases[bias_pos] = optimization_result.x[1+bias_pos_idx] 
        final_nll = compute_nll(labels=labels, preacts=preacts,
                                t=optimal_t, bs=biases)
    else:
        optimal_t = optimization_result.x
        final_nll = compute_nll(labels=labels, preacts=preacts,
                                t=optimal_t, bs=np.zeros(labels.shape[1]))
    if (verbose):
        print("Final NLL & grad is: ",final_nll)

    return (optimal_t, biases)


class NoBiasVectorScaling(CalibratorFactory):

    def __init__(self, lbfgs_kwargs={}, verbose=False):
        self.lbfgs_kwargs = lbfgs_kwargs
        self.verbose = verbose

    def _get_optimal_ws_and_biases(self, preacts, labels):
         
        def eval_func(x):
            ws = np.array(x)

            vs_logits = preacts*ws[None,:]
            log_sum_exp = scipy.special.logsumexp(a=vs_logits, axis=1) 
            exp_vs_logits = np.exp(vs_logits)
            sum_exp = np.sum(exp_vs_logits, axis=1)

            log_likelihoods = (np.sum(vs_logits*labels,axis=1)
                               - log_sum_exp)
            nll = -np.mean(log_likelihoods)

            grads_ws = preacts*(labels - (exp_vs_logits/sum_exp[:,None]))

            #multiply by -1 because we care about *negative* log likelihood
            mean_grads_ws = -np.mean(grads_ws, axis=0)

            return nll, mean_grads_ws

        if (self.verbose):
            original_nll = compute_nll(labels=labels, preacts=preacts,
                                       t=1.0, bs=np.zeros(labels.shape[1]))
            print("Original NLL is: ",original_nll)
            
        optimization_result = scipy.optimize.minimize(
                      fun=eval_func,
                      #fun=lambda x: eval_func(x)[0],
                      x0=np.array([1.0 for x in range(preacts.shape[1])]),
                      bounds=[(0,None) for x in range(preacts.shape[1])],
                      jac=True,
                      method='L-BFGS-B',
                      tol=1e-07,
                      **self.lbfgs_kwargs)
        if (self.verbose):
            print(optimization_result)
        assert optimization_result.success==True, optimization_result
        
        ws = optimization_result.x 
        return ws
        
    def __call__(self, valid_preacts, valid_labels,
                       posterior_supplied=False):
        if (posterior_supplied):
            valid_preacts = inverse_softmax(valid_preacts)  
        assert np.max(np.sum(valid_labels,axis=1)==1.0)
        
        ws = self._get_optimal_ws_and_biases(preacts=valid_preacts,
                                             labels=valid_labels)
        return (lambda preact: vector_scaled_softmax(
                                    preact=(inverse_softmax(preact)
                                            if posterior_supplied else
                                            preact),
                                    ws=ws,
                                    biases=np.zeros(len(ws))))


class VectorScaling(CalibratorFactory):

    def __init__(self, lbfgs_kwargs={}, verbose=False):
        self.lbfgs_kwargs = lbfgs_kwargs
        self.verbose = verbose

    def _get_optimal_ws_and_biases(self, preacts, labels):
         
        def eval_func(x):
            ws = np.array(x[:int(len(x)/2)])
            bs = np.array(x[int(len(x)/2):]) 

            vs_logits = preacts*ws[None,:] + bs[None,:]
            log_sum_exp = scipy.special.logsumexp(a=vs_logits, axis=1) 
            exp_vs_logits = np.exp(vs_logits)
            sum_exp = np.sum(exp_vs_logits, axis=1)

            log_likelihoods = (np.sum(vs_logits*labels,axis=1)
                               - log_sum_exp)
            nll = -np.mean(log_likelihoods)

            grads_ws = preacts*(labels - (exp_vs_logits/sum_exp[:,None]))
            grads_b = labels - (exp_vs_logits/sum_exp[:,None])

            #multiply by -1 because we care about *negative* log likelihood
            mean_grads_ws = -np.mean(grads_ws, axis=0)
            mean_grads_b = -np.mean(grads_b, axis=0) 

            return nll, np.array(list(mean_grads_ws)+list(mean_grads_b))

        if (self.verbose):
            original_nll = compute_nll(labels=labels, preacts=preacts,
                                       t=1.0, bs=np.zeros(labels.shape[1]))
            print("Original NLL is: ",original_nll)
            
        optimization_result = scipy.optimize.minimize(
                      fun=eval_func,
                      #fun=lambda x: eval_func(x)[0],
                      x0=np.array([1.0 for x in range(preacts.shape[1])]
                                  +[0.0 for x in range(preacts.shape[1])]),
                      bounds=[(0,None) for x in range(preacts.shape[1])]
                              +[(None,None) for x in range(preacts.shape[1])],
                      jac=True,
                      method='L-BFGS-B',
                      tol=1e-07,
                      **self.lbfgs_kwargs)
        if (self.verbose):
            print(optimization_result)
        assert optimization_result.success==True, optimization_result
        
        ws = optimization_result.x[:preacts.shape[1]] 
        bs = optimization_result.x[preacts.shape[1]:]
        return ws, bs
        
    def __call__(self, valid_preacts, valid_labels,
                       posterior_supplied=False):
        if (posterior_supplied):
            valid_preacts = inverse_softmax(valid_preacts)  
        assert np.max(np.sum(valid_labels,axis=1)==1.0)
        
        (ws, biases) = self._get_optimal_ws_and_biases(
                                    preacts=valid_preacts,
                                    labels=valid_labels)

        return (lambda preact: vector_scaled_softmax(
                                    preact=(inverse_softmax(preact)
                                            if posterior_supplied else
                                            preact),
                                    ws=ws,
                                    biases=biases))


class TempScaling(CalibratorFactory):

    def __init__(self, ece_bins=15, lbfgs_kwargs={}, verbose=False,
                       bias_positions=[]):
        self.lbfgs_kwargs = lbfgs_kwargs
        self.verbose = verbose
        self.ece_bins = ece_bins
        #the subset of bias positions that we are allowed to optimize for
        self.bias_positions = bias_positions

    def _get_optimal_t_and_biases(self, valid_preacts, valid_labels):
        (optimal_t, biases) = do_tempscale_optimization(
            labels=valid_labels,
            preacts=valid_preacts,
            bias_positions=self.bias_positions,
            verbose=self.verbose,
            lbfgs_kwargs=self.lbfgs_kwargs)
        return (optimal_t, biases)

    def __call__(self, valid_preacts, valid_labels, posterior_supplied=False):

        if (posterior_supplied):
            valid_preacts = inverse_softmax(valid_preacts)
        assert np.max(np.sum(valid_labels,axis=1)==1.0)

        (optimal_t, biases) = self._get_optimal_t_and_biases(
            valid_preacts=valid_preacts,
            valid_labels=valid_labels)

        return (lambda preact: softmax(preact=(inverse_softmax(preact)
                                               if posterior_supplied else
                                               preact),
                                       temp=optimal_t,
                                       biases=biases))


class CrossValidatedBCTS(TempScaling):

    def __init__(self, num_crossvalidation_splits=10,
                       #frac_to_split_with=0.5,
                       #seed=1234,
                       betas_to_try=[0.0, 1e-7, 1e-6, 1e-5,
                                     1e-4, 1e-3, 1e-2, 1e-1], 
                       lbfgs_kwargs={},
                       verbose=False, max_num_bias=None):
        self.num_crossvalidation_splits = num_crossvalidation_splits
        #self.frac_to_split_with = frac_to_split_with
        #self.rng = np.random.RandomState(seed)
        self.betas_to_try = betas_to_try
        self.lbfgs_kwargs = lbfgs_kwargs
        self.verbose = verbose
        self.max_num_bias = max_num_bias

    def _get_optimal_t_and_biases(self, valid_preacts, valid_labels):
        
        heldout_biasdiffs_at_different_betas = []
        for split_num in range(self.num_crossvalidation_splits):
            #get the CV split
            training_preacts = []
            training_labels = []
            cv_heldout_preacts = []
            cv_heldout_labels = [] 
            for idx in range(len(valid_preacts)):
                if ((idx%self.num_crossvalidation_splits)==split_num):
                    cv_heldout_preacts.append(valid_preacts[idx]) 
                    cv_heldout_labels.append(valid_labels[idx])
                else:
                    training_preacts.append(valid_preacts[idx]) 
                    training_labels.append(valid_labels[idx])
            training_preacts = np.array(training_preacts) 
            training_labels = np.array(training_labels)
            cv_heldout_preacts = np.array(cv_heldout_preacts)
            cv_heldout_labels = np.array(cv_heldout_labels)

            thissplit_heldout_biasdiff_at_different_betas = []
            for beta in self.betas_to_try:
                (_t, _biases) = do_regularized_tempscale_optimization(
                    labels=training_labels,
                    preacts=training_preacts,
                    beta=beta,
                    verbose=False,
                    lbfgs_kwargs=self.lbfgs_kwargs) 
                heldout_postsoftmax_preds = softmax(
                    preact=cv_heldout_preacts, temp=_t, biases=_biases)
                thissplit_heldout_biasdiff_at_different_betas.append(
                    scipy.spatial.distance.jensenshannon(
                     p=np.mean(heldout_postsoftmax_preds, axis=0),
                     q=np.mean(cv_heldout_labels, axis=0)))
            heldout_biasdiffs_at_different_betas.append(
                thissplit_heldout_biasdiff_at_different_betas)

        avgacrosssplits_heldout_biasdiffs_at_different_betas = (
            np.mean(np.array(heldout_biasdiffs_at_different_betas), axis=0)) 

        if (self.verbose):
            print("Avg heldout biasdiff history",
                  avgacrosssplits_heldout_biasdiffs_at_different_betas)
        
        best_beta = self.betas_to_try[np.argmin(
            avgacrosssplits_heldout_biasdiffs_at_different_betas)]
        if (self.verbose):
            print("Best beta", best_beta)

        (optimal_t, biases) = do_regularized_tempscale_optimization(
            labels=valid_labels,
            preacts=valid_preacts,
            beta=best_beta,
            verbose=False,
            lbfgs_kwargs=self.lbfgs_kwargs) 

        return (optimal_t, biases)


def increase_num_bias_terms_and_fit_sequentially(
        preacts, labels, total_num_biases,
        verbose, lbfgs_kwargs, heldout_preacts=None, heldout_labels=None): 

    if (total_num_biases is None):
        total_num_biases = preacts.shape[1]

    if (heldout_preacts is not None):
        assert heldout_labels is not None
        heldout_nll_history = []
        heldout_biasdiff_history = []
    else:
        heldout_nll_history = None
        heldout_biasdiff_history = None

    biasdiff_history = []
    bias_positions = []
    for num_biases in range(total_num_biases+1):
        if (verbose):
            print("On bias #",num_biases,"bias positions",bias_positions)
        (optimal_t, biases) = do_tempscale_optimization(
            labels=labels,
            preacts=preacts,
            bias_positions=bias_positions,
            verbose=False,
            lbfgs_kwargs=lbfgs_kwargs)
        if (heldout_preacts is not None):
            heldout_nll = compute_nll(labels=heldout_labels,
                                      preacts=heldout_preacts,
                                      t=optimal_t, bs=biases)
            heldout_nll_history.append(heldout_nll)
            heldout_postsoftmax_preds = softmax(preact=heldout_preacts,
                                                temp=optimal_t, biases=biases)
            heldout_biasdiff_history.append(
                np.max(np.abs(np.mean(heldout_postsoftmax_preds, axis=0)
                              -np.mean(heldout_labels, axis=0))))
        if (num_biases < total_num_biases):
            #determine which position has the biggest remaining bias from
            # *training* set, add that position to bias_positions
            # for the next round
            postsoftmax_preds = softmax(preact=preacts,
                                        temp=optimal_t, biases=biases)
            abs_bias_diff = np.abs(np.mean(postsoftmax_preds, axis=0)
                                   -np.mean(labels, axis=0))
            max_abs_bias_diff = np.max(abs_bias_diff)
            biasdiff_history.append(max_abs_bias_diff)
            #of the positions that have not been bias-corrected, figure out
            #which one has the largest bias; include that in bias_positions
            biaspos_and_biasdiff = [x for x in enumerate(abs_bias_diff) if
                                    x[0] not in bias_positions]
            next_bias_pos,_ = max(biaspos_and_biasdiff, key=lambda x: x[1]) 
            bias_positions.append(next_bias_pos) 

    return (optimal_t, biases, bias_positions,
            biasdiff_history, heldout_nll_history, heldout_biasdiff_history)
        

class Expit(CalibratorFactory):

    def __call__(self, valid_preacts=None, valid_labels=None):
        def func(preact):
            return expit(preact)
        return func
        

class Softmax(CalibratorFactory):

    def __call__(self, valid_preacts=None, valid_labels=None):
        def func(preact):
            return softmax(preact, temp=1.0, biases=None)
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
