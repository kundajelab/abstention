from __future__ import division, print_function, absolute_import
import numpy as np
from collections import OrderedDict


def get_preact_func(model, task_idx):
    from keras import backend as K
    preact_func = K.function([model.layers[0].input, K.learning_phase()],
                                   [model.layers[-2].output])
    def batched_func(data, learning_phase, batch_size):
        to_return = []
        start_idx = 0
        while start_idx < len(data):
            to_return.extend(
                preact_func([data[start_idx:start_idx+batch_size],
                            learning_phase])[0][:, task_idx])
            start_idx += batch_size
        return np.array(to_return)
    return batched_func


def get_embed_func(model, task_idx):
    from keras import backend as K
    embed_func = K.function([model.layers[0].input, K.learning_phase()],
                                   [model.layers[-3].output])
    def batched_func(data, learning_phase, batch_size):
        to_return = []
        start_idx = 0
        while start_idx < len(data):
            to_return.extend(
                embed_func([data[start_idx:start_idx+batch_size],
                            learning_phase])[0][:, task_idx])
            start_idx += batch_size
        return np.array(to_return)
    return batched_func


def obtain_raw_data(preact_func, data, num_dropout_runs, batch_size=50):
    print("Computing deterministic activations")
    deterministic_preacts = np.array(
        preact_func(data=data, learning_phase=0,
                    batch_size=batch_size))
    
    print("Computing nondeterministic activations")
    dropout_run_results = []
    for i in range(num_dropout_runs):
        if ((i+1)%10==0):
            print("Done",i+1,"runs")
        dropout_run_results.append(
            np.array(preact_func(data=data, learning_phase=1,
                                 batch_size=batch_size)).squeeze())
    return deterministic_preacts, np.array(dropout_run_results)


def obtain_embeddings(embed_func, data, batch_size=50):
    print("Computing embeddings")
    embeddings_results = np.array(embed_func(data=data, learning_phase=0,
                                             batch_size=batch_size)).squeeze()
    return embeddings_results


def obtain_posterior_probs_and_uncert_estimates(
    cb_method_name_to_factory,
    valid_labels,
    valid_preacts, valid_dropout_preacts,
    test_preacts, test_dropout_preacts):

    cb_method_name_to_cb_func = OrderedDict()
    for cb_method_name, cb_factory in cb_method_name_to_factory.items():
        cb_method_name_to_cb_func[cb_method_name] =\
            cb_factory(valid_preacts=valid_preacts,
                      valid_labels=valid_labels)
    
    #get all the types of posterior probabilities
    cb_method_name_to_valid_posterior_prob = OrderedDict()
    cb_method_name_to_test_posterior_prob = OrderedDict()
    for cb_method_name, cb_func in cb_method_name_to_cb_func.items():
        cb_method_name_to_valid_posterior_prob[cb_method_name] =\
            cb_func(valid_preacts)
        cb_method_name_to_test_posterior_prob[cb_method_name] =\
            cb_func(test_preacts)
    
    #get all the types of transformations to apply to the
    #preactivations for uncertainty estimation
    uncert_transform_funcs = OrderedDict()
    uncert_transform_funcs['preactivation'] = lambda x: x
    #add all the calibration methods too
    uncert_transform_funcs.update(cb_method_name_to_cb_func) 
    #apply them to validation and test set to
    #get different uncertainty estimates
    transform_name_to_valid_uncert = OrderedDict()
    transform_name_to_test_uncert = OrderedDict()
    for transform_name, transform_func in uncert_transform_funcs.items():
        transform_name_to_valid_uncert[transform_name] =\
            np.std(np.array([transform_func(x) for x in valid_dropout_preacts]), axis=0, ddof=1)
        transform_name_to_test_uncert[transform_name] =\
            np.std(np.array([transform_func(x) for x in test_dropout_preacts]), axis=0, ddof=1) 

    return (cb_method_name_to_valid_posterior_prob,
            cb_method_name_to_test_posterior_prob,
            transform_name_to_valid_uncert,
            transform_name_to_test_uncert)
