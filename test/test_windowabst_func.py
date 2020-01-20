from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
from unittest import skip
import sys
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from abstention.abstention import (MonteCarloWindowAbstDeltaAuroc,
                                   MonteCarloWindowAbstDeltaAuprc,
                                   assertsorted_average_precision_score,
                                   roc_auc_score)
import scipy.stats.mstats

class TestWindowAbst(unittest.TestCase):

    def setUp(self):
        pass

    def calculate_metric_deltas(self, metric_delta_func,
                                           window_size,
                                           metric_eval_func):
        frac_pos = 0.3
        total_num = 10000
        window_size = 100
        np.random.seed(1234)
        scores = np.random.rand(total_num) #random scores
        scores = np.array(sorted(scores))
        labels = np.zeros(total_num)
        for i in range(total_num):
            labels[i] = 1.0 if (np.random.random() < frac_pos) else 0.0
        preabst_metric = metric_eval_func(y_true=labels, y_score=scores)
        metric_deltas = metric_delta_func(labels=labels,
                                          window_size=window_size) 

        max_diff = 0
        for idx, metric_delta in enumerate(metric_deltas):
            retained_labels = np.array(
                list(labels[:idx])+list(labels[idx+window_size:]))
            retained_scores = np.array(
                list(scores[:idx])+list(scores[idx+window_size:]))
            manual_metric_delta = (metric_eval_func(y_true=retained_labels,
                                             y_score=retained_scores)
                                   - preabst_metric) 
            max_diff = max(max_diff, abs(metric_delta-manual_metric_delta))
            assert max_diff < 1e-15, (max_diff, idx,
                                      metric_delta, manual_metric_delta)
        print(max_diff)

    def test_marginal_auroc(self): 
        window_size=100
        self.calculate_metric_deltas(
            metric_delta_func=MonteCarloWindowAbstDeltaAuroc(
               num_to_abstain_on=window_size, 
               return_max_across_windows=False,
               n_samples=None,
               smoothing_window_size=None).calculate_metric_deltas,
            window_size=window_size,
            metric_eval_func=roc_auc_score)

    
    def test_marginal_auprc(self): 
        window_size=100
        self.calculate_metric_deltas(
            metric_delta_func=MonteCarloWindowAbstDeltaAuprc(
               num_to_abstain_on=window_size, 
               return_max_across_windows=False,
               n_samples=None,
               smoothing_window_size=None).calculate_metric_deltas,
            window_size=window_size,
            metric_eval_func=assertsorted_average_precision_score)
