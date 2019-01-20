from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
from unittest import skip
import sys
import os
import numpy as np
from abstention.abstention import (MarginalDeltaAuRocMixin,
                                   MarginalDeltaAuPrcMixin)
import scipy.stats.mstats

class TestMarginals(unittest.TestCase):

    def setUp(self):
        pass

    def apply_marginals_eval(self,
                                 marginal_delta_metric_mixin,
                                 total_num, frac_pos):
        np.random.seed(1234)
        scores = np.random.rand(total_num) #random scores
        scores = sorted(scores)
        labels = np.zeros(total_num)
        for i in range(total_num):
            labels[i] = 1.0 if (np.random.random() < frac_pos) else 0.0
        num_pos = np.sum(labels) 
        num_neg = total_num - num_pos
        pos_cdfs = np.cumsum(labels)/num_pos 
        neg_cdfs = np.cumsum(1-labels)/num_neg
        metric = marginal_delta_metric_mixin.compute_metric(
                                             y_true=labels, y_score=scores)
    
        abstention_scores =\
            marginal_delta_metric_mixin.compute_abstention_score(   
                est_metric=metric,
                est_numpos=num_pos, est_numneg=num_neg,
                ppos=labels, pos_cdfs=pos_cdfs, neg_cdfs=neg_cdfs)
        abstention_ordering = [x[0] for x in
                               sorted(enumerate(abstention_scores),
                                      key=lambda x: x[1])]
        empirical_scores = (np.array(
            [marginal_delta_metric_mixin.compute_metric(
             y_true=np.concatenate([labels[:i],labels[i+1:]]),
             y_score=np.concatenate([scores[:i],scores[i+1:]]))
             for i in range(total_num)])-metric)
        empirical_ordering = [x[0] for x in sorted(enumerate(empirical_scores),
                                                   key=lambda x: x[1])]
        print([x for x in zip(labels, abstention_scores, empirical_scores)
               if (np.abs(x[1]-x[2]) > 1E-7)])
        spearman_corr = scipy.stats.mstats.spearmanr(abstention_scores[:],
                                                     empirical_scores[:])
        print(np.max(np.abs(np.array(abstention_scores)-
                                 np.array(empirical_scores))))
        assert (np.max(np.abs(np.array(abstention_scores)-
                                  np.array(empirical_scores))) < 1E-7) 
        print(spearman_corr)
        assert spearman_corr.correlation > 0.999

    def test_marginal_auroc(self): 
        self.apply_marginals_eval(
            marginal_delta_metric_mixin=MarginalDeltaAuRocMixin(),
            total_num=100, frac_pos=0.1)

    def test_marginal_auprc(self): 
        self.apply_marginals_eval(
            marginal_delta_metric_mixin=MarginalDeltaAuPrcMixin(),
            total_num=1000, frac_pos=0.1)
