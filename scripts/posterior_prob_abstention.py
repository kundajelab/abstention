from __future__ import division, print_function, absolute_import
import argparse
import abstention as ab


def posterior_prob_abstention(options):
    


if __name__=="__main__":
    import argparse;
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_logits", required=True)
    parser.add_argument("--valid_logits_and_labels", required=True)
    parser.add_argument("--calibrator", default="PlattScaling")
    parser.add_argument("--abstention_method", default="marginal_auroc")
    options = parser.parse_args()
    posterior_prob_abstention(options)
