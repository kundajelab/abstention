# Abstention, Calibration & Label Shift

Algorithms for abstention, calibration and domain adaptation under label shift. 

Associated papers:

Shrikumar A\*&dagger;, Alexandari A\*, Kundaje A&dagger;, [A Flexible and Adaptive Framework for Abstention Under Class Imbalance](https://arxiv.org/abs/1802.07024)

Alexandari A\*, Kundaje A&dagger;, Shrikumar A\*&dagger;, [Adapting to Label Shift with Bias-Corrected Calibration](https://arxiv.org/abs/1901.06852)

*co-first authors
&dagger; co-corresponding authors

## Examples

See [https://github.com/blindauth/abstention_experiments](https://github.com/blindauth/abstention_experiments) and [https://github.com/blindauth/labelshiftexperiments](https://github.com/blindauth/labelshiftexperiments) for colab notebooks reproducing the experiments in the papers. 

## Installation

```
pip install abstention
```

## Algorithms implemented

For calibration:
- Platt Scaling
- Isotonic Regression
- Temperature Scaling
- Vector Scaling
- Bias-Corrected Temperature Scaling
- No-Bias Vector Scaling

For domain adaptation to label shift:
- Expectation Maximization (Saerens et al., 2002)
- Black-Box Shift Learning (BBSL) (Lipton et al., 2018)
- Regularized Learning under Label Shifts (RLLS) (Azizzadenesheli et al., 2019)

For abstention:
- Metric-specific abstention methods described in [A Flexible and Adaptive Framework for Abstention Under Class Imbalance](https://arxiv.org/abs/1802.07024), including abstention to optimize auROC, auPRC, sensitivity at a target specificity and weighted Cohen's Kappa
- Jensen-Shannon Divergence from class priors
- Entropy in the predicted class probabilities (Wan, 1990)
- Probability of the highest-predicted class (Hendrycks \& Gimpel, 2016)
- The method of Fumera et al., 2000
- See Colab notebook experiments in [https://github.com/blindauth/abstention_experiments](https://github.com/blindauth/abstention_experiments) for details on how to use the various abstention methods.

## Contact

If you have any questions, please contact:

Avanti Shrikumar: avanti [dot] shrikumar [at] gmail.com

Amr Alexandari: amr [dot] alexandari [at] gmail.com

Anshul Kundaje: akundaje [at] stanford [dot] edu

