# AdaptiveGPFlow
Non stationary Gaussian Process (with Heterscedastic Noise)

**WARNING: This package is still under construction (with limited functionality) and will be updated in near future.**

AdaptiveGPFlow is a python package (released under [MIT License](https://github.com/nawalgao/GPPrefElicit/blob/master/LICENSE)) for modeling non-stationary Gaussian Processes (with Heterscedastic noise)
using [GPflow](https://github.com/GPflow/GPflow), and uses [TensorFlow](http://www.tensorflow.org).

We present a package for implementing fully non-stationary Gaussian process regression (GPR), where all three key parameters – noise variance, signal variance and lengthscale – can be simultaneously input-dependent [paper](https://arxiv.org/pdf/1508.04319.pdf).


It is currently maintained by [Nimish Awalgaonkar and Piyush Pandita](https://www.predictivesciencelab.org/people.html).

# Install
This package was written in `Python 2.7.14`. It is recommended to use the `Anaconda >=5.0.1` distribution, on a empty environment. The package is built on top of `gpflow 0.4.0` which has to be installed from [source]( https://github.com/GPflow/GPflow/releases/tag/0.4.0).
Then, proceed on installing the following
```
conda install numpy scipy matplotlib 
```

# Usage Example
Please check out the notebooks folder for working examples.
# Contributing
If you are interested in contributing to this open source project, contact us through an issue on this repository.

# Citing AdaptiveGPFlow

To cite AdaptiveGPFlow, please reference **paper is currently under review**.
