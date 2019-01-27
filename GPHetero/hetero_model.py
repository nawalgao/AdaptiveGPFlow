import numpy as np
from .hetero_likelihoods import GaussianHeteroNoise, GaussianMod, Gaussian
from gpflow.param import Param
from gpflow import transforms
from gpflow.model import Model
from gpflow.mean_functions import Zero
import tensorflow as tf
from gpflow.param import AutoFlow, DataHolder
from gpflow._settings import settings
float_type = settings.dtypes.float_type


class GPModel(Model):
    """
    A base class for Gaussian process models, that is, those of the form

    .. math::
       :nowrap:

       \\begin{align}
       \\theta & \sim p(\\theta) \\\\
       f       & \sim \\mathcal{GP}(m(x), k(x, x'; \\theta)) \\\\
       f_i       & = f(x_i) \\\\
       y_i\,|\,f_i     & \sim p(y_i|f_i)
       \\end{align}

    This class mostly adds functionality to compile predictions. To use it,
    inheriting classes must define a build_predict function, which computes
    the means and variances of the latent function. This gets compiled
    similarly to build_likelihood in the Model class.

    These predictions are then pushed through the likelihood to obtain means
    and variances of held out data, self.predict_y.

    The predictions can also be used to compute the (log) density of held-out
    data via self.predict_density.

    For handling another data (Xnew, Ynew), set the new value to self.X and self.Y

    >>> m.X = Xnew
    >>> m.Y = Ynew
    """

    def __init__(self, X, Y, kern, likelihood, mean_function, name='model'):
        Model.__init__(self, name)
        self.mean_function = mean_function or Zero()
        self.kern, self.likelihood = kern, likelihood

        if isinstance(X, np.ndarray):
            #: X is a data matrix; each row represents one instance
            X = DataHolder(X)
        if isinstance(Y, np.ndarray):
            #: Y is a data matrix, rows correspond to the rows in X, columns are treated independently
            Y = DataHolder(Y)

        likelihood._check_targets(Y.value)
        self.X, self.Y = X, Y
        self._session = None

    def build_predict(self, *args, **kwargs):
        raise NotImplementedError

    @AutoFlow((float_type, [None, None]))
    def predict_f(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict(Xnew)

    @AutoFlow((float_type, [None, None]))
    def predict_f_full_cov(self, Xnew):
        """
        Compute the mean and covariance matrix of the latent function(s) at the
        points Xnew.
        """
        return self.build_predict(Xnew, full_cov=True)

    @AutoFlow((float_type, [None, None]), (tf.int32, []))
    def predict_f_samples(self, Xnew, num_samples):
        """
        Produce samples from the posterior latent function(s) at the points
        Xnew.
        """
        mu, var = self.build_predict(Xnew, full_cov=True)
        jitter = tf.eye(tf.shape(mu)[0], dtype=float_type) * settings.numerics.jitter_level
        samples = []
        for i in range(self.num_latent):
            L = tf.cholesky(var[:, :, i] + jitter)
            shape = tf.stack([tf.shape(L)[0], num_samples])
            V = tf.random_normal(shape, dtype=settings.dtypes.float_type)
            samples.append(mu[:, i:i + 1] + tf.matmul(L, V))
        return tf.transpose(tf.stack(samples))

    @AutoFlow((float_type, [None, None]))
    def predict_y(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        pred_f_mean, pred_f_var = self.build_predict(Xnew)
        return self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def predict_density(self, Xnew, Ynew):
        """
        Compute the (log) density of the data Ynew at the points Xnew

        Note that this computes the log density of the data individually,
        ignoring correlations between them. The result is a matrix the same
        shape as Ynew containing the log densities.
        """
        pred_f_mean, pred_f_var = self.build_predict(Xnew)
        return self.likelihood.predict_density(pred_f_mean, pred_f_var, Ynew)


class GPModelHeteroNoiseRegression(Model):
    """
    A base class for Gaussian process regression models with heteroscedastic noise,
    wherein, noise is represented by a latent GP N(.)
    """
    def __init__(self, X, Y, kern1, kern2, mean_function, name='hetero_noise_regression_model'):
        Model.__init__(self, name)
        self.mean_function = mean_function or Zero()
        self.kern1, self.kern2 = kern1, kern2
        self.likelihood = GaussianHeteroNoise()
        
        if isinstance(X, np.ndarray):
            #: X is a data matrix; each row represents one instance
            X = DataHolder(X)
        if isinstance(Y, np.ndarray):
            #: Y is a data matrix, rows correspond to the rows in X, columns are treated independently
            Y = DataHolder(Y)
        
        self.X, self.Y = X, Y
        self._session = None
    
    @AutoFlow((float_type, [None, None]))
    def predict_f(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict_f(Xnew)
    
    @AutoFlow((float_type, [None, None]))
    def predict_n(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict_n(Xnew)
 
    
class GPModelAdaptive(Model):
    """
    A base class for adaptive GP (non-stationary lengthscale and signal variance)
    regression models with heteroscedastic noise,
    wherein, noise is represented by a latent GP N(.)
    """
    def __init__(self, X, Y, kern1, kern2, kern3, name='adaptive_gp'):
        Model.__init__(self, name)
        self.kern1, self.kern2, self.kern3 = kern1, kern2, kern3
        self.likelihood = GaussianHeteroNoise()
        
        if isinstance(X, np.ndarray):
            #: X is a data matrix; each row represents one instance
            X = DataHolder(X)
        if isinstance(Y, np.ndarray):
            #: Y is a data matrix, rows correspond to the rows in X, columns are treated independently
            Y = DataHolder(Y)
        
        self.X, self.Y = X, Y
        self._session = None
    
    @AutoFlow((float_type, [None, None]))
    def predict_f(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict_f(Xnew)
    
    @AutoFlow((float_type, [None, None]))
    def predict_n(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict_n(Xnew)


class GPModelAdaptiveLengthscale(Model):
    """
    A base class for adaptive GP (non-stationary lengthscale and signal variance)
    regression models with heteroscedastic noise,
    wherein, noise is represented by a latent GP N(.)
    """
    def __init__(self, X, Y, kern1, nonstat, name='adaptive_lengthscale_gp'):
        Model.__init__(self, name)
        self.kern1 = kern1
        self.nonstat = nonstat
        self.likelihood = Gaussian()
        
        if isinstance(X, np.ndarray):
            #: X is a data matrix; each row represents one instance
            X = DataHolder(X)
        if isinstance(Y, np.ndarray):
            #: Y is a data matrix, rows correspond to the rows in X, columns are treated independently
            Y = DataHolder(Y)
        
        self.X, self.Y = X, Y
        self._session = None
    
    @AutoFlow((float_type, [None, None]))
    def predict_l(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict_l(Xnew)
    
    @AutoFlow((float_type, [None, None]))
    def predict_f(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict_f(Xnew)


class GPModelAdaptLAdaptN(Model):
    """
    A base class for adaptive GP (non-stationary lengthscale and signal variance)
    regression models with heteroscedastic noise,
    wherein, noise is represented by a latent GP N(.)
    """
    def __init__(self, X, Y, kern1, kern2, nonstat, name='adapt_ll_noise_gps'):
        Model.__init__(self, name)
        self.signal_variance = Param(1.0, transforms.positive)
        self.kern1, self.kern2 = kern1, kern2
        self.nonstat = nonstat
        self.likelihood = GaussianHeteroNoise()
        
        if isinstance(X, np.ndarray):
            #: X is a data matrix; each row represents one instance
            X = DataHolder(X)
        if isinstance(Y, np.ndarray):
            #: Y is a data matrix, rows correspond to the rows in X, columns are treated independently
            Y = DataHolder(Y)
        
        self.X, self.Y = X, Y
        self._session = None
    
    @AutoFlow((float_type, [None, None]))
    def predict_l(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict_l(Xnew)
    
    @AutoFlow((float_type, [None, None]))
    def predict_n(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict_n(Xnew)
    
    @AutoFlow((float_type, [None, None]))
    def predict_f(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict_f(Xnew)


        
        
    