import numpy as np
from .hetero_likelihoods import GaussianHeteroNoise, Gaussian
from gpflow.param import Param
from gpflow import transforms
from gpflow.model import Model
from gpflow.mean_functions import Zero
import tensorflow as tf
from gpflow.param import AutoFlow, DataHolder
from gpflow._settings import settings
float_type = settings.dtypes.float_type


class GPModelAdaptiveNoiseLengthscaleMultDim(Model):
    """
    A base class for adaptive GP (non-stationary lengthscale and signal variance)
    regression models with heteroscedastic noise,
    wherein, noise is represented by a latent GP N(.)
    """
    def __init__(self, X, Y, kern, nonstat, noisekern, name='adaptive_noise_lengthscale_gp_multdim'):
        Model.__init__(self, name)
        self.kern_type = kern
        self.nonstat = nonstat
        self.noisekern = noisekern
        self.likelihood = GaussianHeteroNoise()
        if isinstance(X, np.ndarray):
            #: X is a data matrix; each row represents one instance
            X = DataHolder(X)
        if isinstance(Y, np.ndarray):
            #: Y is a data matrix; rows correspond to the rows in X, columns are treated independently
            Y = DataHolder(Y)
        self.likelihood._check_targets(Y.value)
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
    def predict(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict_f(Xnew)

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def pred_cov(self, X1, X2):
        """
        Compute the posterior covariance matrix b/w X1 and X2.
        """
        return self.build_pred_cov_f(X1, X2)

    @AutoFlow((float_type, [None, None]), (tf.int32, []))
    def posterior_samples_n(self, Xnew, num_samples):
        """
        Produce samples from the posterior latent function(s) at the points
        Xnew.
        """
        mu, var = self.build_predict_f(Xnew, full_cov=True)
        jitter = tf.eye(tf.shape(mu)[0], dtype=float_type) * settings.numerics.jitter_level
        mu_n, var_n = self.build_predict_n(Xnew)
        mu_n = tf.square(tf.exp(mu_n))
        A = var[:, :] + jitter
        B = tf.multiply(mu_n, tf.eye(tf.shape(mu_n)[0], dtype=float_type))
        L = tf.cholesky(A + B)
        shape = tf.stack([tf.shape(L)[0], num_samples])
        V = tf.random_normal(shape, dtype=settings.dtypes.float_type)
        samples = mu[:, ] + tf.matmul(L, V)
        return tf.transpose(samples)
    
    @AutoFlow((float_type, [None, None]), (tf.int32, []))
    def posterior_samples(self, Xnew, num_samples):
        """
        Produce samples from the posterior latent function(s) at the points
        Xnew.
        """
        mu, var = self.build_predict_f(Xnew, full_cov=True)
        jitter = tf.eye(tf.shape(mu)[0], dtype=float_type) * settings.numerics.jitter_level
        L = tf.cholesky(var[:, :] + jitter)
        shape = tf.stack([tf.shape(L)[0], num_samples])
        V = tf.random_normal(shape, dtype=settings.dtypes.float_type)
        samples = mu[:, ] + tf.matmul(L, V)
        return tf.transpose(samples)
    
    
class GPModelAdaptiveLengthscaleMultDim(Model):
    """
    A base class for adaptive GP (non-stationary lengthscale and signal variance)
    regression models with heteroscedastic noise,
    wherein, noise is represented by a latent GP N(.)
    """
    def __init__(self, X, Y, kern, nonstat, name='adaptive_lengthscale_gp_multdim'):
        Model.__init__(self, name)
        self.kern_type = kern
        self.nonstat = nonstat
        self.likelihood = Gaussian()
        if isinstance(X, np.ndarray):
            #: X is a data matrix; each row represents one instance
            X = DataHolder(X)
        if isinstance(Y, np.ndarray):
            #: Y is a data matrix; rows correspond to the rows in X, columns are treated independently
            Y = DataHolder(Y)
        self.likelihood._check_targets(Y.value)
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
    def predict(self, Xnew):
        """
        Compute the mean and variance of the latent function(s)
        at the points `Xnew`.
        """
        return self.build_predict_f(Xnew)

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def pred_cov(self, X1, X2):
        """
        Compute the posterior covariance matrix b/w X1 and X2.
        """
        return self.build_pred_cov_f(X1, X2)
    
    @AutoFlow((float_type, [None, None]), (tf.int32, []))
    def posterior_samples(self, Xnew, num_samples):
        """
        Produce samples from the posterior latent function(s) at the points
        Xnew.
        """
        mu, var = self.build_predict_f(Xnew, full_cov=True)
        jitter = tf.eye(tf.shape(mu)[0], dtype=float_type) * settings.numerics.jitter_level
        L = tf.cholesky(var[:, :] + jitter)
        shape = tf.stack([tf.shape(L)[0], num_samples])
        V = tf.random_normal(shape, dtype=settings.dtypes.float_type)
        samples = mu[:, ] + tf.matmul(L, V)
        return tf.transpose(samples)