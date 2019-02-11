from __future__ import print_function, absolute_import
from functools import reduce

import tensorflow as tf
import numpy as np
import gpflow
from gpflow.param import Param, Parameterized, AutoFlow
from gpflow.kernels import Kern
from gpflow import transforms
from gpflow._settings import settings
from gpflow.quadrature import hermgauss, mvhermgauss, mvnquad

float_type = settings.dtypes.float_type
int_type = settings.dtypes.int_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64


class Stationary(Kern):
    """
    Base class for kernels that are stationary, that is, they only depend on

        r = || x - x' ||

    This class handles 'ARD' behaviour, which stands for 'Automatic Relevance
    Determination'. This means that the kernel has one lengthscale per
    dimension, otherwise the kernel is isotropic (has a single lengthscale).
    """

    def __init__(self, input_dim, variance=1.0, lengthscales=None,
                 active_dims=None, ARD=False):
        """
        - input_dim is the dimension of the input to the kernel
        - variance is the (initial) value for the variance parameter
        - lengthscales is the initial value for the lengthscales parameter
          defaults to 1.0 (ARD=False) or np.ones(input_dim) (ARD=True).
        - active_dims is a list of length input_dim which controls which
          columns of X are used.
        - ARD specifies whether the kernel has one lengthscale per dimension
          (ARD=True) or a single lengthscale (ARD=False).
        """
        Kern.__init__(self, input_dim, active_dims)
        self.scoped_keys.extend(['square_dist', 'euclid_dist'])
        self.variance = Param(variance, transforms.positive)
        if ARD:
            if lengthscales is None:
                lengthscales = np.ones(input_dim, np_float_type)
            else:
                # accepts float or array:
                lengthscales = lengthscales * np.ones(input_dim, np_float_type)
            self.lengthscales = Param(lengthscales, transforms.positive)
            self.ARD = True
        else:
            if lengthscales is None:
                lengthscales = 1.0
            self.lengthscales = Param(lengthscales, transforms.positive)
            self.ARD = False

    def square_dist(self, X, X2):
        X = X / self.lengthscales
        Xs = tf.reduce_sum(tf.square(X), 1)
        if X2 is None:
            return -2 * tf.matmul(X, X, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
        else:
            X2 = X2 / self.lengthscales
            X2s = tf.reduce_sum(tf.square(X2), 1)
            return -2 * tf.matmul(X, X2, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))

    def euclid_dist(self, X, X2):
        r2 = self.square_dist(X, X2)
        return tf.sqrt(r2 + 1e-12)

    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))

class RBF(Stationary):
    """
    The radial basis function (RBF) or squared exponential kernel
    """
    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        return self.variance * tf.exp(-self.square_dist(X, X2) / 2)
    
    @AutoFlow((float_type, [None, None]), (float_type, [None, None]),
              (float_type, [None, None]), (float_type, [None, None]))

    def compute_K(self, X1, X2):
        return self.K(X1, X2)

class NonStationaryRBF(Kern):
    """
    Non-stationary 1D RBF kernel
    For more info refer to paper:
    https://arxiv.org/abs/1508.04319
    """
    def __init__(self):
        Kern.__init__(self, input_dim = 1, active_dims= [0])
        self.signal_variance = Param(1.0, transform=transforms.positive)
        
    def K(self, X1, Lexp1, Sexp1, X2, Lexp2, Sexp2):
        """
        X1, X2 : input points
        Lexp1 and Sexp1 are exponential of latent GPs 
        L1(.) representing log of non-stationary lengthscale values at points X1 and
        S1(.) representing log of non-stationary signal variance values at points X1.
        """
        dist_sqr = tf.square(X1 - tf.transpose(X2))
        l_sqr = tf.square(Lexp1) + tf.square(tf.transpose(Lexp2))
        l_2_prod = 2 * Lexp1 * tf.transpose(Lexp2)
        var_prod = Sexp1 * tf.transpose(Sexp2)
        #var_prod = self.signal_variance
        cov = var_prod * tf.sqrt(l_2_prod / l_sqr) * tf.exp(-1 * dist_sqr / l_sqr)
        return cov
        #return dist_sqr, l_sqr, l_2_prod , var_prod, cov
    
    @AutoFlow((float_type, [None, None]), (float_type, [None, None]),
              (float_type, [None, None]), (float_type, [None, None]),
              (float_type, [None, None]), (float_type, [None, None]))
    def compute_K(self, X1, Lexp1, Sexp1, X2, Lexp2, Sexp2):
        return self.K(X1, Lexp1, Sexp1, X2, Lexp2, Sexp2)

class NonStationaryLengthscaleRBF(Kern):
    """
    Non-stationary 1D RBF kernel
    For more info refer to paper:
    https://arxiv.org/abs/1508.04319
    """
    def __init__(self):
        Kern.__init__(self, input_dim = 1, active_dims= [0])
        self.signal_variance = Param(1.0, transform=transforms.positive)
        
    def K(self, X1, Lexp1, X2, Lexp2):
        """
        X1, X2 : input points
        Lexp1 and Sexp1 are exponential of latent GPs 
        L1(.) representing log of non-stationary lengthscale values at points X1 and
        S1(.) representing log of non-stationary signal variance values at points X1.
        """
        dist_sqr = tf.square(X1 - tf.transpose(X2))
        l_sqr = tf.square(Lexp1) + tf.square(tf.transpose(Lexp2))
        l_2_prod = 2 * Lexp1 * tf.transpose(Lexp2)
        var_prod = self.signal_variance
        cov = var_prod * tf.sqrt(l_2_prod / l_sqr) * tf.exp(-1. * dist_sqr / l_sqr)
        return cov
    
    @AutoFlow((float_type, [None, None]), (float_type, [None, None]),
              (float_type, [None, None]), (float_type, [None, None]))
    def compute_K(self, X1, Lexp1, X2, Lexp2):
        return self.K(X1, Lexp1, X2, Lexp2)

class NonStatLRBFMultiD(NonStationaryLengthscaleRBF):
    """
    Non stationary covariance for multi-dimensions
    """
    def __init__(self):
        NonStationaryLengthscaleRBF.__init__(self)

    def _get_squared_distance(self, X1, X2):
        """
        Get the squared Eucledian distance between ```X1``` and ```X2``` where both are nx1 and mx1 tensors repectively. 
        """
        if X2 is None:
            X2 = X
        sqd = tf.squared_difference(tf.tile(X1, tf.constant(tf.transpose(X2).shape)), tf.tile(tf.transpose(X2), tf.constant(X1.shape)))
        return sqd

    def K(self, X1, Lexp1, X2, Lexp2):
        """
        X1, X2 : input points
        Lexp1 and Sexp1 are exponential of latent GPs 
        L1(.) representing log of non-stationary lengthscale values at points X1.
        S1(.) representing log of non-stationary signal variance values at points X1.
        """
        dist_sqr = self._get_squared_distance(X1, X2)
        l_sqr = tf.tile(tf.square(Lexp1), tf.constant(tf.transpose(X2).shape)) + tf.square(tf.transpose(Lexp2), tf.constant(x1.shape))
        l_2_prod = 2 * Lexp1 * tf.transpose(Lexp2)
        cov = tf.sqrt(l_2_prod / l_sqr) * tf.exp(-1. * dist_sqr / l_sqr)
        return cov
    
    def Kgram(self, X1, Lexp1, X2, Lexp2):
        """
        X1mat, X2mat : input feature matrix (N X D)
        Lexpmat1, Lexpmat2 : latent lengthscale matrix (N X D)
        Lexpmat1 : representing log of non-stat lengthscale values for each dimension of X1
        """
        # assert X1.shape[1] == X2.shape[1]
        num_data = X1.shape[0]
        num_feat = X1.shape[1]
        import  pdb
        pdb.set_trace()
        cov = tf.ones(shape=[X1.shape[0], X2.shape[0]])
        for i in xrange(num_feat):
            cov = tf.multiply(self.K(X1[:, i][:, None], Lexp1[:, i][:, None], X2[:, i][:, None], Lexp2[:, i][:, None]), cov)
        return self.variance * cov
     
    @AutoFlow((float_type, [None, None]), (float_type, [None, None]),
              (float_type, [None, None]), (float_type, [None, None]))

    def compute_Ka(self, X1, Lexp1, X2, Lexp2):
        return self.Kgram(X1, Lexp1, X2, Lexp2)

    def is_pos_def(x):
        return np.all(np.linalg.eigvals(x) > 0)

if __name__ == '__main__':
    import numpy as np
    A = np.arange(2, 100)[:,None]
    B = np.arange(1, 100)[:,None]
    C = np.arange(1, 100)[:,None]
    
    Cov = NonStationaryLengthscaleRBF()
    #r = Cov.compute_K(A,A,A,A)
    X = np.random.rand(3, 2)
    #K = gpflow.kernels.RBF(input_dim = 2, ARD = True)
    a = Cov.compute_K(A, A, A, A)
    import pdb
    pdb.set_trace()
    Cov = NonStatLRBFMultiD()
    #r = Cov.compute_K(A,A,A,A)
    X = np.random.rand(3, 2)
    # import pdb
    # pdb.set_trace()
    Cov.compute_Ka(X, X, X, X)
    #K = gpflow.kernels.RBF(input_dim = 2, ARD = True)
    a = Cov.compute_K(X, X, X, X)