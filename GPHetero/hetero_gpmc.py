
import numpy as np
import tensorflow as tf
from .hetero_model import GPModel,GPModelHeteroNoiseRegression, GPModelAdaptive, GPModelAdaptiveLengthscale, GPModelAdaptLAdaptN
from gpflow.param import Param, DataHolder
from .hetero_conditionals import conditional, nonstat_conditional
from .hetero_kernels import NonStationaryRBF, NonStationaryLengthscaleRBF
from gpflow.priors import Gaussian
from gpflow.mean_functions import Zero
from gpflow._settings import settings
float_type = settings.dtypes.float_type


class GPMC(GPModel):
    def __init__(self, X, Y, kern, likelihood,
                 mean_function=None, num_latent=None):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, likelihood, mean_function are appropriate GPflow objects

        This is a vanilla implementation of a GP with a non-Gaussian
        likelihood. The latent function values are represented by centered
        (whitened) variables, so

            v ~ N(0, I)
            f = Lv + m(x)

        with

            L L^T = K

        """
        X = DataHolder(X, on_shape_change='recompile')
        Y = DataHolder(Y, on_shape_change='recompile')
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.num_data = X.shape[0]
        self.num_latent = num_latent or Y.shape[1]
        self.V = Param(np.zeros((self.num_data, self.num_latent)))
        self.V.prior = Gaussian(0., 1.)

    def compile(self, session=None, graph=None, optimizer=None):
        """
        Before calling the standard compile function, check to see if the size
        of the data has changed and add parameters appropriately.

        This is necessary because the shape of the parameters depends on the
        shape of the data.
        """
        if not self.num_data == self.X.shape[0]:
            self.num_data = self.X.shape[0]
            self.V = Param(np.zeros((self.num_data, self.num_latent)))
            self.V.prior = Gaussian(0., 1.)

        return super(GPMC, self).compile(session=session,
                                         graph=graph,
                                         optimizer=optimizer)

    def build_likelihood(self):
        """
        Construct a tf function to compute the likelihood of a general GP
        model.

            \log p(Y, V | theta).

        """
        K = self.kern.K(self.X)
        L = tf.cholesky(K + tf.eye(tf.shape(self.X)[0], dtype=float_type)*settings.numerics.jitter_level)
        F = tf.matmul(L, self.V) + self.mean_function(self.X)

        return tf.reduce_sum(self.likelihood.logp(F, self.Y))

    def build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | (F=LV) )

        where F* are points on the GP at Xnew, F=LV are points on the GP at X.

        """
        mu, var = conditional(Xnew, self.X, self.kern, self.V,
                              full_cov=full_cov,
                              q_sqrt=None, whiten=True)
        return mu + self.mean_function(Xnew), var


class GPMCHeteroNoiseRegression(GPModelHeteroNoiseRegression):
    def __init__(self, X, Y, kern1, kern2,
                 mean_function=None, num_latent=None): 
        """ 
        X is a data matrix, size N x D
        Y is a data matrix, size N x 1
        kern1, kern2, likelihood, mean_function are appropriate GPflow objects
        kern1 : covariance function associated with latent GP F(.)
        kern2 : covairance function associated with latent GP N(.) representing heteroscedastic noise

        This is a vanilla implementation of a GP with a Gaussian
        likelihood and heteroscedatic noise.
        This noise is represented using a new latent GP N(.)
        The latent function values are represented by centered
        (whitened) variables, so

        v1 ~ N(0, I)
        f = L1v1 + m1(x)
        with
        L1 L1^T = K1 and 
    
        v2 ~ N(0, I)
        n = L2v2 
        with
        L2 L2^T = K2
        """
        X = DataHolder(X, on_shape_change='recompile')
        Y = DataHolder(Y, on_shape_change='recompile')
        GPModelHeteroNoiseRegression.__init__(self, X, Y, 
                                              kern1, kern2, mean_function)
        self.num_data = X.shape[0]
        self.num_latent = num_latent or Y.shape[1]
        self.V1 = Param(np.zeros((self.num_data, self.num_latent)))
        self.V1.prior = Gaussian(0., 1.)
        self.V2 = Param(np.zeros((self.num_data, self.num_latent)))
        self.V2.prior = Gaussian(0., 1.)
    
    def compile(self, session=None, graph=None, optimizer=None):
        """
        Before calling the standard compile function, check to see if the size
        of the data has changed and add parameters appropriately.

        This is necessary because the shape of the parameters depends on the
        shape of the data.
        """
        if not self.num_data == self.X.shape[0]:
            self.num_data = self.X.shape[0]
            self.V1 = Param(np.zeros((self.num_data, self.num_latent)))
            self.V1.prior = Gaussian(0., 1.)
            self.V2 = Param(np.zeros((self.num_data, self.num_latent)))
            self.V2.prior = Gaussian(0., 1.)

        return super(GPMCHeteroNoiseRegression, self).compile(session=session,
                                         graph=graph,
                                         optimizer=optimizer)
        
    def build_likelihood(self):
        
        """
        Construct a tf function to compute the likelihood of a general GP
        model with heteroscedatic noise represented using latent GP N(.).

            \log p(Y, V1, V2 | theta).

        """
        K1 = self.kern1.K(self.X)
        L1 = tf.cholesky(K1 + tf.eye(tf.shape(self.X)[0], dtype=float_type)*settings.numerics.jitter_level)
        F = tf.matmul(L1, self.V1) + self.mean_function(self.X)
        
        K2 = self.kern2.K(self.X)
        L2 = tf.cholesky(K2 + tf.eye(tf.shape(self.X)[0], dtype=float_type)*settings.numerics.jitter_level)
        N = tf.matmul(L2, self.V2) 
        
        return tf.reduce_sum(self.likelihood.logp(F, N, self.Y))
    
    def build_predict_f(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | (F=L1V1) )

        where F* are points on the GP at Xnew, F=L1V1 are points on the GP at X.

        """
        mu, var = conditional(Xnew, self.X, self.kern1, self.V1,
                              full_cov=full_cov,
                              q_sqrt=None, whiten=True)
        return mu + self.mean_function(Xnew), var
    
    def build_predict_n(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(N* | (N=L2V2) )

        where N* are points on the GP at Xnew, N=L2V2 are points on the GP at X.

        """
        mu, var = conditional(Xnew, self.X, self.kern2, self.V2,
                              full_cov=full_cov,
                              q_sqrt=None, whiten=True)
        return mu + self.mean_function(Xnew), var
    

class GPMCAdaptive(GPModelAdaptive):
    """ 
    X is a data matrix, size N x D
    Y is a data matrix, size N x 1
    kern1, kern2, kern3, likelihood are appropriate GPflow objects
    kern1: covariance function associated with adaptive lengthscale whose log is represented using GP L(.)
    kern2: covariance function associated with adaptive signal variance whose log is represented using GP S(.)
    kern3: covariance function associated with adaptive heteroscedastic noise whose log is represented using GP N(.)
    
    This is a vanilla implementation of an adaptive GP (non-stationary lengthscale and signal variance) 
    with a Gaussian likelihood and heteroscedatic noise.
    
    v1 ~ N(0, I)
    l = L1v1 
    with
    L1 L1^T = K1 and 
    
    v2 ~ N(0, I)
    s = L2v2
    with
    L2 L2^T = K2 and 
    
    v3 ~ N(0, I)
    n = L3v3
    with
    L3 L3^T = K3
    
    v4 ~ N(0, I)
    f = NonStatLv4
    with
    NonStatL NonStatL^T = NonStatK
    """
    
    def __init__(self, X, Y, kern1, kern2, kern3, num_latent=None): 
        
        X = DataHolder(X, on_shape_change='recompile')
        Y = DataHolder(Y, on_shape_change='recompile')
        GPModelAdaptive.__init__(self, X, Y, kern1, kern2, kern3)
        
        self.num_data = X.shape[0]
        self.num_latent = num_latent or Y.shape[1]
        
        self.V1 = Param(np.zeros((self.num_data, self.num_latent)))
        self.V1.prior = Gaussian(0., 1.)
        self.V2 = Param(np.zeros((self.num_data, self.num_latent)))
        self.V2.prior = Gaussian(0., 1.)
        self.V3 = Param(np.zeros((self.num_data, self.num_latent)))
        self.V3.prior = Gaussian(0., 1.)
        self.V4 = Param(np.zeros((self.num_data, self.num_latent)))
        self.V4.prior = Gaussian(0., 1.)
    
    def compile(self, session=None, graph=None, optimizer=None):
        """
        Before calling the standard compile function, check to see if the size
        of the data has changed and add parameters appropriately.

        This is necessary because the shape of the parameters depends on the
        shape of the data.
        """
        if not self.num_data == self.X.shape[0]:
            self.num_data = self.X.shape[0]
            self.V1 = Param(np.zeros((self.num_data, self.num_latent)))
            self.V1.prior = Gaussian(0., 1.)
            self.V2 = Param(np.zeros((self.num_data, self.num_latent)))
            self.V2.prior = Gaussian(0., 1.)
            self.V3 = Param(np.zeros((self.num_data, self.num_latent)))
            self.V3.prior = Gaussian(0., 1.)
            self.V4 = Param(np.zeros((self.num_data, self.num_latent)))
            self.V4.prior = Gaussian(0., 1.)

        return super(GPMCAdaptive, self).compile(session=session,
                                         graph=graph,
                                         optimizer=optimizer)
    
    def build_likelihood(self):
        
        """
        Construct a tf function to compute the likelihood of an adaptive GP
        model (non-stationary lengthscale and signal variance whose 
        log is represented using latent GPs L(.) and S(.) respectively)
        with heteroscedatic noise represented using latent GP N(.).

            \log p(Y, V1, V2, V3 | theta).

        """
        K1 = self.kern1.K(self.X)
        L1 = tf.cholesky(K1 + tf.eye(tf.shape(self.X)[0], dtype=float_type)*settings.numerics.jitter_level)
        L = tf.matmul(L1, self.V1)
        Lexp = tf.exp(L)
        
        K2 = self.kern2.K(self.X)
        L2 = tf.cholesky(K2 + tf.eye(tf.shape(self.X)[0], dtype=float_type)*settings.numerics.jitter_level)
        S = tf.matmul(L2, self.V2)
        Sexp = tf.exp(S)
        
        K3 = self.kern3.K(self.X)
        L3 = tf.cholesky(K3 + tf.eye(tf.shape(self.X)[0], dtype=float_type)*settings.numerics.jitter_level)
        N = tf.matmul(L3, self.V3)
       
        
        # Non stationary kernel
        NonStatRBF = NonStationaryRBF()
        Knonstat = NonStatRBF.K(self.X, Lexp, Sexp, self.X, Lexp, Sexp)
        Lnonstat = tf.cholesky(Knonstat + tf.eye(tf.shape(self.X)[0], dtype=float_type) * 1e-4)
        F = tf.matmul(Lnonstat, self.V4)
        return tf.reduce_sum(self.likelihood.logp(F, N, self.Y))
    

class GPMCAdaptiveLengthscale(GPModelAdaptiveLengthscale):
    """ 
    X is a data matrix, size N x D
    Y is a data matrix, size N x 1
    kern1
    kern1: covariance function associated with adaptive lengthscale whose log is represented using GP L(.)
    
    
    This is a vanilla implementation of an adaptive GP (non-stationary lengthscale) 
    with a Gaussian likelihood
    
    v1 ~ N(0, I)
    l = L1v1 
    with
    L1 L1^T = K1 and 
    
    v2 ~ N(0, I)
    f = NonStatLv2
    with
    NonStatL NonStatL^T = NonStatK
    """
    def __init__(self, X, Y, kern1, nonstat):
    
        X = DataHolder(X, on_shape_change='recompile')
        Y = DataHolder(Y, on_shape_change='recompile')
        GPModelAdaptiveLengthscale.__init__(self, X, Y, kern1, nonstat)
    
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]
        
        self.V1 = Param(np.zeros((self.num_data, self.num_latent)))
        self.V1.prior = Gaussian(0., 1.)
        self.V2 = Param(np.zeros((self.num_data, self.num_latent)))
        self.V2.prior = Gaussian(0., 1.)
    
    def compile(self, session=None, graph=None, optimizer=None):
        """
        Before calling the standard compile function, check to see if the size
        of the data has changed and add parameters appropriately.

        This is necessary because the shape of the parameters depends on the
        shape of the data.
        """
        if not self.num_data == self.X.shape[0]:
            self.num_data = self.X.shape[0]
            self.V1 = Param(np.zeros((self.num_data, self.num_latent)))
            self.V1.prior = Gaussian(0., 1.)
            self.V2 = Param(np.zeros((self.num_data, self.num_latent)))
            self.V2.prior = Gaussian(0., 1.)
            
        
        return super(GPMCAdaptiveLengthscale, self).compile(session=session,
                                         graph=graph,
                                         optimizer=optimizer)
    def build_likelihood(self):  
        """
        Construct a tf function to compute the likelihood of an adaptive GP
        model (non-stationary lengthscale whose 
        log is represented using latent GP L(.)).
        
            \log p(Y, V1| theta).
        """
        
        K1 = self.kern1.K(self.X)
        L1 = tf.cholesky(K1 + tf.eye(tf.shape(self.X)[0], dtype=float_type)*settings.numerics.jitter_level)
        L = tf.matmul(L1, self.V1)
        self.Lexp = tf.exp(L)
        
        # Non stationary kernel
        Knonstat = self.nonstat.K(self.X, self.Lexp, self.X, self.Lexp)
        Lnonstat = tf.cholesky(Knonstat + tf.eye(tf.shape(self.X)[0], dtype=float_type)*1e-4)
        F = tf.matmul(Lnonstat, self.V2)
        
        return tf.reduce_sum(self.likelihood.logp(F, self.Y))
    
    def build_predict_l(self, Xnew, full_cov=False):
        """
        Predict latent lengthscale GP L(.) at new points Xnew
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(L* | (L=L1V1) )

        where L* are points on the GP at Xnew, N=L1V1 are points on the GP at X.

        """
        mu, var = conditional(Xnew, self.X, self.kern1, self.V1,
                              full_cov=full_cov,
                              q_sqrt=None, whiten=True)
        return mu, var
    
    def build_predict_f(self, Xnew, full_cov=True):
        """
        Predict GP F(.) at new points Xnew
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | (L=LnonstatV2) )

        where L* are points on the GP at Xnew, N=L1V1 are points on the GP at X.

        """
        mu, var = nonstat_conditional(Xnew, self.X,
                                      self.nonstat, self.kern1,
                                      self.V1, self.V2, full_cov)
        return mu, var
    

class GPMCAdaptLAdaptN(GPModelAdaptLAdaptN):
    """ 
    X is a data matrix, size N x D
    Y is a data matrix, size N x 1
    kern1, kern2  are appropriate GPflow objects
    kern1: covariance function associated with adaptive lengthscale whose log is represented using GP L(.)
    kern2: covariance function associated with adaptive heteroscedastic noise whose log is represented using GP N(.)
    
    
    This is a vanilla implementation of an adaptive GP (non-stationary lengthscale) 
    with a Gaussian likelihood
    
    v1 ~ N(0, I)
    l = L1v1 
    with
    L1 L1^T = K1 and 
    
    v2 ~ N(0, I)
    n = L2v2
    with
    L2 L2^T = K2
    
    v3 ~ N(0, I)
    f = NonStatLv3
    with
    NonStatL NonStatL^T = NonStatK
    """
    
    def __init__(self, X, Y, kern1, kern2, nonstat, num_latent=None): 

        X = DataHolder(X, on_shape_change='recompile')
        Y = DataHolder(Y, on_shape_change='recompile')
        GPModelAdaptLAdaptN.__init__(self, X, Y, kern1, kern2, nonstat)
        
        self.num_data = X.shape[0]
        self.num_latent = num_latent or Y.shape[1]
        
        self.V1 = Param(np.zeros((self.num_data, self.num_latent)))
        self.V1.prior = Gaussian(0., 1.)
        self.V2 = Param(np.zeros((self.num_data, self.num_latent)))
        self.V2.prior = Gaussian(0., 1.)
        self.V3 = Param(np.zeros((self.num_data, self.num_latent)))
        self.V3.prior = Gaussian(0., 1.)
    
    def compile(self, session=None, graph=None, optimizer=None):
        """
        Before calling the standard compile function, check to see if the size
        of the data has changed and add parameters appropriately.

        This is necessary because the shape of the parameters depends on the
        shape of the data.
        """
        if not self.num_data == self.X.shape[0]:
            self.num_data = self.X.shape[0]
            self.V1 = Param(np.zeros((self.num_data, self.num_latent)))
            self.V1.prior = Gaussian(0., 1.)
            self.V2 = Param(np.zeros((self.num_data, self.num_latent)))
            self.V2.prior = Gaussian(0., 1.)
            self.V3 = Param(np.zeros((self.num_data, self.num_latent)))
            self.V3.prior = Gaussian(0., 1.)
        
        return super(GPMCAdaptLAdaptN, self).compile(session=session,
                                         graph=graph,
                                         optimizer=optimizer)
    
    def build_likelihood(self):  
        
        """
        Construct a tf function to compute the likelihood of an adaptive GP
        model (non-stationary lengthscale whose 
        log is represented using latent GP L(.)).
        \log p(Y, V1| theta).
        """
        
        K1 = self.kern1.K(self.X)
        L1 = tf.cholesky(K1 + tf.eye(tf.shape(self.X)[0], dtype=float_type)*settings.numerics.jitter_level)
        L = tf.matmul(L1, self.V1)
        self.Lexp = tf.exp(L)
        
        K2 = self.kern2.K(self.X)
        L2 = tf.cholesky(K2 + tf.eye(tf.shape(self.X)[0], dtype=float_type)*settings.numerics.jitter_level)
        N = tf.matmul(L2, self.V2)
    
        Knonstat = self.nonstat.K(self.X, self.Lexp, self.X, self.Lexp)
        Lnonstat = tf.cholesky(Knonstat + tf.eye(tf.shape(self.X)[0], dtype=float_type)*settings.numerics.jitter_level)
        F = tf.matmul(Lnonstat, self.V3)
        
        return tf.reduce_sum(self.likelihood.logp(F, N, self.Y))
    
    def build_predict_l(self, Xnew, full_cov=False):
        """
        Predict latent lengthscale GP L(.) at new points Xnew
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(L* | (L=L1V1) )

        where L* are points on the GP at Xnew, N=L1V1 are points on the GP at X.

        """
        mu, var = conditional(Xnew, self.X, self.kern1, self.V1,
                              full_cov=full_cov,
                              q_sqrt=None, whiten=True)
        
        return mu, var
    
    def build_predict_n(self, Xnew, full_cov=False):
        """
        Predict latent lengthscale GP N(.) at new points Xnew
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(N* | (L=L2V2) )

        where L* are points on the GP at Xnew, N=L2V2 are points on the GP at X.

        """
        mu, var = conditional(Xnew, self.X, self.kern2, self.V2,
                              full_cov=full_cov,
                              q_sqrt=None, whiten=True)
        
        return mu, var
    
    def build_predict_f(self, Xnew, full_cov=True):
        """
        Predict GP F(.) at new points Xnew
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | (L=LnonstatV2) )

        where L* are points on the GP at Xnew, N=L1V1 are points on the GP at X.

        """
        mu, var = nonstat_conditional(Xnew, self.X,
                                      self.nonstat, self.kern1,
                                      self.V1, self.V3, full_cov)
        return mu, var
    
    
        
    
    
    
    
    
    
        
    
    
    
