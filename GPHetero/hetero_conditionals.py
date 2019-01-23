
import tensorflow as tf
from gpflow.scoping import NameScoped
from gpflow._settings import settings
float_type = settings.dtypes.float_type


@NameScoped("conditional")
def conditional(Xnew, X, kern, f, full_cov=False, q_sqrt=None, whiten=False):
    """
    Given F, representing the GP at the points X, produce the mean and
    (co-)variance of the GP at the points Xnew.

    Additionally, there may be Gaussian uncertainty about F as represented by
    q_sqrt. In this case `f` represents the mean of the distribution and
    q_sqrt the square-root of the covariance.

    Additionally, the GP may have been centered (whitened) so that
        p(v) = N( 0, I)
        f = L v
    thus
        p(f) = N(0, LL^T) = N(0, K).
    In this case 'f' represents the values taken by v.

    The method can either return the diagonals of the covariance matrix for
    each output of the full covariance matrix (full_cov).

    We assume K independent GPs, represented by the columns of f (and the
    last dimension of q_sqrt).

     - Xnew is a data matrix, size N x D
     - X are data points, size M x D
     - kern is a GPflow kernel
     - f is a data matrix, M x K, representing the function values at X, for K functions.
     - q_sqrt (optional) is a matrix of standard-deviations or Cholesky
       matrices, size M x K or M x M x K
     - whiten (optional) is a boolean: whether to whiten the representation
       as described above.

    These functions are now considered deprecated, subsumed into this one:
        gp_predict
        gaussian_gp_predict
        gp_predict_whitened
        gaussian_gp_predict_whitened

    """

    # compute kernel stuff
    num_data = tf.shape(X)[0]  # M
    num_func = tf.shape(f)[1]  # K
    Kmn = kern.K(X, Xnew)
    Kmm = kern.K(X) + tf.eye(num_data, dtype=float_type) * settings.numerics.jitter_level
    Lm = tf.cholesky(Kmm)

    # Compute the projection matrix A
    A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = kern.K(Xnew) - tf.matmul(A, A, transpose_a=True)
        shape = tf.stack([num_func, 1, 1])
    else:
        fvar = kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
        shape = tf.stack([num_func, 1])
    fvar = tf.tile(tf.expand_dims(fvar, 0), shape)  # K x N x N or K x N

    # another backsubstitution in the unwhitened case
    if not whiten:
        A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)

    # construct the conditional mean
    fmean = tf.matmul(A, f, transpose_a=True)

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 2:
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # K x M x N
        elif q_sqrt.get_shape().ndims == 3:
            L = tf.matrix_band_part(tf.transpose(q_sqrt, (2, 0, 1)), -1, 0)  # K x M x M
            A_tiled = tf.tile(tf.expand_dims(A, 0), tf.stack([num_func, 1, 1]))
            LTA = tf.matmul(L, A_tiled, transpose_a=True)  # K x M x N
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" %
                             str(q_sqrt.get_shape().ndims))
        if full_cov:
            fvar = fvar + tf.matmul(LTA, LTA, transpose_a=True)  # K x N x N
        else:
            fvar = fvar + tf.reduce_sum(tf.square(LTA), 1)  # K x N
    fvar = tf.transpose(fvar)  # N x K or N x N x K

    return fmean, fvar

@NameScoped("nonstat_conditional")
def nonstat_conditional(Xnew, X, NonStatLRBF, kern1, v1, v2):
    """
    Given F, representing the nonstationary GP (variable lengthscale) at the points X, produce the mean and
    (co-)variance of the GP at the points Xnew.
    """
    
    # compute kernel stuff
    num_data = tf.shape(X)[0]  # M
    num_func = 1  # only one output GP
    
    L_GP_Kmn = kern1.K(X, Xnew)
    L_GP_Kmm = kern1.K(X) + tf.eye(num_data, dtype=float_type) * settings.numerics.jitter_level
    L_GP_Lm = tf.cholesky(L_GP_Kmm)
    
     # Compute the projection matrix A
    L_GP_A = tf.matrix_triangular_solve(L_GP_Lm, L_GP_Kmn, lower=True)
    L_GP_fvar = kern1.Kdiag(Xnew) - tf.reduce_sum(tf.square(L_GP_A), 0)
    shape = tf.stack([num_func, 1])
    L_GP_fvar = tf.tile(tf.expand_dims(L_GP_fvar, 0), shape)  # K x N x N or K x N
    
    # construct the conditional mean
    L_GP_fmean = tf.matmul(L_GP_A, v1, transpose_a=True)
    l_mu_new_exp = tf.exp(L_GP_fmean)
    
    L_GP_fvar = tf.transpose(L_GP_fvar)  # N x K or N x N x K
    
    # Compute non-stationary kernel stuff
    Len = tf.matmul(L_GP_Lm, v1)
    Lexp = tf.exp(Len)
    
    NonStat_Kmn = NonStatLRBF.K(X, Lexp, Xnew, l_mu_new_exp)
    NonStat_Kmm = NonStatLRBF.K(X, Lexp, X, Lexp) + tf.eye(num_data, dtype=float_type) * settings.numerics.jitter_level
    NonStat_Lm = tf.cholesky(NonStat_Kmm)
    
    # Compute the projection matrix A
    NonStat_A = tf.matrix_triangular_solve(NonStat_Lm, NonStat_Kmn, lower = True)
     
    # compute the covariance due to the conditioning
    NonStat_fvar = NonStatLRBF.K(Xnew, l_mu_new_exp, Xnew, l_mu_new_exp) - tf.matmul(NonStat_A, NonStat_A, transpose_a=True)
    #NonStat_shape = tf.stack([num_func, 1, 1])
    #NonStat_fvar = tf.tile(tf.expand_dims(NonStat_fvar, 0), NonStat_shape)  # K x N x N or K x N

    # construct the conditional mean
    NonStat_fmean = tf.matmul(NonStat_A, v2, transpose_a=True)

    NonStat_fvar = tf.transpose(NonStat_fvar)  # N x K or N x N x K
    
    return NonStat_fmean, NonStat_fvar
    