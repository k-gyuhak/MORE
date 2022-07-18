import numpy as np
from scipy import linalg, sparse
from sklearn.utils import check_array
from sklearn.utils.extmath import svd_flip
from sklearn.decomposition import IncrementalPCA

import warnings
warnings.filterwarnings("ignore")

def _safe_accumulator_op(op, x, *args, **kwargs):
    """
    This function provides numpy accumulator functions with a float64 dtype
    when used on a floating point input. This prevents accumulator overflow on
    smaller floating point dtypes.

    Parameters
    ----------
    op : function
        A numpy accumulator function such as np.mean or np.sum
    x : numpy array
        A numpy array to apply the accumulator function
    *args : positional arguments
        Positional arguments passed to the accumulator function after the
        input x
    **kwargs : keyword arguments
        Keyword arguments passed to the accumulator function

    Returns
    -------
    result : The output of the accumulator function passed to this function
    """
    if np.issubdtype(x.dtype, np.floating) and x.dtype.itemsize < 8:
        result = op(x, *args, **kwargs, dtype=np.float64)
    else:
        result = op(x, *args, **kwargs)
    return result

class MyIPCA(IncrementalPCA):
    def __init__(self, n_components, ff=1., whiten=False, copy=True, batch_size=None, max_size=512):
        super(MyIPCA, self).__init__(n_components, whiten, copy, batch_size)
        self.ff = ff

        self.n_samples_seen_ = 1
        self.mean_ = np.random.uniform(0, 1, size=max_size) * 1e-15
        self.explained_variance_ = np.ones(n_components)
        self.components_ = np.hstack((np.eye(n_components),
                                        np.zeros(shape=(n_components, max_size - n_components)))) * 1e-15
        self.singular_values_ = np.ones(n_components)


    def partial_fit(self, X, y=None, check_input=True):
        """Incremental fit with X. All of X is processed as a single batch.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        check_input : bool
            Run check_array on X.

        y : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if check_input:
            if sparse.issparse(X):
                raise TypeError(
                    "IncrementalPCA.partial_fit does not support "
                    "sparse input. Either convert data to dense "
                    "or use IncrementalPCA.fit to do so in batches.")
            X = check_array(X, copy=self.copy, dtype=[np.float64, np.float32])
        n_samples, n_features = X.shape
        if not hasattr(self, 'components_'):
            self.components_ = None

        if self.n_components is None:
            if self.components_ is None:
                self.n_components_ = min(n_samples, n_features) # ONCE N_COMPONENTS_ ARE SET HERE, IT'S FOREVER THIS VALUE.
            else:
                self.n_components_ = self.components_.shape[0]
        elif not 1 <= self.n_components <= n_features:
            raise ValueError("n_components=%r invalid for n_features=%d, need "
                             "more rows than columns for IncrementalPCA "
                             "processing" % (self.n_components, n_features))
        # elif not self.n_components <= n_samples:
        #     raise ValueError("n_components=%r must be less or equal to "
        #                      "the batch number of samples "
        #                      "%d." % (self.n_components, n_samples))
        else:
            self.n_components_ = self.n_components

        # if (self.components_ is not None) and (self.components_.shape[0] !=
        #                                        self.n_components_):
        #     raise ValueError("Number of input features has changed from %i "
        #                      "to %i between calls to partial_fit! Try "
        #                      "setting n_components to a fixed value." %
        #                      (self.components_.shape[0], self.n_components_))

        # This is the first partial_fit
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = 0
            self.mean_ = .0
            self.var_ = .0

        # # Update stats - they are 0 if this is the first step
        # # print(np.repeat(self.n_samples_seen_, X.shape[1])) # AT FIRST, [0 0 0 0 0]. AFTER FITTING 6 SAMPLES, [6 6 6 6 6]. AFTER FITTING 8 ADDITIONAL SAMPLES (6 + 8 = 14), [14 14 14 14 14]. I THINK THE LENGTH IS THE SIZE OF FEATURES
        # col_mean, col_var, n_total_samples = \
        #     _incremental_mean_and_var(
        #         X, last_mean=self.mean_, last_variance=self.var_,
        #         last_sample_count=np.repeat(self.n_samples_seen_, X.shape[1])) # SO... VARIANCE ARE NOT USED FOR COMPUTING SVD... AND THUS COMPUTING THE COVARIANCE
        # n_total_samples = n_total_samples[0]
        # # print(col_mean.shape, col_var.shape) # (5,) (5,) where 5 is the size of features
        # # print(col_var) # WHY IS IT 1D VECTOR? SHOULDN'T BE (D, D) MATRIX?

        n_total_samples = self.n_samples_seen_ + len(X)
        col_mean = (self.mean_ * self.n_samples_seen_ * self.ff +
                    _safe_accumulator_op(np.nansum, X, axis=0)) / (self.ff * self.n_samples_seen_ + len(X))

        # Whitening
        if self.n_samples_seen_ == 0:
            # If it is the first step, simply whiten X
            X -= col_mean
        else:
            col_batch_mean = np.mean(X, axis=0)
            X -= col_batch_mean
            # Build matrix of combined previous basis and new data
            mean_correction = \
                np.sqrt((self.n_samples_seen_ * n_samples) /
                        n_total_samples) * (self.mean_ - col_batch_mean)
            # print((self.singular_values_.reshape((-1, 1)) * self.components_).shape) # (4, 5), where 4 is the n_components, and 5 is the num features
            # print(mean_correction.shape) # (5,), where 5 is the num features
            # print(X.shape) # (num samples provided, 5), where 5 is the num features
            X = np.vstack(((self.ff * self.singular_values_).reshape((-1, 1)) *
                           self.components_, X, mean_correction))
            # print(X.shape) # (num samples provided + n_components + 1, 5), where 5 is the num features

        U, S, V = linalg.svd(X, full_matrices=False)
        U, V = svd_flip(U, V, u_based_decision=False)
        explained_variance = S ** 2 / (n_total_samples - 1)
        # explained_variance_ratio = S ** 2 / np.sum(col_var * n_total_samples)

        self.n_samples_seen_ = n_total_samples
        self.components_ = V[:self.n_components_]
        self.singular_values_ = S[:self.n_components_]
        self.mean_ = col_mean
        # self.var_ = col_var
        self.explained_variance_ = explained_variance[:self.n_components_]
        # self.explained_variance_ratio_ = \
        #     explained_variance_ratio[:self.n_components_]
        if self.n_components_ < n_features:
            self.noise_variance_ = \
                explained_variance[self.n_components_:].mean()
        else:
            self.noise_variance_ = 0.
        return self.n_samples_seen_, self.components_, self.singular_values_, self.mean_, \
                self.explained_variance_, self.noise_variance_

class MovingAvg:
    def __init__(self, ff=1.):
        self.ff = ff
    def partial_fit(self, X):
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = 0
            self.mean_ = .0
        n_total_samples = self.n_samples_seen_ + len(X)
        col_mean = (self.mean_ * self.n_samples_seen_ * self.ff +
                    _safe_accumulator_op(np.nansum, X, axis=0)) / (self.ff * self.n_samples_seen_ + len(X))
        self.n_samples_seen_ = n_total_samples
        self.mean_ = col_mean
