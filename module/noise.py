import numpy as np
from math import sqrt, log, cos, pi, exp
from numpy.linalg import cholesky


def white(sigma, v_vec, mu=0):
    """Returns the given array with white noise as numpy array"""
    noise_signal = np.random.normal(mu, sigma, len(v_vec))
    exp_v = np.add(v_vec, noise_signal)

    return np.array(exp_v)


def sampling_from_prior(p_set, num):
    """
    :param p_set: ParameterSet object
    :param num: Number of sampled items
    :return: A list with num dict elements: [{}, {}, ...]
    """
    tmp = []

    for i in range(num):
        sampled_set = {}
        for param in p_set.params:
            val = np.random.normal(param.mean, param.sigma)
            sampled_set[param.name] = val
        tmp.append(sampled_set)
        sampled_set = {}

    return tmp


def rep_traces(v, sigma, rep):
    """
    This method creates rep number synthetic data whit sigma std white noise

    :param v: Deterministic trace
    :param sigma: noise std
    :param rep: number of repetition
    :return: rep piece synthetic data
    """

    rep_trace = []

    for _ in range(rep):
        noise = np.random.normal(0, sigma, len(v))
        synthetic_data = np.add(v, noise)
        rep_trace.append(synthetic_data)

    return np.array(rep_trace)


def more_w_trace(sigma, model, params, rep):
    """
    Creates synthetic traces for given parameters and given noise effect repetition.
    
    :param sigma: White noise standard deviation
    :param rep: Number of created noisy traces for a single fixed parameter
    :param params: Fixed parameters. Generate synthetic data with these parameters. Type: [{},{},...]
    :return: (num_of_paramset, num_of_rep, trace_len) shaped np.array
    [param1, param2,...] -> [[trace,trace,trace,...], [trace, trace, tracce,...],...] where trace is the sythetic data
    """

    moretrace = []

    for item in params:
        current_param = []
        _, v = model(**item)

        for _ in range(rep):
            noise = np.random.normal(0, sigma, len(v))
            synthetic_data = np.add(v, noise)
            current_param.append(synthetic_data)

        moretrace.append(current_param)
        current_param = []

    return np.array(moretrace)


def colored(D, lamb, dt, v_vec):
    """Returns the given array with added colored noise as numpy array"""
    noise = []
    n, m = np.random.uniform(0.0, 1.0, 2)
    E = exp(-lamb * dt)
    e_0 = sqrt(-2 * D * lamb * log(m)) * cos(2 * pi * n)
    noise.append(e_0)

    for i in range(len(v_vec) - 1):
        a, b = np.random.uniform(0.0, 1.0, 2)
        h = sqrt(-2 * D * lamb * (1 - E ** 2) * log(a)) * cos(2 * pi * b)
        e_next = e_0 * E + h
        noise.append(e_next)
        e_0 = e_next

    return np.add(v_vec, noise)


def noise_from_covmat(covmat, v_vec):

    n = np.random.normal(size=len(v_vec))

    noise = np.dot(cholesky(covmat), n)

    return np.add(v_vec, noise)


def noise_from_cholesky(cholesky, v_vec):
    n = np.random.normal(size=len(v_vec))
    noise = np.dot(cholesky, n)
    return np.add(v_vec, noise)


def more_trace_from_covmat(covmat, model, params, rep):
    """
    Creates @param rep number of noised traces according to @param covmat for the given @param model and params

    :param covmat Covariance matrix of the noise
    :param model Deterministic model to superimpose noise
    :param params parameters for the deterministic model
    :param rep repetition of noisy traces for the given parameter
    """

    moretrace = []
    chol = np.linalg.cholesky(covmat)

    for item in params:
        current_param = []
        _, v = model(**item)

        for _ in range(rep):
            current_param.append(noise_from_cholesky(chol, v))

        moretrace.append(current_param)

    return np.array(moretrace)


def more_trace_from_cholesky(cholesky, model, params, rep):
    """

    """

    moretrace = []

    for item in params:
        current_param = []
        _, v = model(**item)

        for _ in range(rep):
            current_param.append(noise_from_cholesky(cholesky, v))

        moretrace.append(current_param)
        current_param = []

    return np.array(moretrace)


def more_c_trace(D, lamb, dt, model, params, rep):
    """

    :param D:
    :param lamb:
    :param dt:
    :param model:
    :param params:
    :return:
    """

    moretrace = []

    for item in params:
        current_param = []
        _, v = model(**item)

        for _ in range(rep):
            current_param.append(colored(D, lamb, dt, v))

        moretrace.append(current_param)
        current_param = []

    return np.array(moretrace)


# Colored noise with generators and list comprehension ----------------------------------------------------------

def colored_noise_generator(D, lamb, dt):
    """
    An iterable generator function for colored noise.

    :param D: Amplitude of the noise
    :param lamb: Reciprocal of the characteristic time
    :param dt: Time step
    :return : yields the successive value
    """

    e_0 = None
    E = exp(-lamb * dt)

    while True:
        if e_0 is None:
            # Create the first value
            n, m = np.random.uniform(0.0, 1.0, 2)
            e_0 = sqrt(-2 * D * lamb * log(m)) * cos(2 * pi * n)
            yield e_0
        else:
            # Create succession
            a, b = np.random.uniform(0.0, 1.0, 2)
            h = sqrt(-2 * D * lamb * (1 - E ** 2) * log(a)) * cos(2 * pi * b)
            e_next = e_0 * E + h
            e_0 = e_next
            yield e_0


def colored_vector(D, lamb, dt, vec):  # TODO not working properly!!!
    """
    Ads colored noise to the given vector.
    This function uses the colored_noise_generator() generator function

    :param D: amplitude of the noise
    :param lamb: reciprocal of the characteristic time
    :param dt: time step
    :param vec: the vector to extend with noise
    :return: the given vector with noise
    """

    # Create color generator
    noise_generator = colored_noise_generator(D, lamb, dt)

    # iterate through vec and add noise then return the list
    return [x + noise_generator.next() for x in vec]


# Create noise with known autocorrelation function ----------------------------------------------------------------

def near_psd(x, epsilon=0.00001):
    """
    Calculates the nearest positive semi-definite matrix for a correlation/covariance matrix

    :param x : array_like
    Covariance/correlation matrix
    :param epsilon : float
    Eigenvalue limit (usually set to zero to ensure positive definiteness)

    :return:
    near_cov : array_like
    closest positive definite covariance/correlation matrix
    """

    if min(np.linalg.eigvals(x)) > epsilon:
        return x

    # Removing scaling factor of covariance matrix
    n = x.shape[0]
    var_list = np.array([np.sqrt(x[i,i]) for i in xrange(n)])
    y = np.array([[x[i, j]/(var_list[i]*var_list[j]) for i in xrange(n)] for j in xrange(n)])

    # getting the nearest correlation matrix
    eigval, eigvec = np.linalg.eig(y)
    val = np.matrix(np.maximum(eigval, epsilon))
    vec = np.matrix(eigvec)
    T = 1/(np.multiply(vec, vec) * val.T)
    T = np.matrix(np.sqrt(np.diag(np.array(T).reshape(n))))
    B = T * vec * np.diag(np.array(np.sqrt(val)).reshape(n))
    near_corr = B*B.T

    # returning the scaling factors
    near_cov = np.array([[near_corr[i, j]*(var_list[i]*var_list[j]) for i in xrange(n)] for j in xrange(n)])

    print "Transformed to near positive definite matrix!"
    return near_cov


def cov_mat(f, t_vec):
    """
    Produce the covariance matrix for the given autocorrelation function

    :param f: The autocorrelation function of the noise (with fitted parameters)
    :param t_vec: The time series vector
    :return: The covariance matrix for the given autocorrelation function
    """

    return np.array([[f(t_vec[t1] - t_vec[t2]) for t2 in xrange(len(t_vec))] for t1 in xrange(len(t_vec))])


def inv_cov_mat(f, t_vec):
    """
    Produces inverse covariance matrix for given autocorrelation function

    :param f: autocorrrelation function (with adjusted parameters)
    :param t_vec: time vector for the function
    :return: the covariance matrix and the inverse covariance matrix
    """
    covmat = [[f(t_vec[t1] - t_vec[t2]) for t2 in xrange(len(t_vec))] for t1 in xrange(len(t_vec))]

    return np.array(covmat), np.linalg.inv(covmat)


def inv_cov_mat_sd(f, t_vec):
    """
    SOMA-DENDRIT recording version

    Produces inverse covariance matrix for given autocorrelation function

    :param f: autocorrrelation function (with adjusted parameters)
    :param t_vec: time vector for the function
    :return: the covariance matrix and the inverse covariance matrix
    """

    covmat = np.array([[f(t_vec[t1] - t_vec[t2]) for t2 in xrange(len(t_vec))] for t1 in xrange(len(t_vec))])

    # creating big matrix containing soma-dendrit cross-correlation
    s = covmat.shape[0]
    covmat_sd = np.zeros(shape=(s*2, s*2))

    # Fill the diagonal
    covmat_sd[0:s, 0:s] = covmat
    covmat_sd[s:s*2, s:s*2] = covmat

    # Fill the off-diagonal
    covmat_sd[0:s, s:s*2] = 0.5*covmat
    covmat_sd[s:s*2, 0:s] = 0.5*covmat

    inv_covmat_sd = np.linalg.inv(covmat_sd)

    return covmat_sd, inv_covmat_sd


def multivariate_normal(vec, t_vec, f):
    """
    Create a noise sampled from multivariate gaussian.
    When the autocorrelation function of the noise is known.

    :param vec: noise will be added to this vector
    :param t_vec: The time series vector (noise is correlated in time)
    :param f: The autocorrelation function of the noise (with fitted parameters, only one variable f(x))
    :return: vec with noise
    """

    # Generate noise sampled from standard normal pdf
    noise = np.random.normal(size=len(vec))

    # Decompose covariance matrix
    # And transform noise
    noise = np.dot(cholesky(cov_mat(f=f, t_vec=t_vec)), noise)

    return np.add(vec, noise)

