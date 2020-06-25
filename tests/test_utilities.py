#!/usr/bin/env python3
"""
Tests of the functions within utilities.c of the individual-based model OpenABM-Covid19.  

Usage:
With pytest installed (https://docs.pytest.org/en/latest/getting-started.html) tests can be 
run by calling 'pytest' from project folder.  

Created: June 2020
Author: p-robot
"""

import pytest, numpy as np, covid19
from scipy.stats import gamma, geom, nbinom, bernoulli, describe

from . import constant
from . import utilities as utils


def get_gamma_params(mu, sigma):
    """
    Return parameters a,b to gamma distribution from given mean, std
    (as expected by scipy.stats.gamma)
    """
    b = sigma * sigma / mu
    a = mu / b
    return(a, b)


def get_nbinom_params(mu, sigma):
    """
    Return parameters n, p to negative binomial from given mean, std
    (as expected by scipy.stats.nbinom)
    """
    p = mu / sigma / sigma;
    n = mu**2 / ( sigma**2 - mu );
    return(n, p)


def create_c_array(array, ctype):
    """
    Create a C version of a numpy array or Python list

    array : list or numpy array
        Array for which a C representation is wanted
    
    Returns
    -------
    array_c : 
        C array to which the Python array is to be copied
    """
    N = len(array)
    
    if ctype == "double":
        array_c = covid19.doubleArray(N)
    elif ctype == "long":
        array_c = covid19.longArray(N)
    elif ctype == "int":
        array_c = covid19.intArray(N)
    else: 
        print("Error: unknown data type")
        return

    for i in range(N):
        array_c[i] = array[i]
    return(array_c)


def c_array_as_python_list(array_c, N):
    """
    Return python list from a Swig object of a C array.  
    """
    array = []
    for i in range(N):
        array.append(array_c[i])
    return(array)


def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])
    metafunc.parametrize(
        argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
    )

class TestClass(object):
    """
    Test class for testing functions within utilities.c (C internals)
    """
    params = {
        "test_sum_square_diff_array": [dict()],
        "test_gamma_draw_list": [
            dict(n = 1000, mu = 5.6, sigma = 3.7),
            dict(n = 1000, mu = 5.6, sigma = 2.7),
            dict(n = 100, mu = 10.2, sigma = 2.7)],
        "test_bernoulli_draw_list": [
            dict(n = 1000, mu = 5.26),
            dict(n = 1000, mu = 1.31)],
        "test_geometric_max_draw_list": [
            dict(n = 1000, p = 0.1, maxv = 30),
            dict(n = 1000, p = 0.05, maxv = 30)],
        "test_negative_binomial_draw": [
            dict(N = 10000000, mu = 10.2, sigma = 5.5),
            dict(N = 10000000, mu = 20, sigma = 7)],
        "test_discrete_draw": [
            dict(array = [0.1, 0.1, 0.1, 0.1], M = 100000), # NB: GSL normalises the arrays
            dict(array = np.arange(1, 10)/11, M = 100000),
            dict(array = [0.01, 0.01, 0.9], M = 100000)],
        "test_n_unique_elements": [
            dict(array = np.random.randint(1, 1000, 1000))],
        "test_copy_array" : [dict()],
        "test_normalize_array" : [dict()],
        "test_gamma_rate_curve" : [
            dict(N = 35, mu = 10.6, sigma = 3.7, factor = 0.5),
            dict(N = 35, mu = 10.6, sigma = 3.7, factor = 2.5),
            dict(N = 35, mu = 10.6, sigma = 5.7, factor = 0.5),
            dict(N = 500, mu = 10.6, sigma = 7.7, factor = 0.5),
            dict(N = 500, mu = 10.6, sigma = 3.7, factor = 0.5)]
    }
    def test_sum_square_diff_array(self):
        N = 1000 
        np.random.seed(2020)
        array1 = np.random.uniform(0, 1, N)
        array2 = np.random.uniform(0, 1, N)

        sse_np = np.sum((array1 - array2)**2)
        
        array1_c = create_c_array(array1, ctype = "double")
        array2_c = create_c_array(array2, ctype = "double")
        sse_c = covid19.sum_square_diff_array(array1_c, array2_c, N)

        np.testing.assert_almost_equal(sse_np, sse_c, decimal = 10)

    def test_gamma_draw_list(self, n, mu, sigma):
        # Calculate using C
        array_c = covid19.intArray(n)
        covid19.gamma_draw_list(array_c, n, mu, sigma)
        array_c = c_array_as_python_list(array_c, n)

        # Calculate using numpy
        a, b = get_gamma_params(mu, sigma)
        array_np = np.round(gamma.ppf(( np.arange(n)  + 1 )/( n + 1 ), a, loc = 0, scale = b))
        array_np = np.maximum(array_np, 1)
        
        np.testing.assert_array_equal(array_c, array_np)
    
    def test_bernoulli_draw_list(self, n, mu): 
        array_c = covid19.intArray(n)
        covid19.bernoulli_draw_list(array_c, n, mu)
        array_c = c_array_as_python_list(array_c, n)
        
        a = int(np.floor(mu))
        array_np = a + bernoulli.ppf(( np.arange(n)  + 1 )/( n + 1 ), mu - a, loc = 0)
        
        np.testing.assert_equal(np.sort(array_c), array_np)
    
    def test_geometric_max_draw_list(self, n, p, maxv):
        """
        NB: slight mismatches on arrays possibly due to rounding error in C
        """
        array_c = covid19.intArray(n)
        covid19.geometric_max_draw_list(array_c, n, p, maxv)
        array_c = np.array(c_array_as_python_list(array_c, n))

        array_np = geom.ppf( np.linspace(1/n, 1 - 1/n, n) , p ) - 1
        array_np = np.minimum(array_np, maxv)
        
        np.testing.assert_almost_equal(np.mean(array_c), np.mean(array_np), decimal = 2)
    
    def test_negative_binomial_draw(self, N, mu, sigma):
        """
        NB: requires setting the GSL random seed (instantiating the rng object)
        """
        covid19.setup_gsl_rng(2021)
        
        n, p = get_nbinom_params(mu, sigma)
        
        sample_scipy = nbinom.rvs(n = n, p = p, size = N)
        sample_openabm = np.array([covid19.negative_binomial_draw( mu, sigma ) for i in range(N)])
        
        summary_exp = describe(sample_scipy)
        summary_obs = describe(sample_openabm)
        
        np.testing.assert_array_almost_equal(
            [summary_exp.mean, np.sqrt(summary_exp.variance), summary_exp.skewness], 
            [summary_obs.mean, np.sqrt(summary_obs.variance), summary_obs.skewness], 
            decimal = 2)

    def test_discrete_draw(self, array, M):
        """
        NB: requires setting the GSL random seed
        """
        covid19.setup_gsl_rng(2021)
        
        array_c = create_c_array(array, ctype = "double")
        samples = [covid19.discrete_draw(len(array), array_c) for i in range(M)]
        
        values, counts = np.unique(samples, return_counts = True)
        
        np.testing.assert_array_almost_equal(
            counts/np.sum(counts), 
            array/np.sum(array), decimal = 2)
    
    def test_n_unique_elements(self, array):
        N = len(array)
        array_c = create_c_array(array, ctype = "long")
        
        nel_openabm = covid19.n_unique_elements(array_c, N)
        nel_numpy = len(np.unique(array))
        
        np.testing.assert_equal(nel_openabm, nel_numpy)
    
    def test_copy_array(self):
        N = 100
        array = np.random.uniform(1, N, N)
        array_from = create_c_array(array, ctype = "double")
        array_to = covid19.doubleArray(N)
        
        covid19.copy_array(array_to, array_from, N)
        
        # Convert back to Python objects
        array_from = c_array_as_python_list(array_from, N)
        array_to = c_array_as_python_list(array_to, N)
        
        np.testing.assert_array_equal(array_from, array_to)
    
    def test_normalize_array(self):
        """
        NB: normalizes is implemented in-place
        """
        N = 100
        array = np.random.uniform(1, N, N)
        array_c = create_c_array(array, ctype = "double")
        covid19.normalize_array(array_c, N)
        array_c = c_array_as_python_list(array_c, N)
        
        array_numpy = array/np.sum(array)
        
        np.testing.assert_array_almost_equal(array_c, array_numpy, decimal = 10)
    
    def test_gamma_rate_curve(self, N, mu, sigma, factor):
        a, b = get_gamma_params(mu, sigma)
        
        array_scipy = factor * np.diff(gamma.cdf(( np.arange(N + 1)), a, loc = 0, scale = b))
        
        array_c = covid19.doubleArray(N)
        covid19.gamma_rate_curve(array_c, N, mu, sigma, factor)
        array_c = c_array_as_python_list(array_c, N)
        
        np.testing.assert_array_almost_equal(array_c, array_scipy, decimal = 4)
