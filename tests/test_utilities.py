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
from scipy.stats import gamma, geom

from . import constant
from . import utilities as utils

def create_c_array(array, ctype = "double"):
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
    array_c = covid19.doubleArray(N)

    for i in range(N):
        array_c[i] = array[i]
    return(array_c)


def c_array_as_python_list(array_c, N):
    array = []
    for i in range(N):
        array.append(array_c[i])
    return(array)


class TestClass(object):
    """
    Test class
    """
    def test_sum_square_diff_array(self):
        """
        Test that sum_square_diff_array returns the same values as numpy
        """
        N = 1000 
        np.random.seed(2020)
        array1 = np.random.uniform(0, 1, N)
        array2 = np.random.uniform(0, 1, N)

        sse_np = np.sum((array1 - array2)**2)
        
        array1_c = create_c_array(array1)
        array2_c = create_c_array(array2)

        sse_c = covid19.sum_square_diff_array(array1_c, array2_c, N)

        np.testing.assert_almost_equal(sse_np, sse_c, decimal = 10)

    def test_gamma_draw_list(self):
        n = 1000
        mu = 5.6
        sigma = 3.7

        # Calculate using C
        array_c = covid19.intArray(n)
        covid19.gamma_draw_list(array_c, n, mu, sigma)
        array_c = c_array_as_python_list(array_c, n)

        # Calculate using numpy
        b = sigma * sigma / mu
        a = mu / b
        array_np = np.round(gamma.ppf(( np.arange(n)  + 1 )/( n + 1 ), a, loc = 0, scale = b))
        array_np = np.maximum(array_np, 1)

        np.testing.assert_array_equal(array_c, array_np)
    
    def test_bernoulli_draw_list(self): 
        n = 1000
        mu = 5.26

        array_c = covid19.intArray(n)
        covid19.bernoulli_draw_list(array_c, n, mu)
        array_c = c_array_as_python_list(array_c, n)

        a = int(np.floor(mu))
        p = int( (mu - a) * n )
        array_np = np.zeros(n)
        array_np[:p] = a + 1
        array_np[p:] = a
        
        np.testing.assert_array_equal(array_c, array_np)

    def test_geometric_max_draw_list(self):
        n = 1000
        p = 0.1
        maxv = 30

        array_c = covid19.intArray(n)
        covid19.geometric_max_draw_list(array_c, n, p, maxv)
        array_c = c_array_as_python_list(array_c, n)

        array_np = geom.ppf(( np.arange(n)  + 1 )/( n + 1 ), p) - 1
        array_np = np.minimum(array_np, maxv)

        np.testing.assert_array_equal(array_c, array_np)


    # def test_gamma_rate_curve(n, mean, sd, factor):
    #     array_c = covid19.intArray(n)
    #     covid19.gamma_rate_curve(array_c, n, mean, sd, factor)
    #     return(array_c)

    # def test_normalize_array(array):
    #     N = len(array)
    #     covid19.normalize_array(array, N)

    # def test_copy_array(to, from):
    #     N = len(to)
    #     covid19.copy_array(to, from, N)

    # def test_copy_normalize_array(to, from):
    #     N = len(to)
    #     covid19.copy_normalize_array(to, from, N)