#!/usr/bin/env python3

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension


covid19_module = Extension('_covid19',
                           sources=['covid19_wrap.c',
                                    'constant.c',
                                    'demographics.c',
                                    'disease.c',
                                    'individual.c',
                                    'input.c',
                                    'interventions.c',
                                    'model.c',
                                    'network.c',
                                    'params.c',
                                    'utilities.c',
                                    'swig_utils.c'
                                    ],
                           extra_compile_args=['-g', '-Wall', '-fmessage-length=0', '-I$(INC);', '-O0'],
                           extra_link_args=['-lgsl', '-lgslcblas', '-lm', '-O3']
                          )

setup (name = 'covid19',
       version = '0.1',
       author      = "SWIG Docs",
       description = """Simple swig example from docs""",
       ext_modules = [covid19_module],
       py_modules = ["covid19", "COVID19/covid19_model", "COVID19/parameters", "COVID19/model"],
      )
