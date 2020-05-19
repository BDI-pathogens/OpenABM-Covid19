#!/usr/bin/env python3

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension

covid19_module = Extension(
    "_covid19",
    sources=[
        "covid19_wrap.c",
        "constant.c",
        "demographics.c",
        "disease.c",
        "individual.c",
        "input.c",
        "interventions.c",
        "model.c",
        "network.c",
        "params.c",
        "utilities.c",
    ],
    extra_compile_args=["-g", "-Wall", "-fmessage-length=0", "-I$(INC);", "-O0"],
    extra_link_args=["-lgsl", "-lgslcblas", "-lm", "-O3"],
)

setup(
    name="covid19",
    version="0.2",
    author="SWIG Docs",
    description="""Individual-based model for modelling of a COVID-19 outbreak""",
    ext_modules=[covid19_module],
    packages=["COVID19", "adapter_covid19",],
    py_modules=["covid19",],
    install_requires=["click", "matplotlib", "numpy", "pandas", "scipy", "tqdm", "dataclasses"],
)
