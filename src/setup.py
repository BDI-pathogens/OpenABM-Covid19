#!/usr/bin/env python3

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension
from subprocess import check_output

def gsl_config(flag):
    out = check_output(['gsl-config'] + [flag]).decode('utf-8')
    out = out.replace("\n",'') # cut trailing \n
    return out.split(' ')

CFLAGS  = gsl_config('--cflags')
LDFLAGS = gsl_config('--libs')

covid19_module = Extension(
    "_covid19",
    sources=[
        "covid19_wrap.c",
        "constant.c",
        "demographics.c",
        "disease.c",
        "doctor.c",
        "hospital.c",
        "individual.c",
        "input.c",
        "interventions.c",
        "list.c",
        "model.c",
        "network.c",
        "nurse.c",
        "params.c",
        "strain.c",
        "utilities.c",
        "ward.c"
    ],
    extra_compile_args=["-g", "-Wall", "-fmessage-length=0", "-O0"] + CFLAGS,
    extra_link_args=["-lm", "-O3"] + LDFLAGS,
)

setup(
    name="covid19",
    version="0.2",
    author="SWIG Docs",
    description="""Individual-based model for modelling of a COVID-19 outbreak""",
    ext_modules=[covid19_module],
    packages=["COVID19", "adapter_covid19",],
    py_modules=["covid19",],
    package_dir  = {"COVID19" : "COVID19"},
    package_data = {"COVID19" : ["default_params/*.csv"] },
    include_package_data=True,
    install_requires=[
        "click",
        "matplotlib==3.2.2",
        "numpy",
        "pandas",
        "scipy",
        "tqdm",
        "dataclasses",
    ],
)
