#!/usr/bin/env python3

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension
from subprocess import check_output
import os

def usingStats():
    useStats = os.environ.get('USE_STATS')

    if useStats is not None:
        return True
    return False

# This is recommended, but causes the GSL build to fail in Docker
def python_config():
    out = check_output(['python3-config','--ldflags']).decode('utf-8')
    out = out.replace("\n",'') # cut trailing \n
    return out.split(' ')

def gsl_config(flag):
    if usingStats():
        statsRoot=".."
        d = os.environ.get('D')
        if d is not None:
            statsRoot = d
        
        if '--cflags' == flag:
            return ("-DSTATS_ENABLE_STDVEC_WRAPPERS -DSTATS_GO_INLINE -DUSE_STATS -I" + d + "/stats/include -I" + d + "/gcem/include -std=c++17").split(' ')
        else:
            pythonLib = os.environ.get('PYTHONLIB')
            if pythonLib is not None:
                return pythonLib.split(' ')
            else:
                return []
    
    out = check_output(['gsl-config'] + [flag]).decode('utf-8')
    out = out.replace("\n",'') # cut trailing \n
    return out.split(' ')

CFLAGS  = gsl_config('--cflags')
LDFLAGS = gsl_config('--libs') #+ python_config()

srcs = [
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
]
if not usingStats():
  srcs.append("random_gsl.c")
else:
  srcs.append("random_stats.cpp")

lang = "c"
if usingStats():
  lang = "c++"

covid19_module = Extension(
    "_covid19",
    srcs,
    extra_compile_args=["-g", "-Wall", "-fmessage-length=0", "-O2", "-fPIC"] + CFLAGS,
    extra_link_args=["-lm", "-O2", "-fPIC", "-shared"] + LDFLAGS,
    language=lang,
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
