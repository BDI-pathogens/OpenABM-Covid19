OpenABM-Covid19: Agent-based model for modelling the Covid-19 epidemic
========================================================================

Description
-----------

OpenABM-Covid19 is an agent-based model (ABM) developed to simulate the spread of Covid-19 in a city and to analyse the effect of both passive and active intervention strategies.
Interactions between individuals are modelled on networks representing households, work-places and random contacts.
The infection is transmitted between these contacts and the progression of the disease in individuals is modelled.
Instantaneous contract-tracing and quarantining of contacts is modelled allowing the
evaluation of the design and configuration of digital contract-tracing mobile phone apps.

A full description of the model can be found [here](https://github.com/BDI-pathogens/OpenABM-Covid19/blob/master/documentation/covid19.md), a dictionary of input parameters can be found [here](./documentation/parameters/parameter_dictionary.md) and a dictionary of output files can be found [here](./documentation/output_files/output_file_dictionary.md).  
A report evaluating the efficacy of various configurations of digital contract-tracing mobile phone apps can be found [here](https://github.com/BDI-pathogens/covid-19_instant_tracing/blob/master/Report%20-%20Effective%20Configurations%20of%20a%20Digital%20Contact%20Tracing%20App.pdf) and the parameters used in the report are documented [here](https://github.com/BDI-pathogens/covid-19_instant_tracing/tree/master/OpenABM-Covid19%20parameters%20April%202020). 
The model was developed by the Pathogen Dynamics group, at the [Big Data Institute](https://www.bdi.ox.ac.uk/) at the University of Oxford, in conjunction with IBM UK and [Faculty](https://faculty.ai).
More details about our work can be found at [www.coronavirus-fraser-group.org ](https://045.medsci.ox.ac.uk/).  We suggest running from the latest commit in the master branch or from the latest release tag, which are created at major change points.  

### Economics Model

adaptER-covid19, and economics model, is attached to the main OpenABM-Covid19 model so the economic effect of Covid-19 can be modelled jointly with the spread of the disease. More information is [here](src/adapter_covid19/README.md).

Compilation options
-------------------

You can choose to use the GNU Scientific Library (GSL, GNU GPL licensed) (default) or the Stats library (Apache-2.0 licensed). You can also choose to run the Stats library's random number generator in GSL compatibility mode, although
this requires a 1 line code change to the stats library's /include/stats_incl/misc/statslib_options.hpp file to use the mt19937 random number generator and not the mt19937_64 generator.

To enable the stats library instead of GSL export this symbol before compilation:

```bash
export USE_STATS=1
```

When using the stats library, you can also choose to stay random number generator compatible with GSL by also exporting the following symbol before compilation:

```bash
export GSL_COMPAT=1
```

Compilation
-----------

OpenABM-Covid19 requires a C compiler (such as gcc) and the [GSL](https://www.gnu.org/software/gsl/) libraries installed.
Using the optional Stats library instead of GSL requires a C++ compiler, such as g++ included with gcc.
Python installation requires Python 3.7+

```bash
cd OpenABM-Covid19/
make all
```

To install the Python interface, first install [SWIG](http://www.swig.org/), then:

```bash
make install
```

For developers, the following installs the Python interface inplace, so modifications to the code are applied without needing to reinstall
```bash
make dev
```

Usage
-----

```bash
cd OpenABM-Covid19/src
./covid19ibm.exe <input_param_file> <param_line_number> <output_file_dir> <household_demographics_file>
```

where:
* `input_param_file` : is a csv file of parameter values (see [params.h](src/params.h) and the [parameter dictionary](./documentation/parameters/parameter_dictionary.md) for further details of the parameters)
* `param_line_number` : the line number of the parameter file for which to use for the simulation
* `output_file_dir` : path to output directory (this directory must already exist)
* `household_demographics_file` : a csv file from which samples are taken to represent household demographics in the model

We recommend running the model via the Python interface (see Examples section with scripts and notebooks below). Alternatively

```python
from COVID19.model import Model, Parameters
import COVID19.simulation as simulation

params = Parameters(
    input_param_file="./tests/data/baseline_parameters.csv",
    param_line_number=1,
    output_file_dir="./data_test",
    input_households="./tests/data/baseline_household_demographics.csv"
)
params.set_param( "n_total", 10000)

model = simulation.COVID19IBM(model = Model(params))
sim   = simulation.Simulation(env = model, end_time = 10 )
sim.steps( 10 )
print( sim.results )     

```

Examples
-----

The `examples/` directory contains some very simple Python scripts and Jupyter notebooks for running the model. The examples must be run from the example directory. In particular

1. `examples/example_101.py` - the simplest Python script for running the model
2. `examples/example_101.ipynb` - the simplest notebook of running the model and plotting some output
3. `examples/example_102.ipynb` - introduces a lock-down based upon the number of infected people and then after the lock-down turns on digital contact-tracing
4. `examples/example_extended_output.ipynb` - a detailed notebook analysing many aspect of the model and infection transmission.
5. `examples/multi_run_simulator.py` - an example of running the model multi-threaded

C and C++ Examples
------------------

The `examples/` directory also contains some testing examples that exercise the GSL and Stats functions, allowing comparison of output. These support the same compilation
settings mentioned above.

1. `examples/inv_inc_gamma_gsl` and `examples/inv_inc_gamma_stats` compare the incomplete inverse gamma solving algorithms of each library
2. `examples/random_data_gsl` and `examples/random_data_stats` compare every function defined in `random.h` for a variety of inputs. Running these commands and then using a visual text diff programme will show how comparable the results from each library are

_____


Contributing
------------

See document on [contributing](CONTRIBUTING.md) for further guidelines on contributing features and code to OpenABM-Covid19.  


Tests
-----

A full description of the tests run on the model can be found [here](https://github.com/BDI-pathogens/OpenABM-Covid19/blob/master/documentation/covid19_tests.pdf).
Tests are written using [pytest](https://docs.pytest.org/en/latest/getting-started.html) and can be run from the main project directory by calling `pytest`.  Tests require Python 3.7 or later.  Individual tests can be run using, for instance, `pytest tests/test_ibm.py::TestClass::test_hospitalised_zero`.  Tests have been run against modules listed in [tests/requirements.txt](tests/requirements) in case they are to be run within a virtual environment.  

R Packaging
-----------

**Note:** Due to a bug in SWIG [#1901](https://github.com/swig/swig/issues/1901), SWIG 4 is unsupported. You'll need to install version 3.0.12.

POSIX platform and Windows require different setups for building the R package.

### POSIX setup

On MacOS or Linux systems you will then need to generate and run a configure file (this will need installation of `autoconf` such as using Homebrew on MacOS, `brew install autoconf`).

```
autoconf
./configure
```

This will generate a file `src/Makevars` (ignored by git) which includes the necessary GSL compiler and linker flags that R needs when building the C source.

### Windows setup

- Set the environment variable `LIB_GSL` to install prefix of GSL. (It's recommended to compile and install GSL with [CMake](https://github.com/ampl/gsl/) instead of autotools).
- Install [Rtools](https://cran.r-project.org/bin/windows/Rtools/) and add C:\Rtools\usr\bin to your `PATH` environment variable.
- Install swig 3 using [Cygwin](https://www.cygwin.com/) or [MSYS2](https://www.msys2.org/) and add `swig.exe` to your `PATH` environment variable or set `SWIG3`.

You do not need to generate and run a configure script. R will use `src/Makevar.win` which uses the environment variable `LIB_GSL` to find GSL.

### All platforms

To build the R package, you first need to generate the SWIG source files:

```
make Rswig
```

If you have multiple versions of SWIG installed, you can set the `SWIG3` environment variable to the SWIG 3 executable. Example:

```
SWIG3=$(HOME)/swig-3.0.12/bin/swig make Rswig
```

If this is successful, the following files will be generated (ignored by git):

- R/OpenABMCovid19.R
- src/covid19_wrap_R.c

After that, YOU can then open the OpenABM-Covid19.Rproj file in RStudio and build as normal. Remember to re-run `make Rswig` every time about you modify the C source and/or SWIG interface.

You'll need to install some R packages. In the R console, run:
```
> install.packages(c("R6", "roxygen2", "devtools", "xptr"))
```

R console cheat-sheet:
```
> devtools::document() # Generate the R documentation files
> devtools::build()    # Bundle OpenABMCovid19 into a source package tarball
> devtools::install()  # Install the OpenABMCovid19 package.
> devtools::test()     # Run tests.
> devtools::check()    # Check R package.
```
