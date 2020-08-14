OpenABM-Covid19: Agent-based model for modelling the Covid-19 
========================================================================

Description
-----------

OpenABM-Covid19 is an agent-based model (ABM) developed to simulate the spread of Covid-19 in a city and to analyse the effect of both passive and active intervention strategies.
Interactions between individuals are modelled on networks representing households, work-places and random contacts.
The infection is transmitted between these contacts and the progression of the disease in individuals is modelled.
Instantaneous contract-tracing and quarantining of contacts is modelled allowing the
evaluation of the design and configuration of digital contract-tracing mobile phone apps.

A full description of the model can be found [here](https://github.com/roberthinch/OpenABM-Covid19/blob/master/documentation/covid19.md), a dictionary of input parameters can be found [here](./documentation/parameters/parameter_dictionary.md) and a dictionary of output files can be found [here](./documentation/output_files/output_file_dictionary.md).  
A report evaluating the efficacy of various configurations of digital contract-tracing mobile phone apps can be found [here](https://github.com/BDI-pathogens/covid-19_instant_tracing/blob/master/Report%20-%20Effective%20Configurations%20of%20a%20Digital%20Contact%20Tracing%20App.pdf) and the parameters used in the report are documented [here](https://github.com/BDI-pathogens/covid-19_instant_tracing/tree/master/OpenABM-Covid19%20parameters%20April%202020). 
The model was developed by the Pathogen Dynamics group, at the [Big Data Institute](https://www.bdi.ox.ac.uk/) at the University of Oxford, in conjunction with IBM UK and [Faculty](https://faculty.ai).
More details about our work can be found at [www.coronavirus-fraser-group.org ](https://045.medsci.ox.ac.uk/).

### Economics Model

adaptER-covid19, and economics model, is attached to the main OpenABM-Covid19 model so the economic effect of Covid-19 can be modelled jointly with the spread of the disease. More information is [here](src/adapter_covid19/README.md).

Compilation
-----------

OpenABM-Covid19 requires a C compiler (such as gcc) and the [GSL](https://www.gnu.org/software/gsl/) libraries installed.
Python installation requires Python 3.7+

```bash
cd OpenABM-Covid19/src
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

_____

Tests
-----

A full description of the tests run on the model can be found [here](https://github.com/BDI-pathogens/OpenABM-Covid19/blob/master/documentation/covid19_tests.pdf).
Tests are written using [pytest](https://docs.pytest.org/en/latest/getting-started.html) and can be run from the main project directory by calling `pytest`.  Tests require Python 3.7 or later.  Individual tests can be run using, for instance, `pytest tests/test_ibm.py::TestClass::test_hospitalised_zero`.  Tests have been run against modules listed in [tests/requirements.txt](tests/requirements) in case they are to be run within a virtual environment.  

R Packaging
-----------

To build the R source or binary package, you should use the Makefile at the root of the source directory. You must have the following commands included in your `PATH` environment variable.

- `R`
- `swig`
- `git`

The Makefile will automatically run SWIG to generate source code (and other build steps such preprocessing *.in files) before running `R CMD build`.


To build the R source package, use:
```
make Rbuild
```
This will output a source package tar-ball named `OpenABMCovid19_${VERSION}.tar.gz`


To build the R binary package, use:
```
make Rinstall
```
This will output a binary package on your current platform (`.zip` for Window, `.tgz` for macOS).


### Requirements for building on Windows

If you're building on Windows, there are additional tools that are needed to build the binary package. This is the recommended setup:

- Download precompiled 32 and 64 bits libraries of GSL (installed in C:\gsl)
- Install Rtools (in C:\Rtools)
- Install SWIG via Cygwin64

It's adviced to put Rtools & R bin dir at the start of the `PATH` environment and the git & cygwin bin dir at the end.

Example environment:
```batch
@REM R & Rtools at the start
set PATH=C:\Program Files\R\R-3.6.2\bin\x64;%PATH%
set PATH=C:\Rtools\bin;%PATH%

@REM git & swig at the end
set PATH=%PATH%;C:\Program Files\Git\cmd
set PATH=%PATH%;C:\cygwin64\bin
```
