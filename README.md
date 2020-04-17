OpenABM-Covid19: Agent-based model for modelling the Covid-19 
========================================================================

Description
-----------

OpenABM-Covid19 is an agent-based model (ABM) developed to simulate the spread of Covid-19 in a city and to analyse the effect of both passive and active intervention strategies.
Interactions between individuals are modelled on networks representing households, work-places and random contacts.
The infection is transmitted between these contacts and the progression of the disease in individuals is modelled.
Instantaneous contract-tracing and quarantining of contacts is modelled allowing the
evaluation of the design and configuration of digital contract-tracing mobile phone apps.

A full description of the model can be found [here](https://github.com/p-robot/COVID19-IBM/blob/master/documentation/covid19_model.pdf). A report evaluating the efficacy of various configurations of digital contract-tracing mobile phone apps can be found [here](https://github.com/BDI-pathogens/covid-19_instant_tracing/blob/master/Report%20-%20Effective%20Configurations%20of%20a%20Digital%20Contact%20Tracing%20App.pdf). 
The model was developed by the Fraser group at the Big Data Institute at the University of Oxford in conjunction with IBM UK and faculty.ai. More details about our work can be found at [www.coronavirus-fraser-group.org ](https://045.medsci.ox.ac.uk/).


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
make swig-all
```

Usage
-----

```bash
cd OpenABM-Covid19/src
./covid19ibm.exe <input_param_file> <param_line_number> <output_file_dir> <household_demographics_file>
```

where:
* `input_param_file` : is a csv file of parameter values (see [params.h](src/params.h) for description of parameters)
* `param_line_number` : the line number of the parameter file for which to use for the simulation
* `output_file_dir` : path to output directory (this directory must already exist)
* `household_demographics_file` : a csv file from which samples are taken to represent household demographics in the model

Here is an example on how to use the Python interface:

```python
from COVID19.model import Model, Parameters
from COVID19.simulation import Simulation

parameters = Parameters(
    input_param_file="./tests/data/baseline_parameters.csv",
    param_line_number=1,
    output_file_dir="./data_test",
    input_household_file="./tests/data/baseline_household_demographics.csv"
)

model = Model(parameters)

simulation = Simulation(model, end_time=100, verbose=True)
simulation.simulations(n_simulations=3)
print(simulation.results_all_simulations)

```

Tests
-----

A full description of the tests run on the model can be found [here](https://github.com/p-robot/COVID19-IBM/blob/master/documentation/covid19_tests.pdf).
Tests are written using [pytest](https://docs.pytest.org/en/latest/getting-started.html) and can be run from the main project directory by calling `pytest`.  Tests require Python 3.6 or later.  Individual tests can be run using, for instance, `pytest tests/test_ibm.py::TestClass::test_hospitalised_zero`.  Tests have been run against modules listed in [tests/requirements.txt](tests/requirements) in case they are to be run within a virtual environment.  
