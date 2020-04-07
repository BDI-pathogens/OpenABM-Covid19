COVID19-IBM: Individual-based model for modelling of a COVID-19 outbreak
========================================================================


Compilation
-----------

COVID19-IBM requires a C compiler (such as gcc) and the [GSL](https://www.gnu.org/software/gsl/) libraries installed.

```bash
cd COVID19-IBM/src
make clean; make all
```

Usage
-----

```bash
cd COVID19-IBM/src
./covid19ibm.exe <input_param_file> <param_line_number> <output_file_dir> <household_demographics_file>
```

where:
* `input_param_file` : is a csv file of parameter values (see [params.h](src/params.h) for description of parameters)
* `param_line_number` : the line number of the parameter file for which to use for the simulation
* `output_file_dir` : path to output directory
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

Tests are written using [pytest](https://docs.pytest.org/en/latest/getting-started.html) and can be run from the main project directory by calling `pytest`.  Tests require Python 3.6 or later.  Individual tests can be run using, for instance, `pytest tests/test_ibm.py::TestClass::test_hospitalised_zero`.  Tests have been run against modules listed in [tests/requirements.txt](tests/requirements) in case they are to be run within a virtual environment.  
