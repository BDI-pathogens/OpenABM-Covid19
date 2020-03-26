COVID19-IBM: Individual-based model for modelling of a COVID-19 outbreak
========================================================================


Compilation
-----------

```bash
cd COVID19-IBM/src
make clean; make all
```

Usage
-----

```bash
cd COVID19-IBM/src
./covid19ibm.exe <input_param_file> <param_line_number> <household_demographics_file>
```

where:
* `input_param_file` : is a csv file of parameter values (see [params.h](src/params.h) for description of parameters)
* `param_line_number` : the line number of the parameter file for which to use for the simulation
* `household_demographics_file` : a csv file from which samples are taken to represent household demographics in the model

Tests
-----

Tests are written using [pytest](https://docs.pytest.org/en/latest/getting-started.html) and can be run from the main project directory by calling `pytest`.  Individual tests can be run using, for instance, `pytest tests/test_ibm.py::TestClass::test_hospitalised_zero`.

