# AdaptER-Covid19

## Description

AdaptER-Covid19 is an economic model intended to simulate the economic effects of Covid19. For more information, see
the [model documentation](../../documentation/economic_model.pdf)/

## Compilation

See [the root README](../../README.md).

## Usage

AdaptER-Covid19 is intended to be used as part of the wider OpenABM-Covid19 model. Please see [the root README](../../README.md) for usage instructions.

## Examples

[`adapter_covid19/example.py`](example.py) contains an example over how to run the economic model on its own.
This can then be visualised in [`/examples/economics_visualisation.ipynb`](../../examples/economics_visualisation.ipynb).

```bash
python examples/example_run_spread_model_for_economics.py basic
python -m adapter_covid19.example tests/adapter_covid19/data basic
```

Alternatively:
```python
from adapter_covid19.data_structures import Scenario, ModelParams
from adapter_covid19.scenarios import BASIC_SCENARIO
from adapter_covid19.simulator import Simulator

from examples.example_run_spread_model_for_economics import run

# Define data path
data_path = "./tests/adapter_covid19/data"

# Initialize simulator
simulator = Simulator(data_path)

# Define your scenario, or use a predefined one
# scenario = Scenario()
# or
scenario = BASIC_SCENARIO

# Run the spread model for the economics simulator
run(scenario, data_path)

# Run simulation
help(simulator.simulate)  # for info
# Warning, this takes ~20 minutes to run and ~10GB of RAM
econ, states = simulator.simulate(scenario)
```

## Data

Example data is [here](../../tests/adapter_covid19/data) and a data specification can be found [here](../../tests/adapter_covid19/data).

## LICENSE

AdaptER-Covid19 is licensed separately to OpenABM-Covid19 under the [Apache 2.0 License](LICENSE).
