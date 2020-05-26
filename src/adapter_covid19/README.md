# AdaptER-Covid19

## Description

AdaptER-Covid19 is an economic model intended to simulate the economic effects of Covid19. For more information, see
the [model documentation](../../documentation/economic_model.ipynb)/

## Compilation

See [the root README](../../README.md).

## Usage

AdaptER-Covid19 is intended to be used as part of the wider OpenABM-Covid19 model. Please see [the root README](../../README.md) for usage instructions.

## Examples

[`adapter_covid19/example.py`](example.py) contains an example over how to run the economic model on its own.

```bash
python -m adapter_covid19.example
```

Alternatively:
```python
from adapter_covid19.data_structures import Scenario, ModelParams
from adapter_covid19.simulator import Simulator

# Define data path
data_path = f"./tests/adapter_covid19/data"

# Initialize simulator
simulator = Simulator(data_path)

# Define your senarior
scenario = Scenario()

# Run simulation
eco, states = simulator.simulate(scenario)
```

## Data

Example data is [here](../../tests/adapter_covid19/data) and a data specification can be found [here](../../tests/adapter_covid19/data).

## LICENSE

AdaptER-Covid19 is licensed separately to OpenABM-Covid19 under the [Apache 2.0 License](LICENSE).
