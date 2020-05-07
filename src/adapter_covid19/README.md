# AdaptER-Covid19

## Description

AdaptER-Covid19 is an economic model intended to simulate the economic effects of Covid19. For more information, see
the [model documentation](../../documentation/economic_model.md)/

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
import itertools
import numpy as np

from adapter_covid19.datasources import Reader
from adapter_covid19.corporate_bankruptcy import CorporateBankruptcyModel
from adapter_covid19.economics import Economics
from adapter_covid19.gdp import PiecewiseLinearCobbDouglasGdpModel
from adapter_covid19.personal_insolvency import PersonalBankruptcyModel
from adapter_covid19.scenarios import Scenario
from adapter_covid19.enums import Region, Sector, Age

reader = Reader('tests/adapter_covid19/data')
scenario = Scenario()
scenario.load(reader)
init_args = scenario.initialise()
gdp_model = PiecewiseLinearCobbDouglasGdpModel(**init_args.gdp_kwargs)
cb_model = CorporateBankruptcyModel(**init_args.corporate_kwargs)
pb_model = PersonalBankruptcyModel(**init_args.personal_kwargs)
econ = Economics(gdp_model, cb_model, pb_model, **init_args.economics_kwargs)
econ.load(reader)
dead = {key: 0.0 for key in itertools.product(Region, Sector, Age)}
ill = {key: 0.0 for key in itertools.product(Region, Sector, Age)}
states = []
for i in range(10):
    simulate_state = scenario.generate(
        time=i,
        dead=dead,
        ill=ill,
    )
    econ.simulate(simulate_state)
    states.append(simulate_state)
# inspect states
```

## Data

Example data is [here](../../tests/adapter_covid19/data) and a data specification can be found [here](../../tests/adapter_covid19/data).

## LICENSE

AdaptER-Covid19 is licensed separately to OpenABM-Covid19 under the [Apache 2.0 License](LICENSE).
