import logging
import pickle

try:
    from tqdm import tqdm
except:

    def tqdm(x):
        return x


from adapter_covid19.scenarios import *
from adapter_covid19.simulator import Simulator

logger = logging.getLogger(__file__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import sys

    if len(sys.argv) > 1:
        scenario_names = sys.argv[1:]
    else:
        scenario_names = SCENARIOS.keys()

    unknown_scenarios = [n for n in scenario_names if n not in SCENARIOS.keys()]
    if len(unknown_scenarios) > 0:
        logger.error(f"Unkown scenarios! {unknown_scenarios}")
        exit(1)

    s = Simulator()

    for scenario_name in scenario_names:
        logger.info(f"Running scenario {scenario_name}")
        scenario = SCENARIOS[scenario_name]
        result = s.simulate(
            scenario=scenario, show_plots=False, scenario_name=scenario_name
        )
        file_name = f"scenario_{scenario_name}.pkl"
        logger.info(f"Writing file {file_name}")
        with open(file_name, "wb") as f:
            pickle.dump((scenario_name, scenario, result), f)
        logger.info(f"Finished writing")
