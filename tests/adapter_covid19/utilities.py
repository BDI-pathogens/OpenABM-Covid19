import copy
import itertools

from adapter_covid19.data_structures import SimulateState, Utilisation, Utilisations
from adapter_covid19.datasources import Reader
from adapter_covid19.enums import Region, Sector, Age, WorkerState

DATA_PATH = "tests/adapter_covid19/data"

ILL_STATES = {
    WorkerState.ILL_UNEMPLOYED,
    WorkerState.ILL_FURLOUGHED,
    WorkerState.ILL_WFH,
    WorkerState.ILL_WFO,
}

UTILISATION_NO_COVID_NO_LOCKDOWN = Utilisation(
    p_dead=0,
    p_ill_wfo=0,
    p_ill_wfh=0,
    p_ill_furloughed=0,
    p_ill_unemployed=0,
    p_wfh=0,
    p_furloughed=0,
)
UTILISATION_COVID_NO_LOCKDOWN = Utilisation(
    p_dead=0.1,
    p_ill_wfo=0.5,
    p_ill_wfh=0.5,
    p_ill_furloughed=0.5,
    p_ill_unemployed=0.5,
    p_wfh=0.0,
    p_furloughed=0.0,
)
UTILISATION_NO_COVID_LOCKDOWN = Utilisation(
    p_dead=0.0,
    p_ill_wfo=0.0,
    p_ill_wfh=0.0,
    p_ill_furloughed=0.0,
    p_ill_unemployed=0.0,
    p_wfh=0.9,
    p_furloughed=1.0,
    p_not_employed=0.1,
)
UTILISATION_COVID_LOCKDOWN = Utilisation(
    p_dead=0.0001,
    p_ill_wfo=0.01,
    p_ill_wfh=0.01,
    p_ill_furloughed=0.01,
    p_ill_unemployed=0.01,
    p_wfh=0.9,
    p_furloughed=1.0,
    p_not_employed=0.1,
)

ALL_UTILISATIONS = (
    UTILISATION_NO_COVID_NO_LOCKDOWN,
    UTILISATION_COVID_NO_LOCKDOWN,
    UTILISATION_NO_COVID_LOCKDOWN,
    UTILISATION_COVID_LOCKDOWN,
)


def state_from_utilisation(
    utilisation: Utilisation,
    new_spending_day: int = 10 ** 6,
    ccff_day: int = 10 ** 6,
    loan_guarantee_day: int = 10 ** 6,
) -> SimulateState:
    reader = Reader(DATA_PATH)
    utilisations = Utilisations(
        {k: copy.deepcopy(utilisation) for k in itertools.product(Region, Sector, Age)},
        reader=reader,
    )
    lambdas = utilisation.to_lambdas()
    ill = sum(v for k, v in lambdas.items() if k in ILL_STATES)
    dead = lambdas[WorkerState.DEAD]
    state = SimulateState(
        time=0,
        dead=dead,
        ill=ill,
        lockdown=utilisation.p_wfh > 0,
        furlough=utilisation.p_furloughed > 0,
        new_spending_day=new_spending_day,
        ccff_day=ccff_day,
        loan_guarantee_day=loan_guarantee_day,
        utilisations=utilisations,
    )
    return state


def advance_state(state: SimulateState, utilisation: Utilisation,) -> SimulateState:
    new_state = state_from_utilisation(
        utilisation, state.new_spending_day, state.ccff_day, state.loan_guarantee_day,
    )
    new_state.time = state.time + 1
    new_state.previous = state
    return new_state
