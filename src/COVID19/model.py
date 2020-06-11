import logging
import enum
from itertools import chain
from typing import Union
import pandas as pd

import covid19

LOGGER = logging.getLogger(__name__)


class ModelParameterException(Exception):
    pass


class ParameterException(Exception):
    pass


PYTHON_SAFE_UPDATE_PARAMS = [
    "test_on_symptoms",
    "test_on_traced",
    "quarantine_on_traced",
    "traceable_interaction_fraction",
    "tracing_network_depth",
    "allow_clinical_diagnosis",
    "quarantine_household_on_positive",
    "quarantine_household_on_symptoms",
    "quarantine_household_on_traced_positive",
    "quarantine_household_on_traced_symptoms",
    "quarantine_household_contacts_on_positive",
    "quarantine_household_contacts_on_symptoms",
    "quarantine_days",
    "test_order_wait",
    "test_result_wait",
    "self_quarantine_fraction",
    "lockdown_on",
    "lockdown_elderly_on",
    "app_turned_on",
    "app_users_fraction",
    "trace_on_symptoms",
    "trace_on_positive",
    "lockdown_house_interaction_multiplier",
    "lockdown_random_network_multiplier",
    "lockdown_occupation_multiplier_primary_network",
    "lockdown_occupation_multiplier_secondary_network",
    "lockdown_occupation_multiplier_working_network",
    "lockdown_occupation_multiplier_retired_network",
    "lockdown_occupation_multiplier_elderly_network",
    "relative_transmission_household",
    "relative_transmission_occupation",
    "relative_transmission_random",
]


class EVENT_TYPES(enum.Enum):
    SUSCEPTIBLE = 0
    PRESYMPTOMATIC = 1 # Pre-symptompatic, severe disease (progressing to symptomatic severe)
    PRESYMPTOMATIC_MILD = 2 # Pre-symptompatic, mild disease (progressing to symptomatic mild)
    ASYMPTOMATIC = 3 # Asymptompatic (progressing to recovered)
    SYMPTOMATIC = 4 # Symptompatic, severe disease
    SYMPTOMATIC_MILD = 5 # Symptompatic, mild disease
    HOSPITALISED = 6
    CRITICAL = 7
    HOSPITALISED_RECOVERING = 8
    RECOVERED = 9
    DEATH = 10
    QUARANTINED = 11
    QUARANTINE_RELEASE = 12
    TEST_TAKE = 13
    TEST_RESULT = 14
    CASE = 15
    TRACE_TOKEN_RELEASE = 16
    TRANSITION_TO_HOSPITAL = 17
    N_EVENT_TYPES = 18


class OccupationNetworkEnum(enum.Enum):
    _primary_network = 0
    _secondary_network = 1
    _working_network = 2
    _retired_network = 3
    _elderly_network = 4



class AgeGroupEnum(enum.Enum):
    _0_9 = 0
    _10_19 = 1
    _20_29 = 2
    _30_39 = 3
    _40_49 = 4
    _50_59 = 5
    _60_69 = 6
    _70_79 = 7
    _80 = 8


class ChildAdultElderlyEnum(enum.Enum):
    _child = 0
    _adult = 1
    _elderly = 2


class ListIndiciesEnum(enum.Enum):
    _1 = 0
    _2 = 1
    _3 = 2
    _4 = 3
    _5 = 4
    _6 = 5


class TransmissionTypeEnum(enum.Enum):
    _household = 0
    _occupation = 1
    _random = 2


def _get_base_param_from_enum(param):
    base_name, enum_val = None, None
    for en in chain(
        AgeGroupEnum, ChildAdultElderlyEnum, ListIndiciesEnum, TransmissionTypeEnum, OccupationNetworkEnum
    ):
        LOGGER.debug(f"{en.name} =={param[-1 * len(en.name) :]} ")
        if en.name == param[-1 * len(en.name) :]:
            base_name = param.split(en.name)[0]
            enum_val = en.value
            LOGGER.debug(f"Split to {base_name} and {enum_val}")
            break
    return base_name, enum_val


class Parameters(object):
    def __init__(
            self,
            input_param_file: str = None,
            param_line_number: int = 1,
            output_file_dir: str = "./",
            input_households: Union[str, pd.DataFrame] = None,
            read_param_file=True,
    ):
        """[summary]
        
        Arguments:
            object {[type]} -- [description]
        
        Keyword Arguments:
            input_param_file {str} -- [Parameters file path] (default: {None})
            param_line_number {int} -- [Which column of the input param file to read] (default: 1)
            output_file_dir {str} -- [Where to write output files to] (default: {"./"})
            input_households {str} -- [Household demographics file (required)] (default: {None})
            read_param_file {bool} -- [Read param file, all params can be set from python interface] (default: {True})
        
        Raises:
            ParameterException: [Warnings if parameters are not correctly set]
            Sys.exit(0): [Underlaying C code will exist if params are not viable]
        """
        self.c_params = covid19.parameters()
        if input_param_file:
            self.c_params.input_param_file = input_param_file
        elif not input_param_file and read_param_file:
            raise ParameterException(
                "Input param path is None and read param file set to true"
            )
        else:
            LOGGER.info(
                "Have not passed input file for params, use set_param or set_param_dict"
            )
        if param_line_number:
            self.c_params.param_line_number = int(param_line_number)
        self.c_params.output_file_dir = output_file_dir
        if isinstance(input_households, str):
            self.c_params.input_household_file = input_households
            self.household_df = None
        elif isinstance(input_households, pd.DataFrame):
            self.household_df = input_households
        elif not input_households:
            raise ParameterException("Household data must be supplied as a csv")

        if read_param_file and input_param_file != None:
            self._read_and_check_from_file()

        if output_file_dir:
            self.c_params.sys_write_individual = 1
        self.update_lock = False

    def _read_and_check_from_file(self):
        covid19.read_param_file(self.c_params)

    def _read_household_demographics(self):
        if self.household_df is None:
            self._read_household_demographics_file()
        else:
            self._read_household_demographics_df()

    def _read_household_demographics_file(self):
        """[summary]
        Try to read the reference household demographics file
        If we've not set the number of lines to read, parse the file in
        python and inset the line count to the params structure for initilisation
        """
        if self.get_param("N_REFERENCE_HOUSEHOLDS") != 0:
            covid19.read_household_demographics_file(self.c_params)
        else:
            n_ref_hh = -1
            with open(self.c_params.input_household_file, "r") as f:
                for _ in f.readlines():
                    n_ref_hh += 1
            self.set_param("N_REFERENCE_HOUSEHOLDS", n_ref_hh)
            self._read_household_demographics()

    def _read_household_demographics_df(self):
        """[summary]
        """
        if isinstance(self.household_df, pd.DataFrame):
            self.set_param("N_REFERENCE_HOUSEHOLDS", len(self.household_df))
            LOGGER.debug(
                f"setting up ref household memory for {getattr(self.c_params,'N_REFERENCE_HOUSEHOLDS')}"
            )
            covid19.set_up_reference_household_memory(self.c_params)
            LOGGER.debug("memory set up")
            _ = [
                covid19.add_household_to_ref_households(
                    self.c_params,
                    t[0],
                    t[1],
                    t[2],
                    t[3],
                    t[4],
                    t[5],
                    t[6],
                    t[7],
                    t[8],
                    t[9],
                )
                for t in self.household_df.itertuples()
            ]

    def set_param_dict(self, params):
        for k, v in params.items():
            self.set_param(k, v)

    def get_param(self, param):
        """[summary]
        Get the value of a param from the c_params object
        Arguments:
            param {[str]} -- [name of parameters]

        Raises:
            ParameterException: [description]

        Returns:
            [type] -- [value of param]
        """
        if hasattr(covid19, f"get_param_{param}"):
            return getattr(covid19, f"get_param_{param}")(self.c_params)
        elif hasattr(self.c_params, f"{param}"):
            return getattr(self.c_params, f"{param}")
        else:
            param, idx = _get_base_param_from_enum(param)
            LOGGER.debug(
                f"not found full length param, trying get_param_{param} with index getter"
            )
            if hasattr(covid19, f"get_param_{param}"):
                return getattr(covid19, f"get_param_{param}")(self.c_params, idx)
            else:
                LOGGER.debug(f"Could not find get_param_{param} in covid19 getters")
        raise ParameterException(
            f"Can not get param {param} as it doesn't exist in parameters object"
        )

    def set_param(self, param, value):
        """[summary]
        sets parameter on c_params
        Arguments:
            param {[string]} -- [parameter name]
            value {[float or int]} -- [value]

        Raises:
            ParameterException:
        """
        if self.update_lock:
            raise ParameterException(
                (
                    "Parameter set has been exported to model, "
                    "please use model.update_x functions"
                )
            )

        if hasattr(self.c_params, f"{param}"):
            if isinstance(getattr(self.c_params, f"{param}"), int):
                setattr(self.c_params, f"{param}", int(value))
            if isinstance(getattr(self.c_params, f"{param}"), float):
                setattr(self.c_params, f"{param}", float(value))
        elif hasattr(
                covid19, f"set_param_{_get_base_param_from_enum(param)[0]}"
        ):
            param, idx = _get_base_param_from_enum(param)
            setter = getattr(covid19, f"set_param_{param}")
            setter(self.c_params, value, idx)
        elif hasattr(covid19, f"set_param_{param}"):
            setter = getattr(covid19, f"set_param_{param}")
            setter(self.c_params, value)
        else:
            raise ParameterException(
                f"Can not set parameter {param} as it doesn't exist"
            )

    def return_param_object(self):
        """[summary]
        Run a check on the parameters and return if the c code doesn't bail
        Returns:
            [type] -- [description]
        """
        self._read_household_demographics()
        covid19.check_params(self.c_params)
        LOGGER.info(
            (
                "Returning self.c_params into Model object, "
                "future updates to parameters not possible"
            )
        )
        self.update_lock = True
        return self.c_params


class Model:
    def __init__(self, params_object):
        # Store the params object so it doesn't go out of scope and get freed
        self._params_obj = params_object
        # Create C parameters object
        self.c_params = params_object.return_param_object()
        self.c_model = None
        self._create()
        self._is_running = False

    def __del__(self):
        self._destroy()

    def get_param(self, name):
        """[summary]
        Get parameter by name
        Arguments:
            name {[str]} -- [name of param]

        Raises:
            ModelParameterException: [description]

        Returns:
            [type] -- [value of param stored]
        """
        value = None
        try:
            LOGGER.info(f"Getting param {name}")
            split_param, idx = _get_base_param_from_enum(f"get_model_param_{name}")
            if split_param is not None:
                if hasattr(covid19, split_param):
                    value = getattr(covid19, split_param)(self.c_model, idx)
                    LOGGER.info(f"Got {split_param} at index {idx} value {value}")
                else:
                    raise ModelParameterException(f"Parameter {name} not found")
            elif hasattr(covid19, f"get_model_param_{name}"):
                value = getattr(covid19, f"get_model_param_{name}")(self.c_model)
            else:
                raise ModelParameterException(f"Parameter {name} not found")
            if value < 0:
                return False
            else:
                return value
        except AttributeError:
            raise ModelParameterException(f"Parameter {name} not found")

    def update_running_params(self, param, value):
        """[summary]
        a subset of parameters my be updated whilst the model is evaluating,
        these correspond to events

        Arguments:
            param {[str]} -- [name of parameter]
            value {[type]} -- [value to set]

        Raises:
            ModelParameterException: [description]
            ModelParameterException: [description]
            ModelParameterException: [description]
        """
        if param not in PYTHON_SAFE_UPDATE_PARAMS:
            raise ModelParameterException(f"Can not update {param} during running")
        split_param, index = _get_base_param_from_enum(f"set_model_param_{param}")
        if split_param:
            setter = getattr(covid19, split_param)
        elif hasattr(covid19, f"set_model_param_{param}"):
            setter = getattr(covid19, f"set_model_param_{param}")
        else:
            raise ModelParameterException(f"Setting {param} to {value} failed")
        if callable(setter):
            if index is not None:
                args = [self.c_model, value, index]
            else:
                args = [self.c_model, value]
            LOGGER.info(
                f"Updating running params with {args} split_param {split_param} param {split_param}"
            )
            if not setter(*args):
                raise ModelParameterException(f"Setting {param} to {value} failed")

    def get_risk_score(self, day, age_inf, age_sus):
        value = covid19.get_model_param_risk_score(self.c_model, day, age_inf, age_sus)
        if value < 0:
            raise  ModelParameterException( "Failed to get risk score")
        return value
    
    def get_risk_score_household(self, age_inf, age_sus):
        value = covid19.get_model_param_risk_score_household(self.c_model, age_inf, age_sus)
        if value < 0:
            raise  ModelParameterException( "Failed to get risk score household")
        return value
    
    def set_risk_score(self, day, age_inf, age_sus, value):
        ret = covid19.set_model_param_risk_score(self.c_model, day, age_inf, age_sus, value)
        if ret == 0:
            raise  ModelParameterException( "Failed to set risk score")
    
    def set_risk_score_household(self, age_inf, age_sus, value):
        ret = covid19.set_model_param_risk_score_household(self.c_model, age_inf, age_sus, value)
        if ret == 0:
            raise  ModelParameterException( "Failed to set risk score household")

    def _create(self):
        """
        Call C function new_model (renamed create_model)
        """
        LOGGER.info("Started model creation")
        self.c_model = covid19.create_model(self.c_params)
        LOGGER.info("Successfuly created model")

    def _destroy(self):
        """
        Call C function destroy_model and destroy_params
        """
        LOGGER.info("Destroying model")
        covid19.destroy_model(self.c_model)

    def one_time_step(self):
        """
        Call C function on_time_step
        """
        covid19.one_time_step(self.c_model)

    def one_time_step_results(self):
        """
        Get results from one time step
        """
        results = {}
        results["time"] = self.c_model.time
        results["lockdown"] = self.c_params.lockdown_on
        results["test_on_symptoms"] = self.c_params.test_on_symptoms
        results["app_turned_on"] = self.c_params.app_turned_on
        results["total_infected"] = (
                int(covid19.utils_n_total(self.c_model, covid19.PRESYMPTOMATIC))
                + int(covid19.utils_n_total(self.c_model, covid19.PRESYMPTOMATIC_MILD))
                + int(covid19.utils_n_total(self.c_model, covid19.ASYMPTOMATIC))
        )
        for age in AgeGroupEnum:
            key = f"total_infected{age.name}"
            results[key] = sum(
                [
                    covid19.utils_n_total_age(self.c_model, ty, age.value)
                    for ty in [
                        covid19.PRESYMPTOMATIC,
                        covid19.PRESYMPTOMATIC_MILD,
                        covid19.ASYMPTOMATIC,
                    ]
                ]
            )
        results["total_case"] = covid19.utils_n_total(self.c_model, covid19.CASE)
        for age in AgeGroupEnum:
            key = f"total_case{age.name}"
            value = covid19.utils_n_total_age(self.c_model, covid19.CASE, age.value)
            results[key] = value
        results["total_death"] = covid19.utils_n_total(self.c_model, covid19.DEATH)
        for age in AgeGroupEnum:
            key = f"total_death{age.name}"
            value = covid19.utils_n_total_age(self.c_model, covid19.DEATH, age.value)
            results[key] = value
        results["n_presymptom"] = covid19.utils_n_current(
            self.c_model, covid19.PRESYMPTOMATIC
        ) + covid19.utils_n_current(self.c_model, covid19.PRESYMPTOMATIC_MILD)
        results["n_asymptom"] = covid19.utils_n_current(
            self.c_model, covid19.ASYMPTOMATIC
        )
        results["n_quarantine"] = covid19.utils_n_current(
            self.c_model, covid19.QUARANTINED
        )
        results["n_tests"] = covid19.utils_n_daily(
            self.c_model, covid19.TEST_RESULT, int(self.c_model.time) + 1
        )
        results["n_symptoms"] = covid19.utils_n_current(
            self.c_model, covid19.SYMPTOMATIC
        ) + covid19.utils_n_current(self.c_model, covid19.SYMPTOMATIC_MILD)
        results["n_hospital"] = covid19.utils_n_current( self.c_model, covid19.HOSPITALISED )
        results["n_hospitalised_recovering"] = covid19.utils_n_current( self.c_model, covid19.HOSPITALISED_RECOVERING )
        results["n_critical"] = covid19.utils_n_current(self.c_model, covid19.CRITICAL)
        results["n_death"] = covid19.utils_n_current(self.c_model, covid19.DEATH)
        results["n_recovered"] = covid19.utils_n_current(
            self.c_model, covid19.RECOVERED
        )
        results["hospital_admissions"]  = covid19.utils_n_daily(
            self.c_model, covid19.TRANSITION_TO_HOSPITAL, self.c_model.time
        )
        results["hospital_admissions_total"]  = covid19.utils_n_total(
            self.c_model, covid19.TRANSITION_TO_HOSPITAL
        )
        results["hospital_to_critical_daily"] = covid19.utils_n_daily(
            self.c_model, covid19.TRANSITION_TO_CRITICAL, self.c_model.time
        )
        results["hospital_to_critical_total"] = covid19.utils_n_total(
            self.c_model, covid19.TRANSITION_TO_CRITICAL
        )
        return results

    def write_output_files(self):
        """
        Write output files
        """
        covid19.write_output_files(self.c_model, self.c_params)

    def write_individual_file(self):
        covid19.write_individual_file(self.c_model, self.c_params)

    def write_interactions_file(self):
        covid19.write_interactions(self.c_model)

    def write_trace_tokens_timeseries(self):
        covid19.write_trace_tokens_ts(self.c_model)

    def write_trace_tokens(self):
        covid19.write_trace_tokens(self.c_model)

    def write_transmissions(self):
        covid19.write_transmissions(self.c_model)

    def write_quarantine_reasons(self):
        covid19.write_quarantine_reasons(self.c_model, self.c_params)

    def write_occupation_network(self, idx):
        covid19.write_occupation_network(self.c_model, self.c_params, idx)

    def write_household_network(self):
        covid19.write_household_network(self.c_model, self.c_params)

    def write_random_network(self):
        covid19.write_random_network(self.c_model, self.c_params)

    def print_individual(self, idx):
        covid19.print_individual(self.c_model, idx)
