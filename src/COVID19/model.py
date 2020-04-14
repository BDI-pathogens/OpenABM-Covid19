import covid19
import logging
import enum
from itertools import chain

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
    "quarantine_household_on_traced",
    "quarantine_household_contacts_on_positive",
    "quarantine_days",
    "test_order_wait",
    "test_result_wait",
    "self_quarantine_fraction",
    "lockdown_on",
    "app_turned_on",
    "app_users_fraction",
]


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
    _workplace = 1
    _random = 2


class Parameters(object):
    def __init__(
        self,
        input_param_file: str = None,
        param_line_number: int = 1,
        output_file_dir: str = "./",
        input_household_file: str = None,
        read_param_file=True,
    ):
        """[summary]
        
        Arguments:
            object {[type]} -- [description]
        
        Keyword Arguments:
            input_param_file {str} -- [Parameters file path] (default: {None})
            param_line_number {int} -- [Which column of the input param file to read] (default: {1})
            output_file_dir {str} -- [Where to write output files to] (default: {"./"})
            input_household_file {str} -- [Household demographics file (required)] (default: {None})
            read_param_file {bool} -- [Read param file, all params can be set from python interface] (default: {True})
        
        Raises:
            ParameterException: [Warnings if parameters are not correctly set]
            Sys.exit(0): [Underlaying C code will exist if params are not viable]
        """
        self.c_params = covid19.parameters()
        if input_param_file:
            self.c_params.input_param_file = input_param_file
        elif not input_param_file and self.read_param_file:
            raise ParameterException("Input param path is None and read param file set to true")
        else:
            LOGGER.info(
                "Have not passed input file for params, use set_param or set_param_dict"
            )

        if param_line_number:
            self.c_params.param_line_number = int(param_line_number)
        self.c_params.output_file_dir = output_file_dir
        if not input_household_file:
            raise ParamaterException("Household data must be supplied as a csv")
        self.c_params.input_household_file = input_household_file
        if read_param_file and input_param_file != None:
            self._read_and_check_from_file()
        self.update_lock = False

    def _read_and_check_from_file(self):
        covid19.read_param_file(self.c_params)

    def _read_household_demographics(self):
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

    def set_param_dict(self, params):
        for k, v in params.items():
            self.set_param(k, v)

    def _get_base_param_from_age_param(self, param):
        base_name, enum_val = None, None
        for en in chain(
            AgeGroupEnum, ChildAdultElderlyEnum, ListIndiciesEnum, TransmissionTypeEnum
        ):
            if en.name == param[-1 * len(en.name) :]:
                base_name = param.split(en.name)[0]
                enum_val = en.value
                break
        return base_name, enum_val

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
        elif hasattr(
            covid19, f"get_param_{self._get_base_param_from_age_param(param)[0]}"
        ):
            param, idx = self._get_base_param_from_age_param(param)
            return getattr(covid19, f"get_param_{param}")(self.c_params, idx)
        elif hasattr(self.c_params, f"{param}"):
            return getattr(self.c_params, f"{param}")
        else:
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
                f"Parameter set has been exported to model, please use model.update_x functions"
            )

        if hasattr(self.c_params, f"{param}"):
            if isinstance(getattr(self.c_params, f"{param}"), int):
                setattr(self.c_params, f"{param}", int(value))
            if isinstance(getattr(self.c_params, f"{param}"), float):
                setattr(self.c_params, f"{param}", float(value))
            else:
                LOGGER.info(
                    f'param {param} has type {type(getattr(self.c_params, f"{param}"))}'
                )
        elif hasattr(
            covid19, f"set_param_{self._get_base_param_from_age_param(param)[0]}"
        ):
            param, idx = self._get_base_param_from_age_param(param)
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
            "Returning self.c_params into Model object, future updates to parameters not possible"
        )
        self.update_lock = True
        return self.c_params


class Model:
    def __init__(self, params_object):
        # Store the params object so it doesn't go out of scope and get freed
        self._params_obj = params_object
        # Create C parameters object
        self.c_model = None
        self.c_params = params_object.return_param_object()
        self._create()
        self._is_running = False

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
        try:
            LOGGER.info(f"Getting param {name}")
            value = getattr(covid19, f"get_model_param_{name}")(self.c_model)
            if value < 0:
                return False
            else:
                return value
        except AttributeError:
            raise ModelParameterException("Parameter {param} not found")

    def update_running_params(self, param, value):
        """[summary]
        a subset of parameters my be updated whilst the model is evaluating, these correspond to events 
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
        setter = getattr(covid19, f"set_model_param_{param}")
        if callable(setter):
            if not setter(self.c_model, value):
                raise ModelParameterException(f"Setting {param} to {value} failed")
        else:
            raise ModelParameterException(f"Setting {param} to {value} failed")

    def _create(self):
        """
        Call C function new_model (renamed create_model)
        """
        LOGGER.info("Started model creation")
        self.c_model = covid19.create_model(self.c_params)
        LOGGER.info("Successfuly created model")

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
        results["total_case"] = covid19.utils_n_total(self.c_model, covid19.CASE)
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
        results["n_hospital"] = covid19.utils_n_current(
            self.c_model, covid19.HOSPITALISED
        )
        results["n_critical"] = covid19.utils_n_current(self.c_model, covid19.CRITICAL)
        results["n_death"] = covid19.utils_n_current(self.c_model, covid19.DEATH)
        results["n_recovered"] = covid19.utils_n_current(
            self.c_model, covid19.RECOVERED
        )
        return results

    def write_output_files(self):
        """
        Write output files
        """
        covid19.write_output_files(self.c_model, self.c_params)
