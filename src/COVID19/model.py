import covid19
import logging

LOGGER = logging.getLogger(__name__)


class ModelParamaterException(Exception):
    pass


class ParamaterException(Exception):
    pass


PYTHON_SAFE_PARAMS = []

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
]


class Parameters(object):
    def __init__(
        self, input_param_file, param_line_number, output_file_dir, input_household_file
    ):
        self.c_params = covid19.parameters()
        self.c_params.input_param_file = input_param_file
        self.c_params.param_line_number = int(param_line_number)
        self.c_params.output_file_dir = output_file_dir
        self.c_params.input_household_file = input_household_file
        self._read_and_check_from_file()
        self.update_lock = False

    def _read_and_check_from_file(self):
        covid19.read_param_file(self.c_params)
        covid19.read_household_demographics_file(self.c_params)

    def get_param(self, param):
        """
        Get parameter from the C paramater structure
        """
        try:
            return getattr(self.c_params, param)
        except AttributeError:
            raise ParamaterException(
                f"Can not get param {param} as it doesn't exist in paramaters object"
            )

    def set_param(self, param, value):
        if self.update_lock:
            raise ParamaterException(
                "Paramater set has been exported to model, please use model.update_x functions"
            )
        if param not in PYTHON_SAFE_PARAMS:
            raise ParamaterException(
                f"Can not set {param} safely from python please change in your input file"
            )
        if hasattr(self.c_params, param, value):
            if isinstance(getattr(self.c_params, param), int):
                setattr(self.c_params, param, int(value))
            elif isinstance(getattr(self.c_params, param), float):
                setattr(self.c_params, param, float(value))
        else:
            raise ParamaterException(f"Can not set paramater as it doesn't exist")

    def return_param_object(self):
        covid19.check_params(self.c_params)
        LOGGER.info(
            "Returning self.c_params into Model object, future updates to paramaters not possible"
        )
        self.update_lock = True
        return self.c_params


class Model:
    def __init__(self, params_object):
        # Create C parameters object
        self.c_params = params_object.return_param_object()
        self.create()
        self._is_running = False

    def get_param(self, name):
        """
        Get parameter from the C structure
        """
        try:
            if isinstance(getattr(self.c_params, name), int):
                value = covid19.get_param_int(self.c_model, name)
                if value < 0:
                    return False
                else:
                    return value
            elif isinstance(getattr(self.c_params, name), float):
                value = covid19.get_param_double(self.c_model, name)
                if value < 0:
                    return False
                else:
                    return value
        except AttributeError:
            raise ModelParamaterException("Parameter {param} not found")

    def update_running_params(self, param, value):
        if param not in PYTHON_SAFE_UPDATE_PARAMS:
            raise ModelParamaterException(f"Can not update {param} during running")
        # if not hasattr(self.c_model, param):
        #     raise ModelParamaterException(
        #         f"Can not set param {param} as it doesn't exist"
        #     )
        if not covid19.set_param(self.c_model, param, f"{value}"):
            raise ModelParamaterException(f"Setting {param} to {value} failed")

    def create(self):
        """
        Call C function new_model (renamed create_model)
        """
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
        results["social_distancing"] = self.c_params.social_distancing_on
        results["test_on_symptoms"] = self.c_params.test_on_symptoms
        results["app_turned_on"] = self.c_params.app_turned_on
        results["total_infected"] = int(
            covid19.util_n_total(self.c_model, covid19.PRESYMPTOMATIC)
        ) + int(covid19.util_n_total(self.c_model, covid19.ASYMPTOMATIC))
        results["total_case"] = covid19.util_n_total(self.c_model, covid19.CASE)
        results["n_presymptom"] = covid19.util_n_current(
            self.c_model, covid19.PRESYMPTOMATIC
        )
        results["n_asymptom"] = covid19.util_n_current(
            self.c_model, covid19.ASYMPTOMATIC
        )
        results["n_quarantine"] = covid19.util_n_current(
            self.c_model, covid19.QUARANTINED
        )
        results["n_tests"] = covid19.util_n_daily(
            self.c_model, covid19.TEST_RESULT, int(self.c_model.time) + 1
        )
        results["n_sysmptoms"] = covid19.util_n_current(
            self.c_model, covid19.SYMPTOMATIC
        )
        results["n_hospital"] = covid19.util_n_current(
            self.c_model, covid19.HOSPITALISED
        )
        results["n_critical"] = covid19.util_n_current(self.c_model, covid19.CRITICAL)
        results["n_death"] = covid19.util_n_current(self.c_model, covid19.DEATH)
        results["n_recovered"] = covid19.util_n_current(self.c_model, covid19.RECOVERED)
        return results

    def write_output_files(self):
        """
        Write output files
        """
        covid19.write_output_files(self.c_model, self.c_params)

    def __del__(self):
        """
        Call C functions destroy_model and destroy_params
        """
        if self.c_model:
            covid19.destroy_model(self.c_model)
        if self.c_params:
            covid19.destroy_params(self.c_params)
