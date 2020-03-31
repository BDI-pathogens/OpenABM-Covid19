import .covid19

class Model:
    def __init__(self,
                 input_param_file,
                 param_line_number,
                 output_file_dir,
                 input_household_file):
        # Create C parameters object
        self.c_params = covid19.parameters()
        self.c_params.input_param_file = input_param_file
        self.c_params.param_line_number = int(param_line_number)
        self.c_params.output_file_dir = output_file_dir
        self.c_params.sys_write_individual = True
        self.c_params.input_household_file = input_household_file
        
        # Get params from file and check them
        covid19.read_param_file(self.c_params) 
        covid19.check_params(self.c_params)
        covid19.read_household_demographics_file(self.c_params)

    def get_param(self, name):
        """
        Get parameter from the C structure
        """
        try:
            return getattr(self.c_params, name)
        except AttributeError:
            print("Parameter not found")
            return None

    def set_param(self, name, value):
        """
        Set parameter in the C structure
        """
        try:
            if isinstance(getattr(self.c_params, name), int):
                setattr(self.c_params, name, int(value))
            elif isinstance(getattr(self.c_params, name), float):
                setattr(self.c_params, name, float(value))
        except AttributeError:
            print("Parameter not found")
            return None 

    def create(self):
        """
        Call C function new_model (renamed create_model)
        """
        self.c_model = covid19.create_model(self.c_params)

        return self.c_model

    def one_time_step(self, c_model=None):
        """
        Call C function on_time_step
        """
        if not c_model:
            c_model = self.c_model
        covid19.one_time_step(c_model)

    def one_time_step_results(self, c_model=None):
        """
        Get results from one time step
        """
        if not c_model:
            c_model = self.c_model
        results = {}
        results['time'] = c_model.time
        results['social_distancing'] = self.c_params.social_distancing_on
        results['test_on_symptoms'] = self.c_params.test_on_symptoms
        results['app_turned_on'] = self.c_params.app_turned_on
        results['total_infected'] = int(covid19.util_n_total(c_model, covid19.PRESYMPTOMATIC)) + int(covid19.util_n_total(c_model, covid19.ASYMPTOMATIC))
        results['total_case'] = covid19.util_n_total(c_model, covid19.CASE)
        results['n_presymptom'] = covid19.util_n_current(c_model, covid19.PRESYMPTOMATIC)
        results['n_asymptom'] = covid19.util_n_current(c_model, covid19.ASYMPTOMATIC)
        results['n_quarantine'] = covid19.util_n_current(c_model, covid19.QUARANTINED)
        results['n_tests'] = covid19.util_n_daily(c_model, covid19.TEST_RESULT, int(c_model.time) + 1)
        results['n_sysmptoms'] = covid19.util_n_current(c_model, covid19.SYMPTOMATIC)
        results['n_hospital'] = covid19.util_n_current(c_model, covid19.HOSPITALISED)
        results['n_critical'] = covid19.util_n_current(c_model, covid19.CRITICAL)
        results['n_death'] = covid19.util_n_current(c_model, covid19.DEATH)
        results['n_recovered'] = covid19.util_n_current(c_model, covid19.RECOVERED)

        return results

    def write_output_files(self):
        """
        Write output files
        """
        covid19.write_output_files(self.c_model, self.c_params)

    def destroy(self, c_model):
        """
        Call C functions destroy_model and destroy_params
        """
        covid19.destroy_model(c_model)
        covid19.destroy_params(self.c_params)

    def __del__(self):
        covid19.destroy_model(self.c_model)
        covid19.destroy_params(self.c_params)
