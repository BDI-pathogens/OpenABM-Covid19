import sys
from parameters import ParameterSet
sys.path.append("../src/")
import covid19

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
        
        # Get more params and check them
        covid19.read_param_file(self.c_params) 
        covid19.check_params(self.c_params)
        covid19.read_household_demographics_file(self.c_params)

    def get_param(self, param):
        #TODO
        pass

    def set_param(self, param, value):
        #TODO
        pass
        self.c_params.test_on_symptoms = int(value)
        self.c_params.test_on_traced = int(value)
        self.c_params.quarantine_on_traced = int(value)
        self.c_params.traceable_interaction_fraction = float(value)
        self.c_params.tracing_network_depth = int(value)
        self.c_params.allow_clinical_diagnosis = int(value)
        self.c_params.quarantine_household_on_symptoms = int(value)
        self.c_params.quarantine_household_on_positive = int(value)
        self.c_params.quarantine_household_on_traced = int(value)
        self.c_params.quarantine_household_contacts_on_positive = int(value)
        self.c_params.quarantine_days = int(value)
        self.c_params.test_order_wait = int(value)
        self.c_params.test_result_wait = int(value)
        self.c_params.self_quarantine_fraction = float(value)

    def create(self):
        """
        Call C function new_model
        """
        self.model = covid19.create_model(self.c_params)
        print("param_id: {}".format(self.c_params.param_id))
        print("rng_seed: {}".format(self.c_params.rng_seed))
        print("param_line_number: {}".format(self.c_params.param_line_number))

        return self.model

    def one_time_step(self, model):
        """
        Call C function on_time_step
        """
        covid19.one_time_step(model)
        covid19.write_output_files(self.model, self.c_params)

    def destroy(self, model):
        """
        Call C function destroy_model
        """
        covid19.destroy_model(model)
        covid19.destroy_params(self.c_params)


INPUT_PARAM_FILE = "data/baseline_parameters.csv"
PARAM_LINE_NUMBER = 1
INPUT_HOUSEHOLD_FILE = "data/baseline_household_demographics.csv"
OUTPUT_FILE_DIR = "."

model = Model(INPUT_PARAM_FILE,
              PARAM_LINE_NUMBER,
              OUTPUT_FILE_DIR,
              INPUT_HOUSEHOLD_FILE)
step_model = model.create()
model.destroy(step_model)
