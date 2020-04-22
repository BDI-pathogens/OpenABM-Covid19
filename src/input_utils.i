%module input_utils

%include <pybuffer.i>

%pybuffer_string(char *buf);
%inline %{
void utils_update_param_from_buffer(parameters *params, char *buf) {
    update_param_from_buffer(params, buf);
}

void utils_update_household_demographics_param_from_buffer(parameters *params, char *buf) {
    update_household_demographics_param_from_buffer(params, buf);
}
%}
