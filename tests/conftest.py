# content of conftest.py

import pytest, subprocess, shutil, os, sys
from . import constant

sys.path.append("src/COVID19")
from parameters import ParameterSet


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

# "Session" (all files in folder) setup/teardown
@pytest.fixture(scope = "session", autouse = True)
def compile_covid_ibm(request):
    """
    Compile the IBM in a temporary directory
    """
    # Make a temporary copy of the code 
    # (remove this temporary directory if it already exists)
    shutil.rmtree(constant.IBM_DIR_TEST, ignore_errors = True)
    shutil.copytree(constant.IBM_DIR, constant.IBM_DIR_TEST)

    # Construct the compilation command and compile
    compile_command = "make clean; make install"
    completed_compilation = subprocess.run(
        [compile_command], shell = True, cwd = constant.IBM_DIR_TEST, capture_output = True
    )
    def fin():
        # Teardown: remove the temporary code directory (when this class is removed)
        shutil.rmtree(constant.IBM_DIR_TEST, ignore_errors = True)
    request.addfinalizer(fin)


# Method ("function") setup/teardown
@pytest.fixture(scope = "function", autouse = True)
def setup_covid_methods(request):
    """
    Called before each method is run; creates a new data dir, copies test datasets
    """
    os.mkdir(constant.DATA_DIR_TEST)
    shutil.copy(constant.TEST_DATA_TEMPLATE, constant.TEST_DATA_FILE)
    shutil.copy(constant.TEST_HOUSEHOLD_TEMPLATE, constant.TEST_HOUSEHOLD_FILE)
    shutil.copy(constant.TEST_HOSPITAL_TEMPLATE, constant.TEST_HOSPITAL_FILE)

    # Adjust any parameters that need adjusting for all tests
    params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
    params.set_param("n_total", 10000)
    params.set_param("end_time", 100)
    params.write_params(constant.TEST_DATA_FILE)
    def fin():
        """
        At the end of each method (test), remove the directory of test input/output data
        """
        shutil.rmtree(constant.DATA_DIR_TEST, ignore_errors=True)
    request.addfinalizer(fin)
