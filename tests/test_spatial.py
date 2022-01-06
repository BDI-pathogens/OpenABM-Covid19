from pandas.core.frame import DataFrame
import pytest, subprocess
import numpy as np
import pandas as pd

import sys
sys.path.append("src/COVID19")
from parameters import ParameterSet

from . import utilities as utils
from . import constant

# Fix random seed
np.random.seed(0)

def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])
    metafunc.parametrize(
        argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
    )

class TestClass(object):
    params = {
        "test_coordinate_intial" : [dict()],
        "test_coordinate_bespoke": [
            dict(
                df_coords = pd.DataFrame(
                    np.array(
                        [np.arange(10000),
                        np.arange(10000),
                        np.arange(10000)]
                    ).T,
                    columns= ['ID','xcoords','ycoords']
                )
            ),
            dict(
                df_coords = pd.DataFrame(
                    np.array(
                        [np.arange(10000),
                        np.random.randint(10000,size=(10000)),
                        np.random.randint(10000,size=(10000))]
                    ).T,
                    columns= ['ID','xcoords','ycoords']
                )
            ),
            # Include negative values (statistically)
            dict(
                df_coords = pd.DataFrame(
                    np.array(
                        [np.arange(10000),
                        np.random.randint(10000,size=(10000))-5000,
                        np.random.randint(10000,size=(10000))-5000]
                    ).T,
                    columns= ['ID','xcoords','ycoords']
                )
            )
        ]
    }

    def test_coordinate_intial(self):
        """
            Ensure the default value is same as INITIAL_COORDINATE_X/Y
        """
        params = ParameterSet(constant.TEST_DATA_FILE, line_number = 1) 
        params.set_param("n_total", 10000)
        params.set_param("end_time", 0)
        params.write_params(constant.TEST_DATA_FILE)     

        file_output   = open(constant.TEST_OUTPUT_FILE, "w")
        completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)  

        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE, 
            comment="#", sep=",", skipinitialspace = True)

        np.testing.assert_equal([True],['xcoords' in df_indiv.keys()],err_msg="Cannot find 'xcoords' coordinate column in "+constant.TEST_INDIVIDUAL_FILE)
        np.testing.assert_equal([True],['ycoords' in df_indiv.keys()],err_msg="Cannot find 'ycoords' coordinate column in "+constant.TEST_INDIVIDUAL_FILE)
        
        np.testing.assert_array_equal(df_indiv['xcoords'],constant.INITIAL_COORDINATE_X*np.ones(len(df_indiv['xcoords'])),err_msg="Default 'xcoords' values for all individuals not observed.")
        np.testing.assert_array_equal(df_indiv['ycoords'],constant.INITIAL_COORDINATE_Y*np.ones(len(df_indiv['ycoords'])),err_msg="Default 'ycoords' values for all individuals not observed.")

    def test_coordinate_bespoke(self,df_coords,n=128):
        """
            Check whether agents are assigned their coordinates, and kept after simulation
        """
        params = utils.get_params_swig()
        n_total = len(df_coords["ID"])
        params.set_param("n_total", 10000)
        params.set_param("end_time", 128)
        model = utils.get_model_swig( params )
        model.assign_coordinates_individuals(df_coords)

        model.write_individual_file()

        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE, 
            comment="#", sep=",", skipinitialspace = True)

        np.testing.assert_array_equal(df_indiv['xcoords'],df_coords['xcoords'],err_msg="Individual's xcoords do not match prescribed values after their assignment.")
        np.testing.assert_array_equal(df_indiv['ycoords'],df_coords['ycoords'],err_msg="Individual's ycoords do not match prescribed values after their assignment.")

        for t in range(n):
            model.one_time_step()

        model.write_individual_file()

        df_indiv = pd.read_csv(constant.TEST_INDIVIDUAL_FILE, 
            comment="#", sep=",", skipinitialspace = True)

        np.testing.assert_array_equal(df_indiv['xcoords'],df_coords['xcoords'],err_msg="Individual's xcoords do not match prescribed values after simulation steps.")
        np.testing.assert_array_equal(df_indiv['ycoords'],df_coords['ycoords'],err_msg="Individual's ycoords do not match prescribed values after simulation steps.")






