import subprocess, shutil, os
from os.path import join
import numpy as np
import pandas as pd
from parameters import ParameterSet
from numpy.random import randint
import time

rand_seed = randint(0,4000,1)
rand_seed = set(rand_seed)
# Directories
IBM_DIR = "src"
IBM_DIR_TEST = "src_test"
DATA_DIR_TEST = "data_test"


TEST_DATA_TEMPLATE = "./tests/data/test_parameters.csv"
TEST_DATA_FILE = join(DATA_DIR_TEST, "test_parameters.csv")
TEST_OUTPUT_FILE = join(DATA_DIR_TEST, "test_output.csv")

record_list = []
ii = 0
for seed in rand_seed:
    if ii % 10 == 0:
        print(ii)
    # Construct the executable command
    EXE = "covid19ibm.exe"
    command = join(IBM_DIR_TEST, EXE)

    # Make a temporary copy of the code (remove this temporary directory if it already exists)
    shutil.rmtree(IBM_DIR_TEST, ignore_errors = True)
    shutil.copytree(IBM_DIR, IBM_DIR_TEST)

    file_output = open(TEST_OUTPUT_FILE, "w")
    completed_run = subprocess.run([command, TEST_DATA_FILE], stdout=file_output)

    params = ParameterSet(TEST_DATA_TEMPLATE, line_number = 1)
    params.set_param("rng_seed", seed)
    params.write_params(TEST_DATA_FILE)

    file_output = open(TEST_OUTPUT_FILE, "w")
    completed_run = subprocess.run([command, TEST_DATA_FILE], stdout=file_output)

    df_output = pd.read_csv(TEST_OUTPUT_FILE, comment="#", sep=",")
    record_list.append(df_output.values)
    ii += 1

# out_frame = None
#
# completed_run = subprocess.Popen([command, TEST_DATA_FILE], stdout = subprocess.PIPE)
# while not out_frame:
#     try:
#         out_frame = completed_run.communicate()[0]
#     except:
#         print('still_trying')
#         time.sleep(2)
#         ValueError
# completed_run.stdout.close()
# df_output = pd.read_csv(TEST_OUTPUT_FILE, comment="#", sep=",")
#
#
#
#
# p = subprocess.Popen('./hello', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
# p.stdout.readline() # discard welcome message: "This program gener...
# p.stdout.close()
#
#
# ii = 0
# while ii <5:
#     ii+=1
# print(ii)

# while not(os.path.isfile('./1.txt')):
#     run_file = subprocess.run('./hello')
#
# with open('./1.txt') as input:
#     a = input.read()
# print(a)
#
# os.remove('./1.txt')
# print(os.path.isfile('./1.txt'))
#
# string = 'write this to temporary file'
# with open('./line.txt', 'w') as output:
#     output.write(string)
# run_file = subprocess.Popen('./read_write')
# while not(os.path.isfile('./temp_python.txt')):
#     pass
# # Do some python things pass create temp file for c
# string = 'write this to temporary file'
# with open('./temp_python.txt') as input:
#     a = input.read()
# print(a)
# os.remove('temp_python.txt')
#
# string = 'write this to temporary file'
# with open('./temp_c.txt', 'w') as output:
#      output.write(string)
#
# # Check if process has finished
# if run_file.poll() is not None:
#     print('process has finised')