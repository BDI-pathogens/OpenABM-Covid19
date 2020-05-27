#!/usr/bin/env python3
"""
Running the individual-based model, COVID19-IBM

Usage:
python3 run_simulation.py  

Created: May 2020
Author: Dylan Feldner-Busztin
"""

import subprocess, pytest, os, sys
import numpy as np, pandas as pd
from . import constant

print(constant.EVENT_TYPES.ICU)
