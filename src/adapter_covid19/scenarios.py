from __future__ import annotations

from adapter_covid19.data_structures import Scenario, ModelParams

"""
========
Timeline
========

The lockdown in the UK started on 23 March 2020. The strict lockdown ended on 10 May 2020 (inclusive). 
Convention for all scenarios below is that the start time (time 0) is 10 days prior to the start of the
lockdown, i.e., 13 March 2020. (When this convention is changed it should be changed for all scenarios 
consistently.)

Key events on the timeline are

* 13 March - Time 0 - Start of Simulation
* 23 March - Time 10 
    - Start of Lockdown (inclusive)
    - Start of `CCFF <https://www.bankofengland.co.uk/news/2020/march/the-covid-corporate-financing-facility>`_. 
* 31 March - Time 18 - Last day of Q1
* 1 April - Time 19 - First day of Q2
* 11 May - Time 59 - End of Lockdown (exclusive)
* 30 June - Time 109 - Last day of Q2
* 1 July - Time 110 - First day of Q3
* 30 September - Time 201 - Last day of Q3

TODO:
* Confirm start date for furlough scheme
* Confirm start date for new spending
* Confirm start date for loan guarantees

=======================
Institutional Forecasts
=======================

Collecting some key figures from third party publications for reference.

BoE Monetary Policy Report May 2020
-----------------------------------

https://www.bankofengland.co.uk/-/media/boe/files/monetary-policy-report/2020/may/monetary-policy-report-may-2020
 
Key figures in "illustrative scenario": 

* GDP
    * Annual 2020: -14%
    * Q1: -3%
    * Q2: -25%
* Consumption 
    * Household consumption (in March-April): -30%
    * Sales (Q2): -40% to -45%
    * Business Investment (Q2): -40% to -50%


==========
Next Steps
==========

TODO:

* Demand modelling
    * Reduction in business investment
    * Waterfall of demand reduction (reduce expenditure on discretionary items first)
    * Changes of demand imposed by lockdown
        - Constraints imposed by lockdown (e.g. can't go to restaurant) 
        - Substitution of demand (e.g. more communication) 
    * Shift demand of ill individuals toward healthcare
    * Deceased individuals no longer contribute to demand
    * Reduced consumer confidence even when lockdown is lifted (possibly correlated with trajectory of epidemic)
    * Increased savings rate

"""


BASIC_MODEL_PARAMS = ModelParams(
    economics_params={},
    gdp_params={},
    personal_params={
        "default_th": 300,
        "max_earning_furloughed": 30_000,
        "alpha": 5,
        "beta": 20,
    },
    corporate_params={"beta": 1.4, "large_cap_cash_surplus_months": 6,},
)

# Basic Scenario (aligned with actual interventions)
# * Lockdown
# * Furlough
# * Corporate Support

BASIC_SCENARIO = Scenario(
    lockdown_recovery_time=1,
    lockdown_start_time=10,
    lockdown_end_time=59,
    furlough_start_time=10,
    furlough_end_time=202,
    simulation_end_time=202,
    new_spending_day=10,
    ccff_day=10,
    loan_guarantee_day=10,
    model_params=BASIC_MODEL_PARAMS,
)

# Basic No Furlough Scenario
# * Lockdown
# * No Furlough
# * Corporate Support

BASIC_NO_FURLOUGH_SCENARIO = Scenario(
    lockdown_recovery_time=1,
    lockdown_start_time=10,
    lockdown_end_time=59,
    furlough_start_time=10000,
    furlough_end_time=10000,
    simulation_end_time=202,
    new_spending_day=10,
    ccff_day=10,
    loan_guarantee_day=10,
    model_params=BASIC_MODEL_PARAMS,
)

# Basic No Corp Support Scenario
# * Lockdown
# * Furlough
# * No Corporate Support

BASIC_NO_CORP_SUPPORT_SCENARIO = Scenario(
    lockdown_recovery_time=1,
    lockdown_start_time=10,
    lockdown_end_time=59,
    furlough_start_time=10,
    furlough_end_time=202,
    simulation_end_time=202,
    new_spending_day=10000,
    ccff_day=10000,
    loan_guarantee_day=10000,
    model_params=BASIC_MODEL_PARAMS,
)

# Basic No Furlough No Corp Support Scenario
# * Lockdown
# * No Furlough
# * No Corporate Support

BASIC_NO_FURLOUGH_NO_CORP_SUPPORT_SCENARIO = Scenario(
    lockdown_recovery_time=1,
    lockdown_start_time=10,
    lockdown_end_time=59,
    furlough_start_time=10000,
    furlough_end_time=10000,
    simulation_end_time=202,
    new_spending_day=10000,
    ccff_day=10000,
    loan_guarantee_day=10000,
    model_params=BASIC_MODEL_PARAMS,
)

# Basic No Lockdown Scenario
# * No Lockdown
# * No Furlough
# * No Corporate Support

BASIC_NO_LOCKDOWN_SCENARIO = Scenario(
    lockdown_recovery_time=1,
    lockdown_start_time=10000,
    lockdown_end_time=10000,
    furlough_start_time=10000,
    furlough_end_time=10000,
    simulation_end_time=202,
    new_spending_day=10000,
    ccff_day=10000,
    loan_guarantee_day=10000,
    model_params=BASIC_MODEL_PARAMS,
)


