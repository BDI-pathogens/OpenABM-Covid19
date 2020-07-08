# Table: Output file dictionary
| Column name | Description | File type | 
|  ---- | ---- | ---- |
| `time` | Day number of the simulation | timeseries |
| `lockdown` | Is lockdown currently on (1 = Yes; 0 = No) | timeseries |
| `test_on_symptoms` | Is testing on symptoms currently on (1=Yes; 0=No) | timeseries |
| `app_turned_on` | Is the app currently on  (1= Yes; 0 = No) | timeseries |
| `total_infected` | Cumulative infected | timeseries |
| `total_infected_0_9` | Cumulative infected aged 0-9 years | timeseries |
| `total_infected_10_19` | Cumulative infected aged 10-19 years | timeseries |
| `total_infected_20_29` | Cumulative infected aged 20-29 years | timeseries |
| `total_infected_30_39` | Cumulative infected aged 30-39 years | timeseries |
| `total_infected_40_49` | Cumulative infected aged 40-49 years | timeseries |
| `total_infected_50_59` | Cumulative infected aged 50-59 years | timeseries |
| `total_infected_60_69` | Cumulative infected aged 60-69 years | timeseries |
| `total_infected_70_79` | Cumulative infected aged 70-79 years | timeseries |
| `total_infected_80` | Cumulative infected aged 80+ years | timeseries |
| `total_case` | Cumulative cases (a case is defined by a positive test result) | timeseries |
| `total_case_0_9` | Cumulative cases aged 0-9 years | timeseries |
| `total_case_10_19` | Cumulative cases aged 10-19 years | timeseries |
| `total_case_20_29` | Cumulative cases aged 20-29 years | timeseries |
| `total_case_30_39` | Cumulative cases aged 30-39 years | timeseries |
| `total_case_40_49` | Cumulative cases aged 40-49 years | timeseries |
| `total_case_50_59` | Cumulative cases aged 50-59 years | timeseries |
| `total_case_60_69` | Cumulative cases aged 60-69 years | timeseries |
| `total_case_70_79` | Cumulative cases aged 70-79 years | timeseries |
| `total_case_80` | Cumulative cases aged 80+ years | timeseries |
| `total_death` | Cumulative deaths (where COVID19 is the primary cause of death) | timeseries |
| `total_death_0_9` | Cumulative deaths aged 0-9 years | timeseries |
| `total_death_10_19` | Cumulative deaths aged 10-19 years | timeseries |
| `total_death_20_29` | Cumulative deaths aged 20-29 years | timeseries |
| `total_death_30_39` | Cumulative deaths aged 30-39 years | timeseries |
| `total_death_40_49` | Cumulative deaths aged 40-49 years | timeseries |
| `total_death_50_59` | Cumulative deaths aged 50-59 years | timeseries |
| `total_death_60_69` | Cumulative deaths aged 60-69 years | timeseries |
| `total_death_70_79` | Cumulative deaths aged 70-79 years | timeseries |
| `total_death_80` | Cumulative deaths aged 80+ years | timeseries |
| `n_presymptom` | Current number presymptomatic (both mild and severe) | timeseries |
| `n_asymptom` | Current number asymptomatic | timeseries |
| `n_quarantine` | Current number in quarantine | timeseries |
| `n_tests` | Current number of tests reporting results today | timeseries |
| `n_symptoms` | Current number of symptomatic (both mild and severe) | timeseries |
| `n_hospital` | Current number in hospital who have not yet required critical care | timeseries |
| `n_hospitalised_recovering` | Current number in hospital who left critical care but not been discharged | timeseries |
| `n_critical` | Current number in critical care | timeseries |
| `n_death` | Daily number of deaths | timeseries |
| `n_recovered` | Cumulative number recovered | timeseries |
| `hospital_admissions` | Daily hospital admissions | timeseries |
| `hospital_admissions_total` | Cumulative hospital admissions | timeseries |
| `hospital_to_critical_daily` | Daily transitions from hospital to critical | timeseries |
| `hospital_to_critical_total` | Cumulative transitions from hospital to critical | timeseries |
| `n_quarantine_infected` | Current number in quarantine that have ever been infected | timeseries |
| `n_quarantine_recovered` | Current number in quarantine that are recovered | timeseries |
| `n_quarantine_app_user` | Current number of app users in quarantine | timeseries |
| `n_quarantine_app_user_infected` | Current number of app users in quarantine that have ever been infected | timeseries |
| `n_quarantine_app_user_recovered` | Current number of app users in quarantine that are recovered | timeseries |
| `n_quarantine_events` | Daily number of quarantine events | timeseries |
| `n_quarantine_release_events` | Daily number of quarantine release events | timeseries |
| `n_quarantine_events_app_user` | Daily number of quarantine events of app users | timeseries |
| `n_quarantine_release_events_app_user` | Daily number of quarantine release events of app users | timeseries |
| `ID` | Unique identifier of the individual | individual file |
| `current_status` | Disease status of the individual at the point at which the individual file is written to file.  See the transmission file for the status of an individual through time.  Note that this is a variable that may change throughout the simulation and so may be removed from this file in the future.   | individual file |
| `age_group` | Age group of the individual | individual file |
| `occupation_network` | Occupation network to which this individual has membership (coded by the `enum OCCUPATION_NETWORKS` enum in constant.h) | individual file |
| `worker_type` | Type of hospital worker (coded by the `enum WORKER_TYPES` in constant.h) (default -1) | individual file |
| `assigned_worker_ward_type` | Type of ward within which this individual works (coded by the `enum HOSPITAL_WARD_TYPES` in constant.h) (default -1) | individual file |
| `house_no` | Household identifier to which this individual belongs | individual file |
| `quarantined` | Is the individual currently quarantined (1=Yes; 0=No).  See the quarantine reasons at each time step for a complete list of quarantined individuals through time.  Note that this is a variable that may change throughout the simulation and so may be removed from this file in the future.   | individual file |
| `time_quarantined` | Time at which the individual was quarantined if they are currently quarantined.  Note that this is a variable that may change throughout the simulation and so may be removed from this file in the future.   | individual file |
| `test_status` | `quarantine_test_result` attribute of the individual struct.  Currently mainly used for testing purposes.  Takes the following values: -2: the individual is not currently being tested; -1 (-3): a (priority) test has been ordered; 0: the individual is waiting for a result which will be negative; 1: the individual is waiting for a test result which will be positive; note that regardless of the result once the test result has been received this variables returns to not current being tested (-2) | individual file |
| `app_user` | Is this individual an app user  (1=Yes; 0=No) | individual file |
| `mean_interactions` | Number of random daily interactions of the individual (the random_interactions attribute of the individual struct) | individual file |
| `infection_count` | Number of times this individual has been infected with SARS-CoV-2 | individual file |
| `ID_recipient` | Unique identifier of the recipient | transmission file |
| `age_group_recipient` | Age group of the recipient (coded by the `enum AGE_GROUPS` in constant.h) | transmission file |
| `house_no_recipient` | Household identifier of the recipient | transmission file |
| `occupation_network_recipient` | Occupation network of the recipient (coded by the `enum OCCUPATION_NETWORKS` within constant.h) | transmission file |
| `worker_type_recipient` | Type of hospital worker of the recipient (coded by the `enum WORKER_TYPES` in constant.h) (default -1) | transmission file |
| `hospital_state_recipient` | Hospital status of the recipient at time of transmission (coded by `enum EVENT_TYPES` in constant.h) (default NOT_IN_HOSPITAL) | transmission file |
| `infector_network` | Network within which the transmission took place (coded by the `enum INTERACTION_TYPE` within constant.h) | transmission file |
| `generation_time` | Generation time of this transmission event (days; time from infection of the source to transmission) (0 for seed cases) | transmission file |
| `ID_source` | Unique identifier of the source (same as ID_recipient for seed cases) | transmission file |
| `age_group_source` | Age group of the source (coded by the `enum AGE_GROUPS` in constant.h) | transmission file |
| `house_no_source` | Household identifier of the source | transmission file |
| `occupation_network_source` | Occupation network of the source (coded by the `enum OCCUPATION_NETWORKS` within constant.h) | transmission file |
| `worker_type_source` | Type of hospital worker of the source at time of transmission (coded by the `enum WORKER_TYPES` in constant.h) (default -1) | transmission file |
| `hospital_state_source` | Hospital status of the source (coded by `enum EVENT_TYPES` in constant.h) (default NOT_IN_HOSPITAL) | transmission file |
| `time_infected_source` | Time when source was infected | transmission file |
| `status_source` | Infectious status of the source at time of transmission (coded by `enum EVENT_TYPES` within constant.h) | transmission file |
| `time_infected` | Time at which transmission took place (time measured as day number of the simulation) | transmission file |
| `time_presymptomatic` | Time at which the recipient became presymptomatic (-1 if never) | transmission file |
| `time_presymptomatic_mild` | Time at which the recipient became presymptomatic (if mildly infected) (-1 if never) | transmission file |
| `time_presymptomatic_severe` | Time at which the recipient became presymptomatic (if severely infected) (-1 if never) | transmission file |
| `time_symptomatic` | Time at which the recipient became symptomatic (-1 if never) | transmission file |
| `time_symptomatic_mild` | Time at which the recipient became symptomatic (if mildly infected) (-1 if never) | transmission file |
| `time_symptomatic_severe` | Time at which the recipient became symptomatic (if severely infected) (-1 if never) | transmission file |
| `time_asymptomatic` | Time at which the recipient became asymptomatic (-1 if never) | transmission file |
| `time_hospitalised` | Time at which the recipient became hospitalised (-1 if never) | transmission file |
| `time_critical` | Time at which the recipient became critical (-1 if never) | transmission file |
| `time_hospitalised_recovering` | Time at which the recipient was discharged from critical but remained in hospital (-1 if never) | transmission file |
| `time_death` | Time at which the recipient died (-1 if never) | transmission file |
| `time_recovered` | Time at which the recipient recovered (-1 if never) | transmission file |
| `time_susceptible` | Time at which the recipient became susceptible again (if waning immunity is possible) | transmission file |
| `is_case` | Was the recipient identified as a case (positive test result) (1=Yes; 0=No) | transmission file |