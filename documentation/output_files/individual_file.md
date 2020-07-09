# Table: individual file
| Column name | Description | 
|  ---- | ---- |
| `ID` | Unique identifier of the individual |
| `current_status` | Disease status of the individual at the point at which the individual file is written to file.  See the transmission file for the status of an individual through time.  Note that this is a variable that may change throughout the simulation and so may be removed from this file in the future.   |
| `age_group` | Age group of the individual |
| `occupation_network` | Occupation network to which this individual has membership (coded by the `enum OCCUPATION_NETWORKS` enum in constant.h) |
| `worker_type` | Type of hospital worker (coded by the `enum WORKER_TYPES` in constant.h) (default -1) |
| `assigned_worker_ward_type` | Type of ward within which this individual works (coded by the `enum HOSPITAL_WARD_TYPES` in constant.h) (default -1) |
| `house_no` | Household identifier to which this individual belongs |
| `quarantined` | Is the individual currently quarantined (1=Yes; 0=No).  See the quarantine reasons at each time step for a complete list of quarantined individuals through time.  Note that this is a variable that may change throughout the simulation and so may be removed from this file in the future.   |
| `time_quarantined` | Time at which the individual was quarantined if they are currently quarantined.  Note that this is a variable that may change throughout the simulation and so may be removed from this file in the future.   |
| `test_status` | `quarantine_test_result` attribute of the individual struct.  Currently mainly used for testing purposes.  Takes the following values: -2: the individual is not currently being tested; -1 (-3): a (priority) test has been ordered; 0: the individual is waiting for a result which will be negative; 1: the individual is waiting for a test result which will be positive; note that regardless of the result once the test result has been received this variables returns to not current being tested (-2) |
| `app_user` | Is this individual an app user  (1=Yes; 0=No) |
| `mean_interactions` | Number of random daily interactions of the individual (the random_interactions attribute of the individual struct) |
| `infection_count` | Number of times this individual has been infected with SARS-CoV-2 |