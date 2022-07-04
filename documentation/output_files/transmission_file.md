# Table: transmission file
| Column name | Description | 
|  ---- | ---- |
| `ID_recipient` | Unique identifier of the recipient |
| `age_group_recipient` | Age group of the recipient (coded by the `enum AGE_GROUPS` in constant.h) |
| `house_no_recipient` | Household identifier of the recipient |
| `occupation_network_recipient` | Occupation network of the recipient (coded by the `enum OCCUPATION_NETWORKS` within constant.h) |
| `worker_type_recipient` | Type of hospital worker of the recipient (coded by the `enum WORKER_TYPES` in constant.h) (default -1) |
| `hospital_state_recipient` | Hospital status of the recipient at time of transmission (coded by `enum EVENT_TYPES` in constant.h) (default NOT_IN_HOSPITAL) |
| `infector_network` | Network within which the transmission took place (coded by the `enum INTERACTION_TYPE` within constant.h) |
| `generation_time` | Generation time of this transmission event (days; time from infection of the source to transmission) (0 for seed cases) |
| `ID_source` | Unique identifier of the source (same as ID_recipient for seed cases) |
| `age_group_source` | Age group of the source (coded by the `enum AGE_GROUPS` in constant.h) |
| `house_no_source` | Household identifier of the source |
| `occupation_network_source` | Occupation network of the source (coded by the `enum OCCUPATION_NETWORKS` within constant.h) |
| `worker_type_source` | Type of hospital worker of the source at time of transmission (coded by the `enum WORKER_TYPES` in constant.h) (default -1) |
| `hospital_state_source` | Hospital status of the source (coded by `enum EVENT_TYPES` in constant.h) (default NOT_IN_HOSPITAL) |
| `time_infected_source` | Time when source was infected |
| `status_source` | Infectious status of the source at time of transmission (coded by `enum EVENT_TYPES` within constant.h) |
| `time_infected` | Time at which transmission took place (time measured as day number of the simulation) |
| `time_presymptomatic` | Time at which the recipient became presymptomatic (-1 if never) |
| `time_presymptomatic_mild` | Time at which the recipient became presymptomatic (if mildly infected) (-1 if never) |
| `time_presymptomatic_severe` | Time at which the recipient became presymptomatic (if severely infected) (-1 if never) |
| `time_symptomatic` | Time at which the recipient became symptomatic (-1 if never) |
| `time_symptomatic_mild` | Time at which the recipient became symptomatic (if mildly infected) (-1 if never) |
| `time_symptomatic_severe` | Time at which the recipient became symptomatic (if severely infected) (-1 if never) |
| `time_asymptomatic` | Time at which the recipient became asymptomatic (-1 if never) |
| `time_hospitalised` | Time at which the recipient became hospitalised (-1 if never) |
| `time_critical` | Time at which the recipient became critical (-1 if never) |
| `time_hospitalised_recovering` | Time at which the recipient was discharged from critical but remained in hospital (-1 if never) |
| `time_death` | Time at which the recipient died (-1 if never) |
| `time_recovered` | Time at which the recipient recovered (-1 if never) |
| `time_susceptible` | Time at which the recipient became susceptible again (if waning immunity is possible) |
| `is_case` | Was the recipient identified as a case (positive test result) (1=Yes; 0=No) |
| `strain_multiplier` | The relative transmissibility of the strain of the infector (1.0 = default transmissibility) |
