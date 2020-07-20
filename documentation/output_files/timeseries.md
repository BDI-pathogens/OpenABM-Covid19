# Table: timeseries
| Column name | Description | 
|  ---- | ---- |
| `time` | Day number of the simulation |
| `lockdown` | Is lockdown currently on (1 = Yes; 0 = No) |
| `test_on_symptoms` | Is testing on symptoms currently on (1=Yes; 0=No) |
| `app_turned_on` | Is the app currently on  (1= Yes; 0 = No) |
| `total_infected` | Cumulative infected |
| `total_infected_0_9` | Cumulative infected aged 0-9 years |
| `total_infected_10_19` | Cumulative infected aged 10-19 years |
| `total_infected_20_29` | Cumulative infected aged 20-29 years |
| `total_infected_30_39` | Cumulative infected aged 30-39 years |
| `total_infected_40_49` | Cumulative infected aged 40-49 years |
| `total_infected_50_59` | Cumulative infected aged 50-59 years |
| `total_infected_60_69` | Cumulative infected aged 60-69 years |
| `total_infected_70_79` | Cumulative infected aged 70-79 years |
| `total_infected_80` | Cumulative infected aged 80+ years |
| `total_case` | Cumulative cases (a case is defined by a positive test result) |
| `total_case_0_9` | Cumulative cases aged 0-9 years |
| `total_case_10_19` | Cumulative cases aged 10-19 years |
| `total_case_20_29` | Cumulative cases aged 20-29 years |
| `total_case_30_39` | Cumulative cases aged 30-39 years |
| `total_case_40_49` | Cumulative cases aged 40-49 years |
| `total_case_50_59` | Cumulative cases aged 50-59 years |
| `total_case_60_69` | Cumulative cases aged 60-69 years |
| `total_case_70_79` | Cumulative cases aged 70-79 years |
| `total_case_80` | Cumulative cases aged 80+ years |
| `total_death` | Cumulative deaths (where COVID19 is the primary cause of death) |
| `total_death_0_9` | Cumulative deaths aged 0-9 years |
| `total_death_10_19` | Cumulative deaths aged 10-19 years |
| `total_death_20_29` | Cumulative deaths aged 20-29 years |
| `total_death_30_39` | Cumulative deaths aged 30-39 years |
| `total_death_40_49` | Cumulative deaths aged 40-49 years |
| `total_death_50_59` | Cumulative deaths aged 50-59 years |
| `total_death_60_69` | Cumulative deaths aged 60-69 years |
| `total_death_70_79` | Cumulative deaths aged 70-79 years |
| `total_death_80` | Cumulative deaths aged 80+ years |
| `n_presymptom` | Current number presymptomatic (both mild and severe) |
| `n_asymptom` | Current number asymptomatic |
| `n_quarantine` | Current number in quarantine |
| `n_tests` | Current number of tests reporting results today |
| `n_symptoms` | Current number of symptomatic (both mild and severe) |
| `n_hospital` | Current number in hospital who have not yet required critical care |
| `n_hospitalised_recovering` | Current number in hospital who left critical care but not been discharged |
| `n_critical` | Current number in critical care |
| `n_death` | Daily number of deaths |
| `n_recovered` | Cumulative number recovered |
| `hospital_admissions` | Daily hospital admissions |
| `hospital_admissions_total` | Cumulative hospital admissions |
| `hospital_to_critical_daily` | Daily transitions from hospital to critical |
| `hospital_to_critical_total` | Cumulative transitions from hospital to critical |
| `n_quarantine_infected` | Current number in quarantine that have ever been infected |
| `n_quarantine_recovered` | Current number in quarantine that are recovered |
| `n_quarantine_app_user` | Current number of app users in quarantine |
| `n_quarantine_app_user_infected` | Current number of app users in quarantine that have ever been infected |
| `n_quarantine_app_user_recovered` | Current number of app users in quarantine that are recovered |
| `n_quarantine_events` | Daily number of quarantine events |
| `n_quarantine_release_events` | Daily number of quarantine release events |
| `n_quarantine_events_app_user` | Daily number of quarantine events of app users |
| `n_quarantine_release_events_app_user` | Daily number of quarantine release events of app users |