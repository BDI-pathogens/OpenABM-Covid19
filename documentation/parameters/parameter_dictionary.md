# Table: Parameter dictionary
| Name | Value | Symbol | Description | Source | 
|  ---- | ---- | ---- | ---- | ---- |
| `rng_seed` | 1 | - | Random starting seed | - |
| `param_id` | 1 | - | Parameters identifier | - |
| `n_total` | 1000000 | - | Total population simulated | - |
| `mean_work_interactions_child` | 10 | - | Mean daily interactions at work (school) for children (aged 0-19) | Mossong et al, 2008 |
| `mean_work_interactions_adult` | 7 | - | Mean daily interactions at work for adults (aged 20-69) | Mossong et al, 2008 |
| `mean_work_interactions_elderly` | 3 | - | Mean daily interactions at work (or similar) for the elderly (aged 70+) | Mossong et al, 2008 |
| `daily_fraction_work` | 0.5 | - | Fraction of people in work network that an individual interacts with each day | Mossong et al, 2008 |
| `work_network_rewire` | 0.1 | - | Rewire parameter on the Watts-Strogatz work networks | - |
| `mean_random_interactions_child` | 2 | - | Mean number of daily random interactions for children (0-19) | Mossong et al, 2008 |
| `sd_random_interactions_child` | 2 | - | Standard deviation for daily random interactions for children (0-19) | Mossong et al, 2008 |
| `mean_random_interactions_adult` | 4 | - | Mean number of daily random interactions for adults (20-69) | Mossong et al, 2008 |
| `sd_random_interactions_adult` | 4 | - | Standard deviation for daily random interactions for adults (20-69) | Mossong et al, 2008 |
| `mean_random_interactions_elderly` | 3 | - | Mean number of daily random interactions for the elderly (70+) | Mossong et al, 2008 |
| `sd_random_interactions_elderly` | 3 | - | Standard deviation for daily random interactions for the elderly (70+) | Mossong et al, 2008 |
| `random_interaction_distribution` | 1 | - | Distribution used for random interactions (0=fixed, age dep, 1=negative binomial) | - |
| `child_network_adults` | 0.2 | - | Ratio of adults to children in work network for children (0-19) | Mossong et al, 2008 |
| `elderly_network_adults` | 0.2 | - | Ratio of adults to elderly in work network for elderly (70+) | Mossong et al, 2008 |
| `days_of_interactions` | 10 | - | Length of historic interactions traced (days) | - |
| `end_time` | 200 | - | End time (total number of simulated days) | - |
| `n_seed_infection` | 5 | - | Number of infections seeded at simulation start | - |
| `mean_infectious_period` | 5.5 | &#956; | Mean of the generation time distribution (days) | Ferretti et al in prep 2020; Ferretti & Wymant et al 2020; Xia et al 2020; He et al 2020; Cheng et al 2020 |
| `sd_infectious_period` | 2.14 | &#963; | Standard deviation (days) of infectious period | Ferretti et al in prep 2020; Ferretti & Wymant et al 2020; Xia et al 2020; He et al 2020; Cheng et al 2020 |
| `infectious_rate` | 5.64 | *R* | Mean number of individuals infected by each infectious individual with moderate to severe symptoms | Derived from calibration |
| `mean_time_to_symptoms` | 5.42 | &#956;<sub>sym</sub> | Mean time from infection to onset of symptoms (days) | McAloon et al. |
| `sd_time_to_symptoms` | 2.7 | &#963;<sub>sym</sub> | Standard deviation of time from infection to onset of symptoms (days) | McAloon et al. |
| `mean_time_to_hospital` | 5.14 | &#956;<sub>hosp</sub> | Mean time from symptom onset to hospitalisation (days) | Pellis et al, 2020 |
| `mean_time_to_critical` | 2.27 | &#956;<sub>crit</sub> | Mean time from hospitalisation to critical care admission (days) | Personal communication with SPI-M; data soon to be published |
| `sd_time_to_critical` | 2.27 | &#963;<sub>crit</sub> | Standard deviation of time from hospitalisation to critical care admission (days) | Personal communication with SPI-M; data soon to be published |
| `mean_time_to_recover` | 12 | &#956;<sub>rec</sub> | Mean time to recovery if hospitalisation is not required (days) | Yang et al 2020 |
| `sd_time_to_recover` | 5 | &#963;<sub>rec</sub> | Standard deviation of time to recovery if hospitalisaion is not required (days) | Yang et al 2020 |
| `mean_time_to_death` | 11.74 | &#956;<sub>death</sub> | Mean time to death after acquiring critical care (days) | Personal communication with SPI-M; data soon to be published |
| `sd_time_to_death` | 8.79 | &#963;<sub>death</sub> | Standard deviation of time to death after acquiring critical care (days) | Personal communication with SPI-M; data soon to be published |
| `fraction_asymptomatic_0_9` | 0.605 | &#966;<sub>asym</sub>(0-9) | Fraction of infected individuals who are asymptomatic, aged 0-9 | - |
| `fraction_asymptomatic_10_19` | 0.546 | &#966;<sub>asym</sub>(10-19) | Fraction of infected individuals who are asymptomatic, aged 10-19 | Calibration of Riccardo et al. 2020 & Spanish Serology Survey |
| `fraction_asymptomatic_20_29` | 0.483 | &#966;<sub>asym</sub>(20-29) | Fraction of infected individuals who are asymptomatic, aged 20-29 | Calibration of Riccardo et al. 2020 & Spanish Serology Survey |
| `fraction_asymptomatic_30_39` | 0.418 | &#966;<sub>asym</sub>(30-39) | Fraction of infected individuals who are asymptomatic, aged 30-39 | Calibration of Riccardo et al. 2020 & Spanish Serology Survey |
| `fraction_asymptomatic_40_49` | 0.354 | &#966;<sub>asym</sub>(40-49) | Fraction of infected individuals who are asymptomatic, aged 40-49 | Calibration of Riccardo et al. 2020 & Spanish Serology Survey |
| `fraction_asymptomatic_50_59` | 0.294 | &#966;<sub>asym</sub>(50-59) | Fraction of infected individuals who are asymptomatic, aged 50-59 | Calibration of Riccardo et al. 2020 & Spanish Serology Survey |
| `fraction_asymptomatic_60_69` | 0.242 | &#966;<sub>asym</sub>(60-69) | Fraction of infected individuals who are asymptomatic, aged 60-69 | Calibration of Riccardo et al. 2020 & Spanish Serology Survey |
| `fraction_asymptomatic_70_79` | 0.199 | &#966;<sub>asym</sub>(70-79) | Fraction of infected individuals who are asymptomatic, aged 70-79 | Calibration of Riccardo et al. 2020 & Spanish Serology Survey |
| `fraction_asymptomatic_80` | 0.163 | &#966;<sub>asym</sub>(80) | Fraction of infected individuals who are asymptomatic, aged 80+ | - |
| `asymptomatic_infectious_factor` | 0.29 | *A<sub>asym</sub>* | Infectious rate of asymptomatic individuals relative to symptomatic individuals | Luo et al 2020 |
| `mild_fraction_0_9` | 0.387 | &#966;<sub>mild</sub>(0-9) | Fraction of infected individuals with mild symptoms, aged 0-9 | Calibration of Riccardo et al. 2020 & Spanish Serology Survey |
| `mild_fraction_10_19` | 0.435 | &#966;<sub>mild</sub>(10-19) | Fraction of infected individuals with mild symptoms, aged 10-19 | Calibration of Riccardo et al. 2020 & Spanish Serology Survey |
| `mild_fraction_20_29` | 0.478 | &#966;<sub>mild</sub>(20-29) | Fraction of infected individuals with mild symptoms, aged 20-29 | Calibration of Riccardo et al. 2020 & Spanish Serology Survey |
| `mild_fraction_30_39` | 0.512 | &#966;<sub>mild</sub>(30-39) | Fraction of infected individuals with mild symptoms, aged 30-39 | Calibration of Riccardo et al. 2020 & Spanish Serology Survey |
| `mild_fraction_40_49` | 0.532 | &#966;<sub>mild</sub>(40-49) | Fraction of infected individuals with mild symptoms, aged 40-49 | Calibration of Riccardo et al. 2020 & Spanish Serology Survey |
| `mild_fraction_50_59` | 0.541 | &#966;<sub>mild</sub>(50-59) | Fraction of infected individuals with mild symptoms, aged 50-59 | Calibration of Riccardo et al. 2020 & Spanish Serology Survey |
| `mild_fraction_60_69` | 0.543 | &#966;<sub>mild</sub>(60-69) | Fraction of infected individuals with mild symptoms, aged 60-69 | Calibration of Riccardo et al. 2020 & Spanish Serology Survey |
| `mild_fraction_70_79` | 0.541 | &#966;<sub>mild</sub>(70-79) | Fraction of infected individuals with mild symptoms, aged 70-79 | Calibration of Riccardo et al. 2020 & Spanish Serology Survey |
| `mild_fraction_80` | 0.534 | &#966;<sub>mild</sub>(80) | Fraction of infected individuals with mild symptoms, aged 80+ | Calibration of Riccardo et al. 2020 & Spanish Serology Survey |
| `mild_infectious_factor` | 0.48 | *A<sub>mild</sub>* | Infectious rate of mildly symptomatic individuals relative to symptomatic individuals | Luo et al 2020 |
| `mean_asymptomatic_to_recovery` | 15 | &#956;<sub>a,rec</sub> | Mean time from infection to recovery (and no longer infectious) for an asymptomatic individual (days) | Yang et al 2020 |
| `sd_asymptomatic_to_recovery` | 5 | &#963;<sub>a,rec</sub> | Standard deviation from infection to recovery for an asymptomatic individual (days) | Yang et al 2020 |
| `household_size_1` | 7452 | - | Number of UK households with 1 person (thousands) | ONS UK |
| `household_size_2` | 9936 | - | Number of UK households with 2 people (thousands) | ONS UK |
| `household_size_3` | 4416 | - | Number of UK households with 3 people (thousands) | ONS UK |
| `household_size_4` | 4140 | - | Number of UK households with 4 people (thousands) | ONS UK |
| `household_size_5` | 1104 | - | Number of UK households with 5 people (thousands) | ONS UK |
| `household_size_6` | 552 | - | Number of UK households with 6 people (thousands) | ONS UK |
| `population_0_9` | 8054000 | - | UK population aged 0-9 | ONS UK |
| `population_10_19` | 7528000 | - | UK population aged 10-19 | ONS UK |
| `population_20_29` | 8712000 | - | UK population aged 20-29 | ONS UK |
| `population_30_39` | 8835000 | - | UK population aged 30-39 | ONS UK |
| `population_40_49` | 8500000 | - | UK population aged 40-49 | ONS UK |
| `population_50_59` | 8968000 | - | UK population aged 50-59 | ONS UK |
| `population_60_69` | 7069000 | - | UK population aged 60-69 | ONS UK |
| `population_70_79` | 5488000 | - | UK population aged 70-79 | ONS UK |
| `population_80` | 3281000 | - | UK population aged 80+ | ONS UK |
| `daily_non_cov_symptoms_rate` | 0.002 | - | Daily probability of reporting similar symptoms which are not covid-19, including seasonal flu | UK flu survey |
| `relative_susceptibility_0_9` | 0.35 | *S<sub>0-9</sub>* | Relative susceptibility to infection, aged 0-9 | Zhang et al. 2020 |
| `relative_susceptibility_10_19` | 0.69 | *S<sub>10-19</sub>* | Relative susceptibility to infection, aged 10-19 | Zhang et al. 2020 |
| `relative_susceptibility_20_29` | 1.03 | *S<sub>20-29</sub>* | Relative susceptibility to infection, aged 20-29 | Zhang et al. 2020 |
| `relative_susceptibility_30_39` | 1.03 | *S<sub>30-39</sub>* | Relative susceptibility to infection, aged 30-39 | Zhang et al. 2020 |
| `relative_susceptibility_40_49` | 1.03 | *S<sub>40-49</sub>* | Relative susceptibility to infection, aged 40-49 | Zhang et al. 2020 |
| `relative_susceptibility_50_59` | 1.03 | *S<sub>50-59</sub>* | Relative susceptibility to infection, aged 50-59 | Zhang et al. 2020 |
| `relative_susceptibility_60_69` | 1.27 | *S<sub>60-69</sub>* | Relative susceptibility to infection, aged 60-69 | Zhang et al. 2020 |
| `relative_susceptibility_70_79` | 1.52 | *S<sub>70-79</sub>* | Relative susceptibility to infection, aged 70-79 | Zhang et al. 2020 |
| `relative_susceptibility_80` | 1.52 | *S<sub>80</sub>* | Relative susceptibility to infection, aged 80+ | Zhang et al. 2020 |
| `relative_transmission_household` | 2 | *B<sub>home</sub>* | Relative infectious rate of household interaction | - |
| `relative_transmission_occupation` | 1 | *B<sub>occupation</sub>* | Relative infectious rate of workplace interaction | - |
| `relative_transmission_random` | 1 | *B<sub>random</sub>* | Relative infectious rate of random interaction | - |
| `hospitalised_fraction_0_9` | 0.002 | &#966;<sub>hosp</sub>(0-9) | Fraction of infected individuals with severe symptoms aged 0-9 who are hospitalised | Calibration of Riccardo et al. 2020 & Spanish Serology Survey & Ferguson et al. 2020 |
| `hospitalised_fraction_10_19` | 0.009 | &#966;<sub>hosp</sub>(10-19) | Fraction of infected individuals with severe symptoms aged 10-19 who are hospitalised | Calibration of Riccardo et al. 2020 & Spanish Serology Survey & Ferguson et al. 2020 |
| `hospitalised_fraction_20_29` | 0.017 | &#966;<sub>hosp</sub>(20-29) | Fraction of infected individuals with severe symptoms aged 20-29 who are hospitalised | Calibration of Riccardo et al. 2020 & Spanish Serology Survey & Ferguson et al. 2020 |
| `hospitalised_fraction_30_39` | 0.065 | &#966;<sub>hosp</sub>(30-39) | Fraction of infected individuals with severe symptoms aged 30-39 who are hospitalised | Calibration of Riccardo et al. 2020 & Spanish Serology Survey & Ferguson et al. 2020 |
| `hospitalised_fraction_40_49` | 0.186 | &#966;<sub>hosp</sub>(40-49) | Fraction of infected individuals with severe symptoms aged 40-49 who are hospitalised | Calibration of Riccardo et al. 2020 & Spanish Serology Survey & Ferguson et al. 2020 |
| `hospitalised_fraction_50_59` | 0.231 | &#966;<sub>hosp</sub>(50-59) | Fraction of infected individuals with severe symptoms aged 50-59 who are hospitalised | Calibration of Riccardo et al. 2020 & Spanish Serology Survey & Ferguson et al. 2020 |
| `hospitalised_fraction_60_69` | 0.324 | &#966;<sub>hosp</sub>(60-69) | Fraction of infected individuals with severe symptoms aged 60-69 who are hospitalised | Calibration of Riccardo et al. 2020 & Spanish Serology Survey & Ferguson et al. 2020 |
| `hospitalised_fraction_70_79` | 0.387 | &#966;<sub>hosp</sub>(70-79) | Fraction of infected individuals with severe symptoms aged 70-79 who are hospitalised | Calibration of Riccardo et al. 2020 & Spanish Serology Survey & Ferguson et al. 2020 |
| `hospitalised_fraction_80` | 0.439 | &#966;<sub>hosp</sub>(80) | Fraction of infected individuals with severe symptoms aged 80+ who are hospitalised | Calibration of Riccardo et al. 2020 & Spanish Serology Survey & Ferguson et al. 2020 |
| `critical_fraction_0_9` | 0.05 | &#966;<sub>crit</sub>(0-9) | Fraction of hospitalised individuals aged 0-9 who need critical care | Ferguson et al, 2020 |
| `critical_fraction_10_19` | 0.05 | &#966;<sub>crit</sub>(10-19) | Fraction of hospiatlised individuals aged 10-19 who need critical care | Ferguson et al, 2020 |
| `critical_fraction_20_29` | 0.05 | &#966;<sub>crit</sub>(20-29) | Fraction of hospitalised individuals aged 20-29 who need critical care | Ferguson et al, 2020 |
| `critical_fraction_30_39` | 0.05 | &#966;<sub>crit</sub>(30-39) | Fraction of hospitalised individuals aged 30-39 who need critical care | Ferguson et al, 2020 |
| `critical_fraction_40_49` | 0.063 | &#966;<sub>crit</sub>(40-49) | Fraction of hospitalised individuals aged 40-49 who need critical care | Ferguson et al, 2020 |
| `critical_fraction_50_59` | 0.122 | &#966;<sub>crit</sub>(50-59) | Fraction of hospitalised individuals aged 50-59 who need critical care | Ferguson et al, 2020 |
| `critical_fraction_60_69` | 0.274 | &#966;<sub>crit</sub>(60-69) | Fraction of hospitalised individuals aged 60-69 who need critical care | Ferguson et al, 2020 |
| `critical_fraction_70_79` | 0.432 | &#966;<sub>crit</sub>(70-79) | Fraction of hospitalised individuals aged 70-79 who need critical care | Ferguson et al, 2020 |
| `critical_fraction_80` | 0.709 | &#966;<sub>crit</sub>(80) | Fraction of hospitalised individuals aged 80+ who need critical care | Ferguson et al, 2020 |
| `fatality_fraction_0_9` | 0.33 | &#966;<sub>death</sub>(0-9) | Fraction of fatalities amongst individuals in critical care aged 0-9 | Lu et al. 2020, Dong et al. 2020 |
| `fatality_fraction_10_19` | 0.25 | &#966;<sub>death</sub>(10-19) | Fraction of fatalities amongst individuals in critical care aged 10-19 | Lu et al. 2020, Dong et al. 2020 |
| `fatality_fraction_20_29` | 0.5 | &#966;<sub>death</sub>(20-29) | Fraction of fatalities amongst individuals in critical care aged 20-29 | Ferguson et al, 2020 |
| `fatality_fraction_30_39` | 0.5 | &#966;<sub>death</sub>(30-39) | Fraction of fatalities amongst individuals in critical care aged 30-39 | Ferguson et al, 2020 |
| `fatality_fraction_40_49` | 0.5 | &#966;<sub>death</sub>(40-49) | Fraction of fatalities amongst individuals in critical care aged 40-49 | Yang et al 2020 |
| `fatality_fraction_50_59` | 0.69 | &#966;<sub>death</sub>(50-59) | Fraction of fatalities amongst individuals in critical care aged 50-59 | Yang et al 2020 |
| `fatality_fraction_60_69` | 0.65 | &#966;<sub>death</sub>(60-69) | Fraction of fatalities amongst individuals in critical care aged 60-69 | Yang et al 2020 |
| `fatality_fraction_70_79` | 0.88 | &#966;<sub>death</sub>(70-79) | Fraction of fatalities amongst individuals in critical care aged 70-79 | Yang et al 2020 |
| `fatality_fraction_80` | 1 | &#966;<sub>death</sub>(80) | Fraction of fatalities amongst individuals in critical care aged 80+ | Yang et al 2020 |
| `mean_time_hospitalised_recovery` | 8.75 | &#956;<sub>hosp,rec</sub> | Mean time to recover if hospitalised | Personal communication with SPI-M; data soon to be published |
| `sd_time_hospitalised_recovery` | 8.75 | &#963;<sub>hosp,rec</sub> | Standard deviation of time to recover if hospitalised | Personal communication with SPI-M; data soon to be published |
| `mean_time_critical_survive` | 18.8 | &#956;<sub>crit,surv</sub> | Mean time to survive if critical | Personal communication with SPI-M; data soon to be published |
| `sd_time_critical_survive` | 12.21 | &#963;<sub>crit,surv</sub> | Standard deviation of time to survive if critical | Personal communication with SPI-M; data soon to be published |
| `location_death_icu_0_9` | 1 | &#966;<sub>ICU</sub>(0-9) | Proportion of deaths in 0-9 year olds which occur in critical care | - |
| `location_death_icu_10_19` | 1 | &#966;<sub>ICU</sub>(10-19) | Proportion of deaths in 10-19 year olds which occur in critical care | - |
| `location_death_icu_20_29` | 0.9 | &#966;<sub>ICU</sub>(20-29) | Proportion of deaths in 20-29 year olds which occur in critical care | - |
| `location_death_icu_30_39` | 0.9 | &#966;<sub>ICU</sub>(30-39) | Proportion of deaths in 30-39 year olds which occur in critical care | - |
| `location_death_icu_40_49` | 0.8 | &#966;<sub>ICU</sub>(40-49) | Proportion of deaths in 40-49 year olds which occur in critical care | - |
| `location_death_icu_50_59` | 0.8 | &#966;<sub>ICU</sub>(50-59) | Proportion of deaths in 50-59 year olds which occur in critical care | - |
| `location_death_icu_60_69` | 0.4 | &#966;<sub>ICU</sub>(60-69) | Proportion of deaths in 60-69 year olds which occur in critical care | - |
| `location_death_icu_70_79` | 0.4 | &#966;<sub>ICU</sub>(70-79) | Proportion of deaths in 70-79 year olds which occur in critical care | - |
| `location_death_icu_80` | 0.05 | &#966;<sub>ICU</sub>(80) | Proportion of deaths in 80+ year olds which occur in critical care | - |
| `quarantine_length_self` | 7 | - | Maximum number of days quarantine for individuals self-reporting symptoms | - |
| `quarantine_length_traced_symptoms` | 14 | - | Maximum number of days quarantine for individuals who are traced after a contact reported symptoms | - |
| `quarantine_length_traced_positive` | 14 | - | Maximum number of days quarantine for individuals who are traced after a contact tested positive | - |
| `quarantine_length_positive` | 14 | - | Maximum number of days quarantine for individuals with a positive test result | - |
| `quarantine_dropout_self` | 0.02 | - | Daily probability of drop out for an individual quarantining after self-reporting symptoms | - |
| `quarantine_dropout_traced_symptoms` | 0.04 | - | Daily probability of drop out for an individual quarantining after being traced following contact with an individual self-reporting symptoms | - |
| `quarantine_dropout_traced_positive` | 0.03 | - | Daily probability of drop out for an individual quarantining after being traced following contact with an individual who tested positive | - |
| `quarantine_dropout_positive` | 0.01 | - | Daily probability of drop out for an individual quarantining after a positive test result | - |
| `quarantine_compliance_traced_symptoms` | 0.5 | - | Fraction of individuals who initially comply with a quarantine notification after their contact reported symptoms | - |
| `quarantine_compliance_traced_positive` | 0.9 | - | Fraction of individuals who initially comply with a quarantine notification after their contact tested positive | - |
| `test_on_symptoms` | 0 | - | Test individuals who show symptoms (0=no, 1=yes) | - |
| `test_on_traced` | 0 | - | Test individuals who have been contact-traced (0=no, 1=yes) | - |
| `test_release_on_negative` | 1 | - | Release individuals following a negative test (0=no, 1=yes) | - |
| `trace_on_symptoms` | 0 | - | Trace contacts of individuals who show symptoms (0=no, 1=yes) | - |
| `trace_on_positive` | 0 | - | Trace contacts of an individual who tests positive (0=no, 1=yes) | - |
| `retrace_on_positive` | 0 | - | Repeat contract tracing be carried out on a positive test if already traced on symptoms (0=no, 1=yes) | - |
| `quarantine_on_traced` | 0 | - | Quarantine individuals who are traced (0=no, 1=yes) | - |
| `traceable_interaction_fraction` | 0.8 | - | Fraction of interactions that are captured if both users have the app | - |
| `tracing_network_depth` | 0 | - | Depth of interaction network to contact | - |
| `allow_clinical_diagnosis` | 1 | - | Commence contact tracing on a hospital clinical diagnosis | - |
| `quarantine_household_on_positive` | 0 | - | Quarantine household members of a person with a positive test (0=no, 1=yes) | - |
| `quarantine_household_on_symptoms` | 0 | - | Quarantine household members of a person with symptoms (0=no, 1=yes) | - |
| `quarantine_household_on_traced_positive` | 0 | - | Quarantine household members of a person who has been traced following contact with an individual who has tested positive (0=no, 1=yes) | - |
| `quarantine_household_on_traced_symptoms` | 0 | - | Quarantine household members of a person who has been traced following contact with an individual who has self-reported symptoms (0=no, 1=yes) | - |
| `quarantine_household_contacts_on_positive` | 0 | - | Quarantine the contacts of each household member of a person who tests positive (0=no, 1=yes) | - |
| `quarantine_household_contacts_on_symptoms` | 0 | - | Quarantine the contacts of other household members when someone gets symptoms | - |
| `quarantined_daily_interactions` | 0 | - | Daily random interactions of a quarantined individual | - |
| `quarantine_days` | 7 | - | The number of previous days' contacts to be traced and contacted | - |
| `quarantine_smart_release_day` | 0 | - | Release a chain of quarantined people if after this number of days nobody has shown symptoms on the chain | - |
| `hospitalised_daily_interactions` | 0 | - | Daily random interactions of a hospitalised individual | - |
| `test_insensitive_period` | 3 | - | Number of days following infection the test is insensitive | Woelfel et al. 2020 |
| `test_sensitive_period` | 14 | - | Number of days following infection to end of period where test is sensitive | - |
| `test_sensitivity` | 0.8 | - | Sensitivity of test in the time where it is sensitive | - |
| `test_specificity` | 0.999 | - | Specificity of test (at any time) | - |
| `test_order_wait` | 1 | - | Minimum number of days to wait to take a test | - |
| `test_order_wait_priority` | -1 | - | Minimum number of days to wait for a priority test to be taken | - |
| `test_result_wait` | 1 | - | Number of days to wait for a test result | - |
| `test_result_wait_priority` | -1 | - | Number of days to wait for a priority test result | - |
| `priority_test_contacts_0_9` | 1000 | - | Number of contacts that triggers priority test for individuals aged 0-9 years old | - |
| `priority_test_contacts_10_19` | 1000 | - | Number of contacts that triggers priority test for individuals aged 10-19 years old | - |
| `priority_test_contacts_20_29` | 1000 | - | Number of contacts that triggers priority test for individuals aged 20-29 years old | - |
| `priority_test_contacts_30_39` | 1000 | - | Number of contacts that triggers priority test for individuals aged 30-39 years old | - |
| `priority_test_contacts_40_49` | 1000 | - | Number of contacts that triggers priority test for individuals aged 40-49 years old | - |
| `priority_test_contacts_50_59` | 1000 | - | Number of contacts that triggers priority test for individuals aged 50-59 years old | - |
| `priority_test_contacts_60_69` | 1000 | - | Number of contacts that triggers priority test for individuals aged 60-69 years old | - |
| `priority_test_contacts_70_79` | 1000 | - | Number of contacts that triggers priority test for individuals aged 70-79 years old | - |
| `priority_test_contacts_80` | 1000 | - | Number of contacts that triggers priority test for individuals aged 80+ years old | - |
| `self_quarantine_fraction` | 0 | - | Proportion of people who self-quarantine upon symptoms | - |
| `app_users_fraction_0_9` | 0.09 | - | Maximum fraction of the population with smartphones aged 0-9 | OFCOM 3-5 year olds |
| `app_users_fraction_10_19` | 0.8 | - | Maximum fraction of the population with smartphones aged 10-19 | OFCOM 5-15 year olds |
| `app_users_fraction_20_29` | 0.97 | - | Maximum fraction of the population with smartphones aged 20-29 | OFCOM 16-55 year olds |
| `app_users_fraction_30_39` | 0.96 | - | Maximum fraction of the population with smartphones aged 30-39 | OFCOM 16-55 year olds |
| `app_users_fraction_40_49` | 0.94 | - | Maximum fraction of the population with smartphones aged 40-49 | OFCOM 16-55 year olds |
| `app_users_fraction_50_59` | 0.86 | - | Maximum fraction of the population with smartphones aged 50-59 | OFCOM 16-55 year olds |
| `app_users_fraction_60_69` | 0.7 | - | Maximum fraction of the population with smartphones aged 60-69 | OFCOM 55+ year olds |
| `app_users_fraction_70_79` | 0.48 | - | Maximum fraction of the population with smartphones aged 70-79 | OFCOM 55+ year olds |
| `app_users_fraction_80` | 0.32 | - | Maximum fraction of the population with smartphones aged 80+ | OFCOM 55+ year olds |
| `app_turn_on_time` | 10000 | - | Time (days) at which to turn on the app | - |
| `lockdown_occupation_multiplier_primary_network` | 0.22 | - | Relative change in number of occupation network contacts on lockdown for primary age | - |
| `lockdown_occupation_multiplier_secondary_network` | 0.22 | - | Relative change in number of occupation network contacts on lockdown for secondary age | - |
| `lockdown_occupation_multiplier_working_network` | 0.22 | - | Relative change in number of occupation network contacts on lockdown for working age | Based on an estimate of the number of key workers |
| `lockdown_occupation_multiplier_retired_network` | 0.22 | - | Relative change in number of occupation network contacts on lockdown for retired age | - |
| `lockdown_occupation_multiplier_elderly_network` | 0.22 | - | Relative change in number of occupation network contacts on lockdown for elderly | - |
| `lockdown_random_network_multiplier` | 0.22 | - | Relative change in random network contacts on lockdown | - |
| `lockdown_house_interaction_multiplier` | 1.5 | - | Relative change in household network contacts on lockdown | - |
| `lockdown_time_on` | 10000 | - | Time (days) at which to model lockdown starting | - |
| `lockdown_time_off` | 10000 | - | Time (days) at which to model lockdown ending | - |
| `lockdown_elderly_time_on` | 10000 | - | Time (days) at which lockdown starts for elderly people | - |
| `lockdown_elderly_time_off` | 10000 | - | Time (days) at which lockdown ends for elderly people | - |
| `testing_symptoms_time_on` | 10000 | - | Time (days) at which to start testing on symptoms | - |
| `testing_symptoms_time_off` | 10000 | - | Time (days) at which to stop testing on symptoms | - |
| `intervention_start_time` | 0 | - | Time (days) after which interventions can be turned on | - |
| `hospital_on` | 0 | - | Turn on the hospital module  (0=no, 1=yes) | - |
| `manual_trace_on` | 0 | - | Turn on manual tracing (0=no, 1=yes) | - |
| `manual_trace_time_on` | 10000 | - | Time (days) after which manual tracing is turned on | - |
| `manual_trace_on_hospitalization` | 1 | - | Trace when hospitalized if tested positive (no effect if manual_trace_on_positive is on) | - |
| `manual_trace_on_positive` | 0 | - | Trace when hospitalized if tested positive (no effect if manual_trace_on_positive is on) | - |
| `manual_trace_delay` | 1 | - | Delay (days) between triggering manual tracing due to testing/hospitalization and tracing occurring | - |
| `manual_trace_exclude_app_users` | 0 | - | Whether or not to exclude app users when performing manual tracing (exclude=1, include=0) | - |
| `manual_trace_n_workers` | 300 | - | Number of Contact Tracing Workers | NACCHO Position Statement, 2020 |
| `manual_trace_interviews_per_worker_day` | 6 | - | Number of interviews performed per worker per day | https://www.gwhwi.org/estimator-613404.html |
| `manual_trace_notifications_per_worker_day` | 12 | - | Number of trace notifications performed per worker per day | https://www.gwhwi.org/estimator-613404.html |
| `manual_traceable_fraction_household` | 1 | - | The fraction of household contacts that can be successfully traced | - |
| `manual_traceable_fraction_occupation` | 0.8 | - | The fraction of occupation contacts that can be successfully traced | - |
| `manual_traceable_fraction_random` | 0.05 | - | The fraction of random contacts that can be successfully traced | - |