# Table: Active intervention parameters
| Name | Value | Symbol | Description | Source | 
|  ---- | ---- | ---- | ---- | ---- |
| `days_of_interactions` | 10 | - | Length of historic interactions traced (days) | - |
| `quarantine_length_traced_symptoms` | 14 | - | Maximum number of days quarantine for individuals who are traced after a contact reported symptoms | - |
| `quarantine_length_traced_positive` | 14 | - | Maximum number of days quarantine for individuals who are traced after a contact tested positive | - |
| `quarantine_length_positive` | 14 | - | Maximum number of days quarantine for individuals with a positive test result | - |
| `quarantine_dropout_traced_symptoms` | 0.04 | - | Daily probability of drop out for an individual quarantining after being traced following contact with an individual self-reporting symptoms | - |
| `quarantine_dropout_traced_positive` | 0.03 | - | Daily probability of drop out for an individual quarantining after being traced following contact with an individual who tested positive | - |
| `quarantine_dropout_positive` | 0.01 | - | Daily probability of drop out for an individual quarantining after a positive test result | - |
| `quarantine_compliance_traced_symptoms` | 0.5 | - | Fraction of individuals who initially comply with a quarantine notification after their contact reported symptoms | - |
| `quarantine_compliance_traced_positive` | 0.9 | - | Fraction of individuals who initially comply with a quarantine notification after their contact tested positive | - |
| `quarantine_compliance_positive` | 1.0 | - | Fraction of individuals who initially comply with a quarantine after they test positive | - |
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
| `quarantine_days` | 7 | - | The number of previous days' contacts to be traced and contacted | - |
| `quarantine_smart_release_day` | 0 | - | Release a chain of quarantined people if after this number of days nobody has shown symptoms on the chain | - |
| `test_insensitive_period` | 3 | - | Number of days following infection the test is insensitive | Woelfel et al. 2020 |
| `test_sensitive_period` | 14 | - | Number of days following infection to end of period where test is sensitive | - |
| `test_sensitivity` | 0.8 | - | Sensitivity of test in the time where it is sensitive | - |
| `test_specificity` | 0.999 | - | Specificity of test (at any time) | - |
| `test_order_wait` | 1 | - | Minimum number of days to wait to take a test | - |
| `test_result_wait` | 1 | - | Number of days to wait for a test result | - |
| `lateral_flow_test_order_wait` | 1 | - | Number of days to wait to receive a set of lateral flow tests | - |
| `lateral_flow_test_on_symptoms` | 0 | - | Test individuals with Lateral Flow Assay who show symptoms (0=no, 1=yes) | - |
| `lateral_flow_test_on_traced` | 0 | - | Test individuals with Lateral Flow Assay who have been contact-traced (0=no, 1=yes) | - |
| `lateral_flow_test_repeat_count` | 7 | - | Number of daily Lateral Flow Assay tests to take in a row | - |
| `lateral_flow_test_only` | 0 | - | If an individual takes Lateral Flow tests, do not take PCR or quarantine until they receive a test result | - |
| `lateral_flow_test_fraction` | 0.5 | - | The fraction of individuals who take a Lateral Flow Assay test instead of quarantine if offered | - |
| `lateral_flow_test_sensitivity` | 0.95 | - | Peak sensitivity of Lateral Flow Assay | - |
| `lateral_flow_test_specificity` | 0.999 | - | Specificity of Lateral Flow Assay (at any time) | - |
| `app_users_fraction_0_9` | 0.09 | - | Maximum fraction of the population with smartphones aged 0-9 | OFCOM 3-5 year olds |
| `app_users_fraction_10_19` | 0.8 | - | Maximum fraction of the population with smartphones aged 10-19 | OFCOM 5-15 year olds |
| `app_users_fraction_20_29` | 0.97 | - | Maximum fraction of the population with smartphones aged 20-29 | OFCOM 16-55 year olds |
| `app_users_fraction_30_39` | 0.96 | - | Maximum fraction of the population with smartphones aged 30-39 | OFCOM 16-55 year olds |
| `app_users_fraction_40_49` | 0.94 | - | Maximum fraction of the population with smartphones aged 40-49 | OFCOM 16-55 year olds |
| `app_users_fraction_50_59` | 0.86 | - | Maximum fraction of the population with smartphones aged 50-59 | OFCOM 16-55 year olds |
| `app_users_fraction_60_69` | 0.7 | - | Maximum fraction of the population with smartphones aged 60-69 | OFCOM 55+ year olds |
| `app_users_fraction_70_79` | 0.48 | - | Maximum fraction of the population with smartphones aged 70-79 | OFCOM 55+ year olds |
| `app_users_fraction_80` | 0.32 | - | Maximum fraction of the population with smartphones aged 80+ | OFCOM 55+ year olds |