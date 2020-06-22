# Table: Passive intervention parameters
| Name | Value | Symbol | Description | Source | 
|  ---- | ---- | ---- | ---- | ---- |
| `daily_non_cov_symptoms_rate` | 0.002 | - | Daily probability of reporting similar symptoms which are not covid-19, including seasonal flu | UK flu survey |
| `quarantine_length_self` | 7 | - | Maximum number of days quarantine for individuals self-reporting symptoms | - |
| `quarantine_dropout_self` | 0.02 | - | Daily probability of drop out for an individual quarantining after self-reporting symptoms | - |
| `quarantined_daily_interactions` | 0 | - | Daily random interactions of a quarantined individual | - |
| `hospitalised_daily_interactions` | 0 | - | Daily random interactions of a hospitalised individual | - |
| `self_quarantine_fraction` | 0 | - | Proportion of people who self-quarantine upon symptoms | - |
| `lockdown_occupation_multiplier_primary_network` | 0.22 | - | Relative change in number of occupation network contacts on lockdown for primary age | - |
| `lockdown_occupation_multiplier_secondary_network` | 0.22 | - | Relative change in number of occupation network contacts on lockdown for secondary age | - |
| `lockdown_occupation_multiplier_working_network` | 0.22 | - | Relative change in number of occupation network contacts on lockdown for working age | Based on an estimate of the number of key workers |
| `lockdown_occupation_multiplier_retired_network` | 0.22 | - | Relative change in number of occupation network contacts on lockdown for retired age | - |
| `lockdown_occupation_multiplier_elderly_network` | 0.22 | - | Relative change in number of occupation network contacts on lockdown for elderly | - |
| `lockdown_random_network_multiplier` | 0.22 | - | Relative change in random network contacts on lockdown | - |
| `lockdown_house_interaction_multiplier` | 1.5 | - | Relative change in household network contacts on lockdown | - |