# Table: Infection parameters
| Name | Value | Symbol | Description | Source | 
|  ---- | ---- | ---- | ---- | ---- |
| `n_seed_infection` | 10 | - | Number of infections seeded at simulation start | - |
| `mean_infectious_period` | 5.5 | &#956; | Mean of the generation time distribution (days) | Ferretti et al in prep 2020; Ferretti & Wymant et al 2020; Xia et al 2020; He et al 2020; Cheng et al 2020 |
| `sd_infectious_period` | 2.14 | &#963; | Standard deviation (days) of infectious period | Ferretti et al in prep 2020; Ferretti & Wymant et al 2020; Xia et al 2020; He et al 2020; Cheng et al 2020 |
| `infectious_rate` | 5.8 | *R* | Mean number of individuals infected by each infectious individual with moderate to severe symptoms | Derived from calibration |
| `sd_infectiousness_multiplier` | 1.4 | - | SD of the lognormal used to vary the infectiousness of an individual | Derived from calibration |
| `asymptomatic_infectious_factor` | 0.33 | *A<sub>asym</sub>* | Infectious rate of asymptomatic individuals relative to symptomatic individuals | Personal communication, Sun |
| `mild_infectious_factor` | 0.72 | *A<sub>mild</sub>* | Infectious rate of mildly symptomatic individuals relative to symptomatic individuals | Personal communication, Sun |
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