# Table: Disease dynamics parameters
| Name | Value | Symbol | Description | Source | 
|  ---- | ---- | ---- | ---- | ---- |
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
| `mild_fraction_0_9` | 0.387 | &#966;<sub>mild</sub>(0-9) | Fraction of infected individuals with mild symptoms, aged 0-9 | Calibration of Riccardo et al. 2020 & Spanish Serology Survey |
| `mild_fraction_10_19` | 0.435 | &#966;<sub>mild</sub>(10-19) | Fraction of infected individuals with mild symptoms, aged 10-19 | Calibration of Riccardo et al. 2020 & Spanish Serology Survey |
| `mild_fraction_20_29` | 0.478 | &#966;<sub>mild</sub>(20-29) | Fraction of infected individuals with mild symptoms, aged 20-29 | Calibration of Riccardo et al. 2020 & Spanish Serology Survey |
| `mild_fraction_30_39` | 0.512 | &#966;<sub>mild</sub>(30-39) | Fraction of infected individuals with mild symptoms, aged 30-39 | Calibration of Riccardo et al. 2020 & Spanish Serology Survey |
| `mild_fraction_40_49` | 0.532 | &#966;<sub>mild</sub>(40-49) | Fraction of infected individuals with mild symptoms, aged 40-49 | Calibration of Riccardo et al. 2020 & Spanish Serology Survey |
| `mild_fraction_50_59` | 0.541 | &#966;<sub>mild</sub>(50-59) | Fraction of infected individuals with mild symptoms, aged 50-59 | Calibration of Riccardo et al. 2020 & Spanish Serology Survey |
| `mild_fraction_60_69` | 0.543 | &#966;<sub>mild</sub>(60-69) | Fraction of infected individuals with mild symptoms, aged 60-69 | Calibration of Riccardo et al. 2020 & Spanish Serology Survey |
| `mild_fraction_70_79` | 0.541 | &#966;<sub>mild</sub>(70-79) | Fraction of infected individuals with mild symptoms, aged 70-79 | Calibration of Riccardo et al. 2020 & Spanish Serology Survey |
| `mild_fraction_80` | 0.534 | &#966;<sub>mild</sub>(80) | Fraction of infected individuals with mild symptoms, aged 80+ | Calibration of Riccardo et al. 2020 & Spanish Serology Survey |
| `mean_asymptomatic_to_recovery` | 15 | &#956;<sub>a,rec</sub> | Mean time from infection to recovery (and no longer infectious) for an asymptomatic individual (days) | Yang et al 2020 |
| `sd_asymptomatic_to_recovery` | 5 | &#963;<sub>a,rec</sub> | Standard deviation from infection to recovery for an asymptomatic individual (days) | Yang et al 2020 |
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