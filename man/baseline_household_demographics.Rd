%% Documentation of file 'tests/src/baseline_household_demographics.csv'

\name{baseline_household_demographics}
\docType{data}
\alias{baseline_household_demographics}
\title{UK Household Demographics Data}
\description{
  The demographics of the ABM are based upon UK national data for 2018
  from the Office of National Statistics (ONS).

  Individuals are categorised into nine age groups by decade, from age
  group (0-9 years) to (80+ years). The simulation is run on a static
  population and the proportion of individuals in each age group is the
  same as that specified by the population level statistics in Table
  \href{https://github.com/BDI-pathogens/OpenABM-Covid19/blob/master/documentation/parameters/demographic_parameters.md}{Demographics Table}.
  Every individual is part of a household, which forms an important part
  of their daily interactions. Household size data is shown in Table
  \href{https://github.com/BDI-pathogens/OpenABM-Covid19/blob/master/documentation/parameters/demographic_parameters.md}{Demographics Table}.
  Since the duration of the simulated epidemic is less than a year, we
  do not consider changes in the population due to births, deaths due to
  other causes, and migration.
}
\usage{
data('baseline',package='OpenABMCovid19')
baseline_household_demographics
}
\source{
  Obtained from the UK Office of National Statistics (2011) at:
  \url{https://www.ons.gov.uk/peoplepopulationandcommunity/housing/adhocs/11569ct10892011census}
}
\references{
  UK Office of National Statistics (2011):
  \emph{CT1089_2011 Census COVID-19}
}
