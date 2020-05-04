# AdaptER-Covid19

## Introduction

Accurate epidemiological simulations are a key tool for understanding, responding to and managing the Covid-19 pandemic. While understanding health-outcomes of policy interventions is rightly the primary focus of such simulations, there are clear benefits to a holistic simulation that takes into account economic dynamics as well. Created for this purpose, AdaptER-Covid19 (Adaptive Economic Response) is an economic simulation that integrates with the OpenABM-Covid19 simulator in order to jointly model health and economic outcomes of epidemics, with a particular focus on Covid-19.

AdaptER-Covid19 takes as input from the epidemiological part of the model timeseries describing the simulated course of the epidemic (e.g. number of symptomatic individuals) as well as exogenous variables (e.g. whether a lockdown is in place in the given scenario) and simulates the impact on key economic metrics such as GDP, corporate bankruptcy rate and individual insolvency rate. The main focus of the first version of the model is on the short-term effects of labour supply constraints on economic output, as well as second-order effects such as corporate bankruptcies and the longer-term effects these have on capital stock and employment. For this purpose, the UK economy is broken down by sector and region to model local and sector-specific effects of both the epidemic and the policy response. Demand-side effects, such as impact on consumption and investment are not modelled in detail; this will be addressed in future work.

It should also be noted that AdaptER-Covid19 is a simulator rather than a forecast. Its outputs are dependent on both the mathematical model as well as a host of parameter choices on which limited data is available (e.g. worker productivity by region and sector when limited to working from home) and which may vary considerably across scenarios. As such, simulation results from AdaptER-Covid19 may be used to draw qualitative conclusions when comparing different scenarios rather than for quantitative estimates or inference of absolute numbers. To allow the user to properly evaluate simulation results, we describe the current model in detail below.

## Model Overview

To limit complexity, the overall economic simulator is broken down into four linked components. The first three model the impact of the epidemic:

* an *input-output model* of the economy that links primary and intermediate inputs to production, on the one hand, to intermediate and final uses of production, on the other hand, on a sector by sector basis and is used in particular to model the effect of constraining labour supply on GDP,
* a *corporate bankruptcy model* which estimates corporate default rates over time, based on data from the input-output model as well as assumptions around corporate cash flow,
* an *individual solvency model* which estimates risk of individual insolvencies, by region and sector, based on both corporate bankruptcies as well as overall GDP.

Taken together, the above three models constitute a simulation of the short-term shock to the economy caused by the epidemic. Moreover, the destruction of capital through an excess rate of corporate defaults is used to discount GDP going forward. However, this approach is inherently limited to modelling a shortfall of GDP with respect to a baseline and does not account for growth. For this purpose, we add

* a *long-term trend model* which models growth dynamics of the economy based on fixed long-term growth assumptions

and which is overlaid on the combined output of the first three models.

Before we go into further detail on these four component models, we point out key **limitations** in the current approach:

* Some quantities that are already modelled in some of the component models are currently not fully linked with other components. For example, financial distress of households from the individual solvency model should be used to constrain household consumption in the final use section of the input-output model. Similarly, capital destruction given by the corporate bankruptcy model should directly affect capital stock in the primary input section of the input-output model. These feedback loops are currently modelled using ad-hoc GDP discounting. A deeper integration along the lines described above is future work.

* Consumption patterns are not represented in the model yet. Thus, there are a wide range of effects of economic import that are not currently possible to simulate, for example: Consumer reluctance to go to restaurants as case numbers increase, even if a lockdown is not in place. Pent-up demand leading to an increase in spending after a prolonged lockdown is lifted. Shift in demand towards delivery services and home entertainment. Increase in the savings rate across households in response to economic downturn.

* Similarly, changes in investment behaviour of corporates are not simulated with granularity, apart from overall destruction of capital stock due to defaults. Effects that cannot be simulated in detail include: Investment in technology to facilitate working from home. Investment in delivery services and online ordering and divestment from brick-and-mortar stores. Deleveraging of corporates that do not default.

* Government spending is not currently modelled as a separate source of demand, but rather assumed to be constant. In particular, specific economic stimulus packages are not taken into account and will be addressed in future work.

* As mentioned previously, the model currently combines a model of the short-to-medium term downside impact of the epidemic with a simple long-term growth trend model. Modelling the dynamics of demand as described above properly will give a more detailed simulation of the return to growth in the long term.

* The UK government has deployed a wide range of economic policies. Some of these, such as furloughing of workers and tax deferral are included to a first approximation. However, a wide range of other measures, from mortgage holidays and Statutory Sick Pay to the Covid-19 Corporate Financing Facility and the Coronavirus Business Interruption Loan Scheme, are not currently modelled.

* Labour and capital markets are not currently modelled. For example, the dynamics of employees switching firms or industries and of investors redeploying capital is not modelled. Prices for goods and services are assumed to be constant.

Addressing these limitations is left for future work. However, it is essential to keep current limitations of the simulator in mind when interpreting results. Also see the section on parameters below.

## Model Components

### Input-Output Model

The foundation of our model of sectoral interdependencies is the European System of Accounts and in particular the supply and use tables and input-output analytical tables (IOATs) as set out by Eurostat [1] and applied to the UK by the ONS [2]. This framework allows to express total demand for a given product, on the one hand, as the sum of primary inputs and intermediate inputs to production at basic prices and, on the other hand, as the sum of intermediate demand and final uses. Primary inputs are, broadly, broken down into imports, taxes, compensation and operating surplus. Final uses are broken down into final consumption, capital formation and exports. This accounting perspective provides a common reference points that can be referred to across component models, from both a production and consumption perspective. In particular, the matrix of intermediate demand reflects sectoral interdependencies in the UK economy. Note that while IOATs typically categorize demand by product, we assume a categorization by sector, according to the UK's Standard Industrial Classification (SIC 2007), particularly A21 in NACE Rev. 2, see [2, Annex A].

To turn this accounting framework into a model suitable for simulation, we need to specify a production function that, in particular, allows for substitution of one input to production with another, at a cost. Following [3], we adopt a Cobb-Douglas production function that takes intermediate inputs as well as imports, labour and (fixed) capital as arguments. The IOATs are used to calibrate weights in the Cobb-Douglas function. Here we make the crucial assumption that prices for goods are constant. The price of labour responds to exogenous changes to both quantity of labour and productivity imposed by illness or government measures such as lockdown, by region, sector and age-group. This is in contrast to the modelling of capital, where we assume that the stock of capital is fixed at a level given exogenously and the return on capital varies according to the model. Final use is capped at a level given exogenously. See also [4] for further background on modelling of network effects in the economy.

The input-output model currently does not have an explicit representation of time. It takes as parameters the supply of labour, as derived from the output of the epidemiological model, and simulates the economic output as a fraction of pre-crisis GDP. (As mentioned above, other exogenous parameters such as demand and stock of capital can be used in future work.) The model is run repeatedly as parameter values change over time and the output is then integrated to estimate GDP over time.

Note that the code base currently contains a simplified version of the above model based on linear programming (LP) which scales economic output linearly for a given supply of labour by sector, region and age. A full nonlinear programming (NLP) implementation including the Cobb-Douglas production function is in development.

### Corporate Bankruptcy Model

The objective of the bankruptcy model is to model the proportion of companies going insolvent and corresponding reduction to GDP  over time. In this context, it is crucial to distinguish between the probability of a specific company defaulting and the proportion of companies in the population that default. For this purpose we apply the log-logistic (Fisk) distribution to model the variation in number of days until companies go insolvent; this distribution is defined by two parameters (shape and scale) and is commonly used in survival analysis, including in the context of business survival [5,6]. 

The shape parameter of the distribution is assumed to be constant across sectors and consistent before, during, and after the epidemic. The scale parameter is modelled as a function of the mean operating surplus as well as the mean cash buffer that firms have in a given sector. Operating surplus is taken as given by the input-output model, with additional work underway to leverage the granular modelling of primary inputs available in the input-output model. The cash buffer is treated as exogenous parameter that needs to be calibrated. Care is taken to distinguish between small and medium enterprises (SMEs) and large corporations as the survival rates and GDP contributions are quite different for these groups of companies.

One important limitation is that the model currently does not account for interest payments, which form a crucial fixed payment, especially for highly-leverages firms and will be addressed in future work. Another limitation, already mentioned above, is that government support programs are currently not modelled explicitly.

### Individual Insolvency Model

We model the probability of individual insolvency over time, as a function of credit quality by region. The distribution of credit quality within each region is assumed to take the form of a Gaussian mixture model (GMM). The different classes of the GMM are associated with the employment status of an individual, which can be working, furloughed or unemployed. The shape parameter of the Gaussian distributions is fixed. The location parameter is a piecewise linear function of the initial credit quality, before the onset of the simulation, and the cumulative balance of savings the individual has. The balance itself takes savings at the start of the simulation as well as cumulative earnings over the course of the simulation into account. Here, earnings are a function both of the employment status and initial earnings at the start of the simulation. The corporate default rate provided by the bankruptcy model is used to simulate transitions between employment states. Note that this is not an agent-based model, i.e., we are not keeping track of balances by individual but only by group, which limits the ability of the model to simulate the effects of changes in employment status. To address this, we smooth the weights used in the mixture model over time. A more granular modelling of these dynamics is left for future work.

Key limitation of the model is that apart from furloughing, benefits and Covid-19-specific government support programs are not taken into account. Also, as mentioned above, the labour market itself is not modelled, i.e., workers do not have the ability to switch jobs, or find new employment if their previous firm went bankrupt. There is currently no modelling of self-employment, part-time work or short-time work. Finally, while the individual insolvency model does explicitly simulate spending, this is not currently fed back to the demand parameter of the input-output model. All of these are areas of future work.

### Long-term Growth Model

The input-output model simulates economic output as a ratio of GDP at the start of the simulation. This ratio is then discounted based on corporate bankruptcies over time. Taken together, this produces a simulation of the shortfall in economic output due to the epidemic, at a short time horizon. By construction, it does not take growth dynamics into account. As a simple baseline model for economic growth we model the long-term GDP trend by assuming a fixed annual growth rate that persist throughout the simulation, except during lockdown periods. The level of economic output implied by this long term growth model is then multiplied by the discounted GDP ratio supplied by the shortfall model. Note that the shortfall model will dominate the results in the short term.

Modelling long-term growth is a key area for improvement in future work. On the one hand, the long-term growth model itself can be improved considerably, by adopting a more thorough autoregressive approach than assuming fixed growth. On the other hand, as mentioned above, a proper modelling of demand as part of the input-output model will provide more realistic long-term dynamics.

## Parameters

As referred to in the model description above, there are a large number of parameters on which the simulation model depends. These parameters need to be calibrated carefully for the mode to produce realistic results. **Importantly, the code currently does not provide default values for these parameters. Setting parameters is left to the user.** In some places, we may provide *randomized parameters* as a convenience for users when running the code. However, the outputs generated based on randomized parameters should *not* be interpreted as economically meaningful, either in qualitative or quantitative terms.

In some areas the current code does use parameters calibrated on data from the Office for National Statistics. This data is used on the basis of the Open Government License v3.0.

## References

1. Eurostat, *Eurostat Manual of supply, use and input-output tables*, Office for Official Publications of the European Communities, Luxembourg, 2008.
2. R. Wild, "United Kingdom Input-Output Analytical Tables 2010", Office for National Statistics, 2014.
3. H. Fadinger, C. Ghiglino, and M. Teteryatnikova, "Income Diﬀerences and Input-Output Structure", p. 55, 2016.
4. D. Acemoglu, V. M. Carvalho, A. Ozdaglar, and A. Tahbaz-Salehi, "The Network Origins of Aggregate Fluctuations", *Econometrica*, vol. 80, no. 5, pp. 1977–2016, 2012.
5. T. Mahmood, "Survival of newly founded businesses: A log-logistic model approach", *Small Business Economics*, vol. 14, no. 3, pp. 223-237, 2000.
6. P. Fisk, "The graduation of income distributions", Econometrica, vol. 29, no. 2, pp. 171-185, 1961.



## Disclaimers

Contains public sector information licensed under the [Open Government Licence v3.0](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

The data presented herein is solely for illustrative purposes and may include among other things modelling, back testing, simulated results and scenario analyses. The information is based upon certain factors, assumptions and historical information that Goldman Sachs may in its discretion have considered appropriate, however, Goldman Sachs provides no assurance or guarantee that this model will operate or would have operated in the past in a manner consistent with these assumptions. In the event any of the assumptions used do not prove to be true, results are likely to vary materially from the examples shown herein. Additionally, the results may not reflect material economic and market factors, which could impact results.
