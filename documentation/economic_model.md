# AdaptER-Covid19


## Introduction


Accurate epidemiological simulations are a key tool for understanding, responding to and managing the Covid-19 pandemic. While understanding health-outcomes of policy interventions is rightly the primary focus of such simulations, there are clear benefits to a holistic simulation that takes into account economic dynamics as well. Created for this purpose, AdaptER-Covid19 (Adaptive Economic Response) is an economic simulation that integrates with the OpenABM-Covid19 simulator in order to jointly model health and economic outcomes of epidemics, with a particular focus on Covid-19.

AdaptER-Covid19 takes as input from the epidemiological part of the model timeseries describing the simulated course of the epidemic (e.g. number of symptomatic individuals) as well as exogenous variables (e.g. whether a lockdown is in place in the given scenario) and simulates the impact on key economic metrics such as GDP, corporate bankruptcy rate and individual insolvency rate, unemployment, household expenditure, private sector investment, opportunity gaps betweeen supply and demand per sector and behavioural changes caused by the epidemic.

It should also be noted that AdaptER-Covid19 is a simulator rather than a forecast. Its outputs are dependent on both the mathematical model as well as a host of parameter choices on which limited data is available (e.g. worker productivity by region and sector when limited to working from home) and which may vary considerably across scenarios. As such, simulation results from AdaptER-Covid19 may be used to draw qualitative conclusions when comparing different scenarios rather than for quantitative estimates or inference of absolute numbers. To allow the user to properly evaluate simulation results, we describe the current model in detail below.


## Model Overview


The epidemic model produces detailed timeseries of the health impact on the population by region, age and economic sector that are then fed into the economic model at a daily frequency. Government interventions allow to change the trajectory of the epidemic and the economy. Some interventions, such as the imposition of a lockdown, have a direct effect on both the epidemic and the economic model, while other only affect one of the two models directly (e.g. the effect of furloughing on the economic model). Since the joint model runs daily, it is possible to simulate intervention policies that are responsive to events in the epidemic or the economy. Howeverm there is currently not direct feedback loop from the economic model back to the epidemic model.

The economic model is broken down into three main parts. 

* The Input-Output Model, or simply GDP Model, links labour, capital and intermediate inoputs to production to the intermediate and final outputs, while taking interdependencies between sector of industry into account. In particular, it also determines the level of employment. The Input-Output Model is formulated as a constrained optimization problem, solved using linear programming techniques.
* The Corporate Bankruptcy Model is an agent-based model simulating corporate defaults. It is connected to the IO-Model through the net operating surplus of companies, the stock of capital available for production and the level of unemployment caused by corporate bankruptcies. 
* The Individual Insolvency Model takes household earnings and behaviour into account to determine aggregate levels of demand as well as the risk of households becoming insolvent. 

These components are complemented by a "fear factor" driving behavioural changes in household consumption and risk appetite of private sector investors, as well as an "opportunity gap" metric which measures the gap between supply and demand per sector.

Some preliminary calibration of the model has been performed using public data sets for UK available from the Office for National Statistics (ONS). However, many further parameters require careful tuning to create accurate simulation results. In the codebase, we have *randomized* initial settings for these parameters, to make it easy to run the model. These parameters will need to be set by the user  depending on the application. See also the section on parameters below.

Before we go into further detail on the component models, we point out key **limitations** in the current approach:

* Prices for goods and labour are assumed to be fixed. Inflation is not modelled.
* In particular, unemployment is determined based on the assumptions that wages are fixed and workers cannot change industries. A full model of a competitive labour market would be a useful addition to the model.
* Growth of capital in the economy is modelled through the agent-based corporate bankruptcy model as well as through a simplified model of investment by sector that combines long-term growth trends with opportunistic investment behaviour driven by risk appetite and the gap between supply and demand per sector. However, this is not a comprehensive model of the capital market. Also, the corporate bankruptcy model does currently not break down net operating surplus into interest payments and profits.
* Imports and exports are part of the input-output model, however they are currently assumed to be fixed. Connecting these to a model of the global economy would be a straightforward extension.

It is essential to keep current limitations of the simulator in mind when interpreting results. See also the section on directions for future work below.


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


We model the proportion of individual insolvency over time, as a function of credit quality by region.


$$P_r(t)=P(\text{Credit Score}_r(t) < \text{Default Threshold})$$


We assume Credit Score per region at any given time follows a mixture of Gaussian distribution, where 
- Each Guassian component represents one employed sector, and one decile of the population in that sector 
- The weight of each component equals the number of workers per region per sector, divided by the number of deciles
- The shape parameter of the all components is fixed fixed per region
- The location parameter is a piecewise linear function of the initial credit quality before the onset of the simulation, the cumulative balance of savings the individual has, and the current change in balance.


$$\text{Credit Score}_r(t) \sim 
\sum_l \sum_s w_{r, s}^l \mathcal{N}(m_{r, s}^l(t), \sigma_r)$$


$$w_{r^{*}, s^{*}}^l = 
\frac{1}{L} * \frac{N_{r^{*}, s^{*}}}{\sum_{r, s}N_{r, s}} 
\thinspace \forall l \thinspace \text{in  \{1,2,...,L\}}$$


$$m_{r, s}^l(t) = 
\text{Credit Score}_r(0) + 
\alpha * \delta \text{Balance}_{r,s} ^ l(t) + 
\beta * \min(\text{Balance}_{r, s}^l(t), 0)$$


The average balance per region, sector, decile itself takes savings at the start of the simulation as well as cumulative earnings over the course of the simulation into account. We analyze earnings and spendings separately to study the diffrent impact received. 


$$\text{Balance}_{r, s}^l(0) = \text{Saving}_{r, s}^l$$


$$\text{Balance}_{r, s}^l(t) = 
\text{Balance}_{r, s}^l(t-1) + 
\delta \text{Balance}_{r,s} ^ l(t)$$


$$\delta \text{Balance}_{r,s} ^ l(t) = 
\text{Earning}_{r, s}^c(t) - \text{Spending}_{r, s}^c(t)$$


Here, earnings are a function both of the employment status and initial earnings at the start of the simulation. We define a fixed parameter $\eta^c$ to introduce the impact on people's earning given different employment status. The corporate default rate provided by the bankruptcy model is used to simulate transitions between employment states.


$$\text{Earning}_{r, s}^l(t) =
\sum_c \lambda_{r, s}^c(t) * 
\eta^c * 
\text{Earning}_{r, s}^l(0)
$$


We further breakdown spendings into different expense sector($k$). Expenditure in each sector takes function of reduction in income and fear factor into account, bounded by the minimum level of spending to cover basic living cost. The expense from the population that are no longer capable of spending are carefully removed.


$$\text{Spending}_{r, s}^l(t) = \sum_k \text{Spending}_{r, s, k}^l(t)$$


$$\text{Spending}_{r, s, k}^l(t) =
\max(
\text{Spending}_{r, s, k}^l(0) * \frac{\text{Earning}_{r, s}^l(t)}{\text{Earning}_{r, s}^l(0)} * (1 - \text{Fear factor}),
\text{Minimum Spending}_{r, s, k}^l
) * (1 -  \lambda_{r, s}^\text{Dead}(t))
$$


We compare the spot spending at any point in time during the simulation to estimate the reduction in demand.


$$\text{Demand reduction}_k(t) = 1 - 
\frac
{\sum_r \sum_s \sum_l \text{Spending}_{r, s, k}^l(t)}
{\sum_r \sum_s \sum_l \text{Spending}_{r, s, k}^l(0)}$$


Note that this is not an agent-based model, i.e., we are not keeping track of balances by individual but only by group, which limits the ability of the model to simulate the effects of changes in employment status. A more granular modelling of these dynamics is left for future work.

Key limitation of the model is that apart from furloughing, benefits and Covid-19-specific government support programs are not taken into account. Also, as mentioned above, the labour market itself is not modelled, i.e., workers do not have the ability to switch jobs, or find new employment if their previous firm went bankrupt. There is currently no modelling of self-employment, part-time work or short-time work. Finally, while the individual insolvency model does explicitly simulate spending, this is not currently fed back to the demand parameter of the input-output model. All of these are areas of future work.


### Long-term Growth Model


The input-output model simulates economic output as a ratio of GDP at the start of the simulation. This ratio is then discounted based on corporate bankruptcies over time. Taken together, this produces a simulation of the shortfall in economic output due to the epidemic, at a short time horizon. By construction, it does not take growth dynamics into account. As a simple baseline model for economic growth we model the long-term GDP trend by assuming a fixed annual growth rate that persist throughout the simulation, except during lockdown periods. The level of economic output implied by this long term growth model is then multiplied by the discounted GDP ratio supplied by the shortfall model. Note that the shortfall model will dominate the results in the short term.

Modelling long-term growth is a key area for improvement in future work. On the one hand, the long-term growth model itself can be improved considerably, by adopting a more thorough autoregressive approach than assuming fixed growth. On the other hand, as mentioned above, a proper modelling of demand as part of the input-output model will provide more realistic long-term dynamics.


## Parameters


As referred to in the model description above, there are a large number of parameters on which the simulation model depends. These parameters need to be calibrated carefully for the mode to produce realistic results. **Importantly, the code currently does not provide default values for these parameters. Setting parameters is left to the user.** In some places, we may provide *randomized parameters* as a convenience for users when running the code. However, the outputs generated based on randomized parameters should *not* be interpreted as economically meaningful, either in qualitative or quantitative terms.

In some areas the current code does use parameters calibrated on data from the Office for National Statistics. This data is used on the basis of the Open Government License v3.0.


## Directions for Future Work


As indicated above, several directions for future enhancements of the model present themselves, apart from calibration and parameter tuning. First of all, there are a number of iterative enhancements to the model that will be of immediate benefit:

* Tighter integration with spread model, in particular with regard to mapping of agent participation in networks in the spread model to worker states, number of key workers, etc. in the economic model.
* Simulation scenarios that drive both the epidemic and the economic interventions in response to events in the simulation.
* Improvement of runtime performance (i.e. increase of speed of execution and reduction of memory consumption). Only very limited effort has been spent so far on optimizing the implementation, and there are many areas that can be improved.
* There is a wide range of economic variables that are already modelled, but that are currently simply assumed to be constant. Examples include imports and exports or government spending as a final use (source of demand). Exposing these variables to simulation scenarios and then using these to drive the simulation is a straightforeward extension.

Going beyond these iterative improvements, there is also a range of major extensions of the model that would be worthwhile:

* The input model currently uses a piecewise-linear production function. Switching the model to the (non-linear) Cobb-Douglas production function would model substitutability of inputs to production more realistically, as it would make small qunatities of substition more cost effective than in the current model, but large substitutions much more expensive. This would imply replacing the LP solver with an NLP solver, likely using a different solver library than Scipy (e.g. using Pyomo as a generic solver interface).
* As mentioned in the section on limitations, the simulator currently uses simplified models of labour and capital markets. Replacing these with competitive equilibrium models of labour and captial markets would enhance model fidelity. 
* Switching to such competitive equilibrium models would also enable allowing prices to vary, leading to a simulation of inflation.

Finally, in the long-run, an uplift of the data pipelines to incorporate granular and timely data and an automation of the calibration process would allow the model to update "live" in response to current events and simulate the effect of interventions in response to those events.


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
