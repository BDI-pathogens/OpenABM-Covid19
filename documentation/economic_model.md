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


### Parameterization of Labour


In the model, every member of the workforce can be in one of four employment states

* $\text{wfo}$ - employed and not constrained to work from home
* $\text{wfh}$ - employed and constrained to work from home
* $\text{furloughed}$ - furloughed (but not unemployed)
* $\text{unemployed}$ - unemployed (but not furloughed)

In addition, ever individual in the population can be in one of three health states

* $\text{healthy}$ - healthy and able to work
* $\text{ill}$ - ill and unable to work
* $\text{dead}$ - deceased 

Taking combinations of these states yields 9 mutually exclusive and collectively exhaustive states a member of the workforce can be in: $W=\{ \text{healthy wfo}, \text{healthy wfh}, \text{healthy furloughed}, \text{healthy unemployed}, \text{ill wfo}, \text{ill wfh}, \text{ill furloughed}, \text{ill unemployed}, \text{dead}\}$. For each $w\in W$ we define a variable $\lambda^w_{r,s,a}$ which denotes the proportion of workers in region $r$, sector $s$, and age group $a$ who are in state $w$. By construction $\sum_w \lambda^w_{r,s,a} = 1$ for all triples $r,s,a$. Note that we also write $\lambda^w_{s}$ to denote the proportion for the given sector $s$, sutiably aggregated across regions and age groups.


For the purposes of the below models, it will be convenient to define another parameterization of workforce utilisations, which impose conditional structure on these utilisation vectors. We denote this alternative parameterization with $p$. Omitting the indices for region, sector and age group, the parameterization $p$ is defined by 

$$p^\text{dead} = \lambda^{\text{dead}}$$
$$(\lambda^{\text{ill wfo}} + \lambda^{\text{healthy wfo}}) p^\text{ill wfo} = \lambda^{\text{ill wfo}}$$
$$(\lambda^{\text{ill wfh}} + \lambda^{\text{healthy wfh}}) p^\text{ill wfh} = \lambda^{\text{ill wfh}}$$
$$(\lambda^{\text{ill furloughed}} + \lambda^{\text{healthy furloughed}}) p^\text{ill furloughed} = \lambda^{\text{ill furloughed}}$$
$$(\lambda^{\text{ill unemployed}} + \lambda^{\text{healthy unemployed}}) p^\text{ill unemployed} = \lambda^{\text{ill unemployed}}$$
$$(\lambda^{\text{ill wfh}} + \lambda^{\text{healthy wfh}} + \lambda^{\text{ill wfo}} + \lambda^{\text{healthy wfo}}) p^\text{wfh} = \lambda^{\text{ill wfh}} + \lambda^{\text{healthy wfh}}$$
$$(\lambda^{\text{ill furloughed}} + \lambda^{\text{healthy furloughed}} + \lambda^{\text{ill unemployed}} + \lambda^{\text{healthy unemployed}}) p^\text{furloughed} = \lambda^{\text{ill furloughed}} + \lambda^{\text{healthy furloughed}}$$
$$(1- \lambda^\text{dead}) p^\text{furloughed} = \lambda^{\text{ill furloughed}} + \lambda^{\text{healthy furloughed}} + \lambda^{\text{ill unemployed}} + \lambda^{\text{healthy unemployed}}$$


Finally, we define abbreviations
* $\lambda^{\text{working}} = \lambda^{\text{healthy wfo}}$
* $\lambda^{\text{wfh}} = \lambda^{\text{healthy wfh}}$
* $\lambda^{\text{ill}} = \lambda^{\text{ill wfo}} + \lambda^{\text{ill wfh}}$


### Input-Output Model


The foundation of our model of sectoral interdependencies is the European System of Accounts and in particular the supply and use tables and input-output analytical tables (IOATs) as set out by Eurostat [1] and applied to the UK by the ONS [2]. This framework allows to express total demand for a given product, on the one hand, as the sum of primary inputs and intermediate inputs to production at basic prices and, on the other hand, as the sum of intermediate demand and final uses. Primary inputs are, broadly, broken down into imports, taxes, compensation and operating surplus. Final uses are broken down into final consumption, capital formation and exports. This accounting perspective provides a common reference points that can be referred to across component models, from both a production and consumption perspective. In particular, the matrix of intermediate demand reflects sectoral interdependencies in the UK economy. Note that while IOATs typically categorize demand by product, we assume a categorization by sector, according to the UK's Standard Industrial Classification (SIC 2007), particularly A21 in NACE Rev. 2, see [2, Annex A].

To turn this accounting framework into a model suitable for simulation, we need to specify a production function that, in particular, allows for substitution of one input to production with another, at a cost. Following [3], we adopt (a piecewise-linear approximation of) a Cobb-Douglas production function that takes intermediate inputs as well as imports, labour and (fixed) capital as arguments. The IOATs are used to calibrate weights in the Cobb-Douglas function. Here we make the crucial assumption that prices for goods are constant. The price of labour responds to exogenous changes to both quantity of labour and productivity imposed by illness or government measures such as lockdown, by region, sector and age-group. This is in contrast to the modelling of capital, where we assume that the stock of capital is fixed at a level given exogenously and the return on capital varies according to the model. Final use is capped at a level given exogenously. See also [4] for further background on modelling of network effects in the economy.

The input-output model currently does not have an explicit representation of time. It takes as parameters the supply of labour, as derived from the output of the epidemiological model, and simulates the economic output as a fraction of pre-crisis GDP. (As mentioned above, other exogenous parameters such as demand and stock of capital can be used in future work.) The model is run repeatedly as parameter values change over time and the output is then integrated to estimate GDP over time.

This model is formulated as a linear optimization problem that is then solved using linear programming, in particular the `linprog` solver in `scipy`. Note that moving beyond our piecewise linear approximation of a Cobb-Douglas production function to the non-linear version, would require solving a non-linear optimization problem (NLP) instead. For this, the NLP solver in `scipy` is not sufficient. In future work, we recommend adopting `pyomo` to generate the NLP and then using `IPOPT` or a commercial solver for optimization.


#### Notation and Definitions


Most variables and quantities are defined with reference the ONS article on Input-Output Tables [2]. Note that we take the IOATs as being organized industry by industry rather than product by product and assume the tables have been transformed accordingly. The industry classification we use is SIC 2007, NACE Rev. 2 level A21.


With this background, we write:

* $i,j,k$ industrial sectors.
* $\tilde{d}_{i,j}$ input from product $i$ to product $j$ at base prices.
* $\tilde{q}_i$ total demand for product $i$, i.e., total output of product $i$ at base prices
* $\tilde{y}_i$ final use for product $i$ at base prices
* $\tilde{x}_i$ primary inputs to product $i$ at base prices


This allows us to restate key accounting identities from [2] as follows:

$$\underbrace{\text{total demand}_i}_{\tilde{q}_i} = \underbrace{\text{intermediate demand}_i}_{\sum_j {\tilde{d}_{i,j}}} + \underbrace{\text{final use}_i}_{\tilde{y}_i}$$
$$\underbrace{\text{final use}_i}_{\tilde{y}_i} = \text{final consumption}_i + \text{capital formation}_i + \text{exports}$$
$$\underbrace{\text{total demand}_i}_{\tilde{q}_i} = \underbrace{\text{intermediate inputs}_i}_{\sum_j \tilde{d}_{j,i}} + \underbrace{\text{primary inputs}_i}_{\tilde{x}_i}$$
$$ \text{primary inputs}_i = \text{imports}_i + \text{taxes on products}_i + \text{value add}_i$$
$$ \text{value add}_i = \text{taxes on production}_i + \text{compensation}_i + \text{gross operating surplus}_i$$
$$ \text{gross operating surplus}_i = \text{consumption of fixed capital}_i + \text{net operating surplus}_i$$
$$\sum_j \tilde{d}_{i,j} + \tilde{y}_i = \sum_j \tilde{d}_{j,i} + \tilde{x}_i \text{ for all }i$$
$$\sum_i \tilde{x}_i = \sum_i \tilde{y}_i$$
$$\text{GDP} \approx \text{GVA} = \sum_i\text{value add}_i = \sum_i\text{final use}_i - \sum_i\text{imports}_i -\sum_i \text{taxes on products}_i$$


Note also that there is also direct final use of imports, which is not reflected in the above. We will limit ourselves to considering imports only as inputs to domestic production.


Drawing on Fadinger et al. [3], we We combine the above accounting identities with a Cobb-Douglas production function in order to model substitutability of inputs into production. Crucial difference between production functions and accounting identities is that production functions take as input the *quantity* of a given good, while accounting identities are stated in terms of market *value*, where $\text{value}=\text{price}\cdot\text{quantity}$. We will make use of the values stated in the National Accounting section above as follows in the production function and the accounting function. 

| National Accounting | Production Function (qty) | Accounting Function (£) |
| --- | --- | --- |
| Imports | Imports (I) | Imports (I) |
| Taxes on products | - | Other |
| Taxes on production | - | Other |
| Compensation | Labour (L) | Compensation (L) |
| Consumption of fixed capital| Capital (K) | Cost of Capital (K) |
| Net Operating Surplus | Capital (K) | Cost of Capital (K)|



To set up the notation, we introduce a set of labels $M=\{I,L,K\}$, following the above table, and variables

* $\tilde{x}_{m,i}$ value of primary input $m\in M$ to sector $i$
* $\tilde{o}_i$ value of other primary inputs to sector $i$


On the final use side of things, we introduce labels $U=\{C,K,E\}$ to denote consumption, capital formation and exports, respectively and define $\tilde{y}_{u,i}$ to denote value of final use of type $u\in U$ of product $i$.


For variables $\tilde{d}_{i,j}$, $\tilde{x}_{m,i}$, $\tilde{y}_{u,i}$ and $\tilde{q}_i$ defined above, we introduce variables $d_{i,j}$, $x_{m,i}$, $y_{u,i}$ and $q_i$ that represent the associated *quantities*. Crucially, we assume prices for all good and for imports are equal to 1 (and, thus, constant), whence $\tilde{d}_{i,j} = d_{i,j}$, $\tilde{y}_{u,i} = y_{u,i}$,  $\tilde{q}_i= q_i$ and $\tilde{x}_{I,i} =x_{I,i}$. Prices for labour and capital are not assumed to be constant.


In this setting, the Cobb-Douglas production function is 
$$ f^{\text{CD}}(d,x) := \Lambda_i \prod_j d_{j,i}^{\gamma_{j,i}} \prod_{m\in M} x^{\gamma_{m,i}}_{m,i}$$
where 
* $\Lambda_i$ total factor productivity in sector $i$
* $\gamma_{i,j}$ and $\gamma_{m,j}$ for sectors $i,j\in N$ and primary inputs $m\in M$ such that $\sum_{j\in N} \gamma_{j,i} + \sum_{m\in M} \gamma_{m,i} = 1$.


For computational efficiency, we employ a piecewise-linear approximation of the Cobb-Douglas function in the below optimization problem. This approximation is given by the right hand side in the inequality
$$f^{\text{PLCD}}(d,x) := \Lambda_i \left((1-\rho) \min\left(
                                \min_j\left(
                                    \frac{d_{j,i}}{\gamma_{j,i}}
                                \right),
                                \min_m\left(
                                    \frac{x_{m,i}}{\gamma_{m,i}}
                                \right)
                             \right)
                      + \rho \left(\sum_j \gamma_{j,i} d_{j,i} + \sum_m \gamma_{m,i} d_{m,i} \right)\right) $$
where $\rho\in[0,1]$ is a constant.


Many parameters will be defined relative to pre-crisis values for the economy. To this end, we use the latest IOATs from the ONS. Wherever we refer to a constant taken from the IOATs, we will denote this with the superscript $\text{iot}$. For example, $d_{i,j}^{\text{iot}}$ would refer to the value of $d_{i,j}$ as given in the ONS's IO Tables, and $d_{i,j} \leq d_{i,j}^{\text{iot}}$ would express that $d_{i,j}$ in the model should be at most the constant $d_{i,j}^{\text{iot}}$.


From the ONS IO Tables, we can obtain $\tilde{d}_{i,j}^{\text{iot}}$, $\tilde{y}_{i}^{\text{iot}}$ and $\tilde{x}_{L,i}^{\text{iot}}$, $\tilde{x}_{I,i}^{\text{iot}}$ directly. We also define $$o_i^{\text{iot}}=\text{taxes on products}_i^{\text{iot}}+\text{taxes on production}_i^{\text{iot}}$$ and $$\tilde{x}_{K,i}^{\text{iot}} = \text{consumption of fixed capital}_i^{\text{iot}} + \text{net operating surplus}_i^{\text{iot}}.$$


#### Linear Program


With the above setup, we can now proceed to defining the linear program.


##### Parameters


The following parameters are assumed to given, by other model components or set by the user.


* $p^w_i$ labour utilisation parameters as defined in previous section for $w\in \{\text{wfh},\text{ill wfo},\text{ill wfh},\text{ill dead}\}$ and sectors $i$,
* $\kappa_i$ ratio of fixed capital, relative to the pre-crisis level, available in sector $i$
* $\delta_{u,i}$ upper bound on ratio of final use $u\in\{C, K, E\}$ in sector $i$, relative to pre-crisis level (constraint on demand)
* $\tau$ tax *rate* on both products and production, expressed as a ratio of the pre-crisis level
* $h_i\in[0,1]$ the productivity when constrained to working from home in sector $i$, as a ratio of unconstrained productivity
* $\rho\in[0,1]$ a substitution rate controlling the shape of the piecewise linear approximation to Cobb-Douglas
* $b_i\in[0,1]$ lower bound on proportion of people not employed (furloughed or unemploymed) in sector $i$


##### Variables


Using the notation from above, variables in the below are $q_i$, $d_{i,j}$, $x_{I,i}$, $x_{L,i}$, $x_{K,i}$, $\tilde{x}_{L,i}$, $\tilde{x}_{K,i}$, $y_{i}$, $\lambda^{\text{working}}$, $\lambda^{\text{wfh}}$, $\lambda^{\text{ill}}$.


##### Constants


$$\gamma_{i,j} = \frac{\tilde{d}_{i,j}^{\text{iot}}}{\sum_i \tilde{d}_{i,j}^{\text{iot}} + \sum_m \tilde{x}_{m,j}^{\text{iot}}}$$

$$\gamma_{m,j} = \frac{\tilde{x}_{m,j}^{\text{iot}}}{\sum_i \tilde{d}_{i,j}^{\text{iot}} + \sum_m \tilde{x}_{m,j}^{\text{iot}}}$$

$$\Lambda_i = \frac{
\frac{1}
              {1-
                  \frac{o_i^{\text{iot}}}
                       {q_i^{\text{iot}}}\tau
              }
\left(\sum_j \tilde{d}_{j,i}^{\text{iot}} + \sum_m \tilde{x}_{m,i}^{\text{iot}}\right)}{\prod_j (\tilde{d}_{j,i}^{\text{iot}})^{\gamma_{j,i}} \prod_{m\in M} (x^{\text{iot}}_{m,i})^{\gamma_{m,i}}}  $$

$$f^{\text{wfo}}_i = (1-p^{\text{wfh}}_i)(1-p^{\text{ill wfo}}_i)(1-p^{\text{dead}}_i)$$
$$f^{\text{wfh}}_i = p^{\text{wfh}}_i(1-p^{\text{ill wfh}}_i)(1-p^{\text{dead}}_i)$$
$$f^{\text{ill}}_i = (p^{\text{ill wfo}}_i + p^{\text{ill wfh}}_i)(1-p^{\text{dead}}_i)$$


##### Objective


$$ \max \left(\sum_i \tilde{x}_{L,i} + \tilde{x}_{K,i} + \tau q_i \frac{\text{taxes on production}_i^{\text{iot}}}{q_i^{\text{iot}}}\right) + \left( \sum_i \tilde{x}_{K,i} \right)$$


##### Constraints


$$ q_i \leq f^{\text{PLCD}}(d,x)$$

$$ q_i = \frac{1}
              {1-
                  \frac{o_i^{\text{iot}}}
                       {q_i^{\text{iot}}}\tau
              }
              \left(\sum_j d_{j,i} + x_{I,i} + \tilde{x}_{L,i} + \tilde{x}_{K,i}\right) $$
$$ q_i = \sum_j d_{i,j} + \sum_u y_{u,i} $$

$$ x_{K,i} \leq \kappa_i $$

$$ y_i \leq \sum_u \delta_{u,i} \tilde{y}^{\text{iot}}_{u,i}$$

$$ x_{L,i} = \left(\lambda^{\text{working}}_i + h_i\lambda^{\text{wfh}}_i\right) \tilde{x}_{L,i}^{\text{iot}} $$

$$ \tilde{x}_{L,i} = \left( \lambda^{\text{working}}_i + \lambda^{\text{wfh}}_i + \lambda^{\text{ill}}_i \right) \tilde{x}_{L,i}^{\text{iot}}$$

$$ \lambda^{\text{working}}_i \leq (1-b_i) f^\text{wfo}_i $$
$$ \lambda^{\text{wfh}}_i \leq (1-b_i) f^\text{wfh}_i $$
$$ \lambda^{\text{ill}}_i \leq (1-b_i) f^\text{ill}_i $$

$$ \lambda^{\text{wfh}}_i f^\text{wfo}_i = \lambda^{\text{working}}_i f^\text{wfh}_i $$
$$ \lambda^{\text{wfh}}_i f^\text{ill}_i = \lambda^{\text{ill}}_i f^\text{wfh}_i $$
$$ \lambda^{\text{working}}_i f^\text{ill}_i = \lambda^{\text{ill}}_i f^\text{wfo}_i $$


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
\frac{1}{L} \cdot \frac{N_{r^{*}, s^{*}}}{\sum_{r, s}N_{r, s}} 
\thinspace \forall l \thinspace \text{in  \{1,2,...,L\}}$$


$$m_{r, s}^l(t) = 
\text{Credit Score}_r(0) + 
\alpha \cdot \delta \text{Balance}_{r,s} ^ l(t) + 
\beta \cdot \min(\text{Balance}_{r, s}^l(t), 0)$$


The average balance per region, sector, decile itself takes savings at the start of the simulation as well as cumulative earnings over the course of the simulation into account. We analyze earnings and spendings separately to study the diffrent impact received. 


$$\text{Balance}_{r, s}^l(0) = \text{Saving}_{r, s}^l$$


$$\text{Balance}_{r, s}^l(t) = 
\text{Balance}_{r, s}^l(t-1) + 
\delta \text{Balance}_{r,s} ^ l(t)$$


$$\delta \text{Balance}_{r,s} ^ l(t) = 
\text{Earning}_{r, s}^c(t) - \text{Spending}_{r, s}^c(t)$$


Here, earnings are a function both of the employment status and initial earnings at the start of the simulation. We define a fixed parameter $\eta^c$ to introduce the impact on people's earning given different employment status. The corporate default rate provided by the bankruptcy model is used to simulate transitions between employment states.


$$\text{Earning}_{r, s}^l(t) =
\sum_c \lambda_{r, s}^c(t) \cdot
\eta^c \cdot
\text{Earning}_{r, s}^l(0)
$$


We further breakdown spendings into different expense sector($k$). Expenditure in each sector takes function of reduction in income and fear factor into account, bounded by the minimum level of spending to cover basic living cost. The expense from the population that are no longer capable of spending are carefully removed.


$$\text{Spending}_{r, s}^l(t) = \sum_k \text{Spending}_{r, s, k}^l(t)$$


$$\text{Spending}_{r, s, k}^l(t) =
\max\left(
\text{Spending}_{r, s, k}^l(0) \cdot \frac{\text{Earning}_{r, s}^l(t)}{\text{Earning}_{r, s}^l(0)} \cdot (1 - \text{Fear factor}),
\text{Minimum Spending}_{r, s, k}^l
\right) \cdot (1 -  \lambda_{r, s}^\text{Dead}(t))
$$


We compare the spot spending at any point in time during the simulation to estimate the reduction in demand.


$$\text{Demand reduction}_k(t) = 1 - 
\frac
{\sum_r \sum_s \sum_l \text{Spending}_{r, s, k}^l(t)}
{\sum_r \sum_s \sum_l \text{Spending}_{r, s, k}^l(0)}$$


Note that this is not an agent-based model, i.e., we are not keeping track of balances by individual but only by group, which limits the ability of the model to simulate the effects of changes in employment status. A more granular modelling of these dynamics is left for future work.

Key limitation of the model is that apart from furloughing, benefits and Covid-19-specific government support programs are not taken into account. Also, as mentioned above, the labour market itself is not modelled, i.e., workers do not have the ability to switch jobs, or find new employment if their previous firm went bankrupt. There is currently no modelling of self-employment, part-time work or short-time work. Finally, while the individual insolvency model does explicitly simulate spending, this is not currently fed back to the demand parameter of the input-output model. All of these are areas of future work.


### Fear Factor


In general, we expect people to be less motivated to go out and spend when they are mentally scared due to the coronavirus, even they are pysically health and well employed. Hence, we define a fear factor which linearly regresses on lockdown status, the number of ill people and the new dead cases.


$$\text{Fear factor}(t) = w_1 \text{Lock down}(t) + 
   w_2 N_{\text{ill}}(t) + 
    w_3 \delta N_{\text{dead}}(t)$$


### Feedback Loops


**TODO**


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
