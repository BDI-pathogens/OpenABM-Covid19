#!/usr/bin/env python3
"""
Plotting functions for output from the COVID19-IBM

Created: 30 March 2020
Author: p-robot
"""

import numpy as np, pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.stats import gamma
from pandas.api.types import CategoricalDtype
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os, re
from os.path import join

from constant import N_NETWORK_TYPES, HOUSEHOLD, WORK, RANDOM, EVENT_TYPES, EVENT_TYPE_STRING
network_colours = ['#009E73', '#0072B2', '#D55E00']

key_params = [
    "n_total", 
    "n_seed_infection", 
    "infectious_rate", 
    "asymptomatic_infectious_factor", 
    "mild_infectious_factor",
    "daily_non_cov_symptoms_rate"]

sensitivity_analysis_params = [
    "mean_time_to_symptoms",
    "fraction_asymptomatic_0_9",
    "fraction_asymptomatic_10_19",
    "fraction_asymptomatic_20_29",
    "fraction_asymptomatic_30_39",
    "fraction_asymptomatic_40_49",
    "fraction_asymptomatic_50_59",
    "fraction_asymptomatic_60_69",
    "fraction_asymptomatic_70_79",
    "fraction_asymptomatic_80",
    "mild_fraction_0_9",
    "mild_fraction_10_19",
    "mild_fraction_20_29",
    "mild_fraction_30_39",
    "mild_fraction_40_49",
    "mild_fraction_50_59",
    "mild_fraction_60_69",
    "mild_fraction_70_79",
    "mild_fraction_80",
    "daily_non_cov_symptoms_rate",
    "relative_susceptibility_0_9",
    "relative_susceptibility_10_19",
    "relative_susceptibility_20_29",
    "relative_susceptibility_30_39",
    "relative_susceptibility_40_49",
    "relative_susceptibility_50_59",
    "relative_susceptibility_60_69",
    "relative_susceptibility_70_79",
    "relative_susceptibility_80",
    "mean_infectious_period",
    "relative_transmission_household"]


network_labels = [ None ] * N_NETWORK_TYPES
network_labels[HOUSEHOLD] = "Household"
network_labels[WORK] = "Workplace"
network_labels[RANDOM] = "Random"


asymptomatic_cols = [
    "fraction_asymptomatic_0_9",
    "fraction_asymptomatic_10_19",
    "fraction_asymptomatic_20_29",
    "fraction_asymptomatic_30_39",
    "fraction_asymptomatic_40_49",
    "fraction_asymptomatic_50_59",
    "fraction_asymptomatic_60_69",
    "fraction_asymptomatic_70_79",
    "fraction_asymptomatic_80"]


susceptibility_cols = [
    "relative_susceptibility_0_9","relative_susceptibility_10_19",
    "relative_susceptibility_20_29", "relative_susceptibility_30_39", 
    "relative_susceptibility_40_49", "relative_susceptibility_50_59",
    "relative_susceptibility_60_69", "relative_susceptibility_70_79", 
    "relative_susceptibility_80"]

hospitalised_cols = [
    "hospitalised_fraction_0_9", "hospitalised_fraction_10_19",
    "hospitalised_fraction_20_29", "hospitalised_fraction_30_39", 
    "hospitalised_fraction_40_49", "hospitalised_fraction_50_59",
    "hospitalised_fraction_60_69", "hospitalised_fraction_70_79", 
    "hospitalised_fraction_80"]

fatality_cols = [
    "fatality_fraction_0_9", "fatality_fraction_10_19", 
    "fatality_fraction_20_29", "fatality_fraction_30_39", 
    "fatality_fraction_40_49", "fatality_fraction_50_59",
    "fatality_fraction_60_69", "fatality_fraction_70_79", 
    "fatality_fraction_80"]

critical_cols = [
    "critical_fraction_0_9", "critical_fraction_10_19",
    "critical_fraction_20_29", "critical_fraction_30_39", 
    "critical_fraction_40_49", "critical_fraction_50_59",
    "critical_fraction_60_69", "critical_fraction_70_79", 
    "critical_fraction_80"]

mild_cols = [
    "mild_fraction_0_9", "mild_fraction_10_19",
    "mild_fraction_20_29", "mild_fraction_30_39", 
    "mild_fraction_40_49", "mild_fraction_50_59",
    "mild_fraction_60_69", "mild_fraction_70_79", 
    "mild_fraction_80"]

app_users_cols = [
    "app_users_fraction_0_9", "app_users_fraction_10_19",
    "app_users_fraction_20_29", "app_users_fraction_30_39",
    "app_users_fraction_40_49", "app_users_fraction_50_59",
    "app_users_fraction_60_69", "app_users_fraction_70_79",
    "app_users_fraction_80"]

intervention_params = [
    "self_quarantine_fraction", 
    "quarantine_length_self", 
    "quarantine_length_traced", 
    "quarantine_length_positive", 
    "quarantine_dropout_self", 
    "quarantine_dropout_traced", 
    "quarantine_dropout_positive", 
    "test_on_symptoms", 
    "test_on_traced", 
    "quarantine_on_traced", 
    "traceable_interaction_fraction", 
    "tracing_network_depth", 
    "allow_clinical_diagnosis", 
    "quarantine_household_on_positive", 
    "quarantine_household_on_symptoms", 
    "quarantine_household_on_traced", 
    "quarantine_household_contacts_on_positive", 
    "quarantined_daily_interactions", 
    "quarantine_days", 
    "hospitalised_daily_interactions", 
    "test_insensitive_period", 
    "test_order_wait", 
    "test_result_wait", 
    "self_quarantine_fraction", 
    "app_users_fraction_0_9", 
    "app_users_fraction_10_19",
    "app_users_fraction_20_29",
    "app_users_fraction_30_39",
    "app_users_fraction_40_49",
    "app_users_fraction_50_59", 
    "app_users_fraction_60_69",
    "app_users_fraction_70_79",
    "app_users_fraction_80",
    "app_turn_on_time", 
    "lockdown_work_network_multiplier",
    "lockdown_random_network_multiplier",
    "lockdown_house_interaction_multiplier",
    "lockdown_time_on",
    "lockdown_time_off",
    "lockdown_elderly_time_on",
    "lockdown_elderly_time_off",
    "successive_lockdown_time_on",
    "successive_lockdown_duration",
    "successive_lockdown_gap",
    "testing_symptoms_time_on",
    "testing_symptoms_time_off",
    "intervention_start_time"]


def gamma_params(mn, sd):
    """
    Return scale and shape parameters from a Gamma distribution from input mean and sd
    
    Arguments
    ---------
    mn : float
        Mean of the gamma distribution
    sd : float
        Standard deviation of the gamma distribution
    """
    scale = (sd**2)/mn
    shape = mn/scale
    
    return(shape, scale)


def overlapping_bins(start, stop, window, by):
    """Generate overlapping bins"""
    
    bins = []
    for i in np.arange(start, stop - window + 1, step = by):
        bins.append((i, i + window))
    return(bins)


def get_discrete_viridis_colours(n):
    """
    Generate n colours from the viridis colour map
    """
    colourmap = cm.get_cmap('viridis', n)
    colours = [colourmap.colors[n - i - 1] for i in range(n)]
    return(colours)


def ProportionTransmissionsThroughTime(df_trans, groupvar, groups, infectiontimevar,
    start = 1, stop = 100, window = 5, ylims = None):
    """
    Plot proportion of transmissions through time according to disease state
    data are binned into a window of size 'window'
    
    
    Arguments
    ---------
    
    df_trans : pandas.DataFrame
        DataFrame of the transmission file output from COVID19-IBM
    groupvar : str
        Variable name specifying disease state of the infector, typically 'infector_status'
    groups : list
        List of disease states
    infectiontimevar : str
        Variable name specifying time of infection (default -1 if never infected)
    start : int
        Time at which to start plotting
    stop : int
        Time at which to stop plotting
    window : int
        Size of the rolling window over which to summarise transmissions
    
    Returns
    -------
    fig, ax : figure and axis handles to the generated figure using matplotlib.pyplot
    """
    
    cat_type = CategoricalDtype(categories = groups, ordered = False)
    df_trans[groupvar] = df_trans[groupvar].astype(cat_type)
    
    bins = overlapping_bins(start = start, stop = stop, window = window, by = 1)
    
    # Find proportion of transmissions from each "infector_status" through a sliding window
    output = []
    for b in bins:
        # Subset to window of interest
        condition1 = (df_trans[infectiontimevar] >= b[0])
        condition2 = (df_trans.time_infected < b[1])
        df_sub = df_trans[ condition1 & condition2 ]
        
        df_output = df_sub[groupvar].value_counts().reset_index()
        df_output.columns = [groupvar, "freq"]
        
        # Calculation proportion in each disease state
        df_output["proportion"] = df_output.freq/df_output.freq.sum()
        df_output["time"] = b[0]
        output.append(df_output)
    
    df = pd.concat(output)
    
    fig, ax = plt.subplots(nrows = 3)
    
    df_total = df.groupby("time")["freq"].sum().reset_index()
    
    ax[0].plot(df_total.time, df_total.freq, label = "Total incidence", lw = 3, c = "red")
    ax[0].set_xlim([start, stop])
    
    ax[0].set_xlabel(""); ax[0].set_ylabel(""); 
    ax[0].legend(prop = {'size':16})
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)
    
    for tick in ax[0].xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax[0].yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    
    # Add total non-symptomatic
    nonsymptomatic_status = [
        EVENT_TYPES.ASYMPTOMATIC.value, 
        EVENT_TYPES.PRESYMPTOMATIC.value, 
        EVENT_TYPES.PRESYMPTOMATIC_MILD.value]
    
    non = df[df[groupvar].isin(nonsymptomatic_status)]
    df_non = non.groupby("time")["proportion"].sum().reset_index()
    ax[1].plot(df_non.time, 100*df_non.proportion, 
        label = "Non-symptomatic (total)", lw = 3, c = "#E69F00")
    
    # Add total symptomatic
    symptomatic_status = [
        EVENT_TYPES.SYMPTOMATIC.value,
        EVENT_TYPES.SYMPTOMATIC_MILD.value]
    
    sym = df[df[groupvar].isin(symptomatic_status)]
    df_sym = sym.groupby("time")["proportion"].sum().reset_index()
    ax[1].plot(df_sym.time, 100*df_sym.proportion, 
        label = "Symptomatic (total)", lw = 3, c = "#D55E00")
    
    ax[1].set_xlabel(""); 
    ax[1].set_ylabel("Percent incidence\nover next 5 days\n", size = 16)
    ax[1].legend(prop = {'size':16})
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    
    for tick in ax[1].xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax[1].yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    
    ax[1].set_xlim([start, stop])
    
    if ylims:
        ax[1].set_ylim(ylims)
    
    # Subset to those with non-zero counts
    non_zero_status = [s for s in groups if df[df[groupvar] == s].freq.sum() > 0]
    
    for status in non_zero_status:
        df_plot = df[df[groupvar] == status]
        ax[2].plot(df_plot.time, 100*df_plot.proportion, 
            label = EVENT_TYPE_STRING[EVENT_TYPES(status).value], 
            lw = 3)
    
    ax[2].spines["top"].set_visible(False)
    ax[2].spines["right"].set_visible(False)
    ax[2].set_xlabel("Day since infection seeded", size = 20)
    ax[2].set_ylabel("Percent incidence\nover next 5 days\n", size = 16)
    ax[2].legend(prop = {'size':16})
    
    for tick in ax[2].xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax[2].yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    
    ax[2].set_xlim([start, stop])
    
    if ylims:
        ax[2].set_ylim(ylims)
    
    return(fig, ax)


def ParameterAssumptions(df_parameters, xlimits = [0, 30], lw = 3):
    """
    Plot distributions of mean transition times between comparments in the parameters
    
    Arguments
    ---------
    df_parameters : pandas.DataFrame
        DataFrame of parameter values as input first input argument to the COVID19-IBM model
    xlimits : list of ints
        Limits of x axis of gamma distributions showing mean transition times
    lw : float
        Line width used in plotting lines of the PDFs
    
    Returns
    -------
    fig, ax : figure and axis handles to the generated figure using matplotlib.pyplot
    """
    df = df_parameters # for brevity
    x = np.linspace(xlimits[0], xlimits[1], num = 50)
    
    fig, ax = plt.subplots(nrows = 3, ncols = 3)
    
    ####################################
    # Bernoulli of mean time to hospital
    ####################################
    
    height1 = np.ceil(df.mean_time_to_hospital.values[0]) - df.mean_time_to_hospital.values[0]
    height2 = df.mean_time_to_hospital.values[0] - np.floor(df.mean_time_to_hospital.values[0])
    
    x1 = np.floor(df.mean_time_to_hospital.values[0])
    x2 = np.ceil(df.mean_time_to_hospital.values[0])
    ax[0,0].bar([x1, x2], [height1, height2], color = "#0072B2")
    
    ax[0,0].set_ylim([0, 1.0])
    ax[0,0].set_xticks([x1, x2])
    ax[0,0].set_xlabel("Time to hospital\n(from symptoms; days)")
    ax[0,0].set_ylabel("Density")
    ax[0,0].set_title("")
    ax[0,0].spines["top"].set_visible(False)
    ax[0,0].spines["right"].set_visible(False)
    
    ####################################
    # Bernoulli of mean time to critical
    ####################################
    
    height1 = np.ceil(df.mean_time_to_critical[0]) - df.mean_time_to_critical[0]
    height2 = df.mean_time_to_critical[0] - np.floor(df.mean_time_to_critical[0])
    
    x1 = np.floor(df.mean_time_to_critical.values[0])
    x2 = np.ceil(df.mean_time_to_critical.values[0])
    ax[1,0].bar([x1, x2], [height1, height2], color = "#0072B2")
    
    ax[1,0].set_ylabel("Density")
    ax[1,0].set_xticks([x1, x2])
    ax[1,0].set_ylim([0, 1.0])
    ax[1,0].set_xlabel("Time to critical\n(from hospitalised; days)")
    ax[1,0].set_title("")
    ax[1,0].spines["top"].set_visible(False)
    ax[1,0].spines["right"].set_visible(False)
    
    ################################
    # Gamma of mean time to symptoms
    ################################
    
    a, b = gamma_params(df.mean_time_to_symptoms.values, df.sd_time_to_symptoms.values)
    ax[0,1].plot(x, gamma.pdf(x, a = a, loc = 0, scale = b), linewidth= lw, color = "#0072B2")
    ax[0,1].axvline(df.mean_time_to_symptoms.values, color = "#D55E00", 
        linestyle = "dashed", alpha = 0.7)
    ax[0,1].set_xlabel("Time to symptoms\n(from presymptomatic; days)")
    ax[0,1].set_title("")
    ax[0,1].spines["top"].set_visible(False)
    ax[0,1].spines["right"].set_visible(False)
    ax[0,1].text(0.9, 0.7, 'mean: {}\nsd: {}'.format(df.mean_time_to_symptoms.values[0],
        df.sd_time_to_symptoms.values[0]), 
        ha = 'right', va = 'center', transform = ax[0,1].transAxes)
    
    ################################
    # Gamma of mean infectious period
    ################################
    
    a, b = gamma_params(df.mean_infectious_period, df.sd_infectious_period)
    ax[0,2].plot(x, gamma.pdf(x, a = a, loc = 0, scale = b), linewidth= lw, color = "#0072B2")
    ax[0,2].axvline(df.mean_infectious_period.values, color = "#D55E00", 
        linestyle = "dashed", alpha = 0.7)
    ax[0,2].set_xlabel("Infectious period (days)")
    ax[0,2].set_title("")
    ax[0,2].spines["top"].set_visible(False)
    ax[0,2].spines["right"].set_visible(False)
    ax[0,2].text(0.9, 0.7, 'mean: {}\nsd: {}'.format(df.mean_infectious_period.values[0],
        df.sd_infectious_period.values[0]), 
        ha = 'right', va = 'center', transform = ax[0,2].transAxes)
    
    ################################
    # Gamma of mean time to recover
    ################################
    
    a, b = gamma_params(df.mean_time_to_recover, df.sd_time_to_recover)
    ax[1,1].plot(x, gamma.pdf(x, a = a, loc = 0, scale = b), linewidth= lw, color = "#0072B2")
    ax[1,1].axvline(df.mean_time_to_recover.values, color = "#D55E00", 
        linestyle = "dashed", alpha = 0.7)
    ax[1,1].set_xlabel("Time to recover\n(from hospitalised or critical; days)")
    ax[1,1].set_title("")
    ax[1,1].spines["top"].set_visible(False)
    ax[1,1].spines["right"].set_visible(False)
    ax[1,1].text(0.9, 0.7, 'mean: {}\nsd: {}'.format(df.mean_time_to_recover.values[0],
        df.sd_time_to_recover.values[0]), 
        ha = 'right', va = 'center', transform = ax[1,1].transAxes)
    
    ########################################
    # Gamma of mean asymptomatic to recovery
    ########################################
    
    a, b = gamma_params(df.mean_asymptomatic_to_recovery, df.sd_asymptomatic_to_recovery)
    ax[2,0].plot(x, gamma.pdf(x, a = a, loc = 0, scale = b), linewidth= lw, color = "#0072B2")
    ax[2,0].axvline(df.mean_asymptomatic_to_recovery.values, color = "#D55E00", 
        linestyle = "dashed", alpha = 0.7)
    ax[2,0].set_xlabel("Time to recover\n(from asymptomatic; days)")
    ax[2,0].set_title("")
    ax[2,0].spines["top"].set_visible(False)
    ax[2,0].spines["right"].set_visible(False)
    ax[2,0].text(0.9, 0.7, 'mean: {}\nsd: {}'.format(df.mean_asymptomatic_to_recovery.values[0],
        df.sd_asymptomatic_to_recovery.values[0]), 
        ha = 'right', va = 'center', transform = ax[2,0].transAxes)
    
    ########################################
    # Gamma of mean hospitalised to recovery ... if don't go into CRITICAL
    ########################################... if don't go into critical care: FIXME - definitions
    
    a, b = gamma_params(df.mean_time_hospitalised_recovery, df.sd_time_hospitalised_recovery)
    ax[2,1].plot(x, gamma.pdf(x, a = a, loc = 0, scale = b), linewidth= lw, color = "#0072B2")
    ax[2,1].axvline(df.mean_asymptomatic_to_recovery.values, color = "#D55E00", 
        linestyle = "dashed", alpha = 0.7)
    ax[2,1].set_xlabel("Time to recover\n(from hospitalisation to discharge if not ICU\nor from ICU discharge to discharge if ICU; days)")
    ax[2,1].set_title("")
    ax[2,1].spines["top"].set_visible(False)
    ax[2,1].spines["right"].set_visible(False)
    ax[2,1].text(0.9, 0.7, 'mean: {}\nsd: {}'.format(df.mean_time_hospitalised_recovery.values[0],
        df.sd_time_hospitalised_recovery.values[0]), 
        ha = 'right', va = 'center', transform = ax[2,1].transAxes)
    
    #############################
    # Gamma of mean time to death
    #############################
    
    a, b = gamma_params(df.mean_time_to_death.values, df.sd_time_to_death.values)
    ax[1,2].plot(x, gamma.pdf(x, a = a, loc = 0, scale = b), linewidth= lw, c = "#0072B2")
    ax[1,2].axvline(df.mean_time_to_death.values, color = "#D55E00", 
        linestyle = "dashed", alpha = 0.7)
    ax[1,2].set_xlabel("Time to death\n(from critical; days)")
    ax[1,2].set_title("")
    ax[1,2].spines["top"].set_visible(False)
    ax[1,2].spines["right"].set_visible(False)
    ax[1,2].text(0.9, 0.7, 'mean: {}\nsd: {}'.format(df.mean_time_to_death.values[0],
        df.sd_time_to_death.values[0]), 
        ha = 'right', va = 'center', transform = ax[1,2].transAxes)
    
    ########################################
    # Gamma of mean time to survive if critical: FIXME - definitions
    ########################################
    
    a, b = gamma_params(df.mean_time_critical_survive, df.sd_time_critical_survive)
    ax[2,2].plot(x, gamma.pdf(x, a = a, loc = 0, scale = b), linewidth= lw, color = "#0072B2")
    ax[2,2].axvline(df.mean_time_critical_survive.values, color = "#D55E00", 
        linestyle = "dashed", alpha = 0.7)
    ax[2,2].set_xlabel("Time to survive\n(if ICU; days)")
    ax[2,2].set_title("")
    ax[2,2].spines["top"].set_visible(False)
    ax[2,2].spines["right"].set_visible(False)
    ax[2,2].text(0.9, 0.7, 'mean: {}\nsd: {}'.format(df.mean_time_critical_survive.values[0],
        df.sd_time_critical_survive.values[0]), 
        ha = 'right', va = 'center', transform = ax[2,2].transAxes)
    
    plt.subplots_adjust(hspace = 0.5)
    
    return(fig, ax)


def EpidemicCurves(df_timeseries, xlimits = None, lw = 3, timevar = "time"):
    """
    Plot population-level metrics of COVID19 outbreak through time
    
    By default, a figure with four subplots is returned, each with the following plotted:
    1. Cumulative infected with SARS-CoV-2, cumulative recovered
    2. Current number asymptomatic, pre-symptomatic, symtompatic, incident cases
    3. Current number of deaths, hospitalisations, ICU cases
    4. Current number in quarantine, number of daily tests used
    
    Arguments
    ---------
    df_timeseries : pandas.DataFrame
        DataFrame of timeseries output from COVID19-IBM (output which is printed to stdout)
    xlimits : list of ints
        Limits of the x-axis (time)
    lw : float
        Line with used in the plots
    
    Returns
    -------
    fig, ax : figure and axis handles to the generated figure using matplotlib.pyplot
    """
    
    df = df_timeseries # for brevity, keeping arg as-is - more descriptive
    
    df["daily_incidence"] = np.insert(0, 0, np.diff(df.total_infected.values))
    
    # List of dictionaries of what to plot in each panel of the plot
    data = [{
            "total_infected": {"label": "Total infected", "c": "red", "linestyle": "solid"},
            "n_recovered": {"label": "Total recovered", "c": "#009E73", "linestyle": "solid"}
        },
        {
            "n_asymptom":  {"label": "Asymptomatic", "c": "#E69F00", "linestyle": "solid"},
            "n_presymptom":  {"label": "Presymptomatic", "c": "#CC79A7", "linestyle": "solid"},
            "n_symptoms":  {"label": "Symptomatic", "c": "#D55E00", "linestyle": "solid"},
            "daily_incidence": {"label": "Incident cases", "c": "red", "linestyle": "solid"}
        },
        {
            "n_death": {"label": "Deaths", "c": "black", "linestyle": "dashed"},
            "n_hospital": {"label": "Number hospitalised", "c": "#56B4E9", "linestyle": "solid"},
            "n_critical": {"label": "ICU cases", "c": "#0072B2", "linestyle": "solid"}
        },
        {
            "n_quarantine": {"label": "Number in quarantine", "c":  "grey", "linestyle": "solid"},
            "n_tests": {"label": "Tests used", "c": "black", "linestyle": "solid"}
        }]
    
    fig, ax = plt.subplots(nrows = len(data))
    
    for i, panel in enumerate(data):
        
        maximums = []
        for var, props in panel.items():
            
            ax[i].plot(df[timevar], df[var], 
                c = props["c"], 
                linestyle = props["linestyle"],
                linewidth = lw, 
                label = props["label"]
                )
            maximums.append(df[var].max())
            
        ax[i].fill_between(df[timevar], 0, np.max(maximums), 
            where = (df["lockdown"] == 1), alpha = 0.5)
        
        ax[i].set_ylabel("", size = 18)
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].legend(loc = "center left")
        ax[i].set_xlim(xlimits)
        
        if i == 3:
            ax[i].set_xlabel("Day since infection seeded", size = 18)
        else:
            ax[i].set_xticks([])
            ax[i].set_xlabel("")
    
    return(fig, ax)




def BarByGroup(df, groupvar, binvar, bins = None, groups = None, group_labels = None, 
    group_colours = None, xlimits = None, density = False, title = "", xvar = None,
    xlabel = None, ylabel = None, legend_title = "", xticklabels = None):
    """
    Bar plot with multiple groups, with bars plotted side-by-side
    
    Arguments
    ---------
    df : pandas.DataFrame
        DataFrame of model output
    groupvar : str
        Column name of `df` which stores the grouping variable
    binvar : str
        Column name of `df` over which values will be binned 
    bin : int or list
        Either number of bins or list of bins to use
    groups : list
        Subset of categories in `group` column to plot (defaults to unique values in `groupvar` col)
    group_labels : list
        Labels to use for `groups` categories (defaults to `groups` list)
    
    Returns
    -------
    fig, ax : figure and axis handles to the generated figure using matplotlib.pyplot
    """
    
    if not groups:
        groups = df[groupvar].unique()
    
    if not group_labels:
        group_labels = groups
    
    n_groups = len(groups)
    
    if not isinstance(bins, list):
        if binvar == "age_group":
            bin_list = np.arange(0, bins + 1) - 0.1
        else:
            bin_list = np.arange(bins)
    else:
        bin_list = bins
    
    if group_colours is None:
        group_colours = get_discrete_viridis_colours(n_groups)
    
    width = np.diff(bin_list)[0]/(n_groups + 1)
    
    fig, ax = plt.subplots()
    ax.grid(which = 'major', axis = 'y', alpha = 0.7, zorder = 0)
    
    for i, g in enumerate(groups):
        heights, b = np.histogram(df.loc[df[groupvar] == g][binvar], bin_list, density = density)
        
        ax.bar(bin_list[:-1] + width*i, heights, width = width, facecolor = group_colours[i],
            label = group_labels[i], edgecolor = "#0d1a26", linewidth = 0.5, zorder = 3)
    
    ax.set_xlim([-0.5, np.max(bin_list)])

    legend = ax.legend(loc = 'best', borderaxespad = 0, frameon = False, 
        prop = {'size': 16}, fontsize = "x-large")
    legend.set_title(legend_title, prop = {'size':18})
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel(xlabel, size = 16)
    ax.set_ylabel(ylabel, size = 16)
    ax.set_title(title, size = 20)
    
    if xlimits is not None:
        ax.set_xlim(xlimits)
    
    if xticklabels is not None:
        ax.set_xticks(bin_list)
        ax.set_xticklabels(xticklabels, size = 12)
    
    return(fig, ax)



def BarByGroupByPanel(df, groupvar, binvar, panelvar, panels = None, panel_labels = None, 
    groups = None, group_labels = None, NBINS = None, 
    group_colours = None, xlimits = None, density = False, title = "", 
    xlabel = "", ylabel = "", legend_title = "", xticklabels = None):
    """
    
    """
    
    if not panels: 
        panels = df[panelvar].unique()
    n_panels = len(panels)
    
    if not panel_labels:
        panel_labels = panels
    
    if not groups: 
        groups = df[groupvar].unique()
    n_groups = len(groups)
    
    print(groups)
    
    if not group_colours:
        group_colours = get_discrete_viridis_colours(n_groups)
    
    if not group_labels:
        group_labels = groups
    
    bins = np.arange(NBINS)
    width = np.diff(bins)[0]/(n_groups + 1)
    
    fig, ax = plt.subplots(ncols = n_panels)
    
    for j, p in enumerate(panels):
        ax[j].grid(which = 'major', axis = 'y', alpha = 0.7, zorder = 0)
        
        for i, g in enumerate(groups):
            
            df_sub = df.loc[(df[groupvar] == g)&(df[panelvar] == p)]
            print(df_sub.head())
            heights, b = np.histogram(df_sub[binvar], bins)
            
            
            ax[j].bar(bins[:-1] + width*i, heights, width = width, 
                facecolor = group_colours[i], label = group_labels[i], 
                edgecolor = "#0d1a26", linewidth = 0.5, zorder = 3)
        
        ax[j].set_xlim([0, np.max(bins)])
        
        ax[j].spines["top"].set_visible(False)
        ax[j].spines["right"].set_visible(False)
        
        ax[j].set_xlabel(xlabel, size = 16)
        ax[j].set_ylabel("", size = 16)
        ax[j].set_title(panel_labels[j], size = 20)
        
        if xlimits is not None:
            ax[j].set_xlim(xlimits)
            
        if xticklabels is not None:
            ax[j].set_xticklabels(xticklabels, size = 12)
    
    
    legend = ax[j].legend(loc = 'right', borderaxespad = 0, frameon = False, 
        prop = {'size': 16}, fontsize = "x-large")
    legend.set_title(legend_title, prop = {'size':18})
    
    return(fig, ax)


def add_heatmap_to_axes(ax, df, 
        group1var, group2var, bin_list, normalise = False
    ):
    """
    Plot heatmap of transmission events across two grouping variables
    
    
    """
    
    array, xbins, ybins = np.histogram2d(df[group1var].values, df[group2var].values, bin_list)
    
    if normalise:
        array = array/array.sum(axis = 0)
    
    im = ax.imshow(array, origin = 'lower', aspect = "equal")
    
    return(ax, im)


def adjust_ticks(ax, xtick_fontsize = 12, ytick_fontsize = 12, 
    xticklabels = None, yticklabels = None):
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(xtick_fontsize)
    
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(ytick_fontsize)
    
    if xticklabels is not None:
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation = 60)
    
    if yticklabels is not None:
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels)
    
    return(ax)


def transmission_heatmap_by_age(df, 
        group1var, group2var,
        bins = None,
        group_labels = None,
        xlabel = "",
        ylabel = "", title = "",
        legend_title = "", legend_loc = "right",
        xticklabels = None, yticklabels = None,
        normalise = False
    ):
    """
    Plot heatmap of transmission events across two grouping variables
    
    
    """
    if not isinstance(bins, list):
        bin_list = np.arange(bins)
    
    fig, ax = plt.subplots()
    
    ax, im = add_heatmap_to_axes(ax, df, group1var, group2var, bin_list, normalise)
    
    ax = adjust_ticks(ax, xtick_fontsize = 16, ytick_fontsize = 16, xticklabels = xticklabels, 
        yticklabels = yticklabels)
    
    ax.set_xlabel(xlabel, size = 20)
    ax.set_ylabel(ylabel, size = 20)
    ax.set_title(title)
    
    cbar = fig.colorbar(im, fraction = 0.046, pad = 0.04)
    cbar.set_label(legend_title, size = 18)
    cbar.ax.tick_params(labelsize = 14)
    
    return(fig, ax)



def transmission_heatmap_by_age_by_panels(df, 
        group1var, group2var, panelvar, 
        NBINS = None,
        groups = None,
        group_labels = None,
        panels = None,
        panel_labels = None,
        xlabel = "",
        ylabel = "",
        legend_title = "", legend_loc = "right",
        xticklabels = None, yticklabels = None,
        normalise = False
    ):
    """
    Plot subplots of heatmaps of transmissions from one age group to another across
    
    Arguments
    ---------
    group1var : str
        Column name of first grouping variable (x-axis in heatmap)
    group2var : str
        Column name of second grouping variable (x-axis in heatmap)
    panelvar
        Column name of variable for making panels
    NBINS
        Number of bins
    group_labels
    normalise
    
    """
    
    bin_list = np.arange(NBINS)
    
    if not panels: 
        panels = np.unique(df[panelvar])
    n_panels = len(panels)
    
    if not panel_labels:
        panel_labels = panels
    
    fig, ax = plt.subplots(ncols = n_panels)
    
    ax[0].set_ylabel(ylabel, size = 16)
    for i, panel in enumerate(panels):
        
        df_sub = df.loc[df[panelvar] == panel]
        
        ax[i], im = add_heatmap_to_axes(ax[i], df_sub, group1var, group2var, bin_list, normalise)
        
        ax[i] = adjust_ticks(ax[i], xtick_fontsize = 14, ytick_fontsize = 14, 
            xticklabels = xticklabels, yticklabels = yticklabels)
        
        if i > 0:
            ax[i].set_yticks([])
        
        ax[i].set_xlabel(xlabel, size = 16)
        ax[i].set_title(panel_labels[i], size = 20)
    
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes('right', size = "7%", pad = 0.2,)
    
    cbar = fig.colorbar(im, fraction = 0.046, pad = 0.04, cax = cax)
    cbar.set_label(legend_title, size = 18)
    
    return(fig, ax)



def PlotInteractionsByAge(df_interact):
    """
    
    """
    a = 1.0
    
    # Aggregate by age group and ID
    df_agg = df_interact.groupby(["age_group", "ID"]).size().reset_index(name = "counts")
    
    # Define age groups, size of bins
    age_groups = np.unique(df_agg.age_group)
    n_age_groups = len(age_groups)
    bins = np.arange(50)
    age_groups_labels = ["{} - {} years".format(10*i, 10*age_groups[i+1]-1) for i in np.arange(0, n_age_groups-1)]
    age_groups_labels = age_groups_labels + ["80+ years"]
    
    # Split by age group
    hists = [df_agg.loc[df_agg.age_group == age].counts for age in age_groups]

    # Define the colourmap
    colours = get_discrete_viridis_colours(n_age_groups)
    
    
    fig, ax = plt.subplots()
    ax.grid(which = 'major', axis = 'y', alpha = 0.4, zorder = 0)
    ax.hist(hists, bins, stacked = True, label = age_groups_labels, width = 0.8, alpha = a, color = colours, edgecolor = "#0d1a26", linewidth = 0.5, zorder = 3)
    ax.set_xlim([0, np.max(bins)])

    legend = ax.legend(loc = 'right', borderaxespad = 0, frameon = False, prop = {'size': 16}, fontsize = "x-large")
    legend.set_title("Age group", prop = {'size':18})

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel("Number of daily connections", size = 16)
    ax.set_ylabel("Count", size = 16)
    ax.set_title("Number of daily interactions by age group", size = 20)
    
    return(fig, ax)





def PlotREffective(df_indiv, df_trans, start = 1, stop = 100, window = 5):
    
    bins = overlapping_bins(start = start, stop = stop, window = window, by = 1)
    
    Reff = []
    Reff_std = []
    Reff_q025 = []
    Reff_q975 = []
    
    Reff_pre = []
    Reff_asym = []
    Reff_sym = []
    
    for b in bins:
        
        # Of those infectious at time b[0]
        cond1 = (df_indiv["time_presypmtomatic"] < b[0]) & (df_indiv["time_presypmtomatic"] != -1)
        cond2 = (df_indiv["time_symptomatic"] < b[0]) & (df_indiv["time_symptomatic"] != -1)
        cond3 = (df_indiv["time_asymptomatic"] < b[0]) & (df_indiv["time_asymptomatic"] != -1)
        
        cond4 = (df_indiv["time_hospitalised"] > b[0]) | (df_indiv["time_hospitalised"] == -1)
        cond5 = (df_indiv["time_critical"] > b[0]) | (df_indiv["time_critical"] == -1)
        cond6 = (df_indiv["time_death"] > b[0]) | (df_indiv["time_death"] == -1 )
        cond7 = (df_indiv["time_recovered"] > b[0]) | (df_indiv["time_recovered"] == -1)
        
        df_infected = df_indiv[(cond1 | cond2 | cond3) & (cond4 & cond5 & cond6 & cond7)]
        
        cond8 = cond1 & (df_infected["time_symptomatic"] > b[0])
        df_presymptomatic = df_indiv[cond8]
        
        cond9 = cond3 & cond7
        df_asymptomatic = df_indiv[cond9]
        
        cond10 = cond2 & cond4 & cond5 & cond6 & cond7
        df_symptomatic = df_indiv[cond10]
        
        # Find infection events where these people were the infector (ID_2)
        cat_type = CategoricalDtype(categories = df_infected.ID.values, ordered = False)
        R = df_trans.ID_2.astype(cat_type).value_counts().reset_index(name = "R")
        
        R.columns = ["ID", "R"]
        Reff.append(R.R.mean())
        Reff_std.append(R.R.std()/np.sqrt(R.shape[0]))
        Reff_q025.append(np.quantile(R.R.values, 0.025))
        Reff_q975.append(np.quantile(R.R.values, 0.975))
        
        cat_type = CategoricalDtype(categories = df_presymptomatic.ID.values, ordered = False)
        R_pre = df_trans.ID_2.astype(cat_type).value_counts().reset_index(name = "R_pre")
        R_pre.columns = ["ID", "R"]
        Reff_pre.append(R_pre.R.mean())
        
        if(cond9.sum() > 0):
            #df_asymptomatic = df_infected[cond9]
            cat_type = CategoricalDtype(categories = df_asymptomatic.ID.values, ordered = False)
            R_asym = df_trans.ID_2.astype(cat_type).value_counts().reset_index(name = "R_asym")
            R_asym.columns = ["ID", "R"]
            Reff_asym.append(R_asym.R.mean())
        else:
            Reff_asym.append(0)
        
        cat_type = CategoricalDtype(categories = df_symptomatic.ID.values, ordered = False)
        R_sym = df_trans.ID_2.astype(cat_type).value_counts().reset_index(name = "R_sym")
        R_sym.columns = ["ID", "R"]
        Reff_sym.append(R_sym.R.mean())
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(Reff)), Reff, linewidth = 3, 
        c = "#0072B2", label  = "All infectious")
    ax.plot(np.arange(len(Reff)), Reff_asym, linewidth = 3, 
        c = "#E69F00", label = "Asymptomatic")
    ax.plot(np.arange(len(Reff)), Reff_pre, linewidth = 3, 
        c = "#CC79A7", label = "Presymptomatic")
    ax.plot(np.arange(len(Reff)), Reff_sym, linewidth = 3, 
        c = "#D55E00", label = "Symptomatic")
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    
    ax.axhline(1, c = "red", linestyle = "dashed", alpha = 0.6)
    ax.legend(prop = {'size': 16})
    ax.set_xlabel("Time since infection seeded", size = 16)
    ax.set_ylabel("Infections over next {} days".format(window) + 
        "\nper infectious case ($R_{eff}$)",
        size = 16)
    
    return(fig, ax)


def PlotHistIFRByAge(df, 
        numerator_var, denominator_var,
        NBINS = None,
        group_labels = None,
        xlabel = "",
        xticklabels = None,
        density = False
    ):
    """
    Plot IFR by age.  
    """
    
    a = 1.0
    bins = np.arange(0, NBINS + 1) - 0.1
    n_age = len(np.unique(df["age_group"]))
    
    fig, ax = plt.subplots()
    
    height_n, bins_n = np.histogram(df[df[numerator_var] > 0]["age_group"], bins, density = False)
    height_d, bins_d = np.histogram(df[df[denominator_var] > 0]["age_group"], bins, density = False)
    
    heights = np.divide(height_n, height_d)
    
    ax.bar(range(n_age), heights, align = "center",
             alpha = a, color = "#0072B2", edgecolor = "#0d1a26", linewidth = 0.5,
             zorder = 3)
    
    for bi in range(n_age):
        ax.text(bi, heights[bi], str(np.round(heights[bi], 2)), 
            ha = "center", va = "bottom", color = "grey")
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    ax.set_xlim([-0.5, np.max(bins)+0.5])
    ax.set_ylim([0, np.max(heights)*1.1])
    
    if xticklabels is not None:
        ax.set_xticks(bins)
        ax.set_xticklabels(xticklabels, size = 12)
    
    ax.set_xlabel(xlabel, size = 18)
    ax.set_ylabel("Infection fatality ratio (IFR) by age", size = 18)
    
    overall_ifr = height_n.sum()/height_d.sum()
    ax.text(0.05, 0.8, "Overall IFR: {}".format(np.round(overall_ifr, 4)), size = 18, 
        ha = 'left', va = 'center', transform = ax.transAxes, color = "black")
    
    return(fig, ax)


def PlotHistByAge(df, 
        groupvars, 
        NBINS = None,
        group_labels = None,
        xlabel = "",
        xticklabels = None,
        density = False,
        ylim = 0.5
    ):
    """
    
    """
    
    a = 1.0
    bins = np.arange(0, NBINS + 1) - 0.1
    
    # Define number of groups
    n_groups = len(groupvars)
    
    if group_labels is None:
        group_labels = groupvars
    
    fig, ax = plt.subplots(nrows = n_groups)
    
    for axi, var in enumerate(groupvars):
        height, bins, objs = ax[axi].hist(df[df[var] > 0]["age_group"], bins, width = 0.8, 
            alpha = a, color = "#0072B2", edgecolor = "#0d1a26", linewidth = 0.5, 
            zorder = 3, density = density)
        
        for bi in range(len(bins) - 1):
            ax[axi].text(bins[bi] + 0.425, height[bi], str(np.round(height[bi], 2)), 
                ha = "center", va = "bottom", color = "grey")
        
        ax[axi].set_xlim([0, np.max(bins)])
        ax[axi].spines["top"].set_visible(False)
        ax[axi].spines["right"].set_visible(False)
        ax[axi].set_ylim([0, ylim])
        ax[axi].text(0.02, 0.8, group_labels[axi], size = 18, 
            ha = 'left', va = 'center', transform = ax[axi].transAxes, color = "black")
        
        if axi == (n_groups - 1):
            if xticklabels is not None:
                ax[axi].set_xticks(bins + 0.425)
                ax[axi].set_xticklabels(xticklabels, size = 12)
            else:
                ax[axi].set_xticks(bins + 0.425)
                ax[axi].set_xticklabels(bins, size = 12)
        else:
            ax[axi].set_xticks(bins + 0.425)
            ax[axi].set_xticks([])
    
    ax[n_groups-1].set_xlabel(xlabel, size = 18)
    
    plt.subplots_adjust(hspace = 0.5)
    
    return(fig, ax)



def PlotStackedHistByGroup(df, 
        groupvar, countvar,
        NBINS = None,
        group_labels = None,
        xlabel = "",
        ylabel = "",
        title = "",
        legend_title = "", legend_loc = "right",
        xticklabels = None
    ):
    """
    
    """
    
    a = 1.0
    
    # Define groups and size of groups
    groups = np.unique(df[groupvar])
    n_groups = len(groups)
    
    if group_labels is None:
        group_labels = groups
    
    # Split by group
    hists = [df.loc[df[groupvar] == state][countvar] for state in groups]
    
    # Define the colourmap
    colours = get_discrete_viridis_colours(n_groups)
    
    bins = np.arange(NBINS)
    
    fig, ax = plt.subplots()
    
    ax.grid(which = 'major', axis = 'y', alpha = 0.4, zorder = 0)
    ax.hist(hists, bins, stacked = True, label = group_labels, width = 0.8, 
        alpha = a, color = colours, edgecolor = "#0d1a26", linewidth = 0.5, 
        zorder = 3)
    
    ax.set_xlim([0, np.max(bins)])
    
    legend = ax.legend(loc = legend_loc, borderaxespad = 0, 
        frameon = False, prop = {'size': 16}, fontsize = "x-large")
    legend.set_title(legend_title, prop = {'size':18})

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel(xlabel, size = 16)
    ax.set_ylabel(ylabel, size = 16)
    ax.set_title(title, size = 20)
    
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    else:
        ax.set_xticks(bins)
        ax.set_xticklabels(bins)
    
    return(fig, ax)


def PlotStackedHistByGroupByPanel(df, 
        groupvar, countvar, panelvar,
        NBINS = None,
        group_labels = None,
        panel_labels = None,
        xlabel = "",
        ylabel = "",
        title = "",
        legend_title = "", legend_loc = "right",
        xticklabels = None,
        ylims = None
    ):
    """
    
    """
    
    a = 1.0
    
    # Define panels and size of panels
    panels = np.unique(df[panelvar])
    n_panels = len(panels)
    
    if panel_labels is None:
        panel_labels = panels
    
    # Define groups and size of groups
    groups = np.unique(df[groupvar])
    n_groups = len(groups)
    
    if group_labels is None:
        group_labels = groups
    
    # Define the colourmap
    colours = get_discrete_viridis_colours(n_groups)
    
    bins = np.arange(NBINS)
    
    fig, ax = plt.subplots(ncols = n_panels)
    
    for i, panel in enumerate(panels):
        
        # Split by group
        hists = [df.loc[(df[groupvar] == state) & (df[panelvar] == panel)][countvar] for state in groups]
        
        ax[i].grid(which = 'major', axis = 'y', alpha = 0.4, zorder = 0)
        
        ax[i].hist(hists, bins, stacked = True, label = group_labels, width = 0.8, 
            alpha = a, color = colours, edgecolor = "#0d1a26", linewidth = 0.5, 
            zorder = 3)
        
        ax[i].set_xlim([0, np.max(bins)])

        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)

        ax[i].set_xlabel(xlabel, size = 16)
        ax[i].set_title(panel_labels[i], size = 20)
        
        if i == 0:
            ax[i].set_ylabel(ylabel, size = 16)
        
        xticks = np.arange(0, NBINS + 5, 5)
        ax[i].set_xticks(xticks)
        ax[i].set_xticklabels(xticks)
        if ylims:
            ax[i].set_ylim(ylims)
    
    legend = ax[i].legend(loc = legend_loc, borderaxespad = 0, 
    frameon = False, prop = {'size': 10}, fontsize = "x-large")
    legend.set_title(legend_title, prop = {'size':18})

    return(fig, ax)