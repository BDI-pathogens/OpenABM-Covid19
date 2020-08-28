#!/usr/bin/env python3
"""
Plotting functions for files output from the OpenABM-Covid19 model

Created: 30 March 2020
Author: p-robot
"""

from os.path import join

import numpy as np, pandas as pd
from pandas.api.types import CategoricalDtype
from scipy.stats import gamma

from matplotlib import pyplot as plt
from matplotlib import cm

# Colours for plotting the household, work, and random networks
network_colours = ['#009E73', '#0072B2', '#D55E00']

# Nicely printed labels of event types from the EVENT_TYPES enum 
# as defined in OpenABM-Covid19/src/constant.h
EVENT_TYPE_STRING = {
    0: "Susceptible",
    1: "Presymptomatic (severe)",
    2: "Presymptomatic (mild)",
    3: "Asymptomatic",
    4: "Symptomatic (severe)",
    5: "Symptomatic (mild)",
    6: "Hospitalised",
    7: "ICU", 
    8: "Recovering in hospital",
    9: "Recovered",
    10: "Dead",
    11: "Quarantined", 
    12: "Quarantined release",
    13: "Test taken", 
    14: "Test result",
    15: "Case", 
    16: "Trace token release",
    17: "Transition to hospital",
    18: "Transition to critical",
    19: "N event types"
}

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

population_cols = [
    "population_0_9", "population_10_19",
    "population_20_29", "population_30_39",
    "population_40_49", "population_50_59", 
    "population_60_69", "population_70_79", 
    "population_80"]


def get_df_from_params(params, parameter_names):
    """
    Return a pandas dataframe of parameter-value pairs from the passed Parameters object
    Mainly used as a helper function for displaying parameter values.  
    
    Arguments
    ---------
    params : parameters object of class COVID19.model.Parameters
        Parameter object
    parameter_names : list of str
        List of parameter names of interest (column names of parameter input file)
    
    Returns
    -------
    pandas.DataFrame of the parameters of interest
    """
    parameter_values = [params.get_param(p) for p in parameter_names]
    df = pd.DataFrame([parameter_values], columns = parameter_names)
    return(df)


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
    
    Arguments
    ---------
    n : int
        Number of colours to generate on the viridis colour map
    
    Returns
    -------
    List of length n where each elements is an RGBA list defining a colour
    """
    colourmap = cm.get_cmap('viridis', n)
    colours = [colourmap.colors[n - i - 1] for i in range(n)]
    return(colours)


def plot_parameter_assumptions(df_parameters, xlimits = [0, 30], lw = 3):
    """
    Plot distributions of mean transition times between compartments in the parameters of the 
    OpenABM-Covid19 model
    
    Arguments
    ---------
    df_parameters : pandas.DataFrame
        DataFrame of parameter values as input first input argument to the OpenABM-Covid19 model
        This plotting scripts expects the following columns within this dataframe: 
            mean_time_to_hospital
            mean_time_to_critical, sd_time_to_critical
            mean_time_to_symptoms, sd_time_to_symptoms
            mean_infectious_period, sd_infectious_period
            mean_time_to_recover, sd_time_to_recover
            mean_asymptomatic_to_recovery, sd_asymptomatic_to_recovery
            mean_time_hospitalised_recovery, sd_time_hospitalised_recovery
            mean_time_to_death, sd_time_to_death
            mean_time_critical_survive, sd_time_critical_survive
    
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
    # Gamma of mean time to critical
    ####################################
    
    a, b = gamma_params(df.mean_time_to_critical.values, df.sd_time_to_critical.values)
    ax[1,0].plot(x, gamma.pdf(x, a = a, loc = 0, scale = b), linewidth= lw, color = "#0072B2")
    ax[1,0].axvline(df.mean_time_to_critical.values, color = "#D55E00", 
        linestyle = "dashed", alpha = 0.7)
    ax[1,0].set_xlabel("Time to critical\n(from hospitalised; days)")
    ax[1,0].set_title("")
    ax[1,0].spines["top"].set_visible(False)
    ax[1,0].spines["right"].set_visible(False)
    ax[1,0].text(0.9, 0.7, 'mean: {}\nsd: {}'.format(df.mean_time_to_critical.values[0],
        df.sd_time_to_critical.values[0]), 
        ha = 'right', va = 'center', transform = ax[1,0].transAxes)
    
    
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
    # Gamma of mean hospitalised to recovery
    ########################################
    
    a, b = gamma_params(df.mean_time_hospitalised_recovery, df.sd_time_hospitalised_recovery)
    ax[2,1].plot(x, gamma.pdf(x, a = a, loc = 0, scale = b), linewidth= lw, color = "#0072B2")
    ax[2,1].axvline(df.mean_time_hospitalised_recovery.values, color = "#D55E00", 
        linestyle = "dashed", alpha = 0.7)
    ax[2,1].set_xlabel("Time to recover\n(from hospitalisation to hospital discharge if not ICU\nor from ICU discharge to hospital discharge if ICU; days)")
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


def plot_timeseries_curves(df_timeseries, xlimits = None, lw = 3, timevar = "time"):
    """
    Plot population-level metrics of COVID19 outbreak through time
    
    By default, a figure with four subplots is returned, each with the following plotted:
    1. Cumulative infected with SARS-CoV-2, cumulative recovered, number quarantined
    2. Current number asymptomatic, pre-symptomatic, symtompatic, incident cases
    3. Current number of deaths, hospitalisations, ICU cases
    4. Current number of daily tests used
    
    Arguments
    ---------
    df_timeseries : pandas.DataFrame
        DataFrame of timeseries output from COVID19-IBM (output which is printed to stdout)
    xlimits : list of ints
        Limits of the x-axis (time)
    lw : float
        Line with used in the plots
    timevar : str
        Column name within df_timeseries that defines the x-axis
    
    Returns
    -------
    fig, ax : figure and axis handles to the generated figure using matplotlib.pyplot
    """
    
    df = df_timeseries # for brevity, keeping input argument as-is since it's more descriptive
    
    df["daily_incidence"] = np.insert(0, 0, np.diff(df.total_infected.values))
    
    # List of dictionaries of what to plot in each panel of the plot
    data = [{
            "total_infected": {"label": "Total infected", "c": "red", "linestyle": "solid"},
            "n_recovered": {"label": "Total recovered", "c": "#009E73", "linestyle": "solid"},
            "n_quarantine": {"label": "Number in quarantine", "c":  "grey", "linestyle": "solid"}
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
        
        for tick in ax[i].xaxis.get_major_ticks():
            tick.label.set_fontsize(14)
    
        for tick in ax[i].yaxis.get_major_ticks():
            tick.label.set_fontsize(12)
        
        if i == 3:
            ax[i].set_xlabel("Day since infection seeded", size = 18)
        else:
            ax[i].set_xticks([])
            ax[i].set_xlabel("")
    
    return(fig, ax)


def plot_hist_by_group(df, groupvar, binvar, bins = None, groups = None, 
    group_labels = None, group_colours = None, xlimits = None, density = False, 
    title = "", xlabel = "", ylabel = "", legend_title = "", xticklabels = None):
    """
    Histogram with multiple groups, with histogram bars plotted side-by-side for each group
    
    Arguments
    ---------
    df : pandas.DataFrame
        DataFrame of model output
    groupvar : str
        Column name of `df` which stores the grouping variable
    binvar : str
        Column name of `df` over which values will be binned 
    bin : int or list
        Either a number of bins or list of bins to use
    groups : list
        Subset of categories in `group` column to plot (defaults to unique values in `groupvar` col)
    group_labels : list
        Labels to use for `groups` categories (defaults to `groups` list)
    group_colours : list
        Colours to use for the different `groups` categories (defaults to using the viridis 
        colour map with n_groups)
    xlimits : float
        Limit of the x-axis
    density : boolean
        Should histogram be normalised (passed to density arg in np.histogram)
    title, xlabel, ylabel, legend_title : str
        Title, X-axis label, Y-axis label, and legend title respectively
     xticklabels : list of str
        Labels to use for x-ticks
    
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
    #ax.grid(which = 'major', axis = 'y', alpha = 0.7, zorder = 0)
    
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

    ax.set_xlabel(xlabel, size = 18)
    ax.set_ylabel(ylabel, size = 18)
    ax.set_title(title, size = 20)
    
    if xlimits is not None:
        ax.set_xlim(xlimits)
    
    if xticklabels is not None:
        ax.set_xticks(bin_list + n_groups/2*width - width/2.)
        ax.set_xticklabels(xticklabels, size = 14)
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    
    return(fig, ax)


def add_heatmap_to_axes(ax, x, y, bin_list):
    """
    Plot heatmap of 2D histogram.
    
    Used for 2D histograms of transmission events across two grouping variables (e.g. age)
    
    Arguments
    ---------
    ax : object of matplotlib class `Axes`
        Axis object of where to add a heatmap
    x : np.array
        Array of the x values with which to create a histogram
    y : np.array
        Array of the y values with which to create a histogram
    bin_list : list
        List of bins to use in the histogram
    
    Returns
    -------
    (ax, im)
    ax : object of matplotlib class `Axes`
        updated Axes object
    im : matplotlib.image.AxesImage
        AxesImage object returned from matplotlib.pyplot.imshow
    """
    
    array, xbins, ybins = np.histogram2d(x, y, bin_list)
    
    im = ax.imshow(array, origin = "lower", aspect = "equal", vmin = 0)
    
    return(ax, im)


def adjust_ticks(ax, xtick_fontsize = 12, ytick_fontsize = 12, 
    xticklabels = None, yticklabels = None):
    """
    Adjust tick font size and ticklabels in a matplotlib.Axes object
    
    Arguments
    ---------
    ax : object of matplotlib class `Axes`
        Axis object of where to adjust tick fonts/labels
    xtick_fontsize, ytick_fontsize : int
        Font size of x-ticks and y-ticks
    xticklabels, yticklabels : list of str
        List of x and y axis tick labels to change
    
    Returns
    -------
    ax : object of matplotlib class `Axes`
        Returns the modified axis object
    """
    
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


def plot_transmission_heatmap_by_age(df, group1var, group2var, bins = None, 
    group_labels = None, xlabel = "", ylabel = "", title = "", legend_title = "", 
    legend_loc = "right", xticklabels = None, yticklabels = None, normalise = False, 
    vmin = 0, vmax = None):
    """
    Plot 2D histogram (as a heatmap) of transmission events by two grouping variables
    (for instance, age group)
    
    
    Returns
    -------
    fig, ax : figure and axis handles to the generated figure using matplotlib.pyplot
    
    """
    if not isinstance(bins, list):
        bin_list = np.arange(bins)
    
    fig, ax = plt.subplots()
    
    ax, im = add_heatmap_to_axes(ax, df[group1var].values, df[group2var].values, bin_list)
    
    ax = adjust_ticks(ax, xtick_fontsize = 16, ytick_fontsize = 16, 
        xticklabels = xticklabels, yticklabels = yticklabels)
    
    ax.set_xlabel(xlabel, size = 20)
    ax.set_ylabel(ylabel, size = 20)
    ax.set_title(title)
    
    cbar = fig.colorbar(im, fraction = 0.046, pad = 0.04)
    cbar.set_label(legend_title, size = 18)
    cbar.ax.tick_params(labelsize = 14)
    
    return(fig, ax)


def transmission_heatmap_by_age_by_panels(df, 
        group1var, group2var, panelvar, bins = None, 
        groups = None, group_labels = None,
        panels = None, panel_labels = None,
        xlabel = "", ylabel = "",
        legend_title = "", legend_loc = "right",
        xticklabels = None, yticklabels = None,
        normalise = False, title_fontsize = 20,
        spines = False
    ):
    """
    Plot subplots of heatmaps of transmissions from one age group to another across another 
    categorical variable (panelvar)
    
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
    
    if not isinstance(bins, list):
        bin_list = np.arange(bins)
    
    if not panels: 
        panels = np.unique(df[panelvar])
    
    n_panels = len(panels)
    
    if not panel_labels:
        panel_labels = panels
    
    fig, ax = plt.subplots(ncols = n_panels)
    
    ax[0].set_ylabel(ylabel, size = 16)
    
    transmission_arrays = []
    for i, panel in enumerate(panels):
        
        df_sub = df.loc[df[panelvar] == panel]
        
        array, xbins, ybins = np.histogram2d(
            x = df_sub[group1var].values, 
            y = df_sub[group2var].values, 
            bins = bin_list)
        transmission_arrays.append(array)
    
    vmin_panels = 0
    vmax_panels = np.max(np.array(transmission_arrays))
    
    ims = []

    for i, panel in enumerate(panels):
        im = ax[i].imshow(np.ma.masked_where(transmission_arrays[i] == 0, transmission_arrays[i]), 
            origin = "lower", aspect = "equal", 
            vmin = vmin_panels, vmax = vmax_panels)
        
        ims.append(im)
        ax[i] = adjust_ticks(ax[i], xtick_fontsize = 14, ytick_fontsize = 14, 
            xticklabels = xticklabels, yticklabels = yticklabels)
        
        if i > 0:
            ax[i].set_yticks([])
        
        ax[i].set_xlabel(xlabel, size = 16)
        ax[i].set_title(panel_labels[i], size = title_fontsize)
        
        if not spines:
            ax[i].spines["top"].set_visible(False)
            ax[i].spines["right"].set_visible(False)
            ax[i].spines["bottom"].set_visible(False)
            ax[i].spines["left"].set_visible(False)
    
    fig.subplots_adjust(right = 0.85)
    axes_cbar = fig.add_axes([0.9, 0.3, 0.02, 0.4])
    cbar = fig.colorbar(ims[n_panels - 1], cax = axes_cbar)
    
    cbar.set_label(legend_title, size = 18)
    
    return(fig, ax)



def plot_interactions_by_age(df_interact, groupvar, group_labels, 
    xlabel = "", ylabel = "", legend_title  = "", title = "", nbins = 40):
    """
    """
    
    # Aggregate by age group and ID
    df_agg = df_interact.groupby([groupvar, "ID_1"]).size().reset_index(name = "counts")
    
    # Define age groups, size of bins
    groups = np.unique(df_agg[groupvar])
    n_groups = len(groups)
    bins = np.arange(nbins)
    
    # Split by age group
    hists = [df_agg.loc[df_agg[groupvar] == g].counts for g in groups]
    
    # Define the colourmap
    colours = get_discrete_viridis_colours(n_groups)
    
    fig, ax = plt.subplots()
    #ax.grid(which = 'major', axis = 'y', alpha = 0.4, zorder = 0)
    ax.hist(hists, bins, stacked = True, label = group_labels, 
        width = 0.8, color = colours, edgecolor = "#0d1a26", linewidth = 0.5, zorder = 3)
    
    ax.set_xlim([0, np.max(bins)])
    
    legend = ax.legend(loc = 'right', borderaxespad = 0, frameon = False, 
        prop = {'size': 16}, fontsize = "x-large")
    
    legend.set_title(legend_title, prop = {'size':18})

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel(xlabel, size = 16)
    ax.set_ylabel("Count", size = 16)
    ax.set_title(title, size = 20)
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    
    return(fig, ax)


def PlotHistIFRByAge(df, 
        numerator_var, denominator_var,
        age_group_var = "age_group",
        NBINS = None,
        group_labels = None,
        xlabel = "",
        xticklabels = None,
        density = False,
    ):
    """
    Plot IFR by age.  
    """
    
    a = 1.0
    bins = np.arange(0, NBINS + 1) - 0.1
    n_age = len(np.unique(df[age_group_var]))
    
    fig, ax = plt.subplots()
    
    height_n, bins_n = np.histogram(df[df[numerator_var] > 0][age_group_var], 
        bins, density = False)
    height_d, bins_d = np.histogram(df[df[denominator_var] > 0][age_group_var], 
        bins, density = False)
    
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
        age_group_var = "age_group",
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
        height, bins, objs = ax[axi].hist(df[df[var] > 0][age_group_var], bins, width = 0.8, 
            alpha = a, color = "#0072B2", edgecolor = "#0d1a26", linewidth = 0.5, 
            zorder = 3, density = density)
        
        for bi in range(len(bins) - 1):
            ax[axi].text(bins[bi] + 0.425, height[bi], str(np.round(height[bi], 2)), 
                ha = "center", va = "bottom", color = "grey", size = 12)
        
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



def plot_stacked_hist_by_group(df, 
        groupvar, binvar,
        NBINS = None,
        groups = None,
        group_labels = None,
        xlabel = "",
        ylabel = "",
        title = "",
        legend_title = "", legend_loc = "right",
        xticklabels = None
    ):
    """
    """
    # Define groups and size of groups
    if not groups:
        groups = np.unique(df[groupvar])
    n_groups = len(groups)
    
    if group_labels is None:
        group_labels = groups
    
    # Split by group
    hists = [df.loc[df[groupvar] == state][binvar] for state in groups]
    
    # Define the colourmap
    colours = get_discrete_viridis_colours(n_groups)
    
    bins = np.arange(NBINS)
    
    fig, ax = plt.subplots()
    
    #ax.grid(which = 'major', axis = 'y', alpha = 0.4, zorder = 0)
    ax.hist(hists, bins, stacked = True, label = group_labels, width = 0.8, 
        color = colours, edgecolor = "#0d1a26", linewidth = 0.5, 
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
        ax.set_xticks(bins + 0.45)
        ax.set_xticklabels(bins)
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    
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
        
        #ax[i].grid(which = 'major', axis = 'y', alpha = 0.4, zorder = 0)
        
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
