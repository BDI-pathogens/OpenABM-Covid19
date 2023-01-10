"""
Utility functions for examples for OpenABM-Covid19 that use pre-generated spatial networks

Created: 10 Feb 2023
Author: dngrrng
"""
import example_utils as utils
import pandas as pd

from COVID19.model import Model

NETWORK_NAMES = ['primary', 'secondary',
                 'general_workforce', 'retired', 'elderly']

def load_demographics(demographics_path,index=0):
    demographics = pd.read_csv(demographics_path, comment="#", sep=",", skipinitialspace=True)
    demographics.rename(columns={'index': 'ID',
                                 'household.index': 'house_no',
                                 'age': 'age_group',
                                 'coord.x': 'x',
                                 'coord.y': 'y'}, inplace=True)
    # Converting we use network_no=0, as that's for household network
    demographics['network_no'] = index
    # Convert the age to age group (the decade age group of the person and is an integer between 0 (0-9 years) and 8 (80+).)
    demographics['age_group'] = demographics['age_group'].apply(lambda x: int(x / 10))
    return demographics

def coordinates_from_demographics(demographics):
    return demographics.rename(columns={'x':'xcoords','y':'ycoords'})[['ID','xcoords','ycoords']]

def load_edge_networks(project_path, prefix=''):

    networks_dict = {}
    network_col_names = ['ID_1', 'a.x', 'a.y', 'ID_2', 'b.x', 'b.y']

    for i, x in enumerate(NETWORK_NAMES):
        networks_dict[x] = pd.read_csv(project_path+'/'+prefix+"_"+str(i)+"_arcs.csv", comment="#", sep=",", skipinitialspace=True)
        networks_dict[x].columns = network_col_names

    return networks_dict

def update_model_networks(model,networks_dict,daily_fraction=0.5):
    for i,net in enumerate(networks_dict):
        model.delete_network(model.get_network_by_id(i+2))
        model.add_user_network(networks_dict[net],name=net,daily_fraction=daily_fraction)


def load_network_model(dir_path,prefix,demographics_path=None,daily_fraction=0.5,params_extra={}):
    # Load Demographics
    if demographics_path == None: demographics_path = dir_path+'/'+prefix+'_0_nodes.csv'
    demographics = load_demographics(demographics_path)
    coordinates = coordinates_from_demographics(demographics)
    # Load edges
    edge_networks = load_edge_networks(dir_path,prefix)
    # Get params
    params = utils.get_baseline_parameters()
    params.set_param('n_total',len(demographics))
    params.set_demographic_household_table(demographics)
    for p in params_extra:
        params.set_param(p,params_extra[p])
    # Get model
    model = Model(params)
    model.assign_coordinates_individuals(coordinates)
    # Assign networks
    update_model_networks(model,edge_networks,daily_fraction)
    return model