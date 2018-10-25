#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 15:33:17 2018

@author: aristizabal
"""

#%%

from erddapy import ERDDAP
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns # package for nice plotting defaults
sns.set()

#%% Look for datasets 

server = 'https://data.ioos.us/gliders/erddap'

e = ERDDAP(server = server)

# Grab every dataset available
datasets = pd.read_csv(e.get_search_url(response='csv', search_for='all'))

# Search constraints
kw2017 = {
    'min_lon': -100.0,
    'max_lon': -60.0,
    'min_lat': 15.0,
    'max_lat': 45.0,
    'min_time': '2018-06-01T00:00:00Z',
    'max_time': '2018-11-30T00:00:00Z',
}

search_url = e.get_search_url(response='csv', **kw2017)
#print(search_url)

# Grab the results
search = pd.read_csv(search_url)

# Extract the IDs
gliders = search['Dataset ID'].values

msg = 'Found {} Glider Datasets:\n\n{}'.format
print(msg(len(gliders), '\n'.join(gliders)))

#%%

constraints = {
    'time>=': '2018-06-01T00:00:00Z',
    'time<=': '2018-11-30T00:00:00Z',
    'latitude>=': 15.0,
    'latitude<=': 45.0,
    'longitude>=': -100.0,
    'longitude<=': -60.0,
}

variables = ['latitude','longitude','time']

#%%

e = ERDDAP(
    server=server,
    protocol='tabledap',
    response='nc'
)

for id in gliders:
    e.dataset_id=id
    e.constraints=constraints
    e.variables=variables
    
    df = e.to_pandas(
    index_col='time',
    parse_dates=True,
    skiprows=(1,)  # units information can be dropped.
                    ).dropna()
        

#%%
e.dataset_id=gliders[5]
e.constraints=constraints
e.variables=variables
    
df = e.to_pandas(
index_col='time',
parse_dates=True,
skiprows=(1,)  # units information can be dropped.
                    ).dropna()    

