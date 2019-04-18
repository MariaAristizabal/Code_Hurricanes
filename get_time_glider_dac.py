#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 14:46:57 2018

@author: aristizabal
"""

#%%

from erddapy import ERDDAP
import pandas as pd

#%% Look for datasets 

server = 'https://data.ioos.us/gliders/erddap'

e = ERDDAP(server = server)

# Grab every dataset available
datasets = pd.read_csv(e.get_search_url(response='csv', search_for='all'))

kw2017 = {
    'min_lon': -81.0,
    'max_lon': -75.0,
    'min_lat': 25.0,
    'max_lat': 40.0,
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
    'latitude>=': 24.0,
    'latitude<=': 30.0,
    'longitude>=': -92.0,
    'longitude<=': -81.0,
}

variables = [
 'time',
]

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
    
    print(id,df.index[-1])

#print(e.get_download_url())


