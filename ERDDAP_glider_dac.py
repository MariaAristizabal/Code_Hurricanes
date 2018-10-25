#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 16:14:18 2018

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

'''
# Search constraints
kw2017 = {
    'min_lon': -100.0,
    'max_lon': -60.0,
    'min_lat': 15.0,
    'max_lat': 45.0,
    'min_time': '2018-06-01T00:00:00Z',
    'max_time': '2018-11-30T00:00:00Z',
}
'''

kw2017 = {
    'min_lon': -92.0,
    'max_lon': -81.0,
    'min_lat': 24.0,
    'max_lat': 30.0,
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

server = 'https://data.ioos.us/gliders/erddap'

dataset_id = gliders[4]

constraints = {
    'time>=': '2018-06-01T00:00:00Z',
    'time<=': '2018-11-30T00:00:00Z',
    'latitude>=': 24.0,
    'latitude<=': 30.0,
    'longitude>=': -92.0,
    'longitude<=': -81.0,
}

variables = [
 'depth',
 'latitude',
 'longitude',
 'salinity',
 'temperature',
 'time',
]

#%%

e = ERDDAP(
    server=server,
    protocol='tabledap',
    response='nc'
)

e.dataset_id=gliders[4]
e.constraints=constraints
e.variables=variables

print(e.get_download_url())

#%%

df = e.to_pandas(
    index_col='time',
    parse_dates=True,
    skiprows=(1,)  # units information can be dropped.
).dropna()

df.head()

#%%

fig, ax = plt.subplots(figsize=(17, 5))
kw = dict(s=15, c=df['temperature'], marker='o', edgecolor='none')
cs = ax.scatter(df.index, df['depth'], **kw, cmap='RdYlBu_r')

ax.invert_yaxis()
ax.set_xlim(df.index[0], df.index[-1])
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

cbar = fig.colorbar(cs, orientation='vertical', extend='both')
cbar.ax.set_ylabel('Temperature ($^\circ$C)')
ax.set_ylabel('Depth (m)');



