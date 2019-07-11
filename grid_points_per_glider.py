#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 14:17:14 2018

@author: aristizabal
"""

#%% User input

# lat and lon bounds
lon_lim = [-110.0,-10.0]
lat_lim = [15.0,45.0]

# Time bounds
min_time = '2018-06-01T00:00:00Z'
max_time = '2018-11-30T00:00:00Z'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc'

# Server url 
server = 'https://data.ioos.us/gliders/erddap'

#GOFS3.1 outout model location
#catalog31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'
catalog31 = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/hycom_glbv_930_2018010112_t000_ts3z.nc'

#%%
  
#import netCDF4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
from erddapy import ERDDAP
import time
import math

#%% Look for datasets 

server = 'https://data.ioos.us/gliders/erddap'

e = ERDDAP(server = server)

# Grab every dataset available
datasets = pd.read_csv(e.get_search_url(response='csv', search_for='all'))

# Search constraints
kw2018 = {
    'min_lon': -100.0,
    'max_lon': -10.0,
    'min_lat': 15.0,
    'max_lat': 45.0,
    'min_time': '2018-06-01T00:00:00Z',
    'max_time': '2018-11-30T00:00:00Z',
}

search_url = e.get_search_url(response='csv', **kw2018)
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
    'longitude<=': -10.0,
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
        
glider = [l.split('-')[0] for l in gliders]

fund_agency = [None]*(len(glider))
for i,l in enumerate(glider):
    if glider[i][0:2] == 'ng':
        fund_agency[i] = 'Navy'
    if glider[i][0:2] == 'cp':
        fund_agency[i] = 'NSF'
    if glider[i][0:2] == 'sp':
        fund_agency[i] = 'NOAA'
    if glider[i][0:2] == 'SG':
        fund_agency[i] = 'NOAA'
    if glider[i][0:4] == 'ru28':
        fund_agency[i] = 'NJ'

ok = [i for i, l in enumerate(glider) if l == 'bass']
fund_agency[ok[0]] = 'NOAA'

ok = [i for i, l in enumerate(glider) if l == 'bios_minnie']
fund_agency[ok[0]] = 'BIOS'

ok = [i for i, l in enumerate(glider) if l == 'blue']
fund_agency[ok[0]] = 'NOAA'
     
ok = [i for i, l in enumerate(glider) if l == 'blue']
fund_agency[ok[0]] = 'NOAA'

ok = [i for i, l in enumerate(glider) if l == 'pelagia']
fund_agency[ok[0]] = 'NSF'

ok = [i for i, l in enumerate(gliders) if l == 'ramses-20180704T0000']
fund_agency[ok[0]] = 'NOAA'

ok = [i for i, l in enumerate(gliders) if l == 'ramses-20180907T0000']
fund_agency[ok[0]] = 'NSF'

ok = [i for i, l in enumerate(glider) if l == 'Reveille']
fund_agency[ok[0]] = 'NOAA'

ok = [i for i, l in enumerate(glider) if l == 'ru30']
fund_agency[ok[0]] = 'NSF'

ok = [i for i, l in enumerate(glider) if l == 'ru33']
fund_agency[ok[0]] = 'NOAA'

ok = [i for i, l in enumerate(glider) if l == 'sam']
fund_agency[ok[0]] = 'Fl'

ok = [i for i, l in enumerate(glider) if l == 'Sverdrup']
fund_agency[ok[0]] = 'NOAA'

ok = [i for i, l in enumerate(glider) if l == 'sylvia']
fund_agency[ok[0]] = 'NOAA'

ok = [i for i, l in enumerate(glider) if l == 'ud_476']
fund_agency[ok[0]] = 'NOAA'

ok = [i for i, l in enumerate(glider) if l == 'usf']
fund_agency[ok[0]] = 'NOAA'

ok = [i for i, l in enumerate(glider) if l == 'silbo']
fund_agency[ok[0]] = 'TWR'

#%% Glider in each category

n_navy = len([i for i,list in enumerate(fund_agency) if list == 'Navy'])
n_noaa = len([i for i,list in enumerate(fund_agency) if list == 'NOAA'])
n_nsf = len([i for i,list in enumerate(fund_agency) if list == 'NSF'])
n_nj = len([i for i,list in enumerate(fund_agency) if list == 'NJ'])
n_fl = len([i for i,list in enumerate(fund_agency) if list == 'Fl'])
n_bios = len([i for i,list in enumerate(fund_agency) if list == 'BIOS'])
n_twr = len([i for i,list in enumerate(fund_agency) if list == 'TWR'])        
    
#%% Download GOFS 3.1 output

GOFS31 = Dataset(catalog31)

lat31 = GOFS31.variables['lat'][:]
lon31 = GOFS31.variables['lon'][:]
depth31 = GOFS31.variables['depth'][:]

# Make time vector since I can not read the model output.
# NETCDF4 module seems to have problems reading remote files
timeini = datetime.strptime('Jun 1 2018', '%b %d %Y')

timeini = datetime.strptime(min_time, '%Y-%m-%dT%H:%M:%SZ')
timeend = datetime.strptime(max_time, '%Y-%m-%dT%H:%M:%SZ')
dt = datetime.strptime('Jun 1 2018 3', '%b %d %Y %H') - datetime.strptime('Jun 1 2018 0', '%b %d %Y %H') 

tt = timeini
DT = timeend - timeini
ntimesteps = DT.days * 8 # 3 hourly output
tt31 = pd.DataFrame(columns=['time'],index = np.arange(ntimesteps))

for i in np.arange(ntimesteps):
    tt31['time'][i] = tt
    tt += dt
    
#%% Making sure algorithm is correct

variables = ['latitude','longitude','time']

#id = 'ramses-20180907T0000'
#id = 'pelagia-20180910T0000'
#id ='bass-20180808T0000'
#id = 'bios_minnie-20180523T1617'
#id = 'ng309-20180701T0000'
id = 'ru33-20180801T1323'
i#d = 'silbo-20180525T1016'
e.dataset_id = id
e.constraints = constraints
e.variables = variables
    
df = e.to_pandas(
    index_col='time',
    parse_dates=True,
    skiprows=(1,)  # units information can be dropped.
    ).dropna()      
    
#%%  Conversion from glider longitude and latitude to GOFS convention    
  
target_lon = np.empty((len(df['longitude']),))
target_lon[:] = np.nan
for i in range(len(df['longitude'])):
    if df['longitude'][i] < 0: 
        target_lon[i] = 360 + df['longitude'][i]
    else:
        target_lon[i] = df['longitude'][i]
target_lat = df['latitude'][:]

#%% Changing times to timestamp

timeg = [time.mktime(df.index[i].timetuple()) for i in np.arange(len(df))]
time31 = [time.mktime(tt31['time'][i].timetuple()) for i in np.arange(len(tt31))]

# interpolating glider lon and lat to lat and lon on model time
sublon31=np.interp(time31,timeg,target_lon)
sublat31=np.interp(time31,timeg,target_lat)

# getting the model grid positions for sublon31 and sublat31
oklon31=np.round(np.interp(sublon31,lon31,np.arange(len(lon31)))).astype(int)
oklat31=np.round(np.interp(sublat31,lat31,np.arange(len(lat31)))).astype(int)

mat_lat_lon = np.array([oklon31,oklat31])
n_grid_points1 = len(np.unique(mat_lat_lon.T,axis=0))

#%%
fig, ax = plt.subplots(figsize=(7,6), dpi=80, facecolor='w', edgecolor='w')
ax.plot(oklon31,oklat31,'*')

#%% Making sure this is correct

fig, ax = plt.subplots(figsize=(7,6), dpi=80, facecolor='w', edgecolor='w') 
ax.plot(tt31['time'],sublon31,'.-r')
ax.plot(df.index,target_lon,'.-b')
#%%
fig, ax = plt.subplots(figsize=(7,6), dpi=80, facecolor='w', edgecolor='w') 
ax.plot(tt31['time'],sublat31,'.-r')
ax.plot(df.index,target_lat,'.-b')

#%%
meshlon31 = np.meshgrid(lon31,lat31)
meshlat31 = np.meshgrid(lat31,lon31)

#meshlon31_edge = 

fig, ax = plt.subplots(figsize=(7,6), dpi=80, facecolor='w', edgecolor='w')
#ax.pcolor(mesh31[0]) 
ax.plot(target_lon,target_lat,'.-b')
ax.plot(sublon31,sublat31,'.-r')
ax.plot(meshlon31[0][oklat31.min()-10:oklat31.max()+10,oklon31.min()-10:oklon31.max()+10],\
        meshlat31[0][oklon31.min()-10:oklon31.max()+10,oklat31.min()-10:oklat31.max()+10].T,'*k')
ax.plot(lon31[oklon31],lat31[oklat31],'*g')

#%% Calculating number of grid points covered by each glider

n_days = []
n_grid_points = []
for i, id in enumerate(gliders):
    if id[0:3] != 'all':
        print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        index_col='time',
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        
        n_days.append(len(df.index.map(lambda x: x.strftime('%Y-%m-%d')).unique()))
        
        # Conversion from glider longitude and latitude to GOFS convention   
        target_lon = np.empty((len(df['longitude']),))
        target_lon[:] = np.nan
        for l in range(len(df['longitude'])):
            if df['longitude'][l] < 0: 
                target_lon[l] = 360 + df['longitude'][l]
            else:
                target_lon[l] = df['longitude'][l]
            target_lat = df['latitude'][:]
            
        # Changing times to timestamp
        timeg = [time.mktime(df.index[x].timetuple()) for x in np.arange(len(df))]
        time31 = [time.mktime(tt31['time'][x].timetuple()) for x in np.arange(len(tt31))]

        # interpolating glider lon and lat to lat and lon on model time
        sublon31=np.interp(time31,timeg,target_lon)
        sublat31=np.interp(time31,timeg,target_lat)

        # getting the model grid positions for sublon31 and sublat31
        oklon31=np.round(np.interp(sublon31,lon31,np.arange(len(lon31)))).astype(int)
        oklat31=np.round(np.interp(sublat31,lat31,np.arange(len(lat31)))).astype(int)

        mat_lat_lon = np.array([oklon31,oklat31])
        n_grid_points.append(len(np.unique(mat_lat_lon.T,axis=0)))
        
total_grid_points = np.asarray(n_grid_points).sum()
total_days = np.asarray(n_days).sum()
n_grid_points_per_day = np.asarray(n_grid_points)/np.asarray(n_days)
#n_grid_points_per_day = np.ceil(np.asarray(n_grid_points)/np.asarray(n_days))

#%% Plotting the number of grid point in GOFS 3.1 covered by each glider during
#   Hurricane season 2018 

siz=12

funding = list(set(fund_agency))
color_fund1 = ['goldenrod','royalblue','firebrick','forestgreen','darkorange','black','rebeccapurple'] 

fig, ax = plt.subplots(figsize=(14, 12), dpi=80, facecolor='w', edgecolor='w') 
plt.title('Total Number of Grid Points Covered in GOFS 3.1 = ' + str(total_grid_points),fontsize=20)
ax.set_facecolor('lightgrey')
plt.xlabel('Number of Grid Points GOFS 3.1',fontsize = 20)
ax.grid(True)
plt.grid(color='k', linestyle='--', linewidth=1)
ax.set_ylim(-1,len(glider))

glider = [l.split('-')[0] for l in gliders]
ax.set_yticks(np.arange(len(glider)))
ax.set_yticklabels(glider,fontsize=12)
plt.tick_params(labelsize=13)

p1 = plt.barh(np.arange(len(glider)),n_grid_points)
              
for i, id in enumerate(gliders):
    if id[0:3] != 'all':
    
        if fund_agency[i] == 'Navy':
            h0 = plt.barh(i,n_grid_points[i],color=color_fund1[0])
        if fund_agency[i] == 'NOAA':
            h1 = plt.barh(i,n_grid_points[i],color=color_fund1[1])
        if fund_agency[i] == 'NSF':
            h2 = plt.barh(i,n_grid_points[i],color=color_fund1[2])
        if fund_agency[i] == 'NJ':
            h3 = plt.barh(i,n_grid_points[i],color=color_fund1[3])
        if fund_agency[i] == 'Fl':
            h4 = plt.barh(i,n_grid_points[i],color=color_fund1[4])
        if fund_agency[i] == 'BIOS':
            h5 = plt.barh(i,n_grid_points[i],color=color_fund1[5])
        if fund_agency[i] == 'TWR':
            h6 = plt.barh(i,n_grid_points[i],color=color_fund1[6])          
              
ax.legend([h0[0],h1[0],h2[0],h3[0],h4[0],h5[0],h6[0]],['Navy','NOAA','NSF','NJ','FL','BIOS','TWR'],\
          loc=5,fontsize=16)

plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Number_grid_points_hurric_season_2018.png"\
            ,bbox_inches = 'tight',pad_inches = 0) 

#%% Plotting the average number of grid point in GOFS 3.1 covered by each glider per day during
#   Hurricane season 2018 

siz=12

funding = list(set(fund_agency))
color_fund1 = ['goldenrod','royalblue','firebrick','forestgreen','darkorange','black','rebeccapurple'] 

fig, ax = plt.subplots(figsize=(14, 12), dpi=80, facecolor='w', edgecolor='w') 
plt.title('Average Number of grid points Covered Per Day in GOFS 3.1 = ' \
          + str(math.floor((total_grid_points/total_days)*10)/10),fontsize=20)
ax.set_facecolor('lightgrey')
plt.xlabel('Number of Grid Points Covered Per Day GOFS 3.1',fontsize = 20)
ax.grid(True)
plt.grid(color='k', linestyle='--', linewidth=1)
ax.set_ylim(-1,len(glider))

glider = [l.split('-')[0] for l in gliders]
ax.set_yticks(np.arange(len(glider)))
ax.set_yticklabels(glider,fontsize=12)
plt.tick_params(labelsize=13)

p1 = plt.barh(np.arange(len(glider)),n_grid_points_per_day)
              
for i, id in enumerate(gliders):
    if id[0:3] != 'all':
    
        if fund_agency[i] == 'Navy':
            h0 = plt.barh(i,n_grid_points_per_day[i],color=color_fund1[0])
        if fund_agency[i] == 'NOAA':
            h1 = plt.barh(i,n_grid_points_per_day[i],color=color_fund1[1])
        if fund_agency[i] == 'NSF':
            h2 = plt.barh(i,n_grid_points_per_day[i],color=color_fund1[2])
        if fund_agency[i] == 'NJ':
            h3 = plt.barh(i,n_grid_points_per_day[i],color=color_fund1[3])
        if fund_agency[i] == 'Fl':
            h4 = plt.barh(i,n_grid_points_per_day[i],color=color_fund1[4])
        if fund_agency[i] == 'BIOS':
            h5 = plt.barh(i,n_grid_points_per_day[i],color=color_fund1[5])
        if fund_agency[i] == 'TWR':
            h6 = plt.barh(i,n_grid_points_per_day[i],color=color_fund1[6])          
              
ax.legend([h0[0],h1[0],h2[0],h3[0],h4[0],h5[0],h6[0]],['Navy','NOAA','NSF','NJ','FL','BIOS','TWR'],\
          loc=5,fontsize=16)

plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Number_grid_points_per_day_hurric_season_2018.png"\
            ,bbox_inches = 'tight',pad_inches = 0)  