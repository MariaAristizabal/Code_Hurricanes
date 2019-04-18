#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 14:07:55 2019

@author: aristizabal
"""

import netCDF4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import datetime
import pytz
import requests
from bs4 import BeautifulSoup
from erddapy import ERDDAP
import pandas as pd
import xarray as xr

#import xarray as xr

#%% User input

# Directories where increments files reside 
Dir= '/Volumes/aristizabal/GOFS/'

# Gulf of Mexico
lon_lim = [-98,-80]
lat_lim = [16,34]

#Initial and final date
#tini = datetime.datetime(2018,9,11,0,0,0)
#tend = datetime.datetime(2018,9,12,0,0,0)
tini = datetime.datetime(2018,10,10,0,0,0)
tend = datetime.datetime(2018,10,11,0,0,0)

# Time bounds
min_time = '2018-10-10T00:00:00Z'
max_time = '2018-10-11T00:00:00Z'

# Server url 
glider_server = 'https://data.ioos.us/gliders/erddap'

Dir_Argo = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/DataSelection_20190116_210914_7419347'

url = 'https://data.ioos.us/thredds/dodsC/deployments/'

# Michael
jason_url = 'https://podaac-opendap.jpl.nasa.gov:443/opendap/allData/ostm/preview/L2/GPS-OGDR/c608/'
date = ['20181010','20181009','20181008','20181007','20181006']

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

cryosat_folder = '/Volumes/aristizabal/CryoSat_data/'
date = ['2018/09/','2018/10/']

#%% Look for datasets 

e = ERDDAP(server = glider_server)

# Grab every dataset available
datasets = pd.read_csv(e.get_search_url(response='csv', search_for='all'))

kw2018 = {
    'min_lon': lon_lim[0],
    'max_lon': lon_lim[-1],
    'min_lat': lat_lim[0],
    'max_lat': lat_lim[-1],
    'min_time': min_time,
    'max_time': max_time,
}

search_url = e.get_search_url(response='csv', **kw2018)

# Grab the results
search = pd.read_csv(search_url)

# Extract the IDs
gliders = search['Dataset ID'].values

msg = 'Found {} Glider Datasets:\n\n{}'.format
print(msg(len(gliders), '\n'.join(gliders)))

#%%

constraints = {
    'time>=': min_time,
    'time<=': max_time,
    'latitude>=': lat_lim[0],
    'latitude<=': lat_lim[-1],
    'longitude>=': lon_lim[0],
    'longitude<=': lon_lim[-1],
}

variables = [
 'time','latitude','longitude'
]

#%%

e = ERDDAP(
    server=glider_server,
    protocol='tabledap',
    response='nc'
)

'''
for id in gliders:
    e.dataset_id = id
    e.constraints = constraints
    e.variables = variables
    
    df = e.to_pandas(
    index_col='time',
    parse_dates=True,
    skiprows=(1,)  # units information can be dropped.
).dropna()
    
    print(id,df.index[-1])
'''   
 #%% Reading ncoda data

ncoda_files = sorted(glob.glob(os.path.join(Dir,'*seatmp*')))

ncncoda = Dataset(ncoda_files[2]) # Michael
#ncncoda = Dataset(ncoda_files[0]) # Florence
temp_incr = ncncoda.variables['pot_temp'][:]

time_ncoda = ncncoda.variables['MT'] # Michael
time_ncoda = np.transpose(netCDF4.num2date(time_ncoda[:],time_ncoda.units))
depth_ncoda = ncncoda.variables['Depth'][:]
lat_ncoda = ncncoda.variables['Latitude'][:]
lon_ncoda = ncncoda.variables['Longitude'][:]   

#%% Reading bathymetry data

ncbath = Dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
#oklatbath = oklatbath[:,np.newaxis]
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])
#oklonbath = oklonbath[:,np.newaxis]

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
#bath_elevsub = bath_elev[oklatbath,oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath]

#%% Reading Argo data
argo_files = sorted(glob.glob(os.path.join(Dir_Argo,'*.nc')))

ncargo = Dataset(argo_files[0])
argo_id = ncargo.variables['PLATFORM_NUMBER'][:]
argo_lat = ncargo.variables['LATITUDE'][:]
argo_lon = ncargo.variables['LONGITUDE'][:]

argo_tim = ncargo.variables['JULD']#[:]
argo_time = netCDF4.num2date(argo_tim[:],argo_tim.units)

#%% Findind url list from Jason2
 
r = requests.get(jason_url)
data = r.text
soup = BeautifulSoup(data,"lxml")

fold = []
for s in soup.find_all("a"):
    fold.append(s.get("href").split('/')[0])
 
nc_list = []
for f in fold:
    elem = f.split('_')
    for l in elem:    
        for m in date:
            if l == m:
                nc_list.append(f.split('.')[0]+'.nc')
nc_list = list(set(nc_list))

#%% Access CryoSat files

nc_listCry = os.listdir(cryosat_folder)

#%% Tentative Michael path

lonMc = np.array([-84.9,-85.2,-85.3,-85.9,-86.2,-86.4,-86.5,-86.5,-86.3,-86.2,\
                  -86.0,-85.8,-85.5,-85.2,-84.9,-84.5,-84.1])
latMc = np.array([21.2,22.2,23.2,24.1,25.0,26.0,27.1,28.3,28.8,29.1,29.4,29.6,\
                  30.0,30.6,31.1,31.5,31.9])
tMc = ['2018/10/08/15/00','2018/10/08/21/00','2018/10/09/03/00','2018/10/09/09/00',\
       '2018/10/09/15/00','2018/10/09/21/00','2018/10/10/03/00','2018/10/10/09/00',\
       '2018/10/10/11/00','2018/10/10/13/00','2018/10/10/15/00','2018/10/10/16/00',\
       '2018/10/10/18/00','2018/10/10/20/00','2018/10/10/22/00','2018/10/11/00/00',\
       '2018/10/11/02/00']

# Convert time to UTC
pst = pytz.timezone('America/New_York') # time zone
utc = pytz.UTC 

timeMc = [None]*len(tMc) 
for x in range(len(tMc)):
    timeMc[x] = datetime.datetime.strptime(tMc[x], '%Y/%m/%d/%H/%M') # time in time zone
    
    
#%% increment in Gulf Mexico surface
    
siz=12
min_val = -2.4
max_val = 2.4
nlevels = 25

oklonncoda = np.where(np.logical_and(lon_ncoda-360 > lon_lim[0], lon_ncoda-360 < lon_lim[-1]))
oklatncoda = np.where(np.logical_and(lat_ncoda > lat_lim[0], lat_ncoda < lat_lim[-1]))

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)

kw = dict(levels = np.linspace(min_val,max_val,nlevels))
cs = ax.contourf(lon_ncoda[oklonncoda]-360,lat_ncoda[oklatncoda],\
                 temp_incr[0,0,oklatncoda[0][0]:oklatncoda[0][-1]+1,oklonncoda[0][0]:oklonncoda[0][-1]+1],\
                 cmap=plt.get_cmap('seismic'),**kw)
cs=plt.colorbar(cs)
cs.ax.set_ylabel('$(^oc)$',fontsize=16,labelpad=15)
cs.ax.tick_params(labelsize=14) 

#ax.grid(False)
ax.set_ylim([lat_lim[0],lat_lim[-1]])
ax.set_xlim([lon_lim[0],lon_lim[-1]])
ax.plot(lonMc,latMc,'o-',markersize = 5,label = 'Michael Track',color = 'dimgray')
plt.title('Temperature Increments at surface on 2018-10-10',size = 18)

for x in range(0, len(tMc)-1, 2):
    ax.text(lonMc[x],latMc[x],timeMc[x].strftime('%d, %H:%M'),size = 8)

for id in gliders[1:-1]:
    e.dataset_id = id
    e.constraints = constraints
    e.variables = variables
    
    df = e.to_pandas(
    index_col='time',
    parse_dates=True,
    skiprows=(1,)  # units information can be dropped.
    ).dropna()
    
    ax.plot(np.mean(df.longitude),np.mean(df.latitude),'o',markersize = 10, \
                markeredgecolor = 'black',markeredgewidth=2,label =  id.split('-')[0]) 
    legend = ax.legend(loc='upper left',fontsize = siz)
    
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/temp_increment_during_Michael_surf.png"\
            ,bbox_inches = 'tight',pad_inches = 0)   
        
#%% Increments Gulf Mexico at 100 m 
    
siz=12
min_val = -2.4
max_val = 2.4
nlevels = 25

oklonncoda = np.where(np.logical_and(lon_ncoda-360 > lon_lim[0], lon_ncoda-360 < lon_lim[-1]))
oklatncoda = np.where(np.logical_and(lat_ncoda > lat_lim[0], lat_ncoda < lat_lim[-1]))

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)

kw = dict(levels = np.linspace(min_val,max_val,nlevels))
cs = ax.contourf(lon_ncoda[oklonncoda]-360,lat_ncoda[oklatncoda],\
                 temp_incr[0,9,oklatncoda[0][0]:oklatncoda[0][-1]+1,oklonncoda[0][0]:oklonncoda[0][-1]+1],\
                 cmap=plt.get_cmap('seismic'),**kw)
cs=plt.colorbar(cs)
cs.ax.set_ylabel('$(^oc)$',fontsize=16,labelpad=15)
cs.ax.tick_params(labelsize=14) 

#ax.grid(False)
ax.set_ylim([lat_lim[0],lat_lim[-1]])
ax.set_xlim([lon_lim[0],lon_lim[-1]])
ax.plot(lonMc,latMc,'o-',markersize = 5,label = 'Michael Track',color = 'dimgray')
plt.title('Temperature Increments at 100 m on 2018-10-10',size = 18)

for l in argo_files:
    ncargo = Dataset(l)
    argo_lat = ncargo.variables['LATITUDE'][:][0]
    argo_lon = ncargo.variables['LONGITUDE'][:][0]
    ax.plot(argo_lon,argo_lat,'^k',markersize = 10)

for x in range(0, len(tMc)-1, 2):
    ax.text(lonMc[x],latMc[x],timeMc[x].strftime('%d, %H:%M'),size = 8)

for id in gliders[1:-1]:
    e.dataset_id = id
    e.constraints = constraints
    e.variables = variables
    
    df = e.to_pandas(
    index_col='time',
    parse_dates=True,
    skiprows=(1,)  # units information can be dropped.
    ).dropna()
    
    ax.plot(np.mean(df.longitude),np.mean(df.latitude),'o',markersize = 10, \
                markeredgecolor = 'black',markeredgewidth=2,label =  id.split('-')[0]) 
    legend = ax.legend(loc='upper left',fontsize = siz)
    
for l in nc_list:
    print(l)
    ncjason = xr.open_dataset(jason_url + l, decode_times=False) 

    ssha = ncjason.ssha
    lat_jason = ncjason.lat
    lon_jason = ncjason.lon
    time_jason = ncjason.time
    time_jason = np.transpose(netCDF4.num2date(time_jason[:],time_jason.units))
    
    oklon = np.logical_and(lon_jason > lon_lim[0]+360,lon_jason < lon_lim[-1]+360)
    sshasub = ssha[oklon]
    lonsub = lon_jason[oklon]
    latsub = lat_jason[oklon]
    oklat = np.logical_and(latsub > lat_lim[0],latsub < lat_lim[-1])
    if l.split('_')[4][6:]=='10':
        kw = dict(c=sshasub[oklat]*0, marker='.',cmap=plt.get_cmap('gray'),s=1)
    else:
        kw = dict(c=sshasub[oklat]*0, marker='.',cmap=plt.get_cmap('Blues'),s=1)
    ax.scatter(lonsub[oklat]-360, latsub[oklat], **kw) 
    
#for l in nc_list:
dates_michael = ['20181010','20181009','20181008','20181007','20181006']
nc_listsub = []
for l in nc_listCry:
    for m in dates_michael:
        if l.split('_')[6][0:8] == m:
            nc_listsub.append(l) 

for l in nc_listsub:    
    print(l)
    nccryosat = xr.open_dataset(cryosat_folder + l, decode_times=False) 

    ssha = nccryosat.ssha_20_ku
    lat_cryosat = nccryosat.lat_20_ku
    lon_cryosat = nccryosat.lon_20_ku
    
    oklon = np.logical_and(lon_cryosat > lon_lim[0],lon_cryosat < lon_lim[-1])
    sshasub = ssha[oklon]
    lonsub = lon_cryosat[oklon]
    latsub = lat_cryosat[oklon]
    oklat = np.logical_and(latsub > lat_lim[0],latsub < lat_lim[-1])
    if l.split('_')[6][6:8]=='10':
        print('copper')
        kw = dict(c=sshasub[oklat]*0, marker='.',cmap=plt.get_cmap('copper'),s=1)
    else:
        print('blues')
        kw = dict(c=sshasub[oklat]*0, marker='.',cmap=plt.get_cmap('Blues'),s=1)
    ax.scatter(lonsub[oklat], latsub[oklat], **kw) 
    
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/temp_increment_during_Michael_100m.png"\
            ,bbox_inches = 'tight',pad_inches = 0)
