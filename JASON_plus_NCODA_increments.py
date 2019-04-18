#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 16:46:42 2018

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

import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from erddapy import ERDDAP
import pandas as pd

from bs4 import BeautifulSoup
import requests

import xarray as xr

#%% User input

# Directories where increments files reside 
Dir= '/Volumes/aristizabal/GOFS/'

# Gulf of Mexico
#lon_lim = [-98,-78]
#lat_lim = [18,32]

#Initial and final date
#tini = datetime.datetime(2018,9,11,0,0,0)
#tend = datetime.datetime(2018,9,12,0,0,0)
tini = datetime.datetime(2018,10,10,0,0,0)
tend = datetime.datetime(2018,10,11,0,0,0)

# lat and lon bounds
lon_lim = [-100,-10]
lat_lim = [0,50]  

# Time bounds
min_time = '2018-06-01T00:00:00Z'
max_time = '2018-11-30T00:00:00Z'

# Server url 
server = 'https://data.ioos.us/gliders/erddap'

# Bathymetry file
#bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc'
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

# Argo data folder
# Florence
#Dir_Argo = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/DataSelection_20190116_210524_7419306' 
# Michael
Dir_Argo = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/DataSelection_20190116_210914_7419347'

url = 'https://data.ioos.us/thredds/dodsC/deployments/'

# Altimeter from Jason2
# Florence
#jason_url = 'https://podaac-opendap.jpl.nasa.gov:443/opendap/allData/ostm/preview/L2/GPS-OGDR/c605/' 
#date = ['20180911','20180910','20180909','20180908','20180907']

# Michael
jason_url = 'https://podaac-opendap.jpl.nasa.gov:443/opendap/allData/ostm/preview/L2/GPS-OGDR/c608/'
date = ['20181010','20181009','20181008','20181007','20181006']

# Gulf Mexico
id_list = ['rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc',\
           'rutgers/ng261-20180801T0000/ng261-20180801T0000.nc3.nc',\
           'rutgers/ng257-20180801T0000/ng257-20180801T0000.nc3.nc',\
           'rutgers/ng290-20180701T0000/ng290-20180701T0000.nc3.nc',\
           'rutgers/ng230-20180801T0000/ng230-20180801T0000.nc3.nc',\
           'rutgers/ng279-20180801T0000/ng279-20180801T0000.nc3.nc',\
           'rutgers/ng429-20180701T0000/ng429-20180701T0000.nc3.nc',\
           'rutgers/ng258-20180801T0000/ng258-20180801T0000.nc3.nc',\
           'rutgers/ng295-20180701T0000/ng295-20180701T0000.nc3.nc',\
           'rutgers/ng296-20180701T0000/ng296-20180701T0000.nc3.nc',\
           'rutgers/ng309-20180701T0000/ng309-20180701T0000.nc3.nc',\
           'rutgers/ng342-20180701T0000/ng342-20180701T0000.nc3.nc',\
           'rutgers/ng448-20180701T0000/ng448-20180701T0000.nc3.nc',\
           'rutgers/ng450-20180701T0000/ng450-20180701T0000.nc3.nc',\
           'rutgers/ng464-20180701T0000/ng464-20180701T0000.nc3.nc',\
           'rutgers/ng466-20180701T0000/ng466-20180701T0000.nc3.nc',\
           'rutgers/ng512-20180701T0000/ng512-20180701T0000.nc3.nc',\
           'secoora/sam-20180824T0000/sam-20180824T0000.nc3.nc',\
           'gcoos_dmac/Sverdrup-20180509T1742/Sverdrup-20180509T1742.nc3.nc']

'''
# Caribbean:
lon_lim = [-70,-62]
lat_lim = [16,22] 
id_list = ['aoml/SG630-20180716T1220/SG630-20180716T1220.nc3.nc',\
                'rutgers/ng300-20180701T0000/ng300-20180701T0000.nc3.nc',\
                'aoml/SG610-20180719T1146/SG610-20180719T1146.nc3.nc',\
                'aoml/SG635-20180716T1248/SG635-20180716T1248.nc3.nc',\
                'aoml/SG649-20180731T1418/SG649-20180731T1418.nc3.nc',\
                'rutgers/ng291-20180701T0000/ng291-20180701T0000.nc3.nc',\
                'rutgers/ng300-20180701T0000/ng300-20180701T0000.nc3.nc',\
                'rutgers/ng302-20180701T0000/ng302-20180701T0000.nc3.nc',\
                'rutgers/ng467-20180701T0000/ng467-20180701T0000.nc3.nc',\
                'rutgers/ng487-20180701T0000/ng487-20180701T0000.nc3.nc',\
                'rutgers/ng488-20180701T0000/ng488-20180701T0000.nc3.nc',\
                'rutgers/ng616-20180701T0000/ng616-20180701T0000.nc3.nc',\
                'rutgers/ng617-20180701T0000/ng617-20180701T0000.nc3.nc',\
                'rutgers/ng618-20180701T0000/ng618-20180701T0000.nc3.nc',\
                'rutgers/ng619-20180701T0000/ng619-20180701T0000.nc3.nc']

# SAB + MAB
lon_lim = [-81,-70]
lat_lim = [25,35]       

id_list = ['rutgers/blue-20180806T1400/blue-20180806T1400.nc3.nc',\
                'rutgers/ru28-20180920T1334/ru28-20180920T1334.nc3.nc',\
                'rutgers/ru30-20180705T1825/ru30-20180705T1825.nc3.nc',\
                'rutgers/ru33-20180801T1323/ru33-20180801T1323.nc3.nc',\
                'rutgers/sylvia-20180802T0930/sylvia-20180802T0930.nc3.nc',\
                'rutgers/cp_376-20180724T1552/cp_376-20180724T1552.nc3.nc',\
                'rutgers/cp_389-20180724T1620/cp_389-20180724T1620.nc3.nc',\
                'rutgers/cp_336-20180724T1433/cp_336-20180724T1433.nc3.nc',\
                'secoora/ramses-20180907T0000/ramses-20180907T0000.nc3.nc',\
                'secoora/ramses-20180704T0000/ramses-20180704T0000.nc3.nc',\
                'secoora/bass-20180808T0000/bass-20180808T0000.nc3.nc',\
                'secoora/pelagia-20180910T0000/pelagia-20180910T0000.nc3.nc',\
                ]
#'drudnick/sp010-20180620T1455/sp010-20180620T1455.nc3.nc',\
#'drudnick/sp022-20180422T1229/sp022-20180422T1229.nc3.nc',\
#'drudnick/sp066-20180629T1411/sp066-20180629T1411.nc3.nc',\
#'drudnick/sp069-20180411T1516/sp069-20180411T1516.nc3.nc',\
'''

#%% Look for datasets 

e = ERDDAP(server = server)

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
#print(search_url)

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
    server=server,
    protocol='tabledap',
    response='nc'
)

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

'''
# Conversion from glider longitude and latitude to RTOFS convention
target_lon = []
for lon in longlider:
    if lon < 0: 
        target_lon.append(360 + lon)
    else:
        target_lon.append(lon)
target_lon = np.array(target_lon)
target_lat = latglider
'''

#%% Reading Argo data
argo_files = sorted(glob.glob(os.path.join(Dir_Argo,'*.nc')))

ncargo = Dataset(argo_files[0])
argo_id = ncargo.variables['PLATFORM_NUMBER'][:]
argo_lat = ncargo.variables['LATITUDE'][:]
argo_lon = ncargo.variables['LONGITUDE'][:]

argo_tim = ncargo.variables['JULD']#[:]
argo_time = netCDF4.num2date(argo_tim[:],argo_tim.units)

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
    
#%% Tentative Florence path

lonFl = np.array([-61,-63,-66,-69,-72,-77,-78,-79])
latFl = np.array([25,26,27,29,31,34,36,37])
tFl = ['2018/09/10/17/00','2018/09/11/10/00','2018/09/11/14/00','2018/09/12/02/00',
       '2018/09/12/14/00','2018/09/13/14/00','2018/09/14/14/00','2018/09/15/14/00']


# Convert time to UTC
pst = pytz.timezone('America/New_York') # time zone
utc = pytz.UTC 

timeFl = [None]*len(tFl) 
for x in range(len(tFl)):
    d = datetime.datetime.strptime(tFl[x], '%Y/%m/%d/%H/%M') # time in time zone
    d = pst.localize(d) # add time zone to date
    d = d.astimezone(utc)
    timeFl[x] = d.astimezone(utc)
    #print(timeFl[x].strftime('%d, %H %M'))
    
    
#%% increment in Gulf Mexico surface
    
siz=12
#plt.style.use('seaborn-dark')
#plt.style.use('ggplot')

#import seaborn as sns # package for nice plotting defaults
#sns.set() 
#plt.style.use('v2.0')

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
cs = ax.contourf(lon_ncoda[3200:3700]-360,lat_ncoda[2200:3100],temp_incr[0,0,2200:3100,3200:3700],20,cmap=plt.get_cmap('seismic'), vmin=-2.4, vmax=2.4)
plt.colorbar(cs)
ax.set_facecolor('lightgrey')
ax.grid(False)
plt.tick_params(top='on', bottom='on', left='on', right='on')
plt.axis('equal')
plt.axis([lon_lim[0],lon_lim[-1],lat_lim[0],lat_lim[-1]])
ax.plot(lonMc,latMc,'o-',markersize = 5,label = 'Michael Track',color = 'dimgray')
plt.title('Temperature Increments at surface on 2018-10-10',size = 18)

for x in range(0, len(tMc), 2):
    ax.text(lonMc[x],latMc[x],timeMc[x].strftime('%d, %H:%M'),size = 7)

for l in id_list:
    ncng = Dataset(url + l)
    time_n = ncng.variables['time']
    time_ng = netCDF4.num2date(time_n[:],time_n.units)
    oktime = np.logical_and(time_ng >= datetime.datetime(2018, 10, 10,0,0,0),time_ng >= datetime.datetime(2018, 10, 11,0,0,0))
    if np.sum(np.where(oktime)) != 0:
        lat_ng = ncng.variables['latitude'][:]
        lon_ng = ncng.variables['longitude'][:]
        ax.plot(np.mean(lon_ng[oktime]),np.mean(lat_ng[oktime]),'o',markersize = 10, \
                markeredgecolor = 'black',markeredgewidth=2,label = ncng.id.split('-')[0]) 
        legend = ax.legend(loc='upper left',fontsize = siz)
        legend.get_frame().set_color('white') 

    
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/temp_increment_during_Michael_surf.png"\
            ,bbox_inches = 'tight',pad_inches = 0)   
        
#%% Increments Gulf Mexico at 100 m 
    
siz=12

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
cs = ax.contourf(lon_ncoda[3200:3700]-360,lat_ncoda[2200:3100],temp_incr[0,9,2200:3100,3200:3700],20,cmap=plt.get_cmap('seismic'), vmin=-2.4, vmax=2.4)
plt.colorbar(cs)
ax.set_facecolor('lightgrey')
plt.axis('equal')
plt.axis([lon_lim[0],lon_lim[-1],lat_lim[0],lat_lim[-1]])
ax.plot(lonMc,latMc,'o-',markersize = 5,label = 'Michael Track',color = 'dimgray')
plt.title('Temperature Increments at 100 m on 2018-10-10',size = 18)

for x in range(0, len(tMc), 2):
    ax.text(lonMc[x],latMc[x],timeMc[x].strftime('%d, %H:%M'),size = siz)

for l in id_list:
    ncng = Dataset(url + l)
    time_n = ncng.variables['time']
    time_ng = netCDF4.num2date(time_n[:],time_n.units)
    oktime = np.logical_and(time_ng >= datetime.datetime(2018, 10, 10,0,0,0),time_ng >= datetime.datetime(2018, 10, 11,0,0,0))
    if np.sum(np.where(oktime)) != 0:
        lat_ng = ncng.variables['latitude'][:]
        lon_ng = ncng.variables['longitude'][:]
        ax.plot(np.mean(lon_ng[oktime]),np.mean(lat_ng[oktime]),'o',markersize = 10,\
                markeredgecolor = 'black',markeredgewidth=2,label = ncng.id.split('-')[0]) 
        ax.legend(loc='upper left',fontsize = siz)
        
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/temp_increment_during_Michael_100m.png"\
            ,bbox_inches = 'tight',pad_inches = 0)   
        
#%% increment SAB + MAB surface Florence
    
siz=12

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
cs = ax.contourf(lon_ncoda-360,lat_ncoda,temp_incr[0,0,:,:],20,cmap=plt.get_cmap('seismic'), vmin=-4.5, vmax=4.5)
cbar = plt.colorbar(cs)
cbar.set_label('Temperature Increments ($^o$C)',rotation=90,size = 18)
#plt.colorbar()
#ax.contour(bath_lon[oklonbath],bath_lat[oklatbath],bath_elev[np.c_[oklatbath],oklonbath],colors='k')   
plt.axis('equal')
plt.axis([-80,-65,25,42])
ax.plot(lonFl[1:-2],latFl[1:-2],'o-',markersize = 5,color = 'dimgray') #label = 'Florence Track',
for x in range(2,len(tFl)-2):
    ax.text(lonFl[x],latFl[x],timeFl[x].strftime('%d, %H:%M'),size = siz)
plt.title('Temperature Increments at surface on 2018-09-11',size = 18)
ax.set_facecolor('lightgrey')

#for x in range(0, len(tMc), 2):
#    ax.text(lonMc[x],latMc[x],timeMc[x].strftime('%d, %H:%M'),size = siz)

for l in id_list:
    print(l)
    ncng = Dataset(url + l)
    time_n = ncng.variables['time']
    time_ng = netCDF4.num2date(time_n[:],time_n.units)
    oktime = np.logical_and(time_ng >= datetime.datetime(2018,9,11,0,0,0),time_ng <= datetime.datetime(2018,9,12,0,0,0))
    if np.sum(np.where(oktime)) != 0:
        lat_ng = ncng.variables['latitude'][:]
        lon_ng = ncng.variables['longitude'][:]
        print(np.mean(lon_ng[oktime]))
        #print(np.mean(lat_ng[oktime]))
        ax.plot(np.mean(lon_ng[oktime]),np.mean(lat_ng[oktime]),'o',markersize = 10,\
                markeredgecolor='black', markeredgewidth=2,label = ncng.id.split('-')[0]) 
        ax.legend(loc='upper left',fontsize = siz)
        
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/temp_increment_during_Florence_surface.png"\
            ,bbox_inches = 'tight',pad_inches = 0)        
plt.show()  

#%% increment SAB + MAB 60m Florence
    
siz=12

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
cs = ax.contourf(lon_ncoda-360,lat_ncoda,temp_incr[0,6,:,:],20,cmap=plt.get_cmap('seismic'), vmin=-4.5, vmax=4.5)
cbar = plt.colorbar(cs)
cbar.set_label('Temperature Increments ($^o$C)',rotation=90,size = 18)
#plt.colorbar()
#ax.contour(bath_lon[oklonbath],bath_lat[oklatbath],bath_elev[np.c_[oklatbath],oklonbath],colors='k')   
plt.axis('equal')
plt.axis([-80,-65,25,42])
ax.plot(lonFl[1:-2],latFl[1:-2],'o-',markersize = 10,color = 'dimgray') #label = 'Florence Track'
for x in range(2,len(tFl)-2):
    ax.text(lonFl[x],latFl[x],timeFl[x].strftime('%d, %H:%M'),size = siz)
plt.title('Temperature Increments at 60 m',size = 18)

#for x in range(0, len(tMc), 2):
#    ax.text(lonMc[x],latMc[x],timeMc[x].strftime('%d, %H:%M'),size = siz)

for l in id_list:
    print(l)
    ncng = Dataset(url + l)
    time_n = ncng.variables['time']
    time_ng = netCDF4.num2date(time_n[:],time_n.units)
    oktime = np.logical_and(time_ng >= datetime.datetime(2018,9,11,0,0,0),time_ng >= datetime.datetime(2018,9,12,0,0,0))
    if np.sum(np.where(oktime)) != 0:
        lat_ng = ncng.variables['latitude'][:]
        lon_ng = ncng.variables['longitude'][:]
        print(np.mean(lon_ng[oktime]))
        print(np.mean(lat_ng[oktime]))
        ax.plot(np.mean(lon_ng[oktime]),np.mean(lat_ng[oktime]),'o',markersize = 10,\
                markeredgecolor='black', markeredgewidth=2,label = ncng.id.split('-')[0]) 
        ax.legend(loc='upper left',fontsize = siz)
        
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/temp_increment_during_Florence_60m.png")        
#plt.show()  
        
#%% increment Caribbean surface Florence
    
siz=12

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
cs = ax.contourf(lon_ncoda-360,lat_ncoda,temp_incr[0,0,:,:],20,cmap=plt.get_cmap('seismic'), vmin=-4.5, vmax=4.5)
cbar = plt.colorbar(cs)
cbar.set_label('Temperature Increments ($^o$C)',rotation=90,size = 18)
#plt.colorbar()
#ax.contour(bath_lon[oklonbath],bath_lat[oklatbath],bath_elev[np.c_[oklatbath],oklonbath],colors='k')   
plt.axis('equal')
plt.axis([-70,-62,16,22])
ax.plot(lonFl[1:-2],latFl[1:-2],'o-',markersize = 5,color = 'dimgray') #label = 'Florence Track',
plt.title('Temperature Increments at surface on 2018-09-11',size = 18)
ax.set_facecolor('lightgrey')


for l in id_list:
    print(l)
    ncng = Dataset(url + l)
    time_n = ncng.variables['time']
    time_ng = netCDF4.num2date(time_n[:],time_n.units)
    oktime = np.logical_and(time_ng >= datetime.datetime(2018,9,11,0,0,0),time_ng <= datetime.datetime(2018,9,12,0,0,0))
    if np.sum(np.where(oktime)) != 0:
        lat_ng = ncng.variables['latitude'][:]
        lon_ng = ncng.variables['longitude'][:]
        print(np.mean(lon_ng[oktime]))
        #print(np.mean(lat_ng[oktime]))
        ax.plot(np.mean(lon_ng[oktime]),np.mean(lat_ng[oktime]),'o',markersize = 10,\
                markeredgecolor='black', markeredgewidth=2,label = ncng.id.split('-')[0]) 
        ax.legend(loc='upper left',fontsize = siz)
        
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/temp_increment_during_Florence_surface_Carib.png"\
            ,bbox_inches = 'tight',pad_inches = 0)        
plt.show()     

#%% increment Caribbean at 80 m surface Florence
    
siz=12

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
cs = ax.contourf(lon_ncoda-360,lat_ncoda,temp_incr[0,7,:,:],20,cmap=plt.get_cmap('seismic'), vmin=-4.5, vmax=4.5)
cbar = plt.colorbar(cs)
cbar.set_label('Temperature Increments ($^o$C)',rotation=90,size = 18)
#plt.colorbar()
#ax.contour(bath_lon[oklonbath],bath_lat[oklatbath],bath_elev[np.c_[oklatbath],oklonbath],colors='k')   
plt.axis('equal')
plt.axis([-70,-62,16,22])
ax.plot(lonFl[1:-2],latFl[1:-2],'o-',markersize = 5,color = 'dimgray') #label = 'Florence Track',
plt.title('Temperature Increments at 80 m on 2018-09-11',size = 18)
ax.set_facecolor('lightgrey')


for l in id_list:
    print(l)
    ncng = Dataset(url + l)
    time_n = ncng.variables['time']
    time_ng = netCDF4.num2date(time_n[:],time_n.units)
    oktime = np.logical_and(time_ng >= datetime.datetime(2018,9,11,0,0,0),time_ng <= datetime.datetime(2018,9,12,0,0,0))
    if np.sum(np.where(oktime)) != 0:
        lat_ng = ncng.variables['latitude'][:]
        lon_ng = ncng.variables['longitude'][:]
        print(np.mean(lon_ng[oktime]))
        #print(np.mean(lat_ng[oktime]))
        ax.plot(np.mean(lon_ng[oktime]),np.mean(lat_ng[oktime]),'o',markersize = 10,\
                markeredgecolor='black', markeredgewidth=2,label = ncng.id.split('-')[0]) 
        ax.legend(loc='upper left',fontsize = siz)
        
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/temp_increment_during_Florence_80m_Carib.png"\
            ,bbox_inches = 'tight',pad_inches = 0)        
plt.show()   

#%% increment global Surface Florence
    
siz=12

# Get rid off very high values
tincr = temp_incr[0,0,:,:]
tincr[tincr < -4.0] = np.nan 
tincr[tincr > 4.0] = np.nan 

fig = plt.figure(1, figsize=(13,8))
ax = plt.subplot(projection=ccrs.PlateCarree())

#ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
#ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
cs = ax.contourf(lon_ncoda,lat_ncoda,tincr,20,cmap=plt.get_cmap('seismic'), vmin=-4.5, vmax=4.5)
cbar = plt.colorbar(cs,fraction=0.046, pad=0.04)
cbar.set_label('($^o$C)',rotation=270,size = 18,labelpad = 20)

plt.title('{0} {1}'.format('Temperature Increments at Surface on',time_ncoda[0]))
#ax.set_facecolor('lightgrey')
#plt.axis([-360,0,-90,90])

   
# Draw coastlines
ax.coastlines(resolution='50m', color='black', linewidth=1)
ax.add_feature(cartopy.feature.LAND, zorder=0)
ax.add_feature(cartopy.feature.OCEAN, zorder=0, color='lightblue', alpha=0.4)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.2, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 14}
gl.ylabel_style = {'size': 14}

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + '{0}_{1}_{2}_{3}.png'.format('temp_increment_',\
          time_ncoda[0].year,time_ncoda[0].month,time_ncoda[0].day) 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 


#%% increment North Atlantic Surface Florence

# Get rid off very high values
z=0 #depth level
tincr = temp_incr[0,z,:,:]
tincr[tincr < -4.1] = np.nan 
tincr[tincr > 4.1] = np.nan 

fig = plt.figure(1, figsize=(13,8))
ax = plt.subplot()

ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)

cs = ax.contourf(lon_ncoda-360,lat_ncoda,tincr,\
                 levels=np.linspace(-3,3,13),cmap=plt.get_cmap('seismic'),\
                 vmin=-4.0,vmax=4.0)
cbar = plt.colorbar(cs)
cbar.set_label('($^o$C)',rotation=270,size = 18,labelpad = 20)
cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),fontsize=20)

plt.xlim(-100,-10)
plt.ylim(0,50)
plt.grid('on')

plt.title('{0} {1} {2} {3} \n {4} {5} {6} {7}'.format('Temperature Increments at ',np.round(depth_ncoda[z]),' m on',time_ncoda[0],\
          'Observations from ',tini ,'to ', tend),fontsize=20)

for l in argo_files:
    ncargo = Dataset(l)
    argo_lat = ncargo.variables['LATITUDE'][:][0]
    argo_lon = ncargo.variables['LONGITUDE'][:][0]
    ax.plot(argo_lon,argo_lat,'ok',markersize = 5)
    
ax.plot(argo_lon,argo_lat,'ok',markersize = 5,label='Argo Floats')    
ax.legend(loc='best',fontsize = 20) 

for id in gliders:
    if id[0:3] != 'all':
        #print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        index_col='time',
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        oktime = np.logical_and(df.index >= tini,df.index <= tend)
        if np.sum(np.where(oktime)) != 0:
            ax.plot(np.mean(df['longitude'][oktime]),np.mean(df['latitude'][oktime]),'o',\
            markersize = 10, markeredgecolor = 'black',markeredgewidth=2,\
            markerfacecolor = 'none')  

'''    
for l in id_list:
    ncng = Dataset(url + l)
    time_n = ncng.variables['time']
    time_ng = netCDF4.num2date(time_n[:],time_n.units)
    oktime = np.logical_and(time_ng >= datetime.datetime(2018,9,11,0,0,0),time_ng <= datetime.datetime(2018,9,12,0,0,0))
    if np.sum(np.where(oktime)) != 0:
        lat_ng = ncng.variables['latitude'][:]
        lon_ng = ncng.variables['longitude'][:]
        ax.plot(np.mean(lon_ng[oktime]),np.mean(lat_ng[oktime]),'o',markersize = 10, \
                markeredgecolor = 'black',markeredgewidth=2,markerfacecolor = 'none')
ax.plot(np.mean(lon_ng[oktime]),np.mean(lat_ng[oktime]),'o',markersize = 10, \
                markeredgecolor = 'black',markeredgewidth=2,markerfacecolor = 'none',\
                label = 'Gliders')
ax.legend(loc='best',fontsize = 12)  
'''
        
ax.plot(np.mean(np.mean(df['longitude'])),np.mean(np.mean(df['latitude'])),'o',\
        markersize = 10, markeredgecolor = 'black',markeredgewidth=2,\
        markerfacecolor = 'none',label='Gliders') 
ax.legend(loc='best',fontsize = 20)

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + '{0}_{1}_{2}_{3}_{4}.png'.format('temp_increment_Atlant',np.round(depth_ncoda[z]),\
          time_ncoda[0].year,time_ncoda[0].month,time_ncoda[0].day) 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  
 

#%% increment North Atlantic 60 m Florence

# Get rid off very high values
z=6 #depth level
tincr = temp_incr[0,z,:,:]
tincr[tincr < -4.1] = np.nan 
tincr[tincr > 4.1] = np.nan 

fig = plt.figure(1, figsize=(13,8))
ax = plt.subplot()

ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)

cs = ax.contourf(lon_ncoda-360,lat_ncoda,tincr,\
                 levels=np.linspace(-3,3,13),cmap=plt.get_cmap('seismic'),\
                 vmin=-4.0,vmax=4.0)
cbar = plt.colorbar(cs)
cbar.set_label('($^o$C)',rotation=270,size = 20,labelpad = 20)
cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),fontsize=20)

plt.xlim(-100,-10)
plt.ylim(0,50)
plt.grid('on')

plt.title('{0} {1} {2} {3} \n {4} {5} {6} {7}'.format('Temperature Increments at ',np.round(depth_ncoda[z]),' m on',time_ncoda[0],\
          'Observations from ',tini ,'to ', tend),fontsize=20)

for l in argo_files:
    ncargo = Dataset(l)
    argo_lat = ncargo.variables['LATITUDE'][:][0]
    argo_lon = ncargo.variables['LONGITUDE'][:][0]
    ax.plot(argo_lon,argo_lat,'ok',markersize = 5)
    
ax.plot(argo_lon,argo_lat,'ok',markersize = 5,label='Argo Floats')    
ax.legend(loc='best',fontsize = 20) 

for id in gliders:
    if id[0:3] != 'all':
        #print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        index_col='time',
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        oktime = np.logical_and(df.index >= tini,df.index <= tend)
        if np.sum(np.where(oktime)) != 0:
            ax.plot(np.mean(df['longitude'][oktime]),np.mean(df['latitude'][oktime]),'o',\
            markersize = 10, markeredgecolor = 'black',markeredgewidth=2,\
            markerfacecolor = 'none') 
            
ax.plot(np.mean(df['longitude'][oktime]),np.mean(df['latitude'][oktime]),'o',\
        markersize = 10, markeredgecolor = 'black',markeredgewidth=2,\
        markerfacecolor = 'none',label='Gliders') 
ax.legend(loc='best',fontsize = 20)

'''       
l=nc_list[5]
print(l)
ncjason = xr.open_dataset(jason_url + l, decode_times=False) 
ncjason = xr.open_dataset(jason_url + l, decode_times=False) 
ssha = ncjason.ssha
lat_jason = ncjason.lat
lon_jason = ncjason.lon
kw = dict(c=ssha*0, marker='.',cmap=plt.get_cmap('gray'),s=1)
ax.scatter(lon_jason-360, lat_jason, **kw)    
'''

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
    if l.split('_')[4][6:]=='11':
        kw = dict(c=sshasub[oklat]*0, marker='.',cmap=plt.get_cmap('gray'),s=1)
    else:
        kw = dict(c=sshasub[oklat]*0, marker='.',cmap=plt.get_cmap('Blues'),s=1)
    ax.scatter(lonsub[oklat]-360, latsub[oklat], **kw) 
    
#ax.scatter(lonsub[oklat]-360, latsub[oklat], **kw,label='Jason2') 
#ax.legend(loc='best',fontsize = 12)    

plt.show()

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + '{0}_{1}_{2}_{3}_{4}.png'.format('temp_increment_Atlant',np.round(depth_ncoda[z]),\
          time_ncoda[0].year,time_ncoda[0].month,time_ncoda[0].day) 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  


#%%
'''
fig = plt.figure(1, figsize=(13,8))
ax = plt.subplot(projection=ccrs.PlateCarree())

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
    kw = dict(c=sshasub[oklat], marker='.',cmap=plt.get_cmap('gray'),s=1)
    ax.scatter(lonsub[oklat]-360, latsub[oklat], **kw)
'''     
    
fig = plt.figure(1, figsize=(8.12,  5.44))
ax = plt.subplot()
ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
l=nc_list[4]
print(l)
ncjason = xr.open_dataset(jason_url + l, decode_times=False) 
ssha = ncjason.ssha
lat_jason = ncjason.lat
lon_jason = ncjason.lon
kw = dict(c=ssha*0, marker='.',cmap=plt.get_cmap('Blues'),s=1)
ax.scatter(lon_jason-360, lat_jason, **kw,label='Jason2')  
ax.legend(loc='best',fontsize = 12) 

folder = '/Users/aristizabal/Desktop/'
#plt.savefig(folder+'mmm',bbox_inches = 'tight',pad_inches = 0.1) 

#%% increment North Atlantic Surface Michael
    
# Get rid off very high values
z=0 #depth level
tincr = temp_incr[0,z,:,:]
tincr[tincr < -4.1] = np.nan 
tincr[tincr > 4.1] = np.nan 

fig = plt.figure(1, figsize=(13,8))
ax = plt.subplot()

ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)

cs = ax.contourf(lon_ncoda-360,lat_ncoda,tincr,\
                 levels=np.linspace(-3,3,13),cmap=plt.get_cmap('seismic'),\
                 vmin=-4.0,vmax=4.0)
cbar = plt.colorbar(cs)
cbar.set_label('($^o$C)',rotation=270,size = 18,labelpad = 20)
cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),fontsize=20)

plt.xlim(-100,-10)
plt.ylim(0,50)
plt.grid('on')

plt.title('{0} {1} {2} {3} \n {4} {5} {6} {7}'.format('Temperature Increments at ',np.round(depth_ncoda[z]),' m on',time_ncoda[0],\
          'Observations from ',tini ,'to ', tend),fontsize=20)

for l in argo_files:
    ncargo = Dataset(l)
    argo_lat = ncargo.variables['LATITUDE'][:][0]
    argo_lon = ncargo.variables['LONGITUDE'][:][0]
    ax.plot(argo_lon,argo_lat,'ok',markersize = 5)
    
ax.plot(argo_lon,argo_lat,'ok',markersize = 5,label='Argo Floats')    
ax.legend(loc='best',fontsize = 20) 

for id in gliders:
    if id[0:3] != 'all':
        #print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        index_col='time',
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        oktime = np.logical_and(df.index >= tini,df.index <= tend)
        if np.sum(np.where(oktime)) != 0:
            ax.plot(np.mean(df['longitude'][oktime]),np.mean(df['latitude'][oktime]),'o',\
            markersize = 10, markeredgecolor = 'black',markeredgewidth=2,\
            markerfacecolor = 'none')

'''    
for l in id_list:
    ncng = Dataset(url + l)
    time_n = ncng.variables['time']
    time_ng = netCDF4.num2date(time_n[:],time_n.units)
    oktime = np.logical_and(time_ng >= datetime.datetime(2018,9,11,0,0,0),time_ng <= datetime.datetime(2018,9,12,0,0,0))
    if np.sum(np.where(oktime)) != 0:
        lat_ng = ncng.variables['latitude'][:]
        lon_ng = ncng.variables['longitude'][:]
        ax.plot(np.mean(lon_ng[oktime]),np.mean(lat_ng[oktime]),'o',markersize = 10, \
                markeredgecolor = 'black',markeredgewidth=2,markerfacecolor = 'none')
ax.plot(np.mean(lon_ng[oktime]),np.mean(lat_ng[oktime]),'o',markersize = 10, \
                markeredgecolor = 'black',markeredgewidth=2,markerfacecolor = 'none',\
                label = 'Gliders')
ax.legend(loc='best',fontsize = 12)  
'''
        
ax.plot(np.mean(np.mean(df['longitude'])),np.mean(np.mean(df['latitude'])),'o',\
        markersize = 10, markeredgecolor = 'black',markeredgewidth=2,\
        markerfacecolor = 'none',label='Gliders') 
ax.legend(loc='best',fontsize = 20)

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + '{0}_{1}_{2}_{3}_{4}.png'.format('temp_increment_Atlant',np.round(depth_ncoda[z]),\
          time_ncoda[0].year,time_ncoda[0].month,time_ncoda[0].day) 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  
     

#%% increment North Atlantic at 60 m Michael
    
# Get rid off very high values
z=6 #depth level
tincr = temp_incr[0,z,:,:]
tincr[tincr < -4.1] = np.nan 
tincr[tincr > 4.1] = np.nan 

fig = plt.figure(1, figsize=(13,8))
ax = plt.subplot()

ax.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
ax.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[-10000,0],colors='lightskyblue',alpha=0.5)

cs = ax.contourf(lon_ncoda-360,lat_ncoda,tincr,\
                 levels=np.linspace(-3,3,13),cmap=plt.get_cmap('seismic'),\
                 vmin=-4.0,vmax=4.0)
cbar = plt.colorbar(cs)
cbar.set_label('($^o$C)',rotation=270,size = 20,labelpad = 20)
cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),fontsize=20)

plt.xlim(-100,-10)
plt.ylim(0,50)
plt.grid('on')

plt.title('{0} {1} {2} {3} \n {4} {5} {6} {7}'.format('Temperature Increments at ',np.round(depth_ncoda[z]),' m on',time_ncoda[0],\
          'Observations from ',tini ,'to ', tend),fontsize=20)

for l in argo_files:
    ncargo = Dataset(l)
    argo_lat = ncargo.variables['LATITUDE'][:][0]
    argo_lon = ncargo.variables['LONGITUDE'][:][0]
    ax.plot(argo_lon,argo_lat,'ok',markersize = 5)
    
ax.plot(argo_lon,argo_lat,'ok',markersize = 5,label='Argo Floats')    
ax.legend(loc='best',fontsize = 20) 

for id in gliders:
    if id[0:3] != 'all':
        #print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        index_col='time',
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        oktime = np.logical_and(df.index >= tini,df.index <= tend)
        if np.sum(np.where(oktime)) != 0:
            ax.plot(np.mean(df['longitude'][oktime]),np.mean(df['latitude'][oktime]),'o',\
            markersize = 10, markeredgecolor = 'black',markeredgewidth=2,\
            markerfacecolor = 'none') 
            
ax.plot(np.mean(df['longitude'][oktime]),np.mean(df['latitude'][oktime]),'o',\
        markersize = 10, markeredgecolor = 'black',markeredgewidth=2,\
        markerfacecolor = 'none',label='Gliders') 
ax.legend(loc='best',fontsize = 20)

'''       
l=nc_list[5]
print(l)
ncjason = xr.open_dataset(jason_url + l, decode_times=False) 
ncjason = xr.open_dataset(jason_url + l, decode_times=False) 
ssha = ncjason.ssha
lat_jason = ncjason.lat
lon_jason = ncjason.lon
kw = dict(c=ssha*0, marker='.',cmap=plt.get_cmap('gray'),s=1)
ax.scatter(lon_jason-360, lat_jason, **kw)    
'''

for l in nc_list[14:]:
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
    
#ax.scatter(lonsub[oklat]-360, latsub[oklat], **kw,label='Jason2') 
#ax.legend(loc='best',fontsize = 12)    

plt.show()

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + '{0}_{1}_{2}_{3}_{4}.png'.format('temp_increment_Atlant',np.round(depth_ncoda[z]),\
          time_ncoda[0].year,time_ncoda[0].month,time_ncoda[0].day) 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  
 

#%%

l=id_list[10]
ncng = Dataset(url + l)
time_n = ncng.variables['time']
time_ng = netCDF4.num2date(time_n[:],time_n.units)
oktime = np.logical_and(time_ng >= datetime.datetime(2018,10,10,0,0,0),time_ng <= datetime.datetime(2018,10,11,0,0,0))
if np.sum(np.where(oktime)) != 0:
    lat_ng = ncng.variables['latitude'][:]
    lon_ng = ncng.variables['longitude'][:]
    print(np.mean(lon_ng[oktime]),np.mean(lat_ng[oktime]))
        

