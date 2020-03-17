#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 09:51:16 2018

@author: aristizabal
"""

#%% User input

# lat and lon bounds
lon_lim = [-100.0,-10.0]
lat_lim = [0.0,60.0]

# Time bounds
min_time = '2018-06-01T00:00:00Z'
max_time = '2018-11-30T00:00:00Z'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

# Server url 
server = 'https://data.ioos.us/gliders/erddap'

# Jun1 -Jul1
Dir_Argo1 = '/Volumes/aristizabal/ARGO_data/Hurric_season_2018/DataSelection_20190501_195036_7920913'

# Jul1 - Aug1
Dir_Argo2 = '/Volumes/aristizabal/ARGO_data/Hurric_season_2018/DataSelection_20190508_135838_7928110'

# Aug1 - Sep1
Dir_Argo3 = '/Volumes/aristizabal/ARGO_data/Hurric_season_2018/DataSelection_20190508_141321_7928283'

# Sep1 - Oct1
Dir_Argo4 = '/Volumes/aristizabal/ARGO_data/Hurric_season_2018/DataSelection_20190508_190253_7934529'

# Oct1 - Nov1
Dir_Argo5 = '/Volumes/aristizabal/ARGO_data/Hurric_season_2018/DataSelection_20190508_143800_7929144'

# Nov1 - Nov30
Dir_Argo6 = '/Volumes/aristizabal/ARGO_data/Hurric_season_2018/DataSelection_20190508_190433_7934559'

# storm track files
track_folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/KMZ_files/'
basin = 'al'
year = '2018'
fname = '_best_track.kmz' 

#%%

from erddapy import ERDDAP
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import netCDF4 
from datetime import datetime
import glob
import os
from netCDF4 import Dataset
import cmocean
from bs4 import BeautifulSoup
from zipfile import ZipFile
#from matplotlib.dates import date2num

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
    parse_dates=True,
    skiprows=(1,)  # units information can be dropped.
).dropna()
    
    print(id,df.index[-1])


#%% Reading bathymetry data

ncbath = xr.open_dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath] 

#%% Reading Argo data

argo_files1 = sorted(glob.glob(os.path.join(Dir_Argo1,'*profiles*.nc')))
argo_files2 = sorted(glob.glob(os.path.join(Dir_Argo2,'*profiles*.nc')))
argo_files3 = sorted(glob.glob(os.path.join(Dir_Argo3,'*profiles*.nc')))
argo_files4 = sorted(glob.glob(os.path.join(Dir_Argo4,'*profiles*.nc')))
argo_files5 = sorted(glob.glob(os.path.join(Dir_Argo5,'*profiles*.nc')))
argo_files6 = sorted(glob.glob(os.path.join(Dir_Argo6,'*profiles*.nc')))

ncargo = Dataset(argo_files1[-1])
argo_id = ncargo.variables['PLATFORM_NUMBER'][:]
argo_lat = ncargo.variables['LATITUDE'][:]
argo_lon = ncargo.variables['LONGITUDE'][:]

argo_tim = ncargo.variables['JULD']#[:]
argo_time = netCDF4.num2date(argo_tim[:],argo_tim.units) 

#%% Plotting Argo time

plt.figure()
for l in argo_files1:
    ncargo = Dataset(l)
    argo_lat = np.asarray(ncargo.variables['LATITUDE'][:])
    argo_lon = np.asarray(ncargo.variables['LONGITUDE'][:])
    argo_tim = ncargo.variables['JULD']
    argo_time = np.asarray(netCDF4.num2date(argo_tim[:],argo_tim.units))
    plt.plot(argo_time,'*')
    
#%% Finding number of Argo profiles

argo_files = argo_files1 + argo_files2 + argo_files3 + \
             argo_files4 + argo_files5 + argo_files6

Argo_time = np.empty((0))
#Argo_lon = np.empty((0))
#Argo_lat = np.empty((0))
#Argo_temp = np.empty((0))

for i,l in enumerate(argo_files):
    print(len(argo_files),' ', i)
    ncargo = Dataset(l)
    #argo_lat = np.asarray(ncargo.variables['LATITUDE'][:])
    #argo_lon = np.asarray(ncargo.variables['LONGITUDE'][:])
    #argo_temp = np.asarray(ncargo.variables['TEMP'][:,0])
    argo_tim = ncargo.variables['JULD']    
    argo_time = np.asarray(netCDF4.num2date(argo_tim[:],argo_tim.units))
    Argo_time = np.concatenate((Argo_time,argo_time))
    #Argo_lon = np.concatenate((Argo_lon,argo_lon))
    #Argo_lat = np.concatenate((Argo_lat,argo_lat))
    #Argo_temp = np.concatenate((Argo_temp,argo_temp))

Number_argo_profiles = Argo_time.shape

#%% Reading and plotting best track data

track_files = sorted(glob.glob(os.path.join(track_folder+year+'/','*'+basin+'*kmz')))

lev = np.arange(-9000,9100,100)
fig, ax = plt.subplots(figsize=(10, 10))
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo) 
plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell') 
plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')  
plt.yticks([])
plt.xticks([])
plt.title('Glider Tracks and Storm Tracks \n Hurricane Season 2018',fontsize=30)

#%%
for n,f in enumerate(track_files):
    print(f)
    os.system('cp ' + f + ' ' + f[:-3] + 'zip')
    os.system('unzip -o ' + f + ' -d ' + track_files[0][:-4])
    zip_handle = ZipFile(f[:-3]+'zip', 'r')
    kml_file = f.split('/')[-1].split('_')[0] + '.kml'
    kml_best_track = zip_handle.open(kml_file, 'r').read()
    
    # best track coordinates
    soup = BeautifulSoup(kml_best_track,'html.parser')
    
    lon_best_track = np.empty(len(soup.find_all("point")))
    lon_best_track[:] = np.nan
    lat_best_track = np.empty(len(soup.find_all("point")))
    lat_best_track[:] = np.nan
    for i,s in enumerate(soup.find_all("point")):
        #print(s.get_text("coordinates"))
        if len(s.get_text("coordinates").split('coordinates')) != 1:
            lon_best_track[i] = float(s.get_text("coordinates").split('coordinates')[1].split(',')[0])
            lat_best_track[i] = float(s.get_text("coordinates").split('coordinates')[1].split(',')[1])
        
        stormname = soup.find_all("stormname")[-1].get_text('stormname')
    '''  
    cat = []
    for i,s in enumerate(soup.find_all("styleurl")):
        cat.append(s.get_text('#').split('#')[-1]) 
    cat = np.asarray(cat)      
    '''

    plt.text(lon_best_track[0],lat_best_track[0],str(n+1),color='k',\
             bbox=dict(facecolor='white', edgecolor='none',alpha=0.4))
    if np.logical_or(n==1-1,n==6-1):  
        plt.text(lon_best_track[-1],lat_best_track[-1],str(n+1),color='k',\
             bbox=dict(facecolor='white', edgecolor='none',alpha=0.4))
    if np.logical_or(n==7-1,n==14-1):  
        plt.text(lon_best_track[-1],lat_best_track[-1],str(n+1),color='k',\
             bbox=dict(facecolor='white', edgecolor='none',alpha=0.4))
    plt.plot(lon_best_track,lat_best_track,'.-k',label=str(n+1) + ' ' + stormname)

   
for id in gliders:
    if id[0:4] != 'glos':
        #print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        ax.plot(df['longitude (degrees_east)'],\
                df['latitude (degrees_north)'],'.',color='darkorange',markersize=1)   
  
plt.legend(loc='upper left',bbox_to_anchor=[-0.22,1.0])     
plt.axis('scaled') 
#plt.axis([-100,-50,10,50])
plt.axis([-100,-10,10,60])
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/map_gliders_hurric_track_season_2018.png",\
            bbox_inches = 'tight',pad_inches = 0.1)
    
#%% Best Track Michael

lonMc = np.array([-86.9,-86.7,-86.0,\
                  -85.3,-85.4,-85.07,-85.05,\
                  -85.2,-85.7,-86.1,-86.3,\
                  -86.5,-86.6,-86.3,-85.4,\
                  -84.5,-83.2,-81.7,-80.0])

latMc = np.array([18.4,18.7,19.0,\
                  19.7,20.2,20.9,21.7,\
                  22.7,23.6,24.6,25.6,\
                  26.6,27.8,29.0,30.2,\
                  31.5,32.8,34.1,35.6])

tMc = [                   '2018/10/07/06/00','2018/10/07/12/00','2018/10/07/18/00',\
       '2018/10/08/00/00','2018/10/08/06/00','2018/10/08/12/00','2018/10/08/18/00',\
       '2018/10/09/00/00','2018/10/09/06/00','2018/10/09/12/00','2018/10/09/18/00',\
       '2018/10/10/00/00','2018/10/10/06/00','2018/10/10/12/00','2018/10/10/18/00',
       '2018/10/11/00/00','2018/10/11/06/00','2018/10/11/12/00','2018/10/11/18/00']

timeMc = [None]*len(tMc) 
for x in range(len(tMc)):
    timeMc[x] = datetime.strptime(tMc[x], '%Y/%m/%d/%H/%M') # time in time zone  

#%% Map with ARGO durimg hurricane season 2018
    
fig, ax = plt.subplots(figsize=(10, 5))
plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
plt.contourf(bath_lon,bath_lat,bath_elev,cmap='Blues_r')
plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell')
plt.yticks([])
plt.xticks([])
plt.axis([-100,-10,0,50])

for l in argo_files1:
    ncargo = Dataset(l)
    argo_lat = np.asarray(ncargo.variables['LATITUDE'][:])
    argo_lon = np.asarray(ncargo.variables['LONGITUDE'][:])
    ax.plot(argo_lon,argo_lat,'ok-',markersize = 4,markeredgecolor='g')
    
for l in argo_files2:
    ncargo = Dataset(l)
    argo_lat = np.asarray(ncargo.variables['LATITUDE'][:])
    argo_lon = np.asarray(ncargo.variables['LONGITUDE'][:])
    ax.plot(argo_lon,argo_lat,'ok-',markersize = 4,markeredgecolor='g')    
    
for l in argo_files3:
    ncargo = Dataset(l)
    argo_lat = np.asarray(ncargo.variables['LATITUDE'][:])
    argo_lon = np.asarray(ncargo.variables['LONGITUDE'][:])
    ax.plot(argo_lon,argo_lat,'ok-',markersize = 4,markeredgecolor='g')
    
for l in argo_files4:
    ncargo = Dataset(l)
    argo_lat = np.asarray(ncargo.variables['LATITUDE'][:])
    argo_lon = np.asarray(ncargo.variables['LONGITUDE'][:])
    ax.plot(argo_lon,argo_lat,'ok-',markersize = 4,markeredgecolor='g')

for l in argo_files5:
    ncargo = Dataset(l)
    argo_lat = ncargo.variables['LATITUDE'][:]
    argo_lon = ncargo.variables['LONGITUDE'][:]
    ax.plot(argo_lon,argo_lat,'ok-',markersize = 4,markeredgecolor='g')

for l in argo_files6:
    ncargo = Dataset(l)
    argo_lat = np.asarray(ncargo.variables['LATITUDE'][:])
    argo_lon = np.asarray(ncargo.variables['LONGITUDE'][:])
    ax.plot(argo_lon,argo_lat,'ok-',markersize = 4,markeredgecolor='g')        

plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/map_ARGOS_hurric_season_2018_v2.png",\
            bbox_inches = 'tight',pad_inches = 0.1)


#%% Map with ARGO + glider during hurricane season 2018

lev = np.arange(-9000,9100,100)
fig, ax = plt.subplots(figsize=(10, 5))
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo)    
plt.yticks([])
plt.xticks([])
plt.axis([-100,-10,0,50])    
    
#fig, ax = plt.subplots(figsize=(10, 5))
#plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
#plt.contourf(bath_lon,bath_lat,bath_elev,cmap='Blues_r')
#plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell')
#plt.yticks([])
#plt.xticks([])


for l in argo_files1:
    ncargo = Dataset(l)
    argo_lat = np.asarray(ncargo.variables['LATITUDE'][:])
    argo_lon = np.asarray(ncargo.variables['LONGITUDE'][:])
    ax.plot(argo_lon,argo_lat,'ok-',markersize = 4,markeredgecolor='g')
    
for l in argo_files2:
    ncargo = Dataset(l)
    argo_lat = np.asarray(ncargo.variables['LATITUDE'][:])
    argo_lon = np.asarray(ncargo.variables['LONGITUDE'][:])
    ax.plot(argo_lon,argo_lat,'ok-',markersize = 4,markeredgecolor='g')    
    
for l in argo_files3:
    ncargo = Dataset(l)
    argo_lat = np.asarray(ncargo.variables['LATITUDE'][:])
    argo_lon = np.asarray(ncargo.variables['LONGITUDE'][:])
    ax.plot(argo_lon,argo_lat,'ok-',markersize = 4,markeredgecolor='g')
    
for l in argo_files4:
    ncargo = Dataset(l)
    argo_lat = np.asarray(ncargo.variables['LATITUDE'][:])
    argo_lon = np.asarray(ncargo.variables['LONGITUDE'][:])
    ax.plot(argo_lon,argo_lat,'ok-',markersize = 4,markeredgecolor='g')

for l in argo_files5:
    ncargo = Dataset(l)
    argo_lat = ncargo.variables['LATITUDE'][:]
    argo_lon = ncargo.variables['LONGITUDE'][:]
    ax.plot(argo_lon,argo_lat,'ok-',markersize = 4,markeredgecolor='g')

for l in argo_files6:
    ncargo = Dataset(l)
    argo_lat = np.asarray(ncargo.variables['LATITUDE'][:])
    argo_lon = np.asarray(ncargo.variables['LONGITUDE'][:])
    ax.plot(argo_lon,argo_lat,'ok-',markersize = 4,markeredgecolor='g')  

for id in gliders:
    if id[0:4] != 'glos':
        #print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        ax.plot(df['longitude (degrees_east)'],\
                df['latitude (degrees_north)'],'.',color='r',markersize=1)   
        #ax.text(np.mean(df['longitude']),np.mean(df['latitude']),id.split('-')[0])
    
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/map_ARGOS_gliders_hurric_season_2018_v2.png",\
            bbox_inches = 'tight',pad_inches = 0.1)

#%% Map all gliders durimg hurricane season 2018

lev = np.arange(-9000,9100,100)
fig, ax = plt.subplots(figsize=(10, 10))
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo)    
plt.yticks([])
plt.xticks([])
plt.title('Glider Tracks Hurricane Season 2018',fontsize=30)

for id in gliders:
    if id[0:4] != 'glos':
        #print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        ax.plot(df['longitude (degrees_east)'],\
                df['latitude (degrees_north)'],'.',color='darkorange',markersize=1)   
        #ax.text(np.mean(df['longitude']),np.mean(df['latitude']),id.split('-')[0])

plt.axis('scaled') 
plt.axis([-100,-50,10,50])    
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/map_gliders_hurric_season_2018.png",\
            bbox_inches = 'tight',pad_inches = 0.1)
    
    
#%% Reading glider data and plotting lat and lon on the map

fig, ax = plt.subplots(figsize=(10, 5))
plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
plt.contourf(bath_lon,bath_lat,bath_elev,cmap='Blues_r')
plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell')
plt.yticks([])
plt.xticks([])
plt.axis([-100,-10,0,50])
'''  
siz=12

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
'''
#ax.contour(bath_lon[oklonbath],bath_lat[oklatbath],bath_elev[np.c_[oklatbath],oklonbath],colors='k')   
plt.title('Glider Tracks During Hurricane Season 2018')

for id in gliders:
    if id[0:3] != 'all':
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        ax.plot(df['longitude (degrees_east)'],\
                df['latitude (degrees_north)'],'.r', markersize = 1)   
        #ax.legend(loc='upper left',fontsize = 16)
        #ax.text(np.mean(df['longitude (degrees_east)']),\
        #        np.mean(df['latitude (degrees_north)']),)
#plt.axis('equal')
#plt.axis([-94,-82,25,28.5])            
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/map_gliders_hurric_season_2018_Gulf.png")
plt.show() 
 
#%% Reading glider data and plotting lat and lon on the map
    
siz=12

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
#ax.contour(bath_lon[oklonbath],bath_lat[oklatbath],bath_elev[np.c_[oklatbath],oklonbath],colors='k')   
plt.title('Gliders During Hurricane Season 2018')

for id in gliders:
    if id[0:3] != 'all':
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        ax.plot(np.mean(np.mean(df['longitude'])),np.mean(np.mean(df['latitude'])),'*', markersize = 10)   
        #ax.legend(loc='upper left',fontsize = 16)
        ax.text(np.mean(df['longitude']),np.mean(df['latitude']),id.split('-')[0])
        
plt.axis('equal')
plt.axis([-69,-62,16,22])            
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/map_gliders_hurric_season_2018_Caribbean.png")
plt.show() 

#%% Reading glider data and plotting lat and lon on the map
    
siz=12

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
#ax.contour(bath_lon[oklonbath],bath_lat[oklatbath],bath_elev[np.c_[oklatbath],oklonbath],colors='k')   
plt.title('Gliders During Hurricane Season 2018')

for id in gliders:
    if id[0:3] != 'all':
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        ax.plot(np.mean(np.mean(df['longitude'])),np.mean(np.mean(df['latitude'])),'*', markersize = 10)   
        #ax.legend(loc='upper left',fontsize = 16)
        ax.text(np.mean(df['longitude']),np.mean(df['latitude']),id.split('-')[0])
        
plt.axis('equal')
plt.axis([-80,-75,28,36])            
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/map_gliders_hurric_season_2018_SAB.png")
plt.show() 

#%% Reading glider data and plotting lat and lon on the map
    
siz=12

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
#ax.contour(bath_lon[oklonbath],bath_lat[oklatbath],bath_elev[np.c_[oklatbath],oklonbath],colors='k')   
plt.title('Gliders During Hurricane Season 2018')

for id in gliders:
    if id[0:3] != 'all':
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        ax.plot(np.mean(np.mean(df['longitude'])),np.mean(np.mean(df['latitude'])),'*', markersize = 10)   
        #ax.legend(loc='upper left',fontsize = 16)
        ax.text(np.mean(df['longitude']),np.mean(df['latitude']),id.split('-')[0])
        
plt.axis('equal')
plt.axis([-80,-75,28,35.5])            
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/map_gliders_hurric_season_2018_SAB.png")
plt.show() 

#%% Reading glider data and plotting lat and lon on the map
    
siz=12

fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
#ax.contour(bath_lon[oklonbath],bath_lat[oklatbath],bath_elev[np.c_[oklatbath],oklonbath],colors='k')   
plt.title('Gliders During Hurricane Season 2018')

for id in gliders:
    if id[0:3] != 'all':
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        ax.plot(np.mean(np.mean(df['longitude'])),np.mean(np.mean(df['latitude'])),'*', markersize = 10)   
        #ax.legend(loc='upper left',fontsize = 16)
        ax.text(np.mean(df['longitude']),np.mean(df['latitude']),id.split('-')[0])
        
plt.axis('equal')
plt.axis([-75,-68,34.5,42])            
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/map_gliders_hurric_season_2018_MAB.png")
plt.show() 