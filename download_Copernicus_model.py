#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:47:42 2019

@author: aristizabal
"""

#%% User input

# lat and lon bounds
lon_lim = [-110.0,-10.0]
lat_lim = [15.0,45.0]

# urls 
url_glider = 'https://data.ioos.us/gliders/erddap'

# COPERNICUS MARINE ENVIRONMENT MONITORING SERVICE (CMEMS)
url_cmems = 'http://nrt.cmems-du.eu/motu-web/Motu'
service_id = 'GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS'
product_id = 'global-analysis-forecast-phy-001-024'
depth_min = '0.493' 
out_dir = '/Users/aristizabal/Desktop' 

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

#%%

from erddapy import ERDDAP
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cmocean

from datetime import datetime, timedelta

import numpy as np
import xarray as xr

import os

# Increase fontsize of labels globally 
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Get time bounds for the previous day

te = datetime.today() 
tend = datetime(te.year,te.month,te.day)

#ti = datetime.today() - timedelta(1)
ti = datetime.today() - timedelta(1)
tini = datetime(ti.year,ti.month,ti.day)

#%% Look for datasets in IOOS glider dac
 
print('Looking for glider data sets')
e = ERDDAP(server = url_glider)

# Grab every dataset available
datasets = pd.read_csv(e.get_search_url(response='csv', search_for='all'))

# Search constraints
kw = {
    'min_lon': lon_lim[0],
    'max_lon': lon_lim[1],
    'min_lat': lat_lim[0],
    'max_lat': lat_lim[1],
    'min_time': tini.strftime('%Y-%m-%dT%H:%M:%SZ'),
    'max_time': tend.strftime('%Y-%m-%dT%H:%M:%SZ'),
}

search_url = e.get_search_url(response='csv', **kw)
#print(search_url)

# Grab the results
search = pd.read_csv(search_url)

# Extract the IDs
gliders = search['Dataset ID'].values

msg = 'Found {} Glider Datasets:\n\n{}'.format
print(msg(len(gliders), '\n'.join(gliders)))
    
# Setting constraints
constraints = {
        'time>=': tini,
        'time<=': tend,
        'latitude>=': lat_lim[0],
        'latitude<=': lat_lim[1],
        'longitude>=': lon_lim[0],
        'longitude<=': lon_lim[1],
        }

variables = [
        'depth',
        'latitude',
        'longitude',
        'time',
        'temperature',
        'salinity'
        ]

e = ERDDAP(
        server=url_glider,
        protocol='tabledap',
        response='nc'
        )

#%% Reading bathymetry data

ncbath = xr.open_dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

#%%   

for id in gliders:
    print('Reading ' + id )
    e.dataset_id = id
    e.constraints = constraints
    e.variables = variables
    
    # Converting glider data to data frame
    df = e.to_pandas(
            index_col='time (UTC)',
            parse_dates=True,
            skiprows=(1,)  # units information can be dropped.
            ).dropna()
    
    # Coverting glider vectors into arrays
    timeg, ind = np.unique(df.index.values,return_index=True)
    latg = df['latitude (degrees_north)'].values[ind]
    long = df['longitude (degrees_east)'].values[ind]

    dg = df['depth (m)'].values
    #vg = df['temperature (degree_Celsius)'].values
    tg = df[df.columns[3]].values
    sg = df[df.columns[4]].values
    
    delta_z = 0.3
    zn = np.int(np.round(np.max(dg)/delta_z))

    depthg = np.empty((zn,len(timeg)))
    depthg[:] = np.nan
    tempg = np.empty((zn,len(timeg)))
    tempg[:] = np.nan
    saltg = np.empty((zn,len(timeg)))
    saltg[:] = np.nan
    
    # Grid variables
    depthg_gridded = np.arange(0,np.nanmax(dg),delta_z)
    tempg_gridded = np.empty((len(depthg_gridded),len(timeg)))
    tempg_gridded[:] = np.nan
    saltg_gridded = np.empty((len(depthg_gridded),len(timeg)))
    saltg_gridded[:] = np.nan
    
    for i,ii in enumerate(ind):
        if i < len(timeg)-1:
            depthg[0:len(dg[ind[i]:ind[i+1]]),i] = dg[ind[i]:ind[i+1]] 
            tempg[0:len(tg[ind[i]:ind[i+1]]),i] = tg[ind[i]:ind[i+1]]
            saltg[0:len(sg[ind[i]:ind[i+1]]),i] = sg[ind[i]:ind[i+1]]
        else:
            depthg[0:len(dg[ind[i]:len(dg)]),i] = dg[ind[i]:len(dg)] 
            tempg[0:len(tg[ind[i]:len(tg)]),i] = tg[ind[i]:len(tg)]
            saltg[0:len(sg[ind[i]:len(sg)]),i] = sg[ind[i]:len(sg)]

    for t,tt in enumerate(timeg):
        depthu,oku = np.unique(depthg[:,t],return_index=True)
        tempu = tempg[oku,t]
        saltu = saltg[oku,t]
        okdd = np.isfinite(depthu)
        depthf = depthu[okdd]
        tempf = tempu[okdd]
        saltf = saltu[okdd]
        
        okt = np.isfinite(tempf)
        if np.sum(okt) < 3:
            tempg_gridded[:,t] = np.nan
        else:
            okd = depthg_gridded < np.max(depthf[okt])
            tempg_gridded[okd,t] = np.interp(depthg_gridded[okd],depthf[okt],tempf[okt])
            
        oks = np.isfinite(saltf)
        if np.sum(oks) < 3:
            saltg_gridded[:,t] = np.nan
        else:
            okd = depthg_gridded < np.max(depthf[oks])
            saltg_gridded[okd,t] = np.interp(depthg_gridded[okd],depthf[oks],saltf[oks])
            
    # Downloading and reading Copernicus output
    motuc = 'python -m motuclient --motu ' + url_cmems + \
        ' --service-id ' + service_id + \
        ' --product-id ' + product_id + \
        ' --longitude-min ' + str(np.min(long)-1/12) + \
        ' --longitude-max ' + str(np.max(long)+1/12) + \
        ' --latitude-min ' + str(np.min(latg)-1/12) + \
        ' --latitude-max ' + str(np.max(latg)+1/12) + \
        ' --date-min ' + str(tini-timedelta(0.5)) + \
        ' --date-max ' + str(tend+timedelta(0.5)) + \
        ' --depth-min ' + depth_min + \
        ' --depth-max ' + str(np.nanmax(depthg)) + \
        ' --variable ' + 'thetao' + ' ' + \
        ' --variable ' + 'so'  + ' ' + \
        ' --out-dir ' + out_dir + \
        ' --out-name ' + id + '.nc' + ' ' + \
        ' --user ' + 'maristizabalvar' + ' ' + \
        ' --pwd ' +  'MariaCMEMS2018' 
   
    os.system(motuc)   
       
    COP_file = out_dir + '/' + id + '.nc'
    COP = xr.open_dataset(COP_file)
    
    latCOP = COP.latitude[:]
    lonCOP = COP.longitude[:]
    depthCOP = COP.depth[:]
    tCOP = np.asarray(mdates.num2date(mdates.date2num(COP.time[:])))

    tmin = tini - timedelta(0.5)
    tmax = tend + timedelta(0.5)

    oktimeCOP = np.where(np.logical_and(mdates.date2num(tCOP) >= mdates.date2num(tmin),\
                                        mdates.date2num(tCOP) <= mdates.date2num(tmax)))
    timeCOP = tCOP[oktimeCOP]
    
    # Changing times to timestamp
    tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]
    tstamp_COP = [mdates.date2num(timeCOP[i]) for i in np.arange(len(timeCOP))]
    
    # interpolating glider lon and lat to lat and lon on GOFS 3.1 time
    sublonCOP=np.interp(tstamp_COP,tstamp_glider,long)
    sublatCOP=np.interp(tstamp_COP,tstamp_glider,latg)

    # getting the model grid positions for sublonm and sublatm
    oklonCOP=np.round(np.interp(sublonCOP,lonCOP,np.arange(len(lonCOP)))).astype(int)
    oklatCOP=np.round(np.interp(sublatCOP,latCOP,np.arange(len(latCOP)))).astype(int)

    # Getting glider transect from Copernicus model
    print('Getting glider transect from Copernicus model')
    target_tempCOP = np.empty((len(depthCOP),len(oktimeCOP[0])))
    target_tempCOP[:] = np.nan
    for i in range(len(oktimeCOP[0])):
        print(len(oktimeCOP[0]),' ',i)
        target_tempCOP[:,i] = COP.variables['thetao'][oktimeCOP[0][i],:,oklatCOP[i],oklonCOP[i]]
    target_tempCOP[target_tempCOP < -100] = np.nan
    
    target_saltCOP = np.empty((len(depthCOP),len(oktimeCOP[0])))
    target_saltCOP[:] = np.nan
    for i in range(len(oktimeCOP[0])):
        print(len(oktimeCOP[0]),' ',i)
        target_saltCOP[:,i] = COP.variables['so'][oktimeCOP[0][i],:,oklatCOP[i],oklonCOP[i]]
    target_saltCOP[target_saltCOP < -100] = np.nan
    
    os.system('rm ' + out_dir + '/' + id + '.nc')

    # Along track transect temperature figure 
    grid = plt.GridSpec(3, 5, wspace=0.4, hspace=0.3)
    
    min_temp = np.floor(np.min([np.nanmin(df[df.columns[3]]),np.nanmin(target_tempCOP)]))
    max_temp = np.ceil(np.max([np.nanmax(df[df.columns[3]]),np.nanmax(target_tempCOP)]))
    
    min_salt = np.floor(np.min([np.nanmin(df[df.columns[4]]),np.nanmin(target_saltCOP)]))
    max_salt = np.ceil(np.max([np.nanmax(df[df.columns[4]]),np.nanmax(target_saltCOP)]))

    # Along track transect temperature
    fig, ax = plt.subplots(figsize=(14, 12))
    grid = plt.GridSpec(3, 5, wspace=0.4, hspace=0.3)

    # Scatter plot
    ax = plt.subplot(grid[0, :4])
    kw = dict(s=30, c=df[df.columns[3]], marker='*', edgecolor='none')
    cs = ax.scatter(df.index, -df['depth (m)'], **kw, cmap=cmocean.cm.thermal)
    cs.set_clim(min_temp,max_temp)
    ax.set_xlim(tini,tend)
    #ax.set_xlim(df.index[0], df.index[-1])
    xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
    ax.xaxis.set_major_formatter(xfmt)
    ax.set_xticklabels(' ')        
    cbar = fig.colorbar(cs, orientation='vertical')
    cbar.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
    ax.set_ylabel('Depth (m)',fontsize=14)
    plt.title('Along Track Temperature ' + id)
          
    nlevels = max_temp - min_temp + 1
    kw = dict(levels = np.linspace(min_temp,max_temp,nlevels))
    ax = plt.subplot(grid[1, :4])
    #plt.contour(timeg,-depthg_gridded,varg_gridded,colors = 'lightgrey',**kw)
    cs = plt.contourf(timeg,-depthg_gridded,tempg_gridded,cmap=cmocean.cm.thermal,**kw)
    if np.logical_and(min_temp<=26.0,max_temp>=26.0): 
        plt.contour(timeg,-depthg_gridded,tempg_gridded,levels=[26],colors='k')
    cs = fig.colorbar(cs, orientation='vertical') 
    cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)        
    ax.set_xlim(tini,tend)
    ax.set_ylabel('Depth (m)',fontsize=14)
    ax.set_xticklabels(' ')
    plt.title('Along Track Temperature ' + id)
        
    ax = plt.subplot(grid[2, :4])    
    #plt.contour(timeg,-depthg_gridded,varg_gridded,colors = 'lightgrey',**kw)
    cs = plt.contourf(mdates.date2num(timeCOP),-depthCOP,target_tempCOP,cmap=cmocean.cm.thermal,**kw)
    if np.logical_and(min_temp<=26.0,max_temp>=26.0): 
        plt.contour(mdates.date2num(timeCOP),-depthCOP,target_tempCOP,[26],colors='k')
    cs = fig.colorbar(cs, orientation='vertical') 
    cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
    ax.set_xlim(tini,tend)
    ax.set_ylim(-np.max(df['depth (m)']), 0)
    ax.set_ylabel('Depth (m)',fontsize=14)
    xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
    ax.xaxis.set_major_formatter(xfmt)
    #ax.set_xticklabels(' ')  
    plt.title('Along Track Temperature Copernicus')
    
    oklatbath = np.logical_and(bath_lat >= np.min(latg)-5,bath_lat <= np.max(latg)+5)
    oklonbath = np.logical_and(bath_lon >= np.min(long)-5,bath_lon <= np.max(long)+5)

    bath_latsub = bath_lat[oklatbath]
    bath_lonsub = bath_lon[oklonbath]
    bath_elevs = bath_elev[oklatbath,:]
    bath_elevsub = bath_elevs[:,oklonbath] 
    
    ax = plt.subplot(grid[1, 4:])
    plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
    plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,cmap='Blues_r')
    plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
    #plt.yticks([])
    #plt.xticks([])
    plt.axis([np.min(long)-5,np.max(long)+5,np.min(latg)-5,np.max(latg)+5])
    plt.plot(long,latg,'.k')
    plt.title('Track ' + id)
    #plt.axis('equal')
    
    folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
    file = 'along_track_temp_Copernicus' + id + '_' + str(tini).split()[0] + '_' + str(tend).split()[0]
    plt.savefig(folder+file,bbox_inches = 'tight',pad_inches = 0.1) 
      
#%%        
'''        
os.system('python -m motuclient --motu http://nrt.cmems-du.eu/motu-web/Motu 
          --service-id GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS 
          --product-id global-analysis-forecast-phy-001-024 
          --longitude-min -75 --longitude-max -60 
          --latitude-min 30 --latitude-max 55 
          --date-min "2019-06-02 12:00:00" 
          --date-max "2019-06-04 12:00:00" 
          --depth-min 0.493 --depth-max 541.09 
          --variable thetao --variable so 
          --out-dir /Users/aristizabal/Desktop 
          --out-name copernicus.nc 
          --user "maristizabalvar" --pwd "MariaCMEMS2018"')
'''
