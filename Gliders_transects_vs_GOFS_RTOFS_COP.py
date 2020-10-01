#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 13:20:04 2020

@author: aristizabal
"""

#%% User input

# date limits
date_ini = '2020/09/08/00'
date_end = '2020/09/15/00'

# MAB
lon_lim = [-110.0,-10.0]
lat_lim = [15.0,45.0]

glid_ids = ['SG601']#['ng645'] #,'ng314'] #,'Stommel']#,'SG664','ru33','ng645']

# Server location
url_erddap = 'https://data.ioos.us/gliders/erddap'
url_GOFS = 'http://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/ts3z'

# RTOFS files
folder_RTOFS = '/home/coolgroup/RTOFS/forecasts/domains/hurricanes/RTOFS_6hourly_North_Atlantic/'

ncfiles_RTOFS = ['rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f012_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f018_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f024_6hrly_hvr_US_east.nc']
    
# COPERNICUS MARINE ENVIRONMENT MONITORING SERVICE (CMEMS)
url_cmems = 'http://nrt.cmems-du.eu/motu-web/Motu'
service_id = 'GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS'
product_id = 'global-analysis-forecast-phy-001-024'
depth_min = '0.493'
out_dir = '/home/aristizabal/crontab_jobs'
ncCOP_global = '/home/aristizabal/Copernicus/global-analysis-forecast-phy-001-024_1565877333169.nc'

# Bathymetry file
bath_file = '/home/aristizabal/bathymetry_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc';

folder_fig = '/home/aristizabal/Figures/'

#%% Cell #5: Search for glider data sets given a 
#   latitude and longitude box and time window, choose one those data sets 
#   (glider_id), plot a scatter plot of the chosen glider transect, grid 
#   and plot a contour plot of the chosen glider transect 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cmocean
from datetime import datetime, timedelta
import xarray as xr
import os

import sys
sys.path.append('/home/aristizabal/glider_model_comparisons_Python')
       
from read_glider_data import retrieve_dataset_id_erddap_server
from read_glider_data import read_glider_data_erddap_server
from process_glider_data import grid_glider_data
from glider_transect_model_com import get_glider_transect_from_GOFS

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Map
    
def glider_track_map(bath_lon,bath_lat,bath_elev,long,latg,dataset_id):
        
    lev = np.arange(-9000,9100,100)
    fig, ax = plt.subplots(figsize=(6,5))
    plt.contourf(bath_lon,bath_lat,bath_elev,lev,cmap=cmocean.cm.topo)
    plt.plot(long,latg,'.-k')
    plt.plot(long[0],latg[0],'s',color='skyblue',markeredgecolor='k',markersize=5,label=str(timeg[0])[0:16])
    plt.plot(long[-1],latg[-1],'s',color='royalblue',markeredgecolor='k',markersize=5,label=str(timeg[-1])[0:16])
    plt.axis('scaled') 
    plt.legend()   
    plt.title('Glider Track ' + dataset_id,fontsize=16)

#%% RTOFS
                 
def get_glider_transect_from_RTOFS(folder_RTOFS,ncfiles_RTOFS,date_ini,date_end,long,latg,tstamp_glider):
                  
    #Time window
    year_ini = int(date_ini.split('/')[0])
    month_ini = int(date_ini.split('/')[1])
    day_ini = int(date_ini.split('/')[2])
    
    year_end = int(date_end.split('/')[0])
    month_end = int(date_end.split('/')[1])
    day_end = int(date_end.split('/')[2])
    
    tini = datetime(year_ini, month_ini, day_ini)
    tend = datetime(year_end, month_end, day_end)  
    tvec = [tini + timedelta(int(i)) for i in np.arange((tend-tini).days+1)]
    
    # Read RTOFS grid and time
    print('Retrieving coordinates from RTOFS')
    
    if tini.month < 10:
        if tini.day < 10:
            fol = 'rtofs.' + str(tini.year) + '0' + str(tini.month) + '0' + str(tini.day)
        else:
            fol = 'rtofs.' + str(tini.year) + '0' + str(tini.month) + str(tini.day)
    else:
        if tini.day < 10:
            fol = 'rtofs.' + str(tini.year) + str(tini.month) + '0' + str(tini.day)
        else:
            fol = 'rtofs.' + str(tini.year) + str(tini.month) + str(tini.day)
            
    ncRTOFS = xr.open_dataset(folder_RTOFS + fol + '/' + ncfiles_RTOFS[0])
    lat_RTOFS = np.asarray(ncRTOFS.Latitude[:])
    lon_RTOFS = np.asarray(ncRTOFS.Longitude[:])
    depth_RTOFS = np.asarray(ncRTOFS.Depth[:])
    
    tRTOFS = []
    nc_allfiles_RTOFS = []
    for tt in tvec:
        # Read RTOFS grid and time
        if tt.month < 10:
            if tt.day < 10:
                fol = 'rtofs.' + str(tt.year) + '0' + str(tt.month) + '0' + str(tt.day)
            else:
                fol = 'rtofs.' + str(tt.year) + '0' + str(tt.month) + str(tt.day)
        else:
            if tt.day < 10:
                fol = 'rtofs.' + str(tt.year) + str(tt.month) + '0' + str(tt.day)
            else:
                fol = 'rtofs.' + str(tt.year) + str(tt.month) + str(tt.day)
    
        for t in np.arange(len(ncfiles_RTOFS)):
            ncRTOFS = xr.open_dataset(folder_RTOFS + fol + '/' + ncfiles_RTOFS[t])
            nc_allfiles_RTOFS.append(folder_RTOFS + fol + '/' + ncfiles_RTOFS[t])
            tRTOFS.append(np.asarray(ncRTOFS.MT[:])[0])
            
    tstamp_RTOFS = [mdates.date2num(tRTOFS[i]) for i in np.arange(len(tRTOFS))]
    
    sublonRTOFS = np.interp(tstamp_RTOFS,tstamp_glider,long)
    sublatRTOFS = np.interp(tstamp_RTOFS,tstamp_glider,latg)
    
    # getting the model grid positions for sublonm and sublatm
    oklonRTOFS = np.round(np.interp(sublonRTOFS,lon_RTOFS[0,:],np.arange(len(lon_RTOFS[0,:])))).astype(int)
    oklatRTOFS = np.round(np.interp(sublatRTOFS,lat_RTOFS[:,0],np.arange(len(lat_RTOFS[:,0])))).astype(int)
    
    # Getting glider transect from RTOFS
    print('Getting glider transect from RTOFS')
    if len(tRTOFS) == 0:
        temp_RTOFS = np.empty((len(depth_RTOFS),1))
        temp_RTOFS[:] = np.nan
        salt_RTOFS = np.empty((len(depth_RTOFS),1))
        salt_RTOFS[:] = np.nan
    else:
        temp_RTOFS = np.empty((len(depth_RTOFS),len(tRTOFS)))
        temp_RTOFS[:] = np.nan
        salt_RTOFS = np.empty((len(depth_RTOFS),len(tRTOFS)))
        salt_RTOFS[:] = np.nan
        for i in range(len(tRTOFS)):
            print(len(tRTOFS),' ',i)
            nc_file = nc_allfiles_RTOFS[i]
            ncRTOFS = xr.open_dataset(nc_file)
            temp_RTOFS[:,i] = ncRTOFS.variables['temperature'][0,:,oklatRTOFS[i],oklonRTOFS[i]]
            salt_RTOFS[:,i] = ncRTOFS.variables['salinity'][0,:,oklatRTOFS[i],oklonRTOFS[i]]
        
    return temp_RTOFS, salt_RTOFS, tRTOFS, depth_RTOFS

#%% Copernicus
    
def get_glider_transect_from_COP(url_cmems,service_id,product_id,long,latg,\
                                 tini,tend,depth_min,depthg,out_dir,dataset_id,tstamp_glider):
    
    COP_grid = xr.open_dataset(ncCOP_global)
    depthCOP_glob = np.asarray(COP_grid.depth[:])
            
    # Downloading and reading Copernicus output
    motuc = 'python -m motuclient --motu ' + url_cmems + \
    ' --service-id ' + service_id + \
    ' --product-id ' + product_id + \
    ' --longitude-min ' + str(np.min(long)-2/12) + \
    ' --longitude-max ' + str(np.max(long)+2/12) + \
    ' --latitude-min ' + str(np.min(latg)-2/12) + \
    ' --latitude-max ' + str(np.max(latg)+2/12) + \
    ' --date-min ' + str(tini-timedelta(0.5)) + \
    ' --date-max ' + str(tend+timedelta(0.5)) + \
    ' --depth-min ' + depth_min + \
    ' --depth-max ' + str(np.nanmax(depthg)) + \
    ' --variable ' + 'thetao' + ' ' + \
    ' --variable ' + 'so'  + ' ' + \
    ' --out-dir ' + out_dir + \
    ' --out-name ' + dataset_id + '.nc' + ' ' + \
    ' --user ' + 'maristizabalvar' + ' ' + \
    ' --pwd ' +  'MariaCMEMS2018'
    
    os.system(motuc)
    # Check if file was downloaded
    COP_file = out_dir + '/' + dataset_id + '.nc'
    # Check if file was downloaded
    resp = os.system('ls ' + out_dir +'/' + dataset_id + '.nc')
    if resp == 0:
        COP = xr.open_dataset(COP_file)
    
        latCOP = np.asarray(COP.latitude[:])
        lonCOP = np.asarray(COP.longitude[:])
        depthCOP = np.asarray(COP.depth[:])
        tCOP = np.asarray(mdates.num2date(mdates.date2num(COP.time[:])))
    else:
        latCOP = np.empty(depthCOP_glob.shape[0])
        latCOP[:] = np.nan
        lonCOP = np.empty(depthCOP_glob.shape[0])
        lonCOP[:] = np.nan
        tCOP = np.empty(depthCOP_glob.shape[0])
        tCOP[:] = np.nan
    
    #oktimeCOP = np.where(np.logical_and(mdates.date2num(tCOP) >= mdates.date2num(tmin),\
    #                                mdates.date2num(tCOP) <= mdates.date2num(tmax)))
    tstampCOP = mdates.date2num(tCOP)
    oktimeCOP = np.unique(np.round(np.interp(tstamp_glider,tstampCOP,np.arange(len(tstampCOP)))).astype(int))
    timeCOP = tCOP[oktimeCOP]
    
    # Changing times to timestamp
    tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]
    tstamp_COP = [mdates.date2num(timeCOP[i]) for i in np.arange(len(timeCOP))]
    
    # interpolating glider lon and lat to lat and lon on Copernicus time
    sublonCOP = np.interp(tstamp_COP,tstamp_glider,long)
    sublatCOP = np.interp(tstamp_COP,tstamp_glider,latg)
    
    # getting the model grid positions for sublonm and sublatm
    oklonCOP = np.round(np.interp(sublonCOP,lonCOP,np.arange(len(lonCOP)))).astype(int)
    oklatCOP = np.round(np.interp(sublatCOP,latCOP,np.arange(len(latCOP)))).astype(int)
    
    # Getting glider transect from Copernicus model
    print('Getting glider transect from Copernicus model')
    if len(oktimeCOP) == 0:
        temp_COP = np.empty((len(depthCOP),1))
        temp_COP[:] = np.nan
        salt_COP = np.empty((len(depthCOP),1))
        salt_COP[:] = np.nan
    else:
        temp_COP = np.empty((len(depthCOP),len(oktimeCOP)))
        temp_COP[:] = np.nan
        salt_COP = np.empty((len(depthCOP),len(oktimeCOP)))
        salt_COP[:] = np.nan
        for i in range(len(oktimeCOP)):
            print(len(oktimeCOP),' ',i)
            temp_COP[:,i] = COP.variables['thetao'][oktimeCOP[i],:,oklatCOP[i],oklonCOP[i]]
            salt_COP[:,i] = COP.variables['so'][oktimeCOP[i],:,oklatCOP[i],oklonCOP[i]]
    
    os.system('rm ' + out_dir + '/' + dataset_id + '.nc')
    
    return temp_COP, salt_COP, timeCOP, depthCOP

#%% 
def figure_transect_temp(time,depth,temp,date_ini,date_end,max_depth,kw):
    
    #Time window
    year_ini = int(date_ini.split('/')[0])
    month_ini = int(date_ini.split('/')[1])
    day_ini = int(date_ini.split('/')[2])
    
    year_end = int(date_end.split('/')[0])
    month_end = int(date_end.split('/')[1])
    day_end = int(date_end.split('/')[2])
    
    tini = datetime(year_ini, month_ini, day_ini)
    tend = datetime(year_end, month_end, day_end)  
    
    fig, ax = plt.subplots(figsize=(10, 3))
    cs = plt.contourf(time,depth,temp,cmap=cmocean.cm.thermal,**kw)
    plt.contour(time,depth,temp,levels=[26],colors = 'k')
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel('Depth (m)',fontsize=14)
    cbar = plt.colorbar(cs)
    cbar.ax.set_ylabel('($^oC$)',fontsize=14)
    
    xvec = [tini + timedelta(int(dt)) for dt in np.arange((tend-tini).days+1)[::2]]
    plt.xticks(xvec,fontsize=12)
    xfmt = mdates.DateFormatter('%b-%d')
    ax.xaxis.set_major_formatter(xfmt)
    plt.ylim(-np.abs(max_depth),0)
    plt.xlim(tini,timeg[-1])

#%%    
def figure_transect_salt(time,depth,salt,date_ini,date_end,max_depth,kw):
    
    #Time window
    year_ini = int(date_ini.split('/')[0])
    month_ini = int(date_ini.split('/')[1])
    day_ini = int(date_ini.split('/')[2])
    
    year_end = int(date_end.split('/')[0])
    month_end = int(date_end.split('/')[1])
    day_end = int(date_end.split('/')[2])
    
    tini = datetime(year_ini, month_ini, day_ini)
    tend = datetime(year_end, month_end, day_end)  
    
    fig, ax = plt.subplots(figsize=(10, 3))
    cs = plt.contourf(time,depth,salt,cmap=cmocean.cm.haline,**kw)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylabel('Depth (m)',fontsize=14)
    cbar = plt.colorbar(cs)
    cbar.ax.set_ylabel(' ',fontsize=14)
    
    xvec = [tini + timedelta(int(dt)) for dt in np.arange((tend-tini).days+1)[::2]]
    plt.xticks(xvec,fontsize=12)
    xfmt = mdates.DateFormatter('%b-%d')
    ax.xaxis.set_major_formatter(xfmt)
    plt.ylim(-np.abs(max_depth),0)
    plt.xlim(tini,timeg[-1])

#%% Reading bathymetry data

ncbath = xr.open_dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

#%%Time window
year_ini = int(date_ini.split('/')[0])
month_ini = int(date_ini.split('/')[1])
day_ini = int(date_ini.split('/')[2])

year_end = int(date_end.split('/')[0])
month_end = int(date_end.split('/')[1])
day_end = int(date_end.split('/')[2])

tini = datetime(year_ini, month_ini, day_ini)
tend = datetime(year_end, month_end, day_end)  

#%% 

gliders = retrieve_dataset_id_erddap_server(url_erddap,lat_lim,lon_lim,date_ini,date_end)
print(gliders)

for i,ids in enumerate(glid_ids): 
    print(ids)
    okglid = [j for j,id in enumerate(gliders) if id.split('-')[0] == glid_ids[i]][0]
    
    dataset_id = gliders[okglid]
    
    kwargs = dict(date_ini=date_ini,date_end=date_end)
    scatter_plot = 'no'
    
    tempg, saltg, timeg, latg, long, depthg = read_glider_data_erddap_server(url_erddap,dataset_id,\
                                       lat_lim,lon_lim,scatter_plot,**kwargs)
        
    contour_plot = 'no' # default value is 'yes'
    delta_z = 0.3     # default value is 0.3    
        
    tempg_gridded, timegg, depthg_gridded = \
                        grid_glider_data('temperature',dataset_id,tempg,timeg,latg,long,depthg,delta_z,contour_plot)
                        
    saltg_gridded, timegg, depthg_gridded = \
                        grid_glider_data('salinity',dataset_id,saltg,timeg,latg,long,depthg,delta_z,contour_plot)
    
    tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]
    
    #%% GOFS
                        
    # model variable name
    model_name = 'GOFS 3.1'
    var_name_model = 'water_temp'
    contour_plot = 'no'                   
                        
    temp_GOFS, time_GOFS, depth_GOFS, lat_GOFS, lon_GOFS = \
                  get_glider_transect_from_GOFS(url_GOFS,var_name_model,model_name,\
                                            tempg,timeg,latg,long,depthg,contour_plot)
     
    var_name_model = 'salinity'                 
    salt_GOFS, time_GOFS, depth_GOFS, lat_GOFS, lon_GOFS = \
                  get_glider_transect_from_GOFS(url_GOFS,var_name_model,model_name,\
                                            tempg,timeg,latg,long,depthg,contour_plot)    
    
    #%% RTOFS
    
    temp_RTOFS, salt_RTOFS, tRTOFS, depth_RTOFS = \
        get_glider_transect_from_RTOFS(folder_RTOFS,ncfiles_RTOFS,date_ini,date_end,\
                                       long,latg,tstamp_glider)   
    
    #%% Copernicus
    
    temp_COP, salt_COP, time_COP, depth_COP = get_glider_transect_from_COP(url_cmems,service_id,product_id,long,latg,\
                                 tini,tend,depth_min,depthg,out_dir,dataset_id,tstamp_glider)    

    #%% Depth limit
    
    if np.nanmax(depthg) <= 200:
        max_depth = np.nanmax(depthg)
    else:
        max_depth = 200
                                            
    #%% Color limits for plots
        
    maxtg = np.max(tempg[depthg <= max_depth])  
    maxtG = np.max(temp_GOFS[depth_GOFS <= max_depth])
    maxtR = np.max(temp_RTOFS[depth_RTOFS <= max_depth])
    maxtC = np.max(temp_COP[depth_COP <= max_depth])    
    
    mintg = np.min(tempg[depthg <= max_depth])  
    mintG = np.min(temp_GOFS[depth_GOFS <= max_depth])
    mintR = np.min(temp_RTOFS[depth_RTOFS <= max_depth])
    mintC = np.min(temp_COP[depth_COP <= max_depth])
    
    maxt = np.ceil(np.max([maxtg,maxtG,maxtR,maxtC]))
    mint = np.floor(np.min([mintg,mintG,mintR,mintC]))
    kw_temp = dict(levels = np.arange(mint,maxt+1,1))
    
    maxsg = np.max(saltg[depthg <= max_depth])  
    maxsG = np.max(salt_GOFS[depth_GOFS <= max_depth])
    maxsR = np.max(salt_RTOFS[depth_RTOFS <= max_depth])
    maxsC = np.max(salt_COP[depth_COP <= max_depth])    
    
    minsg = np.min(saltg[depthg <= max_depth])  
    minsG = np.min(salt_GOFS[depth_GOFS <= max_depth])
    minsR = np.min(salt_RTOFS[depth_RTOFS <= max_depth])
    minsC = np.min(salt_COP[depth_COP <= max_depth])
    
    maxs = np.ceil(np.max([maxsg,maxsG,maxsR,maxsC]))
    mins = np.floor(np.min([minsg,minsG,minsR,minsC]))
    if mins <= 30 :
        mins = 30
    else:
        mins = mins
    if maxs <= 38 :
        maxs = maxs
    else:
        maxs = 38
    kw_salt = dict(levels = np.arange(mins,maxs+0.2,0.2))
    
    #%% Glider track map
    
    oklatbath = np.logical_and(bath_lat >= latg[0]-5,bath_lat <= latg[-1]+5)
    oklonbath = np.logical_and(bath_lon >= long[0]-5,bath_lon <= long[-1]+5)   
    bath_latsub = bath_lat[oklatbath]
    bath_lonsub = bath_lon[oklonbath]
    bath_elevsub = bath_elev[oklatbath,:][:,oklonbath]
    
    glider_track_map(bath_lonsub,bath_latsub,bath_elevsub,long,latg,dataset_id)
    file = folder_fig + 'glider_track_' + dataset_id + '_' + str(tini)[0:10] + '_' + str(tend)[0:10]
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)
          
    #%% Glider
    figure_transect_temp(timegg,-depthg_gridded,tempg_gridded,date_ini,date_end,max_depth,kw_temp)
    plt.title('Temperature Transect ' + dataset_id,fontsize=16)
    file = folder_fig + 'temp_transect_' + dataset_id + '_' + str(tini)[0:10] + '_' + str(tend)[0:10]
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)
    
    figure_transect_salt(timegg,-depthg_gridded,saltg_gridded,date_ini,date_end,max_depth,kw_salt)
    plt.title('Salinity Transect ' + dataset_id,fontsize=16)
    file = folder_fig + 'salt_transect_' + dataset_id + '_' + str(tini)[0:10] + '_' + str(tend)[0:10]
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)
    
    #%% GOFS
    figure_transect_temp(time_GOFS,-depth_GOFS,temp_GOFS,date_ini,date_end,max_depth,kw_temp)
    plt.title('Along Track Temperature GOFS 3.1 ' + dataset_id ,fontsize=16)
    file = folder_fig + 'GOFS31_along_track_temp_' + dataset_id + '_' + str(tini)[0:10] + '_' + str(tend)[0:10]
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)
    
    figure_transect_salt(time_GOFS,-depth_GOFS,salt_GOFS,date_ini,date_end,max_depth,kw_salt)
    plt.title('Along Track Salinity GOFS 3.1 '+ dataset_id,fontsize=16)
    file = folder_fig + 'GOFS31_along_track_salt_' + dataset_id + '_' + str(tini)[0:10] + '_' + str(tend)[0:10]
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)
    
    #%% RTOFS
    figure_transect_temp(tRTOFS,-depth_RTOFS,temp_RTOFS,date_ini,date_end,max_depth,kw_temp)
    plt.title('Along Track Temperature RTOFS '+ dataset_id,fontsize=16)
    file = folder_fig + 'RTOFS_along_track_temp_' + dataset_id + '_' + str(tini)[0:10] + '_' + str(tend)[0:10]
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)
    
    figure_transect_salt(tRTOFS,-depth_RTOFS,salt_RTOFS,date_ini,date_end,max_depth,kw_salt)
    plt.title('Along Track Salinity RTOFS '+ dataset_id,fontsize=16)
    file = folder_fig + 'RTOFS_along_track_salt_' + dataset_id + '_' + str(tini)[0:10] + '_' + str(tend)[0:10]
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)
    
    #%% COP
    figure_transect_temp(time_COP,-depth_COP,temp_COP,date_ini,date_end,max_depth,kw_temp)
    plt.title('Along Track Temperature Copernicus '+ dataset_id,fontsize=16)
    file = folder_fig + 'COP_along_track_temp_' + dataset_id + '_' + str(tini)[0:10] + '_' + str(tend)[0:10]
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)
    
    figure_transect_salt(time_COP,-depth_COP,salt_COP,date_ini,date_end,max_depth,kw_salt)
    plt.title('Along Track Salinity Copernicus '+ dataset_id,fontsize=16)
    file = folder_fig + 'COP_along_track_salt_' + dataset_id + '_' + str(tini)[0:10] + '_' + str(tend)[0:10]
    plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)