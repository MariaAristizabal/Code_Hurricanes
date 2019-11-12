#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:49:53 2019

@author: root
"""

#%%

# Servers location
url_glider = 'https://data.ioos.us/gliders/erddap'
url_model = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'

# MAB
lon_lim = [-75.0,-72.0]
lat_lim = [38.0,40.0]

# date limits
date_ini = '2018-09-01T00:00:00Z'
date_end = '2018-09-10T00:00:00Z'

# glider variable to retrieve
var_glider = 'temperature'
#var_glider = 'salinity'
delta_z = 0.4 # bin size in the vertical when gridding the variable vertical profile 
              # default value is 0.3  

# model variable name
model_name = 'GOFS 3.1'
var_model = 'water_temp'
#var_model = 'salinity'

#%%

    from erddapy import ERDDAP
    import pandas as pd

    e = ERDDAP(server = url_glider)

    # Search constraints
    kw2018 = {
            'min_lon': lon_lim[0],
            'max_lon': lon_lim[1],
            'min_lat': lat_lim[0],
            'max_lat': lat_lim[1],
            'min_time': date_ini,
            'max_time': date_end,
            }

    search_url = e.get_search_url(response='csv', **kw2018)
    search = pd.read_csv(search_url)
    
    # Extract the IDs
    gliders = search['Dataset ID'].values
    
#%%
    
    dataset_id = gliders[0]
    print(glid)
    
#    timeg,depthg_gridded,varg_gridded,timem,depthm,target_varm = \
#    glider_transect_model_com_erddap_server(url_glider,dataset_id,url_model,\
#                              lat_lim,lon_lim,\
#                              date_ini,date_end,var_glider,var_model,model_name,delta_z=0.4)
    
    import xarray as xr
    import netCDF4
    import numpy as np
    import datetime
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import cmocean

    from read_glider_data import read_glider_data_erddap_server
    from process_glider_data import grid_glider_data_erddap

#%%
    # Read GOFS 3.1 output
    print('Retrieving coordinates from model')
    model = xr.open_dataset(url_model,decode_times=False)
    
    latm = model.lat[:]
    lonm = model.lon[:]
    depthm = model.depth[:]
    ttm = model.time
    tm = netCDF4.num2date(ttm[:],ttm.units) 

    tmin = datetime.datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
    tmax = datetime.datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

    oktimem = np.where(np.logical_and(tm >= tmin, tm <= tmax))
    
    timem = tm[oktimem]
    
#%%
    
    # Read and process glider data
    print('Reading glider data')
    df = read_glider_data_erddap_server(url_glider,dataset_id,var_glider,\
                                        lat_lim,lon_lim,date_ini,date_end,\
                                        scatter_plot='no')
    
    if len(df) != 0:

        depthg_gridded, varg_gridded, timeg, latg, long = \
                       grid_glider_data_erddap(df,dataset_id,var_glider,delta_z=0.2,contour_plot='no')

        # Conversion from glider longitude and latitude to GOFS convention
        target_lon = np.empty((len(long),))
        target_lon[:] = np.nan
        for i,ii in enumerate(long):
            if ii < 0: 
                target_lon[i] = 360 + ii
            else:
                target_lon[i] = ii
        target_lat = latg
        
#%%        

        # Changing times to timestamp
        tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]
        tstamp_model = [mdates.date2num(timem[i]) for i in np.arange(len(timem))]

        # interpolating glider lon and lat to lat and lon on model time
        sublonm=np.interp(tstamp_model,tstamp_glider,target_lon)
        sublatm=np.interp(tstamp_model,tstamp_glider,target_lat)

        # getting the model grid positions for sublonm and sublatm
        oklonm=np.round(np.interp(sublonm,lonm,np.arange(len(lonm)))).astype(int)
        oklatm=np.round(np.interp(sublatm,latm,np.arange(len(latm)))).astype(int)
        
        plt.figure()
        plt.plot(tstamp_glider,target_lon,'.-')
        plt.plot(tstamp_model,sublonm,'.-')
    
        # Getting glider transect from model
        print('Getting glider transect from model. If it breaks is because GOFS 3.1 server is not responding')
        target_varm = np.empty((len(depthm),len(oktimem[0])))
        target_varm[:] = np.nan
        for i in range(len(oktimem[0])):
            print(len(oktimem[0]),' ',i)
            target_varm[:,i] = model.variables[var_model][oktimem[0][i],:,oklatm[i],oklonm[i]]
            
#%%
        # plot
        if var_glider == 'temperature':
            color_map = cmocean.cm.thermal
        else:
            if var_glider == 'salinity':
                color_map = cmocean.cm.haline
            else:
                color_map = 'RdBu_r'
        
        okg = depthg_gridded <= np.max(depthg_gridded) 
        okm = depthm <= np.max(depthg_gridded) 
        min_val = np.floor(np.min([np.nanmin(varg_gridded[okg]),np.nanmin(target_varm[okm])]))
        max_val = np.ceil(np.max([np.nanmax(varg_gridded[okg]),np.nanmax(target_varm[okm])]))
    
        if var_glider == 'salinity':
            kw = dict(levels = np.arange(min_val,max_val+0.25,0.25))
        else:
            nlevels = max_val - min_val + 1
            kw = dict(levels = np.linspace(min_val,max_val,nlevels))
    
        # plot
        fig, ax = plt.subplots(figsize=(12, 6))

        ax = plt.subplot(211)        
        #plt.contour(timeg,-depthg_gridded,varg_gridded,colors = 'lightgrey',**kw)
        cs = plt.contourf(timeg,-depthg_gridded,varg_gridded,cmap=color_map,**kw)
        plt.contour(timeg,-depthg_gridded,varg_gridded,[26],colors='k')

        cs = fig.colorbar(cs, orientation='vertical') 
        cs.ax.set_ylabel(var_glider[0].upper()+var_glider[1:],fontsize=14,labelpad=15)
        
        ax.set_xlim(df.index[0], df.index[-1])
        ax.set_ylim(-np.max(depthg_gridded), 0)
        ax.set_ylabel('Depth (m)',fontsize=14)
        ax.set_xticklabels(' ')
    
        plt.title('Along Track ' + var_glider[0].upper() + var_glider[1:] + ' Profile ' + dataset_id.split('-')[0])
    
        ax = plt.subplot(212)        
        #plt.contour(mdates.date2num(timem),-depthm,target_varm,colors = 'lightgrey',**kw)
        cs = plt.contourf(mdates.date2num(timem),-depthm,target_varm,cmap=color_map,**kw)
        plt.contour(mdates.date2num(timem),-depthm,target_varm,[26],colors='k')
        cs = fig.colorbar(cs, orientation='vertical') 
        cs.ax.set_ylabel(var_glider[0].upper()+var_glider[1:],fontsize=14,labelpad=15)

        ax.set_xlim(df.index[0], df.index[-1])
        ax.set_ylim(-np.max(depthg_gridded), 0)
        ax.set_ylabel('Depth (m)',fontsize=14)
        xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
        ax.xaxis.set_major_formatter(xfmt)

        plt.title('Along Track ' + var_glider[0].upper() + var_glider[1:] + ' Profile ' + model_name)     
    
    else:
        timeg = []
        depthg_gridded = []
        varg_gridded = []
        timem = []
        depthm = []
        target_varm = []