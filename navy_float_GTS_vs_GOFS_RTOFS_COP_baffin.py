#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 17:05:27 2020

@author: aristizabal
"""

#%% User input

#GOFS3.1 output model location
url_GOFS_ts = 'http://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/ts3z'


# RTOFS files
folder_RTOFS = '/home/coolgroup/RTOFS/forecasts/domains/hurricanes/RTOFS_6hourly_North_Atlantic/'

nc_files_RTOFS = ['rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f012_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f018_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f024_6hrly_hvr_US_east.nc']

# COPERNICUS MARINE ENVIRONMENT MONITORING SERVICE (CMEMS)
url_cmems = 'http://nrt.cmems-du.eu/motu-web/Motu'
service_id = 'GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS'
product_id = 'global-analysis-forecast-phy-001-024'
depth_min = '0.493'
out_dir = '/home/aristizabal/crontab_jobs'

# Bathymetry file
#bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc'    
bath_file = '/home/aristizabal/bathymetry_files/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

url_GTS = 'http://osmc.noaa.gov/erddap'

# GoMex
lon_lim = [-98,-80]
lat_lim = [15,32.5]

platf_code_navy = ['4902887','4902888','4903006','4903002']

folder_fig = '/home/aristizabal/Figures/'

#%%

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import netCDF4
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from erddapy import ERDDAP
import pandas as pd
import os

# Do not produce figures on screen
#plt.switch_backend('agg')

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Get time bounds for the previous day
'''
te = datetime.today()
tend = datetime(te.year,te.month,te.day)

ti = datetime.today() - timedelta(1)
tini = datetime(ti.year,ti.month,ti.day)
'''

tini = datetime(2020,7,21)
tend = tini + timedelta(1)


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

#%% GOGF 3.1 grid

try:
    GOFS_ts = xr.open_dataset(url_GOFS_ts,decode_times=False)

    lt_GOFS = np.asarray(GOFS_ts['lat'][:])
    ln_GOFS = np.asarray(GOFS_ts['lon'][:])
    tt = GOFS_ts['time']
    t_GOFS = netCDF4.num2date(tt[:],tt.units)
    depth_GOFS = np.asarray(GOFS_ts['depth'][:])
except Exception as err:
    print(err)
    GOFS_ts = np.nan
    lt_GOFS = np.nan
    ln_GOFS = np.nan
    depth_GOFS = np.nan
    t_GOFS = tini

#%% Look for Navy floats datasets 

e = ERDDAP(server = url_GTS)

# Grab every dataset available
datasets = pd.read_csv(e.get_search_url(response='csv', search_for='all'))

print('Found '+ str(len(datasets)) + ' datasets')
print(datasets['Dataset ID'])

#dataset_id = datasets['Dataset ID'][14]
dataset_id = datasets['Dataset ID'][18]

variables = [
 'platform_code',  
 'time',
 'longitude',
 'latitude', 
 'depth',
 'ztmp',
 'zsal',
]

e = ERDDAP(
    server = url_GTS,
    protocol = 'tabledap',
    response = 'nc',
)

e.dataset_id = dataset_id
e.variables=variables

for idp in platf_code_navy:
    print(idp)
    constraints = {
        'platform_code=': idp,
        'time>=': str(tini),
        'time<=': str(tend),
        }

    e.constraints = constraints

    try:
        print(e.get_download_url())
    
        df = e.to_pandas(
            parse_dates=True,
            skiprows=(1,)  # units information can be dropped.
        ).dropna()
    
        float_id = np.unique(np.asarray(df['platform_code']))[0]
        float_times = np.asarray(df['time (UTC)'])
        float_lons = np.asarray(df['longitude (degrees_east)'])
        float_lats = np.asarray(df['latitude (degrees_north)'])
        float_depths = np.asarray(df['depth (m)'])
        float_temps = np.asarray(df['ztmp (Deg C)'])
        float_salts = np.asarray(df['zsal'])
    
        #%%
        
        tunique = np.unique(float_times)   
        for tu in tunique:
            indt = [ind for ind,t in enumerate(float_times) if t == tu]
            float_time = float_times[indt]
            float_lon = float_lons[indt]
            float_lat = float_lats[indt]
            indd = np.argsort(float_depths[indt])
            float_depth = float_depths[indt][indd]
            float_temp = float_temps[indt][indd]
            float_salt = float_salts[indt][indd]
            
            # GOFS
            print('Retrieving variables from GOFS')
            if isinstance(GOFS_ts,float):
                temp_GOFS = np.nan
                salt_GOFS = np.nan
            else:
                oktt_GOFS = np.where(t_GOFS >= datetime.strptime(float_time[0],'%Y-%m-%dT%H:%M:%SZ'))[0][0]
                oklat_GOFS = np.where(lt_GOFS >= float_lat[0])[0][0]
                oklon_GOFS = np.where(ln_GOFS >= float_lon[0]+360)[0][0]
                temp_GOFS = np.asarray(GOFS_ts['water_temp'][oktt_GOFS,:,oklat_GOFS,oklon_GOFS])
                salt_GOFS = np.asarray(GOFS_ts['salinity'][oktt_GOFS,:,oklat_GOFS,oklon_GOFS])
               
            # RTOFS 
            #Time window
            '''    
            year = int(float_time[0].year)
            month = int(float_time[0].month)
            day = int(float_time[0].day)
            tini = datetime(year, month, day)
            tend = tini + timedelta(days=1)
            '''
    
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
    
            ncRTOFS = xr.open_dataset(folder_RTOFS + fol + '/' + nc_files_RTOFS[0])
            latRTOFS = np.asarray(ncRTOFS.Latitude[:])
            lonRTOFS = np.asarray(ncRTOFS.Longitude[:])
            depth_RTOFS = np.asarray(ncRTOFS.Depth[:])    
                
            tRTOFS = []
            for t in np.arange(len(nc_files_RTOFS)):
                ncRTOFS = xr.open_dataset(folder_RTOFS + fol + '/' + nc_files_RTOFS[t])
                tRTOFS.append(np.asarray(ncRTOFS.MT[:])[0])
    
            tRTOFS = np.asarray([mdates.num2date(mdates.date2num(tRTOFS[t])) \
                      for t in np.arange(len(nc_files_RTOFS))])
    
            oktt_RTOFS = np.where(mdates.date2num(tRTOFS) >= \
                    mdates.date2num(datetime.strptime(float_time[0],'%Y-%m-%dT%H:%M:%SZ')))[0][0] 
            oklat_RTOFS = np.where(latRTOFS[:,0] >= float_lat[0])[0][0]
            oklon_RTOFS = np.where(lonRTOFS[0,:]  >= float_lon[0])[0][0]
    
            nc_file = folder_RTOFS + fol + '/' + nc_files_RTOFS[oktt_RTOFS]
            ncRTOFS = xr.open_dataset(nc_file)
            #time_RTOFS = tRTOFS[oktt_RTOFS]
            temp_RTOFS = np.asarray(ncRTOFS.variables['temperature'][0,:,oklat_RTOFS,oklon_RTOFS])
            salt_RTOFS = np.asarray(ncRTOFS.variables['salinity'][0,:,oklat_RTOFS,oklon_RTOFS])
             
            # Downloading and reading Copernicus output
            motuc = 'python -m motuclient --motu ' + url_cmems + \
            ' --service-id ' + service_id + \
            ' --product-id ' + product_id + \
            ' --longitude-min ' + str(float_lon[0]-2/12) + \
            ' --longitude-max ' + str(float_lon[0]+2/12) + \
            ' --latitude-min ' + str(float_lat[0]-2/12) + \
            ' --latitude-max ' + str(float_lat[0]+2/12) + \
            ' --date-min ' + str(tini-timedelta(0.5)) + \
            ' --date-max ' + str(tend+timedelta(0.5)) + \
            ' --depth-min ' + depth_min + \
            ' --depth-max ' + str(np.nanmax(float_depth)+1000) + \
            ' --variable ' + 'thetao' + ' ' + \
            ' --variable ' + 'so'  + ' ' + \
            ' --out-dir ' + out_dir + \
            ' --out-name ' + str(idp) + '.nc' + ' ' + \
            ' --user ' + 'maristizabalvar' + ' ' + \
            ' --pwd ' +  'MariaCMEMS2018'
    
            os.system(motuc)
            # Check if file was downloaded
    
            COP_file = out_dir + '/' + str(idp) + '.nc'
    
            resp = os.system('ls ' + out_dir +'/' + str(idp) + '.nc')
            if resp == 0:
                COP = xr.open_dataset(COP_file)
    
                latCOP = np.asarray(COP.latitude[:])
                lonCOP = np.asarray(COP.longitude[:])
                depth_COP = np.asarray(COP.depth[:])
                tCOP = np.asarray(mdates.num2date(mdates.date2num(COP.time[:])))
            else:
                latCOP = np.empty(1)
                latCOP[:] = np.nan
                lonCOP = np.empty(1)
                lonCOP[:] = np.nan
                tCOP = np.empty(1)
                tCOP[:] = np.nan
    
            oktimeCOP = np.where(mdates.date2num(tCOP) >= mdates.date2num(tini))[0][0]
            oklonCOP = np.where(lonCOP >= float_lon[0])[0][0]
            oklatCOP = np.where(latCOP >= float_lat[0])[0][0]
    
            temp_COP = np.asarray(COP.variables['thetao'][oktimeCOP,:,oklatCOP,oklonCOP])
            salt_COP = np.asarray(COP.variables['so'][oktimeCOP,:,oklatCOP,oklonCOP])
            
            # Figure temp
            plt.figure(figsize=(5,6))
            plt.plot(float_temp,-float_depth,'.-',linewidth=2,label='Navy Float id '+str(idp))
            plt.plot(temp_GOFS,-depth_GOFS,'.-',linewidth=2,label='GOFS 3.1',color='red')
            plt.plot(temp_RTOFS,-depth_RTOFS,'.-',linewidth=2,label='RTOFS',color='g')
            plt.plot(temp_COP,-depth_COP,'.-',linewidth=2,label='Copernicus',color='darkorchid')
            plt.ylim([-np.max(float_depth)-100,0])
            plt.title('Temperature Profile on '+ str(float_time[0])[0:13] +
                      '\n [lon,lat] = [' \
                      + str(np.round(float_lon[0],3)) +',' +\
                          str(np.round(float_lat[0],3))+']',\
                          fontsize=16)
            plt.ylabel('Depth (m)',fontsize=14)
            plt.xlabel('$^oC$',fontsize=14)
            plt.legend(loc='lower right',fontsize=14)
    
            file = folder_fig + 'Navy_floats_vs_GOFS_RTOFS_COP_temp_' + str(idp)
            plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)
            
            # Figure salinity
            plt.figure(figsize=(5,6))
            plt.plot(float_salt,-float_depth,'.-',linewidth=2,label='Navy Float id '+str(idp))
            plt.plot(salt_GOFS,-depth_GOFS,'.-',linewidth=2,label='GOFS 3.1',color='red')
            plt.plot(salt_RTOFS,-depth_RTOFS,'.-',linewidth=2,label='RTOFS',color='g')
            plt.plot(salt_COP,-depth_COP,'.-',linewidth=2,label='Copernicus',color='darkorchid')
            plt.ylim([-np.max(float_depth)-100,0])
            plt.title('Salinity Profile on '+ str(float_time[0])[0:13] +
                      '\n [lon,lat] = [' \
                      + str(np.round(float_lon[0],3)) +',' +\
                          str(np.round(float_lat[0],3))+']',\
                          fontsize=16)
            plt.ylabel('Depth (m)',fontsize=14)
            plt.legend(loc='lower right',fontsize=14)
    
            file = folder_fig + 'Navy_floats_vs_GOFS_RTOFS_COP_salt_' + str(idp)
            plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)
        
    except Exception as err:
        print(err)
        