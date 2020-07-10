#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 16:50:31 2020

@author: aristizabal
"""

#%% User input

# RU33 (MAB + SAB)
#lon_lim = [-75,-70]
#lat_lim = [36,42]

lon_lim = [-77,-70]
lat_lim = [34,42]


# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/';

#Time window
#date_ini = '2019-08-26T00:00:00Z'
#date_end = '2019-08-27T00:00:00Z'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc';

# url for GOFS 3.1
#url_GOFS = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'
url_GOFS = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/latest'

# url Doppio
url_doppio = 'http://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/2017_da/his/History_Best'

# FTP server RTOFS
ftp_RTOFS = 'ftp.ncep.noaa.gov'

nc_files_RTOFS = ['rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f012_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f018_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f024_6hrly_hvr_US_east.nc']

# COPERNICUS MARINE ENVIRONMENT MONITORING SERVICE (CMEMS)
url_cmems = 'http://nrt.cmems-du.eu/motu-web/Motu'
service_id = 'GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS'
product_id = 'global-analysis-forecast-phy-001-024'
depth_min = '0.493'
out_dir = '/Users/aristizabal/Desktop'
ncCOP_global = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/global-analysis-forecast-phy-001-024_1565877333169.nc'

#%%

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import netCDF4
from datetime import datetime, timedelta
import cmocean
import matplotlib.dates as mdates 
from ftplib import FTP
import seawater

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

#%% Get time bounds for current day

te = datetime.today() + timedelta(1)
tend = datetime(te.year,te.month,te.day)

#ti = datetime.today() - timedelta(1)
ti = datetime.today() 
tini = datetime(ti.year,ti.month,ti.day)

#%% Reading bathymetry data

ncbath = xr.open_dataset(bath_file)
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

#%% Read GOFS 3.1 lat, lon, depth and time

print('Retrieving coordinates from GOFS')

GOFS = xr.open_dataset(url_GOFS,decode_times=False)

latGOFS = np.asarray(GOFS['lat'][:])
lonGOFS = np.asarray(GOFS['lon'][:])
depthGOFS = np.asarray(GOFS['depth'][:])
ttGOFS= GOFS['time']
tGOFS = netCDF4.num2date(ttGOFS[:],ttGOFS.units) 

#tini = datetime.datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
#tend = datetime.datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

oktimeGOFS = np.where(np.logical_and(tGOFS >= tini, tGOFS <= tend))[0]
timeGOFS = tGOFS[oktimeGOFS]

# Conversion from glider longitude and latitude to GOFS convention
lon_limGOFS = np.empty((len(lon_lim),))
lon_limGOFS[:] = np.nan
for i,ii in enumerate(lon_lim):
    if ii < 0: 
        lon_limGOFS[i] = 360 + ii
    else:
        lon_limGOFS[i] = ii
lat_limGOFS = lat_lim

oklonGOFS = np.where(np.logical_and(lonGOFS >= lon_limGOFS[0],lonGOFS <= lon_limGOFS[1]))[0]
oklatGOFS = np.where(np.logical_and(latGOFS >= lat_limGOFS[0],latGOFS <= lat_limGOFS[1]))[0]

#%% load RTOFS nc files

print('Loading 6 hourly RTOFS nc files from FTP server')
for t in np.arange(len(nc_files_RTOFS)):
    #file = out_dir + '/' + nc_files_RTOFS[t]
    file = nc_files_RTOFS[t]

    # Login to ftp file
    ftp = FTP('ftp.ncep.noaa.gov')
    ftp.login()
    ftp.cwd('pub/data/nccf/com/rtofs/prod/')
    if tend.month < 10:
        if tend.day < 10:
            ftp.cwd('rtofs.' + str(tini.year) + '0' + str(tini.month) + '0' + str(tini.day))
        else:
            ftp.cwd('rtofs.' + str(tini.year) + '0' + str(tini.month) + str(tini.day))
    else:
        if tend.day < 10:
            ftp.cwd('rtofs.' + str(tini.year) + str(tini.month) + '0' + str(tini.day))
        else:
            ftp.cwd('rtofs.' + str(tini.year) + str(tini.month) + str(tini.day))

    # Download nc files
    print('loading ' + file)
    ftp.retrbinary('RETR '+file, open(file,'wb').write)

#%% Read RTOFS grid and time
    
print('Retrieving coordinates from RTOFS')
RTOFS = xr.open_dataset(nc_files_RTOFS[0])
latRTOFS = np.asarray(RTOFS.Latitude[:])
lonRTOFS = np.asarray(RTOFS.Longitude[:])
depthRTOFS = np.asarray(RTOFS.Depth[:])

tRTOFS = []
for t in np.arange(len(nc_files_RTOFS)):
    RTOFS = xr.open_dataset(nc_files_RTOFS[t])
    tRTOFS.append(np.asarray(RTOFS.MT[:])[0])

tRTOFS = np.asarray([mdates.num2date(mdates.date2num(tRTOFS[t])) \
          for t in np.arange(len(nc_files_RTOFS))])
    
oktimeRTOFS = np.where(np.logical_and(mdates.date2num(tRTOFS) >= mdates.date2num(tini), 
                                     mdates.date2num(tRTOFS) <= mdates.date2num(tend)))
timeRTOFS = tRTOFS[oktimeRTOFS[0]]

oklonRTOFS = np.where(np.logical_and(lonRTOFS[0,:] >= lon_lim[0],lonRTOFS[0,:] <= lon_lim[1]))[0]
oklatRTOFS = np.where(np.logical_and(latRTOFS[:,0] >= lat_lim[0],latRTOFS[:,0] <= lat_lim[1]))[0]

#%% Downloading and reading Copernicus grid
    
COP_grid = xr.open_dataset(ncCOP_global)

latCOP_glob = np.asarray(COP_grid.latitude[:])
lonCOP_glob = np.asarray(COP_grid.longitude[:])
    
#%% Read Doppio time, lat and lon
print('Retrieving coordinates and time from Doppio ')

doppio = xr.open_dataset(url_doppio,decode_times=False)

latrhodoppio = np.asarray(doppio.variables['lat_rho'][:])
lonrhodoppio = np.asarray(doppio.variables['lon_rho'][:])
srhodoppio = np.asarray(doppio.variables['s_rho'][:])
ttdoppio = doppio.variables['time'][:]
tdoppio = netCDF4.num2date(ttdoppio[:],ttdoppio.attrs['units'])

oktimeDOPP = np.where(np.logical_and(tdoppio >= tini, tdoppio <= tend))
timeDOPP = tdoppio[oktimeDOPP]

#%% Read Doppio S-coordinate parameters

Vtransf = np.asarray(doppio.variables['Vtransform'])
Vstrect = np.asarray(doppio.variables['Vstretching'])
Cs_r = np.asarray(doppio.variables['Cs_r'])
Cs_w = np.asarray(doppio.variables['Cs_w'])
sc_r = np.asarray(doppio.variables['s_rho'])
sc_w = np.asarray(doppio.variables['s_w'])

# depth
h = np.asarray(doppio.variables['h'])
# critical depth parameter
hc = np.asarray(doppio.variables['hc'])

igrid = 1

#%% Calculate non-dimensional potential Energy Anomaly for GOFS 3.1 
    
## Limit the depths to 200m
okdGOFS = np.where(depthGOFS <= 200)[0]

tempGOFS_200 = np.asarray(GOFS.variables['water_temp'][oktimeGOFS[4],okdGOFS,oklatGOFS,oklonGOFS])
saltGOFS_200 = np.asarray(GOFS.variables['salinity'][oktimeGOFS[4],okdGOFS,oklatGOFS,oklonGOFS])
densGOFS_200 = seawater.dens0(saltGOFS_200,tempGOFS_200)

zz = depthGOFS[okdGOFS]
zs = np.tile(zz,(1,tempGOFS_200.shape[1]))
zss = np.tile(zs,(1,tempGOFS_200.shape[2]))
zmat = np.reshape(zss,(tempGOFS_200.shape[0],tempGOFS_200.shape[1],tempGOFS_200.shape[2]), order='F')
land = np.isnan(densGOFS_200)
zmat[land] = np.nan
 
sigma = np.empty((tempGOFS_200.shape[0],tempGOFS_200.shape[1],tempGOFS_200.shape[2]))
sigma[:] = -1.0
for x,xx in enumerate(np.arange(tempGOFS_200.shape[1])):
    for y,yy in enumerate(np.arange(tempGOFS_200.shape[2])):
        sigma[:,x,y] = -zmat[:,x,y]/np.nanmax(zmat[:,x,y])
        sigma[np.isnan(sigma[:,x,y]),x,y] = -1

dens = np.copy(densGOFS_200)
dens[np.isnan(dens)] = 0        
rhomean = -np.trapz(dens,sigma,axis=0)
lands = np.isnan(densGOFS_200[0,:,:])
rhomean[lands] = np.nan

drho = np.empty((tempGOFS_200.shape[0],tempGOFS_200.shape[1],tempGOFS_200.shape[2]))
drho[:] = np.nan
for k,dind in enumerate(np.arange(tempGOFS_200.shape[0])):
    drho[k,:,:] = (rhomean - densGOFS_200[k,:,:])/rhomean
drho[land] = 0

torque = drho * sigma

NPEA_GOFS = np.trapz(torque,sigma,axis=0)
NPEA_GOFS[lands] = np.nan

#%%

kw = dict(levels = np.arange(-0.56,0,0.05))
fig, ax = plt.subplots(figsize=(6, 6)) 
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
plt.contourf(lonGOFS[oklonGOFS]-360,latGOFS[oklatGOFS],NPEA_GOFS*1000,cmap=cmocean.cm.deep_r,**kw)
cb = plt.colorbar()
plt.axis('scaled')
cb.set_label('Stratification Factor ($x1000$)',rotation=270, labelpad=20, fontsize=14)
plt.title('Non-Dimensional Potential Energy Anomaly \n GOFS on '+str(tGOFS[oktimeGOFS[4]])[0:13],\
          fontsize=16)
