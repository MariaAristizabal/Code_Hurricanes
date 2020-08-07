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
#bath_file = '/home/aristizabal/bathymetry_files/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

# url for GOFS 3.1
#url_GOFS = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'
#url_GOFS = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/latest'
url_GOFS = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/ts3z'

# url Doppio
url_doppio = 'http://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/2017_da/his/History_Best'

# FTP server RTOFS
ftp_RTOFS = 'ftp.ncep.noaa.gov'

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
out_dir = '/Users/aristizabal/Desktop'
ncCOP_global = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/global-analysis-forecast-phy-001-024_1565877333169.nc'

# Folder where to save figure
folder_fig = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

#%%

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import netCDF4
from datetime import datetime
from datetime import timedelta
import cmocean
import matplotlib.dates as mdates 
#from ftplib import FTP
import seawater
import os

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Calculate non-dimensional potential Energy Anomaly for GOFS 3.1 

def Non_Dim_PEA(temp,salt,dens,depth):  

    zz = depth
    zs = np.tile(zz,(1,temp.shape[1]))
    zss = np.tile(zs,(1,temp.shape[2]))
    zmat = np.reshape(zss,(temp.shape[0],temp.shape[1],temp.shape[2]), order='F')
    land = np.isnan(dens)
    zmat[land] = np.nan
     
    sigma = np.empty((temp.shape[0],temp.shape[1],temp.shape[2]))
    sigma[:] = -1.0
    for x,xx in enumerate(np.arange(temp.shape[1])):
        for y,yy in enumerate(np.arange(temp.shape[2])):
            sigma[:,x,y] = -zmat[:,x,y]/np.nanmax(zmat[:,x,y])
            sigma[np.isnan(sigma[:,x,y]),x,y] = -1
    
    denss = np.copy(dens)
    denss[np.isnan(denss)] = 0        
    rhomean = -np.trapz(denss,sigma,axis=0)
    lands = np.isnan(dens[0,:,:])
    rhomean[lands] = np.nan
    
    drho = np.empty((temp.shape[0],temp.shape[1],temp.shape[2]))
    drho[:] = np.nan
    for k,dind in enumerate(np.arange(temp.shape[0])):
        drho[k,:,:] = (rhomean - dens[k,:,:])/rhomean
    drho[land] = 0
    
    torque = drho * sigma
    
    NPEA = np.trapz(torque,sigma,axis=0)
    NPEA[lands] = np.nan
    
    return NPEA

#%% Get time bounds for the previous day
'''
te = datetime.today()
tend = datetime(te.year,te.month,te.day)

ti = datetime.today() - timedelta(1)
tini = datetime(ti.year,ti.month,ti.day)
'''

#%% Get time bounds for current day
'''
te = datetime.today() + timedelta(1)
tend = datetime(te.year,te.month,te.day)

#ti = datetime.today() - timedelta(1)
ti = datetime.today() 
tini = datetime(ti.year,ti.month,ti.day)
'''

tini = datetime(2020, 7, 9)
tend = datetime(2020, 7, 10)

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

#%% NPEA GOFS 100
    
## Limit the depths to 100m
okdGOFS = np.where(depthGOFS <= 100)[0]
depthGOFS_100 = depthGOFS[okdGOFS]

tempGOFS_100 = np.asarray(GOFS.variables['water_temp'][oktimeGOFS[4],okdGOFS,oklatGOFS,oklonGOFS])
saltGOFS_100 = np.asarray(GOFS.variables['salinity'][oktimeGOFS[4],okdGOFS,oklatGOFS,oklonGOFS])
densGOFS_100 = seawater.dens0(saltGOFS_100,tempGOFS_100)
    
NPEA_GOFS_100 = Non_Dim_PEA(tempGOFS_100,saltGOFS_100,densGOFS_100,depthGOFS_100)

#%% load RTOFS nc files
'''
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
'''

#%% Read RTOFS grid and time
'''    
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
'''

#%% Read RTOFS grid and time
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
depthRTOFS = np.asarray(ncRTOFS.Depth[:])

tRTOFS = []
for t in np.arange(len(nc_files_RTOFS)):
    ncRTOFS = xr.open_dataset(folder_RTOFS + fol + '/' + nc_files_RTOFS[t])
    tRTOFS.append(np.asarray(ncRTOFS.MT[:])[0])

tRTOFS = np.asarray([mdates.num2date(mdates.date2num(tRTOFS[t])) \
          for t in np.arange(len(nc_files_RTOFS))])

oklonRTOFS = np.where(np.logical_and(lonRTOFS[0,:] >= lon_lim[0],lonRTOFS[0,:] <= lon_lim[1]))[0]
oklatRTOFS = np.where(np.logical_and(latRTOFS[:,0] >= lat_lim[0],latRTOFS[:,0] <= lat_lim[1]))[0]

#%% Downloading and reading Copernicus grid
    
COP_grid = xr.open_dataset(ncCOP_global)

latCOP_glob = np.asarray(COP_grid.latitude[:])
lonCOP_glob = np.asarray(COP_grid.longitude[:])

#%% Downloading and reading Copernicus output
    
motuc = 'python -m motuclient --motu ' + url_cmems + \
        ' --service-id ' + service_id + \
        ' --product-id ' + product_id + \
        ' --longitude-min ' + str(np.min(lon_lim[0])-2/12) + \
        ' --longitude-max ' + str(np.max(lon_lim[1])+2/12) + \
        ' --latitude-min ' + str(np.min(lat_lim[0])-2/12) + \
        ' --latitude-max ' + str(np.max(lat_lim[1])+2/12) + \
        ' --date-min ' + str(tini-timedelta(0.5)) + \
        ' --date-max ' + str(tend+timedelta(0.5)) + \
        ' --depth-min ' + depth_min + \
        ' --depth-max ' + str(1000) + \
        ' --variable ' + 'thetao' + ' ' + \
        ' --variable ' + 'so'  + ' ' + \
        ' --out-dir ' + out_dir + \
        ' --out-name ' + 'MAB' + '.nc' + ' ' + \
        ' --user ' + 'maristizabalvar' + ' ' + \
        ' --pwd ' +  'MariaCMEMS2018'

os.system(motuc)

#%%
COP_file = out_dir + '/' + 'MAB'  + '.nc'
COP = xr.open_dataset(COP_file)

latCOP = np.asarray(COP.latitude[:])
lonCOP = np.asarray(COP.longitude[:])
depthCOP = np.asarray(COP.depth[:])
tCOP = np.asarray(mdates.num2date(mdates.date2num(COP.time[:])))

oktimeCOP = np.where(np.logical_and(mdates.date2num(tCOP) >= mdates.date2num(tini),\
                                        mdates.date2num(tCOP) <= mdates.date2num(tend)))[0]
timeCOP = tCOP[oktimeCOP]

oklonCOP = np.where(np.logical_and(lonCOP >= lon_lim[0],lonCOP <= lon_lim[1]))[0]
oklatCOP = np.where(np.logical_and(latCOP >= lat_lim[0],latCOP <= lat_lim[1]))[0]

#%% NPEA COP 100 m
okdCOP = np.where(depthCOP <= 100)[0]
depthCOP_100 = depthCOP[okdCOP]

tempCOP_100 = np.asarray(COP.variables['thetao'][oktimeCOP,okdCOP,oklatCOP,oklonCOP])[0]
saltCOP_100 = np.asarray(COP.variables['so'][oktimeCOP,okdCOP,oklatCOP,oklonCOP])[0]
densCOP_100 = seawater.dens0(saltCOP_100,tempCOP_100)
    
NPEA_COP_100 = Non_Dim_PEA(tempCOP_100,saltCOP_100,densCOP_100,depthCOP_100)
    
#%% Read Doppio time, lat and lon
print('Retrieving coordinates and time from Doppio ')

doppio = xr.open_dataset(url_doppio,decode_times=False)

latrhodoppio = np.asarray(doppio.variables['lat_rho'][:])
lonrhodoppio = np.asarray(doppio.variables['lon_rho'][:])
srhodoppio = np.asarray(doppio.variables['s_rho'][:])
ttdoppio = doppio.variables['time'][:]
tdoppio = netCDF4.num2date(ttdoppio[:],ttdoppio.attrs['units'])

oktimeDOPP = np.where(np.logical_and(tdoppio >= tini, tdoppio <= tend))[0]
timeDOPP = tdoppio[oktimeDOPP]

#okxiDOPP = np.where(np.logical_and(latrhodoppio[0,:] >= lat_lim[0],latrhodoppio[-1,:] <= lat_lim[1]))[0]
#oketaDOPP = np.where(np.logical_and(lonrhodoppio[:,0] >= lon_lim[0],lonrhodoppio[:,-1] <= lon_lim[1]))[0]

okxiDOPP = np.where(np.logical_and(lonrhodoppio[0,:] >= lon_lim[0],lonrhodoppio[-1,:] <= lon_lim[1]))[0]
oketaDOPP = np.where(np.logical_and(latrhodoppio[:,0] >= lat_lim[0], latrhodoppio[:,-1] >= lat_lim[1]))[0]

#%% Read Doppio S-coordinate parameters

Vtransf = np.asarray(doppio.variables['Vtransform'])
Vstrect = np.asarray(doppio.variables['Vstretching'])
Cs_r = np.asarray(doppio.variables['Cs_r'])
Cs_w = np.asarray(doppio.variables['Cs_w'])
sc_r = np.asarray(doppio.variables['s_rho'])
sc_w = np.asarray(doppio.variables['s_w'])

# critical depth parameter
hc = np.asarray(doppio.variables['hc'])

igrid = 1

#%% NPEA DOPP

# depth
h = np.asarray(doppio.variables['h'][oketaDOPP,okxiDOPP])
zeta = np.asarray(doppio.variables['zeta'][oktimeDOPP[12],oketaDOPP,okxiDOPP])

depthDOPP = np.empty((sc_r.shape[0],zeta.shape[0],zeta.shape[1]))

# Calculate doppio depth
if Vtransf ==1:
    if igrid == 1:
        for k in np.arange(sc_r.shape[0]):
            z0 = (sc_r[k]-Cs_r[k])*hc + Cs_r[k]*h
            depthDOPP[k,:,:] = z0 + zeta * (1.0 + z0/h)

if Vtransf == 2:
    if igrid == 1:
        for k in np.arange(sc_r.shape[0]):
            z0 = (hc*sc_r[k] + Cs_r[k]*h) / (hc+h)
            depthDOPP[k,:,:] = zeta + (zeta+h)*z0
            
tempDOPP_100 = np.empty((depthDOPP.shape[0],depthDOPP.shape[1],depthDOPP.shape[2]))
tempDOPP_100[:] = np.nan
saltDOPP_100 = np.empty((depthDOPP.shape[0],depthDOPP.shape[1],depthDOPP.shape[2]))
saltDOPP_100[:] = np.nan
for eta,eta_pos in enumerate(oketaDOPP):
    print(eta)
    for xi,xi_pos in enumerate(okxiDOPP):
        print(xi)
        okd100 = np.where(depthDOPP[:,eta,xi] >= -100)[0]
        if len(okd100) != 0:
            tempDOPP_100[okd100,eta,xi] = np.asarray(doppio.variables['temp'][oktimeDOPP[12],okd100,eta_pos,xi_pos]) 
            saltDOPP_100[okd100,eta,xi] = saltDOPP = np.asarray(doppio.variables['salt'][oktimeDOPP[12],okd100,eta_pos,xi_pos]) 
        else: 
            tempDOPP_100[:,eta,xi] = np.nan
            saltDOPP_100[:,eta,xi] = np.nan
            
densDOPP_100 = seawater.dens0(saltDOPP_100,tempDOPP_100)
    
NPEA_DOPP_100 = Non_Dim_PEA(tempDOPP_100,saltDOPP_100,densDOPP_100,depthDOPP)

#%% NPEA GOFS

kw = dict(levels = np.arange(-0.56,0,0.05))
fig, ax = plt.subplots(figsize=(6, 6)) 
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
plt.contourf(lonGOFS[oklonGOFS]-360,latGOFS[oklatGOFS],NPEA_GOFS_100*1000,cmap=cmocean.cm.deep_r,**kw)
cb = plt.colorbar()
cs = plt.contour(lonGOFS[oklonGOFS]-360,latGOFS[oklatGOFS],NPEA_GOFS_100*1000,[-0.35],color='k')
fmt = '%r'
plt.clabel(cs,fmt=fmt)
plt.axis('scaled')
cb.set_label('Stratification Factor ($x1000$)',rotation=270, labelpad=20, fontsize=14)
plt.title('Non-Dimensional Potential Energy Anomaly 100 m\n GOFS on '+str(tGOFS[oktimeGOFS[4]])[0:13],\
          fontsize=16)

file = folder_fig + 'MAB_NPEA_100m_GOFS'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)    

#%% NPEA COP

kw = dict(levels = np.arange(-0.56,0,0.05))
fig, ax = plt.subplots(figsize=(6, 6)) 
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
plt.contourf(lonCOP[oklonCOP],latCOP[oklatCOP],NPEA_COP_100*1000,cmap=cmocean.cm.deep_r) #,**kw)
cb = plt.colorbar()
cs = plt.contour(lonCOP[oklonCOP],latCOP[oklatCOP],NPEA_COP_100*1000,[-0.35],color='k')
fmt = '%r'
plt.clabel(cs,fmt=fmt)
plt.axis('scaled')
cb.set_label('Stratification Factor ($x1000$)',rotation=270, labelpad=20, fontsize=14)
plt.title('Non-Dimensional Potential Energy Anomaly 100 m\n COP on '+str(tCOP[oktimeCOP])[0:13],\
          fontsize=16)

file = folder_fig + 'MAB_NPEA_100m_COP'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%% NPEA DOPP

latrhoDOPP = np.asarray(doppio.variables['lat_rho'][oketaDOPP,okxiDOPP])
lonrhoDOPP = np.asarray(doppio.variables['lon_rho'][oketaDOPP,okxiDOPP])

kw = dict(levels = np.arange(-0.56,0,0.05))
fig, ax = plt.subplots(figsize=(6, 6)) 
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,levels=[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,levels=[0,10000],colors='papayawhip',alpha=0.5)
plt.contourf(lonrhoDOPP,latrhoDOPP,NPEA_DOPP_100*1000,cmap=cmocean.cm.deep_r)#,**kw)
cb = plt.colorbar()
cs = plt.contour(lonrhoDOPP,latrhoDOPP,NPEA_DOPP_100*1000,[-0.35],color='k')
fmt = '%r'
plt.clabel(cs,fmt=fmt)
plt.axis('scaled')
cb.set_label('Stratification Factor ($x1000$)',rotation=270, labelpad=20, fontsize=14)
plt.title('Non-Dimensional Potential Energy Anomaly 100 m\n COP on '+str(tCOP[oktimeCOP])[0:13],\
          fontsize=16)

file = folder_fig + 'MAB_NPEA_100m_DOPP'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)      
    
#%% Calculate non-dimensional potential Energy Anomaly for GOFS 3.1 
'''    
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
'''

#%%

#Non_Dim_PEA(tempDOPP_100,saltDOPP_100,densDOPP_100,depthDOPP)

temp = tempDOPP_100
salt = saltDOPP_100
dens = densDOPP_100
depth = depthDOPP

land = np.isnan(dens)

if depth.ndim == 1:
    zz = depth
    zs = np.tile(zz,(1,temp.shape[1]))
    zss = np.tile(zs,(1,temp.shape[2]))
    zmat = np.reshape(zss,(temp.shape[0],temp.shape[1],temp.shape[2]), order='F')
    zmat[land] = np.nan

if depth.ndim == 3:
    zmat = depth
 
sigma = np.empty((temp.shape[0],temp.shape[1],temp.shape[2]))
sigma[:] = -1.0
for x,xx in enumerate(np.arange(temp.shape[1])):
    for y,yy in enumerate(np.arange(temp.shape[2])):
        sigma[:,x,y] = -zmat[:,x,y]/np.nanmax(np.abs(zmat[:,x,y]))
        sigma[np.isnan(sigma[:,x,y]),x,y] = -1

denss = np.copy(dens)
denss[np.isnan(denss)] = 0        
rhomean = -np.trapz(denss,sigma,axis=0)
lands = np.isnan(dens[0,:,:])
rhomean[lands] = np.nan

drho = np.empty((temp.shape[0],temp.shape[1],temp.shape[2]))
drho[:] = np.nan
for k,dind in enumerate(np.arange(temp.shape[0])):
    drho[k,:,:] = (rhomean - dens[k,:,:])/rhomean
drho[land] = 0

torque = drho * sigma

NPEA = np.trapz(torque,sigma,axis=0)
NPEA[lands] = np.nan

