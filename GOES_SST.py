#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:32:22 2019

@author: root
"""
#%%
# GOES 1 day aver
#file_GOES = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GOES-R_SST_ccc1_eb95_ff2a.nc'
file_GOES = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GOES-R_SST_7411_5acf_a553-2.nc'

GOES = xr.open_dataset(file_GOES,decode_times=False)

lat_GOES = np.asarray(GOES.latitude[:])
lon_GOES = np.asarray(GOES.longitude[:])
sst_GOES = np.asarray(GOES.SST[:]) -273.15
tt = GOES.time
time_GOES = netCDF4.num2date(tt[:],tt.units) 

#%%

ypos1 = np.round(np.interp(27.09778,lat_GOES,np.arange(len(lat_GOES)))).astype(int)
xpos1 = np.round(np.interp(-76.60314,lon_GOES,np.arange(len(lon_GOES)))).astype(int)

sst1 = sst_GOES[0,ypos1,xpos1]

#%%

ypos2 = np.round(np.interp(27.77398,lat_GOES,np.arange(len(lat_GOES)))).astype(int)
xpos2 = np.round(np.interp(-76.64989,lon_GOES,np.arange(len(lon_GOES)))).astype(int)

sst2 = sst_GOES[0,ypos2,xpos2]

#%% Reading bathymetry data

lat_lim = [25,30]
lon_lim = [-80,-75]

ncbath = Dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[-1])
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[-1])

bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath]

#%%
kw = dict(levels = np.linspace(0,32,33))
plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(lon_GOES,lat_GOES,sst_GOES[0,:,:],cmap='RdBu_r',**kw)
plt.colorbar()
#plt.plot(-76.60314,27.09778,'*k')
plt.plot(-76.64989,27.77398,'*k')
plt.title('GOES-16 Hourly 2km SST at ' + str(time_GOES[0]),fontsize=14)
