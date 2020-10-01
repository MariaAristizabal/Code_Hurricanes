#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Aug 25 11:17:30 2020

@author: aristizabal
"""

id_storm = '13l'
name = 'laura'
cycle = '2020082500'

file_track = '/home/aristizabal/HWRF_POM_' + id_storm +  '_' + cycle[0:4] + '/HWRF_POM_' + \
             id_storm + '_' + cycle + '/' + name + id_storm + '.' + cycle + '.trak.hwrf.atcfunix'

temp_lim = [25,31.6]
salt_lim = [31,37.1]
temp200_lim = [5,24.6]
salt200_lim = [35.5,37.6]
tempb_lim = [0,25.6]
tempt_lim = [6,31.1]

folder_fig = '/home/aristizabal/Figures/'

#GOFS3.1 output model location
url_GOFS_ts = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/ts3z'

url_GOFS_uv = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z'

url_GOFS_ssh = 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/ssh'

# Bathymetry file
bath_file = '/home/aristizabal/bathymetry_files/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

# RTOFS files
folder_RTOFS = '/home/coolgroup/RTOFS/forecasts/domains/hurricanes/RTOFS_6hourly_North_Atlantic/'

nc_files_RTOFS = ['rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f012_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f018_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f024_6hrly_hvr_US_east.nc']

#%%
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import netCDF4
from datetime import datetime, timedelta
import cmocean
import matplotlib.dates as mdates

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%%
year = int(cycle[0:4])
month = int(cycle[4:6])
day = int(cycle[6:8])
hour = int(cycle[8:10])
tini = datetime(year,month,day,hour) + timedelta(hours=6)

#%% Read track files 
ff_oper = open(file_track,'r')
f_oper = ff_oper.readlines()

latt = []
lonn = []
intt = []
lead_time = []
for l in f_oper:
    lat = float(l.split(',')[6][0:4])/10
    if l.split(',')[6][4] == 'N':
        lat = lat
    else:
        lat = -lat
    lon = float(l.split(',')[7][0:5])/10
    if l.split(',')[7][4] == 'E':
        lon = lon
    else:
        lon = -lon
    latt.append(lat)
    lonn.append(lon)
    intt.append(float(l.split(',')[8]))
    lead_time.append(int(l.split(',')[5][1:4]))

latt = np.asarray(latt)
lonn = np.asarray(lonn)
intt = np.asarray(intt)
lead_time_track, ind = np.unique(lead_time,return_index=True)
lat_forec_track = latt[ind]
lon_forec_track = lonn[ind]
int_forec_track = intt[ind]

lon_forec_cone = []
lat_forec_cone = []
lon_best_track = []
lat_best_track = []

lon_lim = [np.min(lon_forec_track)-5,np.max(lon_forec_track)+5]
lat_lim = [np.min(lat_forec_track)-5,np.max(lat_forec_track)+5]

#%% GOGF 3.1
GOFS_ts = xr.open_dataset(url_GOFS_ts,decode_times=False)
GOFS_uv = xr.open_dataset(url_GOFS_uv,decode_times=False)
GOFS_ssh = xr.open_dataset(url_GOFS_ssh,decode_times=False)

lt_GOFS = np.asarray(GOFS_ts['lat'][:])
ln_GOFS = np.asarray(GOFS_ts['lon'][:])
tt = GOFS_ts['time']
t_GOFS = netCDF4.num2date(tt[:],tt.units)

depth_GOFS = np.asarray(GOFS_ts['depth'][:])

# Conversion from glider longitude and latitude to GOFS convention
lon_limG = np.empty((len(lon_lim),))
lon_limG[:] = np.nan
for i in range(len(lon_lim)):
    if lon_lim[i] < 0:
        lon_limG[i] = 360 + lon_lim[i]
    else:
        lon_limG[i] = lon_lim[i]
    lat_limG = lat_lim

oklon_GOFS = np.where(np.logical_and(ln_GOFS >= lon_limG[0],ln_GOFS <= lon_limG[1]))[0]
oklat_GOFS = np.where(np.logical_and(lt_GOFS >= lat_limG[0],lt_GOFS <= lat_lim[1]))[0]

ttGOFS = np.asarray([datetime(t_GOFS[i].year,t_GOFS[i].month,t_GOFS[i].day,t_GOFS[i].hour) for i in np.arange(len(t_GOFS))])
tstamp_GOFS = [mdates.date2num(ttGOFS[i]) for i in np.arange(len(ttGOFS))]
oktime_GOFS = np.unique(np.round(np.interp(mdates.date2num(tini),tstamp_GOFS,np.arange(len(tstamp_GOFS)))).astype(int))
time_GOFS = ttGOFS[oktime_GOFS][0]

# Conversion from GOFS convention to glider longitude and latitude
ln_GOFSg= np.empty((len(ln_GOFS),))
ln_GOFSg[:] = np.nan
for i in range(len(ln_GOFS)):
    if ln_GOFS[i] > 180:
        ln_GOFSg[i] = ln_GOFS[i] - 360
    else:
        ln_GOFSg[i] = ln_GOFS[i]
lt_GOFSg = lt_GOFS

lat_GOFS= lt_GOFS[oklat_GOFS]
lon_GOFS= ln_GOFS[oklon_GOFS]
lon_GOFSg= ln_GOFSg[oklon_GOFS]
lat_GOFSg= lt_GOFSg[oklat_GOFS]

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
    
#%% loading surface temperature and salinity
sst_GOFS = np.asarray(GOFS_ts['water_temp'][oktime_GOFS,0,oklat_GOFS,oklon_GOFS])[0,:,:]
sss_GOFS = np.asarray(GOFS_ts['salinity'][oktime_GOFS,0,oklat_GOFS,oklon_GOFS])[0,:,:]
ssh_GOFS = np.asarray(GOFS_ssh['surf_el'][oktime_GOFS,oklat_GOFS,oklon_GOFS])[0,:,:]
su_GOFS = np.asarray(GOFS_uv['water_u'][oktime_GOFS,0,oklat_GOFS,oklon_GOFS])[0,:,:]
sv_GOFS = np.asarray(GOFS_uv['water_v'][oktime_GOFS,0,oklat_GOFS,oklon_GOFS])[0,:,:]

#%% Read RTOFS files

year = int(cycle[0:4])
month = int(cycle[4:6])
day = int(cycle[6:8])
hour = int(cycle[8:10])
tini = datetime(year,month,day,hour)

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

ncRTOFS = xr.open_dataset(folder_RTOFS + fol + '/' + nc_files_RTOFS[1])
latRTOFS = np.asarray(ncRTOFS.Latitude[:])
lonRTOFS = np.asarray(ncRTOFS.Longitude[:])
depth_RTOFS = np.asarray(ncRTOFS.Depth[:])

tRTOFS = []
for t in np.arange(len(nc_files_RTOFS)):
    ncRTOFS = xr.open_dataset(folder_RTOFS + fol + '/' + nc_files_RTOFS[t])
    tRTOFS.append(np.asarray(ncRTOFS.MT[:])[0])

tRTOFS = np.asarray([mdates.num2date(mdates.date2num(tRTOFS[t])) \
          for t in np.arange(len(nc_files_RTOFS))])

oklonRTOFS = np.where(np.logical_and(lonRTOFS[0,:] >= lon_lim[0],lonRTOFS[0,:] <= lon_lim[1]))[0]
oklatRTOFS = np.where(np.logical_and(latRTOFS[:,0] >= lat_lim[0],latRTOFS[:,0] <= lat_lim[1]))[0]

t=3
nc_file = folder_RTOFS + fol + '/' + nc_files_RTOFS[t]
ncRTOFS = xr.open_dataset(nc_file)
time_RTOFS = tRTOFS[t]
lon_RTOFS = lonRTOFS[0,oklonRTOFS]
lat_RTOFS = latRTOFS[oklatRTOFS,0]
sst_RTOFS = np.asarray(ncRTOFS.variables['temperature'][0,0,oklatRTOFS,oklonRTOFS])
sss_RTOFS = np.asarray(ncRTOFS.variables['salinity'][0,0,oklatRTOFS,oklonRTOFS])
su_RTOFS = np.asarray(ncRTOFS.variables['u'][0,0,oklatRTOFS,oklonRTOFS])
sv_RTOFS = np.asarray(ncRTOFS.variables['v'][0,0,oklatRTOFS,oklonRTOFS])

#%% Figure sst
kw = dict(levels = np.arange(temp_lim[0],temp_lim[1],0.5))

plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
plt.contourf(lon_GOFSg,lat_GOFSg,sst_GOFS[:,:],cmap=cmocean.cm.thermal,**kw)
plt.plot(lon_forec_track,lat_forec_track,'.-',color='grey')
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14,labelpad=15)
plt.axis('scaled')
plt.xlim(lon_lim[0],lon_lim[1])
plt.ylim(lat_lim[0],lat_lim[1])
plt.title('GOFS SST \n on '+str(time_GOFS)[0:13],fontsize=16)

file = folder_fig +'GOFS_SST'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Figure sss
kw = dict(levels = np.arange(salt_lim[0],salt_lim[1],0.5))

plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
plt.contourf(lon_GOFSg,lat_GOFSg,sss_GOFS,cmap=cmocean.cm.haline,**kw)
plt.plot(lon_forec_track,lat_forec_track,'.-',color='grey')
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
plt.axis('scaled')
plt.xlim(lon_lim[0],lon_lim[1])
plt.ylim(lat_lim[0],lat_lim[1])
plt.title('GOFS SSS \n on '+str(time_GOFS)[0:13],fontsize=16)

file = folder_fig + 'GOFS_SSS'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Figure ssh
kw = dict(levels = np.arange(-1.0,1.1,0.1))

plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
plt.contourf(lon_GOFSg,lat_GOFSg,ssh_GOFS,cmap=cmocean.cm.curl,**kw)
plt.plot(lon_forec_track,lat_forec_track,'.-',color='grey')
plt.plot(lon_forec_cone,lat_forec_cone,'.-b',markersize=1)
plt.plot(lon_best_track,lat_best_track,'or',markersize=3)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('meters',fontsize=14)
plt.axis('scaled')
plt.xlim(lon_lim[0],lon_lim[1])
plt.ylim(lat_lim[0],lat_lim[1])
plt.title('GOFS SSH \n on '+str(time_GOFS)[0:13],fontsize=16)

file = folder_fig + 'GOFS_SSH'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Figure ssh
kw = dict(levels = np.arange(-1.0,1.1,0.1))

plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
plt.contourf(lon_GOFSg,lat_GOFSg,ssh_GOFS,cmap=cmocean.cm.curl,**kw)
plt.plot(lon_forec_track,lat_forec_track,'.-',color='grey')
plt.plot(lon_forec_cone,lat_forec_cone,'.-b',markersize=1)
plt.plot(lon_best_track,lat_best_track,'or',markersize=3)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('meters',fontsize=14)
plt.axis('scaled')
plt.xlim(lon_lim[0],lon_lim[1])
plt.ylim(lat_lim[0],lat_lim[1])
plt.title('GOFS SSH \n on '+str(time_GOFS)[0:13],fontsize=16)
q=plt.quiver(lon_GOFSg[::7],lat_GOFSg[::7],su_GOFS[::7,::7],sv_GOFS[::7,::7] ,scale=3,scale_units='inches',\
          alpha=0.7)
plt.quiverkey(q,np.max(lon_GOFSg)-0.2,np.max(lat_GOFSg)+0.5,1,"1 m/s",coordinates='data',color='k',fontproperties={'size': 14})

file = folder_fig + 'GOFS_SSH_UV'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Figure temp transect along storm path
lon_forec_track_interp = np.interp(lt_GOFS,lat_forec_track,lon_forec_track,left=np.nan,right=np.nan)
lat_forec_track_interp = np.copy(lt_GOFS)
lat_forec_track_interp[np.isnan(lon_forec_track_interp)] = np.nan

lon_forec_track_int = lon_forec_track_interp[np.isfinite(lon_forec_track_interp)]
lat_forec_track_int = lat_forec_track_interp[np.isfinite(lat_forec_track_interp)]

oklon = np.round(np.interp(lon_forec_track_int+360,ln_GOFS,np.arange(len(ln_GOFS)))).astype(int)
oklat = np.round(np.interp(lat_forec_track_int,lt_GOFS,np.arange(len(lt_GOFS)))).astype(int)
okdepth = np.where(depth_GOFS <= 350)[0]

#i=0
trans_temp_GOFS = np.empty((len(depth_GOFS[okdepth]),len(lon_forec_track_int)))
trans_temp_GOFS[:] = np.nan
for x in np.arange(len(lon_forec_track_int)):
    trans_temp_GOFS[:,x] = np.asarray(GOFS_ts['water_temp'][oktime_GOFS,okdepth,oklat[x],oklon[x]])

kw = dict(levels = np.arange(tempt_lim[0],tempt_lim[1],1))

plt.figure()
plt.contourf(lt_GOFS[oklat],-depth_GOFS[okdepth],trans_temp_GOFS,cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.contour(lt_GOFSg[oklat],-depth_GOFS[okdepth],trans_temp_GOFS,[26],color='k')
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.ylabel('Depth (m)',fontsize=14)
plt.xlabel('Latitude ($^o$)',fontsize=14)
plt.title('GOFS Temp. along Forecasted Storm Track \n on '+str(time_GOFS)[0:13],fontsize=16)
plt.xlim(20.8,29.6)

file = folder_fig + 'GOFS_temp_along_forecasted_track_'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% SST
kw = dict(levels = np.arange(temp_lim[0],temp_lim[1],0.5))

fig, ax = plt.subplots()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
plt.contourf(lon_RTOFS,lat_RTOFS,sst_RTOFS,cmap=cmocean.cm.thermal,**kw)
plt.plot(lon_forec_track,lat_forec_track,'.-',color='grey')
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14,labelpad=15)
plt.axis('scaled')
plt.xlim(lon_lim[0],lon_lim[1])
plt.ylim(lat_lim[0],lat_lim[1])
plt.title('RTOFS Oper. SST \n on '+str(time_RTOFS)[0:13],fontsize=16)

file = folder_fig +'RTOFS_SST'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% SSS
kw = dict(levels = np.arange(salt_lim[0],salt_lim[1],0.5))

fig, ax = plt.subplots()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
plt.contourf(lon_RTOFS,lat_RTOFS,sss_RTOFS,cmap=cmocean.cm.haline,**kw)
plt.plot(lon_forec_track,lat_forec_track,'.-',color='grey')
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
plt.axis('scaled')
plt.xlim(lon_lim[0],lon_lim[1])
plt.ylim(lat_lim[0],lat_lim[1])
plt.title('RTOFS Oper. SSS \n on '+str(time_RTOFS)[0:13],fontsize=16)

file = folder_fig +'RTOFS_SSS'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Surface velocity
kw = dict(levels = np.arange(temp_lim[0],temp_lim[1],0.5))

plt.figure()
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
plt.contourf(lon_RTOFS,lat_RTOFS,sst_RTOFS,cmap=cmocean.cm.thermal,**kw)
plt.plot(lon_forec_track,lat_forec_track,'.-',color='grey')
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14,labelpad=15)
plt.axis('scaled')
plt.xlim(lon_lim[0],lon_lim[1])
plt.ylim(lat_lim[0],lat_lim[1])
plt.title('RTOFS SST \n on '+str(time_RTOFS)[0:13],fontsize=16)
q=plt.quiver(lon_RTOFS[::6],lat_RTOFS[::6],su_RTOFS[::6,::6],sv_RTOFS[::6,::6] ,scale=3,scale_units='inches',\
          alpha=0.7)
plt.quiverkey(q,np.max(lon_RTOFS)-0.2,np.max(lat_RTOFS)+0.5,1,"1 m/s",coordinates='data',color='k',fontproperties={'size': 14})

file = folder_fig + 'RTOFS_SST_UV'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

    #%% Figure temp transect along storm path
lon_forec_track_interp = np.interp(latRTOFS[:,0],lat_forec_track,lon_forec_track,left=np.nan,right=np.nan)
lat_forec_track_interp = np.copy(latRTOFS[:,0])
lat_forec_track_interp[np.isnan(lon_forec_track_interp)] = np.nan

lon_forec_track_int = lon_forec_track_interp[np.isfinite(lon_forec_track_interp)]
lat_forec_track_int = lat_forec_track_interp[np.isfinite(lat_forec_track_interp)]

oklon = np.round(np.interp(lon_forec_track_int,lonRTOFS[0,:],np.arange(len(lonRTOFS[0,:])))).astype(int)
oklat = np.round(np.interp(lat_forec_track_int,latRTOFS[:,0],np.arange(len(latRTOFS[:,0])))).astype(int)
okdepth = np.where(depth_RTOFS <= 350)[0]

trans_temp_RTOFS = np.empty((len(depth_RTOFS[okdepth]),len(lon_forec_track_int)))
trans_temp_RTOFS[:] = np.nan
for x in np.arange(len(lon_forec_track_int)):
    trans_temp_RTOFS[:,x] = np.asarray(ncRTOFS.variables['temperature'][0,okdepth,oklat[x],oklon[x]])

kw = dict(levels = np.arange(tempt_lim[0],tempt_lim[1],1))

plt.figure()
plt.contourf(latRTOFS[oklat,0],-depth_RTOFS[okdepth],trans_temp_RTOFS,cmap=cmocean.cm.thermal,**kw)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
plt.contour(latRTOFS[oklat,0],-depth_RTOFS[okdepth],trans_temp_RTOFS,[26],color='k')
cbar.ax.set_ylabel('($^\circ$C)',fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.ylabel('Depth (m)',fontsize=14)
plt.xlabel('Latitude ($^o$)',fontsize=14)
plt.title('RTOFS Temp. along Forecasted Storm Track \n on '+str(time_RTOFS)[0:13],fontsize=16)
plt.xlim(20.8,29.6)

file = folder_fig + 'RTOFS_temp_along_forecasted_track_'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)