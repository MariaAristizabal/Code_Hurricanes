#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:18:38 2019

@author: root
"""

#%% User input

#lon_lim = [-100.0,-55.0]
#lat_lim = [10.0,45.0]

lon_lim = [-80.0,-60.0]
lat_lim = [15.0,35.0]

# Server erddap url IOOS glider dap
server = 'https://data.ioos.us/gliders/erddap'

#gliders sg666, sg665, sg668, silbo
gdata_ng665 = 'http://gliders.ioos.us/thredds/dodsC/deployments/aoml/SG665-20190718T1155/SG665-20190718T1155.nc3.nc'
#gdata_ng666 = 'http://gliders.ioos.us/thredds/dodsC/deployments/aoml/SG666-20190718T1206/SG666-20190718T1206.nc3.nc'
#gdata_ng668 = 'http://gliders.ioos.us/thredds/dodsC/deployments/aoml/SG668-20190819T1217/SG668-20190819T1217.nc3.nc'
#gdata_silbo ='http://gliders.ioos.us/thredds/dodsC/deployments/rutgers/silbo-20190717T1917/silbo-20190717T1917.nc3.nc'

#Time window
date_ini = '2019/08/28/00/00'
date_end = '2019/09/02/00/00'

# url for GOFS 3.1
url_GOFS = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'

# figures
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

# POM output
folder_pom = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/POM_Dorian_npz_files/'
folder_pom_grid = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/'
pom_grid_oper = folder_pom_grid + 'dorian05l.2019082800.pom.grid.oper.nc'
pom_grid_exp = folder_pom_grid + 'dorian05l.2019082800.pom.grid.exp.nc'

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.grid_finder import (FixedLocator,
                                                 DictFormatter)
import xarray as xr
import netCDF4
from datetime import datetime
import matplotlib.dates as mdates
import sys
import seawater as sw

sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/All_code/Remote_repos/glider_model_comparisons_Python')

from read_glider_data import read_glider_data_thredds_server
#from process_glider_data import grid_glider_data_thredd

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% POM Operational and Experimental

# Operational
POM_Dorian_2019082800_oper = np.load(folder_pom + 'pom_oper_Doria_2019082800.npz')
POM_Dorian_2019082800_oper.files

timestamp_pom_oper = POM_Dorian_2019082800_oper['timestamp_pom_oper']
temp_pom_oper = POM_Dorian_2019082800_oper['temp_pom_oper']
salt_pom_oper = POM_Dorian_2019082800_oper['salt_pom_oper']
rho_pom_oper = POM_Dorian_2019082800_oper['rho_pom_oper']

# Experimental
POM_Dorian_2019082800_exp = np.load(folder_pom + 'pom_exp_Doria_2019082800.npz')
POM_Dorian_2019082800_exp.files

timestamp_pom_exp = POM_Dorian_2019082800_exp['timestamp_pom_exp']
temp_pom_exp = POM_Dorian_2019082800_exp['temp_pom_exp']
salt_pom_exp = POM_Dorian_2019082800_exp['salt_pom_exp']
rho_pom_exp = POM_Dorian_2019082800_exp['rho_pom_exp']

#%% Read POM grid

print('Retrieving coordinates from POM')
POM_grid_oper = xr.open_dataset(pom_grid_oper,decode_times=False)
lon_pom_oper = np.asarray(POM_grid_oper['east_e'][:])
lat_pom_oper = np.asarray(POM_grid_oper['north_e'][:])
zlev_pom_oper = np.asarray(POM_grid_oper['zz'][:])
hpom_oper = np.asarray(POM_grid_oper['h'][:])
zmatrix = np.dot(hpom_oper.reshape(-1,1),zlev_pom_oper.reshape(1,-1))
zmatrix_pom_oper = zmatrix.reshape(hpom_oper.shape[0],hpom_oper.shape[1],zlev_pom_oper.shape[0])

POM_grid_exp = xr.open_dataset(pom_grid_exp,decode_times=False)
lon_pom_exp = np.asarray(POM_grid_exp['east_e'][:])
lat_pom_exp = np.asarray(POM_grid_exp['north_e'][:])
zlev_pom_exp = np.asarray(POM_grid_exp['zz'][:])
hpom_exp = np.asarray(POM_grid_exp['h'][:])
zmatrix = np.dot(hpom_exp.reshape(-1,1),zlev_pom_exp.reshape(1,-1))
zmatrix_pom_exp = zmatrix.reshape(hpom_exp.shape[0],hpom_exp.shape[1],zlev_pom_exp.shape[0])

#%% Read GOFS 3.1 grid

print('Retrieving coordinates from GOFS')
GOFS = xr.open_dataset(url_GOFS,decode_times=False)

tt_G = GOFS.time
t_G = netCDF4.num2date(tt_G[:],tt_G.units)

tmin = datetime.strptime(date_ini[0:-3],'%Y/%m/%d/%H')
tmax = datetime.strptime(date_end[0:-3],'%Y/%m/%d/%H')
oktime_GOFS = np.where(np.logical_and(t_G >= tmin, t_G <= tmax))
time_GOFS = np.asarray(t_G[oktime_GOFS])
timestamp_GOFS = mdates.date2num(time_GOFS)

lat_G = np.asarray(GOFS.lat[:])
lon_G = np.asarray(GOFS.lon[:])

# Conversion from glider longitude and latitude to GOFS convention
lon_limG = np.empty((len(lon_lim),))
lon_limG[:] = np.nan
for i in range(len(lon_lim)):
    if lon_lim[i] < 0:
        lon_limG[i] = 360 + lon_lim[i]
    else:
        lon_limG[i] = lon_lim[i]
lat_limG = lat_lim

oklat_GOFS = np.where(np.logical_and(lat_G >= lat_limG[0], lat_G <= lat_limG[1]))
oklon_GOFS = np.where(np.logical_and(lon_G >= lon_limG[0], lon_G <= lon_limG[1]))

lat_GOFS = lat_G[oklat_GOFS]
lon_GOFS = lon_G[oklon_GOFS]

depth_GOFS = np.asarray(GOFS.depth[:])

# Conversion from GOFS convention to glider longitude and latitude
lon_GOFSg= np.empty((len(lon_GOFS),))
lon_GOFSg[:] = np.nan
for i in range(len(lon_GOFS)):
    if lon_GOFS[i] > 180:
        lon_GOFSg[i] = lon_GOFS[i] - 360
    else:
        lon_GOFSg[i] = lon_GOFS[i]
lat_GOFSg = lat_GOFS

#%% Reading glider data

url_glider = gdata_ng665
#url_glider = gdata_ng666
#url_glider = gdata_ng668
#url_glider = gdata_silbo

var = 'temperature'

scatter_plot = 'no'
kwargs = dict(date_ini=date_ini[0:-3],date_end=date_end[0:-3])

varg, latg, long, depthg, timeg, inst_id = \
             read_glider_data_thredds_server(url_glider,var,scatter_plot,**kwargs)

tempg = varg

var = 'salinity'
varg, latg, long, depthg, timeg, inst_id = \
             read_glider_data_thredds_server(url_glider,var,scatter_plot,**kwargs)


saltg = varg

var = 'density'
varg, latg, long, depthg, timeg, inst_id = \
             read_glider_data_thredds_server(url_glider,var,scatter_plot,**kwargs)


densg = varg
depthg = depthg

#%% Grid glider variables according to depth

depthg_gridded = np.arange(0,np.nanmax(depthg),0.5)
tempg_gridded = np.empty((len(depthg_gridded),len(timeg)))
tempg_gridded[:] = np.nan
saltg_gridded = np.empty((len(depthg_gridded),len(timeg)))
saltg_gridded[:] = np.nan
densg_gridded = np.empty((len(depthg_gridded),len(timeg)))
densg_gridded[:] = np.nan

for t,tt in enumerate(timeg):
    depthu,oku = np.unique(depthg[:,t],return_index=True)
    tempu = tempg[oku,t]
    saltu = saltg[oku,t]
    densu = densg[oku,t]
    okdd = np.isfinite(depthu)
    depthf = depthu[okdd]
    tempf = tempu[okdd]
    saltf = saltu[okdd]
    densf = densu[okdd]

    okt = np.isfinite(tempf)
    if np.sum(okt) < 3:
        tempg_gridded[:,t] = np.nan
    else:
        okd = np.logical_and(depthg_gridded >= np.min(depthf[okt]),\
                            depthg_gridded < np.max(depthf[okt]))
        tempg_gridded[okd,t] = np.interp(depthg_gridded[okd],depthf[okt],tempf[okt])

    oks = np.isfinite(saltf)
    if np.sum(oks) < 3:
        saltg_gridded[:,t] = np.nan
    else:
        okd = np.logical_and(depthg_gridded >= np.min(depthf[okt]),\
                            depthg_gridded < np.max(depthf[okt]))
        saltg_gridded[okd,t] = np.interp(depthg_gridded[okd],depthf[oks],saltf[oks])

    okdd = np.isfinite(densf)
    if np.sum(okdd) < 3:
        densg_gridded[:,t] = np.nan
    else:
        okd = np.logical_and(depthg_gridded >= np.min(depthf[okdd]),\
                            depthg_gridded < np.max(depthf[okdd]))
        densg_gridded[okd,t] = np.interp(depthg_gridded[okd],depthf[okdd],densf[okdd])


#%%

# Conversion from glider longitude and latitude to GOFS convention
target_lon = np.empty((len(long),))
target_lon[:] = np.nan
for i,ii in enumerate(long):
    if ii < 0:
        target_lon[i] = 360 + ii
    else:
        target_lon[i] = ii
target_lat = latg

# Changing times to timestamp
tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]
tstamp_model = [mdates.date2num(time_GOFS[i]) for i in np.arange(len(time_GOFS))]

# interpolating glider lon and lat to lat and lon on model time
sublon_GOFS = np.interp(tstamp_model,tstamp_glider,target_lon)
sublat_GOFS = np.interp(tstamp_model,tstamp_glider,target_lat)

# Conversion from GOFS convention to glider longitude and latitude
sublon_GOFSg = np.empty((len(sublon_GOFS),))
sublon_GOFSg[:] = np.nan
for i in range(len(sublon_GOFS)):
    if sublon_GOFS[i] > 180:
        sublon_GOFSg[i] = sublon_GOFS[i] - 360
    else:
        sublon_GOFSg[i] = sublon_GOFS[i]
sublat_GOFSg = sublat_GOFS

# getting the model grid positions for sublonm and sublatm
oklon_GOFS = np.round(np.interp(sublon_GOFS,lon_G,np.arange(len(lon_G)))).astype(int)
oklat_GOFS = np.round(np.interp(sublat_GOFS,lat_G,np.arange(len(lat_G)))).astype(int)

# Getting glider transect from model
print('Getting glider transect from model. If it breaks is because GOFS 3.1 server is not responding')
target_temp_GOFS = np.empty((len(depth_GOFS),len(oktime_GOFS[0])))
target_temp_GOFS[:] = np.nan
target_salt_GOFS = np.empty((len(depth_GOFS),len(oktime_GOFS[0])))
target_salt_GOFS[:] = np.nan
for i in range(len(oktime_GOFS[0])):
    print(len(oktime_GOFS[0]),' ',i)
    target_temp_GOFS[:,i] = GOFS.variables['water_temp'][oktime_GOFS[0][i],:,oklat_GOFS[i],oklon_GOFS[i]]
    target_salt_GOFS[:,i] = GOFS.variables['salinity'][oktime_GOFS[0][i],:,oklat_GOFS[i],oklon_GOFS[i]]

#%% Calculate density for GOFS

target_dens_GOFS = sw.dens(target_salt_GOFS,target_temp_GOFS,np.tile(depth_GOFS,(len(time_GOFS),1)).T)

#%% Retrieve glider transect from POM operational

# Changing times to timestamp
tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]

# interpolating glider lon and lat to lat and lon on model time
sublon_pom = np.interp(timestamp_pom_oper,tstamp_glider,long)
sublat_pom = np.interp(timestamp_pom_oper,tstamp_glider,latg)

# getting the model grid positions for sublonm and sublatm
oklon_pom = np.round(np.interp(sublon_pom,lon_pom_oper[0,:],np.arange(len(lon_pom_oper[0,:])))).astype(int)
oklat_pom = np.round(np.interp(sublat_pom,lat_pom_oper[:,0],np.arange(len(lat_pom_oper[:,0])))).astype(int)

# Getting glider transect from model
target_temp_POM_oper = np.empty((len(zlev_pom_oper),len(timestamp_pom_oper)))
target_temp_POM_oper[:] = np.nan
target_salt_POM_oper = np.empty((len(zlev_pom_oper),len(timestamp_pom_oper)))
target_salt_POM_oper[:] = np.nan
target_rho_POM_oper = np.empty((len(zlev_pom_oper),len(timestamp_pom_oper)))
target_rho_POM_oper[:] = np.nan
for i in range(len(timestamp_pom_oper)):
    print(len(timestamp_pom_oper),' ',i)
    target_temp_POM_oper[:,i] = temp_pom_oper[i,:,oklat_pom[i],oklon_pom[i]]
    target_salt_POM_oper[:,i] = salt_pom_oper[i,:,oklat_pom[i],oklon_pom[i]]
    target_rho_POM_oper[:,i] = rho_pom_oper[i,:,oklat_pom[i],oklon_pom[i]]

target_dens_POM_oper = target_rho_POM_oper * 1000 + 1000
target_dens_POM_oper[target_dens_POM_oper == 1000.0] = np.nan
target_depth_POM_oper = zmatrix_pom_oper[oklat_pom,oklon_pom,:].T

#%% Retrieve glider transect from POM experimental

# Changing times to timestamp
tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]

# interpolating glider lon and lat to lat and lon on model time
sublon_pom = np.interp(timestamp_pom_exp,tstamp_glider,long)
sublat_pom = np.interp(timestamp_pom_exp,tstamp_glider,latg)

# getting the model grid positions for sublonm and sublatm
oklon_pom = np.round(np.interp(sublon_pom,lon_pom_exp[0,:],np.arange(len(lon_pom_exp[0,:])))).astype(int)
oklat_pom = np.round(np.interp(sublat_pom,lat_pom_exp[:,0],np.arange(len(lat_pom_exp[:,0])))).astype(int)

# Getting glider transect from model
target_temp_POM_exp = np.empty((len(zlev_pom_exp),len(timestamp_pom_exp)))
target_temp_POM_exp[:] = np.nan
target_salt_POM_exp = np.empty((len(zlev_pom_exp),len(timestamp_pom_exp)))
target_salt_POM_exp[:] = np.nan
target_rho_POM_exp = np.empty((len(zlev_pom_exp),len(timestamp_pom_exp)))
target_rho_POM_exp[:] = np.nan
for i in range(len(timestamp_pom_exp)):
    print(len(timestamp_pom_exp),' ',i)
    target_temp_POM_exp[:,i] = temp_pom_exp[i,:,oklat_pom[i],oklon_pom[i]]
    target_salt_POM_exp[:,i] = salt_pom_exp[i,:,oklat_pom[i],oklon_pom[i]]
    target_rho_POM_exp[:,i] = rho_pom_exp[i,:,oklat_pom[i],oklon_pom[i]]

target_dens_POM_exp = target_rho_POM_exp * 1000 + 1000
target_dens_POM_exp[target_dens_POM_exp == 1000.0] = np.nan
target_depth_POM_exp = zmatrix_pom_exp[oklat_pom,oklon_pom,:].T

#%%  Calculation of mixed layer depth based on dt, Tmean: mean temp within the
# mixed layer and td: temp at 1 meter below the mixed layer
# for glider data

d10 = np.where(depthg_gridded >= 10)[0][0]
dt = 0.2

MLD_dt = np.empty(len(timeg))
MLD_dt[:] = np.nan
Tmean_dtemp = np.empty(len(timeg))
Tmean_dtemp[:] = np.nan
Smean_dtemp = np.empty(len(timeg))
Smean_dtemp[:] = np.nan
Td = np.empty(len(timeg))
Td[:] = np.nan
for t,tt in enumerate(timeg):
    T10 = tempg_gridded[d10,t]
    delta_T = T10 - tempg_gridded[:,t]
    ok_mld = np.where(delta_T <= dt)[0]
    if ok_mld.size == 0:
        MLD_dt[t] = np.nan
        Tmean_dtemp[t] = np.nan
        Smean_dtemp[t] = np.nan
        Td[t] = np.nan
    else:
        ok_mld_plus1m = np.where(depthg_gridded >= depthg_gridded[ok_mld[-1]] + 1)[0][0]
        MLD_dt[t] = depthg_gridded[ok_mld[-1]]
        Tmean_dtemp[t] = np.nanmean(tempg_gridded[ok_mld,t])
        Smean_dtemp[t] = np.nanmean(saltg_gridded[ok_mld,t])
        Td[t] = tempg_gridded[ok_mld_plus1m,t]

#%%  Calculation of mixed layer depth based on dt, Tmean: mean temp within the
# mixed layer and td: temp at 1 meter below the mixed layer
# for GOFS 3.1 output

d10 = np.where(depth_GOFS >= 10)[0][0]
dt = 0.2

MLD_dt_GOFS = np.empty(len(time_GOFS))
MLD_dt_GOFS[:] = np.nan
Tmean_dtemp_GOFS = np.empty(len(time_GOFS))
Tmean_dtemp_GOFS[:] = np.nan
Smean_dtemp_GOFS = np.empty(len(time_GOFS))
Smean_dtemp_GOFS[:] = np.nan
Td_GOFS = np.empty(len(time_GOFS))
Td_GOFS[:] = np.nan
for t,tt in enumerate(time_GOFS):
    T10 = target_temp_GOFS[d10,t]
    delta_T = T10 - target_temp_GOFS[:,t]
    ok_mld = np.where(delta_T <= dt)[0]
    if ok_mld.size == 0:
        MLD_dt_GOFS[t] = np.nan
        Tmean_dtemp_GOFS[t] = np.nan
        Smean_dtemp_GOFS[t] = np.nan
        Td_GOFS[t] = np.nan
    else:
        ok_mld_plus1m = np.where(depth_GOFS >= depth_GOFS[ok_mld[-1]] + 1)[0][0]
        MLD_dt_GOFS[t] = depth_GOFS[ok_mld[-1]]
        Tmean_dtemp_GOFS[t] = np.nanmean(target_temp_GOFS[ok_mld,t])
        Smean_dtemp_GOFS[t] = np.nanmean(target_salt_GOFS[ok_mld,t])
        Td_GOFS[t] = target_temp_GOFS[ok_mld_plus1m,t]

#%%  Calculation of mixed layer depth based on drho
# for glider data

d10 = np.where(depthg_gridded >= 10)[0][0]
drho = 0.125

MLD_drho = np.empty(len(timeg))
MLD_drho[:] = np.nan
Tmean_drho = np.empty(len(timeg))
Tmean_drho[:] = np.nan
Smean_drho = np.empty(len(timeg))
Smean_drho[:] = np.nan
for t,tt in enumerate(timeg):
    rho10 = densg_gridded[d10,t]
    delta_rho = -(rho10 - densg_gridded[:,t])
    ok_mld = np.where(delta_rho <= drho)
    if ok_mld[0].size == 0:
        MLD_drho[t] = np.nan
        Tmean_drho[t] = np.nan
        Smean_drho[t] = np.nan
    else:
        MLD_drho[t] = depthg_gridded[ok_mld[0][-1]]
        Tmean_drho[t] = np.nanmean(tempg_gridded[ok_mld,t])
        Smean_drho[t] = np.nanmean(saltg_gridded[ok_mld,t])

#%%  Calculation of mixed layer depth based on drho
# for GOFS 3.1 output

d10 = np.where(depth_GOFS >= 10)[0][0]
drho = 0.125

MLD_drho_GOFS = np.empty(len(time_GOFS))
MLD_drho_GOFS[:] = np.nan
Tmean_drho_GOFS = np.empty(len(time_GOFS))
Tmean_drho_GOFS[:] = np.nan
Smean_drho_GOFS = np.empty(len(time_GOFS))
Smean_drho_GOFS[:] = np.nan
for t,tt in enumerate(time_GOFS):
    rho10 = target_dens_GOFS[d10,t]
    delta_rho_GOFS = -(rho10 - target_dens_GOFS[:,t])
    ok_mld = np.where(delta_rho_GOFS <= drho)
    if ok_mld[0].size == 0:
        MLD_drho_GOFS[t] = np.nan
        Tmean_drho_GOFS[t] = np.nan
        Smean_drho_GOFS[t] = np.nan
    else:
        MLD_drho_GOFS[t] = depth_GOFS[ok_mld[0][-1]]
        Tmean_drho_GOFS[t] = np.nanmean(target_temp_GOFS[ok_mld,t])
        Smean_drho_GOFS[t] = np.nanmean(target_salt_GOFS[ok_mld,t])

#%%  Calculation of mixed layer depth based on dt, Tmean: mean temp within the
# mixed layer and td: temp at 1 meter below the mixed layer
# for POM operational

dt = 0.2

MLD_dt_POM_oper = np.empty(len(timestamp_pom_oper))
MLD_dt_POM_oper[:] = np.nan
Tmean_dtemp_POM_oper = np.empty(len(timestamp_pom_oper))
Tmean_dtemp_POM_oper[:] = np.nan
Smean_dtemp_POM_oper = np.empty(len(timestamp_pom_oper))
Smean_dtemp_POM_oper[:] = np.nan
Td_POM_oper = np.empty(len(timestamp_pom_oper))
Td_POM_oper[:] = np.nan
for t,tt in enumerate(timestamp_pom_oper):
    d10 = np.where(target_depth_POM_oper[:,t] >= -10)[0][-1]
    T10 = target_temp_POM_oper[d10,t]
    delta_T = T10 - target_temp_POM_oper[:,t]
    ok_mld = np.where(delta_T <= dt)[0]
    if ok_mld.size == 0:
        MLD_dt_POM_oper[t] = np.nan
        Tmean_dtemp_POM_oper[t] = np.nan
        Smean_dtemp_POM_oper[t] = np.nan
        Td_POM_oper[t] = np.nan
    else:
        ok_mld_plus1m = np.where(target_depth_POM_oper >= target_depth_POM_oper[ok_mld[-1]] + 1)[0][0]
        MLD_dt_POM_oper[t] = target_depth_POM_oper[ok_mld[-1],t]
        Tmean_dtemp_POM_oper[t] = np.nanmean(target_temp_POM_oper[ok_mld,t])
        Smean_dtemp_POM_oper[t] = np.nanmean(target_salt_POM_oper[ok_mld,t])
        Td_POM_oper[t] = target_temp_POM_oper[ok_mld_plus1m,t]

#%%  Calculation of mixed layer depth based on dt, Tmean: mean temp within the
# mixed layer and td: temp at 1 meter below the mixed layer
# for POM experimental

dt = 0.2

MLD_dt_POM_exp = np.empty(len(timestamp_pom_exp))
MLD_dt_POM_exp[:] = np.nan
Tmean_dtemp_POM_exp = np.empty(len(timestamp_pom_exp))
Tmean_dtemp_POM_exp[:] = np.nan
Smean_dtemp_POM_exp = np.empty(len(timestamp_pom_exp))
Smean_dtemp_POM_exp[:] = np.nan
Td_POM_exp = np.empty(len(timestamp_pom_exp))
Td_POM_exp[:] = np.nan
for t,tt in enumerate(timestamp_pom_exp):
    d10 = np.where(target_depth_POM_exp[:,t] >= -10)[0][-1]
    T10 = target_temp_POM_exp[d10,t]
    delta_T = T10 - target_temp_POM_exp[:,t]
    ok_mld = np.where(delta_T <= dt)[0]
    if ok_mld.size == 0:
        MLD_dt_POM_exp[t] = np.nan
        Tmean_dtemp_POM_exp[t] = np.nan
        Smean_dtemp_POM_exp[t] = np.nan
        Td_POM_exp[t] = np.nan
    else:
        ok_mld_plus1m = np.where(target_depth_POM_exp >= target_depth_POM_exp[ok_mld[-1]] + 1)[0][0]
        MLD_dt_POM_exp[t] = target_depth_POM_exp[ok_mld[-1],t]
        Tmean_dtemp_POM_exp[t] = np.nanmean(target_temp_POM_exp[ok_mld,t])
        Smean_dtemp_POM_exp[t] = np.nanmean(target_salt_POM_exp[ok_mld,t])
        Td_POM_exp[t] = target_temp_POM_exp[ok_mld_plus1m,t]

#%%  Calculation of mixed layer depth based on drho
# for POM operational

drho = 0.125

MLD_drho_POM_oper = np.empty(len(timestamp_pom_oper))
MLD_drho_POM_oper[:] = np.nan
Tmean_drho_POM_oper = np.empty(len(timestamp_pom_oper))
Tmean_drho_POM_oper[:] = np.nan
Smean_drho_POM_oper = np.empty(len(timestamp_pom_oper))
Smean_drho_POM_oper[:] = np.nan
for t,tt in enumerate(timestamp_pom_oper):
    d10 = np.where(target_depth_POM_oper[:,t] >= -10)[0][-1]
    rho10 = target_dens_POM_oper[d10,t]
    delta_rho_POM = -(rho10 - target_dens_POM_oper[:,t])
    ok_mld = np.where(delta_rho_POM <= drho)
    if ok_mld[0].size == 0:
        MLD_drho_POM_oper[t] = np.nan
        Tmean_drho_POM_oper[t] = np.nan
        Smean_drho_POM_oper[t] = np.nan
    else:
        MLD_drho_POM_oper[t] = target_depth_POM_oper[ok_mld[0][-1],t]
        Tmean_drho_POM_oper[t] = np.nanmean(target_temp_POM_oper[ok_mld,t])
        Smean_drho_POM_oper[t] = np.nanmean(target_salt_POM_oper[ok_mld,t])


#%%  Calculation of mixed layer depth based on drho
# for POM experimental

drho = 0.125

MLD_drho_POM_exp = np.empty(len(timestamp_pom_exp))
MLD_drho_POM_exp[:] = np.nan
Tmean_drho_POM_exp = np.empty(len(timestamp_pom_exp))
Tmean_drho_POM_exp[:] = np.nan
Smean_drho_POM_exp = np.empty(len(timestamp_pom_exp))
Smean_drho_POM_exp[:] = np.nan
for t,tt in enumerate(timestamp_pom_exp):
    d10 = np.where(target_depth_POM_exp[:,t] >= -10)[0][-1]
    rho10 = target_dens_POM_exp[d10,t]
    delta_rho_POM = -(rho10 - target_dens_POM_exp[:,t])
    ok_mld = np.where(delta_rho_POM <= drho)
    if ok_mld[0].size == 0:
        MLD_drho_POM_exp[t] = np.nan
        Tmean_drho_POM_exp[t] = np.nan
        Smean_drho_POM_exp[t] = np.nan
    else:
        MLD_drho_POM_exp[t] = target_depth_POM_exp[ok_mld[0][-1],t]
        Tmean_drho_POM_exp[t] = np.nanmean(target_temp_POM_exp[ok_mld,t])
        Smean_drho_POM_exp[t] = np.nanmean(target_salt_POM_exp[ok_mld,t])

#%% Surface Heat content for glider

# Heat capacity in J/(kg K)
cp = 3985

OHCg = np.empty((len(timeg)))
OHCg[:] = np.nan
for t,tt in enumerate(timeg):
    ok26 = tempg_gridded[:,t] >= 26
    if np.nanmin(depthg_gridded[ok26])>10:
        OHCg[t] = np.nan
    else:
        rho0 = np.nanmean(densg_gridded[ok26,t])
        OHCg[t] = cp * rho0 * np.trapz(tempg_gridded[ok26,t]-26,depthg_gridded[ok26])

#%% Surface Heat content for GOFS 3.1

# Heat capacity in J/(kg K)
cp = 3985

OHC_GOFS = np.empty((target_temp_GOFS.shape[1]))
OHC_GOFS[:] = np.nan
for t,tt in enumerate(time_GOFS):
    print(t)
    ok26 = target_temp_GOFS[:,t] >= 26
    rho0 = np.nanmean(target_dens_GOFS[ok26,t])
    OHC_GOFS[t] = cp * rho0 * np.trapz(target_temp_GOFS[ok26,t]-26,depth_GOFS[ok26])

#%% Surface Heat content for POM Operational

# Heat capacity in J/(kg K)
cp = 3985

OHC_POM_oper = np.empty((len(timestamp_pom_oper)))
OHC_POM_oper[:] = np.nan
for t,tt in enumerate(timestamp_pom_oper):
    print(t)
    ok26 = target_temp_POM_oper[:,t] >= 26
    rho0 = np.nanmean(target_dens_POM_oper[ok26,t])
    OHC_POM_oper[t] = cp * rho0 * np.trapz(target_temp_POM_oper[ok26,t]-26,-target_depth_POM_oper[ok26,t])

#%% Surface Heat content for POM Experimental

# Heat capacity in J/(kg K)
cp = 3985

OHC_POM_exp = np.empty((len(timestamp_pom_exp)))
OHC_POM_exp[:] = np.nan
for t,tt in enumerate(timestamp_pom_exp):
    print(t)
    ok26 = target_temp_POM_exp[:,t] >= 26
    rho0 = np.nanmean(target_dens_POM_exp[ok26,t])
    OHC_POM_exp[t] = cp * rho0 * np.trapz(target_temp_POM_exp[ok26,t]-26,-target_depth_POM_exp[ok26,t])

#%% Interpolate glider transect onto GOFS time and depth

oktimeg_gofs = np.round(np.interp(tstamp_model,tstamp_glider,np.arange(len(tstamp_glider)))).astype(int)

tempg_to_GOFS = np.empty((target_temp_GOFS.shape[0],target_temp_GOFS.shape[1]))
tempg_to_GOFS[:] = np.nan
saltg_to_GOFS = np.empty((target_temp_GOFS.shape[0],target_temp_GOFS.shape[1]))
saltg_to_GOFS[:] = np.nan
for i in np.arange(len(oktimeg_gofs)):
    pos = np.argsort(depthg[:,oktimeg_gofs[i]])
    tempg_to_GOFS[:,i] = np.interp(depth_GOFS,depthg[pos,oktimeg_gofs[i]],tempg[pos,oktimeg_gofs[i]])
    saltg_to_GOFS[:,i] = np.interp(depth_GOFS,depthg[pos,oktimeg_gofs[i]],saltg[pos,oktimeg_gofs[i]])

MLD_drho_to_GOFS = MLD_drho[oktimeg_gofs]
Tmean_drho_to_GOFS = Tmean_drho[oktimeg_gofs]
Smean_drho_to_GOFS = Smean_drho[oktimeg_gofs]
OHCg_to_GOFS = OHCg[oktimeg_gofs]

#%% Interpolate glider transect onto POM time and depth

oktimeg_pom_oper = np.round(np.interp(timestamp_pom_oper,tstamp_glider,np.arange(len(tstamp_glider)))).astype(int)

tempg_topom_oper = np.empty((target_temp_POM_oper.shape[0],target_temp_POM_oper.shape[1]))
tempg_topom_oper[:] = np.nan
saltg_topom_oper = np.empty((target_temp_POM_oper.shape[0],target_temp_POM_oper.shape[1]))
saltg_topom_oper[:] = np.nan
for i in np.arange(len(oktimeg_pom_oper)):
    pos = np.argsort(depthg[:,oktimeg_pom_oper[i]])
    tempg_topom_oper[:,i] = np.interp(-target_depth_POM_oper[:,i],depthg[pos,oktimeg_pom_oper[i]],tempg[pos,oktimeg_pom_oper[i]])
    saltg_topom_oper[:,i] = np.interp(-target_depth_POM_oper[:,i],depthg[pos,oktimeg_pom_oper[i]],saltg[pos,oktimeg_pom_oper[i]])

MLD_drho_topom_oper = MLD_drho[oktimeg_pom_oper]
Tmean_drho_topom_oper = Tmean_drho[oktimeg_pom_oper]
Smean_drho_to_pom_oper = Smean_drho[oktimeg_pom_oper]
OHCg_to_pom_oper = OHCg[oktimeg_pom_oper]

#%% Interpolate glider transect onto POM time and depth

oktimeg_pom_exp = np.round(np.interp(timestamp_pom_exp,tstamp_glider,np.arange(len(tstamp_glider)))).astype(int)

tempg_topom_exp = np.empty((target_temp_POM_exp.shape[0],target_temp_POM_exp.shape[1]))
tempg_topom_exp[:] = np.nan
saltg_topom_exp = np.empty((target_temp_POM_exp.shape[0],target_temp_POM_exp.shape[1]))
saltg_topom_exp[:] = np.nan
for i in np.arange(len(oktimeg_pom_exp)):
    pos = np.argsort(depthg[:,oktimeg_pom_exp[i]])
    tempg_topom_exp[:,i] = np.interp(-target_depth_POM_exp[:,i],depthg[pos,oktimeg_pom_exp[i]],tempg[pos,oktimeg_pom_exp[i]])
    saltg_topom_exp[:,i] = np.interp(-target_depth_POM_exp[:,i],depthg[pos,oktimeg_pom_exp[i]],saltg[pos,oktimeg_pom_exp[i]])

MLD_drho_topom_exp = MLD_drho[oktimeg_pom_exp]
Tmean_drho_topom_exp = Tmean_drho[oktimeg_pom_exp]
Smean_drho_to_pom_exp = Smean_drho[oktimeg_pom_exp]
OHCg_to_pom_exp = OHCg[oktimeg_pom_exp]

#%% Define dataframe

DF_GOFS_temp_salt = pd.DataFrame(data=np.array([np.ravel(tempg_to_GOFS,order='F'),\
                                      np.ravel(target_temp_GOFS,order='F'),\
                                      np.ravel(saltg_to_GOFS,order='F'),\
                                      np.ravel(target_salt_GOFS,order='F'),\
                                      ]).T,\
                  columns=['temp_obs','temp_GOFS',\
                           'salt_obs','salt_GOFS'])

#%% Define dataframe

DF_POM_temp_salt = pd.DataFrame(data=np.array([np.ravel(tempg_topom_oper,order='F'),\
                                     np.ravel(tempg_topom_exp,order='F'),\
                                     np.ravel(target_temp_POM_oper,order='F'),\
                                     np.ravel(target_temp_POM_exp,order='F'),\
                                     np.ravel(saltg_topom_oper,order='F'),\
                                     np.ravel(saltg_topom_exp,order='F'),\
                                     np.ravel(target_salt_POM_oper,order='F'),\
                                     np.ravel(target_salt_POM_exp,order='F')
                                     ]).T,\
                  columns=['temp_obs_to_oper','temp_obs_to_exp',\
                           'temp_POM_oper','temp_POM_exp',\
                           'salt_obs_to_oper','salt_obs_to_exp',\
                           'salt_POM_oper','salt_POM_exp'])

#%% Define dataframe

DF_GOFS_mld = pd.DataFrame(data=np.array([MLD_drho_to_GOFS,MLD_drho_GOFS,\
                                          Tmean_drho_to_GOFS,Tmean_drho_GOFS]).T,\
                  columns=['MLD_obs','MLD_GOFS',\
                          'Tmean_obs','Tmean_GOFS'])

#%% Define dataframe

DF_POM_mld = pd.DataFrame(data=np.array([MLD_drho_topom_oper,MLD_drho_POM_oper,\
                                         Tmean_drho_topom_oper,Tmean_drho_POM_oper,\
                                         MLD_drho_topom_exp,MLD_drho_POM_exp,\
                                         Tmean_drho_topom_exp,Tmean_drho_POM_exp]).T,\
                  columns=['MLD_obs_to_oper','MLD_POM_oper',\
                          'Tmean_obs_to_oper','Tmean_POM_oper',\
                          'MLD_obs_to_exp','MLD_POM_exp',\
                          'Tmean_obs_to_exp','Tmean_POM_exp'])

#%% DEfine dataframe

DF_GOFS_OHC = pd.DataFrame(data=np.array([OHCg_to_GOFS,OHC_GOFS]).T,\
                  columns=['OHC_obs','OHC_GOFS'])

#%% DEfine dataframe

DF_POM_OHC = pd.DataFrame(data=np.array([OHCg_to_pom_oper,OHCg_to_pom_exp,\
                                          OHC_POM_oper,OHC_POM_exp]).T,\
                  columns=['OHC_obs_to_oper','OHC_obs_to_exp',\
                           'OHC_POM_oper','OHC_POM_exp'])

#%% Temperature statistics.

DF_GOFS = DF_GOFS_temp_salt
DF_POM = DF_POM_temp_salt

NGOFS = len(DF_GOFS)-1  #For Unbiased estimmator.
NPOM = len(DF_POM)-1

cols = ['CORRELATION','OSTD','MSTD','CRMSE','BIAS']

tskill = np.empty((3,5))
tskill[:] = np.nan

#CORR
tskill[0,0] = DF_GOFS.corr()['temp_obs']['temp_GOFS']
tskill[1,0] = DF_POM.corr()['temp_obs_to_oper']['temp_POM_oper']
tskill[2,0] = DF_POM.corr()['temp_obs_to_exp']['temp_POM_exp']

#OSTD
tskill[0,1] = DF_GOFS.std().temp_obs
tskill[1,1] = DF_POM.std().temp_obs_to_oper
tskill[2,1] = DF_POM.std().temp_obs_to_exp

#MSTD
tskill[0,2] = DF_GOFS.std().temp_GOFS
tskill[1,2] = DF_POM.std().temp_POM_oper
tskill[2,2] = DF_POM.std().temp_POM_exp

#CRMSE
tskill[0,3] = np.sqrt(np.nansum(((DF_GOFS.temp_obs-DF_GOFS.mean().temp_obs)-\
                                 (DF_GOFS.temp_GOFS-DF_GOFS.mean().temp_GOFS))**2)/NGOFS)
tskill[1,3] = np.sqrt(np.nansum(((DF_POM.temp_obs_to_exp-DF_POM.mean().temp_obs_to_oper)-\
                                 (DF_POM.temp_POM_oper-DF_POM.mean().temp_POM_oper))**2)/NPOM)
tskill[2,3] = np.sqrt(np.nansum(((DF_POM.temp_obs_to_exp-DF_POM.mean().temp_obs_to_exp)-\
                                 (DF_POM.temp_POM_exp-DF_POM.mean().temp_POM_exp))**2)/NPOM)

#BIAS
tskill[0,4] = DF_GOFS.mean().temp_obs - DF_GOFS.mean().temp_GOFS
tskill[1,4] = DF_POM.mean().temp_obs_to_oper - DF_POM.mean().temp_POM_oper
tskill[2,4] = DF_POM.mean().temp_obs_to_exp - DF_POM.mean().temp_POM_exp

#color
colors = ['indianred','seagreen','darkorchid']

temp_skillscores = pd.DataFrame(tskill,
                        index=['GOFS','POM_oper','POM_exp'],
                        columns=cols)
print(temp_skillscores)

#%% Salinity statistics.

DF_GOFS = DF_GOFS_temp_salt
DF_POM = DF_POM_temp_salt

NGOFS = len(DF_GOFS)-1  #For Unbiased estimmator.
NPOM = len(DF_POM)-1

cols = ['CORRELATION','OSTD','MSTD','CRMSE','BIAS']

tskill = np.empty((3,5))
tskill[:] = np.nan

#CORR
tskill[0,0] = DF_GOFS.corr()['salt_obs']['salt_GOFS']
tskill[1,0] = DF_POM.corr()['salt_obs_to_oper']['salt_POM_oper']
tskill[2,0] = DF_POM.corr()['salt_obs_to_exp']['salt_POM_exp']

#OSTD
tskill[0,1] = DF_GOFS.std().salt_obs
tskill[1,1] = DF_POM.std().salt_obs_to_oper
tskill[2,1] = DF_POM.std().salt_obs_to_exp

#MSTD
tskill[0,2] = DF_GOFS.std().salt_GOFS
tskill[1,2] = DF_POM.std().salt_POM_oper
tskill[2,2] = DF_POM.std().salt_POM_exp

#CRMSE
tskill[0,3] = np.sqrt(np.nansum(((DF_GOFS.salt_obs-DF_GOFS.mean().salt_obs)-\
                                 (DF_GOFS.salt_GOFS-DF_GOFS.mean().salt_GOFS))**2)/NGOFS)
tskill[1,3] = np.sqrt(np.nansum(((DF_POM.salt_obs_to_exp-DF_POM.mean().salt_obs_to_oper)-\
                                 (DF_POM.salt_POM_oper-DF_POM.mean().salt_POM_oper))**2)/NPOM)
tskill[2,3] = np.sqrt(np.nansum(((DF_POM.salt_obs_to_exp-DF_POM.mean().salt_obs_to_exp)-\
                                 (DF_POM.salt_POM_exp-DF_POM.mean().salt_POM_exp))**2)/NPOM)

#BIAS
tskill[0,4] = DF_GOFS.mean().salt_obs - DF_GOFS.mean().salt_GOFS
tskill[1,4] = DF_POM.mean().salt_obs_to_oper - DF_POM.mean().salt_POM_oper
tskill[2,4] = DF_POM.mean().salt_obs_to_exp - DF_POM.mean().salt_POM_exp

#color
colors = ['indianred','seagreen','darkorchid']

salt_skillscores = pd.DataFrame(tskill,
                        index=['GOFS','POM_oper','POM_exp'],
                        columns=cols)
print(salt_skillscores)

#%% Mixed layer statistics.

DF_GOFS = DF_GOFS_mld
DF_POM = DF_POM_mld

NGOFS = len(DF_GOFS)-1  #For Unbiased estimmator.
NPOM = len(DF_POM)-1

cols = ['CORRELATION','OSTD','MSTD','CRMSE','BIAS']

tskill = np.empty((3,5))
tskill[:] = np.nan

#CORR
tskill[0,0] = DF_GOFS.corr()['Tmean_obs']['Tmean_GOFS']
tskill[1,0] = DF_POM.corr()['Tmean_obs_to_oper']['Tmean_POM_oper']
tskill[2,0] = DF_POM.corr()['Tmean_obs_to_exp']['Tmean_POM_exp']

#OSTD
tskill[0,1] = DF_GOFS.std().Tmean_obs
tskill[1,1] = DF_POM.std().Tmean_obs_to_oper
tskill[2,1] = DF_POM.std().Tmean_obs_to_exp

#MSTD
tskill[0,2] = DF_GOFS.std().Tmean_GOFS
tskill[1,2] = DF_POM.std().Tmean_POM_oper
tskill[2,2] = DF_POM.std().Tmean_POM_exp

#CRMSE
tskill[0,3] = np.sqrt(np.nansum(((DF_GOFS.Tmean_obs-DF_GOFS.mean().Tmean_obs)-\
                                 (DF_GOFS.Tmean_GOFS-DF_GOFS.mean().Tmean_GOFS))**2)/NGOFS)
tskill[1,3] = np.sqrt(np.nansum(((DF_POM.Tmean_obs_to_exp-DF_POM.mean().Tmean_obs_to_oper)-\
                                 (DF_POM.Tmean_POM_oper-DF_POM.mean().Tmean_POM_oper))**2)/NPOM)
tskill[2,3] = np.sqrt(np.nansum(((DF_POM.Tmean_obs_to_exp-DF_POM.mean().Tmean_obs_to_exp)-\
                                 (DF_POM.Tmean_POM_exp-DF_POM.mean().Tmean_POM_exp))**2)/NPOM)

#BIAS
tskill[0,4] = DF_GOFS.mean().Tmean_obs - DF_GOFS.mean().Tmean_GOFS
tskill[1,4] = DF_POM.mean().Tmean_obs_to_oper - DF_POM.mean().Tmean_POM_oper
tskill[2,4] = DF_POM.mean().Tmean_obs_to_exp - DF_POM.mean().Tmean_POM_exp

#color
colors = ['indianred','seagreen','darkorchid']

Tmean_mld_skillscores = pd.DataFrame(tskill,
                        index=['GOFS','POM_oper','POM_exp'],
                        columns=cols)
print(Tmean_mld_skillscores)

#%% OHC statistics

DF_GOFS = DF_GOFS_OHC
DF_POM = DF_POM_OHC

NGOFS = len(DF_GOFS)-1  #For Unbiased estimmator.
NPOM = len(DF_POM)-1

cols = ['CORRELATION','OSTD','MSTD','CRMSE','BIAS']

tskill = np.empty((3,5))
tskill[:] = np.nan

#CORR
tskill[0,0] = DF_GOFS.corr()['OHC_obs']['OHC_GOFS']
tskill[1,0] = DF_POM.corr()['OHC_obs_to_oper']['OHC_POM_oper']
tskill[2,0] = DF_POM.corr()['OHC_obs_to_exp']['OHC_POM_exp']

#OSTD
tskill[0,1] = DF_GOFS.std().OHC_obs
tskill[1,1] = DF_POM.std().OHC_obs_to_oper
tskill[2,1] = DF_POM.std().OHC_obs_to_exp

#MSTD
tskill[0,2] = DF_GOFS.std().OHC_GOFS
tskill[1,2] = DF_POM.std().OHC_POM_oper
tskill[2,2] = DF_POM.std().OHC_POM_exp

#CRMSE
tskill[0,3] = np.sqrt(np.nansum(((DF_GOFS.OHC_obs-DF_GOFS.mean().OHC_obs)-\
                                 (DF_GOFS.OHC_GOFS-DF_GOFS.mean().OHC_GOFS))**2)/NGOFS)
tskill[1,3] = np.sqrt(np.nansum(((DF_POM.OHC_obs_to_exp-DF_POM.mean().OHC_obs_to_oper)-\
                                 (DF_POM.OHC_POM_oper-DF_POM.mean().OHC_POM_oper))**2)/NPOM)
tskill[2,3] = np.sqrt(np.nansum(((DF_POM.OHC_obs_to_exp-DF_POM.mean().OHC_obs_to_exp)-\
                                 (DF_POM.OHC_POM_exp-DF_POM.mean().OHC_POM_exp))**2)/NPOM)

#BIAS
tskill[0,4] = DF_GOFS.mean().OHC_obs - DF_GOFS.mean().OHC_GOFS
tskill[1,4] = DF_POM.mean().OHC_obs_to_oper - DF_POM.mean().OHC_POM_oper
tskill[2,4] = DF_POM.mean().OHC_obs_to_exp - DF_POM.mean().OHC_POM_exp

#color
colors = ['indianred','seagreen','darkorchid']

OHC_skillscores = pd.DataFrame(tskill,
                        index=['GOFS','POM_oper','POM_exp'],
                        columns=cols)
print(OHC_skillscores)

#%%

def taylor_template(angle_lim):

    fig = plt.figure()
    tr = PolarAxes.PolarTransform()

    CCgrid= np.concatenate((np.arange(0,10,2.0)/10.,[0.9,0.95,0.99]))
    CCpolar=np.arccos(CCgrid)
    gf=FixedLocator(CCpolar)
    tf=DictFormatter(dict(zip(CCpolar, map(str,CCgrid))))

    STDgrid=np.arange(0,2.0,.5)
    gfs=FixedLocator(STDgrid)
    tfs=DictFormatter(dict(zip(STDgrid, map(str,STDgrid))))

    ra0, ra1 =0, angle_lim
    cz0, cz1 = 0, 2.0
    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=(ra0, ra1, cz0, cz1),
        grid_locator1=gf,
        tick_formatter1=tf,
        grid_locator2=gfs,
        tick_formatter2=tfs)

    ax1 = floating_axes.FloatingSubplot(fig, 111, grid_helper=grid_helper)
    fig.add_subplot(ax1)

    ax1.axis["top"].set_axis_direction("bottom")
    ax1.axis["top"].toggle(ticklabels=True, label=True)
    ax1.axis["top"].major_ticklabels.set_axis_direction("top")
    ax1.axis["top"].label.set_axis_direction("top")
    ax1.axis["top"].label.set_text("Correlation")
    ax1.axis['top'].label.set_size(14)

    ax1.axis["left"].set_axis_direction("bottom")
    ax1.axis["left"].label.set_text("Normalized Standard Deviation")
    ax1.axis['left'].label.set_size(14)

    ax1.axis["right"].set_axis_direction("top")
    ax1.axis["right"].toggle(ticklabels=True)
    ax1.axis["right"].major_ticklabels.set_axis_direction("left")

    ax1.axis["bottom"].set_visible(False)
    ax1 = ax1.get_aux_axes(tr)

    plt.grid(linestyle=':',alpha=0.5)

    return fig,ax1

#%% Create a plotting function for Taylor diagrams.

def taylor(scores,colors,units,angle_lim):

    fig = plt.figure()
    tr = PolarAxes.PolarTransform()

    CCgrid= np.concatenate((np.arange(0,10,2)/10.,[0.9,0.95,0.99]))
    CCpolar=np.arccos(CCgrid)
    gf=FixedLocator(CCpolar)
    tf=DictFormatter(dict(zip(CCpolar, map(str,CCgrid))))

    max_std = np.nanmax([scores.OSTD,scores.MSTD])

    if np.round(scores.OSTD[0],2)<=0.2:
        #STDgrid=np.linspace(0,np.round(scores.OSTD[0]+0.01,2),3)
        STDgrid=np.linspace(0,np.round(scores.OSTD[0]+max_std+0.02,2),3)
    if np.logical_and(np.round(scores.OSTD[0],2)>0.2,np.round(scores.OSTD[0],2)<=1):
        STDgrid=np.linspace(0,np.round(scores.OSTD[0]+0.1,2),3)
    if np.logical_and(np.round(scores.OSTD[0],2)>1,np.round(scores.OSTD[0],2)<=5):
        STDgrid=np.arange(0,np.round(scores.OSTD[0]+2,2),1)
    if np.round(scores.OSTD[0],2)>5:
        STDgrid=np.arange(0,np.round(scores.OSTD[0]+5,1),2)

    gfs=FixedLocator(STDgrid)
    tfs=DictFormatter(dict(zip(STDgrid, map(str,STDgrid))))

    ra0, ra1 =0, angle_lim
    if np.round(scores.OSTD[0],2)<=0.2:
        cz0, cz1 = 0, np.round(max_std+0.1,2)
    else:
        #cz0, cz1 = 0, np.round(scores.OSTD[0]+0.1,2)
        cz0, cz1 = 0, np.round(max_std+0.1,2)
    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=(ra0, ra1, cz0, cz1),
        grid_locator1=gf,
        tick_formatter1=tf,
        grid_locator2=gfs,
        tick_formatter2=tfs)

    ax1 = floating_axes.FloatingSubplot(fig, 111, grid_helper=grid_helper)
    fig.add_subplot(ax1)

    ax1.axis["top"].set_axis_direction("bottom")
    ax1.axis["top"].toggle(ticklabels=True, label=True)
    ax1.axis["top"].major_ticklabels.set_axis_direction("top")
    ax1.axis["top"].label.set_axis_direction("top")
    ax1.axis["top"].label.set_text("Correlation")
    ax1.axis['top'].label.set_size(14)


    ax1.axis["left"].set_axis_direction("bottom")
    ax1.axis["left"].label.set_text("Standard Deviation "+ '(' + units +')' )
    ax1.axis['left'].label.set_size(14)

    ax1.axis["right"].set_axis_direction("top")
    ax1.axis["right"].toggle(ticklabels=True)
    ax1.axis["right"].major_ticklabels.set_axis_direction("left")

    ax1.axis["bottom"].set_visible(False)
    ax1 = ax1.get_aux_axes(tr)

    plt.grid(linestyle=':',alpha=0.5)

    for i,r in enumerate(scores.iterrows()):
        theta=np.arccos(r[1].CORRELATION)
        rr=r[1].MSTD

        ax1.plot(theta,rr,'o',label=r[0],color = colors[i])

    ax1.plot(0,scores.OSTD[0],'o',label='Obs')
    plt.legend(loc='upper right',bbox_to_anchor=[1.3,1.15])
    plt.show()

    rs,ts = np.meshgrid(np.linspace(0,np.round(max_std+0.1,2)),np.linspace(0,angle_lim))

    rms = np.sqrt(scores.OSTD[0]**2 + rs**2 - 2*rs*scores.OSTD[0]*np.cos(ts))

    ax1.contour(ts, rs, rms,5,colors='0.5')
    #contours = ax1.contour(ts, rs, rms,5,colors='0.5')
    #plt.clabel(contours, inline=1, fontsize=10)
    plt.grid(linestyle=':',alpha=0.5)

    for i,r in enumerate(scores.iterrows()):
        #crmse = np.sqrt(r[1].OSTD**2 + r[1].MSTD**2 \
        #           - 2*r[1].OSTD*r[1].MSTD*r[1].CORRELATION)
        crmse = np.sqrt(scores.OSTD[0]**2 + r[1].MSTD**2 \
                   - 2*scores.OSTD[0]*r[1].MSTD*r[1].CORRELATION)
        print(crmse)
        c1 = ax1.contour(ts, rs, rms,[crmse],colors=colors[i])
        plt.clabel(c1, inline=1, fontsize=10,fmt='%1.2f')

    return fig,ax1

#%% Create a plotting function for normalized Taylor diagrams.

def taylor_normalized(scores,colors,angle_lim):

    fig = plt.figure()
    tr = PolarAxes.PolarTransform()

    CCgrid= np.concatenate((np.arange(0,10,2)/10.,[0.9,0.95,0.99]))
    CCpolar=np.arccos(CCgrid)
    gf=FixedLocator(CCpolar)
    tf=DictFormatter(dict(zip(CCpolar, map(str,CCgrid))))

    STDgrid=np.arange(0,2.0,.5)
    gfs=FixedLocator(STDgrid)
    tfs=DictFormatter(dict(zip(STDgrid, map(str,STDgrid))))

    ra0, ra1 =0, angle_lim
    cz0, cz1 = 0, 2
    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=(ra0, ra1, cz0, cz1),
        grid_locator1=gf,
        tick_formatter1=tf,
        grid_locator2=gfs,
        tick_formatter2=tfs)

    ax1 = floating_axes.FloatingSubplot(fig, 111, grid_helper=grid_helper)
    fig.add_subplot(ax1)

    ax1.axis["top"].set_axis_direction("bottom")
    ax1.axis["top"].toggle(ticklabels=True, label=True)
    ax1.axis["top"].major_ticklabels.set_axis_direction("top")
    ax1.axis["top"].label.set_axis_direction("top")
    ax1.axis["top"].label.set_text("Correlation")
    ax1.axis['top'].label.set_size(14)

    ax1.axis["left"].set_axis_direction("bottom")
    ax1.axis["left"].label.set_text("Normalized Standard Deviation")
    ax1.axis['left'].label.set_size(14)

    ax1.axis["right"].set_axis_direction("top")
    ax1.axis["right"].toggle(ticklabels=True)
    ax1.axis["right"].major_ticklabels.set_axis_direction("left")

    ax1.axis["bottom"].set_visible(False)
    ax1 = ax1.get_aux_axes(tr)

    plt.grid(linestyle=':',alpha=0.5)

    for i,r in enumerate(scores.iterrows()):
        theta=np.arccos(r[1].CORRELATION)
        rr=r[1].MSTD/r[1].OSTD
        print(rr)
        print(theta)

        ax1.plot(theta,rr,'o',label=r[0],color = colors[i])

    ax1.plot(0,1,'o',label='Obs')
    plt.legend(loc='upper right',bbox_to_anchor=[1.3,1.15])
    plt.show()

    rs,ts = np.meshgrid(np.linspace(0,2),np.linspace(0,angle_lim))
    rms = np.sqrt(1 + rs**2 - 2*rs*np.cos(ts))

    ax1.contour(ts, rs, rms,3,colors='0.5')
    #contours = ax1.contour(ts, rs, rms,3,colors='0.5')
    #plt.clabel(contours, inline=1, fontsize=10)
    plt.grid(linestyle=':',alpha=0.5)

    for i,r in enumerate(scores.iterrows()):
        crmse = np.sqrt(1 + (r[1].MSTD/scores.OSTD[i])**2 \
                   - 2*(r[1].MSTD/scores.OSTD[i])*r[1].CORRELATION)
        print(crmse)
        c1 = ax1.contour(ts, rs, rms,[crmse],colors=colors[i])
        plt.clabel(c1, inline=1, fontsize=10,fmt='%1.2f')

#%%

fig, ax1 = taylor(temp_skillscores,colors,'$^oC$',np.pi/2)
plt.title('Temperature \n cycle 2019082800',fontsize=16)
#ax1.axis["left"].label.set_text("Standard Deviation ($^oC$)")
#ax1.axis['left'].label.set_size(14)

file = folder + 'Taylor_temperature_2019082800'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%%

taylor(salt_skillscores,colors,'psu',np.pi/2)
plt.title('Salinity \n cycle 2019082800',fontsize=16)
#plt.xlabel('Standard Deviation (psu)')

file = folder + 'Taylor_salinity_2019082800'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)


#%%

taylor(Tmean_mld_skillscores,colors,'$^oC$',np.pi/2)
plt.title('Temperature MLD \n cycle 2019082800',fontsize=16)
plt.xlabel('Standard Deviation ($^oC$)')
plt.text(0.2,0.04,'0.13',color=colors[0])
plt.text(0.15,0.075,'0.10',color=colors[1])
plt.text(0.07,0.107,'0.10',color=colors[2])

file = folder + 'Taylor_temp_mld_2019082800'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%%

taylor_normalized(OHC_skillscores,colors,np.pi/2 + np.pi/16)
plt.title('OHC \n cycle 2019082800',fontsize=16)

file = folder + 'Taylor_ohc_2019082800'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Combine all metrics into one normalized Taylor diagram

fig,ax1 = taylor_template(np.pi/2+np.pi/16)

scores = temp_skillscores
for i,r in enumerate(scores.iterrows()):
    theta=np.arccos(r[1].CORRELATION)
    rr=r[1].MSTD/r[1].OSTD
    ax1.plot(theta,rr,'o',label=r[0],color = colors[i])

scores = salt_skillscores
for i,r in enumerate(scores.iterrows()):
    theta=np.arccos(r[1].CORRELATION)
    rr=r[1].MSTD/r[1].OSTD
    ax1.plot(theta,rr,'*',color = colors[i])

scores = Tmean_mld_skillscores
for i,r in enumerate(scores.iterrows()):
    theta=np.arccos(r[1].CORRELATION)
    rr=r[1].MSTD/r[1].OSTD
    ax1.plot(theta,rr,'s',color = colors[i])

scores = OHC_skillscores
for i,r in enumerate(scores.iterrows()):
    theta=np.arccos(r[1].CORRELATION)
    rr=r[1].MSTD/r[1].OSTD
    ax1.plot(theta,rr,'^',color = colors[i])

ax1.plot(0,1,'o',label='Obs')
ax1.plot(0,0,'ok',label='Temp')
ax1.plot(0,0,'*k',label='Salt')
ax1.plot(0,0,'sk',label='Temp ML')
ax1.plot(0,0,'^k',label='OHC')

plt.legend(loc='upper right',bbox_to_anchor=[1.45,1.2])

rs,ts = np.meshgrid(np.linspace(0,2.0),np.linspace(0,np.pi/2+np.pi/16))
rms = np.sqrt(1 + rs**2 - 2*rs*np.cos(ts))

contours = ax1.contour(ts, rs, rms,3,colors='0.5')
plt.clabel(contours, inline=1, fontsize=10)
plt.grid(linestyle=':',alpha=0.5)

file = folder + 'Taylor_norm_2019082800'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)
