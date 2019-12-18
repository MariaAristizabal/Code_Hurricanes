#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:20:03 2019

@author: root
"""

#%% User input

lon_lim = [-98.5,-60.0]
lat_lim = [10.0,45.0]

#Time window
date_ini = '2019/08/28/00/00'
date_end = '2019/09/02/00/00'

# url for GOFS 3.1
url_GOFS = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'

# folder nc files POM
folder_pom = '/Volumes/aristizabal/POM_Dorian/'
prefix = 'dorian05l.2019082800.pom.00'    

# folder figures
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

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
import seawater as sw
import os
import os.path
import glob

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Reading POM grid files operational

pom_folder_oper = folder_pom + 'POM_Dorian_2019082800_nc_files_oper/'
pom_grid_oper = sorted(glob.glob(os.path.join(pom_folder_oper,'*grid*')))[0]

POM_grid_oper = xr.open_dataset(pom_grid_oper)

lonc_oper = np.asarray(POM_grid_oper['east_e'][:])
latc_oper = np.asarray(POM_grid_oper['north_e'][:])

oklat_pom_oper = np.where(np.logical_and(latc_oper[:,0] >= lat_lim[0], latc_oper[:,0] <= lat_lim[1]))[0]
oklon_pom_oper = np.where(np.logical_and(lonc_oper[0,:] >= lon_lim[0], lonc_oper[0,:] <= lon_lim[1]))[0]

loncc = lonc_oper[oklat_pom_oper,:]
lon_pom_oper = loncc[:,oklon_pom_oper]
latcc = latc_oper[oklat_pom_oper,:]
lat_pom_oper = latcc[:,oklon_pom_oper]
zlevc_oper = np.asarray(POM_grid_oper['zz'][:])
hpom_oper = np.asarray(POM_grid_oper['h'][oklat_pom_oper,oklon_pom_oper])

#%% Reading POM grid files experimental

pom_folder_exp = folder_pom + 'POM_Dorian_2019082800_nc_files_exp/'
pom_grid_exp = sorted(glob.glob(os.path.join(pom_folder_exp,'*grid*')))[0]

POM_grid_exp = xr.open_dataset(pom_grid_exp)

lonc_exp = np.asarray(POM_grid_exp['east_e'][:])
latc_exp = np.asarray(POM_grid_exp['north_e'][:])

oklat_pom_exp = np.where(np.logical_and(latc_exp[:,0] >= lat_lim[0], latc_exp[:,0] <= lat_lim[1]))[0]
oklon_pom_exp = np.where(np.logical_and(lonc_exp[0,:] >= lon_lim[0], lonc_exp[0,:] <= lon_lim[1]))[0]

loncc = lonc_exp[oklat_pom_exp,:]
lon_pom_exp = loncc[:,oklon_pom_exp]
latcc = latc_exp[oklat_pom_exp,:]
lat_pom_exp = latcc[:,oklon_pom_exp]
zlevc_exp = np.asarray(POM_grid_exp['zz'][:])
hpom_exp = np.asarray(POM_grid_exp['h'][oklat_pom_exp,oklon_pom_exp])

#%% POM Operational 

pom_folder_oper = folder_pom + 'POM_Dorian_2019082800_nc_files_oper/'
pom_ncfiles_oper = sorted(glob.glob(os.path.join(pom_folder_oper,prefix+'*.nc')))

time_pom_oper = []
temp_pom_oper = np.empty((len(pom_ncfiles_oper),len(zlevc_oper),hpom_oper.shape[0],hpom_oper.shape[1]))
temp_pom_oper[:] = np.nan
salt_pom_oper = np.empty((len(pom_ncfiles_oper),len(zlevc_oper),hpom_oper.shape[0],hpom_oper.shape[1]))
salt_pom_oper[:] = np.nan
rho_pom_oper = np.empty((len(pom_ncfiles_oper),len(zlevc_oper),hpom_oper.shape[0],hpom_oper.shape[1]))
rho_pom_oper[:] = np.nan
for t,file in enumerate(pom_ncfiles_oper):
    print(t)
    pom = xr.open_dataset(file)

    tpom = pom['time'][:]
    timestamp_pom = mdates.date2num(tpom)[0]
    time_pom_oper.append(mdates.num2date(timestamp_pom))

    temp_pom_oper[t,:,:,:] = np.asarray(pom['t'][0,:,oklat_pom_oper,oklon_pom_oper])
    salt_pom_oper[t,:,:,:] = np.asarray(pom['s'][0,:,oklat_pom_oper,oklon_pom_oper])
    rho_pom_oper[t,:,:,:] = np.asarray(pom['rho'][0,:,oklat_pom_oper,oklon_pom_oper])

temp_pom_oper[temp_pom_oper==0.0] = np.nan
salt_pom_oper[salt_pom_oper==0.0] = np.nan
timestamp_pom_oper = mdates.date2num(time_pom_oper)

zmatrix = np.dot(hpom_oper.reshape(-1,1),zlevc_oper.reshape(1,-1))
zmatrix_pom_oper = zmatrix.reshape(hpom_oper.shape[0],hpom_oper.shape[1],zlevc_oper.shape[0])

#%% POM Experimental 

pom_folder_exp = folder_pom + 'POM_Dorian_2019082800_nc_files_exp/'
pom_ncfiles_exp = sorted(glob.glob(os.path.join(pom_folder_exp,prefix+'*.nc')))

time_pom_exp = []
temp_pom_exp = np.empty((len(pom_ncfiles_exp),len(zlevc_exp),hpom_exp.shape[0],hpom_exp.shape[1]))
temp_pom_exp[:] = np.nan
salt_pom_exp = np.empty((len(pom_ncfiles_exp),len(zlevc_exp),hpom_exp.shape[0],hpom_exp.shape[1]))
salt_pom_exp[:] = np.nan
rho_pom_exp = np.empty((len(pom_ncfiles_exp),len(zlevc_exp),hpom_exp.shape[0],hpom_exp.shape[1]))
rho_pom_exp[:] = np.nan
for t,file in enumerate(pom_ncfiles_exp):
    print(t)
    pom = xr.open_dataset(file)

    tpom = pom['time'][:]
    timestamp_pom = mdates.date2num(tpom)[0]
    time_pom_exp.append(mdates.num2date(timestamp_pom))

    temp_pom_exp[t,:,:,:] = np.asarray(pom['t'][0,:,oklat_pom_exp,oklon_pom_exp])
    salt_pom_exp[t,:,:,:] = np.asarray(pom['s'][0,:,oklat_pom_exp,oklon_pom_exp])
    rho_pom_exp[t,:,:,:] = np.asarray(pom['rho'][0,:,oklat_pom_exp,oklon_pom_exp])

temp_pom_exp[temp_pom_exp==0.0] = np.nan
salt_pom_exp[salt_pom_exp==0.0] = np.nan
timestamp_pom_exp = mdates.date2num(time_pom_exp)

zmatrix = np.dot(hpom_exp.reshape(-1,1),zlevc_exp.reshape(1,-1))
zmatrix_pom_exp = zmatrix.reshape(hpom_exp.shape[0],hpom_exp.shape[1],zlevc_exp.shape[0])

#%% Read GOFS 3.1 coordinates

print('Retrieving coordinates from model')
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

#%% Read GOFS 3.1 output and interpolating into POM operational
'''
oktime_GOFS_to_pom_oper = np.interp(timestamp_pom_oper,timestamp_GOFS,np.arange(len(time_GOFS))).astype(int)
oklon_GOFS_to_pom_oper = np.interp(lon_pom_oper[0,:],lon_GOFSg,np.arange(len(lon_GOFS))).astype(int)
oklat_GOFS_to_pom_oper = np.interp(lat_pom_oper[:,0],lat_GOFSg,np.arange(len(lat_GOFS))).astype(int)

okt = oktime_GOFS[0][oktime_GOFS_to_pom_oper]
oklt = oklat_GOFS[0][oklat_GOFS_to_pom_oper]
okln = oklon_GOFS[0][oklon_GOFS_to_pom_oper]

temp_GOFS_to_pom_oper = np.empty((len(okt),len(zlev_pom_oper),temp_pom_oper.shape[2],temp_pom_oper.shape[3]))
for t in np.arange(len(okt)):
    for y in np.arange(lat_pom_oper.shape[0]):
        for x in np.arange(lon_pom_oper.shape[1]):
            print('t=',t,' y=',y,' x=',x)
            if zmatrix_pom_oper[y,x,-1] == -1.0:
                temp_GOFS_to_pom_oper[t,:,y,x] = np.nan
            else:
                temp_prof = np.asarray(GOFS.water_temp[okt[t],:,oklt[y],okln[x]])
                temp_GOFS_to_pom_oper[t,:,y,x] = np.interp(-zmatrix_pom_oper[y,x,:],depth_GOFS,temp_prof)

np.savez('temp_GOFS_to_POM_2019082800.npz', temp_GOFS_to_pom_oper=temp_GOFS_to_pom_oper)
'''

GOFS_Dorian_2019082800 = np.load('/Volumes/aristizabal/Code/temp_GOFS_to_POM_2019082800.npz')
GOFS_Dorian_2019082800.files
temp_GOFS_to_pom_oper = GOFS_Dorian_2019082800['temp_GOFS_to_pom_oper']

#%% Define dataframe

DF_temp_GOFS_POM = pd.DataFrame(data=np.array([np.ravel(temp_GOFS_to_pom_oper,order='F'),\
                                      np.ravel(temp_pom_oper,order='F')]).T,\
                  columns=['temp_GOFS_to_POM_oper','temp_POM_oper'])
    
#%% Temperature statistics.

DF_temp = DF_temp_GOFS_POM

Ntemp = len(DF_temp)-1  #For Unbiased estimmator.


cols = ['CORRELATION','MSTD','CRMSE','BIAS']
    
tskill = np.empty((3,4))
tskill[:] = np.nan

#CORR
tskill[0,0] = DF_temp.corr()['temp_GOFS_to_POM_oper']['temp_GOFS_to_POM_oper']
tskill[1,0] = DF_temp.corr()['temp_GOFS_to_POM_oper']['temp_POM_oper']
#tskill[1,0] = DF_POM.corr()['temp_obs_to_oper']['temp_POM_oper']


#MSTD
tskill[0,1] = DF_temp.std().temp_GOFS_to_POM_oper
tskill[1,1] = DF_temp.std().temp_POM_oper

#CRMSE
tskill[0,2] = 0
tskill[1,2] = np.sqrt(np.nansum(((DF_temp.temp_GOFS_to_POM_oper-DF_temp.mean().temp_GOFS_to_POM_oper)-\
                                 (DF_temp.temp_POM_oper-DF_temp.mean().temp_POM_oper))**2)/Ntemp)
#tskill[1,3] = np.sqrt(np.nansum(((DF_temp.temp_obs_to_exp-DF_temp.mean().temp_obs_to_oper)-\
#                                 (DF_temp.temp_POM_oper-DF_temp.mean().temp_POM_oper))**2)/NPOM)

#BIAS
tskill[0,3] = 0
tskill[1,3] = DF_temp.mean().temp_GOFS_to_POM_oper - DF_temp.mean().temp_POM_oper
#tskill[1,4] = DF_POM.mean().temp_obs_to_oper - DF_POM.mean().temp_POM_oper

#color
colors = ['indianred','seagreen','darkorchid']
    
temp_skillscores = pd.DataFrame(tskill,
                        index=['GOFS_to_POM_oper','POM_oper','POM_exp'],
                        columns=cols)
print(temp_skillscores)    
        