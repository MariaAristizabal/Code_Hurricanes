#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:51:45 2020

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
gdata_sg665 = 'http://gliders.ioos.us/thredds/dodsC/deployments/aoml/SG665-20190718T1155/SG665-20190718T1155.nc3.nc'
gdata_sg666 = 'http://gliders.ioos.us/thredds/dodsC/deployments/aoml/SG666-20190718T1206/SG666-20190718T1206.nc3.nc'
#gdata_sg668 = 'http://gliders.ioos.us/thredds/dodsC/deployments/aoml/SG668-20190819T1217/SG668-20190819T1217.nc3.nc'
#gdata_silbo ='http://gliders.ioos.us/thredds/dodsC/deployments/rutgers/silbo-20190717T1917/silbo-20190717T1917.nc3.nc'

#gdata = gdata_sg665

# forecasting cycle to be used
cycle = '2019082800'

#Time window
#date_ini = '2019/08/28/00/00'
#date_end = '2019/09/02/06/00'

date_ini = '2019/08/25/00/00'
date_end = '2019/09/05/00/00'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc'

# KMZ file
kmz_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/KMZ_files/al052019_best_track-5.kmz'

# url for GOFS 
url_GOFS = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

# Argo data
# Jun1 -Jul1
Dir_Argo = '/Volumes/aristizabal/ARGO_data/DataSelection_20191014_193816_8936308'

# POM output
folder_pom = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/POM_Dorian_npz_files/'    
folder_pom_grid = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/'
pom_grid_oper = folder_pom_grid + 'dorian05l.2019082800.pom.grid.oper.nc'
pom_grid_exp = folder_pom_grid + 'dorian05l.2019082800.pom.grid.exp.nc'

# folder nc files POM
#folder_pom =  '/Volumes/aristizabal/POM_Dorian/'   

#%%

from matplotlib import pyplot as plt
import numpy as np
import cmocean
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import sys
import xarra as xr
import netCDF4
import seawater as sw

sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/All_code/Remote_repos/glider_model_comparisons_Python')

from read_glider_data import read_glider_data_thredds_server
#from process_glider_data import grid_glider_data_thredd

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Function Grid glider variables according to depth

def varsg_gridded(depth,time,temp,salt,dens,delta_z):
             
    depthg_gridded = np.arange(0,np.nanmax(depth),delta_z)
    tempg_gridded = np.empty((len(depthg_gridded),len(time)))
    tempg_gridded[:] = np.nan
    saltg_gridded = np.empty((len(depthg_gridded),len(time)))
    saltg_gridded[:] = np.nan
    densg_gridded = np.empty((len(depthg_gridded),len(time)))
    densg_gridded[:] = np.nan

    for t,tt in enumerate(time):
        depthu,oku = np.unique(depth[:,t],return_index=True)
        tempu = temp[oku,t]
        saltu = salt[oku,t]
        densu = dens[oku,t]
        okdd = np.isfinite(depthu)
        depthf = depthu[okdd]
        tempf = tempu[okdd]
        saltf = saltu[okdd]
        densf = densu[okdd]
 
        okt = np.isfinite(tempf)
        if np.sum(okt) < 3:
            temp[:,t] = np.nan
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
        
    return depthg_gridded, tempg_gridded, saltg_gridded, densg_gridded

#%% Function Conversion from glider longitude and latitude to GOFS convention

def glider_coor_to_GOFS_coord(long,latg):
    
    target_lon = np.empty((len(long),))
    target_lon[:] = np.nan
    for i,ii in enumerate(long):
        if ii < 0: 
            target_lon[i] = 360 + ii
        else:
            target_lon[i] = ii
    target_lat = latg
    
    return target_lon, target_lat

#%%  Function Conversion from GOFS convention to glider longitude and latitude
    
def GOFS_coor_to_glider_coord(lon_GOFS,lat_GOFS):
    
    lon_GOFSg = np.empty((len(lon_GOFS),))
    lon_GOFSg[:] = np.nan
    for i in range(len(lon_GOFS)):
        if lon_GOFS[i] > 180: 
            lon_GOFSg[i] = lon_GOFS[i] - 360 
        else:
            lon_GOFSg[i] = lon_GOFS[i]
    lat_GOFSg = lat_GOFS
    
    return lon_GOFSg, lat_GOFSg

#%%  Calculation of mixed layer depth based on dt, Tmean: mean temp within the 
# mixed layer and td: temp at 1 meter below the mixed layer          

def MLD_temp_and_dens_criteria(dt,drho,time,depth,temp,salt,dens):

    MLD_temp_crit = np.empty(len(time)) 
    MLD_temp_crit[:] = np.nan
    Tmean_temp_crit = np.empty(len(time)) 
    Tmean_temp_crit[:] = np.nan
    Smean_temp_crit = np.empty(len(time)) 
    Smean_temp_crit[:] = np.nan
    Td_temp_crit = np.empty(len(time)) 
    Td_temp_crit[:] = np.nan
    MLD_dens_crit = np.empty(len(time)) 
    MLD_dens_crit[:] = np.nan
    Tmean_dens_crit = np.empty(len(time)) 
    Tmean_dens_crit[:] = np.nan
    Smean_dens_crit = np.empty(len(time)) 
    Smean_dens_crit[:] = np.nan
    Td_dens_crit = np.empty(len(time)) 
    Td_dens_crit[:] = np.nan
    for t,tt in enumerate(time):
        if depth.ndim == 1:
            d10 = np.where(depth >= 10)[0][0]
        if depth.ndim == 2:
            d10 = np.where(depth[:,t] >= -10)[0][-1]
        T10 = temp[d10,t]
        delta_T = T10 - temp[:,t] 
        ok_mld_temp = np.where(delta_T <= dt)[0]
        rho10 = dens[d10,t]
        delta_rho = -(rho10 - dens[:,t])
        ok_mld_rho = np.where(delta_rho <= drho)[0]
        
        if ok_mld_temp.size == 0:
            MLD_temp_crit[t] = np.nan
            Td_temp_crit[t] = np.nan
            Tmean_temp_crit[t] = np.nan
            Smean_temp_crit[t] = np.nan            
        else:                             
            if depth.ndim == 1:
                MLD_temp_crit[t] = depth[ok_mld_temp[-1]]
                ok_mld_plus1m = np.where(depth >= depth[ok_mld_temp[-1]] + 1)[0][0]                 
            if depth.ndim == 2:
                MLD_temp_crit[t] = depth[ok_mld_temp[-1],t]
                ok_mld_plus1m = np.where(depth >= depth[ok_mld_temp[-1],t] + 1)[0][0]
            Td_temp_crit[t] = temp[ok_mld_plus1m,t]
            Tmean_temp_crit[t] = np.nanmean(temp[ok_mld_temp,t])
            Smean_temp_crit[t] = np.nanmean(salt[ok_mld_temp,t])
                
        if ok_mld_rho.size == 0:
            MLD_dens_crit[t] = np.nan
            Td_dens_crit[t] = np.nan
            Tmean_dens_crit[t] = np.nan
            Smean_dens_crit[t] = np.nan           
        else:
            if depth.ndim == 1:
                MLD_dens_crit[t] = depth[ok_mld_rho[-1]]
                ok_mld_plus1m = np.where(depth >= depth[ok_mld_rho[-1]] + 1)[0][0] 
            if depth.ndim == 2:
                MLD_dens_crit[t] = depth[ok_mld_rho[-1],t]
                ok_mld_plus1m = np.where(depth >= depth[ok_mld_rho[-1],t] + 1)[0][0] 
            Td_dens_crit[t] = temp[ok_mld_plus1m,t]        
            Tmean_dens_crit[t] = np.nanmean(temp[ok_mld_rho,t])
            Smean_dens_crit[t] = np.nanmean(salt[ok_mld_rho,t]) 

    return MLD_temp_crit,Tmean_temp_crit,Smean_temp_crit,Td_temp_crit,\
           MLD_dens_crit,Tmean_dens_crit,Smean_dens_crit,Td_dens_crit,Td_dens_crit

#%% Function Getting glider transect from GOFS
    
def get_glider_transect_from_GOFS(depth_GOFS,oktime_GOFS):
    
    print('Getting glider transect from GOFS')
    target_temp_GOFS = np.empty((len(depth_GOFS),len(oktime_GOFS[0])))
    target_temp_GOFS[:] = np.nan
    target_salt_GOFS = np.empty((len(depth_GOFS),len(oktime_GOFS[0])))
    target_salt_GOFS[:] = np.nan
    for i in range(len(oktime_GOFS[0])):
        print(len(oktime_GOFS[0]),' ',i)
        target_temp_GOFS[:,i] = GOFS.variables['water_temp'][oktime_GOFS[0][i],:,oklat_GOFS[i],oklon_GOFS[i]]
        target_salt_GOFS[:,i] = GOFS.variables['salinity'][oktime_GOFS[0][i],:,oklat_GOFS[i],oklon_GOFS[i]]

    return target_temp_GOFS,target_salt_GOFS

#%%  Function Getting glider transect from POM

def get_glider_transect_from_POM(temp_pom,salt_pom,rho_pom,zlev_pom,timestamp_pom,\
                                  oklat_pom,oklon_pom,zmatrix_pom):
    
    target_temp_POM = np.empty((len(zlev_pom),len(timestamp_pom)))
    target_temp_POM[:] = np.nan
    target_salt_POM = np.empty((len(zlev_pom),len(timestamp_pom)))
    target_salt_POM[:] = np.nan
    target_rho_POM = np.empty((len(zlev_pom),len(timestamp_pom)))
    target_rho_POM[:] = np.nan
    for i in range(len(timestamp_pom)):
        print(len(timestamp_pom),' ',i)
        target_temp_POM[:,i] = temp_pom[i,:,oklat_pom[i],oklon_pom[i]]
        target_salt_POM[:,i] = salt_pom[i,:,oklat_pom[i],oklon_pom[i]]
        target_rho_POM[:,i] = rho_pom[i,:,oklat_pom[i],oklon_pom[i]]
        
    target_dens_POM = target_rho_POM * 1000 + 1000 
    target_dens_POM[target_dens_POM == 1000.0] = np.nan   
    target_depth_POM = zmatrix_pom[oklat_pom,oklon_pom,:].T

    return target_temp_POM, target_salt_POM, target_dens_POM, target_depth_POM

#%% Figure transets

def figure_transect(var1,min_var1,max_var1,nlevels,var2,var3,time,tini,tend,depth,max_depth,color_map):

    if depth.ndim == 1:
        time_matrix = time
    else:
        time_matrix = np.tile(time,(depth.shape[0],1))
    
    kw = dict(levels = np.linspace(min_var1,max_var1,nlevels))
     
    # plot
    fig, ax = plt.subplots(figsize=(12, 2))      
    cs = plt.contourf(time_matrix,depth,var1,cmap=color_map,**kw)
    plt.contour(time_matrix,depth,var1,[26],colors='k')
    plt.plot(time,var2,'-',label='MLD dt',color='indianred',linewidth=2 )
    plt.plot(time,var3,'-',label='MLD drho',color='darkgreen',marker='.',\
             markeredgecolor='k',linewidth=2 )
    
    cs = fig.colorbar(cs, orientation='vertical') 
    cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
            
    ax.set_ylim(-200, 0)
    ax.set_ylabel('Depth (m)',fontsize=14)
    xticks = [tini+nday*timedelta(1) for nday in np.arange(14)]
    xticks = np.asarray(xticks)
    plt.xticks(xticks)
    xfmt = mdates.DateFormatter('%d \n %b')
    ax.xaxis.set_major_formatter(xfmt)
    plt.legend()  
    ax.set_xlim(tini,tend)

           
#%% Reading glider data
    
url_glider = gdata_sg665

var = 'temperature'
scatter_plot = 'no'
kwargs = dict(date_ini=date_ini[0:-3],date_end=date_end[0:-3])

varg, latg, long, depthg, timeg_sg665, inst_id = \
             read_glider_data_thredds_server(url_glider,var,scatter_plot,**kwargs)
             
tempg_sg665 = varg  

var = 'salinity'  
varg, latg, long, depthg, timeg, inst_id = \
             read_glider_data_thredds_server(url_glider,var,scatter_plot,**kwargs)
            
saltg_sg665 = varg
 
var = 'density'  
varg, latg, long, depthg, timeg, inst_id = \
             read_glider_data_thredds_server(url_glider,var,scatter_plot,**kwargs)
             
densg_sg665 = varg
depthg_sg665 = depthg 

#%% Reading glider data
    
url_glider = gdata_sg666

var = 'temperature'
scatter_plot = 'no'
kwargs = dict(date_ini=date_ini[0:-3],date_end=date_end[0:-3])

varg, latg, long, depthg, timeg_sg666, inst_id = \
             read_glider_data_thredds_server(url_glider,var,scatter_plot,**kwargs)
             
tempg_sg666 = varg  

var = 'salinity'  
varg, latg, long, depthg, timeg, inst_id = \
             read_glider_data_thredds_server(url_glider,var,scatter_plot,**kwargs)
            
saltg_sg666 = varg
 
var = 'density'  
varg, latg, long, depthg, timeg, inst_id = \
             read_glider_data_thredds_server(url_glider,var,scatter_plot,**kwargs)
             
densg_sg666 = varg
depthg_sg666 = depthg 

#%% Read GOFS 3.1 grid

#Time window
date_ini = cycle[0:4]+'/'+cycle[4:6]+'/'+cycle[6:8]+'/'+cycle[8:]+'/00/00'
tini = datetime.strptime(date_ini,'%Y/%m/%d/%H/%M/%S')
tend = tini + timedelta(hours=120)
date_end = tend.strftime('%Y/%m/%d/%H/%M/%S')

print('Retrieving coordinates from GOFS')
GOFS = xr.open_dataset(url_GOFS,decode_times=False) 

tt_G = GOFS.time
t_G = netCDF4.num2date(tt_G[:],tt_G.units) 

tmin = datetime.strptime(date_ini[0:-6],'%Y/%m/%d/%H')
tmax = datetime.strptime(date_end[0:-6],'%Y/%m/%d/%H')
oktime_GOFS = np.where(np.logical_and(t_G >= tmin, t_G <= tmax)) 
time_GOFS = np.asarray(t_G[oktime_GOFS])
timestamp_GOFS = mdates.date2num(time_GOFS)

lat_G = np.asarray(GOFS.lat[:])
lon_G = np.asarray(GOFS.lon[:])

# Conversion from glider longitude and latitude to GOFS convention
lon_limG, lat_limG = glider_coor_to_GOFS_coord(lon_lim,lat_lim)

oklat_GOFS = np.where(np.logical_and(lat_G >= lat_limG[0], lat_G <= lat_limG[1])) 
oklon_GOFS = np.where(np.logical_and(lon_G >= lon_limG[0], lon_G <= lon_limG[1])) 

lat_GOFS = lat_G[oklat_GOFS]
lon_GOFS = lon_G[oklon_GOFS]

depth_GOFS = np.asarray(GOFS.depth[:])

# Conversion from GOFS longitude and latitude to glider convention
lon_GOFSg, lat_GOFSg = GOFS_coor_to_glider_coord(lon_GOFS,lat_GOFS)

#%%

# Conversion from glider longitude and latitude to GOFS convention
target_lon, target_lat = glider_coor_to_GOFS_coord(long,latg)

# Changing times to timestamp
tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]
tstamp_model = [mdates.date2num(time_GOFS[i]) for i in np.arange(len(time_GOFS))]

# interpolating glider lon and lat to lat and lon on model time
sublon_GOFS = np.interp(tstamp_model,tstamp_glider,target_lon)
sublat_GOFS = np.interp(tstamp_model,tstamp_glider,target_lat)

# Conversion from GOFS convention to glider longitude and latitude
sublon_GOFSg,sublat_GOFSg = GOFS_coor_to_glider_coord(sublon_GOFS,sublat_GOFS)

# getting the model grid positions for sublonm and sublatm
oklon_GOFS = np.round(np.interp(sublon_GOFS,lon_G,np.arange(len(lon_G)))).astype(int)
oklat_GOFS = np.round(np.interp(sublat_GOFS,lat_G,np.arange(len(lat_G)))).astype(int)
    
# Getting glider transect from model
target_temp_GOFS, target_salt_GOFS = \
                          get_glider_transect_from_GOFS(depth_GOFS,oktime_GOFS) 
                          
#%% Calculate density for GOFS

target_dens_GOFS = sw.dens(target_salt_GOFS,target_temp_GOFS,np.tile(depth_GOFS,(len(time_GOFS),1)).T) 

#%% Grid glider variables according to depth

delta_z = 0.5

depthg_gridded_sg665,tempg_gridded_sg665,saltg_gridded_sg665,densg_gridded_sg665 = \
varsg_gridded(depthg_sg665,timeg_sg665,tempg_sg665,saltg_sg665,densg_sg665,delta_z)

depthg_gridded_sg666,tempg_gridded_sg666,saltg_gridded_sg666,densg_gridded_sg666 = \
varsg_gridded(depthg_sg666,timeg_sg666,tempg_sg666,saltg_sg666,densg_sg666,delta_z) 
    
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

# time POM
time_pom = [tini + timedelta(hours=int(hrs)) for hrs in np.arange(0,126,6)]   

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

#%% Retrieve glider transect from POM operational

# Changing times to timestamp
tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]

# interpolating glider lon and lat to lat and lon on model time
sublon_pom = np.interp(timestamp_pom_oper,tstamp_glider,long)
sublat_pom = np.interp(timestamp_pom_oper,tstamp_glider,latg)

# getting the model grid positions for sublonm and sublatm
oklon_pom = np.round(np.interp(sublon_pom,lon_pom_oper[0,:],np.arange(len(lon_pom_oper[0,:])))).astype(int)
oklat_pom = np.round(np.interp(sublat_pom,lat_pom_oper[:,0],np.arange(len(lat_pom_oper[:,0])))).astype(int)

target_temp_POM_oper, target_salt_POM_oper, target_dens_POM_oper,\
target_depth_POM_oper = \
get_glider_transect_from_POM(temp_pom_oper,salt_pom_oper,rho_pom_oper,zlev_pom_oper,\
                             timestamp_pom_oper,oklat_pom,oklon_pom,zmatrix_pom_oper)
   
#%% Retrieve glider transect from POM experimental

# Changing times to timestamp
tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]

# interpolating glider lon and lat to lat and lon on model time
sublon_pom = np.interp(timestamp_pom_exp,tstamp_glider,long)
sublat_pom = np.interp(timestamp_pom_exp,tstamp_glider,latg)

# getting the model grid positions for sublonm and sublatm
oklon_pom = np.round(np.interp(sublon_pom,lon_pom_exp[0,:],np.arange(len(lon_pom_exp[0,:])))).astype(int)
oklat_pom = np.round(np.interp(sublat_pom,lat_pom_exp[:,0],np.arange(len(lat_pom_exp[:,0])))).astype(int)

target_temp_POM_exp, target_salt_POM_exp, target_dens_POM_exp,\
target_depth_POM_exp = \
get_glider_transect_from_POM(temp_pom_exp,salt_pom_exp,rho_pom_exp,zlev_pom_exp,\
                             timestamp_pom_exp,oklat_pom,oklon_pom,zmatrix_pom_exp)

#%% Calculation of mixed layer depth based on temperature and density critria
# Tmean: mean temp within the mixed layer and 
# td: temp at 1 meter below the mixed layer            

dt = 0.2
drho = 0.125

# for glider data
MLD_temp_crit_glid_sg665, _, _, _, MLD_dens_crit_glid_sg665, Tmean_dens_crit_glid_sg665, \
    Smean_dens_crit_glid_sg665, _, Td_sg665= \
MLD_temp_and_dens_criteria(dt,drho,timeg_sg665,depthg_gridded_sg665,tempg_gridded_sg665,saltg_gridded_sg665,densg_gridded_sg665)

# for glider data
MLD_temp_crit_glid_sg666, _, _, _, MLD_dens_crit_glid_sg666, Tmean_dens_crit_glid_sg666, \
    Smean_dens_crit_glid_sg666, _, Td_sg666 = \
MLD_temp_and_dens_criteria(dt,drho,timeg_sg666,depthg_gridded_sg666,tempg_gridded_sg666,saltg_gridded_sg666,densg_gridded_sg666)

#%% Calculate dTmean/dt

Tmean = Tmean_dens_crit_glid_sg665
timeg = timeg_sg665
dTmean_dt_sg665 = (Tmean[1:]-Tmean[0:-1])/(mdates.date2num(timeg[1:]) - mdates.date2num(timeg[0:-1]))

Tmean = Tmean_dens_crit_glid_sg666
timeg = timeg_sg666
dTmean_dt_sg666 = (Tmean[1:]-Tmean[0:-1])/(mdates.date2num(timeg[1:]) - mdates.date2num(timeg[0:-1]))


#%%

fig,ax1 = plt.subplots(figsize=(7, 4))
plt.plot(timeg_sg665,MLD_dens_crit_glid_sg665,'.-',label='MLD SG665',color='royalblue')
plt.plot(timeg_sg666,MLD_dens_crit_glid_sg666,'.-',label='MLD SG666',color='seagreen')
tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(10,60)))
plt.plot(tDorian,np.arange(10,60),'--k')
plt.legend()
xfmt = mdates.DateFormatter('%d \n %b')
ax1.xaxis.set_major_formatter(xfmt)

file = folder + ' ' + 'MLD_dens_crit_sg665_sg666'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%%

fig,ax1 = plt.subplots(figsize=(7, 4))
plt.plot(timeg_sg665,Tmean_dens_crit_glid_sg665,'.-',label='Temp ml SG665',color='royalblue')
plt.plot(timeg_sg666,Tmean_dens_crit_glid_sg666,'.-',label='Temp ml SG666',color='seagreen')
tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(29,29.5,0.01)))
plt.plot(tDorian,np.arange(29,29.5,0.01),'--k')
plt.legend()
xfmt = mdates.DateFormatter('%d \n %b')
ax1.xaxis.set_major_formatter(xfmt)

file = folder + ' ' + 'Tmean_dens_crit_sg665_sg666'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%%

fig,ax1 = plt.subplots(figsize=(7, 4))
plt.plot(timeg_sg665,Smean_dens_crit_glid_sg665,'.-',label='Salt ml SG665',color='royalblue')
plt.plot(timeg_sg666,Smean_dens_crit_glid_sg666,'.-',label='Salt ml SG666',color='seagreen')
tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(35.5,36.5,0.01)))
plt.plot(tDorian,np.arange(35.5,36.5,0.01),'--k')
plt.legend()
xfmt = mdates.DateFormatter('%d \n %b')
ax1.xaxis.set_major_formatter(xfmt)

file = folder + ' ' + 'Smean_dens_crit_sg665_sg666'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 


#%%

fig,ax1 = plt.subplots(figsize=(7, 4))
plt.plot(timeg_sg665[0:-1],dTmean_dt_sg665,'.-',label='dT/dt ml SG665')
plt.plot(timeg_sg666[0:-1],dTmean_dt_sg666,'.-',label='dT/dt ml SG666')
#tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(35.5,36.5,0.01)))
#plt.plot(tDorian,np.arange(35.5,36.5,0.01),'--k')
plt.legend()
xfmt = mdates.DateFormatter('%d \n %b')
ax1.xaxis.set_major_formatter(xfmt)

#%%

fig,ax1 = plt.subplots(figsize=(7, 4))
plt.plot(timeg_sg665,Tmean_dens_crit_glid_sg665,'.-',color='indianred',label='Temp ml SG665')
plt.plot(timeg_sg665,Td_sg665,'.-',label='Td SG665',color='mediumslateblue')
tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(28.9,29.4,0.01)))
plt.plot(tDorian,np.arange(28.9,29.4,0.01),'--k')
plt.legend()
xfmt = mdates.DateFormatter('%d \n %b')
ax1.xaxis.set_major_formatter(xfmt)

file = folder + ' ' + 'Tmean_Td_dens_crit_sg665'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%%

fig,ax1 = plt.subplots(figsize=(7, 4))
plt.plot(timeg_sg666,Tmean_dens_crit_glid_sg666,'.-',color='indianred',label='Temp ml SG666')
plt.plot(timeg_sg666,Td_sg666,'.-',label='Td SG666',color='mediumslateblue')
tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(28.5,29.4,0.01)))
plt.plot(tDorian,np.arange(28.5,29.4,0.01),'--k')
plt.legend()
xfmt = mdates.DateFormatter('%d \n %b')
ax1.xaxis.set_major_formatter(xfmt)

file = folder + ' ' + 'Tmean_Td_dens_crit_sg666'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%%
tdorian = datetime(2019,8,28,18)
bef = timeg_sg665 <= tdorian
aft = np.logical_and(timeg_sg665 > tdorian, timeg_sg665 <= datetime(2019,8,29,18))

plt.figure()
plt.plot(np.nanmean(tempg_gridded_sg665[:,bef],1),-depthg_gridded_sg665,'.-',color='indianred',label='18 hours Before')
plt.plot(np.nanmean(tempg_gridded_sg665[:,aft],1),-depthg_gridded_sg665,'.-',color='slateblue',label='1 day After')
plt.plot(tempg_gridded_sg665[:,bef],-depthg_gridded_sg665,'-',color='indianred',alpha=0.1)
plt.plot(tempg_gridded_sg665[:,aft],-depthg_gridded_sg665,'-',color='slateblue',alpha=0.1)
plt.legend()
plt.ylim([-100,0])
plt.xlim([26,30])
plt.title('Temperature Profile before and after Dorian SG665',size=16) #+ inst_id.split('-')[0]
plt.ylabel('Depth (m)',size=14)
#plt.xlabel('Density ($kg/m^3$)')

file = folder + ' ' + 'temp_prof_bef_after_Dorian_sg665'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%%

tdorian = datetime(2019,8,28,18)
bef = timeg_sg666 <= tdorian
aft = np.logical_and(timeg_sg666 > tdorian, timeg_sg666 <= datetime(2019,8,29,18))

plt.figure()
plt.plot(np.nanmean(tempg_gridded_sg666[:,bef],1),-depthg_gridded_sg666,'.-',color='indianred',label='18 hours Before')
plt.plot(np.nanmean(tempg_gridded_sg666[:,aft],1),-depthg_gridded_sg666,'.-',color='slateblue',label='1 day After')
plt.plot(tempg_gridded_sg666[:,bef],-depthg_gridded_sg666,'-',color='indianred',alpha=0.1)
plt.plot(tempg_gridded_sg666[:,aft],-depthg_gridded_sg666,'-',color='slateblue',alpha=0.1)
plt.legend()
plt.ylim([-100,0])
plt.xlim([26,30])
plt.title('Temperature Profile before and after Dorian SG666',size=16) #+ inst_id.split('-')[0]
plt.ylabel('Depth (m)',size=14)
#plt.xlabel('Density ($kg/m^3$)')

file = folder + ' ' + 'temp_prof_bef_after_Dorian_sg666'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%% Top 200 m glider salinity from 2019/08/28/00

color_map = cmocean.cm.haline

var1 = saltg_gridded_sg665
var2 = -MLD_temp_crit_glid_sg665
var3 = -MLD_dens_crit_glid_sg665
time = timeg_sg665
depth = -depthg_gridded_sg665
max_depth = -200
min_var1 = 35.5
max_var1 = 37.3
nlevels = 19 #np.round((max_var1 - min_var1)*10+1)
tini = datetime.strptime(date_ini,'%Y/%m/%d/%H/%M')
tend = datetime.strptime(date_end,'%Y/%m/%d/%H/%M')
 
figure_transect(var1,min_var1,max_var1,nlevels,var2,var3,time,tini,tend,depth,max_depth,color_map)

tDorian = np.tile(datetime(2019,8,28,18),len(np.arange(max_depth,0))) #ng665, ng666
plt.plot(tDorian,np.arange(max_depth,0),'--k')
plt.title('Along Track ' + 'Salinity' + ' Profile ' + inst_id,fontsize=14)
file = folder + ' ' + 'along_track_salt_top200 ' + inst_id
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)