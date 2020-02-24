#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:18:51 2019

@author: root
"""

#%% User input

lon_lim = [-80.0,-60.0]
lat_lim = [15.0,35.0]

# Server erddap url IOOS glider dap
server = 'https://data.ioos.us/gliders/erddap'

#gliders sg666, sg665, sg668, silbo
url_aoml = 'http://gliders.ioos.us/thredds/dodsC/deployments/aoml/'
#url_RU = 'http://gliders.ioos.us/thredds/dodsC/deployments/rutgers/'
gdata = [url_aoml+'SG665-20190718T1155/SG665-20190718T1155.nc3.nc',\
         url_aoml+'SG666-20190718T1206/SG666-20190718T1206.nc3.nc',\
         url_aoml+'SG668-20190819T1217/SG668-20190819T1217.nc3.nc',\
         url_aoml+'SG664-20190716T1218/SG664-20190716T1218.nc3.nc',\
         url_aoml+'SG663-20190716T1159/SG663-20190716T1159.nc3.nc',\
         url_aoml+'SG667-20190815T1247/SG667-20190815T1247.nc3.nc']
         
#url_RU+'silbo-20190717T1917/silbo-20190717T1917.nc3.nc']

cycle = '2019082800'    
    
#Time window
#date_ini = '2019/08/28/00/00'
#date_end = '2019/09/02/00/06'

# url for GOFS 3.1
url_GOFS = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'

# figures
folder_fig = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

# folder nc files POM
folder_pom =  '/Volumes/aristizabal/POM_Dorian/'

'''
folder_pom = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/POM_Dorian_npz_files/'    
folder_pom_grid = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/'
pom_grid_oper = folder_pom_grid + 'dorian05l.2019082800.pom.grid.oper.nc'
pom_grid_exp = folder_pom_grid + 'dorian05l.2019082800.pom.grid.exp.nc'
'''
# folde HWRF2010_HYCOM
folder_hycom = '/Volumes/aristizabal/HWRF2020_HYCOM_Dorian/'

###################

# folder nc files POM
folder_pom_oper = folder_pom + 'POM_Dorian_' + cycle + '_nc_files_oper/'
folder_pom_exp = folder_pom + 'HWRF2020_POM_Dorian/' + 'HWRF2020_POM_Dorian_' + cycle + '_nc_files_exp/'
prefix_pom = 'dorian05l.' + cycle + '.pom.00'

pom_grid_oper = folder_pom_oper + 'dorian05l.' + cycle + '.pom.grid.nc'
pom_grid_exp = folder_pom_exp + 'dorian05l.' + cycle + '.pom.grid.nc'

# Dorian track files
track_oper = folder_pom_oper + 'dorian05l.' + cycle + '.trak.hwrf.atcfunix'
track_exp = folder_pom_exp + 'dorian05l.' + cycle + '.trak.hwrf.atcfunix'


##################
# folder ab files HYCOM
folder_hycom_exp = folder_hycom + 'HWRF2020_HYCOM_Dorian_' + cycle + '_ab_files_exp/'
prefix_hycom = 'dorian05l.' + cycle + '.hwrf_rtofs_hat10_3z'

#Dir_HMON_HYCOM = '/Volumes/aristizabal/ncep_model/HMON-HYCOM_Michael/'
Dir_HMON_HYCOM = '/Volumes/aristizabal/ncep_model/HWRF-Hycom-WW3_exp_Michael/'
# RTOFS grid file name
hycom_grid_exp = Dir_HMON_HYCOM + 'hwrf_rtofs_hat10.basin.regional.grid'

# Dorian track files
hycom_track_exp = folder_hycom_exp + 'dorian05l.' + cycle + '.trak.hwrf.atcfunix'

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
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import sys
import seawater as sw
import os
import os.path
import glob 

sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/All_code/Remote_repos/glider_model_comparisons_Python')
from read_glider_data import read_glider_data_thredds_server
#from process_glider_data import grid_glider_data_thredd

import sys
sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/NCEP_scripts')
from utils4HYCOM import readBinz, readgrids

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
    
def get_glider_transect_from_POM(folder_pom,prefix,zlev,zmatrix_pom,lon_pom,lat_pom,tstamp_glider,long,latg):

    ncfiles = sorted(glob.glob(os.path.join(folder_pom,prefix+'*.nc')))

    target_temp_POM = np.empty((len(zlev),len(ncfiles)))
    target_temp_POM[:] = np.nan
    target_salt_POM = np.empty((len(zlev),len(ncfiles)))
    target_salt_POM[:] = np.nan
    target_dens_POM = np.empty((len(zlev),len(ncfiles)))
    target_dens_POM[:] = np.nan
    target_depth_POM = np.empty((len(zlev),len(ncfiles)))
    target_depth_POM[:] = np.nan
    time_POM = []
    
    for x,file in enumerate(ncfiles):
        print(x)
        pom = xr.open_dataset(file)
        
        tpom = pom['time'][:]
        timestamp_pom = mdates.date2num(tpom)[0]
        time_POM.append(mdates.num2date(timestamp_pom))
        
        # Interpolating latg and longlider into RTOFS grid
        sublonpom = np.interp(timestamp_pom,tstamp_glider,long)
        sublatpom = np.interp(timestamp_pom,tstamp_glider,latg)
        oklonpom = np.int(np.round(np.interp(sublonpom,lon_pom[0,:],np.arange(len(lon_pom[0,:])))))
        oklatpom = np.int(np.round(np.interp(sublatpom,lat_pom[:,0],np.arange(len(lat_pom[:,0])))))
        
        target_temp_POM[:,x] = np.asarray(pom['t'][0,:,oklatpom,oklonpom])
        target_salt_POM[:,x] = np.asarray(pom['s'][0,:,oklatpom,oklonpom])
        target_rho_pom = np.asarray(pom['rho'][0,:,oklatpom,oklonpom])
        target_dens_POM[:,x] = target_rho_pom * 1000 + 1000
        target_depth_POM[:,x] = zmatrix_pom[oklatpom,oklonpom,:].T
        
    target_temp_POM[target_temp_POM==0] = np.nan
    target_salt_POM[target_salt_POM==0] = np.nan
    target_dens_POM[target_dens_POM==1000.0] = np.nan
            
    return time_POM, target_temp_POM, target_salt_POM, target_dens_POM, target_depth_POM

'''
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
'''

#%%
    
def get_glider_transect_from_HYCOM(folder_hycom,prefix,nz,lon_hycom,lat_hycom,var,timestamp_glider,lon_glider,lat_glider):

    afiles = sorted(glob.glob(os.path.join(folder_hycom,prefix+'*.a')))    
        
    target_var_hycom = np.empty((nz,len(afiles)))
    target_var_hycom[:] = np.nan
    time_hycom = []
    for x, file in enumerate(afiles):
        print(x)
        #lines=[line.rstrip() for line in open(file[:-2]+'.b')]
    
        #Reading time stamp
        year = int(file.split('.')[1][0:4])
        month = int(file.split('.')[1][4:6])
        day = int(file.split('.')[1][6:8])
        hour = int(file.split('.')[1][8:10])
        dt = int(file.split('.')[3][1:])
        timestamp_hycom = mdates.date2num(datetime(year,month,day,hour)) + dt/24
        time_hycom.append(mdates.num2date(timestamp_hycom))
        
        # Interpolating latg and longlider into HYCOM grid
        sublon_hycom = np.interp(timestamp_hycom,timestamp_glider,lon_glider)
        sublat_hycom = np.interp(timestamp_hycom,timestamp_glider,lat_glider)
        oklon_hycom = np.int(np.round(np.interp(sublon_hycom,lon_hycom[0,:],np.arange(len(lon_hycom[0,:])))))
        oklat_hycom = np.int(np.round(np.interp(sublat_hycom,lat_hycom[:,0],np.arange(len(lat_hycom[:,0])))))
        
        # Reading 3D variable from binary file 
        var_hycom = readBinz(file[:-2],'3z',var)
        #ts=readBin(afile,'archive','temp')
        target_var_hycom[:,x] = var_hycom[oklat_hycom,oklon_hycom,:]
        
    time_hycom = np.asarray(time_hycom)
    #timestamp_hycom = mdates.date2num(time_hycom)
    
    return target_var_hycom, time_hycom

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
           MLD_dens_crit,Tmean_dens_crit,Smean_dens_crit,Td_dens_crit
           
#%% Function Ocean Heat Content

def OHC_surface(time,temp,depth,dens):
    cp = 3985 #Heat capacity in J/(kg K)

    OHC = np.empty((len(time)))
    OHC[:] = np.nan
    for t,tt in enumerate(time):
        ok26 = temp[:,t] >= 26
        if len(depth[ok26]) != 0:
            if np.nanmin(depth[ok26])>10:
                OHC[t] = np.nan  
            else:
                rho0 = np.nanmean(dens[ok26,t])
                if depth.ndim == 1:
                    OHC[t] = np.abs(cp * rho0 * np.trapz(temp[ok26,t]-26,depth[ok26]))
                if depth.ndim == 2:
                    OHC[t] = np.abs(cp * rho0 * np.trapz(temp[ok26,t]-26,depth[ok26,t]))
        else:    
            OHC[t] = np.nan
            
    return OHC

#%%   
def interp_datasets_in_z(temp_orig,salt_orig,depth_orig,depth_target,indexes_time,dim_vars):

    temp_interp = np.empty((dim_vars[0],dim_vars[1]))
    temp_interp[:] = np.nan
    salt_interp = np.empty((dim_vars[0],dim_vars[1]))
    salt_interp[:] = np.nan
    for i in np.arange(len(indexes_time)):
        pos = np.argsort(depth_orig[:,indexes_time[i]])
        if depth_target.ndim == 1:
            temp_interp[:,i] = np.interp(depth_target,depth_orig[pos,indexes_time[i]],temp_orig[pos,indexes_time[i]])
            salt_interp[:,i] = np.interp(depth_target,depth_orig[pos,indexes_time[i]],salt_orig[pos,indexes_time[i]])
        if depth_target.ndim == 2:
            temp_interp[:,i] = np.interp(depth_target[:,i],depth_orig[pos,indexes_time[i]],temp_orig[pos,indexes_time[i]])
            salt_interp[:,i] = np.interp(depth_target[:,i],depth_orig[pos,indexes_time[i]],salt_orig[pos,indexes_time[i]])

    return temp_interp, salt_interp

#%%    
def depth_aver_top_100(depth,var):

    varmean100 = np.empty(var.shape[1])
    varmean100[:] = np.nan
    if depth.ndim == 1:
        okd = np.abs(depth) <= 100
        if len(depth[okd]) != 0:
            for t in np.arange(var.shape[1]):
                if len(np.where(np.isnan(var[okd,t]))[0])>10:
                    varmean100[t] = np.nan
                else:
                    varmean100[t] = np.nanmean(var[okd,t],0)
    else:
        for t in np.arange(depth.shape[1]):
            okd = np.abs(depth[:,t]) <= 100
            if len(depth[okd,t]) != 0:
                if len(np.where(np.isnan(var[okd,t]))[0])>10:
                    varmean100[t] = np.nan
                else:
                    varmean100[t] = np.nanmean(var[okd,t])
            else:
                varmean100[t] = np.nan
    
    return varmean100   

#%%    
def depth_aver_temp_from_100_to_base_mixed_layer(dt,drho,time,depth,temp,salt,dens):

    _, _, _, _, MLD_dens_crit_glid,_,_,_ = \
        MLD_temp_and_dens_criteria(dt,drho,time,depth,temp,salt,dens)    
        
    tempmean100_to_ml = np.empty(temp.shape[1])
    tempmean100_to_ml[:] = np.nan
    if depth.ndim == 1:
       for t in np.arange(temp.shape[1]):
           okd = np.logical_and(np.abs(depth) <= 100,np.abs(depth) >= np.abs(MLD_dens_crit_glid[t]))
           if len(depth[okd]) != 0:
              tempmean100_to_ml[t] = np.nanmean(temp[okd,t],0)
           else:
               tempmean100_to_ml[t] = np.nan
        
    else:
        for t in np.arange(temp.shape[1]):
            okd = np.logical_and(np.abs(depth[:,t]) <= 100,np.abs(depth[:,t]) >= np.abs(MLD_dens_crit_glid[t]))
            if len(depth[okd,t]) != 0:
                tempmean100_to_ml[t] = np.nanmean(temp[okd,t])
            else:
                tempmean100_to_ml[t] = np.nan
    
    return tempmean100_to_ml 

#%% Taylor Diagram

def taylor_template(angle_lim,std_lim):

    fig = plt.figure()
    tr = PolarAxes.PolarTransform()
    
    min_corr = np.round(np.cos(angle_lim),1)
    CCgrid= np.concatenate((np.arange(min_corr*10,10,2.0)/10.,[0.9,0.95,0.99]))
    CCpolar=np.arccos(CCgrid)
    gf=FixedLocator(CCpolar)
    tf=DictFormatter(dict(zip(CCpolar, map(str,CCgrid))))
    
    STDgrid=np.arange(0,std_lim,.5)
    gfs=FixedLocator(STDgrid)
    tfs=DictFormatter(dict(zip(STDgrid, map(str,STDgrid))))
    
    ra0, ra1 =0, angle_lim
    cz0, cz1 = 0, std_lim
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
    
    min_corr = np.round(np.cos(angle_lim),1)
    CCgrid= np.concatenate((np.arange(min_corr*10,10,2.0)/10.,[0.9,0.95,0.99]))
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
    
    min_corr = np.round(np.cos(angle_lim),1)
    CCgrid= np.concatenate((np.arange(min_corr*10,10,2.0)/10.,[0.9,0.95,0.99]))
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

#%% POM Operational and Experimental
'''
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
'''
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

#Time window
date_ini = cycle[0:4]+'-'+cycle[4:6]+'-'+cycle[6:8]+' '+cycle[8:]+':00:00'
tini = datetime.strptime(date_ini,'%Y-%m-%d %H:%M:%S')
tend = tini + timedelta(hours=126)
date_end = str(tend)

print('Retrieving coordinates from GOFS')
GOFS = xr.open_dataset(url_GOFS,decode_times=False) 

tt_G = GOFS.time
t_G = netCDF4.num2date(tt_G[:],tt_G.units) 

tmin = datetime.strptime(date_ini[0:-6],'%Y-%m-%d %H')
tmax = datetime.strptime(date_end[0:-6],'%Y-%m-%d %H')
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

#%% Reading HYCOM grid

# Reading lat and lon
lines_grid = [line.rstrip() for line in open(hycom_grid_exp+'.b')]
lon_hycom = np.array(readgrids(hycom_grid_exp,'plon:',[0]))
lat_hycom = np.array(readgrids(hycom_grid_exp,'plat:',[0]))

# Extracting the longitudinal and latitudinal size array
idm=int([line.split() for line in lines_grid if 'longitudinal' in line][0][0])
jdm=int([line.split() for line in lines_grid if 'latitudinal' in line][0][0])

afiles = sorted(glob.glob(os.path.join(folder_hycom_exp,prefix_hycom+'*.a')))

# Reading depths
lines=[line.rstrip() for line in open(afiles[0][:-2]+'.b')]
z = []
for line in lines[6:]:
    if line.split()[2]=='temp':
        #print(line.split()[1])
        z.append(float(line.split()[1]))
depth_HYCOM_exp = np.asarray(z) 

nz = len(depth_HYCOM_exp) 

#%% Reading glider data

DF_GOFS_temp_salt = pd.DataFrame()
DF_POM_temp_salt = pd.DataFrame()
DF_HYCOM_temp_salt = pd.DataFrame()
DF_GOFS_MLD = pd.DataFrame()
DF_POM_MLD = pd.DataFrame()
DF_HYCOM_MLD = pd.DataFrame()
DF_GOFS_OHC = pd.DataFrame()
DF_POM_OHC = pd.DataFrame()
DF_HYCOM_OHC = pd.DataFrame()
DF_GOFS_T100 = pd.DataFrame()
DF_POM_T100 = pd.DataFrame()
DF_HYCOM_T100 = pd.DataFrame()
DF_GOFS_T100_to_ml = pd.DataFrame()
DF_POM_T100_to_ml = pd.DataFrame()
DF_HYCOM_T100_to_ml = pd.DataFrame()

for f,file in enumerate(gdata):

    #f=0
    #file = gdata[0]
    
    print(file)    
    url_glider = file
    
    var = 'temperature'
    scatter_plot = 'no'
    kwargs = dict(date_ini=datetime.strftime(tmin,'%Y/%m/%d/%H'),\
              date_end=datetime.strftime(tmax,'%Y/%m/%d/%H'))    
    tempg, latg, long, depthg, timeg, inst_id = \
                 read_glider_data_thredds_server(url_glider,var,scatter_plot,**kwargs)
                 
    var = 'salinity'  
    saltg, _, _, _, _, _ = \
                 read_glider_data_thredds_server(url_glider,var,scatter_plot,**kwargs)         
                 
    var = 'density'  
    densg, _, _, _, _, _ = \
                 read_glider_data_thredds_server(url_glider,var,scatter_plot,**kwargs)
                                             
    #%% Grid glider variables according to depth
    
    delta_z = 0.5
    depthg_gridded,tempg_gridded,saltg_gridded,densg_gridded = \
    varsg_gridded(depthg,timeg,tempg,saltg,densg,delta_z)
    
    #%% Get glider transect from GOFS 3.1
    
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
    
    #%% Retrieve glider transect from POM operational
    
    tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]             
    
    folder_pom = folder_pom_oper
    prefix = prefix_pom
    zlev = zlev_pom_oper
    zmatrix_pom = zmatrix_pom_oper
    lon_pom = lon_pom_oper
    lat_pom = lat_pom_oper
    tstamp_glider = tstamp_glider
    long = long
    latg = latg
    
    time_POM_oper, target_temp_POM_oper, target_salt_POM_oper, \
        target_dens_POM_oper, target_depth_POM_oper = \
        get_glider_transect_from_POM(folder_pom,prefix,zlev,zmatrix_pom,lon_pom,lat_pom,\
                                     tstamp_glider,long,latg)
            
    timestamp_POM_oper = mdates.date2num(time_POM_oper)
    
    '''
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
    ''' 
        
    #%% Retrieve glider transect from POM experimental

    tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]             
    
    folder_pom = folder_pom_exp
    prefix = prefix_pom
    zlev = zlev_pom_exp
    zmatrix_pom = zmatrix_pom_exp
    lon_pom = lon_pom_exp
    lat_pom = lat_pom_exp
    tstamp_glider = tstamp_glider
    long = long
    latg = latg
   
    time_POM_exp, target_temp_POM_exp, target_salt_POM_exp,\
        target_dens_POM_exp, target_depth_POM_exp = \
        get_glider_transect_from_POM(folder_pom,prefix,zlev,zmatrix_pom,lon_pom,lat_pom,\
                                     tstamp_glider,long,latg)
    timestamp_POM_exp = mdates.date2num(time_POM_exp)
            
    '''    
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
    '''    
    #%% Get glider transect from HYCOM
    
    folder_hycom = folder_hycom_exp
    prefix = prefix_hycom
        
    # Changing times to timestamp
    tstamp_glider = [mdates.date2num(timeg[i]) for i in np.arange(len(timeg))]
    
    # Conversion from glider longitude and latitude to GOFS convention
    target_lonG, target_latG = glider_coor_to_GOFS_coord(long,latg)
    
    lon_glider = target_lonG 
    lat_glider = target_latG
    
    var = 'temp'
    target_temp_HYCOM_exp, time_HYCOM_exp = \
        get_glider_transect_from_HYCOM(folder_hycom,prefix,nz,\
        lon_hycom,lat_hycom,var,tstamp_glider,lon_glider,lat_glider)

    var = 'salinity'
    target_salt_HYCOM_exp, _ = \
        get_glider_transect_from_HYCOM(folder_hycom,prefix,nz,\
          lon_hycom,lat_hycom,var,tstamp_glider,lon_glider,lat_glider)
        
    #%% Calculate density for HYCOM
    
    target_dens_HYCOM_exp = sw.dens(target_salt_HYCOM_exp,target_temp_HYCOM_exp,np.tile(depth_HYCOM_exp,(len(time_HYCOM_exp),1)).T) 
    
    #%% Calculation of mixed layer depth based on temperature and density critria
    # Tmean: mean temp within the mixed layer and 
    # td: temp at 1 meter below the mixed layer            
    
    dt = 0.2
    drho = 0.125
    
    # for glider data
    MLD_temp_crit_glid, _, _, _, MLD_dens_crit_glid, Tmean_dens_crit_glid, Smean_dens_crit_glid, _ = \
    MLD_temp_and_dens_criteria(dt,drho,timeg,depthg_gridded,tempg_gridded,saltg_gridded,densg_gridded)
    
    # for GOFS 3.1 output 
    MLD_temp_crit_GOFS, _, _, _, MLD_dens_crit_GOFS, Tmean_dens_crit_GOFS, Smean_dens_crit_GOFS, _ = \
    MLD_temp_and_dens_criteria(dt,drho,time_GOFS,depth_GOFS,target_temp_GOFS,target_salt_GOFS,target_dens_GOFS)          
    
    # for POM operational
    MLD_temp_crit_POM_oper, _, _, _, MLD_dens_crit_POM_oper, Tmean_dens_crit_POM_oper, Smean_dens_crit_POM_oper, _ = \
    MLD_temp_and_dens_criteria(dt,drho,timestamp_POM_oper,target_depth_POM_oper,target_temp_POM_oper,target_salt_POM_oper,target_dens_POM_oper)
    
    # for POM experimental
    MLD_temp_crit_POM_exp, _, _, _, MLD_dens_crit_POM_exp, Tmean_dens_crit_POM_exp, Smean_dens_crit_POM_exp, _ = \
    MLD_temp_and_dens_criteria(dt,drho,timestamp_POM_exp,target_depth_POM_exp,target_temp_POM_exp,target_salt_POM_exp,target_dens_POM_exp)
    
    # for HYCOM experimental
    timestamp_HYCOM_exp = mdates.date2num(time_HYCOM_exp)
    MLD_temp_crit_HYCOM_exp, _, _, _, MLD_dens_crit_HYCOM_exp, Tmean_dens_crit_HYCOM_exp, Smean_dens_crit_HYCOM_exp, _ = \
    MLD_temp_and_dens_criteria(dt,drho,timestamp_HYCOM_exp,depth_HYCOM_exp,target_temp_HYCOM_exp,target_salt_HYCOM_exp,target_dens_HYCOM_exp)
    
    #%% Surface Ocean Heat Content
    
    # glider
    OHC_glid = OHC_surface(timeg,tempg_gridded,depthg_gridded,densg_gridded)
    
    # GOFS
    OHC_GOFS = OHC_surface(time_GOFS,target_temp_GOFS,depth_GOFS,target_dens_GOFS)
    
    # POM operational    
    OHC_POM_oper = OHC_surface(timestamp_POM_oper,target_temp_POM_oper,target_depth_POM_oper,target_dens_POM_oper)
    
    # POM experimental
    OHC_POM_exp = OHC_surface(timestamp_POM_exp,target_temp_POM_exp,target_depth_POM_exp,target_dens_POM_exp)
    
    # HYCOM experimental
    OHC_HYCOM_exp = OHC_surface(timestamp_HYCOM_exp,target_temp_HYCOM_exp,depth_HYCOM_exp,target_dens_HYCOM_exp)
    
    #%% Calculate T100
    
    # glider
    T100_glid = depth_aver_top_100(depthg_gridded,tempg_gridded)

    # GOFS
    T100_GOFS = depth_aver_top_100(depth_GOFS,target_temp_GOFS)

    # POM operational
    T100_POM_oper = depth_aver_top_100(target_depth_POM_oper,target_temp_POM_oper) 

    # POM experimental
    T100_POM_exp = depth_aver_top_100(target_depth_POM_exp,target_temp_POM_exp)
    
    # HYCOM experimental
    T100_HYCOM_exp = depth_aver_top_100(depth_HYCOM_exp,target_temp_HYCOM_exp)
    
    #%% Calculate T100_to_ml
    
    dt = 0.2
    drho = 0.125
    
    # glider
    T100_to_ml_glid = depth_aver_temp_from_100_to_base_mixed_layer\
                      (dt,drho,timeg,depthg_gridded,tempg_gridded,saltg_gridded,densg_gridded)
    
    # GOFS
    T100_to_ml_GOFS = depth_aver_temp_from_100_to_base_mixed_layer\
                     (dt,drho,time_GOFS,depth_GOFS,target_temp_GOFS,target_salt_GOFS,target_dens_GOFS)

    # POM operational
    T100_to_ml_POM_oper = depth_aver_temp_from_100_to_base_mixed_layer\
                          (dt,drho,timestamp_POM_oper,target_depth_POM_oper,target_temp_POM_oper,target_salt_POM_oper,target_dens_POM_oper) 

    # POM experimental
    T100_to_ml_POM_exp = depth_aver_temp_from_100_to_base_mixed_layer\
                         (dt,drho,timestamp_POM_exp,target_depth_POM_exp,target_temp_POM_exp,target_salt_POM_exp,target_dens_POM_exp) 
                         
    # POM experimental
    T100_to_ml_HYCOM_exp = depth_aver_temp_from_100_to_base_mixed_layer\
                         (dt,drho,timestamp_HYCOM_exp,depth_HYCOM_exp,target_temp_HYCOM_exp,target_salt_HYCOM_exp,target_dens_HYCOM_exp) 
        
    #%% Interpolate glider transect onto GOFS time and depth
        
    oktimeg_gofs = np.round(np.interp(tstamp_model,tstamp_glider,np.arange(len(tstamp_glider)))).astype(int)    
    
    temp_orig = tempg
    salt_orig = saltg 
    depth_orig = depthg
    depth_target = depth_GOFS
    indexes_time = oktimeg_gofs
    dim_vars = [target_temp_GOFS.shape[0],target_temp_GOFS.shape[1]]
    
    tempg_to_GOFS, saltg_to_GOFS = \
    interp_datasets_in_z(temp_orig,salt_orig,depth_orig,depth_target,indexes_time,dim_vars)
    
    MLD_dens_crit_glid_to_GOFS = MLD_dens_crit_glid[oktimeg_gofs]    
    Tmean_dens_crit_glid_to_GOFS = Tmean_dens_crit_glid[oktimeg_gofs] 
    Smean_dens_crit_glid_to_GOFS = Smean_dens_crit_glid[oktimeg_gofs] 
    OHC_glid_to_GOFS = OHC_glid[oktimeg_gofs] 
    T100_glid_to_GOFS = T100_glid[oktimeg_gofs] 
    T100_to_ml_glid_to_GOFS = T100_to_ml_glid[oktimeg_gofs] 
    
    #%% Interpolate glider transect onto POM operational time and depth
        
    oktimeg_pom_oper = np.round(np.interp(timestamp_POM_oper,tstamp_glider,np.arange(len(tstamp_glider)))).astype(int)   
    
    temp_orig = tempg
    salt_orig = saltg 
    depth_orig = depthg
    depth_target = -target_depth_POM_oper
    indexes_time = oktimeg_pom_oper
    dim_vars = [target_temp_POM_oper.shape[0],target_temp_POM_oper.shape[1]]
    
    tempg_to_POM_oper, saltg_to_POM_oper = \
    interp_datasets_in_z(temp_orig,salt_orig,depth_orig,depth_target,indexes_time,dim_vars)
    
    MLD_dens_crit_glid_to_POM_oper = MLD_dens_crit_glid[oktimeg_pom_oper]    
    Tmean_dens_crit_glid_to_POM_oper = Tmean_dens_crit_glid[oktimeg_pom_oper] 
    Smean_dens_crit_glid_to_POM_oper = Smean_dens_crit_glid[oktimeg_pom_oper] 
    OHC_glid_to_POM_oper = OHC_glid[oktimeg_pom_oper]
    T100_glid_to_POM_oper = T100_glid[oktimeg_pom_oper]
    T100_to_ml_glid_to_POM_oper = T100_to_ml_glid[oktimeg_pom_oper]
    
    #%% Interpolate glider transect onto POM experimental time and depth
        
    oktimeg_pom_exp = np.round(np.interp(timestamp_POM_exp,tstamp_glider,np.arange(len(tstamp_glider)))).astype(int)  
    
    temp_orig = tempg
    salt_orig = saltg 
    depth_orig = depthg
    depth_target = -target_depth_POM_exp
    indexes_time = oktimeg_pom_exp
    dim_vars = [target_temp_POM_exp.shape[0],target_temp_POM_exp.shape[1]]
    
    tempg_to_POM_exp, saltg_to_POM_exp = \
    interp_datasets_in_z(temp_orig,salt_orig,depth_orig,depth_target,indexes_time,dim_vars)
    
    MLD_dens_crit_glid_to_POM_exp = MLD_dens_crit_glid[oktimeg_pom_exp]    
    Tmean_dens_crit_glid_to_POM_exp = Tmean_dens_crit_glid[oktimeg_pom_exp] 
    Smean_dens_crit_glid_to_POM_exp = Smean_dens_crit_glid[oktimeg_pom_exp] 
    OHC_glid_to_POM_exp = OHC_glid[oktimeg_pom_exp]
    T100_glid_to_POM_exp = T100_glid[oktimeg_pom_exp]
    T100_to_ml_glid_to_POM_exp = T100_to_ml_glid[oktimeg_pom_exp]
    
    
    #%% Interpolate glider transect onto HYCOM experimental time and depth
        
    oktimeg_hycom_exp = np.round(np.interp(timestamp_HYCOM_exp,tstamp_glider,np.arange(len(tstamp_glider)))).astype(int)  
    
    temp_orig = tempg
    salt_orig = saltg 
    depth_orig = depthg
    depth_target = depth_HYCOM_exp
    indexes_time = oktimeg_hycom_exp
    dim_vars = [target_temp_HYCOM_exp.shape[0],target_temp_HYCOM_exp.shape[1]]
    
    tempg_to_HYCOM_exp, saltg_to_HYCOM_exp = \
    interp_datasets_in_z(temp_orig,salt_orig,depth_orig,depth_target,indexes_time,dim_vars)
    
    MLD_dens_crit_glid_to_HYCOM_exp = MLD_dens_crit_glid[oktimeg_hycom_exp]    
    Tmean_dens_crit_glid_to_HYCOM_exp = Tmean_dens_crit_glid[oktimeg_hycom_exp] 
    Smean_dens_crit_glid_to_HYCOM_exp = Smean_dens_crit_glid[oktimeg_hycom_exp] 
    OHC_glid_to_HYCOM_exp = OHC_glid[oktimeg_hycom_exp]
    T100_glid_to_HYCOM_exp = T100_glid[oktimeg_hycom_exp]
    T100_to_ml_glid_to_HYCOM_exp = T100_to_ml_glid[oktimeg_hycom_exp]
    
    #%% Define dataframe
    
    df_GOFS_temp_salt = pd.DataFrame(data=np.array([np.ravel(tempg_to_GOFS,order='F'),\
                                          np.ravel(target_temp_GOFS,order='F'),\
                                          np.ravel(saltg_to_GOFS,order='F'),\
                                          np.ravel(target_salt_GOFS,order='F'),\
                                          ]).T,\
                      columns=['temp_obs','temp_GOFS',\
                               'salt_obs','salt_GOFS'])
    
    #%% Define dataframe
    
    df_POM_temp_salt = pd.DataFrame(data=np.array([np.ravel(tempg_to_POM_oper,order='F'),\
                                         np.ravel(tempg_to_POM_exp,order='F'),\
                                         np.ravel(target_temp_POM_oper,order='F'),\
                                         np.ravel(target_temp_POM_exp,order='F'),\
                                         np.ravel(saltg_to_POM_oper,order='F'),\
                                         np.ravel(saltg_to_POM_exp,order='F'),\
                                         np.ravel(target_salt_POM_oper,order='F'),\
                                         np.ravel(target_salt_POM_exp,order='F')
                                         ]).T,\
                      columns=['temp_obs_to_oper','temp_obs_to_exp',\
                               'temp_POM_oper','temp_POM_exp',\
                               'salt_obs_to_oper','salt_obs_to_exp',\
                               'salt_POM_oper','salt_POM_exp'])
        
    #%% Define dataframe
    
    df_HYCOM_temp_salt = pd.DataFrame(data=np.array([np.ravel(tempg_to_HYCOM_exp,order='F'),\
                                          np.ravel(target_temp_HYCOM_exp,order='F'),\
                                          np.ravel(saltg_to_HYCOM_exp,order='F'),\
                                          np.ravel(target_salt_HYCOM_exp,order='F'),\
                                          ]).T,\
                      columns=['temp_obs_to_exp','temp_HYCOM_exp',\
                               'salt_obs_to_exp','salt_HYCOM_exp'])
        
    #%% Define dataframe
    
    df_GOFS_MLD = pd.DataFrame(data=np.array([MLD_dens_crit_glid_to_GOFS,MLD_dens_crit_GOFS,\
                                              Tmean_dens_crit_glid_to_GOFS,Tmean_dens_crit_GOFS,
                                              Smean_dens_crit_glid_to_GOFS,Smean_dens_crit_GOFS]).T,\
                      columns=['MLD_obs','MLD_GOFS',\
                              'Tmean_obs','Tmean_GOFS',
                              'Smean_obs','Smean_GOFS'])  
        
    #%% Define dataframe
    
    df_POM_MLD = pd.DataFrame(data=np.array([MLD_dens_crit_glid_to_POM_oper,MLD_dens_crit_POM_oper,\
                                             Tmean_dens_crit_glid_to_POM_oper,Tmean_dens_crit_POM_oper,\
                                             Smean_dens_crit_glid_to_POM_oper,Smean_dens_crit_POM_oper,\
                                             MLD_dens_crit_glid_to_POM_exp,MLD_dens_crit_POM_exp,\
                                             Tmean_dens_crit_glid_to_POM_exp,Tmean_dens_crit_POM_exp,\
                                             Smean_dens_crit_glid_to_POM_exp,Smean_dens_crit_POM_exp]).T,\
                      columns=['MLD_obs_to_oper','MLD_POM_oper',\
                              'Tmean_obs_to_oper','Tmean_POM_oper',\
                              'Smean_obs_to_oper','Smean_POM_oper',\
                              'MLD_obs_to_exp','MLD_POM_exp',\
                              'Tmean_obs_to_exp','Tmean_POM_exp',\
                              'Smean_obs_to_exp','Smean_POM_exp']) 
        
    #%% Define dataframe
    
    df_HYCOM_MLD = pd.DataFrame(data=np.array([MLD_dens_crit_glid_to_HYCOM_exp,MLD_dens_crit_HYCOM_exp,\
                                              Tmean_dens_crit_glid_to_HYCOM_exp,Tmean_dens_crit_HYCOM_exp,
                                              Smean_dens_crit_glid_to_HYCOM_exp,Smean_dens_crit_HYCOM_exp]).T,\
                      columns=['MLD_obs','MLD_HYCOM_exp',\
                              'Tmean_obs_to_exp','Tmean_HYCOM_exp',
                              'Smean_obs_to_exp','Smean_HYCOM_exp'])  
    
    #%% Define dataframe
    
    df_GOFS_OHC = pd.DataFrame(data=np.array([OHC_glid_to_GOFS,OHC_GOFS]).T,\
                      columns=['OHC_obs','OHC_GOFS'])  
    
    #%% Define dataframe
    
    df_POM_OHC = pd.DataFrame(data=np.array([OHC_glid_to_POM_oper,OHC_glid_to_POM_exp,\
                                              OHC_POM_oper,OHC_POM_exp]).T,\
                      columns=['OHC_obs_to_oper','OHC_obs_to_exp',\
                               'OHC_POM_oper','OHC_POM_exp'])
        
    #%% Define dataframe
    
    df_HYCOM_OHC = pd.DataFrame(data=np.array([OHC_glid_to_HYCOM_exp,OHC_HYCOM_exp]).T,\
                      columns=['OHC_obs_to_exp','OHC_HYCOM_exp'])       
        
    #%% Define dataframe
    
    df_GOFS_T100 = pd.DataFrame(data=np.array([T100_glid_to_GOFS,T100_GOFS]).T,\
                      columns=['T100_obs','T100_GOFS'])  
        
    #%% Define dataframe
    
    df_POM_T100 = pd.DataFrame(data=np.array([T100_glid_to_POM_oper,T100_glid_to_POM_exp,\
                                              T100_POM_oper,T100_POM_exp]).T,\
                      columns=['T100_obs_to_oper','T100_obs_to_exp',\
                               'T100_POM_oper','T100_POM_exp'])
        
    #%% Define dataframe
    
    df_HYCOM_T100 = pd.DataFrame(data=np.array([T100_glid_to_HYCOM_exp,T100_HYCOM_exp]).T,\
                      columns=['T100_obs_to_exp','T100_HYCOM_exp'])  
        
    #%% Define dataframe
    
    df_GOFS_T100_to_ml = pd.DataFrame(data=np.array([T100_to_ml_glid_to_GOFS,T100_to_ml_GOFS]).T,\
                      columns=['T100_to_ml_obs','T100_to_ml_GOFS'])  
        
    #%% Define dataframe
    
    df_POM_T100_to_ml = pd.DataFrame(data=np.array([T100_to_ml_glid_to_POM_oper,T100_to_ml_glid_to_POM_exp,\
                                              T100_to_ml_POM_oper,T100_to_ml_POM_exp]).T,\
                      columns=['T100_to_ml_obs_to_oper','T100_to_ml_obs_to_exp',\
                               'T100_to_ml_POM_oper','T100_to_ml_POM_exp'])
        
    #%% Define dataframe
    
    df_HYCOM_T100_to_ml = pd.DataFrame(data=np.array([T100_to_ml_glid_to_HYCOM_exp,T100_to_ml_HYCOM_exp]).T,\
                      columns=['T100_to_ml_obs_to_exp','T100_to_ml_HYCOM_exp'])  

    #%% Concatenate data frames       
    
    DF_GOFS_temp_salt = pd.concat([DF_GOFS_temp_salt, df_GOFS_temp_salt])
    DF_POM_temp_salt = pd.concat([DF_POM_temp_salt, df_POM_temp_salt])
    DF_HYCOM_temp_salt = pd.concat([DF_HYCOM_temp_salt, df_HYCOM_temp_salt])
    DF_GOFS_MLD = pd.concat([DF_GOFS_MLD, df_GOFS_MLD])
    DF_POM_MLD = pd.concat([DF_POM_MLD, df_POM_MLD])
    DF_HYCOM_MLD = pd.concat([DF_HYCOM_MLD, df_HYCOM_MLD])
    DF_GOFS_OHC = pd.concat([DF_GOFS_OHC, df_GOFS_OHC])
    DF_POM_OHC = pd.concat([DF_POM_OHC, df_POM_OHC])
    DF_HYCOM_OHC = pd.concat([DF_HYCOM_OHC, df_HYCOM_OHC])
    DF_GOFS_T100 = pd.concat([DF_GOFS_T100, df_GOFS_T100])
    DF_POM_T100 = pd.concat([DF_POM_T100, df_POM_T100])
    DF_HYCOM_T100 = pd.concat([DF_HYCOM_T100, df_HYCOM_T100])
    DF_GOFS_T100_to_ml = pd.concat([DF_GOFS_T100_to_ml, df_GOFS_T100_to_ml])
    DF_POM_T100_to_ml = pd.concat([DF_POM_T100_to_ml, df_POM_T100_to_ml])
    DF_HYCOM_T100_to_ml = pd.concat([DF_HYCOM_T100_to_ml, df_HYCOM_T100_to_ml])

#%% Save all data frames 
'''
import feather  

feather.write_dataframe(DF_GOFS_temp_salt,'DF_GOFS_temp_salt.feather')
feather.write_dataframe(DF_GOFS_MLD,'DF_GOFS_MLD.feather')
feather.write_dataframe(DF_GOFS_OHC,'DF_GOFS_OHC.feather')
feather.write_dataframe(DF_GOFS_T100,'DF_GOFS_T100.feather')
feather.write_dataframe(DF_GOFS_T100_to_ml,'DF_GOFS_T100_to_ml.feather')

feather.write_dataframe(DF_POM_temp_salt,'DF_POM_temp_salt.feather')
feather.write_dataframe(DF_POM_MLD,'DF_POM_MLD.feather')
feather.write_dataframe(DF_POM_OHC,'DF_POM_OHC.feather')
feather.write_dataframe(DF_POM_T100,'DF_POM_T100.feather')
feather.write_dataframe(DF_POM_T100_to_ml,'DF_POM_T100_to_ml.feather') 

feather.write_dataframe(DF_HYCOM_temp_salt,'DF_HYCOM_temp_salt.feather')
feather.write_dataframe(DF_HYCOM_MLD,'DF_HYCOM_MLD.feather')
feather.write_dataframe(DF_HYCOM_OHC,'DF_HYCOM_OHC.feather')
feather.write_dataframe(DF_HYCOM_T100,'DF_HYCOM_T100.feather')
feather.write_dataframe(DF_HYCOM_T100_to_ml,'DF_HYCOM_T100_to_ml.feather')
'''

#%% Load all data frames 
'''
import feather

DF_GOFS_temp_salt = feather.read_dataframe('DF_GOFS_temp_salt.feather')
DF_GOFS_MLD = feather.read_dataframe('DF_GOFS_MLD.feather')
DF_GOFS_OHC = feather.read_dataframe('DF_GOFS_OHC.feather')
DF_GOFS_T100 = feather.read_dataframe('DF_GOFS_T100.feather')
DF_GOFS_T100_to_ml = feather.read_dataframe('DF_GOFS_T100_to_ml.feather')

DF_POM_temp_salt = feather.read_dataframe('DF_POM_temp_salt.feather')
DF_POM_MLD = feather.read_dataframe('DF_POM_MLD.feather')
DF_POM_OHC = feather.read_dataframe('DF_POM_OHC.feather')
DF_POM_T100 = feather.read_dataframe('DF_POM_T100.feather')
DF_POM_T100_to_ml = feather.read_dataframe('DF_POM_T100_to_ml.feather')

DF_HYCOM_temp_salt = feather.read_dataframe('DF_HYCOM_temp_salt.feather')
DF_HYCOM_MLD = feather.read_dataframe('DF_HYCOM_MLD.feather')
DF_HYCOM_OHC = feather.read_dataframe('DF_HYCOM_OHC.feather')
DF_HYCOM_T100 = feather.read_dataframe('DF_HYCOM_T100.feather')
DF_HYCOM_T100_to_ml = feather.read_dataframe('DF_HYCOM_T100_to_ml.feather')

''' 
#%% Temperature statistics.

DF_GOFS = DF_GOFS_temp_salt.dropna()
DF_POM = DF_POM_temp_salt.dropna()
DF_HYCOM = DF_HYCOM_temp_salt.dropna()
       
NGOFS = len(DF_GOFS)-1  #For Unbiased estimmator.
NPOM = len(DF_POM)-1
NHYCOM = len(DF_HYCOM)-1

cols = ['CORRELATION','OSTD','MSTD','CRMSE','BIAS']
    
tskill = np.empty((4,5))
tskill[:] = np.nan

#CORR
tskill[0,0] = DF_GOFS.corr()['temp_obs']['temp_GOFS']
tskill[1,0] = DF_POM.corr()['temp_obs_to_oper']['temp_POM_oper']
tskill[2,0] = DF_POM.corr()['temp_obs_to_exp']['temp_POM_exp']
tskill[3,0] = DF_HYCOM.corr()['temp_obs_to_exp']['temp_HYCOM_exp']

#OSTD
tskill[0,1] = DF_GOFS.std().temp_obs
tskill[1,1] = DF_POM.std().temp_obs_to_oper
tskill[2,1] = DF_POM.std().temp_obs_to_exp
tskill[3,1] = DF_HYCOM.std().temp_obs_to_exp

#MSTD
tskill[0,2] = DF_GOFS.std().temp_GOFS
tskill[1,2] = DF_POM.std().temp_POM_oper
tskill[2,2] = DF_POM.std().temp_POM_exp
tskill[3,2] = DF_HYCOM.std().temp_HYCOM_exp

#CRMSE
tskill[0,3] = np.sqrt(np.nansum(((DF_GOFS.temp_obs-DF_GOFS.mean().temp_obs)-\
                                 (DF_GOFS.temp_GOFS-DF_GOFS.mean().temp_GOFS))**2)/NGOFS)
tskill[1,3] = np.sqrt(np.nansum(((DF_POM.temp_obs_to_exp-DF_POM.mean().temp_obs_to_oper)-\
                                 (DF_POM.temp_POM_oper-DF_POM.mean().temp_POM_oper))**2)/NPOM)
tskill[2,3] = np.sqrt(np.nansum(((DF_POM.temp_obs_to_exp-DF_POM.mean().temp_obs_to_exp)-\
                                 (DF_POM.temp_POM_exp-DF_POM.mean().temp_POM_exp))**2)/NPOM)
tskill[3,3] = np.sqrt(np.nansum(((DF_HYCOM.temp_obs_to_exp-DF_HYCOM.mean().temp_obs_to_exp)-\
                                 (DF_HYCOM.temp_HYCOM_exp-DF_HYCOM.mean().temp_HYCOM_exp))**2)/NHYCOM)

#BIAS
tskill[0,4] = DF_GOFS.mean().temp_obs - DF_GOFS.mean().temp_GOFS
tskill[1,4] = DF_POM.mean().temp_obs_to_oper - DF_POM.mean().temp_POM_oper
tskill[2,4] = DF_POM.mean().temp_obs_to_exp - DF_POM.mean().temp_POM_exp
tskill[3,4] = DF_HYCOM.mean().temp_obs_to_exp - DF_HYCOM.mean().temp_HYCOM_exp

#color
colors = ['indianred','seagreen','darkorchid','darkorange']
    
temp_skillscores = pd.DataFrame(tskill,
                        index=['GOFS','POM_oper','POM_exp','HYCOM exp'],
                        columns=cols)
print(temp_skillscores)

#%% Salinity statistics.

DF_GOFS = DF_GOFS_temp_salt.dropna()
DF_POM = DF_POM_temp_salt.dropna()
DF_HYCOM = DF_HYCOM_temp_salt.dropna()

NGOFS = len(DF_GOFS)-1  #For Unbiased estimmator.
NPOM = len(DF_POM)-1
NHYCOM = len(DF_HYCOM)-1

cols = ['CORRELATION','OSTD','MSTD','CRMSE','BIAS']
    
tskill = np.empty((4,5))
tskill[:] = np.nan

#CORR
tskill[0,0] = DF_GOFS.corr()['salt_obs']['salt_GOFS']
tskill[1,0] = DF_POM.corr()['salt_obs_to_oper']['salt_POM_oper']
tskill[2,0] = DF_POM.corr()['salt_obs_to_exp']['salt_POM_exp']
tskill[3,0] = DF_HYCOM.corr()['salt_obs_to_exp']['salt_HYCOM_exp']

#OSTD
tskill[0,1] = DF_GOFS.std().salt_obs
tskill[1,1] = DF_POM.std().salt_obs_to_oper
tskill[2,1] = DF_POM.std().salt_obs_to_exp
tskill[3,1] = DF_HYCOM.std().salt_obs_to_exp

#MSTD
tskill[0,2] = DF_GOFS.std().salt_GOFS
tskill[1,2] = DF_POM.std().salt_POM_oper
tskill[2,2] = DF_POM.std().salt_POM_exp
tskill[3,2] = DF_HYCOM.std().salt_HYCOM_exp

#CRMSE
tskill[0,3] = np.sqrt(np.nansum(((DF_GOFS.salt_obs-DF_GOFS.mean().salt_obs)-\
                                 (DF_GOFS.salt_GOFS-DF_GOFS.mean().salt_GOFS))**2)/NGOFS)
tskill[1,3] = np.sqrt(np.nansum(((DF_POM.salt_obs_to_exp-DF_POM.mean().salt_obs_to_oper)-\
                                 (DF_POM.salt_POM_oper-DF_POM.mean().salt_POM_oper))**2)/NPOM)
tskill[2,3] = np.sqrt(np.nansum(((DF_POM.salt_obs_to_exp-DF_POM.mean().salt_obs_to_exp)-\
                                 (DF_POM.salt_POM_exp-DF_POM.mean().salt_POM_exp))**2)/NPOM)
tskill[3,3] = np.sqrt(np.nansum(((DF_HYCOM.salt_obs_to_exp-DF_HYCOM.mean().salt_obs_to_exp)-\
                                 (DF_HYCOM.salt_HYCOM_exp-DF_HYCOM.mean().salt_HYCOM_exp))**2)/NHYCOM)    

#BIAS
tskill[0,4] = DF_GOFS.mean().salt_obs - DF_GOFS.mean().salt_GOFS
tskill[1,4] = DF_POM.mean().salt_obs_to_oper - DF_POM.mean().salt_POM_oper
tskill[2,4] = DF_POM.mean().salt_obs_to_exp - DF_POM.mean().salt_POM_exp
tskill[3,4] = DF_HYCOM.mean().salt_obs_to_exp - DF_HYCOM.mean().salt_HYCOM_exp

#color
#color
colors = ['indianred','seagreen','darkorchid','darkorange']
    
salt_skillscores = pd.DataFrame(tskill,
                        index=['GOFS','POM_oper','POM_exp','HYCOM_exp'],
                        columns=cols)
print(salt_skillscores)

#%% Mixed layer statistics Temperature.

DF_GOFS = DF_GOFS_MLD.dropna()
DF_POM = DF_POM_MLD.dropna()
DF_HYCOM = DF_HYCOM_MLD.dropna()

NGOFS = len(DF_GOFS)-1  #For Unbiased estimmator.
NPOM = len(DF_POM)-1
NHYCOM = len(DF_HYCOM)-1

cols = ['CORRELATION','OSTD','MSTD','CRMSE','BIAS']
    
tskill = np.empty((4,5))
tskill[:] = np.nan

#CORR
tskill[0,0] = DF_GOFS.corr()['Tmean_obs']['Tmean_GOFS']
tskill[1,0] = DF_POM.corr()['Tmean_obs_to_oper']['Tmean_POM_oper']
tskill[2,0] = DF_POM.corr()['Tmean_obs_to_exp']['Tmean_POM_exp']
tskill[3,0] = DF_HYCOM.corr()['Tmean_obs_to_exp']['Tmean_HYCOM_exp']

#OSTD
tskill[0,1] = DF_GOFS.std().Tmean_obs
tskill[1,1] = DF_POM.std().Tmean_obs_to_oper
tskill[2,1] = DF_POM.std().Tmean_obs_to_exp
tskill[3,1] = DF_HYCOM.std().Tmean_obs_to_exp

#MSTD
tskill[0,2] = DF_GOFS.std().Tmean_GOFS
tskill[1,2] = DF_POM.std().Tmean_POM_oper
tskill[2,2] = DF_POM.std().Tmean_POM_exp
tskill[3,2] = DF_HYCOM.std().Tmean_HYCOM_exp

#CRMSE
tskill[0,3] = np.sqrt(np.nansum(((DF_GOFS.Tmean_obs-DF_GOFS.mean().Tmean_obs)-\
                                 (DF_GOFS.Tmean_GOFS-DF_GOFS.mean().Tmean_GOFS))**2)/NGOFS)
tskill[1,3] = np.sqrt(np.nansum(((DF_POM.Tmean_obs_to_exp-DF_POM.mean().Tmean_obs_to_oper)-\
                                 (DF_POM.Tmean_POM_oper-DF_POM.mean().Tmean_POM_oper))**2)/NPOM)
tskill[2,3] = np.sqrt(np.nansum(((DF_POM.Tmean_obs_to_exp-DF_POM.mean().Tmean_obs_to_exp)-\
                                 (DF_POM.Tmean_POM_exp-DF_POM.mean().Tmean_POM_exp))**2)/NPOM)
tskill[3,3] = np.sqrt(np.nansum(((DF_HYCOM.Tmean_obs_to_exp-DF_HYCOM.mean().Tmean_obs_to_exp)-\
                                 (DF_HYCOM.Tmean_HYCOM_exp-DF_HYCOM.mean().Tmean_HYCOM_exp))**2)/NHYCOM)

#BIAS
tskill[0,4] = DF_GOFS.mean().Tmean_obs - DF_GOFS.mean().Tmean_GOFS
tskill[1,4] = DF_POM.mean().Tmean_obs_to_oper - DF_POM.mean().Tmean_POM_oper
tskill[2,4] = DF_POM.mean().Tmean_obs_to_exp - DF_POM.mean().Tmean_POM_exp
tskill[3,4] = DF_HYCOM.mean().Tmean_obs_to_exp - DF_HYCOM.mean().Tmean_HYCOM_exp

# colors
colors = ['indianred','seagreen','darkorchid','darkorange']
    
Tmean_mld_skillscores = pd.DataFrame(tskill,
                        index=['GOFS','POM_oper','POM_exp', 'HYCOM_exp'],
                        columns=cols)
print(Tmean_mld_skillscores)

#%% Mixed layer statistics Salinity.

DF_GOFS = DF_GOFS_MLD.dropna()
DF_POM = DF_POM_MLD.dropna()
DF_HYCOM = DF_HYCOM_MLD.dropna()

NGOFS = len(DF_GOFS)-1  #For Unbiased estimmator.
NPOM = len(DF_POM)-1
NHYCOM = len(DF_HYCOM)-1

cols = ['CORRELATION','OSTD','MSTD','CRMSE','BIAS']
    
tskill = np.empty((4,5))
tskill[:] = np.nan

#CORR
tskill[0,0] = DF_GOFS.corr()['Smean_obs']['Smean_GOFS']
tskill[1,0] = DF_POM.corr()['Smean_obs_to_oper']['Smean_POM_oper']
tskill[2,0] = DF_POM.corr()['Smean_obs_to_exp']['Smean_POM_exp']
tskill[3,0] = DF_HYCOM.corr()['Smean_obs_to_exp']['Smean_HYCOM_exp']

#OSTD
tskill[0,1] = DF_GOFS.std().Smean_obs
tskill[1,1] = DF_POM.std().Smean_obs_to_oper
tskill[2,1] = DF_POM.std().Smean_obs_to_exp
tskill[3,1] = DF_HYCOM.std().Smean_obs_to_exp

#MSTD
tskill[0,2] = DF_GOFS.std().Smean_GOFS
tskill[1,2] = DF_POM.std().Smean_POM_oper
tskill[2,2] = DF_POM.std().Smean_POM_exp
tskill[3,2] = DF_HYCOM.std().Smean_HYCOM_exp

#CRMSE
tskill[0,3] = np.sqrt(np.nansum(((DF_GOFS.Smean_obs-DF_GOFS.mean().Smean_obs)-\
                                 (DF_GOFS.Smean_GOFS-DF_GOFS.mean().Smean_GOFS))**2)/NGOFS)
tskill[1,3] = np.sqrt(np.nansum(((DF_POM.Smean_obs_to_exp-DF_POM.mean().Smean_obs_to_oper)-\
                                 (DF_POM.Smean_POM_oper-DF_POM.mean().Smean_POM_oper))**2)/NPOM)
tskill[2,3] = np.sqrt(np.nansum(((DF_POM.Smean_obs_to_exp-DF_POM.mean().Smean_obs_to_exp)-\
                                 (DF_POM.Smean_POM_exp-DF_POM.mean().Smean_POM_exp))**2)/NPOM)
tskill[3,3] = np.sqrt(np.nansum(((DF_HYCOM.Smean_obs_to_exp-DF_HYCOM.mean().Smean_obs_to_exp)-\
                                 (DF_HYCOM.Smean_HYCOM_exp-DF_HYCOM.mean().Smean_HYCOM_exp))**2)/NHYCOM)

#BIAS
tskill[0,4] = DF_GOFS.mean().Smean_obs - DF_GOFS.mean().Smean_GOFS
tskill[1,4] = DF_POM.mean().Smean_obs_to_oper - DF_POM.mean().Smean_POM_oper
tskill[2,4] = DF_POM.mean().Smean_obs_to_exp - DF_POM.mean().Smean_POM_exp
tskill[3,4] = DF_HYCOM.mean().Smean_obs_to_exp - DF_HYCOM.mean().Smean_HYCOM_exp

# colors
colors = ['indianred','seagreen','darkorchid','darkorange']
    
Smean_mld_skillscores = pd.DataFrame(tskill,
                        index=['GOFS','POM_oper','POM_exp', 'HYCOM_exp'],
                        columns=cols)
print(Smean_mld_skillscores)

#%% OHC statistics 

DF_GOFS = DF_GOFS_OHC.dropna()
DF_POM = DF_POM_OHC.dropna()
DF_HYCOM = DF_HYCOM_OHC.dropna()

NGOFS = len(DF_GOFS)-1  #For Unbiased estimmator.
NPOM = len(DF_POM)-1
NHYCOM = len(DF_HYCOM)-1

cols = ['CORRELATION','OSTD','MSTD','CRMSE','BIAS']
    
tskill = np.empty((4,5))
tskill[:] = np.nan

#CORR
tskill[0,0] = DF_GOFS.corr()['OHC_obs']['OHC_GOFS']
tskill[1,0] = DF_POM.corr()['OHC_obs_to_oper']['OHC_POM_oper']
tskill[2,0] = DF_POM.corr()['OHC_obs_to_exp']['OHC_POM_exp']
tskill[3,0] = DF_HYCOM.corr()['OHC_obs_to_exp']['OHC_HYCOM_exp']

#OSTD
tskill[0,1] = DF_GOFS.std().OHC_obs
tskill[1,1] = DF_POM.std().OHC_obs_to_oper
tskill[2,1] = DF_POM.std().OHC_obs_to_exp
tskill[3,1] = DF_HYCOM.std().OHC_obs_to_exp

#MSTD
tskill[0,2] = DF_GOFS.std().OHC_GOFS
tskill[1,2] = DF_POM.std().OHC_POM_oper
tskill[2,2] = DF_POM.std().OHC_POM_exp
tskill[3,2] = DF_HYCOM.std().OHC_HYCOM_exp

#CRMSE
tskill[0,3] = np.sqrt(np.nansum(((DF_GOFS.OHC_obs-DF_GOFS.mean().OHC_obs)-\
                                 (DF_GOFS.OHC_GOFS-DF_GOFS.mean().OHC_GOFS))**2)/NGOFS)
tskill[1,3] = np.sqrt(np.nansum(((DF_POM.OHC_obs_to_exp-DF_POM.mean().OHC_obs_to_oper)-\
                                 (DF_POM.OHC_POM_oper-DF_POM.mean().OHC_POM_oper))**2)/NPOM)
tskill[2,3] = np.sqrt(np.nansum(((DF_POM.OHC_obs_to_exp-DF_POM.mean().OHC_obs_to_exp)-\
                                 (DF_POM.OHC_POM_exp-DF_POM.mean().OHC_POM_exp))**2)/NPOM)
tskill[3,3] = np.sqrt(np.nansum(((DF_HYCOM.OHC_obs_to_exp-DF_HYCOM.mean().OHC_obs_to_exp)-\
                                 (DF_HYCOM.OHC_HYCOM_exp-DF_HYCOM.mean().OHC_HYCOM_exp))**2)/NHYCOM)

#BIAS
tskill[0,4] = DF_GOFS.mean().OHC_obs - DF_GOFS.mean().OHC_GOFS
tskill[1,4] = DF_POM.mean().OHC_obs_to_oper - DF_POM.mean().OHC_POM_oper
tskill[2,4] = DF_POM.mean().OHC_obs_to_exp - DF_POM.mean().OHC_POM_exp
tskill[3,4] = DF_HYCOM.mean().OHC_obs_to_exp - DF_HYCOM.mean().OHC_HYCOM_exp

#color
colors = ['indianred','seagreen','darkorchid','darkorange']
    
OHC_skillscores = pd.DataFrame(tskill,
                        index=['GOFS','POM_oper','POM_exp','HYCOM_exp'],
                        columns=cols)
print(OHC_skillscores)

#%% T100 statistics 

DF_GOFS = DF_GOFS_T100.dropna()
DF_POM = DF_POM_T100.dropna()
DF_HYCOM = DF_HYCOM_T100.dropna()

NGOFS = len(DF_GOFS)-1  #For Unbiased estimmator.
NPOM = len(DF_POM)-1
NHYCOM = len(DF_HYCOM)-1

cols = ['CORRELATION','OSTD','MSTD','CRMSE','BIAS']
    
tskill = np.empty((4,5))
tskill[:] = np.nan

#CORR
tskill[0,0] = DF_GOFS.corr()['T100_obs']['T100_GOFS']
tskill[1,0] = DF_POM.corr()['T100_obs_to_oper']['T100_POM_oper']
tskill[2,0] = DF_POM.corr()['T100_obs_to_exp']['T100_POM_exp']
tskill[3,0] = DF_HYCOM.corr()['T100_obs_to_exp']['T100_HYCOM_exp']

#OSTD
tskill[0,1] = DF_GOFS.std().T100_obs
tskill[1,1] = DF_POM.std().T100_obs_to_oper
tskill[2,1] = DF_POM.std().T100_obs_to_exp
tskill[3,1] = DF_HYCOM.std().T100_obs_to_exp

#MSTD
tskill[0,2] = DF_GOFS.std().T100_GOFS
tskill[1,2] = DF_POM.std().T100_POM_oper
tskill[2,2] = DF_POM.std().T100_POM_exp
tskill[3,2] = DF_HYCOM.std().T100_HYCOM_exp

#CRMSE
tskill[0,3] = np.sqrt(np.nansum(((DF_GOFS.T100_obs-DF_GOFS.mean().T100_obs)-\
                                 (DF_GOFS.T100_GOFS-DF_GOFS.mean().T100_GOFS))**2)/NGOFS)
tskill[1,3] = np.sqrt(np.nansum(((DF_POM.T100_obs_to_exp-DF_POM.mean().T100_obs_to_oper)-\
                                 (DF_POM.T100_POM_oper-DF_POM.mean().T100_POM_oper))**2)/NPOM)
tskill[2,3] = np.sqrt(np.nansum(((DF_POM.T100_obs_to_exp-DF_POM.mean().T100_obs_to_exp)-\
                                 (DF_POM.T100_POM_exp-DF_POM.mean().T100_POM_exp))**2)/NPOM)
tskill[3,3] = np.sqrt(np.nansum(((DF_HYCOM.T100_obs_to_exp-DF_HYCOM.mean().T100_obs_to_exp)-\
                                 (DF_HYCOM.T100_HYCOM_exp-DF_HYCOM.mean().T100_HYCOM_exp))**2)/NHYCOM)

#BIAS
tskill[0,4] = DF_GOFS.mean().T100_obs - DF_GOFS.mean().T100_GOFS
tskill[1,4] = DF_POM.mean().T100_obs_to_oper - DF_POM.mean().T100_POM_oper
tskill[2,4] = DF_POM.mean().T100_obs_to_exp - DF_POM.mean().T100_POM_exp
tskill[3,4] = DF_HYCOM.mean().T100_obs_to_exp - DF_HYCOM.mean().T100_HYCOM_exp

#color
colors = ['indianred','seagreen','darkorchid','darkorange']
    
T100_skillscores = pd.DataFrame(tskill,
                        index=['GOFS','POM_oper','POM_exp','HYCOM_exp'],
                        columns=cols)
print(T100_skillscores)

#%% T100_to_ml statistics 

DF_GOFS = DF_GOFS_T100_to_ml.dropna()
DF_POM = DF_POM_T100_to_ml.dropna()
DF_HYCOM = DF_HYCOM_T100_to_ml.dropna()

NGOFS = len(DF_GOFS)-1  #For Unbiased estimmator.
NPOM = len(DF_POM)-1
NHYCOM = len(DF_HYCOM)-1

cols = ['CORRELATION','OSTD','MSTD','CRMSE','BIAS']
    
tskill = np.empty((4,5))
tskill[:] = np.nan

#CORR
tskill[0,0] = DF_GOFS.corr()['T100_to_ml_obs']['T100_to_ml_GOFS']
tskill[1,0] = DF_POM.corr()['T100_to_ml_obs_to_oper']['T100_to_ml_POM_oper']
tskill[2,0] = DF_POM.corr()['T100_to_ml_obs_to_exp']['T100_to_ml_POM_exp']
tskill[3,0] = DF_HYCOM.corr()['T100_to_ml_obs_to_exp']['T100_to_ml_HYCOM_exp']

#OSTD
tskill[0,1] = DF_GOFS.std().T100_to_ml_obs
tskill[1,1] = DF_POM.std().T100_to_ml_obs_to_oper
tskill[2,1] = DF_POM.std().T100_to_ml_obs_to_exp
tskill[3,1] = DF_HYCOM.std().T100_to_ml_obs_to_exp

#MSTD
tskill[0,2] = DF_GOFS.std().T100_to_ml_GOFS
tskill[1,2] = DF_POM.std().T100_to_ml_POM_oper
tskill[2,2] = DF_POM.std().T100_to_ml_POM_exp
tskill[3,2] = DF_HYCOM.std().T100_to_ml_HYCOM_exp

#CRMSE
tskill[0,3] = np.sqrt(np.nansum(((DF_GOFS.T100_to_ml_obs-DF_GOFS.mean().T100_to_ml_obs)-\
                                 (DF_GOFS.T100_to_ml_GOFS-DF_GOFS.mean().T100_to_ml_GOFS))**2)/NGOFS)
tskill[1,3] = np.sqrt(np.nansum(((DF_POM.T100_to_ml_obs_to_exp-DF_POM.mean().T100_to_ml_obs_to_oper)-\
                                 (DF_POM.T100_to_ml_POM_oper-DF_POM.mean().T100_to_ml_POM_oper))**2)/NPOM)
tskill[2,3] = np.sqrt(np.nansum(((DF_POM.T100_to_ml_obs_to_exp-DF_POM.mean().T100_to_ml_obs_to_exp)-\
                                 (DF_POM.T100_to_ml_POM_exp-DF_POM.mean().T100_to_ml_POM_exp))**2)/NPOM)
tskill[3,3] = np.sqrt(np.nansum(((DF_HYCOM.T100_to_ml_obs_to_exp-DF_HYCOM.mean().T100_to_ml_obs_to_exp)-\
                                 (DF_HYCOM.T100_to_ml_HYCOM_exp-DF_HYCOM.mean().T100_to_ml_HYCOM_exp))**2)/NPOM)    

#BIAS
tskill[0,4] = DF_GOFS.mean().T100_to_ml_obs - DF_GOFS.mean().T100_to_ml_GOFS
tskill[1,4] = DF_POM.mean().T100_to_ml_obs_to_oper - DF_POM.mean().T100_to_ml_POM_oper
tskill[2,4] = DF_POM.mean().T100_to_ml_obs_to_exp - DF_POM.mean().T100_to_ml_POM_exp
tskill[3,4] = DF_HYCOM.mean().T100_to_ml_obs_to_exp - DF_HYCOM.mean().T100_to_ml_HYCOM_exp

#color
colors = ['indianred','seagreen','darkorchid','darkorange']
    
T100_to_ml_skillscores = pd.DataFrame(tskill,
                        index=['GOFS','POM_oper','POM_exp','HYCOM_exp'],
                        columns=cols)
print(T100_to_ml_skillscores)
    
#%%    
    
fig, ax1 = taylor(temp_skillscores,colors,'$^oC$',np.pi/2)
plt.title('Temperature \n cycle 2019082800',fontsize=16)

file = folder_fig + 'Taylor_temperature_2019082800'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%%    
    
taylor(salt_skillscores,colors,'psu',np.pi/2)
plt.title('Salinity \n cycle 2019082800',fontsize=16)

file = folder_fig + 'Taylor_salinity_2019082800'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 


#%%    
    
taylor(Tmean_mld_skillscores,colors,'$^oC$',np.pi/2)
plt.title('Temperature MLD \n cycle 2019082800',fontsize=16)

file = folder_fig + 'Taylor_temp_mld_2019082800'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%%    
    
taylor(Smean_mld_skillscores,colors,'psu',np.pi/2)
plt.title('Salinity MLD \n cycle 2019082800',fontsize=16)

file = folder_fig + 'Taylor_salt_mld_2019082800'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%%    
    
taylor_normalized(OHC_skillscores,colors,np.pi/2)
plt.title('OHC \n cycle 2019082800',fontsize=16)

file = folder_fig + 'Taylor_ohc_2019082800'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%%    
    
taylor(T100_skillscores,colors,'$^oC$',np.pi/2)
plt.title('T100 \n cycle 2019082800',fontsize=16)

file = folder_fig + 'Taylor_T100_2019082800'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%%    
    
taylor(T100_to_ml_skillscores,colors,'$^oC$',np.pi/2)
plt.title('Temp Mean from 100 m to Base Mixed Layer  \n cycle 2019082800',fontsize=16)

file = folder_fig + 'Taylor_T100_to_ml_2019082800'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Combine all metrics into one normalized Taylor diagram 

angle_lim = np.pi/2
std_lim = 1.5
fig,ax1 = taylor_template(angle_lim,std_lim)
markers = ['s','X','^','H']
  
scores = temp_skillscores  
for i,r in enumerate(scores.iterrows()):
    theta=np.arccos(r[1].CORRELATION)
    rr=r[1].MSTD/r[1].OSTD
    ax1.plot(theta,rr,markers[i],color = 'darkorange',markersize=8)
ax1.plot(theta,rr,markers[i],label='Temp',color = 'darkorange',markersize=8)
      
scores = salt_skillscores  
for i,r in enumerate(scores.iterrows()):
    theta=np.arccos(r[1].CORRELATION)            
    rr=r[1].MSTD/r[1].OSTD
    ax1.plot(theta,rr,markers[i],color = 'seagreen',markersize=8)
ax1.plot(theta,rr,markers[i],label='Salt',color = 'seagreen',markersize=8)
        
scores = Tmean_mld_skillscores  
for i,r in enumerate(scores.iterrows()):
    theta=np.arccos(r[1].CORRELATION)            
    rr=r[1].MSTD/r[1].OSTD
    ax1.plot(theta,rr,markers[i],color = 'darkorchid',markersize=8)
ax1.plot(theta,rr,markers[i],label='Temp ML',color = 'darkorchid',markersize=8)

scores = Smean_mld_skillscores  
for i,r in enumerate(scores.iterrows()):
    theta=np.arccos(r[1].CORRELATION)            
    rr=r[1].MSTD/r[1].OSTD
    ax1.plot(theta,rr,markers[i],color = 'y',markersize=8)
ax1.plot(theta,rr,markers[i],label='Salt ML',color = 'y',markersize=8) 
       
scores = OHC_skillscores  
for i,r in enumerate(scores.iterrows()):
    theta=np.arccos(r[1].CORRELATION)            
    rr=r[1].MSTD/r[1].OSTD
    ax1.plot(theta,rr,markers[i],color = 'indianred',markersize=8) 
ax1.plot(theta,rr,markers[i],label='OHC',color = 'indianred',markersize=8) 

scores = T100_skillscores  
for i,r in enumerate(scores.iterrows()):
    theta=np.arccos(r[1].CORRELATION)            
    rr=r[1].MSTD/r[1].OSTD
    ax1.plot(theta,rr,markers[i],color = 'royalblue',markersize=8) 
ax1.plot(theta,rr,markers[i],label='T100',color = 'royalblue',markersize=8) 
'''
scores = T100_to_ml_skillscores  
for i,r in enumerate(scores.iterrows()):
    theta=np.arccos(r[1].CORRELATION)            
    rr=r[1].MSTD/r[1].OSTD
    ax1.plot(theta,rr,markers[i],color = 'darkblue',markersize=8) 
ax1.plot(theta,rr,markers[i],label='T100',color = 'darkblue',markersize=8) 
'''        
ax1.plot(0,1,'o',label='Obs',markersize=8) 
ax1.plot(0,0,'sk',label='GOFS',markersize=8)
ax1.plot(0,0,'Xk',label='POM Oper',markersize=8)
ax1.plot(0,0,'^k',label='POM Exp',markersize=8)
ax1.plot(0,0,'Hk',label='HYCOM Exp',markersize=8)
     
plt.legend(loc='upper right',bbox_to_anchor=[1.55,1.2])    

rs,ts = np.meshgrid(np.linspace(0,std_lim),np.linspace(0,angle_lim))
rms = np.sqrt(1 + rs**2 - 2*rs*np.cos(ts))
    
contours = ax1.contour(ts, rs, rms,3,colors='0.5')
plt.clabel(contours, inline=1, fontsize=10)
plt.grid(linestyle=':',alpha=0.5)

file = folder_fig + 'Taylor_norm_2019082800_v2'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Combine all metrics into one normalized Taylor diagram 

angle_lim = np.pi/2
std_lim = 1.5
fig,ax1 = taylor_template(angle_lim,std_lim)
markers = ['s','X','^','H']
  
scores = temp_skillscores  
for i,r in enumerate(scores.iterrows()):
    theta=np.arccos(r[1].CORRELATION)
    rr=r[1].MSTD/r[1].OSTD
    ax1.plot(theta,rr,markers[i],color = 'darkorange',markersize=8)
ax1.plot(theta,rr,markers[i],label='Temp',color = 'darkorange',markersize=8)
        
scores = Tmean_mld_skillscores  
for i,r in enumerate(scores.iterrows()):
    theta=np.arccos(r[1].CORRELATION)            
    rr=r[1].MSTD/r[1].OSTD
    ax1.plot(theta,rr,markers[i],color = 'darkorchid',markersize=8)
ax1.plot(theta,rr,markers[i],label='Temp ML',color = 'darkorchid',markersize=8)
      

scores = T100_skillscores  
for i,r in enumerate(scores.iterrows()):
    theta=np.arccos(r[1].CORRELATION)            
    rr=r[1].MSTD/r[1].OSTD
    ax1.plot(theta,rr,markers[i],color = 'royalblue',markersize=8) 
ax1.plot(theta,rr,markers[i],label='T100',color = 'royalblue',markersize=8) 

scores = T100_to_ml_skillscores  
for i,r in enumerate(scores.iterrows()):
    theta=np.arccos(r[1].CORRELATION)            
    rr=r[1].MSTD/r[1].OSTD
    ax1.plot(theta,rr,markers[i],color = 'darkblue',markersize=8) 
ax1.plot(theta,rr,markers[i],label='T100_ml',color = 'darkblue',markersize=8) 
        
ax1.plot(0,1,'o',label='Obs',markersize=8) 
ax1.plot(0,0,'sk',label='GOFS',markersize=8)
ax1.plot(0,0,'Xk',label='POM Oper',markersize=8)
ax1.plot(0,0,'^k',label='POM Exp',markersize=8)
ax1.plot(0,0,'Hk',label='HYCOM Exp',markersize=8)  
   
plt.legend(loc='upper right',bbox_to_anchor=[1.5,1.2])    

rs,ts = np.meshgrid(np.linspace(0,std_lim),np.linspace(0,angle_lim))
rms = np.sqrt(1 + rs**2 - 2*rs*np.cos(ts))
    
contours = ax1.contour(ts, rs, rms,3,colors='0.5')
plt.clabel(contours, inline=1, fontsize=10)
plt.grid(linestyle=':',alpha=0.5)

file = folder_fig + 'Taylor_norm_2019082800_v3'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Sketch normalized example
angle_lim = np.pi/2
std_lim = 2.0
fig,ax1 = taylor_template(angle_lim,std_lim)
      
scores = OHC_skillscores 

max_std = 2.0
rs,ts = np.meshgrid(np.linspace(0,std_lim),np.linspace(0,angle_lim))
rms = np.sqrt(1 + rs**2 - 2*rs*np.cos(ts))


#rs,ts = np.meshgrid(np.linspace(0,np.round(max_std+0.1,2)),np.linspace(0,angle_lim))   
#rms = np.sqrt(scores.OSTD[0]**2 + rs**2 - 2*rs*scores.OSTD[0]*np.cos(ts))
 
for i,r in enumerate(scores.iterrows()):
    if i==0:
        theta=np.arccos(r[1].CORRELATION)            
        rr=r[1].MSTD/r[1].OSTD
        ax1.plot(theta,rr,'o',color = 'indianred',markersize=8,label='Model')
        
        crmse = np.sqrt(1 + (r[1].MSTD/scores.OSTD[i])**2 \
                   - 2*(r[1].MSTD/scores.OSTD[i])*r[1].CORRELATION) 
        c1 = ax1.contour(ts, rs, rms,[crmse],colors=colors[i])
        plt.clabel(c1, inline=1, fontsize=10,fmt='%1.2f')
        
ax1.plot(0,1,'o',label='Obs',markersize=8)
plt.legend(loc='upper right',bbox_to_anchor=[1.1,1.1]) 
        
file = folder_fig + 'Taylor_norm_example'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 