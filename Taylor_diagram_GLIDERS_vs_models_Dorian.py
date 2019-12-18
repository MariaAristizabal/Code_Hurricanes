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
def interp_datasets_in_z(temp_orig,salt_orig,depth_orig,depth_target,indexes_time,din_vars):

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

def taylor_template(angle_lim):

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
lon_limG, lat_limG = glider_coor_to_GOFS_coord(lon_lim,lat_lim)

oklat_GOFS = np.where(np.logical_and(lat_G >= lat_limG[0], lat_G <= lat_limG[1])) 
oklon_GOFS = np.where(np.logical_and(lon_G >= lon_limG[0], lon_G <= lon_limG[1])) 

lat_GOFS = lat_G[oklat_GOFS]
lon_GOFS = lon_G[oklon_GOFS]

depth_GOFS = np.asarray(GOFS.depth[:])

lon_GOFSg, lat_GOFSg = GOFS_coor_to_glider_coord(lon_GOFS,lat_GOFS)

#%% Reading glider data

DF_GOFS_temp_salt = pd.DataFrame()
DF_POM_temp_salt = pd.DataFrame()
DF_GOFS_MLD = pd.DataFrame()
DF_POM_MLD = pd.DataFrame()
DF_GOFS_OHC = pd.DataFrame()
DF_POM_OHC = pd.DataFrame()

for f,file in enumerate(gdata):
    print(file)    
    url_glider = file
    
    var = 'temperature'
    scatter_plot = 'no'
    kwargs = dict(date_ini=date_ini[0:-3],date_end=date_end[0:-3])    
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
    MLD_temp_crit_glid, _, _, _, MLD_dens_crit_glid, Tmean_dens_crit_glid, Smean_dens_crit_glid, _ = \
    MLD_temp_and_dens_criteria(dt,drho,timeg,depthg_gridded,tempg_gridded,saltg_gridded,densg_gridded)
    
    # for GOFS 3.1 output 
    MLD_temp_crit_GOFS, _, _, _, MLD_dens_crit_GOFS, Tmean_dens_crit_GOFS, Smean_dens_crit_GOFS, _ = \
    MLD_temp_and_dens_criteria(dt,drho,time_GOFS,depth_GOFS,target_temp_GOFS,target_salt_GOFS,target_dens_GOFS)          
    
    # for POM operational
    MLD_temp_crit_POM_oper, _, _, _, MLD_dens_crit_POM_oper, Tmean_dens_crit_POM_oper, Smean_dens_crit_POM_oper, _ = \
    MLD_temp_and_dens_criteria(dt,drho,timestamp_pom_oper,target_depth_POM_oper,target_temp_POM_oper,target_salt_POM_oper,target_dens_POM_oper)
    
    # for POM experimental
    MLD_temp_crit_POM_exp, _, _, _, MLD_dens_crit_POM_exp, Tmean_dens_crit_POM_exp, Smean_dens_crit_POM_exp, _ = \
    MLD_temp_and_dens_criteria(dt,drho,timestamp_pom_exp,target_depth_POM_exp,target_temp_POM_exp,target_salt_POM_exp,target_dens_POM_exp)
    
    #%% Surface Ocean Heat Content
    
    # glider
    OHC_glid = OHC_surface(timeg,tempg_gridded,depthg_gridded,densg_gridded)
    
    # GOFS
    OHC_GOFS = OHC_surface(time_GOFS,target_temp_GOFS,depth_GOFS,target_dens_GOFS)
    
    # POM operational    
    OHC_POM_oper = OHC_surface(timestamp_pom_oper,target_temp_POM_oper,target_depth_POM_oper,target_dens_POM_oper)
    
    # POM experimental
    OHC_POM_exp = OHC_surface(timestamp_pom_exp,target_temp_POM_exp,target_depth_POM_exp,target_dens_POM_exp)
    
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
    
    #%% Interpolate glider transect onto POM operational time and depth
        
    oktimeg_pom_oper = np.round(np.interp(timestamp_pom_oper,tstamp_glider,np.arange(len(tstamp_glider)))).astype(int)   
    
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
    
    #%% Interpolate glider transect onto POM experimental time and depth
        
    oktimeg_pom_exp = np.round(np.interp(timestamp_pom_exp,tstamp_glider,np.arange(len(tstamp_glider)))).astype(int)  
    
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
    
    df_GOFS_MLD = pd.DataFrame(data=np.array([MLD_dens_crit_glid_to_GOFS,MLD_dens_crit_GOFS,\
                                              Tmean_dens_crit_glid_to_GOFS,Tmean_dens_crit_GOFS]).T,\
                      columns=['MLD_obs','MLD_GOFS',\
                              'Tmean_obs','Tmean_GOFS'])  
        
    #%% Define dataframe
    
    df_POM_MLD = pd.DataFrame(data=np.array([MLD_dens_crit_glid_to_POM_oper,MLD_dens_crit_POM_oper,\
                                             Tmean_dens_crit_glid_to_POM_oper,Tmean_dens_crit_POM_oper,\
                                             MLD_dens_crit_glid_to_POM_exp,MLD_dens_crit_POM_exp,\
                                             Tmean_dens_crit_glid_to_POM_exp,Tmean_dens_crit_POM_exp]).T,\
                      columns=['MLD_obs_to_oper','MLD_POM_oper',\
                              'Tmean_obs_to_oper','Tmean_POM_oper',\
                              'MLD_obs_to_exp','MLD_POM_exp',\
                              'Tmean_obs_to_exp','Tmean_POM_exp']) 
    
    #%% Define dataframe
    
    df_GOFS_OHC = pd.DataFrame(data=np.array([OHC_glid_to_GOFS,OHC_GOFS]).T,\
                      columns=['OHC_obs','OHC_GOFS'])  
    
    #%% DEfine dataframe
    
    df_POM_OHC = pd.DataFrame(data=np.array([OHC_glid_to_POM_oper,OHC_glid_to_POM_exp,\
                                              OHC_POM_oper,OHC_POM_exp]).T,\
                      columns=['OHC_obs_to_oper','OHC_obs_to_exp',\
                               'OHC_POM_oper','OHC_POM_exp'])

    #%% Concatenate data frames       
    
    DF_GOFS_temp_salt = pd.concat([DF_GOFS_temp_salt, df_GOFS_temp_salt])
    DF_POM_temp_salt = pd.concat([DF_POM_temp_salt, df_POM_temp_salt])
    DF_GOFS_MLD = pd.concat([DF_GOFS_MLD, df_GOFS_MLD])
    DF_POM_MLD = pd.concat([DF_POM_MLD, df_POM_MLD])
    DF_GOFS_OHC = pd.concat([DF_GOFS_OHC, df_GOFS_OHC])
    DF_POM_OHC = pd.concat([DF_POM_OHC, df_POM_OHC])
    
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

DF_GOFS = DF_GOFS_MLD
DF_POM = DF_POM_MLD

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
    
fig, ax1 = taylor(temp_skillscores,colors,'$^oC$',np.pi/2)
plt.title('Temperature \n cycle 2019082800',fontsize=16)

file = folder + 'Taylor_temperature_2019082800'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%%    
    
taylor(salt_skillscores,colors,'psu',np.pi/2)
plt.title('Salinity \n cycle 2019082800',fontsize=16)

file = folder + 'Taylor_salinity_2019082800'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 


#%%    
    
taylor(Tmean_mld_skillscores,colors,'$^oC$',np.pi/2+np.pi/8)
plt.title('Temperature MLD \n cycle 2019082800',fontsize=16)

file = folder + 'Taylor_temp_mld_2019082800'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%%    
    
taylor_normalized(OHC_skillscores,colors,np.pi/2)
plt.title('OHC \n cycle 2019082800',fontsize=16)

file = folder + 'Taylor_ohc_2019082800'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Combine all metrics into one normalized Taylor diagram 

angle_lim = np.pi/2+np.pi/8
fig,ax1 = taylor_template(angle_lim)
  
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

rs,ts = np.meshgrid(np.linspace(0,2.0),np.linspace(0,angle_lim))
rms = np.sqrt(1 + rs**2 - 2*rs*np.cos(ts))
    
contours = ax1.contour(ts, rs, rms,3,colors='0.5')
plt.clabel(contours, inline=1, fontsize=10)
plt.grid(linestyle=':',alpha=0.5)

file = folder + 'Taylor_norm_2019082800'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Combine all metrics into one normalized Taylor diagram 

angle_lim
fig,ax1 = taylor_template(angle_lim)
markers = ['s','X','^']
  
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
        
scores = OHC_skillscores  
for i,r in enumerate(scores.iterrows()):
    theta=np.arccos(r[1].CORRELATION)            
    rr=r[1].MSTD/r[1].OSTD
    ax1.plot(theta,rr,markers[i],color = 'indianred',markersize=8) 
ax1.plot(theta,rr,markers[i],label='OHC',color = 'indianred',markersize=8) 
        
ax1.plot(0,1,'o',label='Obs',markersize=8) 
ax1.plot(0,0,'sk',label='GOFS',markersize=8)
ax1.plot(0,0,'Xk',label='POM Oper',markersize=8)
ax1.plot(0,0,'^k',label='POM Exp',markersize=8)
     
plt.legend(loc='upper right',bbox_to_anchor=[1.45,1.2])    

rs,ts = np.meshgrid(np.linspace(0,2.0),np.linspace(0,angle_lim))
rms = np.sqrt(1 + rs**2 - 2*rs*np.cos(ts))
    
contours = ax1.contour(ts, rs, rms,3,colors='0.5')
plt.clabel(contours, inline=1, fontsize=10)
plt.grid(linestyle=':',alpha=0.5)

file = folder + 'Taylor_norm_2019082800_v2'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 