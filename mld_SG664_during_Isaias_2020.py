#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 14:49:48 2020

@author: aristizabal
"""

#%% User input

# date limits
date_ini = '2020/07/25/00'
date_end = '2020/08/05/00'

# MAB
lon_lim = [-110.0,-10.0]
lat_lim = [15.0,45.0]

glid_ids = ['SG664','ru33','sam','ng645']

# Server location
url_erddap = 'https://data.ioos.us/gliders/erddap'

#%%

from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import sys
import seawater as sw

sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/All_code/Remote_repos/glider_model_comparisons_Python')
#sys.path.append('/home/aristizabal/glider_model_comparisons_Python')
from read_glider_data import retrieve_dataset_id_erddap_server
from read_glider_data import read_glider_data_erddap_server


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
            if np.nanmin(np.abs(depth[ok26]))>10:
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

#%% Calculate time series of potential Energy Anomaly over the top 100 m

def Potential_Energy_Anomaly(time,depth,density):
    g = 9.8 #m/s
    PEA = np.empty((len(time)))
    PEA[:] = np.nan
    for t,tstamp in enumerate(time):   
        print(t)
        if np.ndim(depth) == 2:
            dindex = np.fliplr(np.where(np.asarray(np.abs(depth[:,t])) <= 100))[0]
        else:
            dindex = np.fliplr(np.where(np.asarray(np.abs(depth)) <= 100))[0]
        if len(dindex) == 0:
            PEA[t] = np.nan
        else:
            if np.ndim(depth) == 2: 
                zz = np.asarray(np.abs(depth[dindex,t]))
            else:
                zz = np.asarray(np.abs(depth[dindex]))
            denss = np.asarray(density[dindex,t])
            ok = np.isfinite(denss)
            z = zz[ok]
            dens = denss[ok]
            if len(z)==0 or len(dens)==0 or np.min(zz) > 10:
                PEA[t] = np.nan
            else:
                if z[-1] - z[0] > 0:
                    # So PEA is < 0
                    #sign = -1
                    # Adding 0 to sigma integral is normalized
                    z = np.append(0,z)
                else:
                    # So PEA is < 0
                    #sign = 1
                    # Adding 0 to sigma integral is normalized
                    z = np.flipud(z)
                    z = np.append(0,z)
                    dens = np.flipud(dens)
    
                # adding density at depth = 0
                densit = np.interp(z,z[1:],dens)
                densit = np.flipud(densit)
                
                # defining sigma
                max_depth = np.nanmax(zz[ok])  
                sigma = -1*z/max_depth
                sigma = np.flipud(sigma)
                
                rhomean = np.trapz(densit,sigma,axis=0)
                drho = rhomean-densit
                torque = drho * sigma
                PEA[t] = g* max_depth * np.trapz(torque,sigma,axis=0) 
                #print(max_depth, ' ',PEA[t]) 
                
    return PEA

#%%Time window
year_ini = int(date_ini.split('/')[0])
month_ini = int(date_ini.split('/')[1])
day_ini = int(date_ini.split('/')[2])

year_end = int(date_end.split('/')[0])
month_end = int(date_end.split('/')[1])
day_end = int(date_end.split('/')[2])

tini = datetime(year_ini, month_ini, day_ini)
tend = datetime(year_end, month_end, day_end)  

#%% 

gliders = retrieve_dataset_id_erddap_server(url_erddap,lat_lim,lon_lim,date_ini,date_end)
print(gliders)

for i,ids in enumerate(glid_ids[0:1]): 
    print(ids)
    okglid = [j for j,id in enumerate(gliders) if id.split('-')[0] == glid_ids[i]][0]
    
    dataset_id = gliders[okglid]
    
    kwargs = dict(date_ini=date_ini,date_end=date_end)
    scatter_plot = 'no'
    
    tempg, saltg, timeg, latg, long, depthg = read_glider_data_erddap_server(url_erddap,dataset_id,\
                                       lat_lim,lon_lim,scatter_plot,**kwargs)
        
    # Calculate density
    densg = sw.dens(saltg,tempg,tempg)     
        
    # Grid glider variables according to depth
    delta_z = 0.5
    depthg_gridded,tempg_gridded,saltg_gridded,densg_gridded = \
    varsg_gridded(depthg,timeg,tempg,saltg,densg,delta_z)  
    
    # MLD
    dt = 0.2
    drho = 0.125
    
    # for glider data
    MLD_temp_crit_glid,Tmean_temp_crit_glid,Smean_temp_crit_glid,_,\
    MLD_dens_crit_glid,Tmean_dens_crit_glid,Smean_dens_crit_glid,_ = \
    MLD_temp_and_dens_criteria(dt,drho,timeg,depthg_gridded,tempg_gridded,saltg_gridded,densg_gridded)

    # Surface Ocean Heat Content
    OHC_glid = OHC_surface(timeg,tempg_gridded,depthg_gridded,densg_gridded)
     
    # PEA
    PEA_glid = Potential_Energy_Anomaly(timeg,depthg,densg)      
    #%% OHC figure
    
    fig,ax = plt.subplots(figsize=(12, 2.8))
    plt.plot(timeg,OHC_glid*10**-7,'-o',color='royalblue',label=dataset_id.split('-')[0],linewidth=4)
    
    ax.set_xlim(timeg[0],timeg[-1])
    plt.ylabel('($KJ/cm^2$)',fontsize = 14)
    plt.title('Ocean Heat Content',fontsize=16)
    
    xvec = [tini + timedelta(int(dt)) for dt in np.arange((tend-tini).days+1)[::2]]
    plt.xticks(xvec,fontsize=12)
    xfmt = mdates.DateFormatter('%b-%d')
    ax.xaxis.set_major_formatter(xfmt)
    plt.xlim(tini,timeg[-1])
    
    #%% Temp ML
    
    fig,ax = plt.subplots(figsize=(12, 2.8))
    plt.plot(timeg,Tmean_dens_crit_glid,'-o',color='royalblue',label='Density criteria',linewidth=4)
    plt.plot(timeg,Tmean_temp_crit_glid,'-o',color='skyblue',label='Temp. Criteria',linewidth=4)
    ax.set_xlim(timeg[0],timeg[-1])
    ax.set_ylabel('($^oC$)',fontsize=14)
    plt.title('Temperature Mixed Layer ' + dataset_id.split('-')[0],fontsize=16)
    plt.legend()
    
    xvec = [tini + timedelta(int(dt)) for dt in np.arange((tend-tini).days+1)[::2]]
    plt.xticks(xvec,fontsize=12)
    xfmt = mdates.DateFormatter('%b-%d')
    ax.xaxis.set_major_formatter(xfmt)
    plt.xlim(tini,timeg[-1])
    
    #%% Sal ML
    
    fig,ax = plt.subplots(figsize=(12, 2.8))
    plt.plot(timeg,Smean_dens_crit_glid,'-o',color='royalblue',label='Density criteria',linewidth=4)
    plt.plot(timeg,Smean_temp_crit_glid,'-o',color='skyblue',label='Temp. Criteria',linewidth=4)
    ax.set_xlim(timeg[0],timeg[-1])
    ax.set_ylabel(' ',fontsize=14)
    plt.title('Salinity Mixed Layer ' + dataset_id.split('-')[0],fontsize=16)
    plt.legend()
    
    xvec = [tini + timedelta(int(dt)) for dt in np.arange((tend-tini).days+1)[::2]]
    plt.xticks(xvec,fontsize=12)
    xfmt = mdates.DateFormatter('%b-%d')
    ax.xaxis.set_major_formatter(xfmt)
    plt.xlim(tini,timeg[-1])
    
    #%% MLD
    
    fig,ax = plt.subplots(figsize=(12, 2.8))
    plt.plot(timeg,MLD_dens_crit_glid,'-o',color='royalblue',label='Density criteria',linewidth=4)
    plt.plot(timeg,MLD_temp_crit_glid,'-o',color='skyblue',label='Temp. Criteria',linewidth=4)
    ax.set_xlim(timeg[0],timeg[-1])
    ax.set_ylabel('(m)',fontsize=14)
    plt.title('Mixed Layer Depth ' + dataset_id.split('-')[0],fontsize=16)
    plt.legend()
    
    xvec = [tini + timedelta(int(dt)) for dt in np.arange((tend-tini).days+1)[::2]]
    plt.xticks(xvec,fontsize=12)
    xfmt = mdates.DateFormatter('%b-%d')
    ax.xaxis.set_major_formatter(xfmt)
    plt.xlim(tini,timeg[-1])
    
    #%% PEA
    
    fig,ax = plt.subplots(figsize=(12, 2.8))
    plt.plot(timeg,PEA_glid,'-o',color='royalblue',linewidth=4)
    ax.set_xlim(timeg[0],timeg[-1])
    ax.set_ylabel('($J/m^3$)',fontsize=14)
    plt.title('Potential Energy Anomaly ' + dataset_id.split('-')[0],fontsize=16)
    
    xvec = [tini + timedelta(int(dt)) for dt in np.arange((tend-tini).days+1)[::2]]
    plt.xticks(xvec,fontsize=12)
    xfmt = mdates.DateFormatter('%b-%d')
    ax.xaxis.set_major_formatter(xfmt)
    plt.xlim(tini,timeg[-1])