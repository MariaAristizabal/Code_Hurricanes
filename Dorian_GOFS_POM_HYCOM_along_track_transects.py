#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 11:06:24 2020

@author: root
"""

#%% User input

lon_lim = [-98.5,-60.0]
lat_lim = [10.0,45.0]

cycle = '2019090118'
#cycle = '2019083018'
#cycle = '2019082800'
#cycle = '2019082918'

delta_lon = 0 # delta longitude around hurricane track to calculate
               # statistics
Nini = 0 # 0 is the start of forecating cycle (2019082800)
      # 1 is 6 hours of forecasting cycle   (2019082806)
      # 2 is 12 hours ...... 20 is 120 hours 

Nend = 22 # indicates how far in the hurricabe track you want
          # include in the analysis. This is helpful if for ex:
          # you onl want to analyse the portion of the track
          # where the storm intensifies
          # 22 corresponds to all the hurricane track forecasted in a cycle
#Nend = 13

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_50.0.nc' 

# KMZ file best track Dorian
kmz_file_Dorian = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/KMZ_files/al052019_best_track-5.kmz'  

# url for GOFS 3.1
url_GOFS = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'

# figures
folder_fig = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp2/'

# folder nc files POM
folder_pom19 =  '/Volumes/aristizabal/HWRF2019_POM_Dorian/'
folder_pom20 =  '/Volumes/aristizabal/HWRF2020_POM_Dorian/'

# folde HWRF2020_HYCOM
folder_hycom20 = '/Volumes/aristizabal/HWRF2020_HYCOM_Dorian/'

###################

# folder nc files POM
folder_pom_oper = folder_pom19 + 'HWRF2019_POM_dorian05l.' + cycle + '_pom_files_oper/'
folder_pom_exp = folder_pom20 + 'HWRF2020_POM_dorian05l.'  + cycle + '_pom_files_exp/'
prefix_pom = 'dorian05l.' + cycle + '.pom.00'

pom_grid_oper = folder_pom_oper + 'dorian05l.' + cycle + '.pom.grid.nc'
pom_grid_exp = folder_pom_exp + 'dorian05l.' + cycle + '.pom.grid.nc'

# Dorian track files
hwrf_pom_track_oper = folder_pom_oper + 'dorian05l.' + cycle + '.trak.hwrf.atcfunix'
hwrf_pom_track_exp = folder_pom_exp + 'dorian05l.' + cycle + '.trak.hwrf.atcfunix'

# folder nc files hwrf
folder_hwrf_pom19_oper = folder_pom19 + 'HWRF2019_POM_dorian05l.' + cycle + '_grb2_to_nc_oper/' 
folder_hwrf_pom20_exp = folder_pom20 + 'HWRF2020_POM_dorian05l.' + cycle + '_grb2_to_nc_exp/' 

##################
# folder ab files HYCOM
folder_hycom_exp = folder_hycom20 + 'HWRF2020_HYCOM_dorian05l.' + cycle + '_hycom_files_exp/'
prefix_hycom = 'dorian05l.' + cycle + '.hwrf_rtofs_hat10_3z'

#Dir_HMON_HYCOM = '/Volumes/aristizabal/ncep_model/HMON-HYCOM_Michael/'
Dir_HMON_HYCOM = '/Volumes/aristizabal/ncep_model/HWRF-Hycom-WW3_exp_Michael/'
# RTOFS grid file name
hycom_grid_exp = Dir_HMON_HYCOM + 'hwrf_rtofs_hat10.basin.regional.grid'

# Dorian track files
hwrf_hycom_track_exp = folder_hycom_exp + 'dorian05l.' + cycle + '.trak.hwrf.atcfunix'

# folder nc files hwrf
folder_hwrf_hycom20_exp = folder_hycom20 + 'HWRF2020_HYCOM_dorian05l.' + cycle + '_grb2_to_nc_exp/' 

#%%
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
import os
import os.path
from bs4 import BeautifulSoup
from zipfile import ZipFile
import glob
import seawater as sw
import cmocean

import sys
sys.path.append('/Users/aristizabal/Desktop/MARACOOS_project/NCEP_scripts')
from utils4HYCOM import readBinz, readgrids

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%%
def MLD_temp_and_dens_criteria(dt,drho,time,depth,temp,salt,dens):

    MLD_temp_crit = np.empty(temp.shape[1]) 
    MLD_temp_crit[:] = np.nan
    Tmean_temp_crit = np.empty(temp.shape[1]) 
    Tmean_temp_crit[:] = np.nan
    Smean_temp_crit = np.empty(temp.shape[1]) 
    Smean_temp_crit[:] = np.nan
    Td_temp_crit = np.empty(temp.shape[1]) 
    Td_temp_crit[:] = np.nan
    MLD_dens_crit = np.empty(temp.shape[1])
    MLD_dens_crit[:] = np.nan
    Tmean_dens_crit = np.empty(temp.shape[1])
    Tmean_dens_crit[:] = np.nan
    Smean_dens_crit = np.empty(temp.shape[1]) 
    Smean_dens_crit[:] = np.nan
    Td_dens_crit = np.empty(temp.shape[1]) 
    Td_dens_crit[:] = np.nan
    for t in np.arange(temp.shape[1]):
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

def OHC_surface(temp,depth,dens):
    cp = 3985 #Heat capacity in J/(kg K)

    OHC = np.empty(temp.shape[1])
    OHC[:] = np.nan
    for t in np.arange(temp.shape[1]):
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

#%% Get storm track from HWRF/POM output

def get_storm_track_POM(file_track):

    ff = open(file_track,'r')
    f = ff.readlines()
    
    latt = []
    lont = []
    lead_time = []
    for l in f:
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
        lont.append(lon)
        lead_time.append(int(l.split(',')[5][1:4]))
    
    latt = np.asarray(latt)
    lont = np.asarray(lont)
    lead_time, ind = np.unique(lead_time,return_index=True)
    lat_track = latt[ind]
    lon_track = lont[ind]  

    return lon_track, lat_track, lead_time

#%%
def get_max_winds_10m_HWRF(folder_nc_files): 
    
    files = sorted(glob.glob(os.path.join(folder_nc_files,'*.nc')))
    max_wind_10m_hwrf = []
    time_hwrf = []
    for i,fl in enumerate(files):
        print(i)
        HWRF = xr.open_dataset(fl)
        t_hwrf = np.asarray(HWRF.variables['time'][:])
        UGRD_hwrf = np.asarray(HWRF.variables['UGRD_10maboveground'][0,:,:])
        VGRD_hwrf = np.asarray(HWRF.variables['VGRD_10maboveground'][0,:,:])
        max_wind_10m_hwrf.append(np.max(np.sqrt(UGRD_hwrf**2 + VGRD_hwrf**2)))
        time_hwrf.append(t_hwrf)
        
    return max_wind_10m_hwrf, time_hwrf

#%% Read best storm track from kmz file
    
def read_kmz_file_storm_best_track(kmz_file):
    
    os.system('cp ' + kmz_file + ' ' + kmz_file[:-3] + 'zip')
    os.system('unzip -o ' + kmz_file + ' -d ' + kmz_file[:-4])
    kmz = ZipFile(kmz_file[:-3]+'zip', 'r')
    kml_file = kmz_file.split('/')[-1].split('_')[0] + '.kml'
    kml_best_track = kmz.open(kml_file, 'r').read()
    
    # best track coordinates
    soup = BeautifulSoup(kml_best_track,'html.parser')
    
    lon_best_track = np.empty(len(soup.find_all("point")))
    lon_best_track[:] = np.nan
    lat_best_track = np.empty(len(soup.find_all("point")))
    lat_best_track[:] = np.nan
    for i,s in enumerate(soup.find_all("point")):
        lon_best_track[i] = float(s.get_text("coordinates").split('coordinates')[1].split(',')[0])
        lat_best_track[i] = float(s.get_text("coordinates").split('coordinates')[1].split(',')[1])
             
    #  get time stamp
    time_best_track = []
    for i,s in enumerate(soup.find_all("atcfdtg")):
        tt = datetime.strptime(s.get_text(' '),'%Y%m%d%H')
        time_best_track.append(tt)
    time_best_track = np.asarray(time_best_track)    
    
    # get type 
    wind_int_mph = []
    for i,s in enumerate(soup.find_all("intensitymph")):
        wind_int_mph.append(s.get_text(' ')) 
    wind_int_mph = np.asarray(wind_int_mph)
    wind_int_mph = wind_int_mph.astype(float)  
    
    wind_int_kt = []
    for i,s in enumerate(soup.find_all("intensity")):
        wind_int_kt.append(s.get_text(' ')) 
    wind_int_kt = np.asarray(wind_int_kt)
    wind_int_kt = wind_int_kt.astype(float)
      
    cat = []
    for i,s in enumerate(soup.find_all("styleurl")):
        cat.append(s.get_text('#').split('#')[-1]) 
    cat = np.asarray(cat)
    
    return lon_best_track, lat_best_track, time_best_track, wind_int_mph, wind_int_kt, cat

#%% Reading POM temperature and salinity for the N time step in forecasting cycle
# following an along track

def get_profiles_from_POM(N,folder_pom,prefix,lon_track,lat_track,\
                                          lon_pom,lat_pom,zlev_pom,zmatrix_pom):   
    
    pom_ncfiles = sorted(glob.glob(os.path.join(folder_pom,prefix+'*.nc')))   
    file = pom_ncfiles[N]
    pom = xr.open_dataset(file)
    tpom = pom['time'][:]
    timestamp_pom = mdates.date2num(tpom)[0]
    time_POM = mdates.num2date(timestamp_pom)
    
    oklon = np.round(np.interp(lon_track,lon_pom[0,:],np.arange(len(lon_pom[0,:])))).astype(int)
    oklat = np.round(np.interp(lat_track,lat_pom[:,0],np.arange(len(lat_pom[:,0])))).astype(int)
    
    temp_POM_along_track = np.empty((len(zlev_pom),len(oklat)))
    temp_POM_along_track[:] = np.nan
    salt_POM_along_track = np.empty((len(zlev_pom),len(oklat)))
    salt_POM_along_track[:] = np.nan
    zmatrix_POM_along_track = np.empty((len(zlev_pom),len(oklat)))
    zmatrix_POM_along_track[:] = np.nan
    dens_POM_along_track = np.empty((len(zlev_pom),len(oklat)))
    dens_POM_along_track[:] = np.nan
    for x,lonn in enumerate(oklon):
        print(x)
        temp_POM_along_track[:,x] = np.asarray(pom['t'][0,:,oklat[x],oklon[x]])
        salt_POM_along_track[:,x] = np.asarray(pom['s'][0,:,oklat[x],oklon[x]])
        zmatrix_POM_along_track[:,x] = zmatrix_pom[oklat[x],oklon[x],:]
        dens_POM_along_track[:,x] = np.asarray(pom['rho'][0,:,oklat[x],oklon[x]])
        
    temp_POM_along_track[temp_POM_along_track==0] = np.nan
    salt_POM_along_track[salt_POM_along_track==0] = np.nan
    dens_POM_along_track = dens_POM_along_track * 1000 + 1000 
    dens_POM_along_track[dens_POM_along_track == 1000.0] = np.nan   
    
    return temp_POM_along_track,salt_POM_along_track,dens_POM_along_track,zmatrix_POM_along_track, time_POM

#%%    
def get_profiles_from_HYCOM(N,folder_hycom,prefix,lon_track,lat_track,\
                   lon_hycom,lat_hycom,var):

    afiles = sorted(glob.glob(os.path.join(folder_hycom,prefix+'*.a')))    
    file = afiles[N]
    
    #Reading time stamp
    year = int(file.split('/')[-1].split('.')[1][0:4])
    month = int(file.split('/')[-1].split('.')[1][4:6])
    day = int(file.split('/')[-1].split('.')[1][6:8])
    hour = int(file.split('/')[-1].split('.')[1][8:10])
    dt = int(file.split('/')[-1].split('.')[3][1:])
    timestamp_hycom = mdates.date2num(datetime(year,month,day,hour)) + dt/24
    time_hycom = mdates.num2date(timestamp_hycom)
    
    # Interpolating lat_track and lon_track into HYCOM grid
    oklon = np.round(np.interp(lon_track+360,lon_hycom[0,:],np.arange(len(lon_hycom[0,:])))).astype(int)
    oklat = np.round(np.interp(lat_track,lat_hycom[:,0],np.arange(len(lat_hycom[:,0])))).astype(int)
    
    # Reading 3D variable from binary file 
    var_hyc = readBinz(file[:-2],'3z',var)
    var_hycom = var_hyc[oklat,oklon,:].T
    
    time_hycom = np.asarray(time_hycom)
    
    return var_hycom, time_hycom

#%% Reading temperature and salinity from DF on target_time following an along track    

def get_profiles_from_GOFS(DF,target_time,lon_track,lat_track):

    depth = np.asarray(DF.depth[:])
    tt_G = DF.time
    t_G = netCDF4.num2date(tt_G[:],tt_G.units)
    oklon = np.round(np.interp(lon_track+360,DF.lon,np.arange(len(DF.lon)))).astype(int)
    oklat = np.round(np.interp(lat_track,DF.lat,np.arange(len(DF.lat)))).astype(int)
    okt = np.where(mdates.date2num(t_G) == mdates.date2num(target_time))[0][0]
    
    temp_along_track = np.empty((len(depth),len(oklon)))
    temp_along_track[:] = np.nan
    salt_along_track = np.empty((len(depth),len(oklon)))
    salt_along_track[:] = np.nan
    for x,lonn in enumerate(oklon):
        print(x)
        temp_along_track[:,x] = np.asarray(DF.water_temp[okt,:,oklat[x],oklon[x]])
        salt_along_track[:,x] = np.asarray(DF.salinity[okt,:,oklat[x],oklon[x]])
    
    return temp_along_track, salt_along_track

#%%   
    
def interp_datasets_in_z(temp_orig,salt_orig,depth_orig,depth_target,nz,nx):

    temp_interp = np.empty((nz,nx))
    temp_interp[:] = np.nan
    salt_interp = np.empty((nz,nx))
    salt_interp[:] = np.nan
    for x in np.arange(nx):
        if depth_target[-1,x] == 1.0:
                temp_interp[:,x] = np.nan
        else:
            temp_interp[:,x] = np.interp(depth_target[:,x],depth_orig,temp_orig[:,x])
            salt_interp[:,x] = np.interp(depth_target[:,x],depth_orig,salt_orig[:,x])
    
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
    
    #ax1.plot(0,1,'o',label='Obs')    
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

#%% Read GOFS 3.1 grid

print('Retrieving coordinates from GOFS')
GOFS = xr.open_dataset(url_GOFS,decode_times=False) 

tt_G = GOFS.time
t_G = netCDF4.num2date(tt_G[:],tt_G.units)
lat_G = np.asarray(GOFS.lat[:])
lon_G = np.asarray(GOFS.lon[:]) 
depth_GOFS = np.asarray(GOFS.depth[:])

#%% Get Dorian track from models

lon_forec_track_pom_oper, lat_forec_track_pom_oper, lead_time_pom_oper = get_storm_track_POM(hwrf_pom_track_oper)

lon_forec_track_pom_exp, lat_forec_track_pom_exp, lead_time_pom_exp = get_storm_track_POM(hwrf_pom_track_exp)

lon_forec_track_hycom_exp, lat_forec_track_hycom_exp, lead_time_hycom_exp = get_storm_track_POM(hwrf_hycom_track_exp)

#%% Get Dorian best track 

lon_best_track, lat_best_track, time_best_track, _, wind_int_kt, _ = \
read_kmz_file_storm_best_track(kmz_file_Dorian)

#%% Obtain lat and lon band around forecated track operational

dlon = 0.1
nlevels = int(2*delta_lon /dlon) + 1

lon_bnd = np.linspace(lon_forec_track_pom_oper[2*Nini:2*Nend-1]-delta_lon,lon_forec_track_pom_oper[2*Nini:2*Nend-1]+delta_lon,nlevels) 
lon_band = lon_bnd.ravel()
lat_bd = np.tile(lat_forec_track_pom_oper[2*Nini:2*Nend-1],lon_bnd.shape[0])
lat_bnd = lat_bd.reshape(lon_bnd.shape[0],lon_bnd.shape[1])
lat_band = lat_bnd.ravel()

#%% Get winds in knots at 10 meterd height
    
folder_nc_files = folder_hwrf_pom19_oper
max_wind_10m_hwrf_pom19_oper, time_hwrf = get_max_winds_10m_HWRF(folder_nc_files)

folder_nc_files = folder_hwrf_pom20_exp
max_wind_10m_hwrf_pom20_exp, _ = get_max_winds_10m_HWRF(folder_nc_files)

folder_nc_files = folder_hwrf_hycom20_exp
max_wind_10m_hwrf_hycom20_exp, _ = get_max_winds_10m_HWRF(folder_nc_files)

#%% wind speed in knots

max_wind_10m_hwrf_pom19_oper = 1.94384 * np.asarray(max_wind_10m_hwrf_pom19_oper)
max_wind_10m_hwrf_pom20_exp = 1.94384 * np.asarray(max_wind_10m_hwrf_pom20_exp)
max_wind_10m_hwrf_hycom20_exp = 1.94384 * np.asarray(max_wind_10m_hwrf_hycom20_exp)

#%% Reading POM operational temperature and salinity for firts time step in forecast cycle 2018082800
# following a band around the forecasted storm track by HWRF/POM

#N = 1
temp_POM_band_oper , salt_POM_band_oper, dens_POM_band_oper,\
zmatrix_POM_band_oper, time_POM = \
get_profiles_from_POM(Nini,folder_pom_oper,prefix_pom,lon_band,lat_band,\
                                          lon_pom_oper,lat_pom_oper,zlev_pom_oper,zmatrix_pom_oper)    
    
#%% Reading POM experimental temperature and salinity for firts time step in forecast cycle 2018082800
# following a band around the forecasted storm track by HWRF/POM

#N = 1
temp_POM_band_exp , salt_POM_band_exp, dens_POM_band_exp, \
zmatrix_POM_band_exp, time_POM = \
get_profiles_from_POM(Nini,folder_pom_exp,prefix_pom,lon_band,lat_band,\
                                          lon_pom_exp,lat_pom_exp,zlev_pom_exp,zmatrix_pom_exp)

#%%    

temp_HYCOM_band_exp, time_HYCOM = \
    get_profiles_from_HYCOM(Nini,folder_hycom_exp,prefix_hycom,\
                            lon_band,lat_band,lon_hycom,lat_hycom,'temp')
        
salt_HYCOM_band_exp, time_HYCOM = \
    get_profiles_from_HYCOM(Nini,folder_hycom_exp,prefix_hycom,\
                            lon_band,lat_band,lon_hycom,lat_hycom,'salinity')

#%% Calculate density for HYCOM

nx = temp_HYCOM_band_exp.shape[1]
dens_HYCOM_band_exp = sw.dens(salt_HYCOM_band_exp,temp_HYCOM_band_exp,np.tile(depth_HYCOM_exp,(nx,1)).T) 
    
#%% Reading GOFS temperature and salinity for firts time step in forecast cycle 2018082800
# following a band around the forecasted storm track by HWRF/POM     

DF = GOFS
target_time = time_POM 

temp_GOFS_band , salt_GOFS_band = \
get_profiles_from_GOFS(DF,target_time,lon_band,lat_band)

#%% Calculate density for GOFS

nx = temp_GOFS_band.shape[1]
dens_GOFS_band = sw.dens(salt_GOFS_band,temp_GOFS_band,np.tile(depth_GOFS,(nx,1)).T)
    
#%% Interpolate GOFS into POM operational 
# Why? because POM vertical grid has more levels in top 2000 meters

temp_orig = temp_GOFS_band
salt_orig = salt_GOFS_band
depth_orig = depth_GOFS
nz = len(zlev_pom_oper)
nx = temp_GOFS_band.shape[1]
depth_target = -zmatrix_POM_band_oper

temp_GOFS_band_to_pom_oper, salt_GOFS_band_to_pom_oper = \
interp_datasets_in_z(temp_orig,salt_orig,depth_orig,depth_target,nz,nx)

#%% Interpolate GOFS into POM experimental 
# Why? because POM vertical grid has more levels in top 2000 meters

temp_orig = temp_GOFS_band
salt_orig = salt_GOFS_band
depth_orig = depth_GOFS
nz = len(zlev_pom_exp)
nx = temp_GOFS_band.shape[1]
depth_target = -zmatrix_POM_band_exp

temp_GOFS_band_to_pom_exp, salt_GOFS_band_to_pom_exp = \
interp_datasets_in_z(temp_orig,salt_orig,depth_orig,depth_target,nz,nx)

#%% Calculation of mixed layer depth based on temperature and density critria
# Tmean: mean temp within the mixed layer and 
# td: temp at 1 meter below the mixed layer            

dt = 0.2
drho = 0.125

# for GOFS 3.1 output 
MLD_temp_crit_GOFS, _, _, _, MLD_dens_crit_GOFS, Tmean_dens_crit_GOFS, Smean_dens_crit_GOFS, _ = \
MLD_temp_and_dens_criteria(dt,drho,target_time,depth_GOFS,temp_GOFS_band,salt_GOFS_band,dens_GOFS_band)          

# for POM operational
MLD_temp_crit_POM_oper, _, _, _, MLD_dens_crit_POM_oper, Tmean_dens_crit_POM_oper, \
Smean_dens_crit_POM_oper, _ = \
MLD_temp_and_dens_criteria(dt,drho,time_POM,zmatrix_POM_band_oper,temp_POM_band_oper,\
                           salt_POM_band_oper,dens_POM_band_oper)

# for POM experimental
MLD_temp_crit_POM_exp, _, _, _, MLD_dens_crit_POM_exp, Tmean_dens_crit_POM_exp, \
Smean_dens_crit_POM_exp, _ = \
MLD_temp_and_dens_criteria(dt,drho,time_POM,zmatrix_POM_band_exp,temp_POM_band_exp,\
                           salt_POM_band_exp,dens_POM_band_exp)
    
# for HYCOM experimental 
depth=depth_HYCOM_exp
temp=np.asarray(temp_HYCOM_band_exp)
salt=np.asarray(salt_HYCOM_band_exp)
dens=np.asarray(dens_HYCOM_band_exp)
time = time_HYCOM 
temp[temp>100]=np.nan
salt[salt>100]=np.nan
dens[dens<1000]=np.nan

MLD_temp_crit_HYCOM_exp, _, _, _, MLD_dens_crit_HYCOM_exp, Tmean_dens_crit_HYCOM_exp, \
Smean_dens_crit_HYCOM_exp, _ = \
MLD_temp_and_dens_criteria(dt,drho,depth,temp,salt,dens) 
    
#%% Surface Ocean Heat Content

# GOFS
OHC_GOFS = OHC_surface(temp_GOFS_band,depth_GOFS,dens_GOFS_band)

# POM operational    
OHC_POM_oper = OHC_surface(temp_POM_band_oper,zmatrix_POM_band_oper,\
                           dens_POM_band_oper)

# POM experimental
OHC_POM_exp = OHC_surface(temp_POM_band_exp,zmatrix_POM_band_exp,\
                           dens_POM_band_exp)
    
# HYCOM experimental
OHC_HYCOM_exp = OHC_surface(temp_HYCOM_band_exp,depth_HYCOM_exp,\
                           dens_HYCOM_band_exp)  

#%% Calculate T100

T100_GOFS = depth_aver_top_100(depth_GOFS,temp_GOFS_band)

T100_POM_oper = depth_aver_top_100(zmatrix_POM_band_oper,temp_POM_band_oper) 

T100_POM_exp = depth_aver_top_100(zmatrix_POM_band_exp,temp_POM_band_exp)

T100_HYCOM_exp = depth_aver_top_100(depth_HYCOM_exp,temp_HYCOM_band_exp)    

#%% Figure forecasted track models vs best track

#Time window
date_ini = cycle[0:4]+'-'+cycle[4:6]+'-'+cycle[6:8]+' '+cycle[8:]+':00:00'
tini = datetime.strptime(date_ini,'%Y-%m-%d %H:%M:%S')
tend = tini + timedelta(hours=float(lead_time_pom_oper[-1]))
date_end = str(tend)

okt = np.logical_and(time_best_track >= tini,time_best_track <= tend)

# time forecasted track_exp
time_forec_track_oper = np.asarray([tini + timedelta(hours = float(t)) for t in lead_time_pom_oper])
oktt = [np.where(t == time_forec_track_oper)[0][0] for t in time_best_track[okt]]

str_time = [str(tt)[5:13] for tt in time_forec_track_oper[oktt]]

lev = np.arange(-9000,9100,100)

fig,ax = plt.subplots()    
#plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo)
plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell') 
plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')   
plt.plot(lon_forec_track_pom_oper[oktt], lat_forec_track_pom_oper[oktt],'X-',color='mediumorchid',\
         markeredgecolor='k',label='POM Oper',markersize=7)
plt.plot(lon_forec_track_pom_exp[oktt], lat_forec_track_pom_exp[oktt],'^-',color='teal',\
         markeredgecolor='k',label='POM Exp',markersize=7)
plt.plot(lon_forec_track_hycom_exp[oktt], lat_forec_track_hycom_exp[oktt],'H-',color='orange',\
         markeredgecolor='k',label='HYCOM Exp',markersize=7)
plt.plot(lon_best_track[okt], lat_best_track[okt],'o-',color='k',label='Best Track')   
plt.legend()
plt.title('Track Forecast Dorian cycle='+ cycle,fontsize=18)
plt.xlim([np.min(lon_forec_track_pom_oper[oktt])-0.5,np.max(lon_forec_track_pom_oper[oktt])+0.5])
plt.ylim([np.min(lat_forec_track_pom_oper[oktt])-0.5,np.max(lat_forec_track_pom_oper[oktt])+0.5])

for i,t in enumerate(str_time[::2]):
    ax.text(lon_forec_track_pom_oper[oktt][::2][i]+0.2,lat_forec_track_pom_oper[oktt][::2][i],str_time[::2][i])

file = folder_fig + 'best_track_vs_forec_track_' + cycle 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Figure forecasted track models vs best track with along track distance

#Time window
date_ini = cycle[0:4]+'-'+cycle[4:6]+'-'+cycle[6:8]+' '+cycle[8:]+':00:00'
tini = datetime.strptime(date_ini,'%Y-%m-%d %H:%M:%S')
tend = tini + timedelta(hours=float(lead_time_pom_oper[-1]))
date_end = str(tend)

okt = np.logical_and(time_best_track >= tini,time_best_track <= tend)

# time forecasted track_exp
time_forec_track_oper = np.asarray([tini + timedelta(hours = float(t)) for t in lead_time_pom_oper])
oktt = [np.where(t == time_forec_track_oper)[0][0] for t in time_best_track[okt]]
str_time = [str(tt)[5:13] for tt in time_forec_track_oper[oktt]]

dist_along_track = np.cumsum(np.append(0,sw.dist(lat_bnd[0],lon_bnd[0],units='km')[0]))

lev = np.arange(-9000,9100,100)

fig,ax = plt.subplots()    
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo)
plt.contourf(bath_lon,bath_lat,bath_elev,[0,10000],colors='seashell') 
plt.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')   
plt.plot(lon_forec_track_pom_oper[oktt], lat_forec_track_pom_oper[oktt],'X-',color='mediumorchid',\
         markeredgecolor='k',label='POM Oper',markersize=7)
plt.plot(lon_forec_track_pom_exp[oktt], lat_forec_track_pom_exp[oktt],'^-',color='teal',\
         markeredgecolor='k',label='POM Exp',markersize=7)
plt.plot(lon_forec_track_hycom_exp[oktt], lat_forec_track_hycom_exp[oktt],'H-',color='orange',\
         markeredgecolor='k',label='HYCOM Exp',markersize=7)
plt.plot(lon_best_track[okt], lat_best_track[okt],'o-',color='k',label='Best Track')   
plt.legend()
plt.title('Track Forecast Dorian cycle='+ cycle,fontsize=18)
plt.xlim([np.min(lon_forec_track_pom_oper[oktt])-0.5,np.max(lon_forec_track_pom_oper[oktt])+0.5])
plt.ylim([np.min(lat_forec_track_pom_oper[oktt])-0.5,np.max(lat_forec_track_pom_oper[oktt])+0.5])

for i,t in enumerate(dist_along_track[oktt][::2]):
    ax.text(lon_forec_track_pom_oper[oktt][::2][i]+0.2,lat_forec_track_pom_oper[oktt][::2][i],np.round(dist_along_track[oktt][::2][i]))

file = folder_fig + 'best_track_vs_forec_track_alog_track_dist' + cycle 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Figure forecasted intensity models vs best intensity

fig,ax1 = plt.subplots(figsize=(10, 5))
plt.ion()
plt.plot(time_best_track,wind_int_kt,'o-k',label='Best')
plt.plot(time_hwrf,max_wind_10m_hwrf_pom19_oper,'X-',color='mediumorchid',label='HWRF2010-POM Oper',markeredgecolor='k',markersize=7)
plt.plot(time_hwrf,max_wind_10m_hwrf_pom20_exp,'^-',color='teal',label='HWRF2020-POM Exp',markeredgecolor='k',markersize=7)
plt.plot(time_hwrf,max_wind_10m_hwrf_hycom20_exp,'H-',color='darkorange',label='HWRF2020-HYCOM Exp',markeredgecolor='k',markersize=7)
plt.legend(loc='lower right')
#plt.legend()
plt.xlim([time_hwrf[0],time_hwrf[-1]])
xfmt = mdates.DateFormatter('%d \n %b')
ax1.xaxis.set_major_formatter(xfmt)
plt.ylim([20,165])
plt.title('Intensity Forecast Dorian '+ cycle,fontsize=18)
plt.ylabel('Max 10m Wind (kt)',fontsize=14)

ax2 = ax1.twinx()
plt.ylim([20,165])
yticks = [64,83,96,113,137]
plt.yticks(yticks,['Cat 1','Cat 2','Cat 3','Cat 4','Cat 5'])
plt.grid(True)

file = folder_fig + 'best_intensity_vs_forec_intensity_' + cycle 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Figure forecasted intensity models vs best intensity

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)

fig,ax1 = plt.subplots(figsize=(10, 5))
plt.ion()
plt.plot(lead_time_pom_oper[::2],wind_int_kt[okt],'o-k',label='Best')
plt.plot(lead_time_pom_oper,max_wind_10m_hwrf_pom19_oper,'X-',color='mediumorchid',label='HWRF2010-POM Oper',markeredgecolor='k',markersize=7)
plt.plot(lead_time_pom_exp,max_wind_10m_hwrf_pom20_exp,'^-',color='teal',label='HWRF2020-POM Exp',markeredgecolor='k',markersize=7)
plt.plot(lead_time_hycom_exp,max_wind_10m_hwrf_hycom20_exp,'H-',color='darkorange',label='HWRF2020-HYCOM Exp',markeredgecolor='k',markersize=7)
plt.legend(loc='lower right')
plt.ylim([20,165])
plt.xlim([0,126])
plt.title('Intensity Forecast Dorian '+ cycle,fontsize=18)
plt.ylabel('Max 10m Wind (kt)',fontsize=14)
plt.xlabel('Forecast Lead Time (Hr)',fontsize=14)
ax1.xaxis.set_major_locator(MultipleLocator(12))
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax1.xaxis.set_minor_locator(MultipleLocator(3))

ax2 = ax1.twinx()
plt.ylim([20,165])
yticks = [64,83,96,113,137]
plt.yticks(yticks,['Cat 1','Cat 2','Cat 3','Cat 4','Cat 5'])
plt.grid(True)

file = folder_fig + 'best_intensity_vs_forec_intensity_' + cycle 
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Intensity error

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)

int_err_hwrf19_pom_oper = (wind_int_kt[okt] - max_wind_10m_hwrf_pom19_oper[::2]) #*100/wind_int_kt[okt]
int_err_hwrf20_pom_exp = (wind_int_kt[okt] - max_wind_10m_hwrf_pom20_exp[::2]) #*100/wind_int_kt[okt]
int_err_hwrf20_hycom_exp = (wind_int_kt[okt] - max_wind_10m_hwrf_hycom20_exp[::2]) #*100/wind_int_kt[okt]

fig,ax1 = plt.subplots(figsize=(10, 5))
plt.ion()
plt.plot(lead_time_pom_oper[::2],int_err_hwrf19_pom_oper,'X-',color='mediumorchid',label='HWRF2019-POM Oper',markeredgecolor='k',markersize=7)
plt.plot(lead_time_pom_exp[::2],int_err_hwrf20_pom_exp,'^-',color='teal',label='HRWF2020-POM Exp',markeredgecolor='k',markersize=7)
plt.plot(lead_time_hycom_exp[::2],int_err_hwrf20_hycom_exp,'H-',color='darkorange',label='HWRF2020-HYCOM Exp',markeredgecolor='k',markersize=7)
plt.plot(lead_time_pom_oper[::2],np.tile(0,len(lead_time_pom_oper[::2])),'--k')
plt.xlim([0,126])
ax1.xaxis.set_major_locator(MultipleLocator(12))
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax1.xaxis.set_minor_locator(MultipleLocator(3))
plt.title('Intensity Forecast Error Dorian '+ cycle,fontsize=18)
plt.ylabel('Forecast Error (Kt)',fontsize=14)
plt.xlabel('Forecast Lead Time (Hr)',fontsize=14)
plt.legend()

#%% Top 200 m GOFS 3.1 temperature along forecasted Dorian track

color_map = cmocean.cm.thermal
       
okm = depth_GOFS <= 200 
#min_val = np.floor(np.min([np.nanmin(tempg_gridded[okg]),np.nanmin(target_temp_GOFS[okm])]))
#max_val = np.ceil(np.max([np.nanmax(tempg_gridded[okg]),np.nanmax(target_temp_GOFS[okm])]))
    
#nlevels = max_val - min_val + 1
#kw = dict(levels = np.linspace(min_val,max_val,nlevels))
kw = dict(levels = np.linspace(16,31,16))

dist_along_track = np.cumsum(np.append(0,sw.dist(lat_bnd[0],lon_bnd[0],units='km')[0]))

fig, ax = plt.subplots(figsize=(12, 2))     
cs = plt.contourf(dist_along_track,-depth_GOFS,temp_GOFS_band,cmap=color_map,**kw)
plt.contour(dist_along_track,-depth_GOFS,temp_GOFS_band,[26],colors='k')
#plt.plot(time_GOFS,-MLD_temp_crit_GOFS,'-o',label='MLD dt',color='indianred' )
#plt.plot(time_GOFS,-MLD_dens_crit_GOFS,'-o',label='MLD drho',color='seagreen' )
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
ax.set_ylim(-200, 0)
ax.set_ylabel('Depth (m)',fontsize=14)
ax.set_xlabel('Distance Along Track (km)',fontsize=14)
plt.title('Along Forecasted Track ' + 'Temperature ' + 'GOFS 3.1 on '+ str(time_POM)[0:13] + ' (cycle= ' + cycle +')',fontsize=14)  

file = folder_fig + ' ' + 'along_track_temp_top200_GOFS_' + cycle + '_' + str(time_POM)[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%% Top 200 m POM oper temperature along forecasted Dorian track

color_map = cmocean.cm.thermal
dist_matrix = np.tile(dist_along_track,(zmatrix_POM_band_oper.shape[0],1))
       
okm = depth_GOFS <= 200 
#min_val = np.floor(np.min([np.nanmin(tempg_gridded[okg]),np.nanmin(target_temp_GOFS[okm])]))
#max_val = np.ceil(np.max([np.nanmax(tempg_gridded[okg]),np.nanmax(target_temp_GOFS[okm])]))
    
#nlevels = max_val - min_val + 1
#kw = dict(levels = np.linspace(min_val,max_val,nlevels))
kw = dict(levels = np.linspace(16,31,16))

fig, ax = plt.subplots(figsize=(12, 2))     
cs = plt.contourf(dist_matrix,zmatrix_POM_band_oper,temp_POM_band_oper,cmap=color_map,**kw)
plt.contour(dist_matrix,zmatrix_POM_band_oper,temp_POM_band_oper,[26],colors='k')
#plt.plot(time_GOFS,-MLD_temp_crit_GOFS,'-o',label='MLD dt',color='indianred' )
#plt.plot(time_GOFS,-MLD_dens_crit_GOFS,'-o',label='MLD drho',color='seagreen' )
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)
ax.set_ylim(-200, 0)
ax.set_ylabel('Depth (m)',fontsize=14) 
ax.set_xlabel('Distance Along Track (km)',fontsize=14)
plt.title('Along Forecasted Track ' + 'Temperature '  + 'POM Operational on '+ str(time_POM)[0:13] + ' (cycle= ' + cycle +')',fontsize=14)  
 

file = folder_fig + ' ' + 'along_track_temp_top200_POM_oper_' + cycle + '_' + str(time_POM)[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%% Top 200 m POM exp temperature along forecasted Dorian track

color_map = cmocean.cm.thermal
lat_matrix = np.tile(lat_bnd,(zmatrix_POM_band_exp.shape[0],1))
       
okm = depth_GOFS <= 200 
#min_val = np.floor(np.min([np.nanmin(tempg_gridded[okg]),np.nanmin(target_temp_GOFS[okm])]))
#max_val = np.ceil(np.max([np.nanmax(tempg_gridded[okg]),np.nanmax(target_temp_GOFS[okm])]))
    
#nlevels = max_val - min_val + 1
#kw = dict(levels = np.linspace(min_val,max_val,nlevels))
kw = dict(levels = np.linspace(16,31,16))

fig, ax = plt.subplots(figsize=(12, 2))     
cs = plt.contourf(dist_matrix,zmatrix_POM_band_exp,temp_POM_band_exp,cmap=color_map,**kw)
plt.contour(dist_matrix,zmatrix_POM_band_exp,temp_POM_band_exp,[26],colors='k')
#plt.plot(time_GOFS,-MLD_temp_crit_GOFS,'-o',label='MLD dt',color='indianred' )
#plt.plot(time_GOFS,-MLD_dens_crit_GOFS,'-o',label='MLD drho',color='seagreen' )
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)

ax.set_ylim(-200, 0)
ax.set_ylabel('Depth (m)',fontsize=14)
ax.set_xlabel('Distance Along Track (km)',fontsize=14)
plt.title('Along Forecasted Track ' + 'Temperature ' + 'POM Experimental on '+str(time_POM)[0:13] + ' (cycle= ' + cycle +')',fontsize=14)  
 

file = folder_fig + ' ' + 'along_track_temp_top200_POM_exp_' + cycle + '_' + str(time_POM)[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%% Top 200 m HYCOM exp temperature along forecasted Dorian track

color_map = cmocean.cm.thermal
#lat_matrix = np.tile(lat_bnd,(zmatrix_POM_band_exp.shape[0],1))
       
okm = depth_HYCOM_exp <= 200 
#min_val = np.floor(np.min([np.nanmin(tempg_gridded[okg]),np.nanmin(target_temp_GOFS[okm])]))
#max_val = np.ceil(np.max([np.nanmax(tempg_gridded[okg]),np.nanmax(target_temp_GOFS[okm])]))
    
#nlevels = max_val - min_val + 1
#kw = dict(levels = np.linspace(min_val,max_val,nlevels))
kw = dict(levels = np.linspace(16,31,16))

fig, ax = plt.subplots(figsize=(12, 2))     
cs = plt.contourf(dist_along_track,-depth_HYCOM_exp,temp_HYCOM_band_exp,cmap=color_map,**kw)
plt.contour(dist_along_track,-depth_HYCOM_exp,temp_HYCOM_band_exp,[26],colors='k')
#plt.plot(time_GOFS,-MLD_temp_crit_GOFS,'-o',label='MLD dt',color='indianred' )
#plt.plot(time_GOFS,-MLD_dens_crit_GOFS,'-o',label='MLD drho',color='seagreen' )
cs = fig.colorbar(cs, orientation='vertical') 
cs.ax.set_ylabel('($^oC$)',fontsize=14,labelpad=15)

ax.set_ylim(-200, 0)
ax.set_ylabel('Depth (m)',fontsize=14)
ax.set_xlabel('Distance Along Track (km)',fontsize=14)
plt.title('Along Forecasted Track ' + 'Temperature ' + 'HYCOM Experimental on '+ str(time_POM)[0:13] + ' (cycle= ' + cycle +')',fontsize=14)  
 

file = folder_fig + ' ' + 'along_track_temp_top200_HYCOM_exp_' + cycle + '_' + str(time_POM)[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%% time series temp ML

fig,ax = plt.subplots(figsize=(12, 2))
plt.plot(dist_along_track,Tmean_dens_crit_GOFS,'--o',color='indianred',label='GOFS 3.1')
plt.plot(dist_along_track,Tmean_dens_crit_POM_oper,'-X',color='mediumorchid',label='POM Oper')
plt.plot(dist_along_track,Tmean_dens_crit_POM_exp,'-^',color='teal',label='POM Exp')
plt.plot(dist_along_track,Tmean_dens_crit_HYCOM_exp,'-H',color='orange',label='HYCOM Exp')
plt.ylabel('($^oC$)',fontsize = 14)
plt.xlabel('Distance Along Track (km)',fontsize = 14)
plt.title('Mixed Layer Temperature Dorian Track on ' + str(time_POM)[0:13] + ' (cycle= ' + cycle +')',fontsize=16)
plt.grid(True)
plt.legend(loc='upper left',bbox_to_anchor=(1,0.9))

file = folder_fig + 'temp_ml_' + cycle + ' ' + str(time_POM)[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%% time series temp OHC

fig,ax = plt.subplots(figsize=(12, 2))
plt.plot(dist_along_track,OHC_GOFS/10**7,'--o',color='indianred',label='GOFS 3.1')
plt.plot(dist_along_track,OHC_POM_oper/10**7,'-X',color='mediumpurple',label='POM Oper')
plt.plot(dist_along_track,OHC_POM_exp/10**7,'-^',color='teal',label='POM Exp')
plt.plot(dist_along_track,OHC_HYCOM_exp/10**7,'-H',color='orange',label='HYCOM Exp')
plt.ylabel('($KJ/cm^2$)',fontsize = 14)
plt.xlabel('Distance Along Track (km)',fontsize = 14)
plt.title('Ocean Heat Content Dorian Track on ' + str(time_POM)[0:13] + ' (cycle= ' + cycle +')',fontsize=16)
plt.grid(True)
plt.legend(loc='upper left',bbox_to_anchor=(1,0.9))

file = folder_fig + 'OHC_' + cycle + ' ' + str(time_POM)[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%% time series temp T100

fig,ax = plt.subplots(figsize=(12, 2))
plt.plot(dist_along_track,T100_GOFS,'--o',color='indianred',label='GOFS 3.1')
plt.plot(dist_along_track,T100_POM_oper,'-X',color='mediumpurple',label='POM Oper')
plt.plot(dist_along_track,T100_POM_exp,'-^',color='teal',label='POM Exp')
plt.plot(dist_along_track,T100_HYCOM_exp,'-H',color='orange',label='HYCOM Exp')
plt.ylabel('($^oC$)',fontsize = 14)
plt.xlabel('Distance Along Track (km)',fontsize = 14)
plt.title('T100 Dorian Track on ' + str(time_POM)[0:13] + ' (cycle= ' + cycle +')',fontsize=16)
plt.grid(True)
plt.legend(loc='upper left',bbox_to_anchor=(1,0.9))

file = folder_fig + 'T100_' + cycle + ' ' + str(time_POM)[0:13]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 


#%% Surface temperature GOFS 3.1

GOFS = xr.open_dataset(url_GOFS,decode_times=False) 

lat_G = np.asarray(GOFS.lat[:])
lon_G = np.asarray(GOFS.lon[:]) 
oklon = np.round(np.interp(lon_forec_track_pom_oper+360,DF.lon,np.arange(len(DF.lon)))).astype(int)
oklat = np.round(np.interp(lat_forec_track_pom_oper,DF.lat,np.arange(len(DF.lat)))).astype(int)
oktime = np.where(mdates.date2num(t_G) == mdates.date2num(time_POM))[0][0]
temp_GOFS = np.asarray(GOFS.water_temp[oktime,0,oklat,oklon])
#temp_gofs = temp_GOFS
#temp_gofs[temp_gofs<27] = 27
#temp_gofs[temp_gofs>32] = 

kw = dict(levels = np.linspace(26,32,17))
plt.figure()
plt.contourf(lon_G[oklon]-360,lat_G[oklat],temp_GOFS,cmap=color_map,**kw)
plt.colorbar()

#%% 100 m temperature GOFS 3.1

GOFS = xr.open_dataset(url_GOFS,decode_times=False) 

lat_G = np.asarray(GOFS.lat[:])
lon_G = np.asarray(GOFS.lon[:]) 
oklon = np.round(np.interp(lon_forec_track_pom_oper+360,DF.lon,np.arange(len(DF.lon)))).astype(int)
oklat = np.round(np.interp(lat_forec_track_pom_oper,DF.lat,np.arange(len(DF.lat)))).astype(int)
oktime = np.where(mdates.date2num(t_G) == mdates.date2num(time_POM))[0][0]
okd = np.where(depth_GOFS == 150)[0][0]
temp_GOFS = np.asarray(GOFS.water_temp[oktime,okd,oklat,oklon])

kw = dict(levels = np.linspace(26,32,17))
plt.figure()
plt.contourf(lon_G[oklon]-360,lat_G[oklat],temp_GOFS,cmap=color_map,**kw)
plt.colorbar()

#%% Surface temperature POM operational

folder_pom =folder_pom_oper
#folder_pom =folder_pom_exp
prefix = prefix_pom
N = Nini

pom_ncfiles = sorted(glob.glob(os.path.join(folder_pom,prefix+'*.nc')))   
file = pom_ncfiles[N]
pom = xr.open_dataset(file)
temp_POM = np.asarray(pom['t'][0,0,:,:])
temp_POM[temp_POM==0] = np.nan
#temp_POM[temp_POM<27] = 27
#temp_POM[temp_POM>32] = 32

kw = dict(levels = np.linspace(16,32,17))
#kw = dict(levels = np.linspace(26,32,17))
plt.figure()
plt.contourf(lon_pom_oper[0,:],lat_pom_oper[:,0],temp_POM,cmap=color_map,**kw)
plt.xlim([np.min(lon_forec_track_pom_oper)-0.5,np.max(lon_forec_track_pom_oper)+0.5])
plt.ylim([np.min(lat_forec_track_pom_oper)-0.5,np.max(lat_forec_track_pom_oper)+0.5])
plt.colorbar()


#%% Temperature 140m to 160m POM operational

pom_ncfiles = sorted(glob.glob(os.path.join(folder_pom,prefix+'*.nc')))
file = pom_ncfiles[N]
pom = xr.open_dataset(file)
tpom = pom['time'][:]
timestamp_pom = mdates.date2num(tpom)[0]
time_POM = mdates.num2date(timestamp_pom)

temp_pom = np.asarray(pom['t'][0,:,:,:])
temp_pom[temp_pom==0] = np.nan
temp_p = temp_pom.reshape(temp_pom.shape[0],temp_pom.shape[1]*temp_pom.shape[2])
temp_pp = temp_p.reshape(temp_pom.shape[0]*temp_pom.shape[1]*temp_pom.shape[2],order='F')

zmatrix_p = zmatrix_pom_oper.reshape(temp_pom.shape[1]*temp_pom.shape[2],temp_pom.shape[0])
zmatrix_pp = zmatrix_p.reshape(temp_pom.shape[1]*temp_pom.shape[2]*temp_pom.shape[0],order='C')

okdepth = np.logical_and(zmatrix_pp < -140,zmatrix_pp > -170)
zmatrix_pp[okdepth]
temp_pp[okdepth]

temp_pom_140m = np.empty((zmatrix_pom_oper.shape[0]*zmatrix_pom_oper.shape[1]))
temp_pom_140m[:] = np.nan

okd = np.where((np.logical_and(zmatrix_p<-140,zmatrix_p>-170)))[0]
temp_pom_140m[okd] = temp_pp[okdepth]
temp_POM_140m = temp_pom_140m.reshape(zmatrix_pom_oper.shape[0],zmatrix_pom_oper.shape[1])

kw = dict(levels = np.linspace(16.5,27,7))
fig,ax = plt.subplots()
plt.contourf(lon_pom_oper[0,:],lat_pom_oper[:,0],temp_POM_140m,cmap=color_map,**kw)
plt.colorbar()
plt.xlim([np.min(lon_forec_track_pom_oper[oktt])-0.5,np.max(lon_forec_track_pom_oper[oktt])+0.5])
plt.ylim([np.min(lat_forec_track_pom_oper[oktt])-0.5,np.max(lat_forec_track_pom_oper[oktt])+0.5])
plt.plot(lon_forec_track_pom_oper[oktt], lat_forec_track_pom_oper[oktt],'X-',color='mediumorchid',\
         markeredgecolor='k',label='POM Oper',markersize=7)
plt.title('POM oper Temp at 124-160 m on ' + str(time_POM)[0:13])

for i,t in enumerate(dist_along_track[oktt][::2]):
    ax.text(lon_forec_track_pom_oper[oktt][::2][i]+0.2,lat_forec_track_pom_oper[oktt][::2][i],np.round(dist_along_track[oktt][::2][i]))

#%% Temperature 140m to 160m POM experimental
N = 10
pom_ncfiles = sorted(glob.glob(os.path.join(folder_pom_exp,prefix_pom+'*.nc')))
file = pom_ncfiles[N]
pom = xr.open_dataset(file)
tpom = pom['time'][:]
timestamp_pom = mdates.date2num(tpom)[0]
time_POM = mdates.num2date(timestamp_pom)

temp_pom = np.asarray(pom['t'][0,:,:,:])
temp_pom[temp_pom==0] = np.nan
temp_p = temp_pom.reshape(temp_pom.shape[0],temp_pom.shape[1]*temp_pom.shape[2])
temp_pp = temp_p.reshape(temp_pom.shape[0]*temp_pom.shape[1]*temp_pom.shape[2],order='F')

zmatrix_p = zmatrix_pom_oper.reshape(temp_pom.shape[1]*temp_pom.shape[2],temp_pom.shape[0])
zmatrix_pp = zmatrix_p.reshape(temp_pom.shape[1]*temp_pom.shape[2]*temp_pom.shape[0],order='C')

okdepth = np.logical_and(zmatrix_pp < -140,zmatrix_pp > -170)
zmatrix_pp[okdepth]
temp_pp[okdepth]

temp_pom_140m = np.empty((zmatrix_pom_oper.shape[0]*zmatrix_pom_oper.shape[1]))
temp_pom_140m[:] = np.nan

okd = np.where((np.logical_and(zmatrix_p<-140,zmatrix_p>-170)))[0]
temp_pom_140m[okd] = temp_pp[okdepth]
temp_POM_140m = temp_pom_140m.reshape(zmatrix_pom_oper.shape[0],zmatrix_pom_oper.shape[1])
temp_pom_140m[temp_pom_140m > 27.0] = 27.0

kw = dict(levels = np.linspace(16.5,27,7))
fig,ax = plt.subplots()
plt.contourf(lon_pom_oper[0,:],lat_pom_oper[:,0],temp_POM_140m,cmap=color_map,**kw)
plt.colorbar()
plt.plot(lon_forec_track_pom_oper[oktt], lat_forec_track_pom_oper[oktt],'X-',color='mediumorchid',\
         markeredgecolor='k',label='POM Oper',markersize=7)
plt.xlim([np.min(lon_forec_track_pom_oper[oktt])-0.5,np.max(lon_forec_track_pom_oper[oktt])+0.5])
plt.ylim([np.min(lat_forec_track_pom_oper[oktt])-0.5,np.max(lat_forec_track_pom_oper[oktt])+0.5])
plt.title('POM exp Temp at 140-160 m on ' + str(time_POM)[0:13])

for i,t in enumerate(dist_along_track[oktt][::2]):
    ax.text(lon_forec_track_pom_oper[oktt][::2][i]+0.2,lat_forec_track_pom_oper[oktt][::2][i],np.round(dist_along_track[oktt][::2][i]))

#%% Surface temperature HYCOM experimental

folder_hycom =folder_hycom_exp
prefix = prefix_hycom
N = Nini
var = 'temp'

afiles = sorted(glob.glob(os.path.join(folder_hycom,prefix+'*.a')))    
file = afiles[N]

# Interpolating lat_track and lon_track into HYCOM grid
oklon = np.round(np.interp(lon_forec_track_pom_oper+360,lon_hycom[0,:],np.arange(len(lon_hycom[0,:])))).astype(int)
oklat = np.round(np.interp(lat_forec_track_pom_oper,lat_hycom[:,0],np.arange(len(lat_hycom[:,0])))).astype(int)

# Reading 3D variable from binary file 
var_hyc = readBinz(file[:-2],'3z',var)
temp_HYCOM = var_hyc[oklat,:,0][:,oklon]

kw = dict(levels = np.linspace(26,32,17))
plt.figure()
plt.contourf(lon_hycom[0,oklon]-360,lat_hycom[oklat,0],temp_HYCOM,cmap=color_map,**kw)
#plt.xlim([np.min(lon_forec_track_pom_oper[oktt])-0.5,np.max(lon_forec_track_pom_oper[oktt])+0.5])
#plt.ylim([np.min(lat_forec_track_pom_oper[oktt])-0.5,np.max(lat_forec_track_pom_oper[oktt])+0.5])
plt.colorbar()

#%% 100 m temperature HYCOM experimental

#Time window
date_ini = cycle[0:4]+'-'+cycle[4:6]+'-'+cycle[6:8]+' '+cycle[8:]+':00:00'
tini = datetime.strptime(date_ini,'%Y-%m-%d %H:%M:%S')
tend = tini + timedelta(hours=float(lead_time_pom_oper[-1]))
date_end = str(tend)

okt = np.logical_and(time_best_track >= tini,time_best_track <= tend)

# time forecasted track_exp
time_forec_track_oper = np.asarray([tini + timedelta(hours = float(t)) for t in lead_time_pom_oper])
oktt = [np.where(t == time_forec_track_oper)[0][0] for t in time_best_track[okt]]
str_time = [str(tt)[5:13] for tt in time_forec_track_oper[oktt]]

dist_along_track = np.cumsum(np.append(0,sw.dist(lat_bnd[0],lon_bnd[0],units='km')[0]))

folder_hycom =folder_hycom_exp
prefix = prefix_hycom
N = 2
var = 'temp'

afiles = sorted(glob.glob(os.path.join(folder_hycom,prefix+'*.a')))    
file = afiles[N]

#Reading time stamp
year = int(file.split('/')[-1].split('.')[1][0:4])
month = int(file.split('/')[-1].split('.')[1][4:6])
day = int(file.split('/')[-1].split('.')[1][6:8])
hour = int(file.split('/')[-1].split('.')[1][8:10])
dt = int(file.split('/')[-1].split('.')[3][1:])
timestamp_hycom = mdates.date2num(datetime(year,month,day,hour)) + dt/24
time_hycom = mdates.num2date(timestamp_hycom)

# Interpolating lat_track and lon_track into HYCOM grid
oklon = np.round(np.interp(lon_forec_track_pom_oper+360,lon_hycom[0,:],np.arange(len(lon_hycom[0,:])))).astype(int)
oklat = np.round(np.interp(lat_forec_track_pom_oper,lat_hycom[:,0],np.arange(len(lat_hycom[:,0])))).astype(int)
okd = np.where(depth_HYCOM_exp == 140)[0][0]

# Reading 3D variable from binary file 
var_hyc = readBinz(file[:-2],'3z',var)
temp_HYCOM = var_hyc[oklat,:,okd][:,oklon]

kw = dict(levels = np.linspace(16.5,27,7))
fig,ax = plt.subplots() 
plt.contourf(lon_hycom[0,oklon]-360,lat_hycom[oklat,0],temp_HYCOM,cmap=color_map,**kw)
plt.plot(lon_forec_track_pom_oper[oktt], lat_forec_track_pom_oper[oktt],'X-',color='mediumorchid',\
         markeredgecolor='k',label='POM Oper',markersize=7)
#plt.xlim([np.min(lon_forec_track_pom_oper[oktt])-0.5,np.max(lon_forec_track_pom_oper[oktt])+0.5])
#plt.ylim([np.min(lat_forec_track_pom_oper[oktt])-0.5,np.max(lat_forec_track_pom_oper[oktt])+0.5])
plt.colorbar()
plt.title('HYCOM Temp at '+ str(depth_HYCOM_exp[okd]) + ' m on ' + str(time_hycom)[0:13])

for i,t in enumerate(dist_along_track[oktt][::2]):
    ax.text(lon_forec_track_pom_oper[oktt][::2][i]+0.2,lat_forec_track_pom_oper[oktt][::2][i],np.round(dist_along_track[oktt][::2][i]))
