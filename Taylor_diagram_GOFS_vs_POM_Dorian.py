#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:20:03 2019

@author: root
"""

#%% User input

lon_lim = [-98.5,-60.0]
lat_lim = [10.0,45.0]

cycle = '2019082800'
#cycle = '2019082918'

delta_lon = 0 # delta longitude around hurricane track to calculate
               # statistics
Nini = 7 # 0 is the start of forecating cycle (2019082800)
      # 1 is 6 hours of forecasting cycle   (2019082806)
      # 2 is 12 hours ...... 20 is 120 hours 

Nend = 22 # indicates how far in the hurricabe track you want
          # include in the analysis. This is helpful if for ex:
          # you onl want to analyse the portion of the track
          # where the storm intensifies
          # 22 corresponds to all the hurricane track forecasted in a cycle
#Nend = 13 

# url for GOFS 3.1
url_GOFS = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'

# KMZ file best track Dorian
kmz_file_Dorian = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/KMZ_files/al052019_best_track-5.kmz'   

# folder figures
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

# folder nc files POM
folder_pom19 =  '/Volumes/aristizabal/HWRF2019_POM_Dorian/'
folder_pom20 =  '/Volumes/aristizabal/HWRF2020_POM_Dorian/'

###################
# folder nc files POM
folder_pom_oper = folder_pom19 + 'HWRF2019_POM_dorian05l.' + cycle + '_pom_files_oper/'
folder_pom_exp = folder_pom20 + 'HWRF2020_POM_dorian05l.'  + cycle + '_pom_files_exp/'
prefix_pom = 'dorian05l.' + cycle + '.pom.00'

pom_grid_oper = folder_pom_oper + 'dorian05l.' + cycle + '.pom.grid.nc'
pom_grid_exp = folder_pom_exp + 'dorian05l.' + cycle + '.pom.grid.nc'

# Dorian track files
track_oper = folder_pom_oper + 'dorian05l.' + cycle + '.trak.hwrf.atcfunix'
track_exp = folder_pom_exp + 'dorian05l.' + cycle + '.trak.hwrf.atcfunix'

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
import os
import os.path
from bs4 import BeautifulSoup
from zipfile import ZipFile
import glob
import seawater as sw

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%%
def MLD_temp_and_dens_criteria(dt,drho,depth,temp,salt,dens):

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
lat_G = np.asarray(GOFS.lat[:])
lon_G = np.asarray(GOFS.lon[:]) 
depth_GOFS = np.asarray(GOFS.depth[:])

#%% Get Dorian track from POM

lon_forec_track_oper, lat_forec_track_oper, lead_time_oper = get_storm_track_POM(track_oper)

lon_forec_track_exp, lat_forec_track_exp, lead_time_exp = get_storm_track_POM(track_exp)

#%% Get Dorian best track 

lon_best_track, lat_best_track, time_best_track, _, _, _ = \
read_kmz_file_storm_best_track(kmz_file_Dorian)

#%% Obtain lat and lon band around forecated track operational

dlon = 0.1
nlevels = int(2*delta_lon /dlon) + 1

lon_bnd = np.linspace(lon_forec_track_oper[2*Nini:2*Nend-1]-delta_lon,lon_forec_track_oper[2*Nini:2*Nend-1]+delta_lon,nlevels) 
lon_band = lon_bnd.ravel()
lat_bd = np.tile(lat_forec_track_oper[2*Nini:2*Nend-1],lon_bnd.shape[0])
lat_bnd = lat_bd.reshape(lon_bnd.shape[0],lon_bnd.shape[1])
lat_band = lat_bnd.ravel()

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
MLD_temp_and_dens_criteria(dt,drho,depth_GOFS,temp_GOFS_band,salt_GOFS_band,dens_GOFS_band)          

# for POM operational
MLD_temp_crit_POM_oper, _, _, _, MLD_dens_crit_POM_oper, Tmean_dens_crit_POM_oper, \
Smean_dens_crit_POM_oper, _ = \
MLD_temp_and_dens_criteria(dt,drho,zmatrix_POM_band_oper,temp_POM_band_oper,\
                           salt_POM_band_oper,dens_POM_band_oper)

# for POM experimental
MLD_temp_crit_POM_exp, _, _, _, MLD_dens_crit_POM_exp, Tmean_dens_crit_POM_exp, \
Smean_dens_crit_POM_exp, _ = \
MLD_temp_and_dens_criteria(dt,drho,zmatrix_POM_band_exp,temp_POM_band_exp,\
                           salt_POM_band_exp,dens_POM_band_exp)

#%% Surface Ocean Heat Content

# GOFS
OHC_GOFS = OHC_surface(temp_GOFS_band,depth_GOFS,dens_GOFS_band)

# POM operational    
OHC_POM_oper = OHC_surface(temp_POM_band_oper,zmatrix_POM_band_oper,\
                           dens_POM_band_oper)

# POM experimental
OHC_POM_exp = OHC_surface(temp_POM_band_exp,zmatrix_POM_band_exp,\
                           dens_POM_band_exp)
    
#%% Calculate T100

T100_GOFS = depth_aver_top_100(depth_GOFS,temp_GOFS_band)

T100_POM_oper = depth_aver_top_100(zmatrix_POM_band_oper,temp_POM_band_oper) 

T100_POM_exp = depth_aver_top_100(zmatrix_POM_band_exp,temp_POM_band_exp)    

#%% Define dataframe

DF_temp_GOFS_POM = pd.DataFrame(data=np.array([np.ravel(temp_GOFS_band_to_pom_oper,order='F'),\
                                      np.ravel(temp_POM_band_oper,order='F'),\
                                      np.ravel(temp_GOFS_band_to_pom_exp,order='F'),\
                                      np.ravel(temp_POM_band_exp,order='F')]).T,\
                  columns=['GOFS_to_POM_oper','POM_oper',\
                           'GOFS_to_POM_exp','POM_exp'])
    
#%% Define dataframe

DF_salt_GOFS_POM = pd.DataFrame(data=np.array([np.ravel(salt_GOFS_band_to_pom_oper,order='F'),\
                                      np.ravel(salt_POM_band_oper,order='F'),\
                                      np.ravel(salt_GOFS_band_to_pom_exp,order='F'),\
                                      np.ravel(salt_POM_band_exp,order='F')]).T,\
                  columns=['GOFS_to_POM_oper','POM_oper',\
                           'GOFS_to_POM_exp','POM_exp'])   
    
#%% Define dataframe

DF_mld_GOFS_POM = pd.DataFrame(data=np.array([MLD_dens_crit_GOFS,MLD_dens_crit_POM_oper,MLD_dens_crit_POM_exp,\
                                          Tmean_dens_crit_GOFS,Tmean_dens_crit_POM_oper,Tmean_dens_crit_POM_exp]).T,\
                      columns=['MLD_GOFS','MLD_POM_oper','MLD_POM_exp',\
                               'Tmean_GOFS','Tmean_POM_oper','Tmean_POM_exp']) 
    
#%% Define dataframe

DF_OHC_GOFS_POM = pd.DataFrame(data=np.array([OHC_GOFS,OHC_POM_oper,OHC_POM_exp]).T,\
                      columns=['OHC_GOFS','OHC_POM_oper','OHC_POM_exp']) 
    
#%% Define dataframe

DF_T100_GOFS_POM = pd.DataFrame(data=np.array([T100_GOFS,T100_POM_oper,T100_POM_exp]).T,\
                      columns=['T100_GOFS','T100_POM_oper','T100_POM_exp'])     
    
#%% Temperature statistics.

DF = DF_temp_GOFS_POM.dropna()

N = len(DF)-1  #For Unbiased estimmator.


cols = ['CORRELATION','OSTD','MSTD','CRMSE','BIAS']
    
tskill = np.empty((3,5))
tskill[:] = np.nan

#CORR
tskill[0,0] = DF.corr()['GOFS_to_POM_oper']['GOFS_to_POM_oper']
tskill[1,0] = DF.corr()['GOFS_to_POM_oper']['POM_oper']
tskill[2,0] = DF.corr()['GOFS_to_POM_exp']['POM_exp']

#OSTD
tskill[0,1] = DF.std().GOFS_to_POM_oper
tskill[1,1] = DF.std().GOFS_to_POM_oper
tskill[2,1] = DF.std().GOFS_to_POM_oper

#MSTD
tskill[0,2] = DF.std().GOFS_to_POM_oper
tskill[1,2] = DF.std().POM_oper
tskill[2,2] = DF.std().POM_exp

#CRMSE
tskill[0,3] = 0
tskill[1,3] = np.sqrt(np.nansum(((DF.GOFS_to_POM_oper-DF.mean().GOFS_to_POM_oper)-\
                                 (DF.POM_oper-DF.mean().POM_oper))**2)/N)
tskill[2,3] = np.sqrt(np.nansum(((DF.GOFS_to_POM_exp-DF.mean().GOFS_to_POM_exp)-\
                                 (DF.POM_exp-DF.mean().POM_exp))**2)/N)

#BIAS
tskill[0,4] = 0
tskill[1,4] = DF.mean().GOFS_to_POM_oper - DF.mean().POM_oper
tskill[2,4] = DF.mean().GOFS_to_POM_exp - DF.mean().POM_exp

#color
colors = ['indianred','seagreen','darkorchid']
    
temp_skillscores = pd.DataFrame(tskill,
                        index=['GOFS_to_POM_oper','POM_oper','POM_exp'],
                        columns=cols)
print(temp_skillscores)

#%% salt statistics.

DF = DF_salt_GOFS_POM.dropna()


N = len(DF)-1  #For Unbiased estimmator.


cols = ['CORRELATION','OSTD','MSTD','CRMSE','BIAS']
    
tskill = np.empty((3,5))
tskill[:] = np.nan

#CORR
tskill[0,0] = DF.corr()['GOFS_to_POM_oper']['GOFS_to_POM_oper']
tskill[1,0] = DF.corr()['GOFS_to_POM_oper']['POM_oper']
tskill[2,0] = DF.corr()['GOFS_to_POM_exp']['POM_exp']

#OSTD
tskill[0,1] = DF.std().GOFS_to_POM_oper
tskill[1,1] = DF.std().GOFS_to_POM_oper
tskill[2,1] = DF.std().GOFS_to_POM_oper

#MSTD
tskill[0,2] = DF.std().GOFS_to_POM_oper
tskill[1,2] = DF.std().POM_oper
tskill[2,2] = DF.std().POM_exp

#CRMSE
tskill[0,3] = 0
tskill[1,3] = np.sqrt(np.nansum(((DF.GOFS_to_POM_oper-DF.mean().GOFS_to_POM_oper)-\
                                 (DF.POM_oper-DF.mean().POM_oper))**2)/N)
tskill[2,3] = np.sqrt(np.nansum(((DF.GOFS_to_POM_exp-DF.mean().GOFS_to_POM_exp)-\
                                 (DF.POM_exp-DF.mean().POM_exp))**2)/N)

#BIAS
tskill[0,4] = 0
tskill[1,4] = DF.mean().GOFS_to_POM_oper - DF.mean().POM_oper
tskill[2,4] = DF.mean().GOFS_to_POM_exp - DF.mean().POM_exp

#color
colors = ['indianred','seagreen','darkorchid']
    
salt_skillscores = pd.DataFrame(tskill,
                        index=['GOFS_to_POM_oper','POM_oper','POM_exp'],
                        columns=cols)
print(salt_skillscores) 
   

#%% MLD statistics.

DF = DF_mld_GOFS_POM.dropna()

N = len(DF)-1  #For Unbiased estimmator.


cols = ['CORRELATION','OSTD','MSTD','CRMSE','BIAS']
    
tskill = np.empty((3,5))
tskill[:] = np.nan

#CORR
tskill[0,0] = DF.corr()['Tmean_GOFS']['Tmean_GOFS']
tskill[1,0] = DF.corr()['Tmean_GOFS']['Tmean_POM_oper']
tskill[2,0] = DF.corr()['Tmean_GOFS']['Tmean_POM_oper']

#OSTD
tskill[0,1] = DF.std().Tmean_GOFS
tskill[1,1] = DF.std().Tmean_GOFS
tskill[2,1] = DF.std().Tmean_GOFS

#MSTD
tskill[0,2] = DF.std().Tmean_GOFS
tskill[1,2] = DF.std().Tmean_POM_oper
tskill[2,2] = DF.std().Tmean_POM_exp

#CRMSE
tskill[0,3] = 0
tskill[1,3] = np.sqrt(np.nansum(((DF.Tmean_GOFS-DF.mean().Tmean_GOFS)-\
                                 (DF.Tmean_POM_oper-DF.mean().Tmean_POM_oper))**2)/N)
tskill[2,3] = np.sqrt(np.nansum(((DF.Tmean_GOFS-DF.mean().Tmean_GOFS)-\
                                 (DF.Tmean_POM_exp-DF.mean().Tmean_POM_exp))**2)/N)

#BIAS
tskill[0,4] = 0
tskill[1,4] = DF.mean().Tmean_GOFS - DF.mean().Tmean_POM_oper
tskill[2,4] = DF.mean().Tmean_GOFS - DF.mean().Tmean_POM_exp

#color
colors = ['indianred','seagreen','darkorchid']
    
Tmean_mld_skillscores = pd.DataFrame(tskill,
                        index=['GOFS_to_POM_oper','POM_oper','POM_exp'],
                        columns=cols)
print(Tmean_mld_skillscores) 

#%% OHC statistics.

DF = DF_OHC_GOFS_POM.dropna()

N = len(DF)-1  #For Unbiased estimmator.


cols = ['CORRELATION','OSTD','MSTD','CRMSE','BIAS']
    
tskill = np.empty((3,5))
tskill[:] = np.nan

#CORR
tskill[0,0] = DF.corr()['OHC_GOFS']['OHC_GOFS']
tskill[1,0] = DF.corr()['OHC_GOFS']['OHC_POM_oper']
tskill[2,0] = DF.corr()['OHC_GOFS']['OHC_POM_oper']

#OSTD
tskill[0,1] = DF.std().OHC_GOFS
tskill[1,1] = DF.std().OHC_GOFS
tskill[2,1] = DF.std().OHC_GOFS

#MSTD
tskill[0,2] = DF.std().OHC_GOFS
tskill[1,2] = DF.std().OHC_POM_oper
tskill[2,2] = DF.std().OHC_POM_exp

#CRMSE
tskill[0,3] = 0
tskill[1,3] = np.sqrt(np.nansum(((DF.OHC_GOFS-DF.mean().OHC_GOFS)-\
                                 (DF.OHC_POM_oper-DF.mean().OHC_POM_oper))**2)/N)
tskill[2,3] = np.sqrt(np.nansum(((DF.OHC_GOFS-DF.mean().OHC_GOFS)-\
                                 (DF.OHC_POM_exp-DF.mean().OHC_POM_exp))**2)/N)

#BIAS
tskill[0,4] = 0
tskill[1,4] = DF.mean().OHC_GOFS - DF.mean().OHC_POM_oper
tskill[2,4] = DF.mean().OHC_GOFS - DF.mean().OHC_POM_exp

#color
colors = ['indianred','seagreen','darkorchid']
    
OHC_skillscores = pd.DataFrame(tskill,
                        index=['GOFS_to_POM_oper','POM_oper','POM_exp'],
                        columns=cols)
print(OHC_skillscores) 

#%% T100 statistics.

DF = DF_T100_GOFS_POM.dropna()

N = len(DF)-1  #For Unbiased estimmator.


cols = ['CORRELATION','OSTD','MSTD','CRMSE','BIAS']
    
tskill = np.empty((3,5))
tskill[:] = np.nan

#CORR
tskill[0,0] = DF.corr()['T100_GOFS']['T100_GOFS']
tskill[1,0] = DF.corr()['T100_GOFS']['T100_POM_oper']
tskill[2,0] = DF.corr()['T100_GOFS']['T100_POM_oper']

#OSTD
tskill[0,1] = DF.std().T100_GOFS
tskill[1,1] = DF.std().T100_GOFS
tskill[2,1] = DF.std().T100_GOFS

#MSTD
tskill[0,2] = DF.std().T100_GOFS
tskill[1,2] = DF.std().T100_POM_oper
tskill[2,2] = DF.std().T100_POM_exp

#CRMSE
tskill[0,3] = 0
tskill[1,3] = np.sqrt(np.nansum(((DF.T100_GOFS-DF.mean().T100_GOFS)-\
                                 (DF.T100_POM_oper-DF.mean().T100_POM_oper))**2)/N)
tskill[2,3] = np.sqrt(np.nansum(((DF.T100_GOFS-DF.mean().T100_GOFS)-\
                                 (DF.T100_POM_exp-DF.mean().T100_POM_exp))**2)/N)

#BIAS
tskill[0,4] = 0
tskill[1,4] = DF.mean().T100_GOFS - DF.mean().T100_POM_oper
tskill[2,4] = DF.mean().T100_GOFS - DF.mean().T100_POM_exp

#color
colors = ['indianred','seagreen','darkorchid']
    
T100_skillscores = pd.DataFrame(tskill,
                        index=['GOFS_to_POM_oper','POM_oper','POM_exp'],
                        columns=cols)
print(T100_skillscores) 

#%% Combine all metrics into one normalized Taylor diagram 

angle_lim = np.pi/2 #+ np.pi/16
std_lim = 2.0
fig,ax1 = taylor_template(angle_lim,std_lim)
markers = ['.','X','^']
  
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

scores = T100_skillscores  
for i,r in enumerate(scores.iterrows()):
    theta=np.arccos(r[1].CORRELATION)            
    rr=r[1].MSTD/r[1].OSTD
    ax1.plot(theta,rr,markers[i],color = 'royalblue',markersize=8) 
ax1.plot(theta,rr,markers[i],label='T100',color = 'royalblue',markersize=8) 
        
#ax1.plot(0,1,'o',label='Obs',markersize=8) 
ax1.plot(0,1,'or',label='GOFS',markersize=8,zorder=10)
ax1.plot(0,0,'Xk',label='POM Oper',markersize=8)
ax1.plot(0,0,'^k',label='POM Exp',markersize=8)
     
plt.legend(loc='upper right',bbox_to_anchor=[1.45,1.2])    

rs,ts = np.meshgrid(np.linspace(0,std_lim),np.linspace(0,angle_lim))
rms = np.sqrt(1 + rs**2 - 2*rs*np.cos(ts))
    
contours = ax1.contour(ts, rs, rms,3,colors='0.5')
plt.clabel(contours, inline=1, fontsize=10)
plt.grid(linestyle=':',alpha=0.5)

file = folder + 'Taylor_norm_'+cycle+'_IC'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Figure forecasted track operational POM

#Time window
date_ini = cycle[0:4]+'-'+cycle[4:6]+'-'+cycle[6:8]+' '+cycle[8:]+':00:00'
tini = datetime.strptime(date_ini,'%Y-%m-%d %H:%M:%S')
tend = tini + timedelta(hours=float(lead_time_oper[-1]))
date_end = str(tend)

okt = np.logical_and(time_best_track >= tini,time_best_track <= tend)

# time forecasted track_exp
time_forec_track_oper = np.asarray([tini + timedelta(hours = float(t)) for t in lead_time_oper])
oktt = [np.where(t == time_forec_track_oper)[0][0] for t in time_best_track[okt]]
    
plt.figure()
plt.plot(lon_forec_track_oper[oktt], lat_forec_track_oper[oktt],'X-',color='mediumorchid',label='POM Oper')
plt.plot(lon_forec_track_exp[oktt], lat_forec_track_exp[oktt],'^-',color='teal',label='POM Exp')
plt.plot(lon_best_track[okt], lat_best_track[okt],'o-',color='k',label='Best Track')   
plt.legend()
plt.title('Track Forecast Dorian '+ cycle,fontsize=18)

file = folder + 'best_track_vs_forec_track_POM_2019082800'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Figure forecasted track band operational POM

okt = np.logical_and(time_best_track >= tini,time_best_track <= tend)

# time forecasted track_exp
time_forec_track_oper = np.asarray([tini + timedelta(hours = float(t)) for t in lead_time_oper])
oktt = [np.where(t == time_forec_track_oper)[0][0] for t in time_best_track[okt]]
    
plt.figure()
plt.plot(lon_band, lat_band,'o',color='lightblue',label='band')
plt.plot(lon_forec_track_oper[oktt], lat_forec_track_oper[oktt],'X-',color='slateblue',label='POM Oper')
#plt.plot(lon_best_track[okt], lat_best_track[okt],'o-',color='red',label='Best Track')   
plt.legend()

file = folder + 'forec_track_band_POM_2019082800'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

