"""
Created on Monday Jan 18 2021

@author: aristizabal
"""

#%% User input

# Limits Caribbean
lon_lim = [-87,-60]
lat_lim = [10,23]

date_ini = '2020/06/19/06'
date_end = '2020/10/25/06'

# RTOFS files
folder_RTOFS = '/home/coolgroup/RTOFS/forecasts/domains/hurricanes/RTOFS_6hourly_North_Atlantic/'

# RTOFS-DA files
folder_RTOFS_DA = '/home/aristizabal/RTOFS-DA/RTOFS_DA_Exp/'

ncfiles_RTOFS = ['rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f012_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f018_6hrly_hvr_US_east.nc',\
                  'rtofs_glo_3dz_f024_6hrly_hvr_US_east.nc']

# Server location
url_erddap = 'https://data.ioos.us/gliders/erddap'

# url for GOFS 3.1
url_GOFS = 'http://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/ts3z'

bath_file = '/home/aristizabal/bathymetry_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

folder_fig = '/www/web/rucool/aristizabal/Figures/'

#%%
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
from datetime import datetime, timedelta
import cmocean
import matplotlib.dates as mdates
import glob
import os
import seawater as sw
from erddapy import ERDDAP

import sys
sys.path.append('/home/aristizabal/glider_model_comparisons_Python')
from read_glider_data import retrieve_dataset_id_erddap_server
from read_glider_data import read_glider_data_erddap_server
from process_glider_data import grid_glider_data

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

def taylor_template(angle_lim,std_lim):

    import mpl_toolkits.axisartist.floating_axes as floating_axes
    from matplotlib.projections import PolarAxes
    from mpl_toolkits.axisartist.grid_finder import (FixedLocator,
                                                 DictFormatter)

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

    import mpl_toolkits.axisartist.floating_axes as floating_axes
    from matplotlib.projections import PolarAxes
    from mpl_toolkits.axisartist.grid_finder import (FixedLocator,
                                                 DictFormatter)

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

def get_glider_transect_from_RTOFS(folder_RTOFS,ncfiles_RTOFS,date_ini,date_end,long,latg,tstamp_glider):

    import os

    #Time window
    year_ini = int(date_ini.split('/')[0])
    month_ini = int(date_ini.split('/')[1])
    day_ini = int(date_ini.split('/')[2])

    year_end = int(date_end.split('/')[0])
    month_end = int(date_end.split('/')[1])
    day_end = int(date_end.split('/')[2])

    tini = datetime(year_ini, month_ini, day_ini)
    tend = datetime(year_end, month_end, day_end)
    tvec = [tini + timedelta(int(i)) for i in np.arange((tend-tini).days)]
    #tvec = [tini + timedelta(int(i)) for i in np.arange((tend-tini).days+1)]

    # Read RTOFS grid and time
    print('Retrieving coordinates from RTOFS')

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

    ncRTOFS = xr.open_dataset(folder_RTOFS + fol + '/' + ncfiles_RTOFS[0])
    lat_RTOFS = np.asarray(ncRTOFS.Latitude[:])
    lon_RTOFS = np.asarray(ncRTOFS.Longitude[:])
    depth_RTOFS = np.asarray(ncRTOFS.Depth[:])

    tRTOFS = []
    nc_allfiles_RTOFS = []
    for tt in tvec:
        # Read RTOFS grid and time
        if tt.month < 10:
            if tt.day < 10:
                fol = 'rtofs.' + str(tt.year) + '0' + str(tt.month) + '0' + str(tt.day)
            else:
                fol = 'rtofs.' + str(tt.year) + '0' + str(tt.month) + str(tt.day)
        else:
            if tt.day < 10:
                fol = 'rtofs.' + str(tt.year) + str(tt.month) + '0' + str(tt.day)
            else:
                fol = 'rtofs.' + str(tt.year) + str(tt.month) + str(tt.day)

        if len(os.listdir(folder_RTOFS + fol + '/')) == 4:

            for t in np.arange(len(ncfiles_RTOFS)):
                ncRTOFS = xr.open_dataset(folder_RTOFS + fol + '/' + ncfiles_RTOFS[t])
                nc_allfiles_RTOFS.append(folder_RTOFS + fol + '/' + ncfiles_RTOFS[t])
                tRTOFS.append(np.asarray(ncRTOFS.MT[:])[0])

    # Getting glider transect from RTOFS
    if len(tRTOFS) == 0:
        temp_RTOFS = np.empty((len(depth_RTOFS),1))
        temp_RTOFS[:] = np.nan
        salt_RTOFS = np.empty((len(depth_RTOFS),1))
        salt_RTOFS[:] = np.nan
    else:

        tstamp_RTOFS = [mdates.date2num(tRTOFS[i]) for i in np.arange(len(tRTOFS))]
        sublonRTOFS = np.interp(tstamp_RTOFS,tstamp_glider,long)
        sublatRTOFS = np.interp(tstamp_RTOFS,tstamp_glider,latg)

        # getting the model grid positions for sublonm and sublatm
        oklonRTOFS = np.round(np.interp(sublonRTOFS,lon_RTOFS[0,:],np.arange(len(lon_RTOFS[0,:])))).astype(int)
        oklatRTOFS = np.round(np.interp(sublatRTOFS,lat_RTOFS[:,0],np.arange(len(lat_RTOFS[:,0])))).astype(int)

        temp_RTOFS = np.empty((len(depth_RTOFS),len(tRTOFS)))
        temp_RTOFS[:] = np.nan
        salt_RTOFS = np.empty((len(depth_RTOFS),len(tRTOFS)))
        salt_RTOFS[:] = np.nan
        for i in range(len(tRTOFS)):
            print(len(tRTOFS),' ',i)
            nc_file = nc_allfiles_RTOFS[i]
            ncRTOFS = xr.open_dataset(nc_file)
            temp_RTOFS[:,i] = ncRTOFS.variables['temperature'][0,:,oklatRTOFS[i],oklonRTOFS[i]]
            salt_RTOFS[:,i] = ncRTOFS.variables['salinity'][0,:,oklatRTOFS[i],oklonRTOFS[i]]

    return temp_RTOFS, salt_RTOFS, tRTOFS, depth_RTOFS

#%%
def interp_datasets_in_z(temp_orig,salt_orig,depth_orig,depth_target,indexes_time,dim_vars):

    temp_interp = np.empty((dim_vars[0],dim_vars[1]))
    temp_interp[:] = np.nan
    salt_interp = np.empty((dim_vars[0],dim_vars[1]))
    salt_interp[:] = np.nan
    for i in np.arange(len(indexes_time)):
        pos = np.argsort(depth_orig[:,indexes_time[i]])
        if depth_target.ndim == 1:
            temp_interp[:,i] = np.interp(depth_target,depth_orig[pos,indexes_time[i]],temp_orig[pos,indexes_time[i]],right=np.nan)
            salt_interp[:,i] = np.interp(depth_target,depth_orig[pos,indexes_time[i]],salt_orig[pos,indexes_time[i]],right=np.nan)
        if depth_target.ndim == 2:
            temp_interp[:,i] = np.interp(depth_target[:,i],depth_orig[pos,indexes_time[i]],temp_orig[pos,indexes_time[i]],right=np.nan)
            salt_interp[:,i] = np.interp(depth_target[:,i],depth_orig[pos,indexes_time[i]],salt_orig[pos,indexes_time[i]],right=np.nan)

    return temp_interp, salt_interp

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

def get_glider_transect_from_GOFS(GOFS,depth_GOFS,oktime_GOFS,oklat_GOFS,oklon_GOFS):

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

def check_dataset_empty(url_erddap,dataset_id,date_ini,date_end,lon_lim,lat_lim):

    from erddapy import ERDDAP

    constraints = {
        'time>=': date_ini,
        'time<=': date_end,
        'latitude>=': lat_lim[0],
        'latitude<=': lat_lim[1],
        'longitude>=': lon_lim[0],
        'longitude<=': lon_lim[1],
        }

    variable_names = [
            'depth',
            'latitude',
            'longitude',
            'time',
            'temperature',
            'salinity'
            ]

    e = ERDDAP(
            server=url_erddap,
            protocol='tabledap',
            response='nc'
            )

    e.dataset_id = dataset_id
    e.constraints = constraints
    e.variables = variable_names

    # Converting glider data to data frame
    # Cheching that data frame has data
    df = e.to_pandas()
    if len(df) < 4:
        empty_dataset = True
    else:
        empty_dataset = False

    return empty_dataset

def MLD_temp_and_dens_criteria(dt,drho,time,depth,temp,salt,dens):

    MLD_temp_crit = np.empty(len(time))
    MLD_temp_crit[:] = np.nan
    Tmean_temp_crit = np.empty(len(time))
    Tmean_temp_crit[:] = np.nan
    Smean_temp_crit = np.empty(len(time))
    Smean_temp_crit[:] = np.nan
    #Td_temp_crit = np.empty(len(time))
    #Td_temp_crit[:] = np.nan
    MLD_dens_crit = np.empty(len(time))
    MLD_dens_crit[:] = np.nan
    Tmean_dens_crit = np.empty(len(time))
    Tmean_dens_crit[:] = np.nan
    Smean_dens_crit = np.empty(len(time))
    Smean_dens_crit[:] = np.nan
    #Td_dens_crit = np.empty(len(time))
    #Td_dens_crit[:] = np.nan
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
            #Td_temp_crit[t] = np.nan
            Tmean_temp_crit[t] = np.nan
            Smean_temp_crit[t] = np.nan
        else:
            if depth.ndim == 1:
                MLD_temp_crit[t] = depth[ok_mld_temp[-1]]
                #ok_mld_plus1m = np.where(depth >= depth[ok_mld_temp[-1]] + 1)[0][0]
            if depth.ndim == 2:
                MLD_temp_crit[t] = depth[ok_mld_temp[-1],t]
                #ok_mld_plus1m = np.where(depth >= depth[ok_mld_temp[-1],t] + 1)[0][0]
            #Td_temp_crit[t] = temp[ok_mld_plus1m,t]
            Tmean_temp_crit[t] = np.nanmean(temp[ok_mld_temp,t])
            Smean_temp_crit[t] = np.nanmean(salt[ok_mld_temp,t])

        if ok_mld_rho.size == 0:
            MLD_dens_crit[t] = np.nan
            #Td_dens_crit[t] = np.nan
            Tmean_dens_crit[t] = np.nan
            Smean_dens_crit[t] = np.nan
        else:
            if depth.ndim == 1:
                MLD_dens_crit[t] = depth[ok_mld_rho[-1]]
                #ok_mld_plus1m = np.where(depth >= depth[ok_mld_rho[-1]] + 1)[0][0]
            if depth.ndim == 2:
                MLD_dens_crit[t] = depth[ok_mld_rho[-1],t]
                #ok_mld_plus1m = np.where(depth >= depth[ok_mld_rho[-1],t] + 1)[0][0]
            #Td_dens_crit[t] = temp[ok_mld_plus1m,t]
            Tmean_dens_crit[t] = np.nanmean(temp[ok_mld_rho,t])
            Smean_dens_crit[t] = np.nanmean(salt[ok_mld_rho,t])

    return MLD_temp_crit,Tmean_temp_crit,Smean_temp_crit,\
           MLD_dens_crit,Tmean_dens_crit,Smean_dens_crit

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

#%% Time window
tini = datetime.strptime(date_ini,"%Y/%m/%d/%H")
tend = datetime.strptime(date_end,"%Y/%m/%d/%H")

#%% Reading glider data
gliders = retrieve_dataset_id_erddap_server(url_erddap,lat_lim,lon_lim,date_ini,date_end)
gliders = [dataset_id for dataset_id in gliders if dataset_id != 'silbo-20190717T1917']
gliders = [dataset_id for dataset_id in gliders if dataset_id.split('-')[-1] != 'delayed']
print(gliders)

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

#%% Make map with glider tracks
constraints = {
    'time>=': date_ini,
    'time<=': date_end,
    'latitude>=': lat_lim[0],
    'latitude<=': lat_lim[-1],
    'longitude>=': lon_lim[0],
    'longitude<=': lon_lim[-1],
}

variables = [
 'time','latitude','longitude'
]

e = ERDDAP(
    server=url_erddap,
    protocol='tabledap',
    response='nc'
)

lev = np.arange(-9000,9100,100)
fig, ax = plt.subplots(figsize=(10, 10))
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,lev,cmap=cmocean.cm.topo)
plt.axis('scaled')

for id in gliders:
    e.dataset_id = id
    e.constraints = constraints
    e.variables = variables

    df = e.to_pandas(
    parse_dates=True)

    print(id,df.index[-1])
    ax.plot(df['longitude (degrees_east)'],\
                df['latitude (degrees_north)'],'.',color='orange',markersize=1)

markers = ['o','v','^','<','>','8','s','p','P','*','h','H','x','X','D','d']
colors = ['green','indianred','purple','c','cadetblue','y','r','b',]

for id in gliders:
    e.dataset_id = id
    e.constraints = constraints
    e.variables = variables

    df = e.to_pandas(
    parse_dates=True)

    if id.split('-')[0][0:2] == 'ng':
        marker = markers[0]
        color = colors[0]
    if id.split('-')[0][0:2] == 'SG':
        marker = markers[1]
        color = colors[1]
    if id.split('-')[0][0:2] == 'ru':
        marker = markers[2]
        color = colors[2]
    ax.plot(np.nanmean(df['longitude (degrees_east)']),\
                np.nanmean(df['latitude (degrees_north)']),marker,markersize=10,color=color,label=id.split('-')[0])

plt.legend(loc='lower left')

file = folder_fig + 'Map_Caribbean_20200619_20201025'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

# Get coordinates from GOFS
print('Retrieving coordinates from GOFS')
GOFS = xr.open_dataset(url_GOFS,decode_times=False)
tt_G = GOFS.time
t_G = netCDF4.num2date(tt_G[:],tt_G.units)
tmin = datetime.strptime(date_ini,'%Y/%m/%d/%H')
tmax = datetime.strptime(date_end,'%Y/%m/%d/%H')
oktime_GOFS = np.where(np.logical_and(t_G >= tmin, t_G <= tmax))
time_GOFS = np.asarray(t_G[oktime_GOFS])
ttGOFS = np.asarray([datetime(time_GOFS[i].year,time_GOFS[i].month,time_GOFS[i].day,\
                    time_GOFS[i].hour) for i in np.arange(len(time_GOFS))])
tstamp_GOFS = mdates.date2num(ttGOFS)

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

#%% Read RTOFS grid
print('Retrieving coordinates from RTOFS')

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

ncRTOFS = xr.open_dataset(folder_RTOFS + fol + '/' + ncfiles_RTOFS[0])
latRTOFS = np.asarray(ncRTOFS.Latitude[:])
lonRTOFS = np.asarray(ncRTOFS.Longitude[:])
depth_RTOFS = np.asarray(ncRTOFS.Depth[:])

#%% Read RTOFS-DA grid
tini = datetime.strptime(date_ini,"%Y/%m/%d/%H")
print('Retrieving coordinates from RTOFS-DA')

ncRTOFS_DA = xr.open_dataset(folder_RTOFS_DA + fol + '/' + ncfiles_RTOFS[0])
latRTOFS_DA = np.asarray(ncRTOFS_DA.Latitude[:])
lonRTOFS_DA = np.asarray(ncRTOFS_DA.Longitude[:])
depth_RTOFS_DA = np.asarray(ncRTOFS_DA.Depth[:])

#%%
DF_RTOFS_temp_salt = pd.DataFrame()
DF_RTOFS_DA_temp_salt = pd.DataFrame()
DF_GOFS_temp_salt = pd.DataFrame()
DF_RTOFS_MLD = pd.DataFrame()
DF_RTOFS_DA_MLD = pd.DataFrame()
DF_GOFS_MLD = pd.DataFrame()
DF_RTOFS_OHC = pd.DataFrame()
DF_RTOFS_DA_OHC = pd.DataFrame()
DF_GOFS_OHC = pd.DataFrame()
DF_RTOFS_T100 = pd.DataFrame()
DF_RTOFS_DA_T100 = pd.DataFrame()
DF_GOFS_T100 = pd.DataFrame()

for f,dataset_id in enumerate(gliders):

    empty_dataset = check_dataset_empty(url_erddap,dataset_id,date_ini,date_end,lon_lim,lat_lim)

    if not empty_dataset:
        print(dataset_id)

        kwargs = dict(date_ini=date_ini,date_end=date_end)
        scatter_plot = 'no'
        tempg, saltg, timeg, latg, long, depthg = read_glider_data_erddap_server(url_erddap,dataset_id,\
                                           lat_lim,lon_lim,scatter_plot,**kwargs)

        # Get glider density
        densg = sw.dens(saltg,tempg,depthg)

        # Grid glider variables according to depth
        delta_z = 0.5
        tempg_gridded, timegg, depthg_gridded = \
        grid_glider_data('temp',dataset_id,tempg,timeg,depthg,delta_z,contour_plot='no')

        saltg_gridded, _, _ = \
        grid_glider_data('temp',dataset_id,saltg,timeg,depthg,delta_z,contour_plot='no')

        densg_gridded, _, _ = \
        grid_glider_data('temp',dataset_id,densg,timeg,depthg,delta_z,contour_plot='no')

        # Changing times to timestamp
        tstamp_glider = mdates.date2num(timeg)

        #%% Get glider transect from GOFS 3.1
        # Conversion from glider longitude and latitude to GOFS convention
        target_lon, target_lat = glider_coor_to_GOFS_coord(long,latg)

        # interpolating glider lon and lat to lat and lon on model time
        sublon_GOFS = np.interp(tstamp_GOFS,tstamp_glider,target_lon)
        sublat_GOFS = np.interp(tstamp_GOFS,tstamp_glider,target_lat)

        # Conversion from GOFS convention to glider longitude and latitude
        sublon_GOFSg,sublat_GOFSg = GOFS_coor_to_glider_coord(sublon_GOFS,sublat_GOFS)

        # getting the model grid positions for sublonm and sublatm
        oklon_GOFS = np.round(np.interp(sublon_GOFS,lon_G,np.arange(len(lon_G)))).astype(int)
        oklat_GOFS = np.round(np.interp(sublat_GOFS,lat_G,np.arange(len(lat_G)))).astype(int)

        # Getting glider transect from model
        print('Getting glider transect from GOFS')
        target_temp_GOFS, target_salt_GOFS = \
                                  get_glider_transect_from_GOFS(GOFS,depth_GOFS,oktime_GOFS,oklat_GOFS,oklon_GOFS)
        # Get GOFS density
        target_dens_GOFS = sw.dens(target_salt_GOFS,target_temp_GOFS,np.tile(depth_GOFS,(len(ttGOFS),1)).T)


        #%% Get glider transect from RTOFS
        print('Getting glider transect from RTOFS')
        target_temp_RTOFS, target_salt_RTOFS, tRTOFS, depth_RTOFS = \
        get_glider_transect_from_RTOFS(folder_RTOFS,ncfiles_RTOFS,date_ini,date_end,long,latg,tstamp_glider)

        # Get RTOFS density
        target_dens_RTOFS = sw.dens(target_salt_RTOFS,target_temp_RTOFS,np.tile(depth_RTOFS,(len(tRTOFS),1)).T)

        # Get glider transect from RTOFS-DA
        print('Getting glider transect from RTOFS-DA')
        target_temp_RTOFS_DA, target_salt_RTOFS_DA, tRTOFS_DA, depth_RTOFS_DA = \
        get_glider_transect_from_RTOFS(folder_RTOFS_DA,ncfiles_RTOFS,date_ini,date_end,long,latg,tstamp_glider)

        # Get RTOFS-DA density
        target_dens_RTOFS_DA = sw.dens(target_salt_RTOFS_DA,target_temp_RTOFS_DA,np.tile(depth_RTOFS_DA,(len(tRTOFS_DA),1)).T)

        tstamp_RTOFS = mdates.date2num(tRTOFS)
        tstamp_RTOFS_DA = mdates.date2num(tRTOFS_DA)

        #%% Calculation of mixed layer depth based on density critria
        dt = 0.2
        drho = 0.125

        # for glider data
        _, _, _, MLD_dens_crit_glid, Tmean_dens_crit_glid, Smean_dens_crit_glid = \
        MLD_temp_and_dens_criteria(dt,drho,timeg,depthg_gridded,tempg_gridded,saltg_gridded,densg_gridded)

        # for RTOFS
        _, _, _, MLD_dens_crit_RTOFS, Tmean_dens_crit_RTOFS, Smean_dens_crit_RTOFS = \
        MLD_temp_and_dens_criteria(dt,drho,tstamp_RTOFS,depth_RTOFS,target_temp_RTOFS,target_salt_RTOFS,target_dens_RTOFS)

        # for RTOFS-DA
        _, _, _, MLD_dens_crit_RTOFS_DA, Tmean_dens_crit_RTOFS_DA, Smean_dens_crit_RTOFS_DA = \
        MLD_temp_and_dens_criteria(dt,drho,tstamp_RTOFS_DA,depth_RTOFS_DA,target_temp_RTOFS_DA,target_salt_RTOFS_DA,target_dens_RTOFS_DA)

        # for GOFS 3.1 output
        _, _, _, MLD_dens_crit_GOFS, Tmean_dens_crit_GOFS, Smean_dens_crit_GOFS = \
        MLD_temp_and_dens_criteria(dt,drho,time_GOFS,depth_GOFS,target_temp_GOFS,target_salt_GOFS,target_dens_GOFS)

        #%% Surface Ocean Heat Content
        # glider
        OHC_glid = OHC_surface(timeg,tempg_gridded,depthg_gridded,densg_gridded)

        # RTOFS
        OHC_RTOFS = OHC_surface(tstamp_RTOFS,target_temp_RTOFS,depth_RTOFS,target_dens_RTOFS)

        # RTOFS_DA
        OHC_RTOFS_DA = OHC_surface(tstamp_RTOFS_DA,target_temp_RTOFS_DA,depth_RTOFS_DA,target_dens_RTOFS_DA)

        # GOFS
        OHC_GOFS = OHC_surface(time_GOFS,target_temp_GOFS,depth_GOFS,target_dens_GOFS)

        #%% Calculate T100
        # glider
        T100_glid = depth_aver_top_100(depthg_gridded,tempg_gridded)

        # RTOFS
        T100_RTOFS = depth_aver_top_100(depth_RTOFS,target_temp_RTOFS)

        # RTOFS-DA
        T100_RTOFS_DA = depth_aver_top_100(depth_RTOFS_DA,target_temp_RTOFS_DA)

        # GOFS
        T100_GOFS = depth_aver_top_100(depth_GOFS,target_temp_GOFS)

        #%% Interpolate glider transect onto RTOFS time and depth
        oktimeg_rtofs = np.round(np.interp(tstamp_RTOFS,tstamp_glider,np.arange(len(tstamp_glider)))).astype(int)
        temp_orig = tempg
        salt_orig = saltg
        depth_orig = depthg
        depth_target = depth_RTOFS
        indexes_time = oktimeg_rtofs
        dim_vars = [target_temp_RTOFS.shape[0],target_temp_RTOFS.shape[1]]

        tempg_to_RTOFS, saltg_to_RTOFS = \
        interp_datasets_in_z(temp_orig,salt_orig,depth_orig,depth_target,indexes_time,dim_vars)

        MLD_dens_crit_glid_to_RTOFS = MLD_dens_crit_glid[oktimeg_rtofs]
        Tmean_dens_crit_glid_to_RTOFS = Tmean_dens_crit_glid[oktimeg_rtofs]
        Smean_dens_crit_glid_to_RTOFS = Smean_dens_crit_glid[oktimeg_rtofs]
        OHC_glid_to_RTOFS = OHC_glid[oktimeg_rtofs]
        T100_glid_to_RTOFS = T100_glid[oktimeg_rtofs]

        #%% Interpolate glider transect onto RTOFS-DA time and depth
        oktimeg_rtofs_da = np.round(np.interp(tstamp_RTOFS_DA,tstamp_glider,np.arange(len(tstamp_glider)))).astype(int)
        temp_orig = tempg
        salt_orig = saltg
        depth_orig = depthg
        depth_target = depth_RTOFS_DA
        indexes_time = oktimeg_rtofs_da
        dim_vars = [target_temp_RTOFS_DA.shape[0],target_temp_RTOFS_DA.shape[1]]

        tempg_to_RTOFS_DA, saltg_to_RTOFS_DA = \
        interp_datasets_in_z(temp_orig,salt_orig,depth_orig,depth_target,indexes_time,dim_vars)

        MLD_dens_crit_glid_to_RTOFS_DA = MLD_dens_crit_glid[oktimeg_rtofs_da]
        Tmean_dens_crit_glid_to_RTOFS_DA = Tmean_dens_crit_glid[oktimeg_rtofs_da]
        Smean_dens_crit_glid_to_RTOFS_DA = Smean_dens_crit_glid[oktimeg_rtofs_da]
        OHC_glid_to_RTOFS_DA = OHC_glid[oktimeg_rtofs_da]
        T100_glid_to_RTOFS_DA = T100_glid[oktimeg_rtofs_da]

        #%% Interpolate glider transect onto GOFS time and depth
        oktimeg_gofs = np.round(np.interp(tstamp_GOFS,tstamp_glider,np.arange(len(tstamp_glider)))).astype(int)
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

        #%% Define dataframe
        df_RTOFS_temp_salt = pd.DataFrame(data=np.array([np.ravel(tempg_to_RTOFS,order='F'),\
                                              np.ravel(target_temp_RTOFS,order='F'),\
                                              np.ravel(saltg_to_RTOFS,order='F'),\
                                              np.ravel(target_salt_RTOFS,order='F'),\
                                              ]).T,\
                          columns=['temp_obs','temp_RTOFS',\
                                   'salt_obs','salt_RTOFS'])

        #%% Define dataframe
        df_RTOFS_DA_temp_salt = pd.DataFrame(data=np.array([np.ravel(tempg_to_RTOFS_DA,order='F'),\
                                              np.ravel(target_temp_RTOFS_DA,order='F'),\
                                              np.ravel(saltg_to_RTOFS_DA,order='F'),\
                                              np.ravel(target_salt_RTOFS_DA,order='F'),\
                                              ]).T,\
                          columns=['temp_obs','temp_RTOFS_DA',\
                                   'salt_obs','salt_RTOFS_DA'])

        #%% Define dataframe
        df_GOFS_temp_salt = pd.DataFrame(data=np.array([np.ravel(tempg_to_GOFS,order='F'),\
                                              np.ravel(target_temp_GOFS,order='F'),\
                                              np.ravel(saltg_to_GOFS,order='F'),\
                                              np.ravel(target_salt_GOFS,order='F'),\
                                              ]).T,\
                          columns=['temp_obs','temp_GOFS',\
                                   'salt_obs','salt_GOFS'])

        #%% Define dataframe
        df_RTOFS_MLD = pd.DataFrame(data=np.array([MLD_dens_crit_glid_to_RTOFS,MLD_dens_crit_RTOFS,\
                                                  Tmean_dens_crit_glid_to_RTOFS,Tmean_dens_crit_RTOFS,
                                                  Smean_dens_crit_glid_to_RTOFS,Smean_dens_crit_RTOFS]).T,\
                          columns=['MLD_obs','MLD_RTOFS',\
                                  'Tmean_obs','Tmean_RTOFS',
                                  'Smean_obs','Smean_RTOFS'])

        #%% Define dataframe
        df_RTOFS_DA_MLD = pd.DataFrame(data=np.array([MLD_dens_crit_glid_to_RTOFS_DA,MLD_dens_crit_RTOFS_DA,\
                                                  Tmean_dens_crit_glid_to_RTOFS_DA,Tmean_dens_crit_RTOFS_DA,
                                                  Smean_dens_crit_glid_to_RTOFS_DA,Smean_dens_crit_RTOFS_DA]).T,\
                          columns=['MLD_obs','MLD_RTOFS_DA',\
                                  'Tmean_obs','Tmean_RTOFS_DA',
                                  'Smean_obs','Smean_RTOFS_DA'])

        #%% Define dataframe
        df_GOFS_MLD = pd.DataFrame(data=np.array([MLD_dens_crit_glid_to_GOFS,MLD_dens_crit_GOFS,\
                                                  Tmean_dens_crit_glid_to_GOFS,Tmean_dens_crit_GOFS,
                                                  Smean_dens_crit_glid_to_GOFS,Smean_dens_crit_GOFS]).T,\
                          columns=['MLD_obs','MLD_GOFS',\
                                  'Tmean_obs','Tmean_GOFS',
                                  'Smean_obs','Smean_GOFS'])

        #%% Define dataframe
        df_RTOFS_OHC = pd.DataFrame(data=np.array([OHC_glid_to_RTOFS,OHC_RTOFS]).T,\
                          columns=['OHC_obs','OHC_RTOFS'])

        #%% Define dataframe
        df_RTOFS_DA_OHC = pd.DataFrame(data=np.array([OHC_glid_to_RTOFS_DA,OHC_RTOFS_DA]).T,\
                          columns=['OHC_obs','OHC_RTOFS_DA'])

        #%% Define dataframe
        df_GOFS_OHC = pd.DataFrame(data=np.array([OHC_glid_to_GOFS,OHC_GOFS]).T,\
                          columns=['OHC_obs','OHC_GOFS'])

        #%% Define dataframe
        df_RTOFS_T100 = pd.DataFrame(data=np.array([T100_glid_to_RTOFS,T100_RTOFS]).T,\
                          columns=['T100_obs','T100_RTOFS'])

        #%% Define dataframe
        df_RTOFS_DA_T100 = pd.DataFrame(data=np.array([T100_glid_to_RTOFS_DA,T100_RTOFS_DA]).T,\
                          columns=['T100_obs','T100_RTOFS_DA'])

        #%% Define dataframe
        df_GOFS_T100 = pd.DataFrame(data=np.array([T100_glid_to_GOFS,T100_GOFS]).T,\
                          columns=['T100_obs','T100_GOFS'])

        #%% Concatenate data frames
        DF_RTOFS_temp_salt = pd.concat([DF_RTOFS_temp_salt, df_RTOFS_temp_salt])
        DF_RTOFS_DA_temp_salt = pd.concat([DF_RTOFS_DA_temp_salt, df_RTOFS_DA_temp_salt])
        DF_GOFS_temp_salt = pd.concat([DF_GOFS_temp_salt, df_GOFS_temp_salt])
        DF_RTOFS_MLD = pd.concat([DF_RTOFS_MLD, df_RTOFS_MLD])
        DF_RTOFS_DA_MLD = pd.concat([DF_RTOFS_DA_MLD, df_RTOFS_DA_MLD])
        DF_GOFS_MLD = pd.concat([DF_GOFS_MLD, df_GOFS_MLD])
        DF_RTOFS_OHC = pd.concat([DF_RTOFS_OHC, df_RTOFS_OHC])
        DF_RTOFS_DA_OHC = pd.concat([DF_RTOFS_DA_OHC, df_RTOFS_DA_OHC])
        DF_GOFS_OHC = pd.concat([DF_GOFS_OHC, df_GOFS_OHC])
        DF_RTOFS_T100 = pd.concat([DF_RTOFS_T100, df_RTOFS_T100])
        DF_RTOFS_DA_T100 = pd.concat([DF_RTOFS_DA_T100, df_RTOFS_DA_T100])
        DF_GOFS_T100 = pd.concat([DF_GOFS_T100, df_GOFS_T100])

#%% Save all data frames
DF_RTOFS_temp_salt.to_pickle('DF_RTOFS_temp_salt_Caribbean.pkl')
DF_RTOFS_MLD.to_pickle('DF_RTOFS_MLD_Caribbean.pkl')
DF_RTOFS_OHC.to_pickle('DF_RTOFS_OHC_Caribbean.pkl')
DF_RTOFS_T100.to_pickle('DF_RTOFS_T100_Caribbean.pkl')

DF_RTOFS_DA_temp_salt.to_pickle('DF_RTOFS_DA_temp_salt_Caribbean.pkl')
DF_RTOFS_DA_MLD.to_pickle('DF_RTOFS_DA_MLD_Caribbean.pkl')
DF_RTOFS_DA_OHC.to_pickle('DF_RTOFS_DA_OHC_Caribbean.pkl')
DF_RTOFS_DA_T100.to_pickle('DF_RTOFS_DA_T100_Caribbean.pkl')

DF_GOFS_temp_salt.to_pickle('DF_GOFS_temp_salt_Caribbean.pkl')
DF_GOFS_MLD.to_pickle('DF_GOFS_MLD_Caribbean.pkl')
DF_GOFS_OHC.to_pickle('DF_GOFS_OHC_Caribbean.pkl')
DF_GOFS_T100.to_pickle('DF_GOFS_T100_Caribbean.pkl')

#%% Load all data frames
'''
DF_RTOFS_temp_salt = pd.read_pickle('DF_RTOFS_temp_salt_Caribbean.pkl')
DF_RTOFS_MLD = pd.read_pickle('DF_RTOFS_MLD_Caribbean.pkl')
DF_RTOFS_OHC = pd.read_pickle('DF_RTOFS_OHC_Caribbean.pkl')
DF_RTOFS_T100 = pd.read_pickle('DF_RTOFS_T100_Caribbean.pkl')

DF_RTOFS_DA_temp_salt = pd.read_pickle('DF_RTOFS_DA_temp_salt_Caribbean.pkl')
DF_RTOFS_DA_MLD = pd.read_pickle('DF_RTOFS_DA_MLD_Caribbean.pkl')
DF_RTOFS_DA_OHC = pd.read_pickle('DF_RTOFS_DA_OHC_Caribbean.pkl')
DF_RTOFS_DA_T100 = pd.read_pickle('DF_RTOFS_DA_T100_Caribbean.pkl')

DF_GOFS_temp_salt = pd.read_pickle('DF_GOFS_temp_salt_Caribbean.pkl')
DF_GOFS_MLD = pd.read_pickle('DF_GOFS_MLD_Caribbean.pkl')
DF_GOFS_OHC = pd.read_pickle('DF_GOFS_OHC_Caribbean.pkl')
DF_GOFS_T100 = pd.read_pickle('DF_GOFS_T100_Caribbean.pkl')
'''

#%% Temperature statistics.
DF_RTOFS = DF_RTOFS_temp_salt.dropna()
DF_RTOFS_DA = DF_RTOFS_DA_temp_salt.dropna()
DF_GOFS = DF_GOFS_temp_salt.dropna()

NRTOFS = len(DF_RTOFS)-1 #For Unbiased estimmator.
NRTOFS_DA = len(DF_RTOFS_DA)-1
NGOFS = len(DF_GOFS)-1

cols = ['CORRELATION','OSTD','MSTD','CRMSE','BIAS']

tskill = np.empty((3,5))
tskill[:] = np.nan

#CORR
tskill[0,0] = DF_RTOFS.corr()['temp_obs']['temp_RTOFS']
tskill[1,0] = DF_RTOFS_DA.corr()['temp_obs']['temp_RTOFS_DA']
tskill[2,0] = DF_GOFS.corr()['temp_obs']['temp_GOFS']

#OSTD
tskill[0,1] = DF_RTOFS.std().temp_obs
tskill[1,1] = DF_RTOFS_DA.std().temp_obs
tskill[2,1] = DF_GOFS.std().temp_obs

#MSTD
tskill[0,2] = DF_RTOFS.std().temp_RTOFS
tskill[1,2] = DF_RTOFS_DA.std().temp_RTOFS_DA
tskill[2,2] = DF_GOFS.std().temp_GOFS

#CRMSE
tskill[0,3] = np.sqrt(np.nansum(((DF_RTOFS.temp_obs-DF_RTOFS.mean().temp_obs)-\
                                 (DF_RTOFS.temp_RTOFS-DF_RTOFS.mean().temp_RTOFS))**2)/NRTOFS)
tskill[1,3] = np.sqrt(np.nansum(((DF_RTOFS_DA.temp_obs-DF_RTOFS_DA.mean().temp_obs)-\
                                 (DF_RTOFS_DA.temp_RTOFS_DA-DF_RTOFS_DA.mean().temp_RTOFS_DA))**2)/NRTOFS_DA)
tskill[2,3] = np.sqrt(np.nansum(((DF_GOFS.temp_obs-DF_GOFS.mean().temp_obs)-\
                                 (DF_GOFS.temp_GOFS-DF_GOFS.mean().temp_GOFS))**2)/NGOFS)

#BIAS
tskill[0,4] = DF_RTOFS.mean().temp_obs - DF_RTOFS.mean().temp_RTOFS
tskill[1,4] = DF_RTOFS_DA.mean().temp_obs - DF_RTOFS_DA.mean().temp_RTOFS_DA
tskill[2,4] = DF_GOFS.mean().temp_obs - DF_GOFS.mean().temp_GOFS

#color
colors = ['indianred','seagreen','darkorchid','darkorange']

temp_skillscores = pd.DataFrame(tskill,
                        index=['RTOFS','RTOFS_DA','GOFS'],
                        columns=cols)

print(temp_skillscores)

#%% Salinity statistics.
DF_RTOFS = DF_RTOFS_temp_salt.dropna()
DF_RTOFS_DA = DF_RTOFS_DA_temp_salt.dropna()
DF_GOFS = DF_GOFS_temp_salt.dropna()

NRTOFS = len(DF_RTOFS)-1  #For Unbiased estimmator.
NRTOFS_DA = len(DF_RTOFS_DA)-1
NGOFS = len(DF_GOFS)-1

cols = ['CORRELATION','OSTD','MSTD','CRMSE','BIAS']

tskill = np.empty((3,5))
tskill[:] = np.nan

#CORR
tskill[0,0] = DF_RTOFS.corr()['salt_obs']['salt_RTOFS']
tskill[1,0] = DF_RTOFS_DA.corr()['salt_obs']['salt_RTOFS_DA']
tskill[2,0] = DF_GOFS.corr()['salt_obs']['salt_GOFS']

#OSTD
tskill[0,1] = DF_RTOFS.std().salt_obs
tskill[1,1] = DF_RTOFS_DA.std().salt_obs
tskill[2,1] = DF_GOFS.std().salt_obs

#MSTD
tskill[0,2] = DF_RTOFS.std().salt_RTOFS
tskill[1,2] = DF_RTOFS_DA.std().salt_RTOFS_DA
tskill[2,2] = DF_GOFS.std().salt_GOFS

#CRMSE
tskill[0,3] = np.sqrt(np.nansum(((DF_RTOFS.salt_obs-DF_RTOFS.mean().salt_obs)-\
                                 (DF_RTOFS.salt_RTOFS-DF_RTOFS.mean().salt_RTOFS))**2)/NRTOFS)
tskill[1,3] = np.sqrt(np.nansum(((DF_RTOFS_DA.salt_obs-DF_RTOFS_DA.mean().salt_obs)-\
                                 (DF_RTOFS_DA.salt_RTOFS_DA-DF_RTOFS_DA.mean().salt_RTOFS_DA))**2)/NRTOFS_DA)
tskill[2,3] = np.sqrt(np.nansum(((DF_GOFS.salt_obs-DF_GOFS.mean().salt_obs)-\
                                 (DF_GOFS.salt_GOFS-DF_GOFS.mean().salt_GOFS))**2)/NGOFS)

#BIAS
tskill[0,4] = DF_RTOFS.mean().salt_obs - DF_RTOFS.mean().salt_RTOFS
tskill[1,4] = DF_RTOFS_DA.mean().salt_obs - DF_RTOFS_DA.mean().salt_RTOFS_DA
tskill[2,4] = DF_GOFS.mean().salt_obs - DF_GOFS.mean().salt_GOFS

#color
colors = ['indianred','seagreen','darkorchid','darkorange']

salt_skillscores = pd.DataFrame(tskill,
                        index=['RTOFS','RTOFS_DA','GOFS'],
                        columns=cols)

print(salt_skillscores)

#%% Mixed layer statistics Temperature.
DF_RTOFS = DF_RTOFS_MLD.dropna()
DF_RTOFS_DA = DF_RTOFS_DA_MLD.dropna()
DF_GOFS = DF_GOFS_MLD.dropna()

NRTOFS = len(DF_RTOFS)-1  #For Unbiased estimmator.
NRTOFS_DA = len(DF_RTOFS_DA)-1
NGOFS = len(DF_GOFS)-1

cols = ['CORRELATION','OSTD','MSTD','CRMSE','BIAS']

tskill = np.empty((3,5))
tskill[:] = np.nan

#CORR
tskill[0,0] = DF_RTOFS.corr()['Tmean_obs']['Tmean_RTOFS']
tskill[1,0] = DF_RTOFS_DA.corr()['Tmean_obs']['Tmean_RTOFS_DA']
tskill[2,0] = DF_GOFS.corr()['Tmean_obs']['Tmean_GOFS']

#OSTD
tskill[0,1] = DF_RTOFS.std().Tmean_obs
tskill[1,1] = DF_RTOFS_DA.std().Tmean_obs
tskill[2,1] = DF_GOFS.std().Tmean_obs

#MSTD
tskill[0,2] = DF_RTOFS.std().Tmean_RTOFS
tskill[1,2] = DF_RTOFS_DA.std().Tmean_RTOFS_DA
tskill[2,2] = DF_GOFS.std().Tmean_GOFS

#CRMSE
tskill[0,3] = np.sqrt(np.nansum(((DF_RTOFS.Tmean_obs-DF_RTOFS.mean().Tmean_obs)-\
                                 (DF_RTOFS.Tmean_RTOFS-DF_RTOFS.mean().Tmean_RTOFS))**2)/NRTOFS)
tskill[1,3] = np.sqrt(np.nansum(((DF_RTOFS_DA.Tmean_obs-DF_RTOFS_DA.mean().Tmean_obs)-\
                                 (DF_RTOFS_DA.Tmean_RTOFS_DA-DF_RTOFS_DA.mean().Tmean_RTOFS_DA))**2)/NRTOFS_DA)
tskill[2,3] = np.sqrt(np.nansum(((DF_GOFS.Tmean_obs-DF_GOFS.mean().Tmean_obs)-\
                                 (DF_GOFS.Tmean_GOFS-DF_GOFS.mean().Tmean_GOFS))**2)/NGOFS)
#BIAS
tskill[0,4] = DF_RTOFS.mean().Tmean_obs - DF_RTOFS.mean().Tmean_RTOFS
tskill[1,4] = DF_RTOFS_DA.mean().Tmean_obs - DF_RTOFS_DA.mean().Tmean_RTOFS_DA
tskill[2,4] = DF_GOFS.mean().Tmean_obs - DF_GOFS.mean().Tmean_GOFS

# colors
colors = ['indianred','seagreen','darkorchid','darkorange']

Tmean_mld_skillscores = pd.DataFrame(tskill,
                        index=['RTOFS','RTOFS_DA','GOFS'],
                        columns=cols)
print(Tmean_mld_skillscores)

#%% Mixed layer statistics Salinity
DF_RTOFS = DF_RTOFS_MLD.dropna()
DF_RTOFS_DA = DF_RTOFS_DA_MLD.dropna()
DF_GOFS = DF_GOFS_MLD.dropna()

NRTOFS = len(DF_RTOFS)-1  #For Unbiased estimmator.
NRTOFS_DA = len(DF_RTOFS_DA)-1
NGOFS = len(DF_GOFS)-1

cols = ['CORRELATION','OSTD','MSTD','CRMSE','BIAS']

tskill = np.empty((3,5))
tskill[:] = np.nan

#CORR
tskill[0,0] = DF_RTOFS.corr()['Smean_obs']['Smean_RTOFS']
tskill[1,0] = DF_RTOFS_DA.corr()['Smean_obs']['Smean_RTOFS_DA']
tskill[2,0] = DF_GOFS.corr()['Smean_obs']['Smean_GOFS']

#OSTD
tskill[0,1] = DF_RTOFS.std().Smean_obs
tskill[1,1] = DF_RTOFS_DA.std().Smean_obs
tskill[2,1] = DF_GOFS.std().Smean_obs

#MSTD
tskill[0,2] = DF_RTOFS.std().Smean_RTOFS
tskill[1,2] = DF_RTOFS_DA.std().Smean_RTOFS_DA
tskill[2,2] = DF_GOFS.std().Smean_GOFS

#CRMSE
tskill[0,3] = np.sqrt(np.nansum(((DF_RTOFS.Smean_obs-DF_RTOFS.mean().Smean_obs)-\
                                 (DF_RTOFS.Smean_RTOFS-DF_RTOFS.mean().Smean_RTOFS))**2)/NRTOFS)
tskill[1,3] = np.sqrt(np.nansum(((DF_RTOFS_DA.Smean_obs-DF_RTOFS_DA.mean().Smean_obs)-\
                                 (DF_RTOFS_DA.Smean_RTOFS_DA-DF_RTOFS_DA.mean().Smean_RTOFS_DA))**2)/NRTOFS_DA)
tskill[2,3] = np.sqrt(np.nansum(((DF_GOFS.Smean_obs-DF_GOFS.mean().Smean_obs)-\
                                 (DF_GOFS.Smean_GOFS-DF_GOFS.mean().Smean_GOFS))**2)/NGOFS)
#BIAS
tskill[0,4] = DF_RTOFS.mean().Smean_obs - DF_RTOFS.mean().Smean_RTOFS
tskill[1,4] = DF_RTOFS_DA.mean().Smean_obs - DF_RTOFS_DA.mean().Smean_RTOFS_DA
tskill[2,4] = DF_GOFS.mean().Smean_obs - DF_GOFS.mean().Smean_GOFS

# colors
colors = ['indianred','seagreen','darkorchid','darkorange']

Smean_mld_skillscores = pd.DataFrame(tskill,
                        index=['RTOFS','RTOFS_DA','GOFS'],
                        columns=cols)
print(Smean_mld_skillscores)

#%% OHC statistics
DF_RTOFS = DF_RTOFS_OHC.dropna()
DF_RTOFS_DA = DF_RTOFS_DA_OHC.dropna()
DF_GOFS = DF_GOFS_OHC.dropna()

NRTOFS = len(DF_RTOFS)-1  #For Unbiased estimmator.
NRTOFS_DA = len(DF_RTOFS_DA)-1
NGOFS = len(DF_GOFS)-1

cols = ['CORRELATION','OSTD','MSTD','CRMSE','BIAS']

tskill = np.empty((3,5))
tskill[:] = np.nan

#CORR
tskill[0,0] = DF_RTOFS.corr()['OHC_obs']['OHC_RTOFS']
tskill[1,0] = DF_RTOFS_DA.corr()['OHC_obs']['OHC_RTOFS_DA']
tskill[2,0] = DF_GOFS.corr()['OHC_obs']['OHC_GOFS']

#OSTD
tskill[0,1] = DF_RTOFS.std().OHC_obs
tskill[1,1] = DF_RTOFS_DA.std().OHC_obs
tskill[2,1] = DF_GOFS.std().OHC_obs

#MSTD
tskill[0,2] = DF_RTOFS.std().OHC_RTOFS
tskill[1,2] = DF_RTOFS_DA.std().OHC_RTOFS_DA
tskill[2,2] = DF_GOFS.std().OHC_GOFS

#CRMSE
tskill[0,3] = np.sqrt(np.nansum(((DF_RTOFS.OHC_obs-DF_RTOFS.mean().OHC_obs)-\
                                 (DF_RTOFS.OHC_RTOFS-DF_RTOFS.mean().OHC_RTOFS))**2)/NRTOFS)
tskill[1,3] = np.sqrt(np.nansum(((DF_RTOFS_DA.OHC_obs-DF_RTOFS_DA.mean().OHC_obs)-\
                                 (DF_RTOFS_DA.OHC_RTOFS_DA-DF_RTOFS_DA.mean().OHC_RTOFS_DA))**2)/NRTOFS_DA)
tskill[2,3] = np.sqrt(np.nansum(((DF_GOFS.OHC_obs-DF_GOFS.mean().OHC_obs)-\
                                 (DF_GOFS.OHC_GOFS-DF_GOFS.mean().OHC_GOFS))**2)/NGOFS)
#BIAS
tskill[0,4] = DF_RTOFS.mean().OHC_obs - DF_RTOFS.mean().OHC_RTOFS
tskill[1,4] = DF_RTOFS_DA.mean().OHC_obs - DF_RTOFS_DA.mean().OHC_RTOFS_DA
tskill[2,4] = DF_GOFS.mean().OHC_obs - DF_GOFS.mean().OHC_GOFS

#color
colors = ['indianred','seagreen','darkorchid','darkorange']

OHC_skillscores = pd.DataFrame(tskill,
                        index=['RTOFS','RTOFS_DA','GOFS'],
                        columns=cols)
print(OHC_skillscores)

#%% T100 statistics
DF_RTOFS = DF_RTOFS_T100.dropna()
DF_RTOFS_DA = DF_RTOFS_DA_T100.dropna()
DF_GOFS = DF_GOFS_T100.dropna()

NRTOFS = len(DF_RTOFS)-1  #For Unbiased estimmator.
NRTOFS_DA = len(DF_RTOFS_DA)-1
NGOFS = len(DF_GOFS)-1

cols = ['CORRELATION','OSTD','MSTD','CRMSE','BIAS']

tskill = np.empty((3,5))
tskill[:] = np.nan

#CORR
tskill[0,0] = DF_RTOFS.corr()['T100_obs']['T100_RTOFS']
tskill[1,0] = DF_RTOFS_DA.corr()['T100_obs']['T100_RTOFS_DA']
tskill[2,0] = DF_GOFS.corr()['T100_obs']['T100_GOFS']

#OSTD
tskill[0,1] = DF_RTOFS.std().T100_obs
tskill[1,1] = DF_RTOFS_DA.std().T100_obs
tskill[2,1] = DF_GOFS.std().T100_obs

#MSTD
tskill[0,2] = DF_RTOFS.std().T100_RTOFS
tskill[1,2] = DF_RTOFS_DA.std().T100_RTOFS_DA
tskill[2,2] = DF_GOFS.std().T100_GOFS

#CRMSE
tskill[0,3] = np.sqrt(np.nansum(((DF_RTOFS.T100_obs-DF_RTOFS.mean().T100_obs)-\
                                 (DF_RTOFS.T100_RTOFS-DF_RTOFS.mean().T100_RTOFS))**2)/NRTOFS)
tskill[1,3] = np.sqrt(np.nansum(((DF_RTOFS_DA.T100_obs-DF_RTOFS_DA.mean().T100_obs)-\
                                 (DF_RTOFS_DA.T100_RTOFS_DA-DF_RTOFS_DA.mean().T100_RTOFS_DA))**2)/NRTOFS_DA)
tskill[2,3] = np.sqrt(np.nansum(((DF_GOFS.T100_obs-DF_GOFS.mean().T100_obs)-\
                                 (DF_GOFS.T100_GOFS-DF_GOFS.mean().T100_GOFS))**2)/NGOFS)

#BIAS
tskill[0,4] = DF_RTOFS.mean().T100_obs - DF_RTOFS.mean().T100_RTOFS
tskill[1,4] = DF_RTOFS_DA.mean().T100_obs - DF_RTOFS_DA.mean().T100_RTOFS_DA
tskill[2,4] = DF_GOFS.mean().T100_obs - DF_GOFS.mean().T100_GOFS

#color
colors = ['indianred','seagreen','darkorchid','darkorange']

T100_skillscores = pd.DataFrame(tskill,
                        index=['RTOFS','RTOFS_DA','GOFS'],
                        columns=cols)
print(T100_skillscores)


##############
#%% Combine all metrics into one normalized Taylor diagram
angle_lim = np.pi/2
std_lim = 2.5

fig,ax1 = taylor_template(angle_lim,std_lim)
markers = ['X','^','s','H']

scores = temp_skillscores
for i,r in enumerate(scores.iterrows()):
    theta=np.arccos(r[1].CORRELATION)
    rr=r[1].MSTD/r[1].OSTD
    if i==2:
        ax1.plot(theta,rr,markers[i],label='Temp',color = 'darkorange',alpha=0.7,markersize=8,markeredgecolor='k')
        ax1.plot(theta,rr,markers[i],fillstyle='none',markersize=8,markeredgecolor='k')
    else:
        ax1.plot(theta,rr,markers[i],color = 'darkorange',alpha=0.7,markersize=8,markeredgecolor='k')
        ax1.plot(theta,rr,markers[i],fillstyle='none',markersize=8,markeredgecolor='k')

scores = salt_skillscores
for i,r in enumerate(scores.iterrows()):
    theta=np.arccos(r[1].CORRELATION)
    rr=r[1].MSTD/r[1].OSTD
    if i==2:
        ax1.plot(theta,rr,markers[i],label='Salt',color = 'seagreen',alpha=0.7,markersize=8,markeredgecolor='k')
        ax1.plot(theta,rr,markers[i],fillstyle='none',markersize=8,markeredgecolor='k')
    else:
        ax1.plot(theta,rr,markers[i],color = 'seagreen',alpha=0.7,markersize=8,markeredgecolor='k')
        ax1.plot(theta,rr,markers[i],fillstyle='none',markersize=8,markeredgecolor='k')

scores = Tmean_mld_skillscores
for i,r in enumerate(scores.iterrows()):
    theta=np.arccos(r[1].CORRELATION)
    rr=r[1].MSTD/r[1].OSTD
    if i==2:
        ax1.plot(theta,rr,markers[i],label='MLT',color = 'darkorchid',alpha=0.7,markersize=8,markeredgecolor='k')
        ax1.plot(theta,rr,markers[i],fillstyle='none',markersize=8,markeredgecolor='k')
    else:
        ax1.plot(theta,rr,markers[i],color = 'darkorchid',alpha=0.7,markersize=8,markeredgecolor='k')
        ax1.plot(theta,rr,markers[i],fillstyle='none',markersize=8,markeredgecolor='k')

scores = Smean_mld_skillscores
for i,r in enumerate(scores.iterrows()):
    theta=np.arccos(r[1].CORRELATION)
    rr=r[1].MSTD/r[1].OSTD
    if i==2:
        ax1.plot(theta,rr,markers[i],label='MLS',color = 'y',alpha=0.7,markersize=8,markeredgecolor='k')
        ax1.plot(theta,rr,markers[i],fillstyle='none',markersize=8,markeredgecolor='k')
    else:
        ax1.plot(theta,rr,markers[i],color = 'y',alpha=0.7,markersize=8,markeredgecolor='k')
        ax1.plot(theta,rr,markers[i],fillstyle='none',markersize=8,markeredgecolor='k')

scores = OHC_skillscores
for i,r in enumerate(scores.iterrows()):
    theta=np.arccos(r[1].CORRELATION)
    rr=r[1].MSTD/r[1].OSTD
    if i==2:
        ax1.plot(theta,rr,markers[i],label='OHC',color = 'indianred',alpha=0.7,markersize=8,markeredgecolor='k')
        ax1.plot(theta,rr,markers[i],fillstyle='none',markersize=8,markeredgecolor='k')
    else:
        ax1.plot(theta,rr,markers[i],color = 'indianred',alpha=0.7,markersize=8,markeredgecolor='k')
        ax1.plot(theta,rr,markers[i],fillstyle='none',markersize=8,markeredgecolor='k')

scores = T100_skillscores
for i,r in enumerate(scores.iterrows()):
    theta=np.arccos(r[1].CORRELATION)
    rr=r[1].MSTD/r[1].OSTD
    if i==2:
        ax1.plot(theta,rr,markers[i],label='T100',color = 'royalblue',alpha=0.7,markersize=8,markeredgecolor='k')
        ax1.plot(theta,rr,markers[i],fillstyle='none',markersize=8,markeredgecolor='k')
    else:
        ax1.plot(theta,rr,markers[i],color = 'royalblue',alpha=0.7,markersize=8,markeredgecolor='k')
        ax1.plot(theta,rr,markers[i],fillstyle='none',markersize=8,markeredgecolor='k')

ax1.plot(0,1,'o',label='Obs',markersize=8,markeredgecolor='k')
ax1.plot(0,0,'Xk',label='RTOFS',markersize=8)
ax1.plot(0,0,'^k',label='RTOFS-DA',markersize=8)
ax1.plot(0,0,'sk',label='GOFS 3.1',markersize=8)

plt.legend(loc='upper left',bbox_to_anchor=[0,2])

rs,ts = np.meshgrid(np.linspace(0,std_lim),np.linspace(0,angle_lim))
rms = np.sqrt(1 + rs**2 - 2*rs*np.cos(ts))

contours = ax1.contour(ts, rs, rms,3,colors='0.5')
plt.clabel(contours, inline=1, fontsize=10)
plt.grid(linestyle=':',alpha=0.5)

file = folder_fig + 'Taylor_norm_RTOFS_RTOFS_DA_GOFS_Caribbean'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)
