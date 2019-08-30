#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 11:02:17 2019

@author: aristizabal
"""
#%% User input

date_ini = '2018-06-01T00:00:00Z'
date_end = '2018-11-30T00:00:00Z'

lon_lim = [-80.0,-60.0]
lat_lim = [15.0,25.0]

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

# urls
url_glider = 'https://data.ioos.us/gliders/erddap'

# downloaded from 
# https://www.esrl.noaa.gov/psd/cgi-bin/db_search/DBSearch.pl?Dataset=NCEP+Reanalysis&Variable=Net+longwave+radiation+flux

net_lwr_nc = '/Volumes/aristizabal/NCEP_reanalysis_data/nlwrs.sfc.gauss.2018.nc' # long wave radiation
net_swr_nc = '/Volumes/aristizabal/NCEP_reanalysis_data/nswrs.sfc.gauss.2018.nc' # short wave radiation
net_lht_nc = '/Volumes/aristizabal/NCEP_reanalysis_data/lhtfl.sfc.gauss.2018.nc' # laten heat flux
net_sht_nc = '/Volumes/aristizabal/NCEP_reanalysis_data/shtfl.sfc.gauss.2018.nc' # sesible heat flux

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'

#%%
from erddapy import ERDDAP
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import xarray as xr
import cmocean 

# Do not produce figures on screen
#plt.switch_backend('agg')
#plt.switch_backend('TkAgg')

# Increase fontsize of labels globally
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('legend',fontsize=14)

#%% Reading bathymetry data

lon_lim2 = [-100.0,-60.0]
lat_lim2 = [5.0,45.0]

ncbath = xr.open_dataset(bath_file)
bath_lat = ncbath.variables['lat'][:]
bath_lon = ncbath.variables['lon'][:]
bath_elev = ncbath.variables['elevation'][:]

# Getting subdomain for plotting glider track on bathymetry
oklatbath = np.logical_and(bath_lat >= lat_lim2[0],bath_lat <= lat_lim2[1])
oklonbath = np.logical_and(bath_lon >= lon_lim2[0],bath_lon <= lon_lim2[1])
        
bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath]

#%% Read net longwave radiation data

tini = datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
tend = datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

Netlwr = xr.open_dataset(net_lwr_nc)
Net_lwr_time = np.asarray(Netlwr.variables['time'][:])
Net_lwr_lat = np.asarray(Netlwr.variables['lat'][:])
Net_lwr_lonn = np.asarray(Netlwr.variables['lon'][:])
Net_lwr = np.asarray(Netlwr.variables['nlwrs'][:])

# Conversion from NCEP reanalaysis longitude convention to geographic convention
Net_lwr_lon = np.empty((len(Net_lwr_lonn),))
Net_lwr_lon[:] = np.nan
for i,ii in enumerate(Net_lwr_lonn):
    if ii > 180: 
        Net_lwr_lon[i] = ii - 360
    else:
        Net_lwr_lon[i] = ii
    
ok = np.argsort(Net_lwr_lon, axis=0, kind='quicksort', order=None)    
Net_lwr_lon =  Net_lwr_lon[ok] 
Net_lwr = Net_lwr[:,:,ok]  

ok_lat = np.where(np.logical_and(Net_lwr_lat >= lat_lim[0],Net_lwr_lat <= lat_lim[1]))
ok_lon = np.where(np.logical_and(Net_lwr_lon >= lon_lim[0],Net_lwr_lon <= lon_lim[1]))
ok_time = np.where(np.logical_and(mdates.date2num(Net_lwr_time) >= mdates.date2num(tini),\
                                  mdates.date2num(Net_lwr_time) <= mdates.date2num(tend)))

net_lwr_lon = Net_lwr_lon[ok_lon[0]]
net_lwr_lat = Net_lwr_lat[ok_lat[0]]
net_lwr_time= Net_lwr_time[ok_time[0]]
net_lwr = np.asarray(Net_lwr[ok_time[0],:,:][:,ok_lat[0],:][:,:,ok_lon[0]]) 


#%% Read net shortwave radiation data

tini = datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
tend = datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

Netswr = xr.open_dataset(net_swr_nc)
Net_swr_time = np.asarray(Netswr.variables['time'][:])
Net_swr_lat = np.asarray(Netswr.variables['lat'][:])
Net_swr_lonn = np.asarray(Netswr.variables['lon'][:])
Net_swr = np.asarray(Netswr.variables['nswrs'][:])

# Conversion from NCEP reanalaysis longitude convention to geographic convention
Net_swr_lon = np.empty((len(Net_swr_lonn),))
Net_swr_lon[:] = np.nan
for i,ii in enumerate(Net_swr_lonn):
    if ii > 180: 
        Net_swr_lon[i] = ii - 360
    else:
        Net_swr_lon[i] = ii
    
ok = np.argsort(Net_swr_lon, axis=0, kind='quicksort', order=None)    
Net_swr_lon =  Net_swr_lon[ok] 
Net_swr = Net_swr[:,:,ok]  

ok_lat = np.where(np.logical_and(Net_swr_lat >= lat_lim[0],Net_swr_lat <= lat_lim[1]))
ok_lon = np.where(np.logical_and(Net_swr_lon >= lon_lim[0],Net_swr_lon <= lon_lim[1]))
ok_time = np.where(np.logical_and(mdates.date2num(Net_swr_time) >= mdates.date2num(tini),\
                                  mdates.date2num(Net_swr_time) <= mdates.date2num(tend)))

net_swr_lon = Net_swr_lon[ok_lon[0]]
net_swr_lat = Net_swr_lat[ok_lat[0]]
net_swr_time= Net_swr_time[ok_time[0]]
net_swr = np.asarray(Net_swr[ok_time[0],:,:][:,ok_lat[0],:][:,:,ok_lon[0]]) 

#%% Read net latent heat flux data

tini = datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
tend = datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

Netlht = xr.open_dataset(net_lht_nc)
Net_lht_time = np.asarray(Netlht.variables['time'][:])
Net_lht_lat = np.asarray(Netlht.variables['lat'][:])
Net_lht_lonn = np.asarray(Netlht.variables['lon'][:])
Net_lht = np.asarray(Netlht.variables['lhtfl'][:])

# Conversion from NCEP reanalaysis longitude convention to geographic convention
Net_lht_lon = np.empty((len(Net_lht_lonn),))
Net_lht_lon[:] = np.nan
for i,ii in enumerate(Net_lht_lonn):
    if ii > 180: 
        Net_lht_lon[i] = ii - 360
    else:
        Net_lht_lon[i] = ii
    
ok = np.argsort(Net_lht_lon, axis=0, kind='quicksort', order=None)    
Net_lht_lon =  Net_lht_lon[ok] 
Net_lht = Net_lht[:,:,ok]  

ok_lat = np.where(np.logical_and(Net_lht_lat >= lat_lim[0],Net_lht_lat <= lat_lim[1]))
ok_lon = np.where(np.logical_and(Net_lht_lon >= lon_lim[0],Net_lht_lon <= lon_lim[1]))
ok_time = np.where(np.logical_and(mdates.date2num(Net_lht_time) >= mdates.date2num(tini),\
                                  mdates.date2num(Net_lht_time) <= mdates.date2num(tend)))

net_lht_lon = Net_lht_lon[ok_lon[0]]
net_lht_lat = Net_lht_lat[ok_lat[0]]
net_lht_time= Net_lht_time[ok_time[0]]
net_lht = np.asarray(Net_lht[ok_time[0],:,:][:,ok_lat[0],:][:,:,ok_lon[0]]) 

#%% Read net sensible heat flux data

tini = datetime.strptime(date_ini,'%Y-%m-%dT%H:%M:%SZ')
tend = datetime.strptime(date_end,'%Y-%m-%dT%H:%M:%SZ')

Netsht = xr.open_dataset(net_sht_nc)
Net_sht_time = np.asarray(Netsht.variables['time'][:])
Net_sht_lat = np.asarray(Netsht.variables['lat'][:])
Net_sht_lonn = np.asarray(Netsht.variables['lon'][:])
Net_sht = np.asarray(Netsht.variables['shtfl'][:])

# Conversion from NCEP reanalaysis longitude convention to geographic convention
Net_sht_lon = np.empty((len(Net_sht_lonn),))
Net_sht_lon[:] = np.nan
for i,ii in enumerate(Net_sht_lonn):
    if ii > 180: 
        Net_sht_lon[i] = ii - 360
    else:
        Net_sht_lon[i] = ii
    
ok = np.argsort(Net_sht_lon, axis=0, kind='quicksort', order=None)    
Net_sht_lon =  Net_sht_lon[ok] 
Net_sht = Net_sht[:,:,ok]  

ok_lat = np.where(np.logical_and(Net_sht_lat >= lat_lim[0],Net_sht_lat <= lat_lim[1]))
ok_lon = np.where(np.logical_and(Net_sht_lon >= lon_lim[0],Net_sht_lon <= lon_lim[1]))
ok_time = np.where(np.logical_and(mdates.date2num(Net_sht_time) >= mdates.date2num(tini),\
                                  mdates.date2num(Net_sht_time) <= mdates.date2num(tend)))

net_sht_lon = Net_sht_lon[ok_lon[0]]
net_sht_lat = Net_sht_lat[ok_lat[0]]
net_sht_time= Net_sht_time[ok_time[0]]
net_sht = np.asarray(Net_sht[ok_time[0],:,:][:,ok_lat[0],:][:,:,ok_lon[0]]) 

#%% Look for datasets in IOOS glider dac
print('Looking for glider data sets')
e = ERDDAP(server = url_glider)

# Grab every dataset available
datasets = pd.read_csv(e.get_search_url(response='csv', search_for='all'))

# Search constraints
kw = {
    'min_lon': lon_lim[0],
    'max_lon': lon_lim[1],
    'min_lat': lat_lim[0],
    'max_lat': lat_lim[1],
    'min_time': date_ini,
    'max_time': date_end,
}

search_url = e.get_search_url(response='csv', **kw)
#print(search_url)

# Grab the results
search = pd.read_csv(search_url)

# Extract the IDs
gliders = search['Dataset ID'].values

msg = 'Found {} Glider Datasets:\n\n{}'.format
print(msg(len(gliders), '\n'.join(gliders)))

# Setting constraints
constraints = {
        'time>=': date_ini,
        'time<=': date_end,
        'latitude>=': lat_lim[0],
        'latitude<=': lat_lim[1],
        'longitude>=': lon_lim[0],
        'longitude<=': lon_lim[1],
        }

variables = [
        'depth',
        'latitude',
        'longitude',
        'time',
        'temperature',
        'salinity',
        'density'
        ]

e = ERDDAP(
        server=url_glider,
        protocol='tabledap',
        response='nc'
        )

#%% Map of North Atlantic with gliders mean position
'''
col = ['red','darkcyan','gold','m','darkorange','crimson','lime',\
       'darkorchid','brown','sienna','yellow','orchid','gray','orange','seagreen']
mark = ['o','*','p','^','D','X','o','*','p','^','D','X','o','*','p','^','D']
edgc = ['k','w','k','w','k','w','k','w','k','w','k','w','k','w','k','k','w','k']

fig, ax = plt.subplots(figsize=(6, 5))
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,cmap='Blues_r')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
#plt.yticks([])
#plt.xticks([])
plt.axis([lon_lim[0],lon_lim[1],lat_lim[0],lat_lim[1]])
plt.title('Gliders in the Caribbean Deployments \nduring 2018 Hurricane Season',fontsize=20)
plt.axis('scaled')

for i,id in enumerate(gliders[:-1]):
    if id[0:3] != 'all':
        #print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables

        df = e.to_pandas(
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        ax.plot(np.nanmean(df['longitude (degrees_east)']),\
                np.nanmean(df['latitude (degrees_north)']),'-',color=col[i],\
                marker = mark[i],markeredgecolor = 'k',markersize=7,\
                label=id.split('-')[0])
        ax.legend(fontsize=14)
        #ax.text(np.mean(df['longitude']),np.mean(df['latitude']),id.split('-')[0])

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + 'map_North_Atlantic_gliders_deployed_Caribbean_hurric_2018.png'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)

#%% Map of North Atlantic with gliders mean position detail

# Getting subdomain for plotting glider track on bathymetry
oklatbath = np.logical_and(bath_lat >= lat_lim[0],bath_lat <= lat_lim[1])
oklonbath = np.logical_and(bath_lon >= lon_lim[0],bath_lon <= lon_lim[1])
        
bath_latsub = bath_lat[oklatbath]
bath_lonsub = bath_lon[oklonbath]
bath_elevs = bath_elev[oklatbath,:]
bath_elevsub = bath_elevs[:,oklonbath]

col = ['red','darkcyan','gold','m','darkorange','crimson','lime',\
       'darkorchid','brown','sienna','yellow','orchid','gray','orange','seagreen']
mark = ['o','*','p','^','D','X','o','*','p','^','D','X','o','*','p','^','D']
edgc = ['k','w','k','w','k','w','k','w','k','w','k','w','k','w','k','k','w','k']

fig, ax = plt.subplots(figsize=(6, 5))
plt.contour(bath_lonsub,bath_latsub,bath_elevsub,[0],colors='k')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,cmap='Blues_r')
plt.contourf(bath_lonsub,bath_latsub,bath_elevsub,[0,10000],colors='seashell')
#plt.yticks([])
#plt.xticks([])
plt.axis([lon_lim[0],lon_lim[1],lat_lim[0],lat_lim[1]])
plt.title('Gliders in the Caribbean Deployments \nduring 2018 Hurricane Season',fontsize=20)
plt.axis('scaled')

for i,id in enumerate(gliders[:-1]):
    if id[0:3] != 'all':
        #print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables

        df = e.to_pandas(
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        ax.plot(np.nanmean(df['longitude (degrees_east)']),\
                np.nanmean(df['latitude (degrees_north)']),'-',color=col[i],\
                marker = mark[i],markeredgecolor = 'k',markersize=7,\
                label=id.split('-')[0])
        ax.legend(fontsize=14)
        #ax.text(np.mean(df['longitude']),np.mean(df['latitude']),id.split('-')[0])

folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'
file = folder + 'map_North_Atlantic_gliders_deployed_Caribbean_hurric_2018_2.png'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)
'''
#%% Let's look first at ng630

id = gliders[11]

print('Reading ' + id )
e.dataset_id = id
e.constraints = constraints
e.variables = variables
    
# checking data frame is not empty
print('checking df')
df = e.to_pandas()
print('checked df')

if len(df.index) != 0: 
    # Converting glider data to data frame
    df = e.to_pandas(
            index_col='time (UTC)',
            parse_dates=True,
            skiprows=(1,)  # units information can be dropped.
            ).dropna()
    # Coverting glider vectors into arrays
    timeg, ind = np.unique(df.index.values,return_index=True)
    latg = df['latitude (degrees_north)'].values[ind]
    long = df['longitude (degrees_east)'].values[ind]
    dg = df['depth (m)'].values
    #vg = df['temperature (degree_Celsius)'].values
    tg = df[df.columns[3]].values
    sg = df[df.columns[4]].values
    dng = df[df.columns[5]].values

    delta_z = 0.5
    zn = np.int(np.round(np.max(dg)/delta_z))
    depthg = np.empty((zn,len(timeg)))
    depthg[:] = np.nan
    tempg = np.empty((zn,len(timeg)))
    tempg[:] = np.nan
    saltg = np.empty((zn,len(timeg)))
    saltg[:] = np.nan
    densg = np.empty((zn,len(timeg)))
    densg[:] = np.nan
        
    # Grid variables
    depthg_gridded = np.arange(0,np.nanmax(dg),delta_z)
    tempg_gridded = np.empty((len(depthg_gridded),len(timeg)))
    tempg_gridded[:] = np.nan
    saltg_gridded = np.empty((len(depthg_gridded),len(timeg)))
    saltg_gridded[:] = np.nan
    densg_gridded = np.empty((len(depthg_gridded),len(timeg)))
    densg_gridded[:] = np.nan
    
    for i,ii in enumerate(ind):
         if i < len(timeg)-1:
             depthg[0:len(dg[ind[i]:ind[i+1]]),i] = dg[ind[i]:ind[i+1]]
             tempg[0:len(tg[ind[i]:ind[i+1]]),i] = tg[ind[i]:ind[i+1]]
             saltg[0:len(sg[ind[i]:ind[i+1]]),i] = sg[ind[i]:ind[i+1]]
             densg[0:len(dg[ind[i]:ind[i+1]]),i] = dng[ind[i]:ind[i+1]]
         else:
             depthg[0:len(dg[ind[i]:len(dg)]),i] = dg[ind[i]:len(dg)]
             tempg[0:len(tg[ind[i]:len(tg)]),i] = tg[ind[i]:len(tg)]
             saltg[0:len(sg[ind[i]:len(sg)]),i] = sg[ind[i]:len(sg)]
             densg[0:len(dg[ind[i]:len(dg)]),i] = dng[ind[i]:len(dg)]
    for t,tt in enumerate(timeg):
        print('t',t)
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
            okd = np.logical_and(depthg_gridded >= np.min(depthf[oks]),\
                                 depthg_gridded < np.max(depthf[oks]))
            saltg_gridded[okd,t] = np.interp(depthg_gridded[okd],depthf[oks],saltf[oks])
            
        okdd = np.isfinite(densf)
        if np.sum(okd) < 3:
            densg_gridded[:,t] = np.nan
        else:
            okd = np.logical_and(depthg_gridded >= np.min(depthf[okdd]),\
                                 depthg_gridded < np.max(depthf[okdd]))
            densg_gridded[okd,t] = np.interp(depthg_gridded[okd],depthf[okdd],densf[okdd])

#%% Temperature transect

fig,ax = plt.subplots(figsize=(15, 4))

nlevels = np.round(np.nanmax(tempg_gridded)) - np.round(np.nanmin(tempg_gridded)) + 1
kw = dict(levels = np.linspace(np.round(np.nanmin(tempg_gridded)),\
                               np.round(np.nanmax(tempg_gridded)),nlevels))
#plt.contour(timeg,-depthg_gridded,varg_gridded,colors = 'lightgrey',**kw)
cs = plt.contourf(timeg,-depthg_gridded,tempg_gridded,cmap=cmocean.cm.thermal,**kw)
plt.xlim(df.index[0], df.index[-1])
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
cbar = fig.colorbar(cs, orientation='vertical')
cbar.ax.set_ylabel('($^\circ$C)',fontsize=16)
ax.set_ylabel('Depth (m)',fontsize=16)  
plt.title('Temperature Transect ' + id,fontsize=18)  

file = folder + ' ' + id + '_temp'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Density transect

fig,ax = plt.subplots(figsize=(15, 4))

nlevels = np.round(np.nanmax(densg_gridded)) - np.round(np.nanmin(densg_gridded)) + 1
kw = dict(levels = np.linspace(np.round(np.nanmin(densg_gridded)),\
                               np.round(np.nanmax(densg_gridded)),nlevels))
#plt.contour(timeg,-depthg_gridded,varg_gridded,colors = 'lightgrey',**kw)
cs = plt.contourf(timeg,-depthg_gridded,densg_gridded,cmap=cmocean.cm.dense,**kw)
plt.xlim(df.index[0], df.index[-1])
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
cbar = fig.colorbar(cs, orientation='vertical')
cbar.ax.set_ylabel('$kg/m^3$',fontsize=16)
ax.set_ylabel('Depth (m)',fontsize=16)  
plt.title('Density Transect ' + id,fontsize=18)  

file = folder + ' ' + id + '_dens'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)  

#%%  Calculation of mixed layer depth based on dt, Tmean: mean temp within the 
# mixed layer and td: temp at 1 meter below the mixed layer            

d10 = np.where(depthg_gridded >= 10)[0][0]
dt = 0.2

MLD_dt = np.empty(len(timeg)) 
MLD_dt[:] = np.nan
Tmean = np.empty(len(timeg)) 
Tmean[:] = np.nan
Td = np.empty(len(timeg)) 
Td[:] = np.nan
for t,tt in enumerate(timeg):
    T10 = tempg_gridded[d10,t]
    delta_T = T10 - tempg_gridded[:,t] 
    ok_mld = np.where(delta_T <= dt)[0]    
    if ok_mld.size == 0:
        MLD_dt[t] = np.nan
        Tmean[t] = np.nan
        Td[t] = np.nan
    else:
        ok_mld_plus1m = np.where(depthg_gridded >= depthg_gridded[ok_mld[-1]] + 1)[0][0]
        MLD_dt[t] = depthg_gridded[ok_mld[-1]]
        Tmean[t] = np.nanmean(tempg_gridded[ok_mld,t])
        Td[t] = tempg_gridded[ok_mld_plus1m,t]
        
#%% Tmean and Td
        
fig,ax = plt.subplots(figsize=(12, 4))
plt.plot(timeg,Tmean,'.-k',label='Tmean')
plt.plot(timeg,Td,'.-g',label='Td')
plt.legend()
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.xlim([timeg[0],timeg[-1]])

file = folder + ' ' + id + '_Tmean_Td'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%%  Calculation of mixed layer depth based on drho

d10 = np.where(depthg_gridded >= 10)[0][0]
drho = 0.125

MLD_drho = np.empty(len(timeg)) 
MLD_drho[:] = np.nan
for t,tt in enumerate(timeg):
    rho10 = densg_gridded[d10,t]
    delta_rho = -(rho10 - densg_gridded[:,t]) 
    ok_mld = np.where(delta_rho <= drho)
    if ok_mld[0].size == 0:
        MLD_drho[t] = np.nan
    else:
        MLD_drho[t] = depthg_gridded[ok_mld[0][-1]] 
        
#%% 

fig,ax = plt.subplots(figsize=(12, 4))
plt.plot(timeg,MLD_dt,'.-k',label='MLD_dt')
plt.plot(timeg,MLD_drho,'.-g',label='MLD_drho')
plt.legend() 
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.xlim([timeg[0],timeg[-1]])

file = folder + ' ' + id + '_MLD_dt_drho'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% MLD_dt and MLD_drho

fig,ax = plt.subplots(figsize=(10, 4))

nlevels = np.round(np.nanmax(tempg_gridded)) - np.round(np.nanmin(tempg_gridded)) + 1
kw = dict(levels = np.linspace(15,31,17))
#plt.contour(timeg,-depthg_gridded,varg_gridded,colors = 'lightgrey',**kw)
cs = plt.contourf(timeg,-depthg_gridded,tempg_gridded,cmap=cmocean.cm.thermal,**kw)
plt.contour(timeg,-depthg_gridded,tempg_gridded,[26],colors='grey')
plt.plot(timeg,-MLD_dt,'.k',label='MLD_dt')
plt.plot(timeg,-MLD_drho,'.g',label='MLD_drho')
plt.xlim(df.index[0], df.index[-1])
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
cbar = fig.colorbar(cs, orientation='vertical')
cbar.ax.set_ylabel('($^\circ$C)',fontsize=16)
ax.set_ylabel('Depth (lm)',fontsize=16)  
plt.title('Temperature Transect ' + id,fontsize=18) 
plt.ylim(-200,0) 
plt.legend()

file = folder + ' ' + id + '_MLD'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)         
                
#%% Calculate dTmean/dt

dTmean_dt = (Tmean[1:]-Tmean[0:-1])/(mdates.date2num(timeg[1:]) - mdates.date2num(timeg[0:-1]))
        
# convert dTmean/dt in m/6 hours
dTmean_dt = dTmean_dt / 4 

timeg_mid = (mdates.date2num(timeg[1:]) + mdates.date2num(timeg[0:-1]))/2

#%% dTmean/dt

fig,ax = plt.subplots(figsize=(12, 4))
plt.plot(timeg_mid,dTmean_dt,'.-k',label='dTmean_dt')
plt.legend()
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.xlim([timeg[0],timeg[-1]])

file = folder + ' ' + id + '_Tmean/Td'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)   

#%%  Interpolate MLD to NCEP reanalysis time

MLD_dt_interp = np.interp(mdates.date2num(net_lwr_time),mdates.date2num(timeg),MLD_dt)        
      
plt.figure()
plt.plot(timeg,MLD_dt,'.-k')
plt.plot(net_lwr_time,MLD_dt_interp,'.-g')

#%%  Calculate dT/dt due to Net long wave radiation

rho0 = 1027 #kg/m^3
c_rho = 4300 # specific heat capacity at constant pressure J/kg ^oC

# Interpolate long and latg onto reanalysis grid
oklon_lwr = np.round(np.interp(long,net_lwr_lon,np.arange(len(net_lwr_lon)))).astype(int)
oklat_lwr = np.round(np.interp(latg,net_lwr_lat,np.arange(len(net_lwr_lat)))).astype(int)

dT_dt_lwr = np.empty(len(net_lwr_time))
dT_dt_lwr[:] = np.nan
for tind,t in enumerate(net_lwr_time):
    dT_dt_lwr[tind] = net_lwr[tind,oklat_lwr[tind],oklon_lwr[tind]]/(rho0 * c_rho * MLD_dt_interp[tind])
  
# conversion to ^oC/6h 
dT_dt_lwr = dT_dt_lwr * 3600 * 6

#%%  Calculate dT/dt due to Net short wave radiation

rho0 = 1027 #kg/m^3
c_rho = 4300 # specific heat capacity at constant pressure J/kg ^oC

# Interpolate long and latg onto reanalysis grid
oklon_swr = np.round(np.interp(long,net_swr_lon,np.arange(len(net_swr_lon)))).astype(int)
oklat_swr = np.round(np.interp(latg,net_swr_lat,np.arange(len(net_swr_lat)))).astype(int)

dT_dt_swr = np.empty(len(net_swr_time))
dT_dt_swr[:] = np.nan
for tind,t in enumerate(net_swr_time):
    dT_dt_swr[tind] = net_swr[tind,oklat_swr[tind],oklon_swr[tind]]/(rho0 * c_rho * MLD_dt_interp[tind])
  
# conversion to ^oC/6h 
dT_dt_swr = dT_dt_swr * 3600 * 6

#%%  Calculate dT/dt due to Net latent heat flux

rho0 = 1027 #kg/m^3
c_rho = 4300 # specific heat capacity at constant pressure J/kg ^oC

# Interpolate long and latg onto reanalysis grid
oklon_lht = np.round(np.interp(long,net_lht_lon,np.arange(len(net_lht_lon)))).astype(int)
oklat_lht = np.round(np.interp(latg,net_lht_lat,np.arange(len(net_lht_lat)))).astype(int)

dT_dt_lht = np.empty(len(net_lht_time))
dT_dt_lht[:] = np.nan
for tind,t in enumerate(net_lht_time):
    dT_dt_lht[tind] = net_lht[tind,oklat_lht[tind],oklon_lht[tind]]/(rho0 * c_rho * MLD_dt_interp[tind])
  
# conversion to ^oC/6h 
dT_dt_lht = dT_dt_lht * 3600 * 6

#%%  Calculate dT/dt due to Net sensible heat flux

rho0 = 1027 #kg/m^3
c_rho = 4300 # specific heat capacity at constant pressure J/kg ^oC

# Interpolate long and latg onto reanalysis grid
oklon_sht = np.round(np.interp(long,net_sht_lon,np.arange(len(net_sht_lon)))).astype(int)
oklat_sht = np.round(np.interp(latg,net_sht_lat,np.arange(len(net_sht_lat)))).astype(int)

dT_dt_sht = np.empty(len(net_sht_time))
dT_dt_sht[:] = np.nan
for tind,t in enumerate(net_sht_time):
    dT_dt_sht[tind] = net_sht[tind,oklat_sht[tind],oklon_sht[tind]]/(rho0 * c_rho * MLD_dt_interp[tind])
  
# conversion to ^oC/6h 
dT_dt_sht = dT_dt_sht * 3600 * 6

#%% dT/dt due to lwr, swr, lht, sht

fig,ax = plt.subplots(figsize=(12, 4))
plt.plot(net_lht_time,dT_dt_lht,'.-b',label='dT/dt lht')
plt.plot(net_sht_time,dT_dt_sht,'.-r',label='dT/dt sht')
plt.plot(net_swr_time,dT_dt_swr,'.-g',label='dT/dt lwr')
plt.plot(net_lwr_time,dT_dt_lwr,'.-k',label='dT/dt lwr')
plt.legend()
plt.ylim([-0.14,0.14])
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.xlim([timeg[0],timeg[-1]])

file = folder + ' ' + id + '_dT_dt_lwr_swr_sht_lht'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)   

#%% Calculate dh/dt (Tmean - Td)/h

dh_dt = (MLD_dt[1:] - MLD_dt[0:-1]) / (mdates.date2num(timeg[1:]) - mdates.date2num(timeg[0:-1]))

# convert dt_dt in m/6 hours
dh_dt = dh_dt / 4

timeg_mid = (mdates.date2num(timeg[1:]) + mdates.date2num(timeg[0:-1]))/2

Tmean_mid = np.interp(timeg_mid,mdates.date2num(timeg),Tmean)
Td_mid = np.interp(timeg_mid,mdates.date2num(timeg),Td)
MLD_dt_mid = np.interp(timeg_mid,mdates.date2num(timeg),MLD_dt)

dT_dt_MLD_deep = dh_dt * (Tmean_mid - Td_mid)/MLD_dt_mid 

#%% 

fig,ax = plt.subplots(figsize=(12,4))
plt.plot(timeg_mid,dT_dt_MLD_deep,'.-k',label='dT/dt MLD deepening')
plt.legend()
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
plt.xlim([timeg[0],timeg[-1]])

file = folder + ' ' + id + '_dT_dt_entrainment'
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)   

        
#%%
'''
t=100            
plt.figure()
plt.plot(tempg_gridded[:,t],-depthg_gridded,'*b')
plt.plot(tempg[:,t],-depthg[:,t],'o-k')        
'''        
#%%
'''       
plt.figure()
plt.plot(delta_T,-depthg_gridded,'.-')
'''
#%%
'''
plt.figure()
plt.plot(delta_rho,-depthg_gridded,'.-')
'''
     