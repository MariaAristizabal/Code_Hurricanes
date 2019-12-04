#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:06:33 2018

@author: aristizabal
"""

#%% User input

# lat and lon bounds
lon_lim = [-100.0,0.0]
lat_lim = [15.0,45.0]

# Time bounds
min_time = '2019-06-01T00:00:00Z'
max_time = '2019-11-30T00:00:00Z'

# Bathymetry file
bath_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc'

# Server url 
server = 'https://data.ioos.us/gliders/erddap'

#%%

from erddapy import ERDDAP
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#import time
from datetime import datetime
#from matplotlib.dates import date2num

import numpy as np

#%% Look for datasets 

server = 'https://data.ioos.us/gliders/erddap'

e = ERDDAP(server = server)

# Grab every dataset available
datasets = pd.read_csv(e.get_search_url(response='csv', search_for='all'))

# Search constraints
kw = {
    'min_lon': lon_lim[0],
    'max_lon': lon_lim[-1],
    'min_lat': lat_lim[0],
    'max_lat': lat_lim[-1],
    'min_time': min_time,
    'max_time': max_time,
}

search_url = e.get_search_url(response='csv', **kw)
#print(search_url)

# Grab the results
search = pd.read_csv(search_url)

# Extract the IDs
gliders = search['Dataset ID'].values

msg = 'Found {} Glider Datasets:\n\n{}'.format
print(msg(len(gliders), '\n'.join(gliders)))

#%%

constraints = {
    'time>=': min_time,
    'time<=': max_time,
    'latitude>=': lat_lim[0],
    'latitude<=': lat_lim[-1],
    'longitude>=': lon_lim[0],
    'longitude<=': lon_lim[-1],
}

variables = ['latitude','longitude','time']

#%%


e = ERDDAP(
    server=server,
    protocol='tabledap',
    response='nc'
)
'''
for id in gliders:
    e.dataset_id=id
    e.constraints=constraints
    e.variables=variables
    
    df = e.to_pandas(
    index_col='time (UTC)',
    parse_dates=True,
    skiprows=(1,)  # units information can be dropped.
                    ).dropna()
        
'''

#%% Reading glider data and plotting lat and lon on the map
'''
#plt.style.use('classic')    
siz=12

fig, ax = plt.subplots(figsize=(10, 8), dpi=80, facecolor='w', edgecolor='w') 
ax.contour(bath_lon,bath_lat,bath_elev,[-1000],colors='silver')
ax.contour(bath_lon,bath_lat,bath_elev,[0],colors='k')
#ax.contour(bath_lon[oklonbath],bath_lat[oklatbath],bath_elev[np.c_[oklatbath],oklonbath],colors='k')   
plt.axis('equal')
#plt.axis([lon_lim[0],lon_lim[-1],lat_lim[0],lat_lim[-1]])
plt.title('Gliders During Hurricane Season 2018')

for id in gliders:
    if id[0:3] != 'all':
        #print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        index_col='time',
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        ax.plot(np.mean(np.mean(df['longitude'])),np.mean(np.mean(df['latitude'])),'*',markersize = 10 )   
        #ax.text(np.mean(df['longitude']),np.mean(df['latitude']),id.split('-')[0])
    
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/map_gliders_hurric_season_2018.png")
plt.show() 
'''
        
#%%
        
glider = [l.split('-')[0] for l in gliders]

fund_agency = [None]*(len(glider))
for i,l in enumerate(glider):
    if glider[i][0:2] == 'ng':
        fund_agency[i] = 'Navy'
    if glider[i][0:2] == 'SG':
        fund_agency[i] = 'NOAA'
    if glider[i][0:2] == 'sp':
        fund_agency[i] = 'NOAA'
    if glider[i] == 'angus':
        fund_agency[i] = 'NOAA'
    if glider[i] == 'usf':
        fund_agency[i] = 'NOAA'    
    if glider[i] == 'bios_anna':
        fund_agency[i] = 'NSF'
    if glider[i] == 'bios_jack':
        fund_agency[i] = 'NSF'
    if glider[i] == 'bios_minnie':
        fund_agency[i] = 'Simmons'
    if glider[i] == 'blue':
        fund_agency[i] = 'NOAA'
    if glider[i] == 'franklin':
        fund_agency[i] = 'NOAA'
    if glider[i] == 'mote':
        fund_agency[i] = 'NOAA'
    if glider[i] == 'ru22':
        fund_agency[i] = 'NOAA'
    if glider[i] == 'ru28':
        fund_agency[i] = 'NJDEP'
    if glider[i] == 'ru29':
        fund_agency[i] = 'Vetlesen'
    if glider[i] == 'ru30':
        fund_agency[i] = 'NOAA'
    if glider[i] == 'sam':
        fund_agency[i] = 'NOAA'
    if glider[i] == 'sbu01':
        fund_agency[i] = 'NYDEC'
    if glider[i] == 'SG636':
        fund_agency[i] = 'Shell'
    if glider[i] == 'silbo':
        fund_agency[i] = 'TWR'
    if glider[i] == 'Sverdrup':
        fund_agency[i] = 'NOAA'
    if glider[i] == 'glos_236':
        fund_agency[i] = 'NOAA'
    if glider[i] == 'ru33':
        fund_agency[i] = 'NOAA'

#%% Glider in each category

n_navy = len([i for i,list in enumerate(fund_agency) if list == 'Navy'])
n_noaa = len([i for i,list in enumerate(fund_agency) if list == 'NOAA'])
n_nsf = len([i for i,list in enumerate(fund_agency) if list == 'NSF'])
n_twr = len([i for i,list in enumerate(fund_agency) if list == 'TWR'])
n_simmons = len([i for i,list in enumerate(fund_agency) if list == 'Simmons'])
n_nydec = len([i for i,list in enumerate(fund_agency) if list == 'NYDEC'])
n_shell = len([i for i,list in enumerate(fund_agency) if list == 'Shell'])
n_njdep = len([i for i,list in enumerate(fund_agency) if list == 'NJDEP'])
n_vetlesen = len([i for i,list in enumerate(fund_agency) if list == 'Vetlesen'])

#%% Pie chart of number of gliders in each category

labels = 'NOAA - '+str(n_noaa),'Navy - '+str(n_navy),'NSF - '+str(n_nsf),\
         'NYDEC - '+str(n_nydec),'NJDEP - '+str(n_njdep),\
         'Simmons - '+str(n_simmons),'Shell - '+str(n_shell),\
         'TWR - '+str(n_twr),'Vetlesen - '+str(n_vetlesen) 
siz = [n_noaa,n_navy,n_nsf,n_nydec,n_njdep,n_simmons,n_shell,n_twr,n_vetlesen]
sizes = np.ndarray.tolist(np.multiply(siz,1/np.sum(siz)))
colors = ['royalblue','goldenrod','firebrick','forestgreen','darkorange','black',\
          'rebeccapurple','aqua','yellowgreen'] 
#explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')


plt.figure()
patches, texts = plt.pie(sizes,startangle=90,colors=colors) #,autopct='%2d')
plt.legend(patches,labels,loc='best',bbox_to_anchor=(0.85, 1))
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Total Number of Gliders = '+str(len(gliders)),fontsize=16)

plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/pie_chart_number_gliders_hurica_season_2019.png"\
            ,bbox_inches = 'tight',pad_inches = 0.1)

#%% Plotting the deployment window of all glider in Hurricane season 2019 
      
siz=12

funding = list(set(fund_agency))
color_fund1 = ['royalblue','goldenrod','firebrick','forestgreen','darkorange','black',\
          'rebeccapurple','aqua','yellowgreen'] 

fig, ax = plt.subplots(figsize=(14, 12), dpi=80, facecolor='w', edgecolor='w') 
plt.title('Gliders During Hurricane Season 2019',fontsize=24)
ax.set_facecolor('lightgrey')

for i, id in enumerate(gliders):
    if id[0:3] != 'all':
        print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        index_col='time (UTC)',
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        if fund_agency[i] == 'NOAA':
            h0 = ax.plot(df.index,np.tile(i,len(df.index)),'s',markersize = 10,\
                    color=color_fund1[0],markeredgewidth=0.1,markeredgecolor=color_fund1[0],zorder=0)
        if fund_agency[i] == 'Navy':
            h1 = ax.plot(df.index,np.tile(i,len(df.index)),'s',markersize = 10,\
                    color=color_fund1[1],markeredgewidth=0.1,markeredgecolor=color_fund1[1],zorder=0)
        if fund_agency[i] == 'NSF':
            h2 = ax.plot(df.index,np.tile(i,len(df.index)),'s',markersize = 10,\
                    color=color_fund1[2],markeredgewidth=0.1,markeredgecolor=color_fund1[2],zorder=0)
        if fund_agency[i] == 'NYDEC':
            h3 = ax.plot(df.index,np.tile(i,len(df.index)),'s',markersize = 10,\
                    color=color_fund1[3],markeredgewidth=0.1,markeredgecolor=color_fund1[3],zorder=0)
        if fund_agency[i] == 'NJDEP':
            h4 = ax.plot(df.index,np.tile(i,len(df.index)),'s',markersize = 10,\
                    color=color_fund1[4],markeredgewidth=0.1,markeredgecolor=color_fund1[4],zorder=0)
        if fund_agency[i] == 'Simmons':
            h5 = ax.plot(df.index,np.tile(i,len(df.index)),'s',markersize = 10,\
                    color=color_fund1[5],markeredgewidth=0.1,markeredgecolor=color_fund1[5],zorder=0)
        if fund_agency[i] == 'Shell':
            h6 = ax.plot(df.index,np.tile(i,len(df.index)),'s',markersize = 10,\
                    color=color_fund1[6],markeredgewidth=0.1,markeredgecolor=color_fund1[6],zorder=0)
        if fund_agency[i] == 'TWR':
            h7 = ax.plot(df.index,np.tile(i,len(df.index)),'s',markersize = 10,\
                    color=color_fund1[7],markeredgewidth=0.1,markeredgecolor=color_fund1[7],zorder=0)
        if fund_agency[i] == 'Vetlesen':
            h8 = ax.plot(df.index,np.tile(i,len(df.index)),'s',markersize = 10,\
                    color=color_fund1[8],markeredgewidth=0.1,markeredgecolor=color_fund1[8],zorder=0)
           
glider = [l.split('-')[0] for l in gliders]
#glider = [l for l in gliders]
ax.set_yticks(np.arange(len(glider)))
plt.tick_params(labelsize=20)
ax.plot(np.tile(datetime(2019,8,24,0,0,0),len(gliders)+2),np.arange(-1,len(gliders)+1),'k')
ax.plot(np.tile(datetime(2019,9,7,0,0,0),len(gliders)+2),np.arange(-1,len(gliders)+1),'k')
ax.plot(np.tile(datetime(2019,9,20,0,0,0),len(gliders)+2),np.arange(-1,len(gliders)+1),'k')
ax.plot(np.tile(datetime(2019,9,27,0,0,0),len(gliders)+2),np.arange(-1,len(gliders)+1),'k')
ax.legend([h0[0],h1[0],h2[0],h3[0],h4[0],h5[0],h6[0],h7[0],h8[0]],\
          ['NOAA - '+str(n_noaa),'Navy - '+str(n_navy),'NSF - '+str(n_nsf),\
         'NYDEC - '+str(n_nydec),'NJDEP - '+str(n_njdep),\
         'Simmons - '+str(n_simmons),'Shell - '+str(n_shell),\
         'TWR - '+str(n_twr),'Vetlesen - '+str(n_vetlesen)],\
          loc='center left',fontsize=20,bbox_to_anchor=(0, 0.5))

xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
ax.set_xlabel('2019 Date (DD-Month UTC)',fontsize=24)
#plt.grid(color='k', linestyle='--', linewidth=1)
ax.set_ylim(-1,len(glider))

ax.grid(True)
plt.grid(color='k', linestyle='--', linewidth=1)

wd = datetime(2019,8,24,0,0,0) - datetime(2019,9,7,0,0,0)
rect = plt.Rectangle((datetime(2019,9,7,0,0,0),-1), wd, len(glider)+1, color='k', alpha=0.3, zorder=10)
ax.add_patch(rect)

wd = datetime(2019,9,20,0,0,0) - datetime(2019,9,27,0,0,0)
rect = plt.Rectangle((datetime(2019,9,27,0,0,0),-1), wd, len(glider)+1, color='k', alpha=0.3, zorder=10)
ax.add_patch(rect)

ax.set_yticklabels(glider,fontsize=14)
   
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Time_window_gliders_hurric_season_2019.png"\
            ,bbox_inches = 'tight',pad_inches = 0.1)
#plt.show() 

#%% Total number of profiles

n_profiles = []
profiles = pd.DataFrame(columns=['# profiles'],index=gliders)
variables = ['latitude','longitude','time']

for i, id in enumerate(gliders):
    if id[0:3] != 'all':
        #print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        index_col='time (UTC)',
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        n_profiles.append(len(df.index))
        profiles['# profiles'][i] = len(df.index)

# Correction just for Ramses and Pelagia
#profiles['# profiles'][profiles.index=='ramses-20180704T0000'] = 5748
#profiles['# profiles'][profiles.index=='ramses-20180907T0000'] = 8473        
#profiles['# profiles'][profiles.index=='pelagia-20180910T0000'] = 875
#profiles['# profiles'][profiles.index=='ng309-20180701T0000'] = 1331
total_profiles = profiles['# profiles'].sum()                         
                                                            
#%% Total number of profiles for each funding agency
nprof_navy = 0
nprof_noaa = 0
nprof_nsf = 0
nprof_nydec = 0
nprof_njdep = 0
nprof_simmons = 0
nprof_shell = 0
nprof_twr = 0
nprof_vetlesen = 0
for i,glid in enumerate(profiles.index):
    if fund_agency[i] == 'Navy':
        print('YES',glid)
        nprof_navy += profiles['# profiles'][i]
    if fund_agency[i] == 'NOAA':
        print('YES',glid)
        nprof_noaa += profiles['# profiles'][i]
    if fund_agency[i] == 'NSF':
        print('YES',glid)
        nprof_nsf += profiles['# profiles'][i]                           
    if fund_agency[i] == 'NYDEC':
        print('YES',glid)
        nprof_nydec += profiles['# profiles'][i]                           
    if fund_agency[i] == 'NJDEP':
        print('YES',glid)
        nprof_njdep += profiles['# profiles'][i]                           
    if fund_agency[i] == 'Simmons':
        print('YES',glid)
        nprof_simmons += profiles['# profiles'][i]  
    if fund_agency[i] == 'Shell':
        print('YES',glid)
        nprof_shell += profiles['# profiles'][i]                              
    if fund_agency[i] == 'TWR':
        print('YES',glid)
        nprof_twr += profiles['# profiles'][i]
    if fund_agency[i] == 'Vetlesen':
        print('YES',glid)
        nprof_vetlesen += profiles['# profiles'][i]
                              
#%% Pie chart of number of profiles in each category

labels = 'NOAA - '+str(nprof_noaa),'Navy - '+str(nprof_navy),'NSF - '+str(nprof_nsf),\
         'NYDEC - '+str(nprof_nydec),'NJDEP - '+str(nprof_njdep),\
         'Simmons - '+str(nprof_simmons),'Shell - '+str(nprof_shell),\
         'TWR - '+str(nprof_twr),'Vetlesen - '+str(nprof_vetlesen)
sizes = [nprof_noaa,nprof_navy,nprof_nsf,nprof_nydec,nprof_njdep,\
         nprof_simmons,nprof_shell,nprof_twr,nprof_vetlesen]
colors = ['royalblue','goldenrod','firebrick','forestgreen',\
          'darkorange','black','rebeccapurple','aqua','yellowgreen']  
#explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

plt.figure()
patches, texts = plt.pie(sizes,startangle=90,colors=colors) #,autopct='%2d')
plt.legend(patches,labels,loc='best',bbox_to_anchor=(0.85, 1))
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Total Number of Profiles = '+str(total_profiles),fontsize=16)

plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/pie_chart_number_profiles_hurrica_season_2019.png"\
            ,bbox_inches = 'tight',pad_inches = 0.0)
                                            
