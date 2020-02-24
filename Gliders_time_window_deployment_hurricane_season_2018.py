#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:06:33 2018

@author: aristizabal
"""

#%% User input

# lat and lon bounds
lon_lim = [-110.0,-10.0]
lat_lim = [15.0,45.0]

# Time bounds
min_time = '2018-06-01T00:00:00Z'
max_time = '2018-11-30T00:00:00Z'

# Time bounds
#min_time = '2019-06-01T00:00:00Z'
#max_time = '2019-09-26T00:00:00Z'

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
# get rid off glos gliders (great lakes ocean observing)
gliders = np.concatenate((gliders[0:6],gliders[8:])) 

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
#%%
e.dataset_id=gliders[5]
e.constraints=constraints
e.variables=variables
    
df = e.to_pandas(
        index_col='time (UTC)',
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
                    ).dropna()    


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
    if glider[i][0:2] == 'cp':
        fund_agency[i] = 'NSF'
    if glider[i][0:2] == 'sp':
        fund_agency[i] = 'NOAA'
    if glider[i][0:2] == 'SG':
        fund_agency[i] = 'NOAA'
    if glider[i][0:4] == 'ru28':
        fund_agency[i] = 'NJ'

ok = [i for i, l in enumerate(glider) if l == 'bass']
fund_agency[ok[0]] = 'NOAA'

ok = [i for i, l in enumerate(glider) if l == 'bios_minnie']
fund_agency[ok[0]] = 'BIOS'

ok = [i for i, l in enumerate(glider) if l == 'blue']
fund_agency[ok[0]] = 'NOAA'
     
ok = [i for i, l in enumerate(glider) if l == 'blue']
fund_agency[ok[0]] = 'NOAA'

ok = [i for i, l in enumerate(glider) if l == 'pelagia']
fund_agency[ok[0]] = 'NSF'

ok = [i for i, l in enumerate(gliders) if l == 'ramses-20180704T0000']
fund_agency[ok[0]] = 'NOAA'

ok = [i for i, l in enumerate(gliders) if l == 'ramses-20180907T0000']
fund_agency[ok[0]] = 'NSF'

ok = [i for i, l in enumerate(glider) if l == 'Reveille']
fund_agency[ok[0]] = 'NOAA'

ok = [i for i, l in enumerate(glider) if l == 'ru30']
fund_agency[ok[0]] = 'NSF'

ok = [i for i, l in enumerate(glider) if l == 'ru33']
fund_agency[ok[0]] = 'NOAA'

ok = [i for i, l in enumerate(glider) if l == 'sam']
fund_agency[ok[0]] = 'FL'

ok = [i for i, l in enumerate(glider) if l == 'Sverdrup']
fund_agency[ok[0]] = 'NOAA'

ok = [i for i, l in enumerate(glider) if l == 'sylvia']
fund_agency[ok[0]] = 'NOAA'

ok = [i for i, l in enumerate(glider) if l == 'ud_476']
fund_agency[ok[0]] = 'NOAA'

ok = [i for i, l in enumerate(glider) if l == 'usf']
fund_agency[ok[0]] = 'NOAA'

ok = [i for i, l in enumerate(glider) if l == 'silbo']
fund_agency[ok[0]] = 'TWR'

#%% Glider in each category

n_navy = len([i for i,list in enumerate(fund_agency) if list == 'Navy'])
n_noaa = len([i for i,list in enumerate(fund_agency) if list == 'NOAA'])
n_nsf = len([i for i,list in enumerate(fund_agency) if list == 'NSF'])
n_nj = len([i for i,list in enumerate(fund_agency) if list == 'NJ'])
n_fl = len([i for i,list in enumerate(fund_agency) if list == 'FL'])
n_bios = len([i for i,list in enumerate(fund_agency) if list == 'BIOS'])
n_twr = len([i for i,list in enumerate(fund_agency) if list == 'TWR'])

#%% Pie chart of number of gliders in each category

labels = 'Navy - '+str(n_navy), 'NOAA - '+str(n_noaa), 'NSF -'+str(n_nsf),\
    'NJ - '+str(n_nj), 'FL - '+str(n_fl), 'BIOS - '+str(n_bios),\
        'TWR - '+str(n_twr)
siz = [n_navy,n_noaa,n_nsf,n_nj,n_fl,n_bios,n_twr]
sizes = np.ndarray.tolist(np.multiply(siz,1/np.sum(siz)))
colors = ['goldenrod','royalblue','firebrick','forestgreen',\
          'darkorange','black','rebeccapurple'] 
#explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

plt.figure()
patches, texts = plt.pie(sizes,startangle=90,colors=colors) #,autopct='%2d')
plt.legend(patches,labels,loc='best',bbox_to_anchor=(0.85, 1))
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Total Number of Gliders = '+str(len(glider)),fontsize=16)

plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/pie_chart_number_gliders_hurica_season_2018.png"\
            ,bbox_inches = 'tight',pad_inches = 0.1)
      
#%% Plotting the deployment window of all glider in Hurricane season 2018 
      
siz=12

funding = list(set(fund_agency))
color_fund1 = ['goldenrod','royalblue','firebrick','forestgreen','darkorange','black','rebeccapurple'] 
color_fund2 = ['navy','deepskyblue','indianred','mediumseagreen','sandybrown','dimgrey','teal'] 

fig, ax = plt.subplots(figsize=(14, 12), dpi=80, facecolor='w', edgecolor='w') 
plt.title('Gliders During Hurricane Season 2018',fontsize=24)
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
        if fund_agency[i] == 'Navy':
            h0 = ax.plot(df.index,np.tile(i,len(df.index)),'s',markersize = 10,\
                    color=color_fund1[0],markeredgewidth=0.1,markeredgecolor=color_fund1[0],zorder=0)
        if fund_agency[i] == 'NOAA':
            h1 = ax.plot(df.index,np.tile(i,len(df.index)),'s',markersize = 10,\
                    color=color_fund1[1],markeredgewidth=0.1,markeredgecolor=color_fund1[1],zorder=0)
        if fund_agency[i] == 'NSF':
            h2 = ax.plot(df.index,np.tile(i,len(df.index)),'s',markersize = 10,\
                    color=color_fund1[2],markeredgewidth=0.1,markeredgecolor=color_fund1[2],zorder=0)
        if fund_agency[i] == 'NJ':
            h3 = ax.plot(df.index,np.tile(i,len(df.index)),'s',markersize = 10,\
                    color=color_fund1[3],markeredgewidth=0.1,markeredgecolor=color_fund1[3],zorder=0)
        if fund_agency[i] == 'Fl':
            h4 = ax.plot(df.index,np.tile(i,len(df.index)),'s',markersize = 10,\
                    color=color_fund1[4],markeredgewidth=0.1,markeredgecolor=color_fund1[4],zorder=0)
        if fund_agency[i] == 'BIOS':
            h5 = ax.plot(df.index,np.tile(i,len(df.index)),'s',markersize = 10,\
                    color=color_fund1[5],markeredgewidth=0.1,markeredgecolor=color_fund1[5],zorder=0)
        if fund_agency[i] == 'TWR':
            h6 = ax.plot(df.index,np.tile(i,len(df.index)),'s',markersize = 10,\
                    color=color_fund1[6],markeredgewidth=0.1,markeredgecolor=color_fund1[6],zorder=0)
           
glider = [l.split('-')[0] for l in gliders]
ax.set_yticks(np.arange(len(glider)))
plt.tick_params(labelsize=20)
ax.plot(np.tile(datetime(2018,9,11,18,0,0),len(gliders)+2),np.arange(-1,len(gliders)+1),'k')
ax.plot(np.tile(datetime(2018,9,13,18,0,0),len(gliders)+2),np.arange(-1,len(gliders)+1),'k')
ax.plot(np.tile(datetime(2018,10,8,15,0,0),len(gliders)+2),np.arange(-1,len(gliders)+1),'k')
ax.plot(np.tile(datetime(2018,10,10,18,0,0),len(gliders)+2),np.arange(-1,len(gliders)+1),'k')
ax.legend([h0[0],h1[0],h2[0],h3[0],h4[0],h5[0],h6[0]],['Navy - 30','NOAA - 21','NSF - 6','NJ - 2','FL - 1','BIOS - 1','TWR - 1'],\
          loc='center left',fontsize=20,bbox_to_anchor=(0, 0.4))

xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
ax.set_xlabel('2018 Date (DD-Month UTC)',fontsize=24)
#plt.grid(color='k', linestyle='--', linewidth=1)
ax.set_ylim(-1,len(glider))

ax.grid(True)
plt.grid(color='k', linestyle='--', linewidth=1)

wd = datetime(2018,9,13,18,0,0) - datetime(2018,9,11,18,0,0)
rect = plt.Rectangle((datetime(2018,9,11,18,8,0),-1), wd, len(glider)+1, color='k', alpha=0.3, zorder=10)
ax.add_patch(rect)

wd = datetime(2018,10,10,18,0,0) - datetime(2018,10,8,15,0,0)
rect = plt.Rectangle((datetime(2018,10,8,18,0,0),-1), wd, len(glider)+1, color='k', alpha=0.3, zorder=10)
ax.add_patch(rect)

ax.set_yticklabels(glider,fontsize=14)
   
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Time_window_gliders_hurric_season_2018.png"\
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
profiles['# profiles'][profiles.index=='ramses-20180704T0000'] = 5748
profiles['# profiles'][profiles.index=='ramses-20180907T0000'] = 8473        
profiles['# profiles'][profiles.index=='pelagia-20180910T0000'] = 875
profiles['# profiles'][profiles.index=='ng309-20180701T0000'] = 1331
total_profiles = profiles['# profiles'].sum()                         
                                                            
#%% Total number of profiles for each funding agency
nprof_navy = 0
nprof_noaa = 0
nprof_nsf = 0
nprof_nj = 0
nprof_fl = 0
nprof_bios = 0
nprof_twr = 0
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
    if fund_agency[i] == 'NJ':
        print('YES',glid)
        nprof_nj += profiles['# profiles'][i]                           
    if fund_agency[i] == 'FL':
        print('YES',glid)
        nprof_fl += profiles['# profiles'][i]                           
    if fund_agency[i] == 'BIOS':
        print('YES',glid)
        nprof_bios += profiles['# profiles'][i]                              
    if fund_agency[i] == 'TWR':
        print('YES',glid)
        nprof_twr += profiles['# profiles'][i]
                              
#%% Pie chart of number of profiles in each category

labels = 'Navy', 'NOAA', 'NSF', 'NJ', 'FL', 'BIOS', 'TWR'
sizes = [nprof_navy,nprof_noaa,nprof_nsf,nprof_nj,nprof_fl,nprof_bios,nprof_twr]
colors = ['goldenrod','royalblue','firebrick','forestgreen','darkorange','black','rebeccapurple'] 
#explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

plt.figure()
patches, texts = plt.pie(sizes,startangle=90,colors=colors) #,autopct='%2d')
#plt.legend(patches,labels,loc='best',bbox_to_anchor=(0.85, 1))
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/pie_chart_number_profiles_hurrica_season_2018.png"\
            ,bbox_inches = 'tight',pad_inches = 0.0)
                                            
#%% Total number of profiles with depths > 5 m

'''                         
n_profiles_long = []
profiles_long = pd.DataFrame(columns=['# profiles long'],index=gliders)
constraints = {
    'time>=': '2018-06-01T00:00:00Z',
    'time<=': '2018-11-30T00:00:00Z',
    'latitude>=': 15.0,
    'latitude<=': 45.0,
    'longitude>=': -100.0,
    'longitude<=': -10.0,
}                                                                            
variables = ['latitude','longitude','time','depth']

for i, id in enumerate(gliders):
    if id[0:3] != 'all':
        print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        index_col='time',
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        
        time_all_prof = df.index.map(lambda x: x).unique()
        
        n_prof_long = 0
        time_all_prof = df.index.map(lambda x: x).unique()
        for time_prof in time_all_prof: 
            if df['depth'][df.index.map(lambda x: x==time_prof)].max() > 5.0:
                n_prof_long += 1
        n_profiles_long.append(n_prof_long) 
        profiles_long['# profiles long'][i] = n_prof_long

total_profiles_long = profiles_long['# profiles long'].sum()                          
                          
'''

#%%

variables = ['latitude','longitude','time','depth','temperature']
#variables = ['latitude','longitude','time','depth']
#variables = ['latitude','longitude','time']

#id = 'ramses-20180907T0000'
#id = 'pelagia-20180910T0000'
#id ='bass-20180808T0000'
#id = 'bios_minnie-20180523T1617'
id = 'ng309-20180701T0000'
e.dataset_id = id
e.constraints = constraints
e.variables = variables
    
df = e.to_pandas(
    index_col='time (UTC)',
    parse_dates=True,
    skiprows=(1,)  # units information can be dropped.
    ).dropna()  

n_profiles_long = 0
max_depth_prof = []
num_depths_prof = []
time_all_prof = df.index.map(lambda x: x).unique()
for i, time_prof in enumerate(time_all_prof): 
    if df['depth'][df.index.map(lambda x: x==time_prof)].max() > 5.0:
    #if df['depth'][df.index.map(lambda x: x==time_prof)].max() > 50.0:
        max_depth_prof.append(df['depth'][df.index.map(lambda x: x==time_prof)].max())
        num_depths_prof.append(len(df['depth'][df.index.map(lambda x: x==time_prof)]))
        n_profiles_long += 1
       
        
#%%

fig, ax = plt.subplots(figsize=(12, 5))
kw = dict(s=15, c=df['temperature'], marker='o', edgecolor='none')
cs = ax.scatter(df.index, df['depth'], **kw, cmap='RdYlBu_r')

ax.invert_yaxis()
ax.set_xlim(df.index[0], df.index[-1])
xfmt = mdates.DateFormatter('%H:%Mh\n%d-%b')
ax.xaxis.set_major_formatter(xfmt)

cbar = fig.colorbar(cs, orientation='vertical', extend='both')
cbar.ax.set_ylabel('Temperature ($^\circ$C)')
ax.set_ylabel('Depth (m)');
ax.grid(True) 
ax.set_xlim(df.index.unique()[1280],df.index.unique()[-1])    

      
        
#%% Figure total number of profiles

siz=12

funding = list(set(fund_agency))
color_fund1 = ['goldenrod','royalblue','firebrick','forestgreen','darkorange','black','rebeccapurple'] 

fig, ax = plt.subplots(figsize=(14, 12), dpi=80, facecolor='w', edgecolor='w') 
plt.title('Total Number of Profiles = ' + str(total_profiles),fontsize=24)
ax.set_facecolor('lightgrey')
plt.xlabel('Number of Profiles',fontsize = 24)
ax.grid(True)
plt.grid(color='k', linestyle='--', linewidth=1)
ax.set_ylim(-1,len(glider))

glider = [l.split('-')[0] for l in gliders]
ax.set_yticks(np.arange(len(glider)))
plt.tick_params(labelsize=20)

p1 = plt.barh(np.arange(len(glider)),profiles['# profiles'])
              
for i, id in enumerate(gliders):
    if id[0:3] != 'all':
    
        if fund_agency[i] == 'Navy':
            h0 = plt.barh(i,profiles['# profiles'][i],color=color_fund1[0])
        if fund_agency[i] == 'NOAA':
            h1 = plt.barh(i,profiles['# profiles'][i],color=color_fund1[1])
        if fund_agency[i] == 'NSF':
            h2 = plt.barh(i,profiles['# profiles'][i],color=color_fund1[2])
        if fund_agency[i] == 'NJ':
            h3 = plt.barh(i,profiles['# profiles'][i],color=color_fund1[3])
        if fund_agency[i] == 'Fl':
            h4 = plt.barh(i,profiles['# profiles'][i],color=color_fund1[4])
        if fund_agency[i] == 'BIOS':
            h5 = plt.barh(i,profiles['# profiles'][i],color=color_fund1[5])
        if fund_agency[i] == 'TWR':
            h6 = plt.barh(i,profiles['# profiles'][i],color=color_fund1[6])          

ax.set_yticklabels(glider,fontsize=14)              
ax.legend([h0[0],h1[0],h2[0],h3[0],h4[0],h5[0],h6[0]],['Navy','NOAA','NSF','NJ','FL','BIOS','TWR'],\
          loc='center right',fontsize=20)

plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Number_profiles_hurric_season_2018.png"\
            ,bbox_inches = 'tight',pad_inches = 0.1)  

#%% Total number of days

n_days = []
days = pd.DataFrame(columns=['# days'],index=gliders)

for i, id in enumerate(gliders):
    if id[0:3] != 'all':
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        index_col='time (UTC)',
        parse_dates=True,
        skiprows=(1,)  # units information can be dropped.
            ).dropna()
        #dt = df.index[-1] - df.index[0]
        #days['# days'][i] = dt.days
        n_days.append(len(df.index.map(lambda x: x.strftime('%Y-%m-%d')).unique()))
        days['# days'][i] = len(df.index.map(lambda x: x.strftime('%Y-%m-%d')).unique())  

total_days = days['# days'].sum()                 
                           
#%% Total number of days for each funding agency
        
ndays_navy = 0
ndays_noaa = 0
ndays_nsf = 0
ndays_nj = 0
ndays_fl = 0
ndays_bios = 0
ndays_twr = 0
for i,glid in enumerate(profiles.index):
    if fund_agency[i] == 'Navy':
        print('YES',glid)
        ndays_navy += days['# days'][i]
    if fund_agency[i] == 'NOAA':
        print('YES',glid)
        ndays_noaa += days['# days'][i]
    if fund_agency[i] == 'NSF':
        print('YES',glid)
        ndays_nsf += days['# days'][i]                        
    if fund_agency[i] == 'NJ':
        print('YES',glid)
        ndays_nj += days['# days'][i]                          
    if fund_agency[i] == 'FL':
        print('YES',glid)
        ndays_fl += days['# days'][i]                              
    if fund_agency[i] == 'BIOS':
        print('YES',glid)
        ndays_bios += days['# days'][i]                                 
    if fund_agency[i] == 'TWR':
        print('YES',glid)
        ndays_twr += days['# days'][i]   
                              
#%% Pie chart of number of profiles in each category

labels = 'Navy', 'NOAA', 'NSF', 'NJ', 'FL', 'BIOS', 'TWR'
sizes = [ndays_navy,ndays_noaa,ndays_nsf,ndays_nj,ndays_fl,ndays_bios,ndays_twr]
colors = ['goldenrod','royalblue','firebrick','forestgreen','darkorange','black','rebeccapurple'] 
#explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

plt.figure()
patches, texts = plt.pie(sizes,startangle=90,colors=colors) #,autopct='%2d')
#plt.legend(patches,labels,loc='best',bbox_to_anchor=(0.85, 1))
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/pie_chart_number_days_hurrica_season_2018.png"\
            ,bbox_inches = 'tight',pad_inches = 0.0)                           

#%% Figure total number of days

fig, ax = plt.subplots(figsize=(14, 12), dpi=80, facecolor='w', edgecolor='w') 
plt.title('Total Number of Days = ' + str(total_days),fontsize=24)
ax.set_facecolor('lightgrey')
plt.xlabel('Number of Days',fontsize = 24)
ax.grid(True)
plt.grid(color='k', linestyle='--', linewidth=1)
ax.set_ylim(-1,len(glider))

glider = [l.split('-')[0] for l in gliders]
ax.set_yticks(np.arange(len(glider)))
plt.tick_params(labelsize=20)

p1 = plt.barh(np.arange(len(glider)),days['# days'])
              
for i, id in enumerate(gliders):
    if id[0:3] != 'all':
    
        if fund_agency[i] == 'Navy':
            h0 = plt.barh(i,days['# days'][i],color=color_fund1[0])
        if fund_agency[i] == 'NOAA':
            h1 = plt.barh(i,days['# days'][i],color=color_fund1[1])
        if fund_agency[i] == 'NSF':
            h2 = plt.barh(i,days['# days'][i],color=color_fund1[2])
        if fund_agency[i] == 'NJ':
            h3 = plt.barh(i,days['# days'][i],color=color_fund1[3])
        if fund_agency[i] == 'Fl':
            h4 = plt.barh(i,days['# days'][i],color=color_fund1[4])
        if fund_agency[i] == 'BIOS':
            h5 = plt.barh(i,days['# days'][i],color=color_fund1[5])
        if fund_agency[i] == 'TWR':
            h6 = plt.barh(i,days['# days'][i],color=color_fund1[6])          

ax.set_yticklabels(glider,fontsize=14)              
ax.legend([h0[0],h1[0],h2[0],h3[0],h4[0],h5[0],h6[0]],['Navy','NOAA','NSF','NJ','FL','BIOS','TWR'],\
          loc=5,fontsize=20)

plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Number_days_hurric_season_2018.png"\
            ,bbox_inches = 'tight',pad_inches = 0.1)  

#%% Number of profiles per day per glider

n_profiles = profiles['# profiles']
n_days = days['# days']
n_profiles_per_day = np.asarray(n_profiles)/np.asarray(n_days)
n_profiles_per_day_aver = total_profiles/total_days

#%% Figure profiles per day per glider

fig, ax = plt.subplots(figsize=(14, 12), dpi=80, facecolor='w', edgecolor='w') 
plt.title('Average Number of Profiels per Days = ' + str(round(n_profiles_per_day_aver,1)),fontsize=20)
ax.set_facecolor('lightgrey')
plt.xlabel('Number of Profiles per Day',fontsize = 20)
ax.grid(True)
plt.grid(color='k', linestyle='--', linewidth=1)
ax.set_ylim(-1,len(glider))

glider = [l.split('-')[0] for l in gliders]
ax.set_yticks(np.arange(len(glider)))
ax.set_yticklabels(glider,fontsize=12)
plt.tick_params(labelsize=13)

p1 = plt.barh(np.arange(len(glider)),n_profiles_per_day)
              
for i, id in enumerate(gliders):
    if id[0:3] != 'all':
    
        if fund_agency[i] == 'Navy':
            h0 = plt.barh(i,n_profiles_per_day[i],color=color_fund1[0])
        if fund_agency[i] == 'NOAA':
            h1 = plt.barh(i,n_profiles_per_day[i],color=color_fund1[1])
        if fund_agency[i] == 'NSF':
            h2 = plt.barh(i,n_profiles_per_day[i],color=color_fund1[2])
        if fund_agency[i] == 'NJ':
            h3 = plt.barh(i,n_profiles_per_day[i],color=color_fund1[3])
        if fund_agency[i] == 'Fl':
            h4 = plt.barh(i,n_profiles_per_day[i],color=color_fund1[4])
        if fund_agency[i] == 'BIOS':
            h5 = plt.barh(i,n_profiles_per_day[i],color=color_fund1[5])
        if fund_agency[i] == 'TWR':
            h6 = plt.barh(i,n_profiles_per_day[i],color=color_fund1[6])          
              
ax.legend([h0[0],h1[0],h2[0],h3[0],h4[0],h5[0],h6[0]],['Navy','NOAA','NSF','NJ','FL','BIOS','TWR'],\
          loc=0,fontsize=16)

plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Number_profiles_per_days_hurric_season_2018.png"\
            ,bbox_inches = 'tight',pad_inches = 0) 

#%% Figure profiles per day per glider zoom-in 0 to 100 days

fig, ax = plt.subplots(figsize=(14, 12), dpi=80, facecolor='w', edgecolor='w') 
plt.title('Average Number of Profiels per Days = ' + str(round(n_profiles_per_day_aver,1)),fontsize=20)
ax.set_facecolor('lightgrey')
plt.xlabel('Number of Profiles per Day',fontsize = 20)
ax.grid(True)
plt.grid(color='k', linestyle='--', linewidth=1)
ax.set_ylim(-1,len(glider))
ax.set_xlim(0,100)

glider = [l.split('-')[0] for l in gliders]
ax.set_yticks(np.arange(len(glider)))
ax.set_yticklabels(glider,fontsize=12)
plt.tick_params(labelsize=13)

p1 = plt.barh(np.arange(len(glider)),n_profiles_per_day)
              
for i, id in enumerate(gliders):
    if id[0:3] != 'all':
    
        if fund_agency[i] == 'Navy':
            h0 = plt.barh(i,n_profiles_per_day[i],color=color_fund1[0])
        if fund_agency[i] == 'NOAA':
            h1 = plt.barh(i,n_profiles_per_day[i],color=color_fund1[1])
        if fund_agency[i] == 'NSF':
            h2 = plt.barh(i,n_profiles_per_day[i],color=color_fund1[2])
        if fund_agency[i] == 'NJ':
            h3 = plt.barh(i,n_profiles_per_day[i],color=color_fund1[3])
        if fund_agency[i] == 'Fl':
            h4 = plt.barh(i,n_profiles_per_day[i],color=color_fund1[4])
        if fund_agency[i] == 'BIOS':
            h5 = plt.barh(i,n_profiles_per_day[i],color=color_fund1[5])
        if fund_agency[i] == 'TWR':
            h6 = plt.barh(i,n_profiles_per_day[i],color=color_fund1[6])          
              
ax.legend([h0[0],h1[0],h2[0],h3[0],h4[0],h5[0],h6[0]],['Navy','NOAA','NSF','NJ','FL','BIOS','TWR'],\
          loc=0,fontsize=16)

plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Number_profiles_per_days_hurric_season_2018_detail.png"\
            ,bbox_inches = 'tight',pad_inches = 0) 
    
  #%%
'''
fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w') 
plt.title('Gliders During Hurricane Season 2018')

ax.plot(np.tile(datetime(2018,9,11),len(gliders)),np.arange(len(gliders)),'k')
ax.legend([h0[0],h1[0],h2[0],h3[0]],['Navy','NOAA','NSF'],fontsize =16)
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)
ax.set_xlabel('2018 Date (DD-Month UTC)',fontsize=16)

ax.set_ylim(0,60)
wd = datetime(2018,9,11) - datetime(2018,9,9)
rect = plt.Rectangle((datetime(2018,9,11),0), wd, 60, color='k', alpha=0.3)
ax.add_patch(rect)

e.dataset_id = id
e.constraints = constraints
e.variables = variables
    
df = e.to_pandas(
index_col='time',
parse_dates=True,
skiprows=(1,)  # units information can be dropped.
).dropna()
ax.plot(df.index,np.tile(i,len(df.index)),'s',markersize = 10,color=color_fund1[0],zorder=0)

ax.grid(True)
plt.grid(color='k', linestyle='--', linewidth=1)

'''