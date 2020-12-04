#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 13:38:10 2020

@author: aristizabal
"""

#%% User input

# lat and lon bounds
lon_lim = [-100.0,0.0]
lat_lim = [15.0,45.0]

# Time bounds
min_time = '2020-06-01T00:00:00Z'
max_time = '2020-11-30T00:00:00Z'

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
#from datetime import datetime
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
gliders = np.concatenate((gliders[0:14],gliders[15:])) 

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
        
#%%
        
glider = [l.split('-')[0] for l in gliders]
#glider = np.concatenate((glider[0:14],glider[15:]))

fund_agency = [None]*(len(glider))
for i,l in enumerate(glider):
    if glider[i][0:2] == 'ng':
        fund_agency[i] = 'Navy'
    if glider[i][0:2] == 'SG':
        fund_agency[i] = 'NOAA'
    if glider[i][0:2] == 'sp':
        fund_agency[i] = 'NOAA'
    if glider[i][0:2] == 'cp':
        fund_agency[i] = 'NSF'
    if glider[i][0:2] == 'ud':
        fund_agency[i] = 'NOAA'
    if glider[i] == 'amelia':
        fund_agency[i] = 'NOAA'
    if glider[i] == 'bios_jack':
        fund_agency[i] = 'NSF'
    if glider[i] == 'blue':
        fund_agency[i] = 'NOAA'
    if glider[i] == 'franklin':
        fund_agency[i] = 'NOAA'
    if glider[i] == 'maracoos_01':
        fund_agency[i] = 'NOAA'
    if glider[i] == 'mote':
        fund_agency[i] = 'NOAA'
    if glider[i] == 'pelagia':
        fund_agency[i] = 'NOAA'
    if glider[i] == 'ru28':
        fund_agency[i] = 'NJDEP'
    if glider[i] == 'ru29':
        fund_agency[i] = 'Vetlesen'
    if glider[i] == 'ru33':
        fund_agency[i] = 'NOAA'
    if glider[i] == 'ru34':
        fund_agency[i] = 'Ocean Wind'
    if glider[i] == 'sam':
        fund_agency[i] = 'NOAA'
    if glider[i] == 'stella':
        fund_agency[i] = 'FL + NOAA'
    if glider[i] == 'Stommel':
        fund_agency[i] = 'NOAA'
    if glider[i] == 'Sverdrup':
        fund_agency[i] = 'NOAA'
    if glider[i] == 'sylvia':
        fund_agency[i] = 'NOAA'
    if glider[i] == 'usf':
        fund_agency[i] = 'FL + NOAA'  
    
#%% Glider in each category

n_navy = len([i for i,list in enumerate(fund_agency) if list == 'Navy'])
n_noaa = len([i for i,list in enumerate(fund_agency) if list == 'NOAA'])
n_nsf = len([i for i,list in enumerate(fund_agency) if list == 'NSF'])
n_njdep = len([i for i,list in enumerate(fund_agency) if list == 'NJDEP'])
n_vetlesen = len([i for i,list in enumerate(fund_agency) if list == 'Vetlesen'])
n_oceanwind = len([i for i,list in enumerate(fund_agency) if list == 'Ocean Wind'])
n_flnoaa = len([i for i,list in enumerate(fund_agency) if list == 'FL + NOAA'])

#%% Pie chart of number of gliders in each category

labels = 'NOAA - '+str(n_noaa),'Navy - '+str(n_navy),'NSF - '+str(n_nsf),\
         'NJDEP - '+str(n_njdep),'Vetlesen - '+str(n_vetlesen),\
         'Ocean Wind - '+str(n_oceanwind), 'FL + NOAA - '+str(n_flnoaa)  
             
siz = [n_noaa,n_navy,n_nsf,n_njdep,n_vetlesen,n_oceanwind,n_flnoaa]
sizes = np.ndarray.tolist(np.multiply(siz,1/np.sum(siz)))
colors = ['royalblue','goldenrod','firebrick','forestgreen','darkorange','black',\
          'rebeccapurple','aqua','yellowgreen'] 

plt.figure()
patches, texts = plt.pie(sizes,startangle=90,colors=colors) #,autopct='%2d')
plt.legend(patches,labels,loc='best',bbox_to_anchor=(0.85, 1))
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Total Number of Gliders = '+str(len(glider)),fontsize=16)

plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/pie_chart_number_gliders_hurica_season_2020.png"\
            ,bbox_inches = 'tight',pad_inches = 0.1)

#%% Plotting the deployment window of all glider in Hurricane season 
      
siz=12

funding = list(set(fund_agency))
color_fund1 = ['royalblue','goldenrod','firebrick','forestgreen','darkorange','black',\
          'rebeccapurple','aqua','yellowgreen'] 

fig, ax = plt.subplots(figsize=(14, 12), dpi=80, facecolor='w', edgecolor='w') 
plt.title('Gliders During Hurricane Season 2020',fontsize=24)
ax.set_facecolor('lightgrey')

for i, id in enumerate(gliders):
    if id[0:4] != 'glos':
        print(id)
        e.dataset_id = id
        e.constraints = constraints
        e.variables = variables
    
        df = e.to_pandas(
        index_col='time (UTC)',parse_dates=True).dropna()
        df.index = mdates.date2num(df.index)
        if fund_agency[i] == 'NOAA':
            h0 = ax.plot(df.index,np.tile(i,len(df.index)),'s',markersize = 10,\
                    color=color_fund1[0],markeredgewidth=0.1,markeredgecolor=color_fund1[0],zorder=0)
        if fund_agency[i] == 'Navy':
            h1 = ax.plot(df.index,np.tile(i,len(df.index)),'s',markersize = 10,\
                    color=color_fund1[1],markeredgewidth=0.1,markeredgecolor=color_fund1[1],zorder=0)
        if fund_agency[i] == 'NSF':
            h2 = ax.plot(df.index,np.tile(i,len(df.index)),'s',markersize = 10,\
                    color=color_fund1[2],markeredgewidth=0.1,markeredgecolor=color_fund1[2],zorder=0)
        if fund_agency[i] == 'NJDEP':
            h3 = ax.plot(df.index,np.tile(i,len(df.index)),'s',markersize = 10,\
                    color=color_fund1[3],markeredgewidth=0.1,markeredgecolor=color_fund1[3],zorder=0)
        if fund_agency[i] == 'Vetlesen':
            h4 = ax.plot(df.index,np.tile(i,len(df.index)),'s',markersize = 10,\
                    color=color_fund1[4],markeredgewidth=0.1,markeredgecolor=color_fund1[4],zorder=0)
        if fund_agency[i] == 'Ocean Wind':
            h5 = ax.plot(df.index,np.tile(i,len(df.index)),'s',markersize = 10,\
                    color=color_fund1[5],markeredgewidth=0.1,markeredgecolor=color_fund1[5],zorder=0)
        if fund_agency[i] == 'FL + NOAA':
            h6 = ax.plot(df.index,np.tile(i,len(df.index)),'s',markersize = 10,\
                    color=color_fund1[6],markeredgewidth=0.1,markeredgecolor=color_fund1[6],zorder=0)
 
ax.legend([h0[0],h1[0],h2[0],h3[0],h4[0],h5[0],h6[0]],\
          ['NOAA - '+str(n_noaa),'Navy - '+str(n_navy),'NSF - '+str(n_nsf),\
         'NJDEP - '+str(n_njdep),'Vetlesen - '+str(n_vetlesen),\
         'Ocean Wind - '+str(n_oceanwind), 'FL + NOAA - '+str(n_flnoaa)],\
          loc='center left',fontsize=20,bbox_to_anchor=(0, 0.5))

glider = [l.split('-')[0] for l in gliders]
ax.set_yticks(np.arange(len(glider)))
ax.set_ylim(-1,len(glider))
ax.set_yticklabels(glider,fontsize=14)

ax.set_xlabel('2020 Date (DD-Month UTC)',fontsize=24)
xfmt = mdates.DateFormatter('%d-%b')
ax.xaxis.set_major_formatter(xfmt)   

plt.tick_params(labelsize=12)
   
ax.grid(True)
plt.grid(color='k', linestyle='--', linewidth=1)
   
plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Time_window_gliders_hurric_season_2020.png"\
            ,bbox_inches = 'tight',pad_inches = 0.1)

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

total_profiles = profiles['# profiles'].sum()                         
                                                            
#%% Total number of profiles for each funding agency
nprof_navy = 0
nprof_noaa = 0
nprof_nsf = 0
nprof_njdep = 0
nprof_vetlesen = 0
nprof_oceanwind = 0
nprof_flnoaa = 0

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
    if fund_agency[i] == 'NJDEP':
        print('YES',glid)
        nprof_njdep += profiles['# profiles'][i]                           
    if fund_agency[i] == 'Vetlesen':
        print('YES',glid)
        nprof_vetlesen += profiles['# profiles'][i]
    if fund_agency[i] == 'Ocean Wind':
        print('YES',glid)
        nprof_oceanwind += profiles['# profiles'][i]
    if fund_agency[i] == 'FL + NOAA':
        print('YES',glid)
        nprof_flnoaa += profiles['# profiles'][i]
                              
#%% Pie chart of number of profiles in each category

labels = 'NOAA - '+str(nprof_noaa),'Navy - '+str(nprof_navy),'NSF - '+str(nprof_nsf),\
         'NJDEP - '+str(nprof_njdep),'Vetlesen - '+str(nprof_vetlesen),\
         'Ocean Wind - '+str(nprof_oceanwind),'FL + NOAA - '+str(nprof_flnoaa)    
sizes = [nprof_noaa,nprof_navy,nprof_nsf,nprof_njdep,nprof_vetlesen,\
         nprof_oceanwind,nprof_flnoaa]
colors = ['royalblue','goldenrod','firebrick','forestgreen',\
          'darkorange','black','rebeccapurple','aqua','yellowgreen']  
#explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

plt.figure()
patches, texts = plt.pie(sizes,startangle=90,colors=colors) #,autopct='%2d')
plt.legend(patches,labels,loc='best',bbox_to_anchor=(0.85, 1))
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Total Number of Profiles = '+str(total_profiles),fontsize=16)

plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/pie_chart_number_profiles_hurrica_season_2020.png"\
            ,bbox_inches = 'tight',pad_inches = 0.0)
                                            
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
ndays_njdep = 0
ndays_vetlesen = 0
ndays_oceanwind = 0
ndays_flnoaa = 0
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
    if fund_agency[i] == 'NJDEP':
        print('YES',glid)
        ndays_njdep += days['# days'][i]                          
    if fund_agency[i] == 'Vetlesen':
        print('YES',glid)
        ndays_vetlesen += days['# days'][i]                              
    if fund_agency[i] == 'Ocean Wind':
        print('YES',glid)
        ndays_oceanwind += days['# days'][i]                                 
    if fund_agency[i] == 'FL + NOAA':
        print('YES',glid)
        ndays_flnoaa += days['# days'][i]   
                              
#%% Pie chart of number of days in each category

labels = 'NOAA - '+str(ndays_noaa),'Navy - '+str(ndays_navy),'NSF - '+str(ndays_nsf),\
         'NJDEP - '+str(ndays_njdep),'Vetlesen - '+str(ndays_vetlesen),\
         'Ocean Wind - '+str(ndays_oceanwind),'FL + NOAA - '+str(ndays_flnoaa)    
sizes = [ndays_noaa,ndays_navy,ndays_nsf,ndays_njdep,ndays_vetlesen,\
         ndays_oceanwind,ndays_flnoaa]
colors = ['royalblue','goldenrod','firebrick','forestgreen',\
          'darkorange','black','rebeccapurple','aqua','yellowgreen']  

plt.figure()
patches, texts = plt.pie(sizes,startangle=90,colors=colors) #,autopct='%2d')
plt.legend(patches,labels,loc='best',bbox_to_anchor=(0.85, 1))
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Total Number of Days = '+str(total_days),fontsize=16)

plt.savefig("/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/pie_chart_number_days_hurrica_season_2020.png"\
            ,bbox_inches = 'tight',pad_inches = 0.0)                           

#%% Figure total number of days
'''
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
'''