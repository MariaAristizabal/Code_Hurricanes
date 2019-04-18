#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 14:46:22 2018

@author: aristizabal
"""

#%%
#import urllib.request
from bs4 import BeautifulSoup
import requests
#import os
from datetime import datetime
import time
from netCDF4 import Dataset
import netCDF4

#%% Initial inputs 

# url where data resides
url = 'https://data.ioos.us/gliders/thredds/catalog/deployments/'
catalog = 'catalog.html'
nc_url = 'https://data.ioos.us/thredds/dodsC/deployments/'

# Initial and fina date from which we wish to download data
yearini = '20180101T0000'
dateini = '20180601T0000'
dateend = '20181201T0000'

# lat and lon bounds
lon_lim = [-110.0,0.0]
lat_lim = [14.0,45.0]

#%% Find URL address for nc files in ioos thredds server
'''
r = requests.get(url + catalog)
data = r.text
soup = BeautifulSoup(data,"lxml")

folders = []
for s in soup.find_all("a"):
    folders.append(s.get("href").split('/')[0])

glider_folders = folders[1:-4]

id_list = []
for l in glider_folders:
    url2 = url + l + '/' + catalog
    r = requests.get(url2)
    data = r.text
    soup2 = BeautifulSoup(data,"lxml")
    for i, s in enumerate(soup2.find_all("a")):
        if i>0 and i<len(soup2.find_all("a"))-4:
            date_file = [l for l in s.get("href").split('/')[0].split('-') if l[0:2]=='20']
            if time.mktime(datetime.strptime(date_file[0][0:13], "%Y%m%dT%H%M").timetuple()) >= \
            time.mktime(datetime.strptime(dateini, "%Y%m%dT%H%M").timetuple()):
                id_list.append(nc_url + l + '/' + \
                               s.get("href").split('/')[0] + '/' + s.get("href").split('/')[0] + '.nc3.nc')
                print(date_file[0])
                
'''                
#%% Find URL address for nc files in ioos thredds server including lat and lon 

'''
r = requests.get(url + catalog)
data = r.text
soup = BeautifulSoup(data,"lxml")

folders = []
for s in soup.find_all("a"):
    folders.append(s.get("href").split('/')[0])

glider_folders = folders[1:-4]

id_list = []
for l in glider_folders:
    url2 = url + l + '/' + catalog
    r = requests.get(url2)
    data = r.text
    soup2 = BeautifulSoup(data,"lxml")
    for i, s in enumerate(soup2.find_all("a")):
        if i>0 and i<len(soup2.find_all("a"))-4:
            date_file = [l for l in s.get("href").split('/')[0].split('-') if l[0:2]=='20']
            if time.mktime(datetime.strptime(date_file[0][0:13], "%Y%m%dT%H%M").timetuple()) >= \
            time.mktime(datetime.strptime(dateini, "%Y%m%dT%H%M").timetuple()):
                file = nc_url + l + '/' + \
                               s.get("href").split('/')[0] + '/' + s.get("href").split('/')[0] + '.nc3.nc'
                               
                ncglider = Dataset(file)
                latitude = ncglider.variables['latitude'][:]
                longitude = ncglider.variables['longitude'][:]
                if longitude.min() >= lon_lim[0]  and longitude.max() <= lon_lim[1]:
                    if latitude.min() >= lat_lim[0]  and latitude.max() <= lat_lim[1]:
                        id_list.append(file)
                        print(s.get("href").split('/')[0])
                        #print(date_file[0])
                        

'''
#%% Find URL address for nc files in ioos thredds server including lat and lon 
#   and time
                        

r = requests.get(url + catalog)
data = r.text
soup = BeautifulSoup(data,"lxml")

folders = []
for s in soup.find_all("a"):
    folders.append(s.get("href").split('/')[0])

glider_folders = folders[1:-4] # firts and last three lines are other links

id_list = []
for l in glider_folders:
    
    url2 = url + l + '/' + catalog
    r = requests.get(url2)
    data = r.text
    soup2 = BeautifulSoup(data,"lxml")
    for i, s in enumerate(soup2.find_all("a")):
        if i>0 and i<len(soup2.find_all("a"))-4: # firts and last three lines are other links
            #print(l,' ',i)
            url3 = url + l + '/' + s.get("href").split('/')[0] + '/' + catalog #url where nc file is listed
            #print(url3)
            r = requests.get(url3)
            data3 = r.text
            soup3 = BeautifulSoup(data3,"lxml")
            if len(soup3.find_all("a")) >= 6: # checking file exists
                date_file = [l for l in s.get("href").split('/')[0].split('-') if l[0:2]=='20']
                if time.mktime(datetime.strptime(date_file[0][0:13], "%Y%m%dT%H%M").timetuple()) >= \
                    time.mktime(datetime.strptime(yearini, "%Y%m%dT%H%M").timetuple()): # checking initial date of file is within bounds
                        file = nc_url + l + '/' + \
                               s.get("href").split('/')[0] + '/' + s.get("href").split('/')[0] + '.nc3.nc'
                        
                        if s.get("href").split('/')[0] == 'sp034-20180514T1938' \
                           or s.get("href").split('/')[0] == 'sp022-20180422T1229'\
                           or s.get("href").split('/')[0] == 'sp013-20180927T1717'\
                           or s.get("href").split('/')[0] == 'sp010-20180620T1455':
                            print(s.get("href").split('/')[0], ' time is bad in nc file !!')
                        else:
                            ncglider = Dataset(file)
                            latitude = ncglider.variables['latitude'][:]
                            longitude = ncglider.variables['longitude'][:]
                            tnc = ncglider.variables['time']
                            tnc = netCDF4.num2date(tnc[:],tnc.units) 
                            if longitude.min() >= lon_lim[0]  and longitude.max() <= lon_lim[1]:
                                if latitude.min() >= lat_lim[0]  and latitude.max() <= lat_lim[1]:
                                    if tnc[0][-1] > datetime.strptime(dateini, "%Y%m%dT%H%M"): 
                                        id_list.append(file)
                                        print(s.get("href").split('/')[0])
    

#%%
         
#file = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/silbo-20180525T1016/silbo-20180525T1016.nc3.nc'                          
'''
l = 'aoml'
ii = 7
l = 'mkhoward'
ii = 17
url2 = url + l + '/' + catalog
r = requests.get(url2)
data = r.text
soup2 = BeautifulSoup(data,"lxml")
for i, s in enumerate(soup2.find_all("a")):
    if i==ii:
        url3 = url + l + '/' + s.get("href").split('/')[0]
        r = requests.get(url3)
        data3 = r.text
        soup3 = BeautifulSoup(data3,"lxml")
        for x in soup3.find_all("a"):
            if x.get("href").find('.nc3.nc') != -1: # checking nc file exists
            
                date_file = [l for l in s.get("href").split('/')[0].split('-') if l[0:2]=='20']  
                file = nc_url + l + '/' + \
                               s.get("href").split('/')[0] + '/' + s.get("href").split('/')[0] + '.nc3.nc'           
                ncglider = Dataset(file)
                latitude = ncglider.variables['latitude'][:]
                longitude = ncglider.variables['longitude'][:]
                if longitude.min() >= lon_lim[0]  and longitude.max() <= lon_lim[1]:
                    if latitude.min() >= lat_lim[0]  and latitude.max() <= lat_lim[1]:
                        id_list.append(file)
                        print(l)
                        print(i)
                        print(date_file[0])
'''
                      