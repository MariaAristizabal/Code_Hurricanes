
#%% User input

# Glider data url address

'''
#Gulf of Mexico
gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc';
# lat and lon of area
lon_lim = [-100,-80]
lat_lim = [  18, 32]
# Time window
date_ini = '2018/10/7/00/00'
date_end = '2018/10/13/00/00'
# time of hurricane passage
thurr = '2018/10/10/06/00'
'''

'''
# MAB + SAB
# RU33 (MAB + SAB)
gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru33-20180801T1323/ru33-20180801T1323.nc3.nc';
# lat and lon of area
lon_lim = [-81,-70]
lat_lim = [30,42]
#Time window
date_ini = '2018/09/06/00/00'
date_end = '2018/09/15/00/00'
# time of hurricane passage
thurr = '2018/09/14/18/00'
'''

# Caribbean
gdata = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng467-20180701T0000/ng467-20180701T0000.nc3.nc'
# lat and lon of area
lon_lim = [-68,-64]
lat_lim = [15,20]
#Time window
date_ini = '2018/09/06/00/00'
date_end = '2018/09/15/00/00'
# time of hurricane passage
thurr = '2018/09/14/18/00'

# url for GOFS 3.1
catalog31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z'

# Bathymetry data
bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc'

# In[1]:

from matplotlib import pyplot as plt
#import cmocean
import numpy as np
import xarray as xr
import matplotlib.dates as mdates
import datetime

plt.style.use('seaborn-poster')
plt.style.use('ggplot')
import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


# In[3]:

def dens0(s, t):
    s, t = list(map(np.asanyarray, (s, t)))
    T68 = T68conv(t)
    # UNESCO 1983 Eqn.(13) p17.
    b = (8.24493e-1, -4.0899e-3, 7.6438e-5, -8.2467e-7, 5.3875e-9)
    c = (-5.72466e-3, 1.0227e-4, -1.6546e-6)
    d = 4.8314e-4
    return (smow(t) + (b[0] + (b[1] + (b[2] + (b[3] + b[4] * T68) * T68) *
            T68) * T68) * s + (c[0] + (c[1] + c[2] * T68) * T68) * s * s ** 0.5 + d * s ** 2)

def smow(t):
    t = np.asanyarray(t)
    a = (999.842594, 6.793952e-2, -9.095290e-3, 1.001685e-4, -1.120083e-6, 6.536332e-9)
    T68 = T68conv(t)
    return (a[0] + (a[1] + (a[2] + (a[3] + (a[4] + a[5] * T68) * T68) * T68) * T68) * T68)
    
def T68conv(T90):
    T90 = np.asanyarray(T90)
    return T90 * 1.00024

#%% Glider data
        
dglider = xr.open_dataset(gdata,decode_times=False) 

inst_id = dglider.id.split('_')[0]
inst_name = inst_id.split('-')[0]  

latitude = dglider.latitude[0] 
longitude = dglider.longitude[0]
temperature = dglider.temperature[0]
salinity = dglider.salinity[0]
density = dglider.density[0]
depth = dglider.depth[0]

## Change time into standardized mdates datenums 
seconds_since1970 = dglider.time[0]
timei = datetime.datetime.strptime(dglider.time.time_origin,'%d-%b-%Y %H:%M:%S')
timei + datetime.timedelta(seconds=int(seconds_since1970[0]))
time = np.empty(len(seconds_since1970))
for ind, hrs in enumerate(seconds_since1970):
    time[ind] = mdates.date2num(timei + datetime.timedelta(seconds=int(hrs)))

# Find time window of interest
tti = mdates.date2num(datetime.datetime.strptime(date_ini,'%Y/%m/%d/%H/%M'))     
tte = mdates.date2num(datetime.datetime.strptime(date_end,'%Y/%m/%d/%H/%M')) 
oktimeg = np.logical_and(time >= tti,time <= tte)

# Fiels within time window
timeg = time[oktimeg]
latg = latitude[oktimeg]
long = longitude[oktimeg]
tempg =  temperature[oktimeg,:]
saltg = salinity[oktimeg,:]
depthg = depth[oktimeg,:]
densg = density[oktimeg,:]

if np.nanmin(densg) < 100:
    densg = densg + 1000
    
# Change glider lot and lat to GOFS 3.1 convention
target_lon = np.empty((len(long),))
target_lon[:] = np.nan
for i in range(len(long)):
    if long[i] < 0: 
        target_lon[i] = 360 + long[i]
    else:
        target_lon[i] = long[i]
target_lat = latg

#%% GOFS 3.1

## This takes a couple seconds because the thredds for GOFS is sloooow
df = xr.open_dataset(catalog31,decode_times=False)

## Decode the GOFS3.1 time into standardized mdates datenums 
hours_since2000 = df.time
time_naut       = datetime.datetime(2000,1,1)
time31 = np.ones_like(hours_since2000)
for ind, hrs in enumerate(hours_since2000):
    time31[ind] = mdates.date2num(time_naut+datetime.timedelta(hours=int(hrs)))
   
    #%%
## Find the dates of import
dini = mdates.date2num(datetime.datetime.strptime(date_ini,'%Y/%m/%d/%H/%M')) # October 7, 2018
dend = mdates.date2num(datetime.datetime.strptime(date_end,'%Y/%m/%d/%H/%M')) # October 11, 2018
formed  = int(np.where(time31 == dini)[0][0])
dissip  = int(np.where(time31 == dend)[0][0])
oktime31 = np.arange(formed,dissip+1,dtype=int)

lat31 = df.lat
lon31 = df.lon
depth31 = df.depth

### Build the bbox for the xy data
botm  = np.where(df.lat > lat_lim[0])[0][0]
top   = np.where(df.lat > lat_lim[1])[0][0]
half  = int(len(df.lon)/2)

left  = np.where(df.lon > lon_lim[0]+360)[0][0]
right = np.where(df.lon > lon_lim[1]+360)[0][0]
lat100= df.lat[botm:top]
lon100= df.lon[left:right]
X, Y = np.meshgrid(lon100,lat100)

#%% Read Bathymetry data

Bathy = xr.open_dataset(bath_data,decode_times=False) 

latbath = Bathy.lat
lonbath = Bathy.lon
elevbath = Bathy.elevation

# Subsample bathymetry data
oklatbath = np.logical_and(latbath > lat_lim[0],latbath < lat_lim[1])
oklonbath = np.logical_and(lonbath > lon_lim[0],lonbath < lon_lim[1])
sublatbath = latbath[oklatbath]
sublonbath = lonbath[oklonbath]
subelevbath = elevbath[oklatbath,oklonbath]

#%%

# interpolating glider lon and lat to lat and lon on model time
sublon31 = np.interp(time31,timeg,target_lon)
sublat31 = np.interp(time31,timeg,target_lat)

# getting the model grid positions for sublon31 and sublat31
oklon31 = np.round(np.interp(sublon31,lon31,np.arange(len(lon31)))).astype(int)
oklat31 = np.round(np.interp(sublat31,lat31,np.arange(len(lat31)))).astype(int)

#%% Get glider transect from model

target_temp31 = np.empty((len(depth31),len(oktime31)))
target_temp31[:] = np.nan
target_salt31 = np.empty((len(depth31),len(oktime31)))
target_salt31[:] = np.nan
for i in range(len(oktime31)):
    target_temp31[:,i] = df.water_temp[oktime31[i],:,oklat31[i],oklon31[i]]
    target_salt31[:,i] = df.salinity[oktime31[i],:,oklat31[i],oklon31[i]]

target_temp31[target_temp31 < -100] = np.nan

#%% Calculate time series of non-dimensional potential Energy Anomaly for GOFS 3.1
# over the top 100 m

sea31 = np.empty((len(time31[oktime31])))
sea31[:] = np.nan

for t,tstamp in enumerate(time31[oktime31]):
    print(t)
    dindex = np.where(depth31 <= 100)[0]
    zz = np.asarray(depth31[dindex])
        
    T100   = target_temp31[:,t]
    S100   = target_salt31[:,t]
    dennss  = dens0(S100,T100)
    denss = dennss[dindex]
    
    ok = np.isfinite(denss)
    dens = denss[ok]
    z = zz[ok]
        
    # defining sigma
    max_depth = np.nanmax(z) 
    sigma = -1*z/max_depth
    sigma = np.flipud(sigma)

    density = np.flipud(dens)
    rhomean = np.trapz(density,sigma,axis=0)           
    drho = (rhomean-density)/rhomean
    torque = drho * sigma
    sea31[t] = -np.trapz(torque,sigma,axis=0) 
    print(sea31[t])     
    
    
#%% Calculate time series of non-dimensional potential Energy Anomaly for glider
# over the top 100 m

seag = np.empty((len(timeg)))
seag[:] = np.nan
for t,tstamp in enumerate(timeg):   
    print(t)
    dindex = np.fliplr(np.where(np.asarray(depthg[t,:]) <= 100))[0]
    if len(dindex) == 0:
        seag[t] = np.nan
    else:
        zz = np.asarray(depthg[t,dindex])
        denss = np.asarray(densg[t,dindex])
        ok = np.isfinite(denss)
        z = zz[ok]
        dens = denss[ok]
        if len(z)==0 or len(dens)==0:
            seag [t] = np.nan
        else:
            if z[-1] - z[0] > 0:
                z = np.append(0,z)
            else:
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

            drho = (rhomean-densit)/rhomean
            torque = drho * sigma
            seag[t] = -np.trapz(torque,sigma,axis=0) 
            print(seag[t])
            
#%%    
'''
t=288          
dindex = np.fliplr(np.where(np.asarray(depthg[t,:]) <= 100))[0]
zz = np.asarray(depthg[t,dindex])
denss = np.asarray(densg[t,dindex])
ok = np.isfinite(denss)
z = zz[ok]
dens = denss[ok]
if z[-1] - z[0] > 0:
    z = np.append(0,z)
    #z_surf = np.append(np.arange(0,z[0],0.5),z)
else:
    z = np.flipud(z)
    z = np.append(0,z)
    #z_surf = np.append(np.arange(0,z[0],0.5),z)
    dens = np.flipud(dens)
        
# ading density at depth = 0
densit = np.interp(z,z[1:],dens)
densit= np.flipud(densit)
    
# defining sigma
max_depth = np.nanmax(zz[ok])  
sigma = -1*z/max_depth
sigma = np.flipud(sigma)
    
rhomean = np.trapz(densit,sigma,axis=0)
drho = (rhomean-densit)/rhomean
torque = drho * sigma
seag[t] = -np.trapz(torque,sigma,axis=0) 
print(seag[t]) 
'''           
            
#%% Calculate time series of original potential Energy Anomaly for glider
# over the top 100 m

g = 9.8 #m/s
seag_orig = np.empty((len(timeg)))
seag_orig[:] = np.nan
for t,tstamp in enumerate(timeg):   
    print(t)
    dindex = np.fliplr(np.where(np.asarray(depthg[t,:]) <= 100))[0]
    if len(dindex) == 0:
        seag_orig[t] = np.nan
    else:
        zz = np.asarray(depthg[t,dindex])
        denss = np.asarray(densg[t,dindex])
        ok = np.isfinite(denss)
        z = zz[ok]
        dens = denss[ok]
        if len(z)==0 or len(dens)==0:
            seag_orig [t] = np.nan
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
            seag_orig[t] = -g* max_depth * np.trapz(torque,sigma,axis=0) 
            print(max_depth, ' ',seag_orig[t]) 
            
#%%  Calculate time series of original potential Energy Anomaly for GOFS 3.1
# over the top 100 m

sea31_orig = np.empty((len(time31[oktime31])))
sea31_orig[:] = np.nan

for t,tstamp in enumerate(time31[oktime31]):
    print(t)
    dindex = np.where(depth31 <= 100)[0]
    zz = np.asarray(depth31[dindex])
        
    T100   = target_temp31[:,t]
    S100   = target_salt31[:,t]
    dennss  = dens0(S100,T100)
    denss = dennss[dindex]
    
    ok = np.isfinite(denss)
    dens = denss[ok]
    z = zz[ok]
        
    # defining sigma
    max_depth = np.nanmax(z) 
    sigma = -1*z/max_depth
    sigma = np.flipud(sigma)

    density = np.flipud(dens)
    rhomean = np.trapz(density,sigma,axis=0)           
    drho = rhomean-density
    torque = drho * sigma
    sea31_orig[t] = -g* max_depth * np.trapz(torque,sigma,axis=0) 
    print(max_depth, ' ',seag_orig[t]) 
    
#%%
'''
t=0
dindex = np.where(depth31 <= 100)[0]
zz = np.asarray(depth31[dindex])
        
T100   = target_temp31[:,t]
S100   = target_salt31[:,t]
dennss  = dens0(S100,T100)
denss = dennss[dindex]
    
ok = np.isfinite(denss)
dens = denss[ok]
z = zz[ok]
        
# defining sigma
max_depth = np.nanmax(z) 
sigma = -1*z/max_depth
sigma = np.flipud(sigma)

density = np.flipud(dens)
rhomean = np.trapz(density,sigma,axis=0)           
drho = rhomean-density
torque = drho * sigma
sea31_orig[t] = -g* max_depth * np.trapz(torque,sigma,axis=0) 
np.sum(drho[0:-1,:,:] *np.diff(sigma,axis=0),axis=0)
print(max_depth, ' ',seag_orig[t]) 
'''
    
#%% Quality Control profiles for calculation PEA
            
for t,tstamp in enumerate(timeg):   
    print(t)
    dindex = np.fliplr(np.where(depthg[t,:] <= 100))[0]
    if len(dindex) != 0:
        zz = np.asarray(depthg[t,dindex])
        denss = np.asarray(densg[t,dindex])
        ok = np.isfinite(denss)
        z = zz[ok]
        dens = denss[ok]
        #if len(z) < 40:
        if len(z) < 150:    
            seag[t] = np.nan
            seag_orig[t] = np.nan
        print(t,' ', 'len= ',len(z)) 
       
        
#%%
'''        
t=288
dindex = np.fliplr(np.where(depthg[t,:] <= 100))[0]
zz = np.asarray(depthg[t,dindex])
denss = np.asarray(densg[t,dindex])
ok = np.isfinite(denss)
z = zz[ok]
dens = denss[ok]
#if len(z) < 40:
if len(z) < 150:    
    seag[t] = np.nan
    seag_orig[t] = np.nan
print('len= ',len(z)) 
'''       
        
#%% Interpolate to GOFS 3.1 time

ok = np.isfinite(seag)        
seag_int = np.interp(time31[oktime31],timeg[ok],seag[ok])
ok = np.isfinite(denss)          

ok = np.isfinite(seag_orig)        
seag_orig_int = np.interp(time31[oktime31],timeg[ok],seag_orig[ok])    
            
#%% Plot time series of non-dimensional PEA for GOFS 3.1 and glider

tMic = datetime.datetime.strptime(thurr,'%Y/%m/%d/%H/%M')
tMic = np.tile(tMic,len(np.arange(np.nanmin(seag),0,-np.nanmin(seag)/20)))

fig= plt.figure(1, figsize=(13,6)) 
ax =plt.subplot()
ax.plot(mdates.num2date(timeg),seag,'.',color = 'aqua',label = inst_name)
ax.plot(mdates.num2date(time31[oktime31]),sea31,'.-',label = 'GOFS 3.1',color = 'red') 
ax.plot(mdates.num2date(time31[oktime31]),seag_int,'.-', label = inst_name, color='blue')
ax.legend(loc='upper left',fontsize = 20)
ax.plot(tMic,np.arange(np.nanmin(seag),0,-np.nanmin(seag)/20),'--k')
#ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.title('Non-dimensional Potential Energy Anomaly (100 m)',size=20)

xfmt = mdates.DateFormatter('%m-%d')
ax.xaxis.set_major_formatter(xfmt)

#plt.ylim(-0.0006,0)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/non_dim_SEA_GOFS31_vs_" \
       + inst_name + ".png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

         

#%% Figure time series original PEA

tMic = datetime.datetime.strptime(thurr,'%Y/%m/%d/%H/%M')
tMic = np.tile(tMic,len(np.arange(np.nanmin(seag_orig),0,-np.nanmin(seag_orig)/20)))

fig= plt.figure(1, figsize=(13,6)) 
ax =plt.subplot()
ax.plot(mdates.num2date(timeg),seag_orig,'.',color = 'aqua',label = inst_name)
ax.plot(mdates.num2date(time31[oktime31]),sea31_orig,'.-',label = 'GOFS 3.1',color = 'red') 
ax.plot(mdates.num2date(time31[oktime31]),seag_orig_int,'.-', label = inst_name, color='blue')
ax.legend(loc='upper left',fontsize = 20)
ax.plot(tMic,np.arange(np.nanmin(seag_orig),0,-np.nanmin(seag_orig)/20),'--k')
#ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.ylabel('$w/m^2$')
plt.title('Original Potential Energy Anomaly (100 m)',size=20)

xfmt = mdates.DateFormatter('%m-%d')
ax.xaxis.set_major_formatter(xfmt)
#plt.ylim(-500,0)

#plt.ylim(-0.0006,0)

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/orig_SEA_GOFS31_vs_" \
       + inst_name + ".png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)             

#%% Calculate non-dimensional potential Energy Anomaly for GOFS 3.1 
    
## Limit the depths to 100m
dindex = df.depth <= 100

tindex = time31[oktime31] ==  mdates.date2num(datetime.datetime(2018, 9, 13, 0, 0))
t#index = time31[oktime31] ==  mdates.date2num(datetime.datetime(2018, 10, 9, 0, 0))

T100   = df.water_temp[oktime31[tindex][0],dindex,botm:top,left:right]
S100   = df.salinity[oktime31[tindex][0],dindex,botm:top,left:right]
denss  = dens0(S100,T100)

ok = np.isfinite(denss)  
density = np.empty((T100.shape[0],T100.shape[1],T100.shape[2]))
#density[:] = np.nan
density[ok] = denss[ok]

zz = np.asarray(depth31[dindex])
zs = np.tile(zz,(1,T100.shape[1]))
zss = np.tile(zs,(1,T100.shape[2]))
zmat = np.reshape(zss,(T100.shape[0],T100.shape[1],T100.shape[2]), order='F')
bad = np.isnan(denss)
zmat[bad] = np.nan
 
sigma = np.empty((T100.shape[0],T100.shape[1],T100.shape[2]))
sigma[:] = -1.0
for x,xx in enumerate(np.arange(zmat.shape[1])):
    for y,yy in enumerate(np.arange(zmat.shape[2])):
        sigma[:,x,y] = -zmat[:,x,y]/np.nanmax(zmat[:,x,y])
        sigma[np.isnan(sigma[:,x,y]),x,y] = -1
        
rhomean = -np.trapz(density,sigma,axis=0)
#rhomean = -np.nansum(density[1:,:,:] *np.diff(sigma,axis=0),axis=0)
nok = np.isnan(T100[0,:,:])
rhomean[nok] = np.nan


drho = np.empty((T100.shape[0],T100.shape[1],T100.shape[2]))
drho[:] = np.nan
for k,dind in enumerate(np.arange(T100.shape[0])):
    drho[k,:,:] = (rhomean - density[k,:,:])/rhomean
drho[bad] = 0

torque = drho * sigma

#SEA31 = np.trapz(drho,sigma,axis=0)
#SEA31 = np.nansum(torque[0:-1,:,:] *np.diff(sigma,axis=0),axis=0)
SEA31 = np.trapz(torque,sigma,axis=0)
#SEA31 = np.nansum(drho[0:-1,:,:] *np.diff(sigma,axis=0),axis=0)
SEA31[nok] = np.nan

#%% Calculate original potential Energy Anomaly for GOFS 3.1 
    
## Limit the depths to 100m
dindex = df.depth <= 100

tindex = time31[oktime31] ==  mdates.date2num(datetime.datetime(2018, 9, 13, 0, 0))
#tindex = time31[oktime31] ==  mdates.date2num(datetime.datetime(2018, 10, 9, 0, 0))

T100   = df.water_temp[oktime31[tindex][0],dindex,botm:top,left:right]
S100   = df.salinity[oktime31[tindex][0],dindex,botm:top,left:right]
denss  = dens0(S100,T100)

ok = np.isfinite(denss)  
density = np.empty((T100.shape[0],T100.shape[1],T100.shape[2]))
#density[:] = np.nan
density[ok] = denss[ok]

zz = np.asarray(depth31[dindex])
zs = np.tile(zz,(1,T100.shape[1]))
zss = np.tile(zs,(1,T100.shape[2]))
zmat = np.reshape(zss,(T100.shape[0],T100.shape[1],T100.shape[2]), order='F')
bad = np.isnan(denss)
zmat[bad] = np.nan
 
sigma = np.empty((T100.shape[0],T100.shape[1],T100.shape[2]))
sigma[:] = -1.0
for x,xx in enumerate(np.arange(zmat.shape[1])):
    for y,yy in enumerate(np.arange(zmat.shape[2])):
        sigma[:,x,y] = -zmat[:,x,y]/np.nanmax(zmat[:,x,y])
        sigma[np.isnan(sigma[:,x,y]),x,y] = -1
        
rhomean = -np.trapz(density,sigma,axis=0)
#rhomean = -np.nansum(density[1:,:,:] *np.diff(sigma,axis=0),axis=0)
nok = np.isnan(T100[0,:,:])
rhomean[nok] = np.nan


drho = np.empty((T100.shape[0],T100.shape[1],T100.shape[2]))
drho[:] = np.nan
for k,dind in enumerate(np.arange(T100.shape[0])):
    drho[k,:,:] = (rhomean - density[k,:,:])
drho[bad] = 0

torque = drho * sigma

#SEA31 = np.trapz(drho,sigma,axis=0)
#SEA31_orig = g* np.nanmax(zmat,axis=0)*np.nansum(torque[0:-1,:,:] *np.diff(sigma,axis=0),axis=0)
SEA31_orig = g* np.nanmax(zmat,axis=0)*np.trapz(torque,sigma,axis=0)
SEA31_orig[nok] = np.nan

#%% Tentative Michael path
'''
lonMc = np.array([-84.9,-85.2,-85.3,-85.9,-86.2,-86.4,-86.5,-86.5,-86.3,-86.2,\
                  -86.0,-85.8,-85.5,-85.2,-84.9,-84.5,-84.1])
latMc = np.array([21.2,22.2,23.2,24.1,25.0,26.0,27.1,28.3,28.8,29.1,29.4,29.6,\
                  30.0,30.6,31.1,31.5,31.9])
tMc = ['2018/10/08/15/00','2018/10/08/21/00','2018/10/09/03/00','2018/10/09/09/00',\
       '2018/10/09/15/00','2018/10/09/21/00','2018/10/10/03/00','2018/10/10/09/00',\
       '2018/10/10/11/00','2018/10/10/13/00','2018/10/10/15/00','2018/10/10/16/00',\
       '2018/10/10/18/00','2018/10/10/20/00','2018/10/10/22/00','2018/10/11/00/00',\
       '2018/10/11/02/00']

# Convert time to UTC
#pst = pytz.timezone('America/New_York') # time zone
#utc = pytz.UTC 

timeMc = [None]*len(tMc) 
for x in range(len(tMc)):
    timeMc[x] = datetime.datetime.strptime(tMc[x], '%Y/%m/%d/%H/%M') # time in time zone   

'''

#%%

fig = plt.figure(1, figsize=(13,8))
ax = plt.subplot(projection=ccrs.PlateCarree())
plot_date = mdates.num2date(time31[oktime31[tindex][0]])

var = subelevbath
#var = rhomean

max_v = np.nanmax(abs(var))

kw = dict(levels=np.linspace(np.nanmin(var),0,10), 
         cmap=plt.cm.Spectral_r, 
          transform=ccrs.PlateCarree())

#kw = dict(levels=np.linspace(-0.2,0,101), 
#         cmap=cmocean.cm.curl_r, 
#        transform=ccrs.PlateCarree())

plt.contourf(sublonbath, sublatbath, var, **kw)
plt.colorbar()
plt.contour(sublonbath, sublatbath, var, \
            levels=np.linspace(np.nanmin(var),0,10),colors='k',linestyles='-',\
            linewidths=1)
ax.plot(np.mean(long),np.mean(latg),'k*',markersize=15)


#%% Figure non-dimensional PEA GOFS 3.1 

ok200 = np.where(subelevbath == -200)
ok100 = np.where(subelevbath == -100)
ok50 = np.where(subelevbath == -50)
ok20 = np.where(subelevbath == -20)
ok30 = np.where(subelevbath == -30)
#SEA31[SEA31 < -0.2*10**-3] = np.nan

fig = plt.figure(1, figsize=(13,8))
ax = plt.subplot(projection=ccrs.PlateCarree())
plot_date = mdates.num2date(time31[oktime31[tindex][0]])
plt.title('Non-dimensional Simpson Energy Anomaly (100$m$) \n GOFS 3.1 on {}'.format(plot_date))

var = SEA31 *1000
#var = rhomean

max_v = np.nanmax(abs(var))

kw = dict(levels=np.linspace(np.nanmin(var),0,101), 
         cmap=plt.cm.Spectral_r, 
          transform=ccrs.PlateCarree())

#kw = dict(levels=np.linspace(-0.2,0,101), 
#         cmap=cmocean.cm.curl_r, 
#        transform=ccrs.PlateCarree())

plt.contourf(lon100, lat100, var, **kw)
#ax.plot(sublonbath[ok200[1]],sublatbath[ok200[0]],'.k')
#ax.plot(sublonbath[ok50[1]],sublatbath[ok50[0]],'.k',markersize=3)
#ax.plot(sublonbath[ok20[1]],sublatbath[ok20[0]],'.k',markersize=2)
#ax.plot(sublonbath[ok30[1]],sublatbath[ok30[0]],'.k',markersize=2)
#ax.plot(sublonbath[ok100[1]],sublatbath[ok100[0]],'.k',markersize=4)
cb = plt.colorbar(format='%.1e')#,ticks=[-6e-1,-5e-1,-4e-1, -3e-1, -2e-1, -1e-1,0])
#cb = plt.colorbar()
#plt.clim(-1.2*10**-1,0)
cb.set_label('Stratification Factor ($x1000$)',rotation=270, labelpad=25, fontsize=12)

'''
# Michael track
ax.plot(lonMc,latMc,'o-',markersize = 8,label = 'Michael Track',color = 'dimgray')

for x in range(0, len(tMc)-1, 2):
    ax.text(lonMc[x],latMc[x],timeMc[x].strftime('%d, %H:%M'),size = 12)
'''    

ax.plot(np.mean(long),np.mean(latg),'k*',markersize=15)
#ax.plot(lon100[98]-360,lat100[112],'r*',markersize=15)

### High resolution coastline is set by resolution='10m'
ax.coastlines(resolution='50m', color='black', linewidth=1)
ax.add_feature(cartopy.feature.LAND, zorder=0)
ax.add_feature(cartopy.feature.OCEAN, zorder=0, color='gray', alpha=0.1)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.2, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 14}
gl.ylabel_style = {'size': 14}

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/SEA_GOFS31_MAB+SAB.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 

#%% Figure original PEA GOFS 3.1 

ok1000 = np.where(subelevbath == -1000)
ok200 = np.where(subelevbath == -200)
ok100 = np.where(subelevbath == -100)
ok50 = np.where(subelevbath == -50)
ok20 = np.where(subelevbath == -20)
ok30 = np.where(subelevbath == -30)
#SEA31[SEA31 < -0.2*10**-3] = np.nan

fig = plt.figure(1, figsize=(13,8))
ax = plt.subplot(projection=ccrs.PlateCarree())
plot_date = mdates.num2date(time31[oktime31[tindex][0]])
plt.title('Original Potential Energy Anomaly (100$m$) \n GOFS 3.1 on {}'.format(plot_date))

#var = np.nanmax(zmat,axis=0)
var = SEA31_orig
#var = rhomean

max_v = np.nanmax(abs(var))

kw = dict(levels=np.linspace(np.nanmin(var),0,101), 
         cmap=plt.cm.Spectral_r, 
          transform=ccrs.PlateCarree())

#kw = dict(levels=np.linspace(-0.2,0,101), 
#         cmap=cmocean.cm.curl_r, 
#        transform=ccrs.PlateCarree())

plt.contourf(lon100, lat100, var, **kw)
#ax.plot(sublonbath[ok200[1]],sublatbath[ok200[0]],'.k')
#ax.plot(sublonbath[ok50[1]],sublatbath[ok50[0]],'.k',markersize=3)
#ax.plot(sublonbath[ok20[1]],sublatbath[ok20[0]],'.k',markersize=2)
#ax.plot(sublonbath[ok30[1]],sublatbath[ok30[0]],'.k',markersize=2)
#ax.plot(sublonbath[ok100[1]],sublatbath[ok100[0]],'.k',markersize=4)
cb = plt.colorbar(format='%d')#,ticks=[-6e-1,-5e-1,-4e-1, -3e-1, -2e-1, -1e-1,0])
#cb = plt.colorbar()
#plt.clim(-1.2*10**-1,0)
cb.set_label('($w/m^2$)',rotation=270, labelpad=25, fontsize=12)

'''
# Michael track
ax.plot(lonMc,latMc,'o-',markersize = 8,label = 'Michael Track',color = 'dimgray')

for x in range(0, len(tMc)-1, 2):
    ax.text(lonMc[x],latMc[x],timeMc[x].strftime('%d, %H:%M'),size = 12)
    
'''
ax.plot(np.mean(long),np.mean(latg),'k*',markersize=15)

### High resolution coastline is set by resolution='10m'
ax.coastlines(resolution='50m', color='black', linewidth=1)
ax.add_feature(cartopy.feature.LAND, zorder=0)
ax.add_feature(cartopy.feature.OCEAN, zorder=0, color='gray', alpha=0.1)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.2, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 14}
gl.ylabel_style = {'size': 14}

file = "/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/SEA_original_GOFS31_SAB+MAB.png"
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1) 


#%%

'''
The green is the non-stratified water, which leaves the red to indicate a level 
of stratification. In the Gulf of Mexico where the waters are warmer than 
the Mid-Atlantic and the thermocline is deeper, a significant Simpson Energy Anomaly 
( ϕ100m>0.3) indicated a shallower thermocline and less energy for tropical cyclones 
to uptake during a storm passage. Simultaneously, a low SEA indicates a deep 
thermocline and rich energy stores for potential tropical storm developement. 
Of specific note is that the shelf-regions of the Gulf are all destratified, 
with the exception of freshwater plumes from rivers.
'''