#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 22:01:13 2019

@author: aristizabal
"""
#%% User input

mat_file = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/ng487_depth_aver_vel.mat'

# Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/';

#%%

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, timedelta

import scipy.io as sio
ng487 = sio.loadmat(mat_file)

#%%
uglider = ng487['ug'][:,0]
vglider = ng487['vg'][:,0]
tstamp_glider = ng487['timeg'][:,0]

tstamp_31 =  ng487['time31'][:,0]
oktime31 = ng487['oktime31'][:,0]
depth31 = ng487['depth31'][:,0]

u31 = ng487['target_u31']
v31 = ng487['target_v31']

#%% Changing timestamps to datenum

timeglid = []
tim31 = []
for i in np.arange(len(tstamp_glider)):
    timeglid.append(datetime.fromordinal(int(tstamp_glider[i])) + \
        timedelta(days=tstamp_glider[i]%1) - timedelta(days = 366))
    tim31.append(datetime.fromordinal(int(tstamp_31[i])) + \
        timedelta(days=tstamp_31[i]%1) - timedelta(days = 366))
    
timeglider = np.asarray(timeglid)
time31 = np.asarray(tim31)
tt31 = time31[oktime31]  

#%% Putting velocity vectors on the same time stamp
Ug,oku = np.unique(uglider,return_index=True)
Vg = vglider[oku]
Xg = timeglider[oku]
Yg = np.zeros(len(Xg))

X31 = time31[oktime31]
Y31 = np.zeros(len(X31))
# Depth average velocoty in to 200 m
okd = depth31 < 200
U31 = np.mean(u31[okd,:],0)
V31 = np.mean(v31[okd,:],0)

ttg = np.asarray(tstamp_glider)[oku]
tt31 = np.asarray(tstamp_31)[oktime31]

U31_interp = np.interp(ttg,tt31,U31)
V31_interp = np.interp(ttg,tt31,V31)

#%% Scatter plot velocity
plt.subplots(figsize=(5.6, 5.6))

x = np.arange(-0.4,0.4,0.001)
y = np.zeros(len(x))
plt.plot(x,y,'-k',zorder=0)
y = np.arange(-0.4,0.4,0.001)
x = np.zeros(len(x))
plt.plot(x,y,'-k',zorder=5)
plt.plot(Ug,Vg,'.b',zorder=10,label=ng487['inst_id'][0].split('-')[0])
plt.plot(U31,V31,'.r',zorder=15,label='GOFS 3.1')
plt.ylabel('v (m/s)',size=18)
plt.xlabel('u (m/s)',size=18)
plt.legend(loc='upper right',fontsize = 18)
plt.title('200 m Depth Average Velocity \n' + timeglid[0].strftime("%Y-%m-%d") \
          + ' - ' + timeglid[-1].strftime("%Y-%m-%d") , fontsize = 20)

plt.grid(True)
plt.ylim(-0.4,0.4)
plt.xlim(-0.4,0.4)
plt.xticks(np.arange(-0.4,0.5,0.1))

file = folder + 'depth_ave_vel_' + ng487['inst_id'][0].split('-')[0]
plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.1)   

#%% Velocity plot glider vs model

fig,ax = plt.subplots()
fig.set_size_inches(12.48,2.9)
Q1 = plt.quiver(Xg,Yg,Ug,Vg,scale=0.2,scale_units='inches',color='b',\
                label=ng487['inst_id'][0].split('-')[0],alpha=0.5)
Q2 = plt.quiver(Xg,Yg,U31_interp,V31_interp,scale=0.2,scale_units='inches',\
                color='r',label='GOFS 3.1',alpha=0.5)
qk = plt.quiverkey(Q1, 0.88, 0.9, 0.1, r'$0.1 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
plt.legend()
plt.title('Depth Average Velocity ',size=20)
xfmt = mdates.DateFormatter('%d-%b-%y')
ax.xaxis.set_major_formatter(xfmt)
ax.xaxis.label.set_size(30)
plt.yticks([])
plt.xticks(fontsize=12)

#file = folder + '{0}_{1}_{2}_{3}.png'.format('Depth_avg_velocity',inst_id.split('-')[0]\
#                     timeglider[0],timeglider[-1])

file = folder + 'Depth_Average_Velocity_' + ng487['inst_id'][0].split('-')[0]

plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.2) 

#%% Velocity plot glider vs model rotated

Ug,oku = np.unique(uglider,return_index=True)
Vg = vglider[oku]
Xg = timeglider[oku]
Yg = np.zeros(len(Xg))

X31 = time31[oktime31]
Y31 = np.zeros(len(X31))
# Depth average velocoty in to 200 m
okd = depth31 < 200
U31 = np.mean(u31[okd,:],0)
V31 = np.mean(v31[okd,:],0)

ttg = np.asarray(tstamp_glider)[oku]
tt31 = np.asarray(tstamp_31)[oktime31]

U31_interp = np.interp(ttg,tt31,U31)
V31_interp = np.interp(ttg,tt31,V31)

fig,ax = plt.subplots()
fig.set_size_inches(12.48,2.9)
Q1 = plt.quiver(Xg,Yg,Vg,Ug,scale=0.3,scale_units='inches',color='b',\
                label=ng487['inst_id'][0].split('-')[0],alpha=0.5)
Q2 = plt.quiver(Xg,Yg,V31_interp,U31_interp,scale=0.3,scale_units='inches',\
                color='r',label='GOFS 3.1',alpha=0.5)
qk = plt.quiverkey(Q1, 0.88, 0.9, 0.1, r'$0.1 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
plt.legend()
plt.title('Depth Average Velocity ',size=20)
xfmt = mdates.DateFormatter('%d-%b-%y')
ax.xaxis.set_major_formatter(xfmt)
ax.xaxis.label.set_size(30)
plt.yticks([])
plt.xticks(fontsize=12)

#file = folder + '{0}_{1}_{2}_{3}.png'.format('Depth_avg_velocity',inst_id.split('-')[0]\
#                     timeglider[0],timeglider[-1])

file = folder + 'Depth_Average_Velocity_' + ng487['inst_id'][0].split('-')[0] \
              +'_rotated'

plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.2) 

#%% Velocity plot glider vs model rotated 60 degrees 
# (approximately along the channel axis)

Ug,oku = np.unique(uglider,return_index=True)
Vg = vglider[oku]
Xg = timeglider[oku]
Yg = np.zeros(len(Xg))

X31 = time31[oktime31]
Y31 = np.zeros(len(X31))
# Depth average velocoty in to 200 m
okd = depth31 < 200
U31 = np.mean(u31[okd,:],0)
V31 = np.mean(v31[okd,:],0)

ttg = np.asarray(tstamp_glider)[oku]
tt31 = np.asarray(tstamp_31)[oktime31]

U31_interp = np.interp(ttg,tt31,U31)
V31_interp = np.interp(ttg,tt31,V31)

# Rotate velocities 60 degrees
alpha = np.radians(60)
UUg = np.cos(alpha)*Ug - np.sin(alpha)*Vg
VVg = np.sin(alpha)*Ug + np.cos(alpha)*Vg
UU31_interp = np.cos(alpha)*U31_interp - np.sin(alpha)*V31_interp
VV31_interp = np.sin(alpha)*U31_interp + np.cos(alpha)*V31_interp


fig,ax = plt.subplots()
fig.set_size_inches(12.48,2.9)
Q1 = plt.quiver(Xg,Yg,UUg,VVg,scale=0.2,scale_units='inches',color='b',\
                label=ng487['inst_id'][0].split('-')[0],alpha=0.5)
Q2 = plt.quiver(Xg,Yg,UU31_interp,VV31_interp,scale=0.2,scale_units='inches',\
                color='r',label='GOFS 3.1',alpha=0.5)
qk = plt.quiverkey(Q1, 0.88, 0.9, 0.1, r'$0.1 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
plt.legend()
plt.title('Depth Average Velocity ',size=20)
xfmt = mdates.DateFormatter('%d-%b-%y')
ax.xaxis.set_major_formatter(xfmt)
ax.xaxis.label.set_size(30)
plt.yticks([])
plt.xticks(fontsize=12)

#file = folder + '{0}_{1}_{2}_{3}.png'.format('Depth_avg_velocity',inst_id.split('-')[0]\
#                     timeglider[0],timeglider[-1])

file = folder + 'Depth_Average_Velocity_' + ng487['inst_id'][0].split('-')[0]\
              + '_rotated60'

plt.savefig(file,bbox_inches = 'tight',pad_inches = 0.2) 