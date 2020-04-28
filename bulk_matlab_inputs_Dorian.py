#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 13:41:42 2020

@author: root
"""

#%%
SG665

ur = 36;
zr = 10; 
Ta = 2;
zt = 3;
rh = 97;
zq = 3; 
Pa = 990;
Ts = 29.15;

H = bulk(ur,zr,Ta,zt,rh,zq,Pa,Ts)

#%%
SG666

ur = 36.2;
zr = 10; 
Ta = 28.08;
zt = 3;
rh = 97;
zq = 3; 
Pa = 990;
Ts = 29.21;

H = bulk(ur,zr,Ta,zt,rh,zq,Pa,Ts)