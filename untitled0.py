#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:59:28 2019

@author: aristizabal
"""

import numpy.ma as ma

lonlat=parse_b(gridfile,'relax')
ijdm = lonlat['idm'] * lonlat['jdm']
npad = 4096 - (ijdm%4096)
fld2 = ma.array([],fill_value=1e30)

aFile = gridfile+'.a'
fid   = open(aFile,'rb')

layers=[1]
    
for lyr in layers:
    fid.seek((lonlat[fieldName][lyr-1]-1)*4*(npad+ijdm),0)
    fld = fid.read(ijdm*4)
    fld = struct.unpack('>'+str(ijdm)+'f',fld)
    fld = np.array(fld)

       if pntidx is not None:
         fld = fld[pntidx]
         if fld2.size == 0:
            fld2=fld.copy()
         else:
            fld2=np.vstack((fld2,fld))
       else:
         fld = ma.reshape(fld,(lonlat['jdm'],lonlat['idm']))
         if fld2.size == 0:
            fld2=fld.copy()
         else:
            fld2=ma.dstack((fld2,fld))

    fid.close()
    fld2=ma.masked_greater(fld2,1e10)
    return fld2
